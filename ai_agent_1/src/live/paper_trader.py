"""Real-time candle polling + strategy execution for OANDA demo."""

import csv
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.data.oanda_client import OandaClient
from src.indicators.ppst import compute_ppst
from src.indicators.entry_circles import compute_entry_circles
from src.strategy.base_strategy import BaseStrategy
from src.strategy.time_filter import compute_time_filter
from src.strategy.position import Position
from src.backtest.portfolio import Portfolio
from src.live.order_manager import OrderManager
from src.utils.forex_utils import format_price

logger = logging.getLogger(__name__)


class PaperTrader:
    """Live paper trading on OANDA demo account."""

    def __init__(
        self,
        client: OandaClient,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        instrument: str = "EUR_USD",
        granularity: str = "M1",
        ppst_params: Optional[dict] = None,
        time_filter_params: Optional[dict] = None,
        poll_interval: int = 5,
        max_candles: int = 500,
        use_circles: bool = False,
        logs_dir: str = "logs",
        performance_monitor=None,
    ):
        self.client = client
        self.strategy = strategy
        self.portfolio = portfolio
        self.instrument = instrument
        self.granularity = granularity
        self.ppst_params = ppst_params or {}
        self.time_filter_params = time_filter_params or {}
        self.poll_interval = poll_interval
        self.max_candles = max_candles
        self.use_circles = use_circles
        self.logs_dir = logs_dir

        self.order_manager = OrderManager(client, instrument)
        self.performance_monitor = performance_monitor
        self.position: Optional[Position] = None
        self.trade_id: Optional[str] = None
        self.prev_sl: Optional[float] = None
        self.last_processed_time: Optional[pd.Timestamp] = None
        self.running = False

        # Session stats
        self.session_trades = 0
        self.session_pnl = 0.0
        self.session_start_balance = portfolio.balance

        # Set up trade log CSV
        os.makedirs(self.logs_dir, exist_ok=True)
        self.trades_csv = os.path.join(self.logs_dir, "trades.csv")
        self._init_trades_csv()

    def _init_trades_csv(self) -> None:
        """Initialize trades CSV with header if it doesn't exist."""
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "event", "trade_id", "direction", "entry_price",
                    "units", "sl_price", "tp_price", "close_price", "close_reason",
                    "pnl", "balance",
                ])

    def _log_trade_event(self, event: str, **kwargs) -> None:
        """Append a trade event to the CSV log."""
        with open(self.trades_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                event,
                kwargs.get("trade_id", self.trade_id or ""),
                kwargs.get("direction", ""),
                kwargs.get("entry_price", ""),
                kwargs.get("units", ""),
                kwargs.get("sl_price", ""),
                kwargs.get("tp_price", ""),
                kwargs.get("close_price", ""),
                kwargs.get("close_reason", ""),
                kwargs.get("pnl", ""),
                kwargs.get("balance", self.portfolio.balance),
            ])

    def run(self) -> None:
        """Main trading loop. Polls for new candles and executes strategy."""
        self.running = True
        logger.info(f"Starting paper trader: {self.instrument} {self.granularity}")
        logger.info(f"PPST params: {self.ppst_params}")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info(f"Account balance: ${self.portfolio.balance:,.2f}")
        logger.info(f"Risk per trade: {self.portfolio.risk_percent}%")

        # Load performance monitor state
        if self.performance_monitor is not None:
            self.performance_monitor.load_state()
            self.portfolio.risk_percent = self.performance_monitor.current_risk_percent
            logger.info(f"Monitor active: risk={self.performance_monitor.current_risk_percent:.2f}%")

        # Try to recover existing position from OANDA
        self._recover_position()

        while self.running:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Stopping paper trader...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)

            time.sleep(self.poll_interval)

    def stop(self) -> None:
        """Stop the trading loop."""
        self.running = False

    def get_session_summary(self) -> dict:
        """Return session statistics."""
        return {
            "trades": self.session_trades,
            "pnl": self.session_pnl,
            "start_balance": self.session_start_balance,
            "end_balance": self.portfolio.balance,
            "return_pct": ((self.portfolio.balance - self.session_start_balance)
                          / self.session_start_balance * 100)
                          if self.session_start_balance > 0 else 0.0,
        }

    def _recover_position(self) -> None:
        """Try to recover an existing OANDA position on startup."""
        oanda_pos = self.order_manager.get_current_position()
        if oanda_pos is None:
            logger.info("No existing position on OANDA — starting flat")
            return

        self.trade_id = oanda_pos["trade_id"]
        direction = oanda_pos["direction"]
        entry_price = oanda_pos["entry_price"]
        units = oanda_pos["units"]
        sl_price = oanda_pos.get("sl_price")
        tp_price = oanda_pos.get("tp_price")

        self.position = Position(
            direction=direction,
            entry_price=entry_price,
            units=units,
            sl_price=sl_price if sl_price is not None else entry_price,
            tp_price=tp_price,
        )
        self.prev_sl = sl_price

        logger.info(
            f"Recovered position: {'LONG' if direction == 1 else 'SHORT'} "
            f"{units} units @ {format_price(entry_price)}, "
            f"SL={format_price(sl_price) if sl_price else 'None'}, "
            f"TP={format_price(tp_price) if tp_price else 'None'}, "
            f"trade_id={self.trade_id}"
        )

    def _tick(self) -> None:
        """Single iteration: fetch candles, compute indicators, run strategy."""
        # Fetch recent candles
        df = self.client.get_latest_candles(
            self.instrument, self.granularity, self.max_candles
        )

        if df.empty:
            return

        # Only process complete candles
        complete = df[df["complete"] == True].copy()
        if complete.empty:
            return

        # Check for new candle
        latest_time = complete["time"].iloc[-1]
        if self.last_processed_time is not None and latest_time <= self.last_processed_time:
            return

        self.last_processed_time = latest_time

        # Drop 'complete' column for indicator computation
        complete = complete.drop(columns=["complete"])

        # Compute indicators
        data = compute_ppst(complete, **self.ppst_params)

        if self.use_circles:
            circles = compute_entry_circles(data)
            for col in circles.columns:
                data[col] = circles[col].values

        tf = compute_time_filter(data["time"], **self.time_filter_params)
        data["can_trade"] = tf["can_trade"].values
        data["entering_quiet"] = tf["entering_quiet"].values

        # Process latest bar
        idx = len(data) - 1
        row = data.iloc[idx]
        can_trade = bool(row["can_trade"])
        entering_quiet = bool(row["entering_quiet"])

        # Sync position state with OANDA
        oanda_pos = self.order_manager.get_current_position()
        if oanda_pos is None and self.position is not None:
            # Position was closed externally (SL/TP hit on OANDA)
            self._handle_external_close()
        elif oanda_pos is not None and self.position is None and self.trade_id is None:
            # Untracked position exists — recover it
            logger.warning("Untracked position found on OANDA — recovering")
            self._recover_position()
            return

        # Run strategy
        action = self.strategy.on_bar(idx, row, self.position, can_trade, entering_quiet)
        action_type = action.get("action", "none")

        logger.debug(
            f"Bar {row['time']}: close={row['close']:.5f}, "
            f"trend={row['trend']}, can_trade={can_trade}, action={action_type}"
        )

        # Sync SL changes to OANDA (trailing SL / break-even)
        if self.position is not None and self.trade_id is not None:
            self._sync_sl_to_oanda()

        # Execute actions
        if action_type == "close" and self.position is not None:
            self._execute_close(action.get("close_reason", "strategy_close"))

        elif action_type in ("open_long", "open_short"):
            if self.position is None:
                self._execute_open(action, row)

        elif action_type in ("reverse_to_long", "reverse_to_short"):
            if self.position is not None:
                self._execute_close("reversal")
            self._execute_open(action, row)

    def _sync_sl_to_oanda(self) -> None:
        """Sync local SL changes (trailing/break-even) to OANDA."""
        current_sl = self.position.sl_price
        if self.prev_sl is not None and abs(current_sl - self.prev_sl) < 1e-6:
            return  # No change

        logger.info(
            f"SL changed: {format_price(self.prev_sl) if self.prev_sl else 'None'} "
            f"-> {format_price(current_sl)}"
        )
        result = self.order_manager.modify_sl(self.trade_id, current_sl)
        if result:
            self.prev_sl = current_sl
            self._log_trade_event(
                "SL_UPDATE",
                sl_price=format_price(current_sl),
            )

    def _handle_external_close(self) -> None:
        """Handle position closed externally (SL/TP hit on OANDA side)."""
        logger.info("Position closed externally (SL/TP hit on OANDA)")

        # Sync balance from OANDA account
        pnl = self._sync_balance()

        self._log_trade_event(
            "CLOSE",
            direction="LONG" if self.position.is_long else "SHORT",
            entry_price=format_price(self.position.entry_price),
            units=self.position.units,
            close_reason="external_sl_tp",
            balance=self.portfolio.balance,
        )

        self.session_trades += 1
        self._notify_monitor(pnl)
        self.position = None
        self.trade_id = None
        self.prev_sl = None

    def _sync_balance(self) -> float:
        """Fetch current OANDA account balance and sync portfolio.

        Returns:
            PnL delta from the balance sync (0.0 on error).
        """
        try:
            account = self.client.get_account()
            account_info = account.get("account", {})
            new_balance = float(account_info.get("balance", self.portfolio.balance))
            old_balance = self.portfolio.balance
            pnl = new_balance - old_balance

            self.portfolio.balance = new_balance
            self.portfolio.peak_balance = max(self.portfolio.peak_balance, new_balance)
            self.session_pnl += pnl

            logger.info(
                f"Balance synced: ${old_balance:,.2f} -> ${new_balance:,.2f} "
                f"(PnL: ${pnl:+,.2f})"
            )
            return pnl
        except Exception as e:
            logger.error(f"Failed to sync balance: {e}")
            return 0.0

    def _execute_close(self, reason: str) -> None:
        """Close current position on OANDA."""
        self.order_manager.close_all_positions()

        # Sync balance after close
        pnl = self._sync_balance()

        self._log_trade_event(
            "CLOSE",
            direction="LONG" if self.position.is_long else "SHORT",
            entry_price=format_price(self.position.entry_price),
            units=self.position.units,
            close_reason=reason,
            balance=self.portfolio.balance,
        )

        logger.info(f"Closed position: {reason}")
        self.session_trades += 1
        self._notify_monitor(pnl)
        self.position = None
        self.trade_id = None
        self.prev_sl = None

    def _execute_open(self, action: dict, row: pd.Series) -> None:
        """Execute a new position open on OANDA."""
        is_long = "long" in action["action"]
        direction = 1 if is_long else -1
        close_price = row["close"]

        entry_price = self.portfolio.adjust_entry_for_spread(close_price, is_long)
        sl = action["sl"]

        if is_long:
            sl_distance = entry_price - sl
        else:
            sl_distance = sl - entry_price

        if sl_distance <= 0:
            return

        # Recalculate TP from adjusted entry
        tp = action.get("tp")
        if tp is not None:
            original_sl_dist = abs(close_price - sl)
            if original_sl_dist > 0:
                rr = abs(tp - close_price) / original_sl_dist
                if is_long:
                    tp = entry_price + sl_distance * rr
                else:
                    tp = entry_price - sl_distance * rr

        units = self.portfolio.compute_position_size(sl_distance)
        if units <= 0:
            return

        result = self.order_manager.open_position(direction, units, sl, tp)
        if result:
            self.trade_id = result.get("trade_id")
            self.position = Position(
                direction=direction,
                entry_price=entry_price,
                units=units,
                sl_price=sl,
                tp_price=tp,
                entry_time=str(row["time"]),
            )
            self.prev_sl = sl

            self._log_trade_event(
                "OPEN",
                trade_id=self.trade_id,
                direction="LONG" if is_long else "SHORT",
                entry_price=format_price(entry_price),
                units=int(units),
                sl_price=format_price(sl),
                tp_price=format_price(tp) if tp else "",
            )

            logger.info(
                f"Opened {'LONG' if is_long else 'SHORT'}: "
                f"entry={entry_price:.5f}, sl={sl:.5f}, "
                f"tp={tp:.5f if tp else 'None'}, units={units:.0f}, "
                f"trade_id={self.trade_id}"
            )

    def _notify_monitor(self, pnl: float) -> None:
        """Notify performance monitor of a trade close and update risk."""
        if self.performance_monitor is None:
            return

        new_risk = self.performance_monitor.on_trade_close(
            timestamp_utc=datetime.now(timezone.utc),
            pnl=pnl,
            balance=self.portfolio.balance,
            peak_balance=self.portfolio.peak_balance,
            drawdown=self.portfolio.drawdown,
        )
        self.portfolio.risk_percent = new_risk
        self.performance_monitor.save_state()
