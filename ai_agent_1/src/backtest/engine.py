"""Bar-by-bar backtest engine.

Processes each bar:
1. Check SL/TP fills on H/L
2. Run strategy
3. Process actions (open/close/reverse)
4. Record equity
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.indicators.ppst import compute_ppst
from src.indicators.entry_circles import compute_entry_circles
from src.indicators.volume_filter import compute_relative_volume
from src.indicators.vwap import compute_session_vwap
from src.indicators.rsi import compute_rsi
from src.strategy.base_strategy import BaseStrategy
from src.strategy.time_filter import compute_time_filter
from src.strategy.position import Position
from src.backtest.portfolio import Portfolio


class BacktestEngine:
    """Bar-by-bar backtest loop."""

    def __init__(
        self,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        ppst_params: Optional[dict] = None,
        time_filter_params: Optional[dict] = None,
        conservative_fills: bool = True,
        use_circles: bool = False,
        strategy_type: str = "ppst",
        vwap_params: Optional[dict] = None,
    ):
        self.strategy = strategy
        self.portfolio = portfolio
        self.ppst_params = ppst_params or {}
        self.time_filter_params = time_filter_params or {}
        self.conservative_fills = conservative_fills
        self.use_circles = use_circles
        self.strategy_type = strategy_type
        self.vwap_params = vwap_params or {}

        self.position: Optional[Position] = None
        self.trade_log: list = []
        self.bar_count = 0

    def run(self, df: pd.DataFrame) -> dict:
        """Run backtest on candle data.

        Args:
            df: DataFrame with columns: time, open, high, low, close, volume

        Returns:
            Dict with results: trade_log, equity_curve, stats
        """
        if self.strategy_type == "vwap":
            data = self._prepare_vwap_data(df)
            warmup = self.vwap_params.get("rsi_period", 14) + 2
        else:
            data = self._prepare_ppst_data(df)
            pivot_period = self.ppst_params.get("pivot_period", 2)
            atr_period = self.ppst_params.get("atr_period", 10)
            warmup = max(pivot_period * 2 + 1, atr_period + 1, 20)

        # Compute time filter
        tf = compute_time_filter(data["time"], **self.time_filter_params)
        data["can_trade"] = tf["can_trade"].values
        data["entering_quiet"] = tf["entering_quiet"].values

        n = len(data)
        self.bar_count = n

        # Step 5: Bar-by-bar loop
        for i in range(warmup, n):
            row = data.iloc[i]
            high = row["high"]
            low = row["low"]
            close = row["close"]
            can_trade = bool(row["can_trade"])
            entering_quiet = bool(row["entering_quiet"])

            # --- Check SL/TP fills on current bar's H/L ---
            if self.position is not None:
                sl_hit = self.position.check_sl_hit(high, low)
                tp_hit = self.position.check_tp_hit(high, low)

                # Compute realistic fill prices accounting for gaps
                # If SL/TP is gapped through at open, fill at open (not at stop/limit)
                open_price = row["open"]
                sl_fill = self.position.sl_price
                tp_fill = self.position.tp_price if self.position.tp_price else None

                if self.position.is_long:
                    # Long SL: sell stop. If open gaps below SL, fill at open (worse)
                    if sl_hit and open_price <= self.position.sl_price:
                        sl_fill = open_price
                    # Long TP: sell limit. If open gaps above TP, fill at open (better)
                    if tp_hit and tp_fill is not None and open_price >= tp_fill:
                        tp_fill = open_price
                else:
                    # Short SL: buy stop. If open gaps above SL, fill at open (worse)
                    if sl_hit and open_price >= self.position.sl_price:
                        sl_fill = open_price
                    # Short TP: buy limit. If open gaps below TP, fill at open (better)
                    if tp_hit and tp_fill is not None and open_price <= tp_fill:
                        tp_fill = open_price

                if sl_hit and tp_hit:
                    # Conservative: assume SL hit first
                    if self.conservative_fills:
                        self._close_position(sl_fill, "SL Hit", i)
                    else:
                        self._close_position(tp_fill, "TP Hit", i)
                elif sl_hit:
                    self._close_position(sl_fill, "SL Hit", i)
                elif tp_hit:
                    self._close_position(tp_fill, "TP Hit", i)

            # --- Run strategy ---
            action = self.strategy.on_bar(i, row, self.position, can_trade, entering_quiet)

            # --- Process actions ---
            action_type = action.get("action", "none")

            if action_type == "close" and self.position is not None:
                self._close_position(close, action.get("close_reason", "Strategy Close"), i)

            elif action_type == "open_long" and self.position is None:
                self._open_position(1, close, action.get("sl"), action.get("tp"), i, row)

            elif action_type == "open_short" and self.position is None:
                self._open_position(-1, close, action.get("sl"), action.get("tp"), i, row)

            elif action_type == "reverse_to_long":
                if self.position is not None:
                    self._close_position(close, action.get("close_reason", "Reversal"), i)
                self._open_position(1, close, action.get("sl"), action.get("tp"), i, row)

            elif action_type == "reverse_to_short":
                if self.position is not None:
                    self._close_position(close, action.get("close_reason", "Reversal"), i)
                self._open_position(-1, close, action.get("sl"), action.get("tp"), i, row)

            # Record equity
            unrealized = 0.0
            if self.position is not None:
                if self.position.is_long:
                    unrealized = (close - self.position.entry_price) * self.position.units
                else:
                    unrealized = (self.position.entry_price - close) * self.position.units
            self.portfolio.record_equity(i, unrealized)

        # Close any remaining position at last bar
        if self.position is not None:
            last_close = data.iloc[-1]["close"]
            self._close_position(last_close, "End of Data", n - 1)

        return {
            "trade_log": self.trade_log,
            "equity_curve": self.portfolio.equity_curve,
            "total_bars": n,
            "warmup_bars": warmup,
        }

    def _open_position(
        self,
        direction: int,
        close_price: float,
        sl: float,
        tp: Optional[float],
        bar_idx: int,
        row: pd.Series,
    ) -> None:
        """Open a new position with spread adjustment and compound sizing."""
        is_long = direction == 1
        entry_price = self.portfolio.adjust_entry_for_spread(close_price, is_long)

        # Recalculate SL distance after spread adjustment
        if is_long:
            sl_distance = entry_price - sl
        else:
            sl_distance = sl - entry_price

        if sl_distance <= 0:
            return

        # Recalculate TP if RR-based (to account for spread-adjusted entry)
        if tp is not None:
            # Recalc TP from adjusted entry
            if is_long:
                # Original tp was close + sl_dist * rr. Recalc with new entry.
                # We keep same RR ratio: tp = entry + sl_distance * rr_ratio
                # But we don't know rr_ratio here, so recompute from original intent
                original_sl_dist = close_price - sl
                if original_sl_dist > 0:
                    rr = (tp - close_price) / original_sl_dist
                    tp = entry_price + sl_distance * rr
            else:
                original_sl_dist = sl - close_price
                if original_sl_dist > 0:
                    rr = (close_price - tp) / original_sl_dist
                    tp = entry_price - sl_distance * rr

        units = self.portfolio.compute_position_size(sl_distance)
        if units <= 0:
            return

        self.position = Position(
            direction=direction,
            entry_price=entry_price,
            units=units,
            sl_price=sl,
            tp_price=tp,
            entry_bar=bar_idx,
            entry_time=str(row.get("time", "")),
        )

    def _prepare_ppst_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute PPST indicator and optional circles."""
        ppst_only = {k: v for k, v in self.ppst_params.items() if k != "volume_filter_period"}
        data = compute_ppst(df, **ppst_only)
        if self.use_circles:
            circles = compute_entry_circles(data)
            for col in circles.columns:
                data[col] = circles[col].values

        # Compute relative volume for optional volume filter
        vol_period = self.ppst_params.get("volume_filter_period", 20)
        if "volume" in df.columns:
            data["rel_volume"] = compute_relative_volume(df["volume"].values, vol_period)

        return data

    def _prepare_vwap_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute session VWAP + RSI indicators."""
        data = df.copy()
        vwap_result = compute_session_vwap(
            df,
            session_start_hour_utc=self.vwap_params.get("session_start_utc", 8),
            band_mult=self.vwap_params.get("vwap_band_mult", 1.5),
        )
        data["vwap"] = vwap_result["vwap"].values
        data["vwap_upper"] = vwap_result["vwap_upper"].values
        data["vwap_lower"] = vwap_result["vwap_lower"].values

        rsi_period = self.vwap_params.get("rsi_period", 14)
        data["rsi"] = compute_rsi(df["close"].values, period=rsi_period)
        return data

    def _close_position(self, price: float, reason: str, bar_idx: int) -> None:
        """Close current position and record trade."""
        if self.position is None:
            return

        pnl = self.position.close_position(price, reason, bar_idx)
        self.portfolio.record_trade(pnl)
        self.trade_log.append(self.position.to_dict())
        self.position = None
