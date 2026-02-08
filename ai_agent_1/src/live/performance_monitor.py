"""Autonomous performance monitor for live trading.

Tracks weekly returns, drawdown, and adaptively reduces risk during bad periods.
Acts as a governor — only reduces risk below baseline, never increases above it.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Optional

from src.utils.timezone_utils import to_pt

logger = logging.getLogger(__name__)


def get_forex_week_start(dt_pt) -> datetime:
    """Get the Sunday 2pm PT that starts this forex week.

    Replicates logic from src/backtest/metrics.py lines 163-177.
    """
    wd = dt_pt.weekday()  # Mon=0..Sun=6
    if wd == 6:  # Sunday
        if dt_pt.hour < 14:
            # Before market open — belongs to previous week
            base = dt_pt.replace(hour=14, minute=0, second=0, microsecond=0)
            return base - __import__("datetime").timedelta(days=7)
        else:
            return dt_pt.replace(hour=14, minute=0, second=0, microsecond=0)
    else:
        # Go back to most recent Sunday
        days_since_sunday = (wd + 1) % 7
        sunday = dt_pt - __import__("datetime").timedelta(days=days_since_sunday)
        return sunday.replace(hour=14, minute=0, second=0, microsecond=0)


class PerformanceMonitor:
    """Monitors live trading performance and adjusts risk adaptively.

    Risk rules (most conservative wins):
    1. DD Danger (>= 20%): 0.5%
    2. DD Caution (>= 10%): 0.75%
    3. Gain Protection (week return >= 8%): 0.75%
    4. 2-Week Losing Streak: 0.75%
    5. 1-Week Losing: 1.125%
    6. Baseline: 1.5%
    """

    def __init__(
        self,
        baseline_risk: float = 1.5,
        min_risk: float = 0.5,
        dd_caution_threshold: float = 0.10,
        dd_danger_threshold: float = 0.20,
        gain_protect_threshold: float = 8.0,
        gain_protect_risk: float = 0.75,
        logs_dir: str = "logs",
    ):
        self.baseline_risk = baseline_risk
        self.min_risk = min_risk
        self.dd_caution_threshold = dd_caution_threshold
        self.dd_danger_threshold = dd_danger_threshold
        self.gain_protect_threshold = gain_protect_threshold
        self.gain_protect_risk = gain_protect_risk
        self.logs_dir = logs_dir

        # Current state
        self.current_risk_percent = baseline_risk
        self.week_start_balance: Optional[float] = None
        self.week_pnl = 0.0
        self.current_week_key: Optional[str] = None
        self.prev_week_returns: list[float] = []  # last 4 completed weeks

        # Paths
        self.state_path = os.path.join(logs_dir, "monitor_state.json")
        self.audit_csv = os.path.join(logs_dir, "risk_adjustments.csv")

        os.makedirs(logs_dir, exist_ok=True)
        self._init_audit_csv()

    def _init_audit_csv(self) -> None:
        """Initialize audit CSV with header if it doesn't exist."""
        if not os.path.exists(self.audit_csv):
            with open(self.audit_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "event", "rule", "old_risk", "new_risk",
                    "drawdown", "week_return", "balance",
                ])

    def _log_audit(self, event: str, rule: str, old_risk: float,
                   new_risk: float, drawdown: float, week_return: float,
                   balance: float) -> None:
        """Append an entry to the risk adjustments audit CSV."""
        with open(self.audit_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                event, rule, f"{old_risk:.4f}", f"{new_risk:.4f}",
                f"{drawdown:.4f}", f"{week_return:.2f}", f"{balance:.2f}",
            ])

    def on_trade_close(self, timestamp_utc: datetime, pnl: float,
                       balance: float, peak_balance: float,
                       drawdown: float) -> float:
        """Called after every trade close. Returns new effective risk percent.

        Args:
            timestamp_utc: UTC time of trade close
            pnl: PnL of the closed trade
            balance: Current account balance after close
            peak_balance: Peak balance for drawdown calc
            drawdown: Current drawdown ratio (0.0 to 1.0)

        Returns:
            New risk percent to use
        """
        dt_pt = to_pt(timestamp_utc)
        week_key = get_forex_week_start(dt_pt).isoformat()

        # Handle week transition
        if self.current_week_key is not None and week_key != self.current_week_key:
            self._close_week(balance)

        # Initialize week if needed
        if self.current_week_key is None or week_key != self.current_week_key:
            self.current_week_key = week_key
            self.week_start_balance = balance - pnl  # balance before this trade
            self.week_pnl = 0.0

        # Record trade
        self.week_pnl += pnl

        # Compute week return
        week_return_pct = 0.0
        if self.week_start_balance and self.week_start_balance > 0:
            week_return_pct = (self.week_pnl / self.week_start_balance) * 100.0

        # Evaluate rules
        old_risk = self.current_risk_percent
        new_risk = self.compute_effective_risk(drawdown, week_return_pct)
        self.current_risk_percent = new_risk

        if abs(new_risk - old_risk) > 1e-6:
            rule = self._get_active_rule(drawdown, week_return_pct)
            logger.warning(
                f"Risk adjusted: {old_risk:.2f}% -> {new_risk:.2f}% "
                f"(rule: {rule}, DD: {drawdown:.1%}, week: {week_return_pct:+.1f}%)"
            )
            self._log_audit("RISK_CHANGE", rule, old_risk, new_risk,
                            drawdown, week_return_pct, balance)

        return new_risk

    def compute_effective_risk(self, drawdown: float,
                               week_return_pct: float = 0.0) -> float:
        """Evaluate all rules and return the minimum (most conservative) risk.

        Args:
            drawdown: Current drawdown ratio (0.0 to 1.0)
            week_return_pct: Current week return as percentage

        Returns:
            Effective risk percent
        """
        candidates = [self.baseline_risk]

        # Rule 1: DD Danger
        if drawdown >= self.dd_danger_threshold:
            candidates.append(self.min_risk)

        # Rule 2: DD Caution
        if drawdown >= self.dd_caution_threshold:
            candidates.append(0.75)

        # Rule 3: Gain Protection
        if week_return_pct >= self.gain_protect_threshold:
            candidates.append(self.gain_protect_risk)

        # Rule 4: 2-Week Losing Streak
        if len(self.prev_week_returns) >= 2:
            if self.prev_week_returns[-1] < 0 and self.prev_week_returns[-2] < 0:
                candidates.append(0.75)

        # Rule 5: 1-Week Losing
        if len(self.prev_week_returns) >= 1 and self.prev_week_returns[-1] < 0:
            candidates.append(self.baseline_risk * 0.75)  # 1.5 * 0.75 = 1.125

        return max(min(candidates), self.min_risk)

    def _get_active_rule(self, drawdown: float,
                         week_return_pct: float) -> str:
        """Return the name of the most restrictive active rule."""
        effective = self.compute_effective_risk(drawdown, week_return_pct)

        if drawdown >= self.dd_danger_threshold and effective <= self.min_risk:
            return "DD_DANGER"
        if drawdown >= self.dd_caution_threshold and effective <= 0.75:
            return "DD_CAUTION"
        if week_return_pct >= self.gain_protect_threshold and effective <= self.gain_protect_risk:
            return "GAIN_PROTECT"
        if (len(self.prev_week_returns) >= 2
                and self.prev_week_returns[-1] < 0
                and self.prev_week_returns[-2] < 0
                and effective <= 0.75):
            return "2_WEEK_LOSING"
        if (len(self.prev_week_returns) >= 1
                and self.prev_week_returns[-1] < 0
                and effective <= self.baseline_risk * 0.75):
            return "1_WEEK_LOSING"
        return "BASELINE"

    def _close_week(self, current_balance: float) -> None:
        """Finalize the current week and record its return."""
        if self.week_start_balance and self.week_start_balance > 0:
            week_return = (self.week_pnl / self.week_start_balance) * 100.0
        else:
            week_return = 0.0

        self.prev_week_returns.append(week_return)
        # Keep only last 4 weeks
        if len(self.prev_week_returns) > 4:
            self.prev_week_returns = self.prev_week_returns[-4:]

        logger.info(
            f"Week closed: PnL=${self.week_pnl:+,.2f}, "
            f"Return={week_return:+.2f}%, "
            f"Recent weeks: {[f'{r:+.1f}%' for r in self.prev_week_returns]}"
        )

    def save_state(self) -> None:
        """Persist monitor state to JSON for crash recovery."""
        state = {
            "current_risk_percent": self.current_risk_percent,
            "week_start_balance": self.week_start_balance,
            "week_pnl": self.week_pnl,
            "current_week_key": self.current_week_key,
            "prev_week_returns": self.prev_week_returns,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug(f"Monitor state saved to {self.state_path}")

    def load_state(self) -> bool:
        """Load monitor state from JSON. Returns True if state was loaded."""
        if not os.path.exists(self.state_path):
            logger.info("No monitor state file found — starting fresh")
            return False

        try:
            with open(self.state_path) as f:
                state = json.load(f)

            self.current_risk_percent = state.get("current_risk_percent", self.baseline_risk)
            self.week_start_balance = state.get("week_start_balance")
            self.week_pnl = state.get("week_pnl", 0.0)
            self.current_week_key = state.get("current_week_key")
            self.prev_week_returns = state.get("prev_week_returns", [])

            logger.info(
                f"Monitor state loaded: risk={self.current_risk_percent:.2f}%, "
                f"week_pnl=${self.week_pnl:+,.2f}, "
                f"prev_weeks={len(self.prev_week_returns)}"
            )
            return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load monitor state: {e} — starting fresh")
            return False
