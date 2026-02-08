"""Circle entry strategy.

Port of ppst_v1_bt.pine entry logic.

Multiple entry modes:
- Signal Swing: enter on any signal swing
- Signal + Circle Same Bar: signal and circle on same bar
- Circle within Time Window: circle within X minutes of signal
- Nth Circle: enter on specific circle number after signal

SL: Pivot/SuperTrend midpoint method with buffer
TP: RR-based, fixed pips, or none
Break-even: move SL to entry + offset when profit reaches trigger
Trailing: SuperTrend takes over when it passes original SL
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base_strategy import BaseStrategy
from src.strategy.position import Position
from src.utils.forex_utils import pips_to_price, price_to_pips


class PPSTCircleStrategy(BaseStrategy):
    """Circle entry strategy from ppst_v1_bt.pine."""

    def __init__(
        self,
        entry_mode: str = "Signal Swing",
        entry_window_min: int = 30,
        entry_circle_num: int = 1,
        max_entry_dist_pips: float = 8.5,
        sl_buffer_pips: float = 1.0,
        tp_type: str = "Risk Ratio",
        tp_ratio: float = 1.5,
        tp_fixed_pips: float = 30.0,
        be_enabled: bool = True,
        be_trigger_pips: float = 15.0,
        be_offset_pips: float = 2.0,
        timeframe_minutes: int = 1,
        volume_threshold: float = 0.0,
    ):
        self.entry_mode = entry_mode
        self.entry_window_min = entry_window_min
        self.entry_circle_num = entry_circle_num
        self.max_entry_dist_pips = max_entry_dist_pips
        self.sl_buffer_pips = sl_buffer_pips
        self.tp_type = tp_type
        self.tp_ratio = tp_ratio
        self.tp_fixed_pips = tp_fixed_pips
        self.be_enabled = be_enabled
        self.be_trigger_pips = be_trigger_pips
        self.be_offset_pips = be_offset_pips
        self.timeframe_minutes = timeframe_minutes
        self.volume_threshold = volume_threshold

    def _calc_pivot_sl(self, is_long: bool, pivot_level: float, supertrend: float) -> float:
        """Calculate SL using pivot and supertrend midpoint method.

        Port of ppst_v1_bt.pine calc_pivot_sl() (lines 106-124).
        """
        buffer = pips_to_price(self.sl_buffer_pips)

        if is_long:
            if pivot_level < supertrend:
                return supertrend - buffer
            midpoint = (pivot_level + supertrend) / 2
            return max(midpoint, supertrend) - buffer
        else:
            if pivot_level > supertrend:
                return supertrend + buffer
            midpoint = (pivot_level + supertrend) / 2
            return min(midpoint, supertrend) + buffer

    def _is_valid_entry_distance(self, is_long: bool, open_price: float, supertrend: float) -> bool:
        """Check if entry is within max distance from SuperTrend."""
        max_dist = pips_to_price(self.max_entry_dist_pips)
        if is_long:
            return open_price <= supertrend + max_dist
        else:
            return open_price >= supertrend - max_dist

    def _compute_tp(self, entry_price: float, sl_distance: float, is_long: bool) -> Optional[float]:
        """Compute TP price based on tp_type."""
        if self.tp_type == "Risk Ratio":
            if is_long:
                return entry_price + sl_distance * self.tp_ratio
            else:
                return entry_price - sl_distance * self.tp_ratio
        elif self.tp_type == "Fixed Pips":
            fixed = pips_to_price(self.tp_fixed_pips)
            if is_long:
                return entry_price + fixed
            else:
                return entry_price - fixed
        return None

    def _determine_entry(self, row: pd.Series, idx: int) -> tuple:
        """Determine if entry should fire based on mode.

        Returns: (should_enter_long, should_enter_short)
        """
        buy_signal = bool(row.get("buy_signal", False))
        sell_signal = bool(row.get("sell_signal", False))
        trend = int(row["trend"])
        long_circle = bool(row.get("long_circle", False))
        short_circle = bool(row.get("short_circle", False))
        last_signal_bar = int(row.get("last_signal_bar", 0))
        circle_count = int(row.get("circle_count_since_signal", 0))

        if self.entry_mode == "Signal Swing":
            return buy_signal, sell_signal

        elif self.entry_mode == "Signal + Circle Same Bar":
            return (buy_signal and long_circle), (sell_signal and short_circle)

        elif self.entry_mode == "Circle within Time Window":
            minutes_since = (idx - last_signal_bar) * self.timeframe_minutes
            long_ok = long_circle and minutes_since <= self.entry_window_min and trend == 1
            short_ok = short_circle and minutes_since <= self.entry_window_min and trend == -1
            return long_ok, short_ok

        elif self.entry_mode == "Nth Circle":
            long_ok = long_circle and circle_count == self.entry_circle_num and trend == 1
            short_ok = short_circle and circle_count == self.entry_circle_num and trend == -1
            return long_ok, short_ok

        return False, False

    def on_bar(
        self,
        idx: int,
        row: pd.Series,
        position: Optional[Position],
        can_trade: bool,
        entering_quiet: bool,
    ) -> dict:
        result = {"action": "none"}

        # Trail SL with SuperTrend (with buffer, only when better than original)
        # CRITICAL: Use tup for longs, tdown for shorts — NOT trailing_sl which
        # flips bands on trend reversal and causes SL to jump to the wrong band.
        if position is not None:
            if position.is_long:
                trail_val = row["tup"]
            else:
                trail_val = row["tdown"]
            position.trail_sl_with_supertrend(trail_val, self.sl_buffer_pips)

            # Break-even check
            if self.be_enabled:
                position.check_break_even(
                    row["close"],
                    self.be_trigger_pips,
                    self.be_offset_pips,
                )

        # Only enter when no position (ppst_v1_bt doesn't do reversals)
        if position is not None:
            return result

        # Check entry conditions
        long_entry, short_entry = self._determine_entry(row, idx)

        # Volume confirmation filter: skip entry if volume below threshold
        if self.volume_threshold > 0 and (long_entry or short_entry):
            rel_vol = row.get("rel_volume", 0.0)
            if np.isnan(rel_vol) or rel_vol < self.volume_threshold:
                return result

        # At entry time, trailing_sl is correct (trend matches direction):
        # Long entries only happen during trend==1, so trailing_sl==tup
        # Short entries only happen during trend==-1, so trailing_sl==tdown
        trailing_sl_val = row["trailing_sl"]

        if long_entry and can_trade:
            open_price = row["open"]
            close_price = row["close"]
            support_val = row["support"]
            # Use tup directly for long entry SL calc (same as trailing_sl when trend==1)
            tup_val = row["tup"]

            if np.isnan(support_val) or np.isnan(tup_val):
                return result

            if not self._is_valid_entry_distance(True, open_price, tup_val):
                return result

            sl = self._calc_pivot_sl(True, support_val, tup_val)
            sl_distance = close_price - sl
            if sl_distance <= 0:
                return result

            tp = self._compute_tp(close_price, sl_distance, True)

            return {
                "action": "open_long",
                "sl": sl,
                "tp": tp,
            }

        if short_entry and can_trade:
            open_price = row["open"]
            close_price = row["close"]
            resistance_val = row["resistance"]
            # Use tdown directly for short entry SL calc (same as trailing_sl when trend==-1)
            tdown_val = row["tdown"]

            if np.isnan(resistance_val) or np.isnan(tdown_val):
                return result

            if not self._is_valid_entry_distance(False, open_price, tdown_val):
                return result

            sl = self._calc_pivot_sl(False, resistance_val, tdown_val)
            sl_distance = sl - close_price
            if sl_distance <= 0:
                return result

            tp = self._compute_tp(close_price, sl_distance, False)

            return {
                "action": "open_short",
                "sl": sl,
                "tp": tp,
            }

        return result
