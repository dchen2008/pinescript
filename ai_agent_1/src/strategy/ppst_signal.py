"""Signal-swing entry strategy.

Port of ppst_official_bt.pine entry logic (lines 231-259).

Entry: On buy/sell signal (trend change) when canTrade=true
SL: SuperTrend trailing line (TUp for long, TDown for short)
TP: RR-based from actual SL distance
Trail SL: with SuperTrend, only favorable direction
Reversal: sell signal while long -> close long + open short
Close on quiet window entry.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base_strategy import BaseStrategy
from src.strategy.position import Position


class PPSTSignalStrategy(BaseStrategy):
    """Signal-swing entry strategy from ppst_official_bt.pine."""

    def __init__(
        self,
        use_rr_tp: bool = True,
        rr_ratio: float = 1.5,
        trail_with_supertrend: bool = True,
        only_long: bool = False,
    ):
        self.use_rr_tp = use_rr_tp
        self.rr_ratio = rr_ratio
        self.trail_with_supertrend = trail_with_supertrend
        self.only_long = only_long

    def on_bar(
        self,
        idx: int,
        row: pd.Series,
        position: Optional[Position],
        can_trade: bool,
        entering_quiet: bool,
    ) -> dict:
        result = {"action": "none"}

        buy_signal = bool(row.get("buy_signal", False))
        sell_signal = bool(row.get("sell_signal", False))
        tup = row["tup"]
        tdown = row["tdown"]
        trailing_sl_val = row["trailing_sl"]

        # Close on quiet window entry
        if entering_quiet and position is not None:
            return {"action": "close", "close_reason": "Quiet Close"}

        # Trail SL with SuperTrend (only favorable direction)
        # IMPORTANT: Use TUp for longs, TDown for shorts — NOT trailing_sl
        # (trailing_sl = TDown when trend=-1, which would jump long SL above entry)
        if self.trail_with_supertrend and position is not None:
            if position.is_long:
                position.trail_sl_with_supertrend(tup)
            else:
                position.trail_sl_with_supertrend(tdown)

        # Entry logic: reversal strategy
        # Buy signal while in short -> close short via reversal open
        # Sell signal while in long -> close long via reversal open
        if buy_signal and can_trade:
            close_price = row["close"]
            sl = tup

            if np.isnan(sl):
                return result

            # Spread adjustment done in engine
            sl_distance = close_price - sl
            if sl_distance <= 0:
                return result

            tp = None
            if self.use_rr_tp:
                tp = close_price + sl_distance * self.rr_ratio

            # If in short position, this is a reversal
            if position is not None and position.is_short:
                return {
                    "action": "reverse_to_long",
                    "sl": sl,
                    "tp": tp,
                    "close_reason": "Reversal",
                }
            elif position is None:
                return {
                    "action": "open_long",
                    "sl": sl,
                    "tp": tp,
                }

        if sell_signal and not self.only_long and can_trade:
            close_price = row["close"]
            sl = tdown

            if np.isnan(sl):
                return result

            sl_distance = sl - close_price
            if sl_distance <= 0:
                return result

            tp = None
            if self.use_rr_tp:
                tp = close_price - sl_distance * self.rr_ratio

            # If in long position, this is a reversal
            if position is not None and position.is_long:
                return {
                    "action": "reverse_to_short",
                    "sl": sl,
                    "tp": tp,
                    "close_reason": "Reversal",
                }
            elif position is None:
                return {
                    "action": "open_short",
                    "sl": sl,
                    "tp": tp,
                }

        return result
