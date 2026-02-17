"""Position lifecycle: SL/TP/trailing/break-even management."""

from dataclasses import dataclass, field
from typing import Optional

from src.utils.forex_utils import pips_to_price, price_to_pips


@dataclass
class Position:
    """Represents an open trading position."""

    direction: int  # 1 = long, -1 = short
    entry_price: float
    units: float
    sl_price: float
    tp_price: Optional[float] = None
    original_sl: Optional[float] = None
    be_triggered: bool = False
    supertrend_trailing: bool = False
    entry_bar: int = 0
    entry_time: Optional[str] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None
    close_bar: Optional[int] = None
    pnl: float = 0.0
    entry_source: str = "signal"  # "signal" or "wcse"
    entry_pattern: str = ""       # "E2", "E3", "WX", or "" (signal)
    wx_entry_bar: int = 0         # bar_index when WX entry placed

    def __post_init__(self):
        if self.original_sl is None:
            self.original_sl = self.sl_price

    @property
    def is_long(self) -> bool:
        return self.direction == 1

    @property
    def is_short(self) -> bool:
        return self.direction == -1

    def check_sl_hit(self, high: float, low: float) -> bool:
        """Check if SL was hit during this bar."""
        if self.is_long:
            return low <= self.sl_price
        else:
            return high >= self.sl_price

    def check_tp_hit(self, high: float, low: float) -> bool:
        """Check if TP was hit during this bar."""
        if self.tp_price is None:
            return False
        if self.is_long:
            return high >= self.tp_price
        else:
            return low <= self.tp_price

    def trail_sl_with_supertrend(
        self,
        trailing_sl: float,
        sl_buffer_pips: float = 0.0,
    ) -> None:
        """Trail SL using SuperTrend, only moves favorably.

        For official strategy: direct SuperTrend trailing (no buffer).
        For circle strategy: SuperTrend trailing with buffer, only when better than original.
        """
        import math
        if math.isnan(trailing_sl):
            return

        if sl_buffer_pips > 0:
            # Circle strategy: SuperTrend trailing with buffer
            buffer = pips_to_price(sl_buffer_pips)
            if self.is_long:
                new_sl = trailing_sl - buffer
                if new_sl > self.original_sl and new_sl > self.sl_price:
                    self.supertrend_trailing = True
                    self.sl_price = new_sl
            else:
                new_sl = trailing_sl + buffer
                if new_sl < self.original_sl and new_sl < self.sl_price:
                    self.supertrend_trailing = True
                    self.sl_price = new_sl
        else:
            # Official strategy: direct trailing (only favorable direction)
            if self.is_long:
                self.sl_price = max(self.sl_price, trailing_sl)
            else:
                self.sl_price = min(self.sl_price, trailing_sl)

    def trail_sl_raw(self, new_sl: float) -> None:
        """Trail SL, only tightening, no original_sl guard (for WX after N bars)."""
        if self.is_long and new_sl > self.sl_price:
            self.sl_price = new_sl
        elif self.is_short and new_sl < self.sl_price:
            self.sl_price = new_sl

    def check_break_even(
        self,
        close: float,
        be_trigger_pips: float,
        be_offset_pips: float,
    ) -> None:
        """Check and apply break-even logic."""
        if self.be_triggered:
            return

        profit_pips = price_to_pips(
            (close - self.entry_price) if self.is_long
            else (self.entry_price - close)
        )

        if profit_pips >= be_trigger_pips:
            self.be_triggered = True
            be_price = self.entry_price + self.direction * pips_to_price(be_offset_pips)

            if self.is_long:
                self.sl_price = max(self.sl_price, be_price)
            else:
                self.sl_price = min(self.sl_price, be_price)

    def close_position(self, price: float, reason: str, bar_idx: int) -> float:
        """Close the position and compute PnL.

        Returns:
            PnL in price units (multiply by units for dollar PnL)
        """
        self.close_price = price
        self.close_reason = reason
        self.close_bar = bar_idx

        if self.is_long:
            price_pnl = price - self.entry_price
        else:
            price_pnl = self.entry_price - price

        self.pnl = price_pnl * self.units
        return self.pnl

    def to_dict(self) -> dict:
        """Convert to dict for trade log."""
        return {
            "direction": "LONG" if self.is_long else "SHORT",
            "entry_price": self.entry_price,
            "units": self.units,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "original_sl": self.original_sl,
            "close_price": self.close_price,
            "close_reason": self.close_reason,
            "entry_bar": self.entry_bar,
            "close_bar": self.close_bar,
            "pnl": self.pnl,
            "be_triggered": self.be_triggered,
            "supertrend_trailing": self.supertrend_trailing,
            "entry_source": self.entry_source,
            "entry_pattern": self.entry_pattern,
        }
