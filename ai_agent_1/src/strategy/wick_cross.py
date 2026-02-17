"""Wick Cross SuperTrend Entry (WCSE) — pattern detection and state machine.

Three close-price entry patterns near SuperTrend support/resistance:
  A. Engulfing 2-candle (E2): pullback C1 near ST, C2 engulfs C1
  B. Engulfing 3-candle (E3): C1 near ST, C2+C3 net move engulfs C1
  C. Wick Cross (WX): wick pierces ST, body stays on trend side
"""

from dataclasses import dataclass

from src.utils.forex_utils import pips_to_price


@dataclass
class WickCrossState:
    """Tracks WCSE arm/disarm and entry-times gating per swing."""

    armed_dir: int = 0        # 1=LONG, -1=SHORT, 0=off
    tp_exits_count: int = 0   # only TP exits count toward entry_times limit

    def arm(self, direction: int) -> None:
        """Arm WCSE for a new swing direction. Resets TP count on direction change."""
        if direction != self.armed_dir:
            self.tp_exits_count = 0
        self.armed_dir = direction

    def disarm(self) -> None:
        """Disarm WCSE (trend change or actionable signal)."""
        self.armed_dir = 0
        self.tp_exits_count = 0

    def is_active(self, position_is_none: bool) -> bool:
        """True if armed and no open position."""
        return self.armed_dir != 0 and position_is_none

    def can_enter(self, entry_times: int) -> bool:
        """Check if entry-times limit allows another entry.

        entry_times=0 means unlimited.
        """
        if entry_times == 0:
            return True
        return self.tp_exits_count < entry_times

    def record_tp_hit(self) -> None:
        """Record a TP exit (counts toward entry_times limit)."""
        self.tp_exits_count += 1


def detect_engulf_2(
    c1_ohlc: tuple,
    c2_ohlc: tuple,
    st: float,
    direction: int,
    c1_close_pips: float = 0.8,
    c1_body_pips: float = 0.1,
    c2_body_pips: float = 0.6,
) -> bool:
    """Pattern A: 2-candle engulfing near SuperTrend.

    LONG: c1 RED near TUp, c2 GREEN engulfs c1.
    SHORT: c1 GREEN near TDown, c2 RED engulfs c1.

    Args:
        c1_ohlc: (open, high, low, close) of bar i-1.
        c2_ohlc: (open, high, low, close) of bar i.
        st: SuperTrend value at c1's bar (TUp for long, TDown for short).
        direction: 1=LONG, -1=SHORT.
        c1_close_pips: Max distance from c1 close to ST.
        c1_body_pips: Min body size for c1.
        c2_body_pips: Min body size for c2.

    Returns:
        True if the 2-candle engulfing pattern matches.
    """
    c1_close_price = pips_to_price(c1_close_pips)
    c1_body_price = pips_to_price(c1_body_pips)
    c2_body_price = pips_to_price(c2_body_pips)

    o1, _, _, c1 = c1_ohlc
    o2, _, _, c2 = c2_ohlc

    if direction == 1:
        # c1: RED, body >= min, close near TUp
        if c1 >= o1:
            return False
        c1_body = o1 - c1
        if c1_body < c1_body_price:
            return False
        if abs(c1 - st) > c1_close_price:
            return False
        # c2: GREEN, body >= min, engulfs c1
        if c2 <= o2:
            return False
        c2_body = c2 - o2
        if c2_body < c2_body_price:
            return False
        if c2_body <= c1_body:
            return False
        return True
    else:
        # c1: GREEN, body >= min, close near TDown
        if c1 <= o1:
            return False
        c1_body = c1 - o1
        if c1_body < c1_body_price:
            return False
        if abs(c1 - st) > c1_close_price:
            return False
        # c2: RED, body >= min, engulfs c1
        if c2 >= o2:
            return False
        c2_body = o2 - c2
        if c2_body < c2_body_price:
            return False
        if c2_body <= c1_body:
            return False
        return True


def detect_engulf_3(
    c1_ohlc: tuple,
    c2_ohlc: tuple,
    c3_ohlc: tuple,
    st: float,
    direction: int,
    c1_close_pips: float = 0.8,
    c1_body_pips: float = 0.1,
    c2_body_pips: float = 0.6,
) -> bool:
    """Pattern B: 3-candle engulfing near SuperTrend.

    LONG: c1 (i-2) RED near TUp, c3 (i) GREEN, net move c3.close - c2.open > c1 body.
    SHORT: c1 (i-2) GREEN near TDown, c3 (i) RED, net move c2.open - c3.close > c1 body.

    Args:
        c1_ohlc: (open, high, low, close) of bar i-2.
        c2_ohlc: (open, high, low, close) of bar i-1.
        c3_ohlc: (open, high, low, close) of bar i.
        st: SuperTrend value at c1's bar (i-2).
        direction: 1=LONG, -1=SHORT.
        c1_close_pips: Max distance from c1 close to ST.
        c1_body_pips: Min body size for c1.
        c2_body_pips: Min body size for c3.

    Returns:
        True if the 3-candle engulfing pattern matches.
    """
    c1_close_price = pips_to_price(c1_close_pips)
    c1_body_price = pips_to_price(c1_body_pips)
    c3_body_price = pips_to_price(c2_body_pips)

    o1, _, _, c1 = c1_ohlc
    o2, _, _, _ = c2_ohlc
    o3, _, _, c3 = c3_ohlc

    if direction == 1:
        # c1: RED, body >= min, close near TUp
        if c1 >= o1:
            return False
        c1_body = o1 - c1
        if c1_body < c1_body_price:
            return False
        if abs(c1 - st) > c1_close_price:
            return False
        # c3: GREEN, body >= min
        if c3 <= o3:
            return False
        c3_body = c3 - o3
        if c3_body < c3_body_price:
            return False
        # net move: c3.close - c2.open > c1 body
        net_move = c3 - o2
        if net_move <= c1_body:
            return False
        return True
    else:
        # c1: GREEN, body >= min, close near TDown
        if c1 <= o1:
            return False
        c1_body = c1 - o1
        if c1_body < c1_body_price:
            return False
        if abs(c1 - st) > c1_close_price:
            return False
        # c3: RED, body >= min
        if c3 >= o3:
            return False
        c3_body = o3 - c3
        if c3_body < c3_body_price:
            return False
        # net move: c2.open - c3.close > c1 body
        net_move = o2 - c3
        if net_move <= c1_body:
            return False
        return True


def detect_wick_cross(
    candle_ohlc: tuple,
    st: float,
    direction: int,
    wx_st_pips: float = 0.4,
) -> bool:
    """Pattern C: wick pierces SuperTrend, body stays on trend side.

    LONG: GREEN candle, low < TUp, TUp - low >= wx_st_pips, close > TUp.
    SHORT: RED candle, high > TDown, high - TDown >= wx_st_pips, close < TDown.

    Args:
        candle_ohlc: (open, high, low, close) of current bar.
        st: SuperTrend value (TUp for long, TDown for short).
        direction: 1=LONG, -1=SHORT.
        wx_st_pips: Min wick depth through SuperTrend.

    Returns:
        True if the wick cross pattern matches.
    """
    wx_st_price = pips_to_price(wx_st_pips)
    o, h, l, c = candle_ohlc

    if direction == 1:
        # GREEN candle, wick below TUp, depth >= threshold, close above TUp
        if c <= o:
            return False
        if l >= st:
            return False
        if st - l < wx_st_price:
            return False
        if c <= st:
            return False
        return True
    else:
        # RED candle, wick above TDown, depth >= threshold, close below TDown
        if c >= o:
            return False
        if h <= st:
            return False
        if h - st < wx_st_price:
            return False
        if c >= st:
            return False
        return True
