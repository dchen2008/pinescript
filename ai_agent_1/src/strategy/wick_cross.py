"""Wick Cross SuperTrend Entry (WCSE) — pattern detection and state machine.

Detects a 3-candle pullback-rejection pattern near SuperTrend support/resistance
as an alternative entry when the normal signal gets volume-filtered.
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


def detect_wick_cross_long(
    o2: float, h2: float, l2: float, c2: float, tup2: float,
    o1: float, h1: float, l1: float, c1: float, tup1: float,
    oi: float, ci: float,
    c1_body_pips: float = 3.0,
    c1_wick_pips: float = 0.5,
    c1_close_pips: float = 1.0,
    c2_body_pips: float = 2.0,
    c2_wick_pips: float = 0.5,
    c2_close_pips: float = 1.0,
) -> bool:
    """Detect LONG wick-cross pattern near TUp (support).

    Args:
        o2,h2,l2,c2,tup2: 1st candle (bar i-2) OHLC + SuperTrend TUp
        o1,h1,l1,c1,tup1: 2nd candle (bar i-1) OHLC + SuperTrend TUp
        oi,ci: 3rd candle (bar i) open + close
        c1/c2 body/wick/close pips: pattern thresholds

    Returns:
        True if the 3-candle LONG pattern matches.
    """
    c1_body_price = pips_to_price(c1_body_pips)
    c1_wick_price = pips_to_price(c1_wick_pips)
    c1_close_price = pips_to_price(c1_close_pips)
    c2_body_price = pips_to_price(c2_body_pips)
    c2_wick_price = pips_to_price(c2_wick_pips)
    c2_close_price = pips_to_price(c2_close_pips)

    # --- 1st candle (bar i-2): RED, body >= threshold ---
    if c2 >= o2:  # not red
        return False
    body1 = o2 - c2
    if body1 < c1_body_price:
        return False

    # Touch or cross: (b) low within wick_pips of TUp OR close within close_pips of TUp
    #                 (c) low < TUp AND close > TUp (wick crossed, close didn't)
    touch_b = (l2 >= tup2) and ((l2 - tup2) <= c1_wick_price or abs(c2 - tup2) <= c1_close_price)
    cross_c = (l2 < tup2) and (c2 > tup2)
    if not (touch_b or cross_c):
        return False

    # --- 2nd candle (bar i-1): GREEN, body >= threshold ---
    if c1 <= o1:  # not green
        return False
    body2 = c1 - o1
    if body2 < c2_body_price:
        return False

    # Touch: low within wick_pips of TUp OR close within close_pips of TUp
    touch_2 = (l1 >= tup1) and ((l1 - tup1) <= c2_wick_price or abs(c1 - tup1) <= c2_close_price)
    if not touch_2:
        return False

    # No cross: low must stay >= TUp
    if l1 < tup1:
        return False

    # --- 3rd candle (bar i): GREEN (close > open) ---
    if ci <= oi:
        return False

    return True


def detect_engulfing_near_st(
    bars: list,        # [(o, h, l, c), ...] last 4 bars (i-3, i-2, i-1, i)
    st_values: list,   # ST value for each bar (TUp for long, TDown for short)
    direction: int,    # 1=LONG, -1=SHORT
    c_body_pips: float = 0.2,
    c_wick_pips: float = 5.0,
    c_close_pips: float = 5.0,
    engulf_body_pips: float = 1.0,
) -> bool:
    """Detect engulfing pattern near SuperTrend (WCSE2).

    Looks for a 2-4 candle pattern where the last bar engulfs 1-3 preceding
    small candles, at least one of which is near SuperTrend.

    Args:
        bars: Last 4 bars as (open, high, low, close) tuples.
        st_values: SuperTrend value for each bar (TUp for long, TDown for short).
        direction: 1 for LONG (bullish engulfing near TUp), -1 for SHORT.
        c_body_pips: Min body size for pre-engulfing candles (filters dojis).
        c_wick_pips: Wick proximity to ST for c1 identification.
        c_close_pips: Close proximity to ST for c1 identification.
        engulf_body_pips: Min body size for the engulfing candle.

    Returns:
        True if an engulfing pattern near ST is detected.
    """
    c_body_price = pips_to_price(c_body_pips)
    c_wick_price = pips_to_price(c_wick_pips)
    c_close_price = pips_to_price(c_close_pips)
    engulf_body_price = pips_to_price(engulf_body_pips)

    # Last bar = potential engulfing candle
    eng_o, eng_h, eng_l, eng_c = bars[-1]

    # Check color: GREEN for LONG, RED for SHORT
    if direction == 1 and eng_c <= eng_o:
        return False
    if direction == -1 and eng_c >= eng_o:
        return False

    # Check engulfing body size
    eng_body = abs(eng_c - eng_o)
    if eng_body < engulf_body_price:
        return False

    eng_body_lo = min(eng_o, eng_c)
    eng_body_hi = max(eng_o, eng_c)

    # Try k=1,2,3 (engulf last 1, 2, or 3 bars before current)
    for k in range(1, min(4, len(bars))):
        pre_bars = bars[-(k + 1):-1]  # k bars before engulfing candle
        pre_st = st_values[-(k + 1):-1]

        # Check all pre-engulfing bars have body >= c_body_pips
        all_body_ok = True
        for o, h, l, c in pre_bars:
            if abs(c - o) < c_body_price:
                all_body_ok = False
                break
        if not all_body_ok:
            continue

        # Check at least one bar is near ST (c1 identification)
        has_near_st = False
        for idx, (o, h, l, c) in enumerate(pre_bars):
            st = pre_st[idx]
            if direction == 1:
                # LONG: check proximity to TUp (support)
                if abs(l - st) <= c_wick_price or abs(c - st) <= c_close_price:
                    has_near_st = True
                    break
            else:
                # SHORT: check proximity to TDown (resistance)
                if abs(h - st) <= c_wick_price or abs(c - st) <= c_close_price:
                    has_near_st = True
                    break
        if not has_near_st:
            continue

        # Compute combined body range of pre-engulfing bars
        combined_lo = min(min(o, c) for o, h, l, c in pre_bars)
        combined_hi = max(max(o, c) for o, h, l, c in pre_bars)

        # Check engulfing body covers combined range
        if eng_body_lo <= combined_lo and eng_body_hi >= combined_hi:
            return True

    return False


def detect_wick_cross_short(
    o2: float, h2: float, l2: float, c2: float, tdown2: float,
    o1: float, h1: float, l1: float, c1: float, tdown1: float,
    oi: float, ci: float,
    c1_body_pips: float = 3.0,
    c1_wick_pips: float = 0.5,
    c1_close_pips: float = 1.0,
    c2_body_pips: float = 2.0,
    c2_wick_pips: float = 0.5,
    c2_close_pips: float = 1.0,
) -> bool:
    """Detect SHORT wick-cross pattern near TDown (resistance).

    Mirror of detect_wick_cross_long:
    - 1st candle (i-2): GREEN, pulls up toward TDown
    - 2nd candle (i-1): RED, bounces off TDown
    - 3rd candle (i): RED (close < open)

    Args:
        o2,h2,l2,c2,tdown2: 1st candle (bar i-2) OHLC + SuperTrend TDown
        o1,h1,l1,c1,tdown1: 2nd candle (bar i-1) OHLC + SuperTrend TDown
        oi,ci: 3rd candle (bar i) open + close
        c1/c2 body/wick/close pips: pattern thresholds

    Returns:
        True if the 3-candle SHORT pattern matches.
    """
    c1_body_price = pips_to_price(c1_body_pips)
    c1_wick_price = pips_to_price(c1_wick_pips)
    c1_close_price = pips_to_price(c1_close_pips)
    c2_body_price = pips_to_price(c2_body_pips)
    c2_wick_price = pips_to_price(c2_wick_pips)
    c2_close_price = pips_to_price(c2_close_pips)

    # --- 1st candle (bar i-2): GREEN, body >= threshold ---
    if c2 <= o2:  # not green
        return False
    body1 = c2 - o2
    if body1 < c1_body_price:
        return False

    # Touch or cross: (b) high within wick_pips of TDown OR close within close_pips
    #                 (c) high > TDown AND close < TDown (wick crossed, close didn't)
    touch_b = (h2 <= tdown2) and ((tdown2 - h2) <= c1_wick_price or abs(c2 - tdown2) <= c1_close_price)
    cross_c = (h2 > tdown2) and (c2 < tdown2)
    if not (touch_b or cross_c):
        return False

    # --- 2nd candle (bar i-1): RED, body >= threshold ---
    if c1 >= o1:  # not red
        return False
    body2 = o1 - c1
    if body2 < c2_body_price:
        return False

    # Touch: high within wick_pips of TDown OR close within close_pips
    touch_2 = (h1 <= tdown1) and ((tdown1 - h1) <= c2_wick_price or abs(c1 - tdown1) <= c2_close_price)
    if not touch_2:
        return False

    # No cross: high must stay <= TDown
    if h1 > tdown1:
        return False

    # --- 3rd candle (bar i): RED (close < open) ---
    if ci >= oi:
        return False

    return True
