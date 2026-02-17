"""Tests for WCSE pattern detection and state machine."""

import pytest

from src.strategy.wick_cross import (
    WickCrossState,
    detect_engulf_2,
    detect_engulf_3,
    detect_wick_cross,
)
from src.utils.forex_utils import pips_to_price


# ---------------------------------------------------------------------------
# WickCrossState tests
# ---------------------------------------------------------------------------

class TestWickCrossState:
    def test_initial_state(self):
        wcs = WickCrossState()
        assert wcs.armed_dir == 0
        assert wcs.tp_exits_count == 0
        assert not wcs.is_active(position_is_none=True)

    def test_arm_long(self):
        wcs = WickCrossState()
        wcs.arm(1)
        assert wcs.armed_dir == 1
        assert wcs.is_active(position_is_none=True)
        assert not wcs.is_active(position_is_none=False)

    def test_arm_short(self):
        wcs = WickCrossState()
        wcs.arm(-1)
        assert wcs.armed_dir == -1
        assert wcs.is_active(position_is_none=True)

    def test_disarm(self):
        wcs = WickCrossState()
        wcs.arm(1)
        wcs.disarm()
        assert wcs.armed_dir == 0
        assert wcs.tp_exits_count == 0
        assert not wcs.is_active(position_is_none=True)

    def test_arm_direction_change_resets_tp_count(self):
        wcs = WickCrossState()
        wcs.arm(1)
        wcs.record_tp_hit()
        assert wcs.tp_exits_count == 1
        wcs.arm(-1)  # direction change
        assert wcs.tp_exits_count == 0

    def test_arm_same_direction_keeps_tp_count(self):
        wcs = WickCrossState()
        wcs.arm(1)
        wcs.record_tp_hit()
        wcs.arm(1)  # same direction
        assert wcs.tp_exits_count == 1

    def test_can_enter_unlimited(self):
        wcs = WickCrossState()
        wcs.arm(1)
        assert wcs.can_enter(entry_times=0)
        wcs.record_tp_hit()
        wcs.record_tp_hit()
        wcs.record_tp_hit()
        assert wcs.can_enter(entry_times=0)

    def test_can_enter_limited(self):
        wcs = WickCrossState()
        wcs.arm(1)
        assert wcs.can_enter(entry_times=2)
        wcs.record_tp_hit()
        assert wcs.can_enter(entry_times=2)
        wcs.record_tp_hit()
        assert not wcs.can_enter(entry_times=2)

    def test_can_enter_one(self):
        wcs = WickCrossState()
        wcs.arm(1)
        assert wcs.can_enter(entry_times=1)
        wcs.record_tp_hit()
        assert not wcs.can_enter(entry_times=1)


# ---------------------------------------------------------------------------
# detect_engulf_2 tests (Pattern A)
# ---------------------------------------------------------------------------

class TestDetectEngulf2:
    """Test the 2-candle engulfing pattern near SuperTrend."""

    # Helper: build candle tuple (open, high, low, close)
    # EUR/USD prices near 1.08000, pip = 0.0001

    def test_long_match(self):
        """LONG: c1 RED near TUp, c2 GREEN engulfs c1."""
        tup = 1.08000
        # c1: RED, close near TUp, body = 2 pips
        c1 = (1.08020, 1.08025, 1.07995, 1.08000)  # open > close, close = TUp
        # c2: GREEN, body > c1 body (3 pips > 2 pips)
        c2 = (1.08005, 1.08040, 1.08000, 1.08035)
        assert detect_engulf_2(c1, c2, tup, 1, c1_close_pips=0.8, c1_body_pips=0.1, c2_body_pips=0.6)

    def test_short_match(self):
        """SHORT: c1 GREEN near TDown, c2 RED engulfs c1."""
        tdown = 1.08100
        # c1: GREEN, close near TDown, body = 2 pips
        c1 = (1.08080, 1.08110, 1.08075, 1.08100)
        # c2: RED, body > c1 body (3 pips > 2 pips)
        c2 = (1.08095, 1.08100, 1.08060, 1.08065)
        assert detect_engulf_2(c1, c2, tdown, -1, c1_close_pips=0.8, c1_body_pips=0.1, c2_body_pips=0.6)

    def test_c1_too_far_from_st(self):
        """c1 close too far from TUp → no match."""
        tup = 1.08000
        # c1: RED, but close is 2 pips above TUp (outside 0.8 pip threshold)
        c1 = (1.08040, 1.08045, 1.08015, 1.08020)
        c2 = (1.08025, 1.08060, 1.08020, 1.08055)
        assert not detect_engulf_2(c1, c2, tup, 1, c1_close_pips=0.8)

    def test_c2_doesnt_engulf(self):
        """c2 body smaller than c1 → no engulfing."""
        tup = 1.08000
        c1 = (1.08030, 1.08035, 1.07995, 1.08000)  # body = 3 pips
        c2 = (1.08005, 1.08020, 1.08000, 1.08015)  # body = 1 pip (< 3 pips)
        assert not detect_engulf_2(c1, c2, tup, 1, c1_close_pips=0.8, c1_body_pips=0.1, c2_body_pips=0.6)

    def test_wrong_colors(self):
        """c1 GREEN (should be RED for long) → no match."""
        tup = 1.08000
        c1 = (1.07990, 1.08010, 1.07985, 1.08000)  # GREEN (close > open)
        c2 = (1.08005, 1.08040, 1.08000, 1.08035)
        assert not detect_engulf_2(c1, c2, tup, 1)

    def test_c1_body_too_small(self):
        """c1 body below minimum → no match."""
        tup = 1.08000
        # c1: RED, body = 0.05 pips (below 0.1 pip min)
        c1 = (1.08001, 1.08005, 1.07998, 1.08000)  # body = 0.1 pip
        # With c1_body_pips=0.2, the 0.1 pip body is too small
        c2 = (1.08005, 1.08040, 1.08000, 1.08035)
        assert not detect_engulf_2(c1, c2, tup, 1, c1_body_pips=0.2)


# ---------------------------------------------------------------------------
# detect_engulf_3 tests (Pattern B)
# ---------------------------------------------------------------------------

class TestDetectEngulf3:
    """Test the 3-candle engulfing pattern near SuperTrend."""

    def test_long_match(self):
        """LONG: c1 RED near TUp, c3 GREEN, net move > c1 body."""
        tup = 1.08000
        # c1 (i-2): RED, close near TUp, body = 2 pips
        c1 = (1.08020, 1.08025, 1.07995, 1.08000)
        # c2 (i-1): any candle, open = 1.08010
        c2 = (1.08010, 1.08015, 1.08005, 1.08008)
        # c3 (i): GREEN, body = 4 pips, net_move = c3.close - c2.open = 1.08045 - 1.08010 = 3.5 pips > 2 pips
        c3 = (1.08005, 1.08050, 1.08000, 1.08045)
        assert detect_engulf_3(c1, c2, c3, tup, 1, c1_close_pips=0.8, c1_body_pips=0.1, c2_body_pips=0.6)

    def test_short_match(self):
        """SHORT: c1 GREEN near TDown, c3 RED, net move > c1 body."""
        tdown = 1.08100
        # c1 (i-2): GREEN, close near TDown, body = 2 pips
        c1 = (1.08080, 1.08105, 1.08075, 1.08100)
        # c2 (i-1): any candle, open = 1.08090
        c2 = (1.08090, 1.08095, 1.08085, 1.08088)
        # c3 (i): RED, body = 4 pips, net_move = c2.open - c3.close = 1.08090 - 1.08055 = 3.5 pips > 2 pips
        c3 = (1.08095, 1.08100, 1.08050, 1.08055)
        assert detect_engulf_3(c1, c2, c3, tdown, -1, c1_close_pips=0.8, c1_body_pips=0.1, c2_body_pips=0.6)

    def test_net_move_insufficient(self):
        """Net move doesn't exceed c1 body → no match."""
        tup = 1.08000
        c1 = (1.08030, 1.08035, 1.07995, 1.08000)  # body = 3 pips
        c2 = (1.08010, 1.08015, 1.08005, 1.08008)   # open = 1.08010
        # net_move = 1.08035 - 1.08010 = 2.5 pips < 3 pips (c1 body)
        c3 = (1.08005, 1.08040, 1.08000, 1.08035)
        assert not detect_engulf_3(c1, c2, c3, tup, 1, c1_close_pips=0.8, c1_body_pips=0.1, c2_body_pips=0.6)

    def test_c3_wrong_color(self):
        """c3 RED for LONG pattern → no match."""
        tup = 1.08000
        c1 = (1.08020, 1.08025, 1.07995, 1.08000)
        c2 = (1.08010, 1.08015, 1.08005, 1.08008)
        c3 = (1.08040, 1.08045, 1.08000, 1.08005)  # RED (close < open)
        assert not detect_engulf_3(c1, c2, c3, tup, 1)


# ---------------------------------------------------------------------------
# detect_wick_cross tests (Pattern C)
# ---------------------------------------------------------------------------

class TestDetectWickCross:
    """Test the wick cross SuperTrend pattern."""

    def test_long_match(self):
        """LONG: GREEN candle, wick below TUp, close above TUp."""
        tup = 1.08000
        # GREEN candle, low pierces below TUp by 1 pip, close above TUp
        candle = (1.08005, 1.08020, 1.07990, 1.08015)
        assert detect_wick_cross(candle, tup, 1, wx_st_pips=0.4)

    def test_short_match(self):
        """SHORT: RED candle, wick above TDown, close below TDown."""
        tdown = 1.08100
        # RED candle, high pierces above TDown by 1 pip, close below TDown
        candle = (1.08095, 1.08110, 1.08080, 1.08085)
        assert detect_wick_cross(candle, tdown, -1, wx_st_pips=0.4)

    def test_wrong_color_long(self):
        """RED candle for LONG → no match."""
        tup = 1.08000
        candle = (1.08015, 1.08020, 1.07990, 1.08005)  # RED (close < open)
        assert not detect_wick_cross(candle, tup, 1, wx_st_pips=0.4)

    def test_insufficient_depth(self):
        """Wick doesn't pierce deep enough → no match."""
        tup = 1.08000
        # low = 1.07998, depth = 0.2 pips (< 0.4 threshold)
        candle = (1.08005, 1.08020, 1.07998, 1.08015)
        assert not detect_wick_cross(candle, tup, 1, wx_st_pips=0.4)

    def test_close_wrong_side(self):
        """Close below TUp for LONG → no match."""
        tup = 1.08000
        candle = (1.07990, 1.08010, 1.07980, 1.07995)  # GREEN but close < TUp
        assert not detect_wick_cross(candle, tup, 1, wx_st_pips=0.4)

    def test_wick_doesnt_cross(self):
        """Wick stays above TUp for LONG → no match."""
        tup = 1.08000
        candle = (1.08005, 1.08020, 1.08002, 1.08015)  # low > TUp
        assert not detect_wick_cross(candle, tup, 1, wx_st_pips=0.4)
