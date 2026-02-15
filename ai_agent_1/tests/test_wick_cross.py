"""Tests for WCSE pattern detection and state machine."""

import pytest

from src.strategy.wick_cross import (
    WickCrossState,
    detect_wick_cross_long,
    detect_wick_cross_short,
)


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
# detect_wick_cross_long tests
# ---------------------------------------------------------------------------

class TestDetectWickCrossLong:
    """LONG pattern: RED candle touches TUp, GREEN candle bounces, GREEN confirm."""

    # Helper: build a valid touch pattern (low near TUp, not crossing)
    # TUp = 1.08000
    # C1 (i-2): RED, o=1.08060, h=1.08070, l=1.08003, c=1.08020  (body=4 pips, low 0.3 pips above TUp)
    # C2 (i-1): GREEN, o=1.08010, h=1.08050, l=1.08004, c=1.08040 (body=3 pips, low 0.4 pips above TUp)
    # C3 (i): GREEN, o=1.08020, c=1.08050

    def _valid_touch_long(self):
        tup = 1.08000
        return dict(
            o2=1.08060, h2=1.08070, l2=1.08003, c2=1.08020, tup2=tup,
            o1=1.08010, h1=1.08050, l1=1.08004, c1=1.08040, tup1=tup,
            oi=1.08020, ci=1.08050,
            c1_body_pips=3.0, c1_wick_pips=0.5, c1_close_pips=1.0,
            c2_body_pips=2.0, c2_wick_pips=0.5, c2_close_pips=1.0,
        )

    def test_valid_touch_pattern(self):
        """Valid LONG touch pattern should return True."""
        assert detect_wick_cross_long(**self._valid_touch_long()) is True

    def test_valid_cross_pattern(self):
        """1st candle wick crosses below TUp but closes above → valid."""
        params = self._valid_touch_long()
        # low below TUp, close above TUp
        params["l2"] = 1.07990  # below TUp
        params["c2"] = 1.08020  # above TUp
        assert detect_wick_cross_long(**params) is True

    def test_reject_green_first_candle(self):
        """1st candle must be RED."""
        params = self._valid_touch_long()
        params["o2"] = 1.08010  # open < close → GREEN
        params["c2"] = 1.08060
        assert detect_wick_cross_long(**params) is False

    def test_reject_small_body_first(self):
        """1st candle body too small."""
        params = self._valid_touch_long()
        params["o2"] = 1.08030  # body = 1 pip (< 3 pip threshold)
        params["c2"] = 1.08020
        assert detect_wick_cross_long(**params) is False

    def test_reject_first_candle_too_far_from_tup(self):
        """1st candle low too far from TUp (no touch, no cross)."""
        params = self._valid_touch_long()
        params["l2"] = 1.08020  # 2 pips above TUp
        params["c2"] = 1.08020  # also 2 pips above TUp
        assert detect_wick_cross_long(**params) is False

    def test_reject_red_second_candle(self):
        """2nd candle must be GREEN."""
        params = self._valid_touch_long()
        params["o1"] = 1.08040  # open > close → RED
        params["c1"] = 1.08010
        assert detect_wick_cross_long(**params) is False

    def test_reject_small_body_second(self):
        """2nd candle body too small."""
        params = self._valid_touch_long()
        params["o1"] = 1.08030  # body = 0.5 pip (< 2 pip threshold)
        params["c1"] = 1.08035
        assert detect_wick_cross_long(**params) is False

    def test_reject_second_candle_crosses_tup(self):
        """2nd candle low below TUp → invalid (no cross allowed on C2)."""
        params = self._valid_touch_long()
        params["l1"] = 1.07990  # below TUp
        assert detect_wick_cross_long(**params) is False

    def test_reject_red_third_candle(self):
        """3rd candle must be GREEN."""
        params = self._valid_touch_long()
        params["oi"] = 1.08050
        params["ci"] = 1.08020  # close < open → RED
        assert detect_wick_cross_long(**params) is False

    def test_reject_flat_third_candle(self):
        """3rd candle close == open → not green."""
        params = self._valid_touch_long()
        params["oi"] = 1.08030
        params["ci"] = 1.08030
        assert detect_wick_cross_long(**params) is False

    def test_close_proximity_first_candle(self):
        """1st candle: low far but close within close_pips of TUp → valid."""
        params = self._valid_touch_long()
        params["l2"] = 1.08008  # 0.8 pips above TUp (> wick threshold)
        params["c2"] = 1.08005  # 0.5 pips above TUp (within close_pips=1.0)
        params["o2"] = 1.08050  # body = 4.5 pips
        assert detect_wick_cross_long(**params) is True


# ---------------------------------------------------------------------------
# detect_wick_cross_short tests
# ---------------------------------------------------------------------------

class TestDetectWickCrossShort:
    """SHORT pattern: GREEN candle touches TDown, RED candle bounces, RED confirm."""

    # TDown = 1.09000
    # C1 (i-2): GREEN, o=1.08940, h=1.08997, l=1.08930, c=1.08980 (body=4 pips, high 0.3 pips below TDown)
    # C2 (i-1): RED, o=1.08990, h=1.08996, l=1.08950, c=1.08960 (body=3 pips, high 0.4 pips below TDown)
    # C3 (i): RED, o=1.08980, c=1.08950

    def _valid_touch_short(self):
        tdown = 1.09000
        return dict(
            o2=1.08940, h2=1.08997, l2=1.08930, c2=1.08980, tdown2=tdown,
            o1=1.08990, h1=1.08996, l1=1.08950, c1=1.08960, tdown1=tdown,
            oi=1.08980, ci=1.08950,
            c1_body_pips=3.0, c1_wick_pips=0.5, c1_close_pips=1.0,
            c2_body_pips=2.0, c2_wick_pips=0.5, c2_close_pips=1.0,
        )

    def test_valid_touch_pattern(self):
        """Valid SHORT touch pattern should return True."""
        assert detect_wick_cross_short(**self._valid_touch_short()) is True

    def test_valid_cross_pattern(self):
        """1st candle wick crosses above TDown but closes below → valid."""
        params = self._valid_touch_short()
        params["h2"] = 1.09010  # above TDown
        params["c2"] = 1.08980  # below TDown
        assert detect_wick_cross_short(**params) is True

    def test_reject_red_first_candle(self):
        """1st candle must be GREEN for short pattern."""
        params = self._valid_touch_short()
        params["o2"] = 1.08990
        params["c2"] = 1.08940  # open > close → RED
        assert detect_wick_cross_short(**params) is False

    def test_reject_small_body_first(self):
        """1st candle body too small."""
        params = self._valid_touch_short()
        params["o2"] = 1.08970  # body = 1 pip
        params["c2"] = 1.08980
        assert detect_wick_cross_short(**params) is False

    def test_reject_green_second_candle(self):
        """2nd candle must be RED for short pattern."""
        params = self._valid_touch_short()
        params["o1"] = 1.08960
        params["c1"] = 1.08990  # close > open → GREEN
        assert detect_wick_cross_short(**params) is False

    def test_reject_second_candle_crosses_tdown(self):
        """2nd candle high above TDown → invalid."""
        params = self._valid_touch_short()
        params["h1"] = 1.09010  # above TDown
        assert detect_wick_cross_short(**params) is False

    def test_reject_green_third_candle(self):
        """3rd candle must be RED."""
        params = self._valid_touch_short()
        params["oi"] = 1.08950
        params["ci"] = 1.08980  # close > open → GREEN
        assert detect_wick_cross_short(**params) is False

    def test_first_candle_too_far_from_tdown(self):
        """1st candle high too far from TDown."""
        params = self._valid_touch_short()
        params["h2"] = 1.08980  # 2 pips below TDown
        params["c2"] = 1.08980  # also 2 pips below TDown
        assert detect_wick_cross_short(**params) is False

    def test_close_proximity_first_candle(self):
        """1st candle: high far but close within close_pips of TDown → valid."""
        params = self._valid_touch_short()
        params["h2"] = 1.08992  # 0.8 pips below TDown (> wick threshold)
        params["c2"] = 1.08995  # 0.5 pips below TDown (within close_pips=1.0)
        params["o2"] = 1.08950  # body = 4.5 pips
        assert detect_wick_cross_short(**params) is True
