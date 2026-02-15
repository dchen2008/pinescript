"""Tests for the HMTS (Huge Movement Trade Strategy)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.strategy.hmts import HMTSStrategy
from src.strategy.position import Position


def make_row(close, high=None, low=None, tup=np.nan, tdown=np.nan,
             buy_signal=False, sell_signal=False):
    """Create a mock bar row."""
    if high is None:
        high = close + 0.0002
    if low is None:
        low = close - 0.0002
    return pd.Series({
        "close": close,
        "high": high,
        "low": low,
        "open": close,
        "tup": tup,
        "tdown": tdown,
        "trailing_sl": tup if not math.isnan(tup) else tdown,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
    })


def make_strategy(**kwargs):
    """Create HMTS strategy with test-friendly defaults."""
    defaults = {
        "movement_candles": 10,
        "movement_pips": 60.0,
        "revert_candles": 6,
        "revert_min_pips": 10.0,
        "revert_max_pips": 30.0,
        "hmts_tp_rr": 7.0,
        "allow_second_entry": True,
        "base_rr_ratio": 1.5,
        "trail_with_supertrend": True,
        "s1_sl_spread_buffer": False,  # Disable spread buffer for simpler test math
        "s2_revert_cross_limit": 8.0,
        "spread_pips": 1.5,
    }
    defaults.update(kwargs)
    return HMTSStrategy(**defaults)


def make_position(direction, entry_price, sl_price, tp_price=None, entry_bar=0):
    """Create a Position for testing."""
    return Position(
        direction=direction,
        entry_price=entry_price,
        units=1000,
        sl_price=sl_price,
        tp_price=tp_price,
        entry_bar=entry_bar,
    )


class TestPatternDetection:
    """Test 3-signal HMTS pattern detection."""

    def test_pattern_detection_short_to_hmts(self):
        """S1 sell -> S2 buy (60+ pips) -> S3 sell (10-30 pip revert) -> HMTS active."""
        strat = make_strategy()

        # S1: sell signal at 1.0800, tdown at 1.0810
        s1_row = make_row(1.0800, tdown=1.0810, sell_signal=True)
        result = strat.on_bar(0, s1_row, None, True, False)
        assert result["action"] == "open_short"

        pos = make_position(-1, 1.0800, 1.0810, entry_bar=0)

        # Some bars pass...
        for i in range(1, 4):
            row = make_row(1.0800 - i * 0.0020, tdown=1.0810 - i * 0.0010)
            strat.on_bar(i, row, pos, True, False)

        # S2: buy signal at 1.0730 (70 pips from S1 close of 1.0800), 5 bars later
        s2_row = make_row(1.0730, tup=1.0720, buy_signal=True)
        result = strat.on_bar(5, s2_row, pos, True, False)
        # Should hold position (not reverse)
        assert result["action"] == "none"
        assert strat.state == strat.WATCHING_REVERT

        # S3: sell signal at 1.0745 (15 pip revert from S2), 2 bars later
        s3_row = make_row(1.0745, tdown=1.0755, sell_signal=True)
        result = strat.on_bar(7, s3_row, pos, True, False)
        assert result["action"] == "none"
        assert strat.state == strat.HMTS_ACTIVE

    def test_pattern_detection_long_to_hmts(self):
        """S1 buy -> S2 sell (60+ pips) -> S3 buy (revert) -> HMTS active."""
        strat = make_strategy()

        # S1: buy signal at 1.0800, tup at 1.0790
        s1_row = make_row(1.0800, tup=1.0790, buy_signal=True)
        result = strat.on_bar(0, s1_row, None, True, False)
        assert result["action"] == "open_long"

        pos = make_position(1, 1.0800, 1.0790, entry_bar=0)

        # S2: sell signal at 1.0870 (70 pips up), 5 bars later
        s2_row = make_row(1.0870, tdown=1.0880, sell_signal=True)
        result = strat.on_bar(5, s2_row, pos, True, False)
        assert result["action"] == "none"
        assert strat.state == strat.WATCHING_REVERT

        # S3: buy signal at 1.0855 (15 pip revert from S2)
        s3_row = make_row(1.0855, tup=1.0845, buy_signal=True)
        result = strat.on_bar(7, s3_row, pos, True, False)
        assert result["action"] == "none"
        assert strat.state == strat.HMTS_ACTIVE


class TestPatternRejection:
    """Test cases where HMTS pattern is not met."""

    def test_movement_too_small(self):
        """Movement < 60 pips -> normal reversal."""
        strat = make_strategy()

        # S1: sell at 1.0800
        s1_row = make_row(1.0800, tdown=1.0810, sell_signal=True)
        strat.on_bar(0, s1_row, None, True, False)

        pos = make_position(-1, 1.0800, 1.0810, entry_bar=0)

        # S2: buy at 1.0760 (only 40 pips — below 60 threshold)
        s2_row = make_row(1.0760, tup=1.0750, buy_signal=True)
        result = strat.on_bar(5, s2_row, pos, True, False)
        # Should reverse normally
        assert result["action"] == "reverse_to_long"
        assert strat.state == strat.IDLE

    def test_revert_too_large(self):
        """Revert >= max_pips -> close position."""
        strat = make_strategy(revert_max_pips=30.0)

        # S1: sell at 1.0800
        s1_row = make_row(1.0800, tdown=1.0810, sell_signal=True)
        strat.on_bar(0, s1_row, None, True, False)
        pos = make_position(-1, 1.0800, 1.0810, entry_bar=0)

        # S2: buy at 1.0730 (70 pips)
        s2_row = make_row(1.0730, tup=1.0720, buy_signal=True)
        strat.on_bar(5, s2_row, pos, True, False)
        assert strat.state == strat.WATCHING_REVERT

        # S3: sell at 1.0765 (35 pip revert — exceeds max_pips of 30)
        s3_row = make_row(1.0765, tdown=1.0775, sell_signal=True)
        result = strat.on_bar(7, s3_row, pos, True, False)
        assert result["action"] == "close"
        assert result["close_reason"] == "HMTS Revert Failed"
        assert strat.state == strat.IDLE

    def test_too_many_candles_for_revert(self):
        """Revert takes too many candles -> close position."""
        strat = make_strategy(revert_candles=3)

        # S1: sell at 1.0800
        s1_row = make_row(1.0800, tdown=1.0810, sell_signal=True)
        strat.on_bar(0, s1_row, None, True, False)
        pos = make_position(-1, 1.0800, 1.0810, entry_bar=0)

        # S2: buy at 1.0730 (70 pips, 5 bars)
        s2_row = make_row(1.0730, tup=1.0720, buy_signal=True)
        strat.on_bar(5, s2_row, pos, True, False)
        assert strat.state == strat.WATCHING_REVERT

        # No S3 for 4 bars (exceeds revert_candles=3)
        for i in range(6, 10):
            row = make_row(1.0735, tdown=1.0745)
            result = strat.on_bar(i, row, pos, True, False)

        # Should have closed on timeout
        assert result["action"] == "close"
        assert result["close_reason"] == "HMTS Revert Timeout"
        assert strat.state == strat.IDLE

    def test_too_many_candles_for_movement(self):
        """Movement takes too many candles -> normal reversal."""
        strat = make_strategy(movement_candles=5)

        # S1: sell at 1.0800
        s1_row = make_row(1.0800, tdown=1.0810, sell_signal=True)
        strat.on_bar(0, s1_row, None, True, False)
        pos = make_position(-1, 1.0800, 1.0810, entry_bar=0)

        # S2: buy at 1.0730 (70 pips, but 8 bars > movement_candles=5)
        s2_row = make_row(1.0730, tup=1.0720, buy_signal=True)
        result = strat.on_bar(8, s2_row, pos, True, False)
        assert result["action"] == "reverse_to_long"
        assert strat.state == strat.IDLE


class TestHoldBehavior:
    """Test that position is held during HMTS pattern."""

    def test_s2_holds_position(self):
        """At S2, position is held (not reversed)."""
        strat = make_strategy()

        # S1: buy at 1.0800
        s1_row = make_row(1.0800, tup=1.0790, buy_signal=True)
        strat.on_bar(0, s1_row, None, True, False)
        pos = make_position(1, 1.0800, 1.0790, entry_bar=0)

        # S2: sell at 1.0870 (70 pips)
        s2_row = make_row(1.0870, tdown=1.0880, sell_signal=True)
        result = strat.on_bar(5, s2_row, pos, True, False)
        assert result["action"] == "none"
        # Position still exists (not closed or reversed)
        assert strat.state == strat.WATCHING_REVERT
        assert strat.hmts_direction == 1  # Long position held


class TestTPUpdate:
    """Test TP is set correctly after S3 confirmation."""

    def test_tp_set_at_rr_7(self):
        """After S3 confirmation, TP is set at entry + sl_distance * 7."""
        strat = make_strategy(hmts_tp_rr=7.0)

        # S1: sell at 1.0800
        s1_row = make_row(1.0800, tdown=1.0810, sell_signal=True)
        strat.on_bar(0, s1_row, None, True, False)
        pos = make_position(-1, 1.0800, 1.0810, entry_bar=0)

        # S2: buy at 1.0730 (70 pips)
        s2_row = make_row(1.0730, tup=1.0720, buy_signal=True)
        strat.on_bar(5, s2_row, pos, True, False)

        # S3: sell at 1.0745 (15 pip revert)
        s3_row = make_row(1.0745, tdown=1.0755, sell_signal=True)
        strat.on_bar(7, s3_row, pos, True, False)

        assert strat.state == strat.HMTS_ACTIVE
        # TP = entry - sl_distance * 7.0
        sl_distance = abs(pos.entry_price - 1.0810)  # = 0.0010
        expected_tp = pos.entry_price - sl_distance * 7.0
        assert pos.tp_price == pytest.approx(expected_tp)


class TestSLUpdate:
    """Test SL is updated after S3 confirmation."""

    def test_sl_updated_to_s3_supertrend(self):
        """After S3, SL should be S3's SuperTrend."""
        strat = make_strategy()

        # S1: sell at 1.0800
        s1_row = make_row(1.0800, tdown=1.0810, sell_signal=True)
        strat.on_bar(0, s1_row, None, True, False)
        pos = make_position(-1, 1.0800, 1.0810, entry_bar=0)

        # S2: buy at 1.0730 (70 pips)
        s2_row = make_row(1.0730, tup=1.0720, buy_signal=True)
        strat.on_bar(5, s2_row, pos, True, False)

        # S3: sell at 1.0745, tdown=1.0760
        s3_row = make_row(1.0745, tdown=1.0760, sell_signal=True)
        strat.on_bar(7, s3_row, pos, True, False)

        assert strat.state == strat.HMTS_ACTIVE
        assert pos.sl_price == pytest.approx(1.0760)


class TestSecondEntry:
    """Test re-entry after TP hit."""

    def test_second_entry_after_tp(self):
        """After TP hit with allow_second_entry=true, re-enters same direction."""
        strat = make_strategy(allow_second_entry=True, hmts_tp_rr=7.0)

        # Set up HMTS_ACTIVE state manually
        strat.state = strat.HMTS_ACTIVE
        strat.hmts_direction = -1
        strat.was_in_position = True
        strat.last_known_direction = -1
        strat.last_known_tp = 1.0700
        strat.last_known_sl = 1.0760

        # Position disappeared (engine closed it on TP hit)
        # Bar where TP was hit (low reached 1.0700)
        row = make_row(1.0710, high=1.0720, low=1.0695, tdown=1.0730)
        result = strat.on_bar(20, row, None, True, False)

        assert result["action"] == "open_short"
        assert result["sl"] is not None

    def test_no_reentry_after_sl(self):
        """After SL hit, returns to IDLE (no re-entry)."""
        strat = make_strategy()

        # Set up HMTS_ACTIVE state manually
        strat.state = strat.HMTS_ACTIVE
        strat.hmts_direction = -1
        strat.was_in_position = True
        strat.last_known_direction = -1
        strat.last_known_tp = 1.0700
        strat.last_known_sl = 1.0760

        # Bar where SL was hit but TP was NOT (high reached SL, low didn't reach TP)
        row = make_row(1.0755, high=1.0765, low=1.0740, tdown=1.0770)
        result = strat.on_bar(20, row, None, True, False)

        assert result["action"] == "none"
        assert strat.state == strat.IDLE


class TestCounterTrendSuppression:
    """Test that counter-trend signals are ignored during HMTS."""

    def test_ignore_counter_signal_watching_revert(self):
        """During WATCHING_REVERT, opposite signals are ignored."""
        strat = make_strategy()

        # S1: sell at 1.0800
        s1_row = make_row(1.0800, tdown=1.0810, sell_signal=True)
        strat.on_bar(0, s1_row, None, True, False)
        pos = make_position(-1, 1.0800, 1.0810, entry_bar=0)

        # S2: buy at 1.0730 (70 pips)
        s2_row = make_row(1.0730, tup=1.0720, buy_signal=True)
        strat.on_bar(5, s2_row, pos, True, False)
        assert strat.state == strat.WATCHING_REVERT

        # Counter-trend: another buy signal while we're watching for sell S3
        counter_row = make_row(1.0735, tup=1.0725, buy_signal=True)
        result = strat.on_bar(6, counter_row, pos, True, False)
        assert result["action"] == "none"

    def test_ignore_counter_signal_hmts_active(self):
        """During HMTS_ACTIVE, all signals are ignored."""
        strat = make_strategy()

        # Manually put into HMTS_ACTIVE
        strat.state = strat.HMTS_ACTIVE
        strat.hmts_direction = -1
        pos = make_position(-1, 1.0800, 1.0760, tp_price=1.0730)

        # Buy signal fires (counter-trend for short)
        row = make_row(1.0770, tup=1.0760, buy_signal=True)
        result = strat.on_bar(15, row, pos, True, False)
        assert result["action"] == "none"

        # Sell signal fires (same direction but still ignored in HMTS_ACTIVE)
        row2 = make_row(1.0750, tdown=1.0760, sell_signal=True)
        result2 = strat.on_bar(16, row2, pos, True, False)
        assert result2["action"] == "none"


class TestQuietWindowClose:
    """Test quiet window behavior."""

    def test_quiet_window_resets_state(self):
        """Entering quiet window closes position and resets HMTS state."""
        strat = make_strategy()

        # Put into WATCHING_REVERT
        strat.state = strat.WATCHING_REVERT
        strat.hmts_direction = -1
        pos = make_position(-1, 1.0800, 1.0810)

        row = make_row(1.0750, tdown=1.0760)
        result = strat.on_bar(10, row, pos, True, entering_quiet=True)

        assert result["action"] == "close"
        assert result["close_reason"] == "Quiet Close"
        assert strat.state == strat.IDLE
