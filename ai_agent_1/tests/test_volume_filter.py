"""Tests for volume filter indicator and strategy integration."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.volume_filter import compute_relative_volume
from src.strategy.ppst_circle import PPSTCircleStrategy


class TestComputeRelativeVolume:
    def test_basic_calculation(self):
        """Verify relative volume against manual SMA calculation."""
        volume = np.array([100, 200, 150, 120, 180], dtype=float)
        period = 3
        rel = compute_relative_volume(volume, period)

        # First 2 bars (indices 0,1) should be NaN
        assert np.isnan(rel[0])
        assert np.isnan(rel[1])

        # Index 2: SMA = (100+200+150)/3 = 150, rel = 150/150 = 1.0
        assert rel[2] == pytest.approx(1.0)

        # Index 3: SMA = (200+150+120)/3 = 156.67, rel = 120/156.67 = 0.766
        sma3 = (200 + 150 + 120) / 3
        assert rel[3] == pytest.approx(120 / sma3, rel=1e-6)

        # Index 4: SMA = (150+120+180)/3 = 150, rel = 180/150 = 1.2
        sma4 = (150 + 120 + 180) / 3
        assert rel[4] == pytest.approx(180 / sma4, rel=1e-6)

    def test_nan_before_period(self):
        """Bars before SMA lookback should be NaN."""
        volume = np.array([100, 200, 150, 120, 180, 160, 140], dtype=float)
        period = 5
        rel = compute_relative_volume(volume, period)

        for i in range(period - 1):
            assert np.isnan(rel[i])
        # Index 4 (period-1) should have a value
        assert not np.isnan(rel[period - 1])

    def test_zero_volume(self):
        """Zero average volume should produce NaN (no division by zero)."""
        volume = np.array([0, 0, 0, 0, 0], dtype=float)
        period = 3
        rel = compute_relative_volume(volume, period)
        # All NaN since SMA is 0
        assert np.all(np.isnan(rel))

    def test_short_array(self):
        """Array shorter than period returns all NaN."""
        volume = np.array([100, 200], dtype=float)
        rel = compute_relative_volume(volume, period=5)
        assert len(rel) == 2
        assert np.all(np.isnan(rel))

    def test_default_period(self):
        """Default period is 20."""
        volume = np.ones(25, dtype=float) * 100
        rel = compute_relative_volume(volume)
        # NaN for first 19 bars
        assert np.isnan(rel[18])
        # Value at index 19
        assert rel[19] == pytest.approx(1.0)


class TestVolumeThresholdIntegration:
    """Test that PPSTCircleStrategy respects volume_threshold."""

    def _make_entry_row(self, rel_volume=2.0):
        """Create a minimal row that would trigger a long entry in Nth Circle mode."""
        # tup at 1.0798 so open (1.0800) is only 2 pips away — within max_entry_dist_pips
        # support at 1.0790 so SL midpoint gives positive sl_distance from close
        return pd.Series({
            "open": 1.0800,
            "high": 1.0810,
            "low": 1.0790,
            "close": 1.0805,
            "volume": 100,
            "trend": 1,
            "tup": 1.0798,
            "tdown": 1.0830,
            "trailing_sl": 1.0798,
            "support": 1.0790,
            "resistance": 1.0835,
            "buy_signal": False,
            "sell_signal": False,
            "long_circle": True,
            "short_circle": False,
            "last_signal_bar": 0,
            "circle_count_since_signal": 1,
            "rel_volume": rel_volume,
        })

    def test_threshold_blocks_low_volume(self):
        """Entry should be blocked when rel_volume < threshold."""
        strategy = PPSTCircleStrategy(
            entry_mode="Nth Circle",
            entry_circle_num=1,
            volume_threshold=1.5,
            tp_ratio=2.0,
        )
        row = self._make_entry_row(rel_volume=1.0)  # Below threshold
        result = strategy.on_bar(idx=50, row=row, position=None, can_trade=True, entering_quiet=False)
        assert result["action"] == "none"

    def test_threshold_allows_high_volume(self):
        """Entry should proceed when rel_volume >= threshold."""
        strategy = PPSTCircleStrategy(
            entry_mode="Nth Circle",
            entry_circle_num=1,
            volume_threshold=1.5,
            tp_ratio=2.0,
        )
        row = self._make_entry_row(rel_volume=2.0)  # Above threshold
        result = strategy.on_bar(idx=50, row=row, position=None, can_trade=True, entering_quiet=False)
        assert result["action"] == "open_long"

    def test_threshold_disabled(self):
        """When threshold=0, all entries should proceed regardless of volume."""
        strategy = PPSTCircleStrategy(
            entry_mode="Nth Circle",
            entry_circle_num=1,
            volume_threshold=0.0,
            tp_ratio=2.0,
        )
        row = self._make_entry_row(rel_volume=0.5)  # Low volume, but filter disabled
        result = strategy.on_bar(idx=50, row=row, position=None, can_trade=True, entering_quiet=False)
        assert result["action"] == "open_long"

    def test_threshold_blocks_nan_volume(self):
        """Entry should be blocked when rel_volume is NaN (warmup period)."""
        strategy = PPSTCircleStrategy(
            entry_mode="Nth Circle",
            entry_circle_num=1,
            volume_threshold=1.5,
            tp_ratio=2.0,
        )
        row = self._make_entry_row(rel_volume=np.nan)
        result = strategy.on_bar(idx=50, row=row, position=None, can_trade=True, entering_quiet=False)
        assert result["action"] == "none"

    def test_threshold_exact_boundary(self):
        """Entry at exactly the threshold should proceed."""
        strategy = PPSTCircleStrategy(
            entry_mode="Nth Circle",
            entry_circle_num=1,
            volume_threshold=1.5,
            tp_ratio=2.0,
        )
        row = self._make_entry_row(rel_volume=1.5)  # Exactly at threshold
        result = strategy.on_bar(idx=50, row=row, position=None, can_trade=True, entering_quiet=False)
        assert result["action"] == "open_long"
