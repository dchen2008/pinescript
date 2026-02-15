"""Test time filter: quiet window, market hours."""

import pandas as pd
import pytest

from src.strategy.time_filter import compute_time_filter


class TestQuietWindow:
    def test_1329_pt_is_tradable(self):
        """13:29 PT should be tradable (before quiet window)."""
        # 13:29 PT = 21:29 UTC (during PST, +8h)
        times = pd.Series([pd.Timestamp("2025-01-06 21:29:00", tz="UTC")])  # Monday
        result = compute_time_filter(times)
        assert result["can_trade"].iloc[0] == True

    def test_1330_pt_is_quiet(self):
        """13:30 PT should be quiet (start of quiet window)."""
        times = pd.Series([pd.Timestamp("2025-01-06 21:30:00", tz="UTC")])
        result = compute_time_filter(times)
        assert result["not_in_quiet"].iloc[0] == False
        assert result["can_trade"].iloc[0] == False

    def test_1630_pt_is_tradable(self):
        """16:30 PT should be tradable (end of quiet window, >= means exit)."""
        times = pd.Series([pd.Timestamp("2025-01-07 00:30:00", tz="UTC")])  # Tuesday 00:30 UTC = Monday 16:30 PT
        result = compute_time_filter(times)
        assert result["not_in_quiet"].iloc[0] == True

    def test_1500_pt_is_quiet(self):
        """15:00 PT should be in quiet window."""
        times = pd.Series([pd.Timestamp("2025-01-06 23:00:00", tz="UTC")])
        result = compute_time_filter(times)
        assert result["not_in_quiet"].iloc[0] == False

    def test_quiet_disabled(self):
        """With quiet window disabled, all times tradable."""
        times = pd.Series([pd.Timestamp("2025-01-06 21:30:00", tz="UTC")])
        result = compute_time_filter(times, use_quiet_window=False)
        assert result["not_in_quiet"].iloc[0] == True


class TestMarketHours:
    def test_sunday_1400_pt_open(self):
        """Sunday 14:00 PT should be market open."""
        # Sunday 14:00 PT = Sunday 22:00 UTC (PST +8)
        times = pd.Series([pd.Timestamp("2025-01-05 22:00:00", tz="UTC")])  # Sunday
        result = compute_time_filter(times)
        assert result["is_market_open"].iloc[0] == True

    def test_sunday_1300_pt_closed(self):
        """Sunday 13:00 PT should be market closed."""
        times = pd.Series([pd.Timestamp("2025-01-05 21:00:00", tz="UTC")])
        result = compute_time_filter(times)
        assert result["is_market_open"].iloc[0] == False

    def test_wednesday_midday_open(self):
        """Wednesday 12:00 PT should be market open."""
        times = pd.Series([pd.Timestamp("2025-01-08 20:00:00", tz="UTC")])  # Wed
        result = compute_time_filter(times)
        assert result["is_market_open"].iloc[0] == True

    def test_friday_1400_pt_closed(self):
        """Friday 14:00 PT should be market closed (>= close hour)."""
        # Friday 14:00 PT = Friday 22:00 UTC
        times = pd.Series([pd.Timestamp("2025-01-10 22:00:00", tz="UTC")])  # Friday
        result = compute_time_filter(times)
        assert result["is_market_open"].iloc[0] == False

    def test_friday_1359_pt_open(self):
        """Friday 13:59 PT should still be market open."""
        times = pd.Series([pd.Timestamp("2025-01-10 21:59:00", tz="UTC")])
        result = compute_time_filter(times)
        assert result["is_market_open"].iloc[0] == True

    def test_saturday_closed(self):
        """Saturday should be market closed."""
        times = pd.Series([pd.Timestamp("2025-01-11 12:00:00", tz="UTC")])  # Saturday
        result = compute_time_filter(times)
        assert result["is_market_open"].iloc[0] == False

    def test_market_disabled(self):
        """With market window disabled, all times open."""
        times = pd.Series([pd.Timestamp("2025-01-11 12:00:00", tz="UTC")])
        result = compute_time_filter(times, use_market_window=False)
        assert result["is_market_open"].iloc[0] == True


class TestEnteringQuiet:
    def test_transition_detected(self):
        """Should detect transition from not-quiet to quiet."""
        times = pd.Series([
            pd.Timestamp("2025-01-06 21:29:00", tz="UTC"),  # 13:29 PT - not quiet
            pd.Timestamp("2025-01-06 21:30:00", tz="UTC"),  # 13:30 PT - quiet starts
            pd.Timestamp("2025-01-06 21:31:00", tz="UTC"),  # 13:31 PT - still quiet
        ])
        result = compute_time_filter(times)
        assert result["entering_quiet"].iloc[0] == False
        assert result["entering_quiet"].iloc[1] == True
        assert result["entering_quiet"].iloc[2] == False  # Not a transition
