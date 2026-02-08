"""Tests for the autonomous performance monitor."""

import json
import os
import pytest
from datetime import datetime, timezone, timedelta

from src.live.performance_monitor import PerformanceMonitor, get_forex_week_start
from src.utils.timezone_utils import to_pt


@pytest.fixture
def monitor(tmp_path):
    """Create a PerformanceMonitor with a temp logs dir."""
    return PerformanceMonitor(
        baseline_risk=1.5,
        min_risk=0.5,
        dd_caution_threshold=0.10,
        dd_danger_threshold=0.20,
        gain_protect_threshold=8.0,
        gain_protect_risk=0.75,
        logs_dir=str(tmp_path),
    )


class TestForexWeekStart:
    def test_monday(self):
        """Monday belongs to the previous Sunday's week."""
        # Monday 10am PT -> previous Sunday 2pm PT
        dt = datetime(2025, 3, 3, 10, 0)  # Monday
        result = get_forex_week_start(dt)
        assert result.weekday() == 6  # Sunday
        assert result.hour == 14
        assert result.day == 2  # March 2 (previous Sunday)

    def test_sunday_after_2pm(self):
        """Sunday 3pm starts a new forex week."""
        dt = datetime(2025, 3, 2, 15, 0)  # Sunday 3pm
        result = get_forex_week_start(dt)
        assert result.weekday() == 6
        assert result.hour == 14
        assert result.day == 2  # Same Sunday

    def test_sunday_before_2pm(self):
        """Sunday 1pm belongs to previous week."""
        dt = datetime(2025, 3, 2, 13, 0)  # Sunday 1pm
        result = get_forex_week_start(dt)
        assert result.weekday() == 6
        assert result.hour == 14
        assert result.day == 23  # Feb 23 (previous Sunday)

    def test_friday(self):
        """Friday maps back to previous Sunday."""
        dt = datetime(2025, 3, 7, 10, 0)  # Friday
        result = get_forex_week_start(dt)
        assert result.weekday() == 6
        assert result.hour == 14
        assert result.day == 2  # March 2 Sunday


class TestRiskRules:
    def test_baseline(self, monitor):
        """Default risk is baseline when no rules triggered."""
        risk = monitor.compute_effective_risk(drawdown=0.0, week_return_pct=0.0)
        assert risk == 1.5

    def test_dd_caution(self, monitor):
        """10% drawdown -> 0.75%."""
        risk = monitor.compute_effective_risk(drawdown=0.10, week_return_pct=0.0)
        assert risk == 0.75

    def test_dd_danger(self, monitor):
        """20% drawdown -> 0.5% (min risk)."""
        risk = monitor.compute_effective_risk(drawdown=0.20, week_return_pct=0.0)
        assert risk == 0.5

    def test_dd_danger_extreme(self, monitor):
        """50% drawdown still floors at min_risk."""
        risk = monitor.compute_effective_risk(drawdown=0.50, week_return_pct=0.0)
        assert risk == 0.5

    def test_gain_protection(self, monitor):
        """8% week return -> 0.75% to protect gains."""
        risk = monitor.compute_effective_risk(drawdown=0.0, week_return_pct=8.0)
        assert risk == 0.75

    def test_gain_protection_above_threshold(self, monitor):
        """12% week return also triggers gain protection."""
        risk = monitor.compute_effective_risk(drawdown=0.0, week_return_pct=12.0)
        assert risk == 0.75

    def test_1_week_losing(self, monitor):
        """Last week negative -> 1.125% (baseline * 0.75)."""
        monitor.prev_week_returns = [-2.0]
        risk = monitor.compute_effective_risk(drawdown=0.0, week_return_pct=0.0)
        assert risk == pytest.approx(1.125)

    def test_2_week_losing_streak(self, monitor):
        """Last 2 weeks negative -> 0.75%."""
        monitor.prev_week_returns = [-1.0, -3.0]
        risk = monitor.compute_effective_risk(drawdown=0.0, week_return_pct=0.0)
        assert risk == 0.75

    def test_2_week_losing_only_last_negative(self, monitor):
        """Only last week negative (not 2-week streak) -> 1.125%."""
        monitor.prev_week_returns = [5.0, -1.0]
        risk = monitor.compute_effective_risk(drawdown=0.0, week_return_pct=0.0)
        assert risk == pytest.approx(1.125)

    def test_no_weeks_history(self, monitor):
        """No week history -> baseline."""
        monitor.prev_week_returns = []
        risk = monitor.compute_effective_risk(drawdown=0.0, week_return_pct=0.0)
        assert risk == 1.5


class TestRulePriority:
    def test_dd_danger_beats_all(self, monitor):
        """DD danger (0.5%) wins over all other rules."""
        monitor.prev_week_returns = [-1.0, -3.0]  # 2-week losing = 0.75
        risk = monitor.compute_effective_risk(drawdown=0.25, week_return_pct=10.0)
        assert risk == 0.5  # DD danger

    def test_dd_caution_beats_1_week_losing(self, monitor):
        """DD caution (0.75%) beats 1-week losing (1.125%)."""
        monitor.prev_week_returns = [-2.0]
        risk = monitor.compute_effective_risk(drawdown=0.12, week_return_pct=0.0)
        assert risk == 0.75

    def test_gain_protect_and_dd_caution_same_level(self, monitor):
        """Both at 0.75%, minimum still 0.75%."""
        risk = monitor.compute_effective_risk(drawdown=0.10, week_return_pct=9.0)
        assert risk == 0.75

    def test_multiple_rules_most_conservative_wins(self, monitor):
        """When multiple rules fire, the lowest risk wins."""
        monitor.prev_week_returns = [-1.0, -2.0]  # 2-week losing = 0.75
        # DD caution = 0.75, gain protect = 0.75, 1-week losing = 1.125
        risk = monitor.compute_effective_risk(drawdown=0.15, week_return_pct=8.0)
        assert risk == 0.75


class TestOnTradeClose:
    def _utc(self, year, month, day, hour=12, minute=0):
        """Helper to create UTC datetimes."""
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)

    def test_first_trade(self, monitor):
        """First trade initializes week tracking."""
        risk = monitor.on_trade_close(
            timestamp_utc=self._utc(2025, 3, 3, 15),  # Monday
            pnl=50.0,
            balance=10050.0,
            peak_balance=10050.0,
            drawdown=0.0,
        )
        assert risk == 1.5  # baseline
        assert monitor.week_pnl == 50.0
        assert monitor.week_start_balance == 10000.0

    def test_multiple_trades_same_week(self, monitor):
        """Multiple trades in the same week accumulate PnL."""
        ts = self._utc(2025, 3, 3, 15)
        monitor.on_trade_close(ts, pnl=50.0, balance=10050.0,
                               peak_balance=10050.0, drawdown=0.0)
        monitor.on_trade_close(ts, pnl=-20.0, balance=10030.0,
                               peak_balance=10050.0, drawdown=0.0)
        assert monitor.week_pnl == pytest.approx(30.0)

    def test_week_transition(self, monitor):
        """Week transition records previous week and starts new one."""
        # Week 1 trade
        week1 = self._utc(2025, 3, 3, 15)  # Monday March 3
        monitor.on_trade_close(week1, pnl=100.0, balance=10100.0,
                               peak_balance=10100.0, drawdown=0.0)

        # Week 2 trade (next Monday)
        week2 = self._utc(2025, 3, 10, 15)  # Monday March 10
        monitor.on_trade_close(week2, pnl=50.0, balance=10150.0,
                               peak_balance=10150.0, drawdown=0.0)

        assert len(monitor.prev_week_returns) == 1
        # Week 1 return: 100/10000 = 1.0%
        assert monitor.prev_week_returns[0] == pytest.approx(1.0)

    def test_dd_triggers_risk_reduction(self, monitor):
        """Drawdown during trade triggers risk reduction."""
        risk = monitor.on_trade_close(
            timestamp_utc=self._utc(2025, 3, 3, 15),
            pnl=-500.0,
            balance=9500.0,
            peak_balance=10000.0,
            drawdown=0.05,  # 5% — no rule yet
        )
        assert risk == 1.5

        risk = monitor.on_trade_close(
            timestamp_utc=self._utc(2025, 3, 3, 16),
            pnl=-600.0,
            balance=8900.0,
            peak_balance=10000.0,
            drawdown=0.11,  # 11% — DD caution
        )
        assert risk == 0.75

    def test_gain_protection_mid_week(self, monitor):
        """Big winning week triggers gain protection."""
        # Start with $10000, week return = 9% after this trade
        risk = monitor.on_trade_close(
            timestamp_utc=self._utc(2025, 3, 3, 15),
            pnl=900.0,
            balance=10900.0,
            peak_balance=10900.0,
            drawdown=0.0,
        )
        # Week return: 900/10000 = 9% >= 8% threshold
        assert risk == 0.75


class TestStatePersistence:
    def test_save_and_load(self, monitor, tmp_path):
        """State persists across save/load cycle."""
        monitor.current_risk_percent = 0.75
        monitor.week_start_balance = 10000.0
        monitor.week_pnl = 200.0
        monitor.current_week_key = "2025-03-02T14:00:00"
        monitor.prev_week_returns = [1.5, -2.0, 3.0]

        monitor.save_state()

        # Create new monitor and load
        monitor2 = PerformanceMonitor(logs_dir=str(tmp_path))
        assert monitor2.load_state() is True

        assert monitor2.current_risk_percent == 0.75
        assert monitor2.week_start_balance == 10000.0
        assert monitor2.week_pnl == 200.0
        assert monitor2.current_week_key == "2025-03-02T14:00:00"
        assert monitor2.prev_week_returns == [1.5, -2.0, 3.0]

    def test_load_missing_file(self, monitor):
        """Loading with no file returns False and keeps defaults."""
        assert monitor.load_state() is False
        assert monitor.current_risk_percent == 1.5

    def test_load_corrupted_file(self, tmp_path):
        """Loading corrupted JSON returns False."""
        state_path = tmp_path / "monitor_state.json"
        state_path.write_text("not valid json{{{")

        mon = PerformanceMonitor(logs_dir=str(tmp_path))
        assert mon.load_state() is False
        assert mon.current_risk_percent == 1.5


class TestEdgeCases:
    def _utc(self, year, month, day, hour=12, minute=0):
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)

    def test_prev_weeks_capped_at_4(self, monitor):
        """Only the last 4 completed weeks are retained."""
        monitor.prev_week_returns = [1.0, 2.0, 3.0, 4.0]
        monitor.current_week_key = "2025-03-02T14:00:00"
        monitor.week_start_balance = 10000.0
        monitor.week_pnl = 50.0

        # Close week by transitioning to a new one
        monitor._close_week(10500.0)
        assert len(monitor.prev_week_returns) == 4
        # Oldest (1.0) dropped, new one (0.5%) added
        assert monitor.prev_week_returns[0] == 2.0

    def test_zero_balance_no_crash(self, monitor):
        """Edge case: zero balance doesn't crash."""
        risk = monitor.on_trade_close(
            timestamp_utc=self._utc(2025, 3, 3, 15),
            pnl=-10000.0,
            balance=0.0,
            peak_balance=10000.0,
            drawdown=1.0,
        )
        # Should hit DD danger
        assert risk == 0.5

    def test_drawdown_recovery_restores_baseline(self, monitor):
        """When drawdown recovers below threshold, risk returns to baseline."""
        # First: in drawdown (10% DD caution)
        risk = monitor.on_trade_close(
            timestamp_utc=self._utc(2025, 3, 3, 15),
            pnl=-1100.0,
            balance=8900.0,
            peak_balance=10000.0,
            drawdown=0.11,
        )
        assert risk == 0.75  # DD caution

        # Then: new week with fresh balance, no drawdown, small trade
        # Week PnL resets since it's a new week
        risk = monitor.on_trade_close(
            timestamp_utc=self._utc(2025, 3, 10, 15),  # Next Monday = new week
            pnl=50.0,
            balance=8950.0,
            peak_balance=8950.0,
            drawdown=0.0,
        )
        # Week return: 50/8900 = 0.56% (< 8% threshold)
        # No DD, no gain protect, but prev_week was negative -> 1-week losing
        assert risk == pytest.approx(1.125)  # 1-week losing rule

    def test_audit_csv_created(self, monitor, tmp_path):
        """Audit CSV is created with header on initialization."""
        csv_path = tmp_path / "risk_adjustments.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "timestamp" in content
        assert "rule" in content

    def test_risk_never_exceeds_baseline(self, monitor):
        """Risk is always <= baseline, regardless of conditions."""
        # Even with great conditions, risk stays at baseline
        monitor.prev_week_returns = [5.0, 10.0, 8.0, 12.0]
        risk = monitor.compute_effective_risk(drawdown=0.0, week_return_pct=3.0)
        assert risk <= monitor.baseline_risk
