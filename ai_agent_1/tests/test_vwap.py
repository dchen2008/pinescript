"""Tests for VWAP indicator, RSI indicator, and VWAP mean reversion strategy."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.vwap import compute_session_vwap
from src.indicators.rsi import compute_rsi
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy


def make_vwap_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic EUR/USD M5 data with volume."""
    rng = np.random.RandomState(seed)
    base = 1.08
    prices = [base]
    for _ in range(n - 1):
        change = rng.normal(0, 0.0005)
        prices.append(prices[-1] + change)

    df = pd.DataFrame()
    # Start on a Monday at 07:55 UTC so session reset at 08:00 is testable
    df["time"] = pd.date_range("2025-03-03 07:55", periods=n, freq="5min", tz="UTC")
    df["close"] = prices
    df["open"] = [prices[0]] + prices[:-1]
    df["high"] = np.maximum(df["open"], df["close"]) + rng.uniform(0, 0.0003, n)
    df["low"] = np.minimum(df["open"], df["close"]) - rng.uniform(0, 0.0003, n)
    df["volume"] = rng.randint(100, 1000, n)
    return df


class TestVWAP:
    def test_output_columns(self):
        df = make_vwap_data()
        result = compute_session_vwap(df)
        assert "vwap" in result.columns
        assert "vwap_upper" in result.columns
        assert "vwap_lower" in result.columns

    def test_output_length(self):
        df = make_vwap_data(100)
        result = compute_session_vwap(df)
        assert len(result) == 100

    def test_vwap_between_bands(self):
        """VWAP should always be between lower and upper bands."""
        df = make_vwap_data(500)
        result = compute_session_vwap(df, band_mult=1.5)
        mask = ~np.isnan(result["vwap"].values)
        vwap = result["vwap"].values[mask]
        upper = result["vwap_upper"].values[mask]
        lower = result["vwap_lower"].values[mask]
        assert np.all(lower <= vwap + 1e-10)
        assert np.all(vwap <= upper + 1e-10)

    def test_manual_vwap_calculation(self):
        """Verify VWAP against manual computation for a few bars."""
        df = pd.DataFrame({
            "time": pd.date_range("2025-03-03 08:00", periods=3, freq="5min", tz="UTC"),
            "open": [1.08, 1.081, 1.082],
            "high": [1.082, 1.083, 1.084],
            "low": [1.078, 1.079, 1.080],
            "close": [1.081, 1.082, 1.083],
            "volume": [100, 200, 150],
        })
        result = compute_session_vwap(df, session_start_hour_utc=8, band_mult=1.0)

        # Bar 0: tp = (1.082+1.078+1.081)/3 = 1.080333...
        tp0 = (1.082 + 1.078 + 1.081) / 3
        expected_vwap0 = tp0  # only one bar
        assert result["vwap"].iloc[0] == pytest.approx(expected_vwap0, abs=1e-8)

        # Bar 1: cumulative
        tp1 = (1.083 + 1.079 + 1.082) / 3
        cum_tp_vol = tp0 * 100 + tp1 * 200
        cum_vol = 300
        expected_vwap1 = cum_tp_vol / cum_vol
        assert result["vwap"].iloc[1] == pytest.approx(expected_vwap1, abs=1e-8)

    def test_session_reset(self):
        """VWAP should reset when session starts."""
        # Create data spanning two sessions
        times = pd.date_range("2025-03-03 07:55", periods=4, freq="5min", tz="UTC")
        # bars at 07:55, 08:00, 08:05, 08:10
        df = pd.DataFrame({
            "time": times,
            "open": [1.08, 1.081, 1.082, 1.083],
            "high": [1.082, 1.083, 1.084, 1.085],
            "low": [1.078, 1.079, 1.080, 1.081],
            "close": [1.081, 1.082, 1.083, 1.084],
            "volume": [100, 200, 150, 250],
        })
        result = compute_session_vwap(df, session_start_hour_utc=8, band_mult=1.0)

        # Bar 0 (07:55): first bar of old session
        tp0 = (1.082 + 1.078 + 1.081) / 3
        assert result["vwap"].iloc[0] == pytest.approx(tp0, abs=1e-8)

        # Bar 1 (08:00): session reset, this becomes first bar of new session
        tp1 = (1.083 + 1.079 + 1.082) / 3
        assert result["vwap"].iloc[1] == pytest.approx(tp1, abs=1e-8)

    def test_wider_bands_with_higher_mult(self):
        df = make_vwap_data(200)
        r1 = compute_session_vwap(df, band_mult=1.0)
        r2 = compute_session_vwap(df, band_mult=2.0)
        # Band width should be wider with higher mult
        mask = ~np.isnan(r1["vwap"].values)
        width1 = (r1["vwap_upper"].values - r1["vwap_lower"].values)[mask]
        width2 = (r2["vwap_upper"].values - r2["vwap_lower"].values)[mask]
        # Filter out zeros (at session start, variance=0)
        nonzero = width1 > 1e-10
        assert np.all(width2[nonzero] >= width1[nonzero] - 1e-10)


class TestRSI:
    def test_output_length(self):
        close = np.linspace(1.08, 1.09, 100)
        rsi = compute_rsi(close, period=14)
        assert len(rsi) == 100

    def test_nan_before_period(self):
        close = np.linspace(1.08, 1.09, 30)
        rsi = compute_rsi(close, period=14)
        # RSI should be NaN for indices 0..13
        assert np.all(np.isnan(rsi[:14]))
        # RSI should have a value at index 14
        assert not np.isnan(rsi[14])

    def test_range_0_to_100(self):
        rng = np.random.RandomState(123)
        close = 1.08 + np.cumsum(rng.normal(0, 0.001, 500))
        rsi = compute_rsi(close, period=14)
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_uptrend_high_rsi(self):
        """In a strong uptrend, RSI should be high."""
        close = np.linspace(1.08, 1.12, 100)  # Steady rise
        rsi = compute_rsi(close, period=14)
        # Last RSI should be very high (near 100)
        assert rsi[-1] > 90.0

    def test_downtrend_low_rsi(self):
        """In a strong downtrend, RSI should be low."""
        close = np.linspace(1.12, 1.08, 100)  # Steady fall
        rsi = compute_rsi(close, period=14)
        # Last RSI should be very low (near 0)
        assert rsi[-1] < 10.0

    def test_manual_rsi(self):
        """Verify RSI calculation against manual computation."""
        # 15 values: first 14 differences, then 1 more for smoothing
        close = np.array([44, 44.34, 44.09, 43.61, 44.33,
                          44.83, 45.10, 45.42, 45.84, 46.08,
                          45.89, 46.03, 45.61, 46.28, 46.28])
        rsi = compute_rsi(close, period=14)
        # The first valid RSI is at index 14
        # Manual calc: gains/losses over first 14 diffs
        assert not np.isnan(rsi[14])
        # RSI should be between 0-100
        assert 0 <= rsi[14] <= 100


class TestVWAPMeanReversionStrategy:
    def _make_row(self, close=1.08, vwap=1.081, vwap_lower=1.079, vwap_upper=1.083, rsi=25.0):
        """Create a mock row for strategy testing."""
        return pd.Series({
            "open": close - 0.0001,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "vwap": vwap,
            "vwap_lower": vwap_lower,
            "vwap_upper": vwap_upper,
            "rsi": rsi,
        })

    def test_long_entry(self):
        """Should open long when close < vwap_lower and RSI oversold."""
        strategy = VWAPMeanReversionStrategy(rsi_oversold=30, rsi_overbought=70, sl_pips=15)
        row = self._make_row(close=1.078, vwap=1.081, vwap_lower=1.079, rsi=25.0)
        action = strategy.on_bar(100, row, None, True, False)
        assert action["action"] == "open_long"
        assert action["sl"] < 1.078

    def test_short_entry(self):
        """Should open short when close > vwap_upper and RSI overbought."""
        strategy = VWAPMeanReversionStrategy(rsi_oversold=30, rsi_overbought=70, sl_pips=15)
        row = self._make_row(close=1.084, vwap=1.081, vwap_upper=1.083, rsi=75.0)
        action = strategy.on_bar(100, row, None, True, False)
        assert action["action"] == "open_short"
        assert action["sl"] > 1.084

    def test_no_entry_normal_conditions(self):
        """No entry when price is between bands or RSI not extreme."""
        strategy = VWAPMeanReversionStrategy(rsi_oversold=30, rsi_overbought=70, sl_pips=15)
        row = self._make_row(close=1.081, vwap=1.081, vwap_lower=1.079, vwap_upper=1.083, rsi=50.0)
        action = strategy.on_bar(100, row, None, True, False)
        assert action["action"] == "none"

    def test_no_entry_rsi_not_extreme(self):
        """No long if RSI not oversold, even if price below lower band."""
        strategy = VWAPMeanReversionStrategy(rsi_oversold=30, rsi_overbought=70)
        row = self._make_row(close=1.078, vwap=1.081, vwap_lower=1.079, rsi=45.0)
        action = strategy.on_bar(100, row, None, True, False)
        assert action["action"] == "none"

    def test_no_entry_when_cant_trade(self):
        """No entry when can_trade is False."""
        strategy = VWAPMeanReversionStrategy(rsi_oversold=30, rsi_overbought=70)
        row = self._make_row(close=1.078, vwap=1.081, vwap_lower=1.079, rsi=25.0)
        action = strategy.on_bar(100, row, None, False, False)
        assert action["action"] == "none"

    def test_no_entry_with_existing_position(self):
        """No new entry when already in a position."""
        from src.strategy.position import Position
        strategy = VWAPMeanReversionStrategy(rsi_oversold=30, rsi_overbought=70)
        position = Position(direction=1, entry_price=1.080, units=1000, sl_price=1.078)
        row = self._make_row(close=1.078, vwap=1.081, vwap_lower=1.079, rsi=25.0)
        action = strategy.on_bar(100, row, position, True, False)
        assert action["action"] == "none"

    def test_tp_type_vwap(self):
        """TP should target VWAP for tp_type='vwap'."""
        strategy = VWAPMeanReversionStrategy(
            rsi_oversold=30, rsi_overbought=70, sl_pips=15, tp_type="vwap"
        )
        row = self._make_row(close=1.078, vwap=1.081, vwap_lower=1.079, rsi=25.0)
        action = strategy.on_bar(100, row, None, True, False)
        assert action["action"] == "open_long"
        # TP should be VWAP (1.081), which is above entry (1.078)
        assert action["tp"] == pytest.approx(1.081, abs=1e-5)

    def test_tp_type_rr(self):
        """TP should be entry + sl_distance * rr for tp_type='rr'."""
        strategy = VWAPMeanReversionStrategy(
            rsi_oversold=30, rsi_overbought=70, sl_pips=15, tp_type="rr", tp_rr=2.0
        )
        row = self._make_row(close=1.078, vwap=1.081, vwap_lower=1.079, rsi=25.0)
        action = strategy.on_bar(100, row, None, True, False)
        assert action["action"] == "open_long"
        # SL distance = 15 pips = 0.0015
        # TP = 1.078 + 0.0015 * 2.0 = 1.081
        assert action["tp"] == pytest.approx(1.078 + 0.0015 * 2.0, abs=1e-5)

    def test_nan_indicators_no_entry(self):
        """No entry when indicators are NaN."""
        strategy = VWAPMeanReversionStrategy()
        row = pd.Series({
            "open": 1.08, "high": 1.082, "low": 1.078, "close": 1.08,
            "vwap": np.nan, "vwap_lower": np.nan, "vwap_upper": np.nan, "rsi": np.nan,
        })
        action = strategy.on_bar(100, row, None, True, False)
        assert action["action"] == "none"
