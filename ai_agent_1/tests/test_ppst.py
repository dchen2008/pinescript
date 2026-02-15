"""Test PPST indicator against known PineScript behavior."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.ppst import (
    compute_true_range,
    compute_rma,
    detect_pivots,
    compute_ppst,
)


def make_sample_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic EUR/USD-like candle data."""
    rng = np.random.RandomState(seed)
    base = 1.08
    prices = [base]
    for _ in range(n - 1):
        change = rng.normal(0, 0.0005)
        prices.append(prices[-1] + change)

    df = pd.DataFrame()
    df["time"] = pd.date_range("2025-01-06 00:00", periods=n, freq="1min", tz="UTC")
    df["close"] = prices
    df["open"] = [prices[0]] + prices[:-1]
    df["high"] = np.maximum(df["open"], df["close"]) + rng.uniform(0, 0.0003, n)
    df["low"] = np.minimum(df["open"], df["close"]) - rng.uniform(0, 0.0003, n)
    df["volume"] = rng.randint(10, 200, n)
    return df


class TestTrueRange:
    def test_first_bar(self):
        high = np.array([1.085])
        low = np.array([1.080])
        close = np.array([1.083])
        tr = compute_true_range(high, low, close)
        assert tr[0] == pytest.approx(0.005)

    def test_gap_up(self):
        high = np.array([1.085, 1.090])
        low = np.array([1.080, 1.087])
        close = np.array([1.083, 1.089])
        tr = compute_true_range(high, low, close)
        # Bar 1: max(0.003, |1.090-1.083|, |1.087-1.083|) = max(0.003, 0.007, 0.004)
        assert tr[1] == pytest.approx(0.007)


class TestRMA:
    def test_basic(self):
        values = np.arange(1.0, 11.0)  # [1, 2, ..., 10]
        period = 3
        rma = compute_rma(values, period)
        # First value: SMA of [1,2,3] = 2.0
        assert rma[2] == pytest.approx(2.0)
        # Next: (2.0 * 2 + 4) / 3 = 8/3 ≈ 2.6667
        assert rma[3] == pytest.approx(8.0 / 3.0)

    def test_nan_before_period(self):
        values = np.ones(10)
        rma = compute_rma(values, 5)
        assert np.isnan(rma[3])
        assert rma[4] == pytest.approx(1.0)


class TestPivots:
    def test_simple_pivot_high(self):
        # Create clear pivot high at index 5
        high = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1])
        low = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        ph, pl = detect_pivots(high, low, period=2)
        # Pivot high at index 5, detected at index 7 (5 + 2)
        assert ph[7] == pytest.approx(1.5)

    def test_simple_pivot_low(self):
        low = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4])
        high = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        ph, pl = detect_pivots(high, low, period=2)
        # Pivot low at index 5, detected at index 7
        assert pl[7] == pytest.approx(1.0)


class TestPPST:
    def test_output_columns(self):
        df = make_sample_data(200)
        result = compute_ppst(df, pivot_period=2, atr_factor=3.0, atr_period=10)
        expected_cols = [
            "pivot_high", "pivot_low", "center", "atr",
            "up_band", "dn_band", "tup", "tdown",
            "trend", "trailing_sl", "buy_signal", "sell_signal",
            "support", "resistance",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_trend_values(self):
        df = make_sample_data(500)
        result = compute_ppst(df)
        # Trend should only be -1, 0, or 1
        unique_trends = set(result["trend"].unique())
        assert unique_trends.issubset({-1, 0, 1})

    def test_buy_sell_signals_are_trend_changes(self):
        df = make_sample_data(500)
        result = compute_ppst(df)
        trend = result["trend"].values
        buy = result["buy_signal"].values
        sell = result["sell_signal"].values

        for i in range(1, len(result)):
            if buy[i]:
                assert trend[i] == 1 and trend[i - 1] == -1
            if sell[i]:
                assert trend[i] == -1 and trend[i - 1] == 1

    def test_trailing_sl_follows_trend(self):
        df = make_sample_data(500)
        result = compute_ppst(df)
        trend = result["trend"].values
        tsl = result["trailing_sl"].values
        tup = result["tup"].values
        tdown = result["tdown"].values

        for i in range(1, len(result)):
            if trend[i] == 1 and not np.isnan(tup[i]):
                assert tsl[i] == pytest.approx(tup[i])
            elif trend[i] == -1 and not np.isnan(tdown[i]):
                assert tsl[i] == pytest.approx(tdown[i])

    def test_center_weighted_average(self):
        """Verify center = (center_prev * 2 + lastpp) / 3."""
        df = make_sample_data(200)
        result = compute_ppst(df)
        center = result["center"].values
        ph = result["pivot_high"].values
        pl = result["pivot_low"].values

        # Find bars with pivot points and check the formula
        for i in range(1, len(result)):
            lastpp = None
            if not np.isnan(ph[i]):
                lastpp = ph[i]
            elif not np.isnan(pl[i]):
                lastpp = pl[i]

            if lastpp is not None and not np.isnan(center[i - 1]):
                expected = (center[i - 1] * 2 + lastpp) / 3
                assert center[i] == pytest.approx(expected, abs=1e-10), \
                    f"Center mismatch at bar {i}: {center[i]} != {expected}"

    def test_tup_only_increases_when_above(self):
        """TUp should only increase (never decrease) when close[i-1] > TUp[i-1]."""
        df = make_sample_data(500)
        result = compute_ppst(df)
        tup = result["tup"].values
        close = result["close"].values

        for i in range(2, len(result)):
            if not np.isnan(tup[i]) and not np.isnan(tup[i - 1]):
                if close[i - 1] > tup[i - 1]:
                    assert tup[i] >= tup[i - 1], \
                        f"TUp decreased at bar {i}: {tup[i]} < {tup[i-1]}"
