"""Wilder's RSI (Relative Strength Index).

Uses Wilder's smoothing (same as RMA) for average gain/loss,
matching PineScript's ta.rsi() behavior.
"""

import numpy as np


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Wilder's RSI.

    Args:
        close: Array of close prices.
        period: RSI lookback period. Default 14.

    Returns:
        Array of RSI values (0-100). NaN before enough data.
    """
    n = len(close)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        return rsi

    # Calculate price changes
    delta = np.diff(close)  # length n-1
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)

    # First average: SMA of first `period` values
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Subsequent: Wilder's smoothing (same as RMA)
    # avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi
