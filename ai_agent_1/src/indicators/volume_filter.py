"""Relative volume filter.

Computes volume relative to its simple moving average.
Values > 1.0 indicate above-average volume.
"""

import numpy as np


def compute_relative_volume(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Compute relative volume: volume / SMA(volume, period).

    Args:
        volume: Array of tick volume values.
        period: SMA lookback period. Default 20.

    Returns:
        Array of relative volume ratios. NaN before `period` bars.
    """
    n = len(volume)
    rel_vol = np.full(n, np.nan)

    if n < period:
        return rel_vol

    # Compute rolling SMA using cumsum for efficiency
    cumsum = np.cumsum(volume, dtype=np.float64)
    sma = np.full(n, np.nan)
    sma[period - 1] = cumsum[period - 1] / period
    for i in range(period, n):
        sma[i] = (cumsum[i] - cumsum[i - period]) / period

    # Avoid division by zero
    mask = sma > 0
    rel_vol[mask] = volume[mask] / sma[mask]

    return rel_vol
