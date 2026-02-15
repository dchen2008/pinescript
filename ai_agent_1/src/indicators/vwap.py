"""Session VWAP (Volume Weighted Average Price) with standard deviation bands.

VWAP = cumsum(typical_price * volume) / cumsum(volume)
Resets at session start (configurable).

For forex (24h market), session-based reset is more meaningful than daily reset.
Uses OANDA tick volume as proxy for real volume (~0.7-0.8 correlation).
"""

import numpy as np
import pandas as pd


def compute_session_vwap(
    df: pd.DataFrame,
    session_start_hour_utc: int = 8,
    band_mult: float = 1.5,
) -> pd.DataFrame:
    """Compute session-based VWAP with standard deviation bands.

    Args:
        df: DataFrame with columns: time, high, low, close, volume
        session_start_hour_utc: Hour (UTC) to reset VWAP. Default 8 = London open.
        band_mult: Standard deviation multiplier for bands.

    Returns:
        DataFrame with columns: vwap, vwap_upper, vwap_lower
    """
    n = len(df)
    times = pd.to_datetime(df["time"])
    if times.dt.tz is not None:
        times_utc = times.dt.tz_convert("UTC")
    else:
        times_utc = times.dt.tz_localize("UTC")

    hours = times_utc.dt.hour.values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    volume = df["volume"].values.astype(float)

    typical_price = (high + low + close) / 3.0

    vwap = np.full(n, np.nan)
    vwap_upper = np.full(n, np.nan)
    vwap_lower = np.full(n, np.nan)

    cum_tp_vol = 0.0
    cum_vol = 0.0
    cum_tp2_vol = 0.0  # for variance: sum(tp^2 * vol)

    prev_hour = -1

    for i in range(n):
        cur_hour = hours[i]

        # Reset at session start: when hour transitions to session_start_hour_utc
        if cur_hour == session_start_hour_utc and prev_hour != session_start_hour_utc:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            cum_tp2_vol = 0.0

        prev_hour = cur_hour

        vol = volume[i]
        tp = typical_price[i]

        cum_tp_vol += tp * vol
        cum_vol += vol
        cum_tp2_vol += tp * tp * vol

        if cum_vol > 0:
            v = cum_tp_vol / cum_vol
            vwap[i] = v
            # Variance = E[X^2] - E[X]^2
            variance = cum_tp2_vol / cum_vol - v * v
            std = np.sqrt(max(variance, 0.0))
            vwap_upper[i] = v + band_mult * std
            vwap_lower[i] = v - band_mult * std

    return pd.DataFrame({
        "vwap": vwap,
        "vwap_upper": vwap_upper,
        "vwap_lower": vwap_lower,
    })
