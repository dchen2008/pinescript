"""Pivot Point SuperTrend (PPST) indicator.

Faithful port of ppst_official_bt.pine lines 90-116.

Key implementation details:
1. Pivot detection (period=2): pivothigh(2,2) / pivotlow(2,2)
   - Detected with 2-bar delay, value is from 2 bars ago
2. Center line: center = (center_prev * 2 + lastpp) / 3 (weighted average, ph priority)
3. ATR: Wilder's RMA (not SMA): atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
4. Bands: Up = center - factor * ATR (support), Dn = center + factor * ATR (resistance)
5. Trailing: TUp[i] = max(Up, TUp[i-1]) if close[i-1] > TUp[i-1] else Up (sequential loop)
6. Trend: 1 if close > TDown[i-1], -1 if close < TUp[i-1], else prev (default 1)
7. Signals: buy = trend==1 and trend[1]==-1, sell = trend==-1 and trend[1]==1
"""

import numpy as np
import pandas as pd


def compute_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Compute True Range."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return tr


def compute_rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA (Running Moving Average / Exponential Moving Average).

    rma[i] = (rma[i-1] * (period-1) + values[i]) / period

    PineScript's atr() uses this internally.
    First value is SMA of first `period` values.
    """
    n = len(values)
    rma = np.full(n, np.nan)

    # First value: SMA of first period values
    if n < period:
        return rma

    rma[period - 1] = np.mean(values[:period])

    # Subsequent values: Wilder's smoothing
    for i in range(period, n):
        rma[i] = (rma[i - 1] * (period - 1) + values[i]) / period

    return rma


def detect_pivots(high: np.ndarray, low: np.ndarray, period: int) -> tuple:
    """Detect pivot highs and lows.

    pivothigh(prd, prd): bar at index i is a pivot high if high[i] is the highest
    in the range [i-prd, i+prd]. Since we need prd bars after, the pivot is
    confirmed at bar i+prd (detected with prd-bar delay).

    Returns:
        pivot_high: array of NaN except at bars where pivot is detected (value = high of pivot bar)
        pivot_low: array of NaN except at bars where pivot is detected (value = low of pivot bar)
    """
    n = len(high)
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)

    for i in range(period, n - period):
        # PineScript's ta.pivothigh uses lastindexof internally:
        # ties on the LEFT are allowed (>=), ties on the RIGHT disqualify (>).
        left_h = high[i - period: i]
        right_h = high[i + 1: i + period + 1]
        if (len(left_h) == 0 or high[i] >= np.max(left_h)) and \
           (len(right_h) == 0 or high[i] > np.max(right_h)):
            pivot_high[i + period] = high[i]

        left_l = low[i - period: i]
        right_l = low[i + 1: i + period + 1]
        if (len(left_l) == 0 or low[i] <= np.min(left_l)) and \
           (len(right_l) == 0 or low[i] < np.min(right_l)):
            pivot_low[i + period] = low[i]

    return pivot_high, pivot_low


def compute_ppst(
    df: pd.DataFrame,
    pivot_period: int = 2,
    atr_factor: float = 3.0,
    atr_period: int = 10,
) -> pd.DataFrame:
    """Compute the full PPST indicator.

    Args:
        df: DataFrame with columns: open, high, low, close
        pivot_period: Pivot detection lookback/forward (default 2)
        atr_factor: ATR multiplier for bands (default 3.0)
        atr_period: ATR smoothing period (default 10)

    Returns:
        DataFrame with all original columns plus indicator columns:
        pivot_high, pivot_low, center, atr, up_band, dn_band,
        tup, tdown, trend, trailing_sl, buy_signal, sell_signal,
        support, resistance
    """
    result = df.copy()
    n = len(df)

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_ = df["open"].values

    # Step 1: Detect pivots
    pivot_high, pivot_low = detect_pivots(high, low, pivot_period)
    result["pivot_high"] = pivot_high
    result["pivot_low"] = pivot_low

    # Step 2: Compute center line
    # center = (center_prev * 2 + lastpp) / 3
    # ph takes priority over pl (lastpp = ph if ph else pl)
    center = np.full(n, np.nan)
    for i in range(n):
        ph = pivot_high[i]
        pl = pivot_low[i]
        # lastpp: ph takes priority
        if not np.isnan(ph):
            lastpp = ph
        elif not np.isnan(pl):
            lastpp = pl
        else:
            lastpp = np.nan

        if not np.isnan(lastpp):
            if np.isnan(center[i - 1] if i > 0 else np.nan):
                center[i] = lastpp
            else:
                center[i] = (center[i - 1] * 2 + lastpp) / 3
        else:
            center[i] = center[i - 1] if i > 0 else np.nan

    result["center"] = center

    # Step 3: Compute ATR using Wilder's RMA
    tr = compute_true_range(high, low, close)
    atr = compute_rma(tr, atr_period)
    result["atr"] = atr

    # Step 4: Compute bands
    # Up = center - factor * ATR  (support band)
    # Dn = center + factor * ATR  (resistance band)
    up_band = center - atr_factor * atr
    dn_band = center + atr_factor * atr
    result["up_band"] = up_band
    result["dn_band"] = dn_band

    # Step 5: Trailing bands (MUST be sequential loop - self-referencing)
    # TUp[i] = max(Up[i], TUp[i-1]) if close[i-1] > TUp[i-1] else Up[i]
    # TDown[i] = min(Dn[i], TDown[i-1]) if close[i-1] < TDown[i-1] else Dn[i]
    tup = np.full(n, np.nan)
    tdown = np.full(n, np.nan)
    trend = np.full(n, 0, dtype=int)

    for i in range(1, n):
        if np.isnan(up_band[i]) or np.isnan(dn_band[i]):
            tup[i] = tup[i - 1]
            tdown[i] = tdown[i - 1]
            trend[i] = trend[i - 1] if trend[i - 1] != 0 else 1
            continue

        # TUp calculation
        if not np.isnan(tup[i - 1]) and close[i - 1] > tup[i - 1]:
            tup[i] = max(up_band[i], tup[i - 1])
        else:
            tup[i] = up_band[i]

        # TDown calculation
        if not np.isnan(tdown[i - 1]) and close[i - 1] < tdown[i - 1]:
            tdown[i] = min(dn_band[i], tdown[i - 1])
        else:
            tdown[i] = dn_band[i]

        # Trend calculation
        # Trend = close > TDown[i-1] ? 1 : close < TUp[i-1] ? -1 : nz(Trend[i-1], 1)
        if not np.isnan(tdown[i - 1]) and close[i] > tdown[i - 1]:
            trend[i] = 1
        elif not np.isnan(tup[i - 1]) and close[i] < tup[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1] if trend[i - 1] != 0 else 1

    result["tup"] = tup
    result["tdown"] = tdown
    result["trend"] = trend

    # Step 6: Trailing SL line
    trailing_sl = np.where(trend == 1, tup, tdown)
    result["trailing_sl"] = trailing_sl

    # Step 7: Buy/Sell signals
    # buy = trend==1 and trend[1]==-1
    # sell = trend==-1 and trend[1]==1
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    for i in range(1, n):
        buy_signal[i] = trend[i] == 1 and trend[i - 1] == -1
        sell_signal[i] = trend[i] == -1 and trend[i - 1] == 1

    result["buy_signal"] = buy_signal
    result["sell_signal"] = sell_signal

    # Step 8: Support/Resistance (rolling pivot levels)
    # support = pl if pl else support[1]
    # resistance = ph if ph else resistance[1]
    support = np.full(n, np.nan)
    resistance = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(pivot_low[i]):
            support[i] = pivot_low[i]
        elif i > 0:
            support[i] = support[i - 1]

        if not np.isnan(pivot_high[i]):
            resistance[i] = pivot_high[i]
        elif i > 0:
            resistance[i] = resistance[i - 1]

    result["support"] = support
    result["resistance"] = resistance

    return result
