"""Entry circle detection (4-condition logic).

Faithful port of ppst_v1_circle.pine lines 100-152.

LONG Entry Circle conditions (all must be true):
1. open < resistance AND close > resistance (candle crosses resistance from below)
2. body_mid > resistance (50%+ of body above resistance)
3. close > center (close above center line)

SHORT Entry Circle conditions (all must be true):
1. open > support AND close < support (candle crosses support from above)
2. body_mid < support (50%+ of body below support)
3. close < center (close below center line)

On signal bar: use original resistance/support
After signal bar: use entry_resistance/entry_support (formed after signal)
"""

import numpy as np
import pandas as pd


def compute_entry_circles(df: pd.DataFrame) -> pd.DataFrame:
    """Detect entry circle conditions on each bar.

    Requires df to have: open, high, low, close, trend, buy_signal, sell_signal,
    center, support, resistance, pivot_high, pivot_low

    Returns:
        DataFrame with columns: long_circle, short_circle,
        long_circle_on_signal, long_circle_after_signal,
        short_circle_on_signal, short_circle_after_signal,
        entry_resistance, entry_support
    """
    n = len(df)
    open_ = df["open"].values
    close = df["close"].values
    trend = df["trend"].values
    buy_signal = df["buy_signal"].values
    sell_signal = df["sell_signal"].values
    center = df["center"].values
    support = df["support"].values
    resistance = df["resistance"].values
    pivot_high = df["pivot_high"].values
    pivot_low = df["pivot_low"].values

    # Candle body midpoint
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    body_mid = (body_top + body_bottom) / 2

    # Track state: entry_resistance/support form AFTER signal
    has_ph_after_signal = np.zeros(n, dtype=bool)
    has_pl_after_signal = np.zeros(n, dtype=bool)
    entry_resistance = np.full(n, np.nan)
    entry_support = np.full(n, np.nan)

    # Circle count and signal timing
    last_signal_bar = np.zeros(n, dtype=int)
    circle_count_since_signal = np.zeros(n, dtype=int)

    long_circle_on_signal = np.zeros(n, dtype=bool)
    long_circle_after_signal = np.zeros(n, dtype=bool)
    short_circle_on_signal = np.zeros(n, dtype=bool)
    short_circle_after_signal = np.zeros(n, dtype=bool)

    _has_ph = False
    _has_pl = False
    _entry_res = np.nan
    _entry_sup = np.nan
    _last_sig_bar = 0
    _circle_count = 0

    for i in range(n):
        # Reset on new signals
        if buy_signal[i]:
            _has_ph = False
            _entry_res = np.nan
            _last_sig_bar = i
            _circle_count = 0
        if sell_signal[i]:
            _has_pl = False
            _entry_sup = np.nan
            _last_sig_bar = i
            _circle_count = 0

        # Detect new PH after BUY signal
        if trend[i] == 1 and not np.isnan(pivot_high[i]):
            _has_ph = True
            _entry_res = pivot_high[i]

        # Detect new PL after SELL signal
        if trend[i] == -1 and not np.isnan(pivot_low[i]):
            _has_pl = True
            _entry_sup = pivot_low[i]

        has_ph_after_signal[i] = _has_ph
        has_pl_after_signal[i] = _has_pl
        entry_resistance[i] = _entry_res
        entry_support[i] = _entry_sup
        last_signal_bar[i] = _last_sig_bar

        # LONG circle on signal bar
        if buy_signal[i] and not np.isnan(resistance[i]):
            if (open_[i] < resistance[i] and close[i] > resistance[i]
                    and body_mid[i] > resistance[i] and close[i] > center[i]):
                long_circle_on_signal[i] = True

        # LONG circle after signal
        if (trend[i] == 1 and not buy_signal[i] and _has_ph
                and not np.isnan(_entry_res)):
            if (open_[i] < _entry_res and close[i] > _entry_res
                    and body_mid[i] > _entry_res and close[i] > center[i]):
                long_circle_after_signal[i] = True

        # SHORT circle on signal bar
        if sell_signal[i] and not np.isnan(support[i]):
            if (open_[i] > support[i] and close[i] < support[i]
                    and body_mid[i] < support[i] and close[i] < center[i]):
                short_circle_on_signal[i] = True

        # SHORT circle after signal
        if (trend[i] == -1 and not sell_signal[i] and _has_pl
                and not np.isnan(_entry_sup)):
            if (open_[i] > _entry_sup and close[i] < _entry_sup
                    and body_mid[i] < _entry_sup and close[i] < center[i]):
                short_circle_after_signal[i] = True

        # Count circles
        is_long_circle = long_circle_on_signal[i] or long_circle_after_signal[i]
        is_short_circle = short_circle_on_signal[i] or short_circle_after_signal[i]
        if is_long_circle or is_short_circle:
            _circle_count += 1

        circle_count_since_signal[i] = _circle_count

    long_circle = long_circle_on_signal | long_circle_after_signal
    short_circle = short_circle_on_signal | short_circle_after_signal

    return pd.DataFrame({
        "long_circle": long_circle,
        "short_circle": short_circle,
        "long_circle_on_signal": long_circle_on_signal,
        "long_circle_after_signal": long_circle_after_signal,
        "short_circle_on_signal": short_circle_on_signal,
        "short_circle_after_signal": short_circle_after_signal,
        "entry_resistance": entry_resistance,
        "entry_support": entry_support,
        "last_signal_bar": last_signal_bar,
        "circle_count_since_signal": circle_count_since_signal,
    }, index=df.index)
