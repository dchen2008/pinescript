#!/usr/bin/env python3
"""PPST Indicator Diagnostics — cross-reference Python vs TradingView values.

Outputs detailed bar-level and signal-level data for the first N signals
to help identify divergence between the Python and PineScript implementations.

Usage:
    cd /Users/boscini/github/pinescript/ai_agent_1
    python3 -m scripts.ppst_diagnostics
"""

import numpy as np
import pandas as pd

from src.data.data_manager import load_candles
from src.indicators.ppst import compute_ppst


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INSTRUMENT = "EUR_USD"
GRANULARITY = "M5"
PIVOT_PERIOD = 2
ATR_FACTOR = 5.0
ATR_PERIOD = 10
CONTEXT_BARS = 3          # bars before/after each signal to print
NUM_SIGNALS_DETAIL = 10   # signals with surrounding bar context
NUM_SIGNALS_SUMMARY = 20  # signals in the summary table


def fmt_price(val, decimals=5):
    """Format a price value with fixed decimals, or '-' if NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-".rjust(decimals + 3)
    return f"{val:.{decimals}f}"


def fmt_int(val):
    """Format an integer or NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return str(int(val))


def print_bar_header():
    """Print the column header for bar-level output."""
    hdr = (
        f"{'idx':>6}  {'time':>19}  "
        f"{'open':>9}  {'high':>9}  {'low':>9}  {'close':>9}  "
        f"{'center':>9}  {'atr':>9}  "
        f"{'up_band':>9}  {'dn_band':>9}  "
        f"{'tup':>9}  {'tdown':>9}  "
        f"{'trend':>5}  {'trail_sl':>9}  "
        f"{'ph':>9}  {'pl':>9}  "
        f"{'buy':>3}  {'sell':>3}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_bar_row(idx, row):
    """Print a single bar row."""
    print(
        f"{idx:>6}  {str(row['time']):>19}  "
        f"{fmt_price(row['open'])}  {fmt_price(row['high'])}  "
        f"{fmt_price(row['low'])}  {fmt_price(row['close'])}  "
        f"{fmt_price(row['center'])}  {fmt_price(row['atr'])}  "
        f"{fmt_price(row['up_band'])}  {fmt_price(row['dn_band'])}  "
        f"{fmt_price(row['tup'])}  {fmt_price(row['tdown'])}  "
        f"{int(row['trend']):>5}  {fmt_price(row['trailing_sl'])}  "
        f"{fmt_price(row['pivot_high'])}  {fmt_price(row['pivot_low'])}  "
        f"{'Y' if row['buy_signal'] else '.':>3}  "
        f"{'Y' if row['sell_signal'] else '.':>3}"
    )


def main():
    # ------------------------------------------------------------------
    # 1. Load data and compute PPST
    # ------------------------------------------------------------------
    print(f"Loading {INSTRUMENT} {GRANULARITY} data...")
    df = load_candles(INSTRUMENT, GRANULARITY)
    print(f"  Loaded {len(df)} candles: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    print(f"\nComputing PPST (pivot={PIVOT_PERIOD}, atr_factor={ATR_FACTOR}, atr_period={ATR_PERIOD})...")
    data = compute_ppst(df, pivot_period=PIVOT_PERIOD, atr_factor=ATR_FACTOR, atr_period=ATR_PERIOD)

    # ------------------------------------------------------------------
    # 2. Find all signal bars
    # ------------------------------------------------------------------
    buy_idxs = data.index[data["buy_signal"]].tolist()
    sell_idxs = data.index[data["sell_signal"]].tolist()

    all_signals = []
    for i in buy_idxs:
        all_signals.append((i, "BUY"))
    for i in sell_idxs:
        all_signals.append((i, "SELL"))
    all_signals.sort(key=lambda x: x[0])

    print(f"\n  Total BUY signals:  {len(buy_idxs)}")
    print(f"  Total SELL signals: {len(sell_idxs)}")
    print(f"  Total signals:      {len(all_signals)}")

    # ------------------------------------------------------------------
    # 3. Warmup diagnostics — show how center / ATR / bands initialize
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("WARMUP DIAGNOSTICS — first 20 bars (center & ATR initialization)")
    print("=" * 120)
    print_bar_header()
    for idx in range(min(20, len(data))):
        print_bar_row(idx, data.iloc[idx])

    # ------------------------------------------------------------------
    # 4. First pivot events
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("FIRST 10 PIVOT EVENTS (where center line changes)")
    print("=" * 120)

    pivot_bars = data.index[data["pivot_high"].notna() | data["pivot_low"].notna()].tolist()
    print(f"  Total pivot events: {len(pivot_bars)}")
    print()
    print_bar_header()
    for count, pidx in enumerate(pivot_bars[:10]):
        # Print the pivot bar and one bar before/after for context
        start = max(0, pidx - 1)
        end = min(len(data), pidx + 2)
        for idx in range(start, end):
            marker = " <<< PIVOT" if idx == pidx else ""
            # We print the row then append the marker on the same line
            print_bar_row(idx, data.iloc[idx])
        print()  # blank line between pivot events

    # ------------------------------------------------------------------
    # 5. Detailed bar context around first N signals
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print(f"DETAILED BAR CONTEXT — first {NUM_SIGNALS_DETAIL} signals (+/- {CONTEXT_BARS} bars)")
    print("=" * 120)

    for sig_num, (sig_idx, sig_type) in enumerate(all_signals[:NUM_SIGNALS_DETAIL]):
        row = data.iloc[sig_idx]
        print(f"\n--- Signal #{sig_num + 1}: {sig_type} at bar {sig_idx} "
              f"({row['time']}) ---")
        print_bar_header()
        start = max(0, sig_idx - CONTEXT_BARS)
        end = min(len(data), sig_idx + CONTEXT_BARS + 1)
        for idx in range(start, end):
            print_bar_row(idx, data.iloc[idx])
        print()

    # ------------------------------------------------------------------
    # 6. Signal summary table (first 20 signals)
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print(f"SIGNAL SUMMARY — first {NUM_SIGNALS_SUMMARY} signals")
    print("=" * 120)

    hdr = (
        f"{'#':>3}  {'type':>4}  {'idx':>6}  {'time':>19}  "
        f"{'close':>9}  "
        f"{'tup':>9}  {'tdown':>9}  {'trail_sl':>9}  "
        f"{'center':>9}  {'atr':>9}  "
        f"{'up_band':>9}  {'dn_band':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for sig_num, (sig_idx, sig_type) in enumerate(all_signals[:NUM_SIGNALS_SUMMARY]):
        row = data.iloc[sig_idx]
        print(
            f"{sig_num + 1:>3}  {sig_type:>4}  {sig_idx:>6}  "
            f"{str(row['time']):>19}  "
            f"{fmt_price(row['close'])}  "
            f"{fmt_price(row['tup'])}  {fmt_price(row['tdown'])}  "
            f"{fmt_price(row['trailing_sl'])}  "
            f"{fmt_price(row['center'])}  {fmt_price(row['atr'])}  "
            f"{fmt_price(row['up_band'])}  {fmt_price(row['dn_band'])}"
        )

    # ------------------------------------------------------------------
    # 7. Consecutive trend values around each signal (sanity check)
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print(f"TREND TRANSITION CHECK — first {NUM_SIGNALS_SUMMARY} signals")
    print("  Shows trend[i-2], trend[i-1], trend[i], trend[i+1] around signal bar")
    print("=" * 120)

    hdr2 = f"{'#':>3}  {'type':>4}  {'idx':>6}  {'t[i-2]':>6}  {'t[i-1]':>6}  {'t[i]':>6}  {'t[i+1]':>6}"
    print(hdr2)
    print("-" * len(hdr2))

    trend_arr = data["trend"].values
    for sig_num, (sig_idx, sig_type) in enumerate(all_signals[:NUM_SIGNALS_SUMMARY]):
        vals = []
        for offset in [-2, -1, 0, 1]:
            j = sig_idx + offset
            if 0 <= j < len(trend_arr):
                vals.append(f"{trend_arr[j]:>6}")
            else:
                vals.append(f"{'N/A':>6}")
        print(f"{sig_num + 1:>3}  {sig_type:>4}  {sig_idx:>6}  {'  '.join(vals)}")

    # ------------------------------------------------------------------
    # 8. ATR detail around first signal (for manual verification)
    # ------------------------------------------------------------------
    if all_signals:
        first_sig_idx = all_signals[0][0]
        print("\n" + "=" * 120)
        print(f"ATR DETAIL — bars {max(0, first_sig_idx - 15)} to {first_sig_idx + 2}")
        print("  For manual Wilder's RMA verification")
        print("=" * 120)

        from src.indicators.ppst import compute_true_range
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        tr = compute_true_range(high, low, close)

        hdr3 = (
            f"{'idx':>6}  {'time':>19}  "
            f"{'high':>9}  {'low':>9}  {'close':>9}  "
            f"{'TR':>9}  {'ATR(rma)':>9}"
        )
        print(hdr3)
        print("-" * len(hdr3))

        atr_arr = data["atr"].values
        start = max(0, first_sig_idx - 15)
        end = min(len(data), first_sig_idx + 3)
        for idx in range(start, end):
            print(
                f"{idx:>6}  {str(data.iloc[idx]['time']):>19}  "
                f"{fmt_price(high[idx])}  {fmt_price(low[idx])}  {fmt_price(close[idx])}  "
                f"{fmt_price(tr[idx])}  {fmt_price(atr_arr[idx])}"
            )

    print("\n\nDiagnostics complete.")


if __name__ == "__main__":
    main()
