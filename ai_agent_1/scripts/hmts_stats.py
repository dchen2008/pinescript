#!/usr/bin/env python3
"""HMTS statistics: analyze M5 data for huge movement events and reversals.

Counts distinct large-movement events (non-overlapping) and measures
post-event reversals to help calibrate HMTS parameters.

Usage:
    python3 scripts/hmts_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_manager import load_candles


def count_huge_movements(df: pd.DataFrame, window: int, pip_thresholds: list[float]) -> dict:
    """Count distinct huge movement events per year.

    For each rolling window, compute max(high) - min(low). Group consecutive
    qualifying bars into single events (non-overlapping).

    Returns:
        {year: {threshold: count}}
    """
    high = df["high"].values
    low = df["low"].values
    times = df["time"].values

    n = len(df)
    # Compute rolling range (max high - min low) for each window
    ranges = np.full(n, np.nan)
    for i in range(window - 1, n):
        win_high = high[i - window + 1:i + 1].max()
        win_low = low[i - window + 1:i + 1].min()
        ranges[i] = (win_high - win_low) / 0.0001  # in pips

    # Extract years
    years = pd.DatetimeIndex(times).year

    results = {}
    for year in sorted(set(years)):
        year_mask = years == year
        year_ranges = ranges[year_mask]

        results[year] = {}
        for threshold in pip_thresholds:
            # Find qualifying bars
            qualifying = year_ranges >= threshold
            # Group consecutive qualifying bars into single events
            count = 0
            in_event = False
            for q in qualifying:
                if q and not np.isnan(q):
                    if not in_event:
                        count += 1
                        in_event = True
                else:
                    in_event = False
            results[year][threshold] = count

    return results


def count_reversals(df: pd.DataFrame, move_window: int, move_threshold: float,
                    reversal_windows: list[int], reversal_thresholds: list[float]) -> dict:
    """Count reversal events after huge movements.

    Detect huge movement events (non-overlapping), then from each event's
    extreme point, measure the max reversal in the next M candles.

    Returns:
        {reversal_window: {year: {threshold: count}}}
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    times = df["time"].values
    n = len(df)
    years = pd.DatetimeIndex(times).year

    # First find all huge movement events (non-overlapping)
    events = []  # list of (end_idx, direction, extreme_price)
    i = move_window - 1
    while i < n:
        win_high = high[i - move_window + 1:i + 1].max()
        win_low = low[i - move_window + 1:i + 1].min()
        range_pips = (win_high - win_low) / 0.0001

        if range_pips >= move_threshold:
            # Determine direction: was it an up move or down move?
            high_idx = i - move_window + 1 + np.argmax(high[i - move_window + 1:i + 1])
            low_idx = i - move_window + 1 + np.argmin(low[i - move_window + 1:i + 1])

            if high_idx > low_idx:
                # Up move: extreme is the high
                direction = 1
                extreme_price = win_high
            else:
                # Down move: extreme is the low
                direction = -1
                extreme_price = win_low

            events.append((i, direction, extreme_price))
            # Skip ahead to avoid overlapping events
            i += move_window
        else:
            i += 1

    # For each reversal window, measure reversals
    results = {}
    for rev_window in reversal_windows:
        results[rev_window] = {}
        for year in sorted(set(years)):
            results[rev_window][year] = {}
            for rev_thresh in reversal_thresholds:
                count = 0
                for end_idx, direction, extreme_price in events:
                    if years[end_idx] != year:
                        continue
                    # Measure max reversal in next rev_window candles
                    rev_end = min(end_idx + rev_window + 1, n)
                    if rev_end <= end_idx + 1:
                        continue

                    if direction == 1:
                        # Up move, reversal is price dropping from high
                        min_after = low[end_idx + 1:rev_end].min()
                        reversal_pips = (extreme_price - min_after) / 0.0001
                    else:
                        # Down move, reversal is price rising from low
                        max_after = high[end_idx + 1:rev_end].max()
                        reversal_pips = (max_after - extreme_price) / 0.0001

                    if reversal_pips >= rev_thresh:
                        count += 1

                results[rev_window][year][rev_thresh] = count

    return results


def print_movement_table(results: dict, window: int, thresholds: list[float]):
    """Print formatted movement count table."""
    header = f"Window: {window} candles ({window * 5} min)"
    print(header)

    # Table header
    cols = [f">{int(t)} pips" for t in thresholds]
    print(f"{'Year':<8}|" + "|".join(f"{c:>10}" for c in cols))
    print("-" * 8 + "|" + "|".join("-" * 10 for _ in cols))

    for year in sorted(results.keys()):
        row = f"{year:<8}|"
        for t in thresholds:
            row += f"{results[year].get(t, 0):>10}|"
        print(row)
    print()


def print_reversal_table(results: dict, rev_window: int, years: list, thresholds: list[float]):
    """Print formatted reversal count table."""
    header = f"Window: {rev_window} candles ({rev_window * 5} min)"
    print(header)

    cols = [f">{int(t)} pips" for t in thresholds]
    print(f"{'Year':<8}|" + "|".join(f"{c:>10}" for c in cols))
    print("-" * 8 + "|" + "|".join("-" * 10 for _ in cols))

    for year in sorted(years):
        row = f"{year:<8}|"
        for t in thresholds:
            row += f"{results.get(year, {}).get(t, 0):>10}|"
        print(row)
    print()


def main():
    print("Loading EUR_USD M5 data...")
    df = load_candles("EUR_USD", "M5")
    print(f"Loaded {len(df)} candles: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print()

    # --- Huge Movement Events ---
    movement_windows = [3, 5, 7, 10]
    movement_thresholds = [50.0, 60.0, 70.0, 80.0, 90.0]

    print("=" * 60)
    print("=== Huge Movement Events (M5 EUR/USD) ===")
    print("=" * 60)
    print()

    for window in movement_windows:
        results = count_huge_movements(df, window, movement_thresholds)
        print_movement_table(results, window, movement_thresholds)

    # --- Reversal Events ---
    move_window = 10  # Base detection: 10 candles
    move_threshold = 60.0  # >= 60 pips
    reversal_windows = [3, 5, 7, 10]
    reversal_thresholds = [10.0, 20.0, 30.0, 40.0]

    years = sorted(set(pd.DatetimeIndex(df["time"]).year))

    print("=" * 60)
    print(f"=== Reversal Events (M5 EUR/USD) ===")
    print(f"(After a >={int(move_threshold)} pip move detected in {move_window} candles,")
    print(f" measure bounce from extreme)")
    print("=" * 60)
    print()

    rev_results = count_reversals(df, move_window, move_threshold,
                                  reversal_windows, reversal_thresholds)

    for rev_window in reversal_windows:
        print_reversal_table(rev_results[rev_window], rev_window, years, reversal_thresholds)


if __name__ == "__main__":
    main()
