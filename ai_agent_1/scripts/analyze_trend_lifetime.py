#!/usr/bin/env python3
"""Analyze SuperTrend trend segment lifetimes and circle timing.

Computes how long each M1 SuperTrend trend lives before reversing,
and when circles occur relative to the trend start.

Usage:
    python -m scripts.analyze_trend_lifetime
    python -m scripts.analyze_trend_lifetime --atr-factor 5.0
    python -m scripts.analyze_trend_lifetime --granularity M5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_manager import load_candles
from src.indicators.ppst import compute_ppst
from src.indicators.entry_circles import compute_entry_circles


def analyze_trend_lifetimes(
    df: pd.DataFrame,
    atr_factor: float = 3.0,
    timeframe_minutes: int = 1,
) -> dict:
    """Analyze trend segment lifetimes and circle timing.

    Args:
        df: Candle data with time, open, high, low, close
        atr_factor: ATR multiplier for PPST bands
        timeframe_minutes: Minutes per bar (1 for M1, 5 for M5)

    Returns:
        Dict with 'segments' DataFrame and 'circles' DataFrame
    """
    # Compute PPST + circles
    data = compute_ppst(df, atr_factor=atr_factor)
    circles = compute_entry_circles(data)
    for col in circles.columns:
        data[col] = circles[col].values

    trend = data["trend"].values
    buy_signal = data["buy_signal"].values
    sell_signal = data["sell_signal"].values
    long_circle = data["long_circle"].values
    short_circle = data["short_circle"].values
    times = data["time"].values

    n = len(data)

    # Find trend segments: each starts at a signal bar, ends at next opposite signal
    segments = []
    circle_records = []

    seg_start = None
    seg_direction = None
    seg_circles = []

    for i in range(n):
        # New signal starts a new segment
        if buy_signal[i] or sell_signal[i]:
            # Close previous segment
            if seg_start is not None:
                duration_bars = i - seg_start
                duration_min = duration_bars * timeframe_minutes
                segments.append({
                    "start_bar": seg_start,
                    "end_bar": i,
                    "start_time": str(times[seg_start]),
                    "end_time": str(times[i]),
                    "direction": seg_direction,
                    "duration_bars": duration_bars,
                    "duration_min": duration_min,
                    "circle_count": len(seg_circles),
                })
                # Record circle timing within segment
                for c in seg_circles:
                    circle_records.append({
                        "segment_start_bar": seg_start,
                        "circle_bar": c["bar"],
                        "bars_after_signal": c["bar"] - seg_start,
                        "min_after_signal": (c["bar"] - seg_start) * timeframe_minutes,
                        "circle_num": c["num"],
                        "direction": seg_direction,
                        "segment_duration_min": duration_min,
                    })

            # Start new segment
            seg_start = i
            seg_direction = "long" if buy_signal[i] else "short"
            seg_circles = []

        # Track circles in current segment
        if seg_start is not None:
            is_circle = (long_circle[i] and seg_direction == "long") or \
                        (short_circle[i] and seg_direction == "short")
            if is_circle:
                seg_circles.append({
                    "bar": i,
                    "num": len(seg_circles) + 1,
                })

    # Close final segment
    if seg_start is not None:
        duration_bars = n - 1 - seg_start
        duration_min = duration_bars * timeframe_minutes
        segments.append({
            "start_bar": seg_start,
            "end_bar": n - 1,
            "start_time": str(times[seg_start]),
            "end_time": str(times[n - 1]),
            "direction": seg_direction,
            "duration_bars": duration_bars,
            "duration_min": duration_min,
            "circle_count": len(seg_circles),
        })
        for c in seg_circles:
            circle_records.append({
                "segment_start_bar": seg_start,
                "circle_bar": c["bar"],
                "bars_after_signal": c["bar"] - seg_start,
                "min_after_signal": (c["bar"] - seg_start) * timeframe_minutes,
                "circle_num": c["num"],
                "direction": seg_direction,
                "segment_duration_min": duration_min,
            })

    segments_df = pd.DataFrame(segments)
    circles_df = pd.DataFrame(circle_records)

    return {"segments": segments_df, "circles": circles_df}


def print_statistics(segments_df: pd.DataFrame, circles_df: pd.DataFrame, label: str):
    """Print comprehensive statistics."""
    print(f"\n{'='*70}")
    print(f"  TREND LIFETIME ANALYSIS — {label}")
    print(f"{'='*70}")

    durations = segments_df["duration_min"]

    print(f"\n--- Trend Segment Duration Statistics ---")
    print(f"  Total segments:       {len(segments_df)}")
    print(f"  Mean duration:        {durations.mean():.1f} min")
    print(f"  Median duration:      {durations.median():.1f} min")
    print(f"  Std deviation:        {durations.std():.1f} min")
    print(f"  25th percentile:      {durations.quantile(0.25):.1f} min")
    print(f"  50th percentile:      {durations.quantile(0.50):.1f} min")
    print(f"  75th percentile:      {durations.quantile(0.75):.1f} min")
    print(f"  90th percentile:      {durations.quantile(0.90):.1f} min")
    print(f"  95th percentile:      {durations.quantile(0.95):.1f} min")
    print(f"  Max duration:         {durations.max():.0f} min")

    # Histogram buckets
    buckets = [
        ("1-10 min", 1, 10),
        ("10-30 min", 10, 30),
        ("30-60 min", 30, 60),
        ("60-120 min", 60, 120),
        ("120-240 min", 120, 240),
        ("240+ min", 240, float("inf")),
    ]
    print(f"\n--- Duration Distribution ---")
    total = len(durations)
    for name, lo, hi in buckets:
        count = ((durations >= lo) & (durations < hi)).sum()
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {name:>14s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Key hypothesis check
    die_61 = (durations <= 61).sum()
    die_30 = (durations <= 30).sum()
    print(f"\n--- Hypothesis Check ---")
    print(f"  Trends dying within 30 min: {die_30:5d} ({die_30/total*100:.1f}%)")
    print(f"  Trends dying within 61 min: {die_61:5d} ({die_61/total*100:.1f}%)")

    # Circle statistics
    if len(circles_df) > 0:
        print(f"\n--- Circle Timing Within Trend ---")
        print(f"  Total circles:        {len(circles_df)}")
        print(f"  Segments with 0 circles: {(segments_df['circle_count'] == 0).sum()}")
        print(f"  Segments with 1+ circle: {(segments_df['circle_count'] >= 1).sum()}")
        print(f"  Segments with 2+ circles: {(segments_df['circle_count'] >= 2).sum()}")
        print(f"  Segments with 3+ circles: {(segments_df['circle_count'] >= 3).sum()}")

        mins = circles_df["min_after_signal"]
        print(f"\n  Circle timing (minutes after signal):")
        print(f"    Mean:    {mins.mean():.1f} min")
        print(f"    Median:  {mins.median():.1f} min")
        print(f"    25th:    {mins.quantile(0.25):.1f} min")
        print(f"    75th:    {mins.quantile(0.75):.1f} min")

        # Same-bar circles
        same_bar = (circles_df["bars_after_signal"] == 0).sum()
        within_10 = (circles_df["min_after_signal"] <= 10).sum()
        within_30 = (circles_df["min_after_signal"] <= 30).sum()
        within_60 = (circles_df["min_after_signal"] <= 60).sum()
        total_c = len(circles_df)
        print(f"\n  Circle timing buckets:")
        print(f"    Same bar (0 min):   {same_bar:5d} ({same_bar/total_c*100:.1f}%)")
        print(f"    Within 10 min:      {within_10:5d} ({within_10/total_c*100:.1f}%)")
        print(f"    Within 30 min:      {within_30:5d} ({within_30/total_c*100:.1f}%)")
        print(f"    Within 60 min:      {within_60:5d} ({within_60/total_c*100:.1f}%)")

        # First circle timing
        first_circles = circles_df[circles_df["circle_num"] == 1]
        if len(first_circles) > 0:
            fc_mins = first_circles["min_after_signal"]
            print(f"\n  First circle timing:")
            print(f"    Mean:    {fc_mins.mean():.1f} min")
            print(f"    Median:  {fc_mins.median():.1f} min")
            print(f"    Same bar: {(first_circles['bars_after_signal'] == 0).sum()} / {len(first_circles)}")

    # Direction breakdown
    for direction in ["long", "short"]:
        subset = segments_df[segments_df["direction"] == direction]
        if len(subset) > 0:
            d = subset["duration_min"]
            print(f"\n--- {direction.upper()} Trends ---")
            print(f"  Count: {len(subset)}, Mean: {d.mean():.1f} min, Median: {d.median():.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Analyze SuperTrend trend lifetimes")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", default="M1")
    parser.add_argument("--atr-factor", type=float, default=None,
                        help="Single ATR factor (default: sweep 3.0-5.0)")
    args = parser.parse_args()

    timeframe_minutes = 1 if args.granularity == "M1" else 5

    print(f"Loading {args.instrument} {args.granularity} data...")
    df = load_candles(args.instrument, args.granularity)
    print(f"Loaded {len(df)} candles")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.atr_factor is not None:
        atr_factors = [args.atr_factor]
    else:
        atr_factors = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0]

    all_segments = []
    all_circles = []

    for atr_f in atr_factors:
        print(f"\nComputing PPST with ATR factor = {atr_f}...")
        result = analyze_trend_lifetimes(df, atr_factor=atr_f, timeframe_minutes=timeframe_minutes)
        segments_df = result["segments"]
        circles_df = result["circles"]

        segments_df["atr_factor"] = atr_f
        circles_df["atr_factor"] = atr_f

        all_segments.append(segments_df)
        all_circles.append(circles_df)

        label = f"ATR={atr_f}, {args.granularity}"
        print_statistics(segments_df, circles_df, label)

    # Save combined results
    combined_segments = pd.concat(all_segments, ignore_index=True)
    combined_circles = pd.concat(all_circles, ignore_index=True)

    seg_file = results_dir / f"trend_lifetime_analysis_{args.granularity}.csv"
    circle_file = results_dir / f"circle_timing_analysis_{args.granularity}.csv"
    combined_segments.to_csv(seg_file, index=False)
    combined_circles.to_csv(circle_file, index=False)

    print(f"\n{'='*70}")
    print(f"Results saved to:")
    print(f"  {seg_file}")
    print(f"  {circle_file}")


if __name__ == "__main__":
    main()
