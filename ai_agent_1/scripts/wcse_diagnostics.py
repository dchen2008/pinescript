#!/usr/bin/env python3
"""WCSE diagnostics: understand why so few patterns match."""
import sys
sys.argv = ['vol']
import math
import numpy as np
from src.data.data_manager import load_candles
from src.indicators.ppst import compute_ppst
from src.indicators.volume_filter import compute_relative_volume
from src.vol import compute_time_window
from src.utils.forex_utils import pips_to_price

df = load_candles('EUR_USD', 'M5')
print(f'Loaded {len(df)} candles\n')

# Compute indicators
ppst_params = {'pivot_period': 2, 'atr_factor': 5.0, 'atr_period': 10}
data = compute_ppst(df, **ppst_params)
data["rel_volume"] = compute_relative_volume(df["volume"].values, 30)

is_quiet_arr = compute_time_window(
    data["time"], "America/Los_Angeles", 12, 0, 14, 30, True)

# WCSE params (loose thresholds from tests)
C1_BODY = pips_to_price(1.0)
C1_WICK = pips_to_price(2.0)
C1_CLOSE = pips_to_price(3.0)
C2_BODY = pips_to_price(0.5)
C2_WICK = pips_to_price(2.0)
C2_CLOSE = pips_to_price(3.0)

warmup = 22
n = len(data)

# Simulate arming
armed_dir = 0
armed_bars = 0  # total bars while armed & no position (simplified: assume no position)

# Rejection counters
rej = {
    'c1_color': 0,
    'c1_body': 0,
    'c1_proximity': 0,
    'c2_color': 0,
    'c2_body': 0,
    'c2_proximity': 0,
    'c2_crosses': 0,
    'c3_color': 0,
    'pass': 0,
}

# Also track body size distribution when armed
c1_bodies = []
c2_bodies = []
c1_distances = []  # distance from ST
c2_distances = []

for i in range(warmup, n):
    row = data.iloc[i]
    trend = int(row.get("trend", 0))
    buy_raw = bool(row.get("buy_signal", False))
    sell_raw = bool(row.get("sell_signal", False))
    rel_vol = row["rel_volume"]
    is_quiet = bool(is_quiet_arr[i])
    is_low_vol = not np.isnan(rel_vol) and rel_vol < 1.0

    is_vol_filtered = (buy_raw or sell_raw) and is_low_vol and not is_quiet
    is_filtered = is_quiet or is_low_vol

    # Arm logic
    if buy_raw and is_vol_filtered and trend == 1:
        armed_dir = 1
    elif sell_raw and is_vol_filtered and trend == -1:
        armed_dir = -1
    if (buy_raw and not is_filtered) or (sell_raw and not is_filtered):
        armed_dir = 0
    if armed_dir == 1 and trend != 1:
        armed_dir = 0
    if armed_dir == -1 and trend != -1:
        armed_dir = 0

    if armed_dir == 0 or i < warmup + 2:
        continue

    armed_bars += 1

    # Check pattern
    r2 = data.iloc[i - 2]
    r1 = data.iloc[i - 1]
    o2, h2, l2, c2 = r2["open"], r2["high"], r2["low"], r2["close"]
    o1, h1, l1, c1 = r1["open"], r1["high"], r1["low"], r1["close"]
    oi, ci = row["open"], row["close"]

    if armed_dir == 1:
        tup2, tup1 = r2["tup"], r1["tup"]
        if math.isnan(tup2) or math.isnan(tup1):
            continue

        # C1: must be RED
        if c2 >= o2:
            rej['c1_color'] += 1
            continue
        body1 = o2 - c2
        c1_bodies.append(body1 / 0.0001)  # in pips
        if body1 < C1_BODY:
            rej['c1_body'] += 1
            continue
        # C1: proximity to TUp
        dist1 = l2 - tup2  # distance of low from TUp (positive = above)
        close_dist1 = abs(c2 - tup2)
        c1_distances.append(dist1 / 0.0001)
        touch_b = (l2 >= tup2) and ((l2 - tup2) <= C1_WICK or close_dist1 <= C1_CLOSE)
        cross_c = (l2 < tup2) and (c2 > tup2)
        if not (touch_b or cross_c):
            rej['c1_proximity'] += 1
            continue

        # C2: must be GREEN
        if c1 <= o1:
            rej['c2_color'] += 1
            continue
        body2 = c1 - o1
        c2_bodies.append(body2 / 0.0001)
        if body2 < C2_BODY:
            rej['c2_body'] += 1
            continue
        # C2: proximity
        dist2 = l1 - tup1
        c2_distances.append(dist2 / 0.0001)
        touch_2 = (l1 >= tup1) and ((l1 - tup1) <= C2_WICK or abs(c1 - tup1) <= C2_CLOSE)
        if not touch_2:
            rej['c2_proximity'] += 1
            continue
        if l1 < tup1:
            rej['c2_crosses'] += 1
            continue

        # C3: must be GREEN
        if ci <= oi:
            rej['c3_color'] += 1
            continue
        rej['pass'] += 1

    elif armed_dir == -1:
        td2, td1 = r2["tdown"], r1["tdown"]
        if math.isnan(td2) or math.isnan(td1):
            continue

        # C1: must be GREEN
        if c2 <= o2:
            rej['c1_color'] += 1
            continue
        body1 = c2 - o2
        c1_bodies.append(body1 / 0.0001)
        if body1 < C1_BODY:
            rej['c1_body'] += 1
            continue
        dist1 = td2 - h2
        close_dist1 = abs(c2 - td2)
        c1_distances.append(dist1 / 0.0001)
        touch_b = (h2 <= td2) and ((td2 - h2) <= C1_WICK or close_dist1 <= C1_CLOSE)
        cross_c = (h2 > td2) and (c2 < td2)
        if not (touch_b or cross_c):
            rej['c1_proximity'] += 1
            continue
        # C2: must be RED
        if c1 >= o1:
            rej['c2_color'] += 1
            continue
        body2 = o1 - c1
        c2_bodies.append(body2 / 0.0001)
        if body2 < C2_BODY:
            rej['c2_body'] += 1
            continue
        dist2 = td1 - h1
        c2_distances.append(dist2 / 0.0001)
        touch_2 = (h1 <= td1) and ((td1 - h1) <= C2_WICK or abs(c1 - td1) <= C2_CLOSE)
        if not touch_2:
            rej['c2_proximity'] += 1
            continue
        if h1 > td1:
            rej['c2_crosses'] += 1
            continue
        # C3: must be RED
        if ci >= oi:
            rej['c3_color'] += 1
            continue
        rej['pass'] += 1

# Print results
print("=" * 55)
print("  WCSE Pattern Rejection Funnel")
print("=" * 55)
print(f"  Total armed bars checked:  {armed_bars:,}")
print()
total_checked = armed_bars
for stage, label in [
    ('c1_color', 'C1 wrong color (need RED/GREEN)'),
    ('c1_body', 'C1 body too small (< 1 pip)'),
    ('c1_proximity', 'C1 not near SuperTrend'),
    ('c2_color', 'C2 wrong color'),
    ('c2_body', 'C2 body too small (< 0.5 pip)'),
    ('c2_proximity', 'C2 not near SuperTrend'),
    ('c2_crosses', 'C2 crosses SuperTrend'),
    ('c3_color', 'C3 wrong color'),
    ('pass', 'PATTERN MATCH'),
]:
    count = rej[stage]
    pct_of_total = count / armed_bars * 100 if armed_bars > 0 else 0
    remaining = armed_bars - sum(rej[k] for k in list(rej.keys())[:list(rej.keys()).index(stage)])
    pct_of_remaining = count / remaining * 100 if remaining > 0 else 0
    marker = ">>>" if stage == 'pass' else "   "
    print(f"  {marker} {label:<35} {count:>7,} ({pct_of_total:>5.1f}% of total, {pct_of_remaining:>5.1f}% of remaining)")

print()
print("=" * 55)
print("  Body Size Distribution (pips) when armed")
print("=" * 55)
if c1_bodies:
    import statistics
    arr = sorted(c1_bodies)
    print(f"  C1 bodies (all, n={len(arr)}):")
    print(f"    Min: {arr[0]:.2f}  P25: {arr[len(arr)//4]:.2f}  Median: {arr[len(arr)//2]:.2f}  P75: {arr[3*len(arr)//4]:.2f}  Max: {arr[-1]:.2f}")
    pct_above_1 = sum(1 for b in arr if b >= 1.0) / len(arr) * 100
    pct_above_05 = sum(1 for b in arr if b >= 0.5) / len(arr) * 100
    print(f"    >= 0.5 pip: {pct_above_05:.1f}%  |  >= 1.0 pip: {pct_above_1:.1f}%")

if c2_bodies:
    arr = sorted(c2_bodies)
    print(f"  C2 bodies (after C1 pass, n={len(arr)}):")
    print(f"    Min: {arr[0]:.2f}  P25: {arr[len(arr)//4]:.2f}  Median: {arr[len(arr)//2]:.2f}  P75: {arr[3*len(arr)//4]:.2f}  Max: {arr[-1]:.2f}")

print()
print("=" * 55)
print("  Distance from SuperTrend (pips) when armed")
print("=" * 55)
if c1_distances:
    arr = sorted(c1_distances)
    print(f"  C1 low/high distance to ST (n={len(arr)}):")
    print(f"    Min: {arr[0]:.2f}  P25: {arr[len(arr)//4]:.2f}  Median: {arr[len(arr)//2]:.2f}  P75: {arr[3*len(arr)//4]:.2f}  Max: {arr[-1]:.2f}")
    pct_within_2 = sum(1 for d in arr if abs(d) <= 2.0) / len(arr) * 100
    pct_within_5 = sum(1 for d in arr if abs(d) <= 5.0) / len(arr) * 100
    pct_within_10 = sum(1 for d in arr if abs(d) <= 10.0) / len(arr) * 100
    print(f"    Within 2 pips: {pct_within_2:.1f}%  |  5 pips: {pct_within_5:.1f}%  |  10 pips: {pct_within_10:.1f}%")

if c2_distances:
    arr = sorted(c2_distances)
    print(f"  C2 low/high distance to ST (n={len(arr)}):")
    print(f"    Min: {arr[0]:.2f}  P25: {arr[len(arr)//4]:.2f}  Median: {arr[len(arr)//2]:.2f}  P75: {arr[3*len(arr)//4]:.2f}  Max: {arr[-1]:.2f}")
