#!/usr/bin/env python3
"""WCSE1 Stop Order optimizer — 2-stage grid search with multiprocessing.

Precomputes PPST indicators once, then sweeps WCSE/vol params fast.

Stage 1: vol_threshold × sma_period × stop_buffer × tp_rr × entry_times
Stage 2: Top N from stage 1 × c1_body × c2_body × wick/close groups

Usage:
    python3 -m scripts.optimize_wcse
    python3 -m scripts.optimize_wcse --workers 6
"""

import argparse
import csv
import itertools
import math
import multiprocessing
import os
import sys
import time
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd

from src.data.data_manager import load_candles
from src.indicators.ppst import compute_ppst
from src.indicators.volume_filter import compute_relative_volume
from src.backtest.portfolio import Portfolio
from src.strategy.position import Position
from src.strategy.wick_cross import (
    WickCrossState,
    detect_wcse1_setup,
    check_stop_fill,
)
from src.utils.forex_utils import pips_to_price


# ── Shared precomputed data (set in main, inherited by forked workers) ──
PPST_DATA = None        # DataFrame with PPST columns
REL_VOL = {}            # {sma_period: np.array} precomputed for each SMA period


def _fast_backtest(params: dict) -> dict:
    """Run the bar-by-bar backtest loop using precomputed indicators.

    This skips compute_ppst() and compute_relative_volume() — the expensive parts.
    """
    data = PPST_DATA
    n = len(data)

    vol_threshold = params["vol_threshold"]
    sma_period = params["sma_period"]
    rel_vol_arr = REL_VOL[sma_period]

    # Portfolio
    initial_capital = 10000.0
    portfolio = Portfolio(
        initial_capital=initial_capital,
        risk_percent=1.0,
        max_position_units=500000,
        spread_pips=1.5,
    )

    sl_buffer_price = pips_to_price(3.0)
    warmup = max(5, 11, sma_period + 1, 20)

    position = None
    trade_log = []
    deferred_dir = 0
    deferred_age = 0
    filtered_exit_countdown = 0
    filtered_exit_dir = 0

    stats = {
        "traded": 0,
        "wcse_entries": 0,
        "wcse_patterns": 0,
    }

    # WCSE state
    wcse = WickCrossState()
    wcse_entry_info = []
    wcse1_pending_stop = None

    # WCSE params
    entry_times = params["entry_times"]
    c1_body = params["c1_body"]
    c1_wick = params["c1_wick"]
    c1_close = params["c1_close"]
    c2_body = params["c2_body"]
    c2_wick = params["c2_wick"]
    c2_close = params["c2_close"]
    tp_rr = params["tp_rr"]
    stop_buffer_price = pips_to_price(params["stop_buffer"])
    spread_price = portfolio.spread_price

    # Precompute numpy arrays for fast access
    opens = data["open"].values
    highs = data["high"].values
    lows = data["low"].values
    closes = data["close"].values
    tups = data["tup"].values
    tdowns = data["tdown"].values
    trends = data["trend"].values.astype(int)
    buys = data["buy_signal"].values.astype(bool) if "buy_signal" in data.columns else np.zeros(n, dtype=bool)
    sells = data["sell_signal"].values.astype(bool) if "sell_signal" in data.columns else np.zeros(n, dtype=bool)

    for i in range(warmup, n):
        high = highs[i]
        low = lows[i]
        close = closes[i]
        open_price = opens[i]
        tup = tups[i]
        tdown = tdowns[i]
        trend = trends[i]

        buy_raw = buys[i]
        sell_raw = sells[i]

        rel_vol = rel_vol_arr[i]
        is_low_vol = (not np.isnan(rel_vol) and rel_vol < vol_threshold)

        # ── SL/TP fills ──
        if position is not None:
            sl_hit = position.check_sl_hit(high, low)
            tp_hit = position.check_tp_hit(high, low)

            sl_fill = position.sl_price
            tp_fill = position.tp_price

            if position.is_long:
                if sl_hit and open_price <= position.sl_price:
                    sl_fill = open_price
                if tp_hit and tp_fill is not None and open_price >= tp_fill:
                    tp_fill = open_price
            else:
                if sl_hit and open_price >= position.sl_price:
                    sl_fill = open_price
                if tp_hit and tp_fill is not None and open_price <= tp_fill:
                    tp_fill = open_price

            if sl_hit and tp_hit:
                pnl = position.close_position(sl_fill, "SL Hit", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None
            elif sl_hit:
                pnl = position.close_position(sl_fill, "SL Hit", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None
            elif tp_hit:
                if position.entry_source == "wcse":
                    wcse.record_tp_hit()
                pnl = position.close_position(tp_fill, "TP Hit", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None

        # ── Trail SL ──
        if position is not None:
            if position.is_long and not math.isnan(tup):
                position.trail_sl_with_supertrend(tup, sl_buffer_pips=3.0)
            elif position.is_short and not math.isnan(tdown):
                position.trail_sl_with_supertrend(tdown, sl_buffer_pips=3.0)

        # ── Filter logic ──
        is_vol_filtered = ((buy_raw or sell_raw) and is_low_vol)
        is_filtered = is_low_vol

        # ── Volume Recovery (simplified: disabled) ──
        # ── Filtered Exit (simplified: disabled) ──

        buy_signal = buy_raw and not is_filtered
        sell_signal = sell_raw and not is_filtered

        # ── WCSE1 pending stop fill ──
        wcse_entry = False
        if wcse1_pending_stop is not None and position is None:
            ps = wcse1_pending_stop
            filled, fill_price = check_stop_fill(
                (open_price, high, low, close),
                ps["stop_price"], ps["direction"],
            )
            if filled:
                if ps["direction"] == 1:
                    spread_entry = fill_price + spread_price
                else:
                    spread_entry = fill_price - spread_price
                sl_dist_actual = abs(spread_entry - ps["sl"])
                if sl_dist_actual > 0:
                    tp_actual = spread_entry + ps["direction"] * sl_dist_actual * tp_rr
                    units_actual = portfolio.compute_position_size(sl_dist_actual)
                    if units_actual > 0:
                        position = Position(
                            direction=ps["direction"],
                            entry_price=fill_price,
                            units=units_actual,
                            sl_price=ps["sl"],
                            tp_price=tp_actual,
                            entry_bar=i,
                            entry_time="",
                            entry_source="wcse",
                        )
                        wcse_entry = True
                        stats["wcse_entries"] += 1
                        stats["traded"] += 1
                        wcse_entry_info.append((i, fill_price, ps["sl"], ps["direction"], sl_dist_actual))

                        sb_sl_hit = position.check_sl_hit(high, low)
                        sb_tp_hit = position.check_tp_hit(high, low)
                        if sb_sl_hit and sb_tp_hit:
                            pnl = position.close_position(position.sl_price, "SL Hit", i)
                            portfolio.record_trade(pnl)
                            trade_log.append(position.to_dict())
                            position = None
                        elif sb_sl_hit:
                            pnl = position.close_position(position.sl_price, "SL Hit", i)
                            portfolio.record_trade(pnl)
                            trade_log.append(position.to_dict())
                            position = None
                        elif sb_tp_hit:
                            wcse.record_tp_hit()
                            pnl = position.close_position(position.tp_price, "TP Hit", i)
                            portfolio.record_trade(pnl)
                            trade_log.append(position.to_dict())
                            position = None
            wcse1_pending_stop = None

        # ── WCSE arm/disarm ──
        if buy_raw and is_vol_filtered and trend == 1:
            wcse.arm(1)
        elif sell_raw and is_vol_filtered and trend == -1:
            wcse.arm(-1)
        if (buy_raw and not is_filtered) or (sell_raw and not is_filtered):
            wcse.disarm()
        if wcse.armed_dir == 1 and trend != 1:
            wcse.disarm()
        if wcse.armed_dir == -1 and trend != -1:
            wcse.disarm()

        # ── WCSE1 setup detection ──
        if not wcse_entry and i >= warmup + 1 and wcse.is_active(position is None):
            if not wcse.can_enter(entry_times):
                pass
            else:
                wcse_dir = wcse.armed_dir
                if wcse_dir == 1:
                    c1_st = tups[i - 1]
                    c2_st = tup
                else:
                    c1_st = tdowns[i - 1]
                    c2_st = tdown

                valid, breakout_level = detect_wcse1_setup(
                    (opens[i-1], highs[i-1], lows[i-1], closes[i-1]),
                    (open_price, high, low, close),
                    c1_st, c2_st, wcse_dir,
                    c1_body_pips=c1_body, c1_wick_pips=c1_wick, c1_close_pips=c1_close,
                    c2_body_pips=c2_body, c2_wick_pips=c2_wick, c2_close_pips=c2_close,
                )

                if valid:
                    stats["wcse_patterns"] += 1
                    stop_price = breakout_level + wcse_dir * stop_buffer_price

                    if wcse_dir == 1:
                        sl = tup
                        se = stop_price + spread_price
                        sl_dist = se - sl
                    else:
                        sl = tdown
                        se = stop_price - spread_price
                        sl_dist = sl - se

                    if sl_dist > 0:
                        tp = se + wcse_dir * sl_dist * tp_rr
                        units = portfolio.compute_position_size(sl_dist)
                        if units > 0:
                            wcse1_pending_stop = {
                                "direction": wcse_dir,
                                "stop_price": stop_price,
                                "sl": sl,
                                "tp": tp,
                                "units": units,
                                "placed_bar": i,
                            }

        # ── Execute buy signal ──
        if not wcse_entry and buy_signal and (position is None or (position is not None and position.is_short)):
            stats["traded"] += 1
            if position is not None and position.is_short:
                pnl = position.close_position(close, "Reversal", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None

            sl_base = tup if not math.isnan(tup) else np.nan
            if not math.isnan(sl_base):
                spread_entry = close + spread_price
                sl = sl_base - sl_buffer_price
                sl_dist = spread_entry - sl
                if sl_dist > 0:
                    tp = spread_entry + sl_dist * 2.0  # tp_ratio=2.0 (fixed)
                    units = portfolio.compute_position_size(sl_dist)
                    if units > 0:
                        position = Position(
                            direction=1, entry_price=close, units=units,
                            sl_price=sl, tp_price=tp,
                            entry_bar=i, entry_time="",
                        )

        # ── Execute sell signal ──
        elif not wcse_entry and sell_signal and (position is None or (position is not None and position.is_long)):
            stats["traded"] += 1
            if position is not None and position.is_long:
                pnl = position.close_position(close, "Reversal", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None

            sl_base = tdown if not math.isnan(tdown) else np.nan
            if not math.isnan(sl_base):
                spread_entry = close - spread_price
                sl = sl_base + sl_buffer_price
                sl_dist = sl - spread_entry
                if sl_dist > 0:
                    tp = spread_entry - sl_dist * 2.0
                    units = portfolio.compute_position_size(sl_dist)
                    if units > 0:
                        position = Position(
                            direction=-1, entry_price=close, units=units,
                            sl_price=sl, tp_price=tp,
                            entry_bar=i, entry_time="",
                        )

        # ── Record equity ──
        unrealized = 0.0
        if position is not None:
            if position.is_long:
                unrealized = (close - position.entry_price) * position.units
            else:
                unrealized = (position.entry_price - close) * position.units
        portfolio.record_equity(i, unrealized)

    # Close remaining position
    if position is not None:
        last_close = closes[-1]
        pnl = position.close_position(last_close, "End of Data", n - 1)
        portfolio.record_trade(pnl)
        trade_log.append(position.to_dict())

    return trade_log, portfolio.equity_curve, stats, wcse_entry_info


def run_single(params: dict) -> dict:
    """Run one backtest and return params + results."""
    try:
        trade_log, equity_curve, stats, _ = _fast_backtest(params)
    except Exception as e:
        return {**params, "error": str(e)}

    initial = 10000.0
    n_trades = len(trade_log)
    if n_trades == 0:
        return {
            **params,
            "trades": 0, "wcse_entries": 0, "wr": 0, "pf": 0,
            "net_pct": 0, "max_dd_pct": 0, "sharpe": 0,
        }

    wins = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
    wr = len(wins) / n_trades * 100
    pf = gross_win / gross_loss

    final_equity = equity_curve[-1]["equity"] if equity_curve else initial
    net_pct = (final_equity - initial) / initial * 100

    peak = initial
    max_dd = 0
    for ec in equity_curve:
        eq = ec["equity"]
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd * 100

    # Sharpe (weekly — sample every ~1400 bars for M5)
    if len(equity_curve) > 1:
        eq_series = pd.Series([e["equity"] for e in equity_curve])
        weekly_eq = eq_series.iloc[::1400]
        if len(weekly_eq) > 2:
            weekly_ret = weekly_eq.pct_change().dropna()
            sharpe = (weekly_ret.mean() / weekly_ret.std() * (52 ** 0.5)) if weekly_ret.std() > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    return {
        **params,
        "trades": n_trades,
        "wcse_entries": stats.get("wcse_entries", 0),
        "wr": round(wr, 2),
        "pf": round(pf, 3),
        "net_pct": round(net_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "sharpe": round(sharpe, 2),
    }


def make_stage1_grid():
    """Stage 1: outer params (vol + WCSE entry config)."""
    vol_thresholds = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    sma_periods = [20, 25, 30, 35, 40]
    stop_buffers = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    tp_rrs = [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0]
    entry_times_list = [1, 2, 3]

    c1_body, c2_body = 1.0, 0.5
    c1_wick, c1_close = 5.0, 5.0
    c2_wick, c2_close = 5.0, 5.0

    grid = []
    for vt, sma, sb, rr, et in itertools.product(
        vol_thresholds, sma_periods, stop_buffers, tp_rrs, entry_times_list
    ):
        grid.append({
            "vol_threshold": vt, "sma_period": sma,
            "stop_buffer": sb, "tp_rr": rr, "entry_times": et,
            "c1_body": c1_body, "c1_wick": c1_wick, "c1_close": c1_close,
            "c2_body": c2_body, "c2_wick": c2_wick, "c2_close": c2_close,
        })
    return grid


def make_stage2_grid(top_configs: list):
    """Stage 2: sweep c1/c2 pattern params for top N configs."""
    c1_bodies = [0.5, 0.8, 1.0, 1.3]
    c2_bodies = [0.3, 0.5, 0.8, 1.0]
    wc_groups = [
        (3.0, 3.0),
        (4.0, 4.0),
        (5.0, 5.0),
        (5.0, 7.0),
        (7.0, 7.0),
    ]

    grid = []
    for cfg in top_configs:
        for c1b, c2b in itertools.product(c1_bodies, c2_bodies):
            for (c1w, c1c), (c2w, c2c) in itertools.product(wc_groups, wc_groups):
                grid.append({
                    "vol_threshold": cfg["vol_threshold"],
                    "sma_period": cfg["sma_period"],
                    "stop_buffer": cfg["stop_buffer"],
                    "tp_rr": cfg["tp_rr"],
                    "entry_times": cfg["entry_times"],
                    "c1_body": c1b, "c1_wick": c1w, "c1_close": c1c,
                    "c2_body": c2b, "c2_wick": c2w, "c2_close": c2c,
                })
    return grid


def run_stage(grid, workers, label):
    """Run a grid of backtests in parallel."""
    total = len(grid)
    print(f"\n{'='*60}")
    print(f"  {label}: {total} combinations, {workers} workers")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    results = []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single, grid, chunksize=8)):
            results.append(res)
            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{total}] {rate:.1f} runs/sec, ETA {eta:.0f}s")
                sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s ({total/elapsed:.1f} runs/sec)")

    results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)
    return results


def save_results(results, filepath):
    """Save results to CSV."""
    if not results:
        return
    keys = results[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved {len(results)} results to {filepath}")


def print_top(results, n=15):
    """Print top N results."""
    print(f"\n  Top {n} by Sharpe:")
    print(f"  {'VT':>4} {'SMA':>4} {'Buf':>4} {'RR':>4} {'ET':>3} "
          f"{'C1B':>4} {'C2B':>4} {'C1W':>4} {'C2W':>4} "
          f"{'Trd':>4} {'WCSE':>4} {'WR%':>5} {'PF':>5} {'Ret%':>6} {'DD%':>5} {'Shp':>5}")
    print(f"  {'-'*95}")
    for r in results[:n]:
        if "error" in r:
            continue
        print(f"  {r['vol_threshold']:>4.1f} {r['sma_period']:>4} {r['stop_buffer']:>4.1f} "
              f"{r['tp_rr']:>4.1f} {r['entry_times']:>3} "
              f"{r['c1_body']:>4.1f} {r['c2_body']:>4.1f} "
              f"{r['c1_wick']:>4.1f} {r['c2_wick']:>4.1f} "
              f"{r['trades']:>4} {r['wcse_entries']:>4} "
              f"{r['wr']:>5.1f} {r['pf']:>5.3f} "
              f"{r['net_pct']:>6.1f} {r['max_dd_pct']:>5.1f} {r['sharpe']:>5.2f}")


def main():
    multiprocessing.set_start_method("fork")
    parser = argparse.ArgumentParser(description="WCSE1 Optimizer")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top configs to carry into stage 2")
    parser.add_argument("--stage1-only", action="store_true",
                        help="Skip stage 2")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # ── Precompute indicators once ──
    global PPST_DATA, REL_VOL
    print(f"Loading {args.instrument} {args.granularity} data...")
    df = load_candles(args.instrument, args.granularity)
    print(f"Loaded {len(df)} candles")

    print("Precomputing PPST indicators...")
    PPST_DATA = compute_ppst(df, pivot_period=2, atr_factor=5.0, atr_period=10)

    print("Precomputing volume SMAs...")
    for sma in [20, 25, 30, 35, 40]:
        REL_VOL[sma] = compute_relative_volume(df["volume"].values, sma)
    print("Precomputation done.\n")

    # ── Stage 1 ──
    grid1 = make_stage1_grid()
    results1 = run_stage(grid1, args.workers,
                         "Stage 1: Vol × WCSE outer params")
    save_results(results1, "results/wcse_opt_stage1.csv")
    print_top(results1, 20)

    if args.stage1_only:
        return

    # ── Stage 2 ──
    seen = set()
    top_configs = []
    for r in results1:
        if "error" in r:
            continue
        key = (r["vol_threshold"], r["sma_period"], r["stop_buffer"], r["tp_rr"], r["entry_times"])
        if key not in seen:
            seen.add(key)
            top_configs.append(r)
        if len(top_configs) >= args.top_n:
            break

    print(f"\n  Carrying {len(top_configs)} configs into Stage 2")

    grid2 = make_stage2_grid(top_configs)
    results2 = run_stage(grid2, args.workers,
                         "Stage 2: c1/c2 pattern params")
    save_results(results2, "results/wcse_opt_stage2.csv")
    print_top(results2, 20)

    # ── Combined ──
    all_results = results1 + results2
    all_results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)
    save_results(all_results, "results/wcse_opt_all.csv")
    print(f"\n  Overall top 20:")
    print_top(all_results, 20)


if __name__ == "__main__":
    main()
