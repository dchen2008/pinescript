#!/usr/bin/env python3
"""HMTS parameter optimizer.

Sweeps key HMTS parameters to find profitable combinations on M5 data.
Two-phase approach: coarse grid then fine-tune around best results.
"""

import sys
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_manager import load_candles
from src.strategy.hmts import HMTSStrategy
from src.backtest.engine import BacktestEngine
from src.backtest.portfolio import Portfolio


# Global: load data once per worker process
_df_cache = {}

def _get_data():
    pid = id(None)  # same per process
    if "df" not in _df_cache:
        _df_cache["df"] = load_candles("EUR_USD", "M5")
    return _df_cache["df"]


def run_single(params: dict) -> dict:
    """Run a single backtest with given params, return summary stats."""
    config_path = Path("config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tf_cfg = config["time_filter"]
    time_filter_params = {k: tf_cfg[k] for k in [
        "use_quiet_window", "quiet_start_hour", "quiet_start_min",
        "quiet_end_hour", "quiet_end_min", "use_market_window",
        "market_open_day", "market_open_hour", "market_close_day", "market_close_hour",
    ]}

    sizing_cfg = config["sizing"]
    risk_pct = params.get("risk_percent", sizing_cfg["risk_percent"])
    portfolio = Portfolio(
        initial_capital=sizing_cfg["initial_capital"],
        risk_percent=risk_pct,
        max_position_units=sizing_cfg["max_position_units"],
        spread_pips=config["spread_pips"],
    )

    atr_factor = params.get("atr_factor", config["ppst"]["atr_factor"])
    ppst_params = {
        "pivot_period": config["ppst"]["pivot_period"],
        "atr_factor": atr_factor,
        "atr_period": config["ppst"]["atr_period"],
    }

    strategy = HMTSStrategy(
        movement_candles=params["movement_candles"],
        movement_pips=params["movement_pips"],
        revert_candles=params["revert_candles"],
        revert_min_pips=params["revert_min_pips"],
        revert_max_pips=params["revert_max_pips"],
        hmts_tp_rr=params["hmts_tp_rr"],
        allow_second_entry=params.get("allow_second_entry", True),
        base_rr_ratio=params["base_rr_ratio"],
        trail_with_supertrend=params.get("trail_with_supertrend", False),
        s1_sl_spread_buffer=True,
        s2_revert_cross_limit=params["s2_revert_cross_limit"],
        spread_pips=config["spread_pips"],
    )

    engine = BacktestEngine(
        strategy=strategy, portfolio=portfolio,
        ppst_params=ppst_params, time_filter_params=time_filter_params,
        conservative_fills=config["backtest"]["conservative_fills"],
        use_circles=False, strategy_type="ppst",
    )

    df = _get_data()
    result = engine.run(df)

    trades = result["trade_log"]
    equity = result["equity_curve"]
    n_trades = len(trades)
    initial = 10000.0

    if n_trades == 0:
        return {**params, "trades": 0, "net_pct": 0, "pf": 0, "wr": 0,
                "max_dd_pct": 0, "final_equity": initial, "avg_weekly": 0}

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
    pf = gross_profit / gross_loss
    wr = len(wins) / n_trades * 100

    final = equity[-1]["equity"] if equity else initial
    net_pct = (final - initial) / initial * 100

    peak = initial
    max_dd = 0
    for entry in equity:
        eq = entry["equity"]
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Weekly returns
    eq_arr = np.array([e["equity"] for e in equity])
    block = 5 * 24 * 12
    weekly_returns = []
    for i in range(block, len(eq_arr), block):
        prev = eq_arr[i - block]
        if prev > 0:
            weekly_returns.append((eq_arr[i] - prev) / prev * 100)
    avg_weekly = np.mean(weekly_returns) if weekly_returns else 0

    return {
        **params,
        "trades": n_trades,
        "net_pct": round(net_pct, 2),
        "pf": round(pf, 3),
        "wr": round(wr, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "final_equity": round(final, 2),
        "avg_weekly": round(avg_weekly, 2),
    }


def run_grid(param_grid: dict, label: str = ""):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    param_dicts = [dict(zip(keys, c)) for c in combos]

    print(f"\n{'='*80}")
    print(f"  {label}: {total} combinations")
    print(f"{'='*80}")

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_single, p): p for p in param_dicts}
        for future in as_completed(futures):
            completed += 1
            try:
                res = future.result()
                results.append(res)
                if completed % 50 == 0 or completed == total:
                    best = max(results, key=lambda x: x["net_pct"])
                    print(f"  [{completed}/{total}] Best: net={best['net_pct']:.1f}% "
                          f"PF={best['pf']:.2f} WR={best['wr']:.1f}% #{best['trades']} "
                          f"DD={best['max_dd_pct']:.1f}%")
            except Exception as e:
                print(f"  ERROR: {e}")

    results.sort(key=lambda x: x["net_pct"], reverse=True)
    return results


def print_top(results, n=15):
    print(f"\n{'Net%':>8} {'PF':>6} {'WR%':>6} {'#Tr':>5} {'DD%':>6} {'Final$':>12} "
          f"{'AvgWk%':>7} | {'mvP':>4} {'mvC':>4} {'rvMn':>4} {'rvMx':>4} "
          f"{'rvC':>4} {'tpRR':>5} {'bRR':>4} {'xL':>3} {'rsk':>4} {'atr':>4}")
    print("-" * 110)
    for r in results[:n]:
        print(f"{r['net_pct']:>8.1f} {r['pf']:>6.2f} {r['wr']:>6.1f} {r['trades']:>5} "
              f"{r['max_dd_pct']:>6.1f} {r['final_equity']:>12.2f} {r['avg_weekly']:>7.2f} | "
              f"{r['movement_pips']:>4.0f} {r['movement_candles']:>4} {r['revert_min_pips']:>4.0f} "
              f"{r['revert_max_pips']:>4.0f} {r['revert_candles']:>4} {r['hmts_tp_rr']:>5.1f} "
              f"{r['base_rr_ratio']:>4.1f} {r['s2_revert_cross_limit']:>3.0f} "
              f"{r['risk_percent']:>4.1f} {r['atr_factor']:>4.1f}")


def main():
    # Phase 1: Corrected grid — no trail, no base TP, wider candle windows
    # Key finding: HMTS needs positions to survive until S2 (opposite signal)
    # With trailing SL, positions close before S2 fires -> HMTS never triggers
    # Wide candle windows (200-500) needed because signal spacing is 30-200+ bars
    coarse = {
        "movement_pips": [30, 50, 70],
        "movement_candles": [200, 500],
        "revert_min_pips": [5, 10],
        "revert_max_pips": [30, 50, 80],
        "revert_candles": [15, 30, 50],
        "hmts_tp_rr": [0, 5.0, 7.0, 10.0],
        "base_rr_ratio": [100.0],  # No TP on base trades
        "trail_with_supertrend": [False],  # No trail on base (keep positions alive)
        "s2_revert_cross_limit": [20],
        "risk_percent": [1.0, 3.0, 5.0],
        "atr_factor": [3.0, 4.0, 5.0],
    }
    r1 = run_grid(coarse, "Phase 1: Coarse Grid")
    print("\n--- PHASE 1 TOP 15 ---")
    print_top(r1, 15)

    # Phase 2: Fine-tune around top results
    if r1:
        best = r1[0]

        bmp = best["movement_pips"]
        bmc = best["movement_candles"]
        brmin = best["revert_min_pips"]
        brmax = best["revert_max_pips"]
        brc = best["revert_candles"]
        btprr = best["hmts_tp_rr"]
        brisk = best["risk_percent"]
        batr = best["atr_factor"]

        fine = {
            "movement_pips": sorted(set([max(20, bmp-10), bmp, bmp+10])),
            "movement_candles": sorted(set([max(100, bmc-100), bmc, min(800, bmc+100)])),
            "revert_min_pips": sorted(set([max(3, brmin-3), brmin, brmin+3])),
            "revert_max_pips": sorted(set([max(20, brmax-10), brmax, brmax+15, brmax+30])),
            "revert_candles": sorted(set([max(10, brc-5), brc, brc+10, brc+20])),
            "hmts_tp_rr": sorted(set([max(0, btprr-2), btprr, btprr+2, btprr+4])),
            "base_rr_ratio": [100.0],
            "trail_with_supertrend": [False],
            "s2_revert_cross_limit": [10, 20, 40, 80],
            "risk_percent": sorted(set([max(1.0, brisk-1), brisk, brisk+2, min(15.0, brisk+5)])),
            "atr_factor": sorted(set([max(2.0, batr-1), batr, batr+1])),
        }
        r2 = run_grid(fine, "Phase 2: Fine-Tune")
        print("\n--- PHASE 2 TOP 15 ---")
        print_top(r2, 15)

        # Combine and show overall best
        all_results = r1 + r2
        all_results.sort(key=lambda x: x["net_pct"], reverse=True)
        print("\n--- OVERALL TOP 20 ---")
        print_top(all_results, 20)

        # Show top by PF (edge quality, min 100 trades)
        pf_sorted = [r for r in all_results if r["trades"] >= 100]
        pf_sorted.sort(key=lambda x: x["pf"], reverse=True)
        print("\n--- TOP 15 BY PF (min 100 trades) ---")
        print_top(pf_sorted, 15)

        # Save
        df_out = pd.DataFrame(all_results)
        out_path = Path("results/hmts_optimizer.csv")
        out_path.parent.mkdir(exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
