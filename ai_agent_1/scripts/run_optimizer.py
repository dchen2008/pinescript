#!/usr/bin/env python3
"""Run parameter grid sweep optimization.

Usage:
    python scripts/run_optimizer.py
    python scripts/run_optimizer.py --strategy signal --granularity M1
"""

import argparse
from pathlib import Path

import yaml

from src.data.data_manager import load_candles
from src.backtest.optimizer import run_optimization


def main():
    parser = argparse.ArgumentParser(description="Run PPST parameter optimization")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--strategy", choices=["signal", "circle", "vwap"], default="signal")
    parser.add_argument("--instrument", default=None, help="Override instrument")
    parser.add_argument("--granularity", default=None, help="Override timeframe")
    parser.add_argument("--workers", type=int, default=None, help="Override max workers")
    parser.add_argument("--risk", type=float, default=None, help="Override risk percent")
    parser.add_argument("--entry-modes", nargs="+", default=None,
                        help="Circle entry modes to sweep")
    parser.add_argument("--window-mins", nargs="+", type=int, default=None,
                        help="Entry window minutes for 'Circle within Time Window'")
    parser.add_argument("--circle-nums", nargs="+", type=int, default=None,
                        help="Circle numbers for 'Nth Circle' mode")
    parser.add_argument("--volume-thresholds", nargs="+", type=float, default=None,
                        help="Volume thresholds to sweep (e.g. 0.0 1.25 1.5 2.0)")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    instrument = args.instrument or config["instrument"]
    granularity = args.granularity or config["timeframes"][0]

    opt_cfg = config["optimizer"]
    tf_cfg = config["time_filter"]
    sizing_cfg = config["sizing"]

    time_filter_params = {
        "use_quiet_window": tf_cfg["use_quiet_window"],
        "quiet_start_hour": tf_cfg["quiet_start_hour"],
        "quiet_start_min": tf_cfg["quiet_start_min"],
        "quiet_end_hour": tf_cfg["quiet_end_hour"],
        "quiet_end_min": tf_cfg["quiet_end_min"],
        "use_market_window": tf_cfg["use_market_window"],
        "market_open_day": tf_cfg["market_open_day"],
        "market_open_hour": tf_cfg["market_open_hour"],
        "market_close_day": tf_cfg["market_close_day"],
        "market_close_hour": tf_cfg["market_close_hour"],
    }

    print(f"Loading {instrument} {granularity} data...")
    df = load_candles(instrument, granularity)
    print(f"Loaded {len(df)} candles")

    risk_percent = args.risk if args.risk is not None else sizing_cfg["risk_percent"]

    # Build VWAP grid from config if applicable
    vwap_grid = None
    if args.strategy == "vwap":
        vwap_opt = config.get("vwap_optimizer", {})
        vwap_grid = {
            "session_starts": vwap_opt.get("session_starts", [8, 13]),
            "vwap_band_mults": vwap_opt.get("vwap_band_mults", [1.0, 1.5, 2.0]),
            "rsi_periods": vwap_opt.get("rsi_periods", [7, 14]),
            "rsi_thresholds": [tuple(t) for t in vwap_opt.get("rsi_thresholds", [[30, 70], [25, 75]])],
            "sl_pips": vwap_opt.get("sl_pips", [10, 15, 20]),
            "tp_types": vwap_opt.get("tp_types", ["vwap", "rr"]),
            "tp_rrs": vwap_opt.get("tp_rrs", [1.0, 1.5, 2.0]),
        }

    results = run_optimization(
        df=df,
        strategy_type=args.strategy,
        atr_factors=opt_cfg["atr_factors"],
        rr_ratios=opt_cfg["rr_ratios"],
        tp_types=opt_cfg["tp_types"],
        entry_modes=args.entry_modes,
        entry_window_mins=args.window_mins,
        entry_circle_nums=args.circle_nums,
        time_filter_params=time_filter_params,
        initial_capital=sizing_cfg["initial_capital"],
        risk_percent=risk_percent,
        spread_pips=config["spread_pips"],
        max_workers=args.workers or opt_cfg["max_workers"],
        vwap_grid=vwap_grid,
        volume_thresholds=args.volume_thresholds,
    )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f"optimization_{args.strategy}_{granularity}.csv"
    results.to_csv(output_file, index=False)

    print(f"\nOptimization Results (Top 10):")
    print(results.head(10).to_string(index=False))
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
