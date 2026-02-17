#!/usr/bin/env python3
"""PPST Volume + Time Filter Backtest.

Port of ppst_vol_tf_filter_bt.pine.

Extends vol.py with Entry Time Window (whitelist).
Entries only during the specified time window; exits fire at any time.

Usage:
    cd ai_agent_1
    python3 -m src.vol_tf
    python3 -m src.vol_tf --granularity M5 --etw-start 16:00 --etw-end 16:30
    python3 -m src.vol_tf --granularity M5 --etw-start 09:00 --etw-end 11:00 --etw-tz America/New_York
"""

import argparse

from src.data.data_manager import load_candles
from src.backtest.report import generate_report
from src.vol import (
    run_vol_backtest,
    build_common_args,
    build_params,
    parse_time,
    print_filter_stats,
    print_wcse_mfe,
)


def main():
    parser = argparse.ArgumentParser(description="PPST Volume + Time Filter Backtest")
    build_common_args(parser)

    # Entry time window args
    parser.add_argument("--no-etw", action="store_true", help="Disable entry time window")
    parser.add_argument("--etw-start", default="16:00", help="Entry window start HH:MM")
    parser.add_argument("--etw-end", default="16:30", help="Entry window end HH:MM")
    parser.add_argument("--etw-tz", default="America/Los_Angeles")

    args = parser.parse_args()

    ppst_params, vol_params, quiet_params, strategy_params, sizing_params, wcse_params = build_params(args)

    # Entry window params
    etw_start = parse_time(args.etw_start)
    etw_end = parse_time(args.etw_end)
    entry_window_params = {
        "enabled": not args.no_etw,
        "start_hour": etw_start[0], "start_min": etw_start[1],
        "end_hour": etw_end[0], "end_min": etw_end[1],
        "tz": args.etw_tz,
    }

    # Load data
    print(f"Loading {args.instrument} {args.granularity} data...")
    df = load_candles(args.instrument, args.granularity)
    print(f"Loaded {len(df)} candles: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Run backtest
    etw_label = f"{args.etw_start}-{args.etw_end}" if not args.no_etw else "off"
    print(f"Running vol+tf backtest (ETW: {etw_label})...")
    result = run_vol_backtest(
        df, ppst_params, vol_params, quiet_params, strategy_params, sizing_params,
        entry_window_params=entry_window_params,
        wcse_params=wcse_params,
    )

    # Generate report
    close_mode = "swing" if args.swing_close else f"tp{args.tp_type}"
    label = args.label or (
        f"vol_tf_{args.granularity}_atr{args.atr_factor}"
        f"_sma{args.vol_sma}_vf{args.vol_threshold}"
        f"_sl{args.sl_buffer}_r{args.risk}"
        f"_{close_mode}"
        f"_etw{args.etw_start.replace(':', '')}-{args.etw_end.replace(':', '')}"
    )
    report = generate_report(
        result["trade_log"], result["equity_curve"],
        df["time"], sizing_params["initial_capital"], label,
    )

    print_filter_stats(result["stats"], vol_params, quiet_params, entry_window_params,
                       wcse_params=wcse_params)
    print_wcse_mfe(result.get("wcse_mfe_rrs", []))

    print(f"\nFiles saved to: results/")
    for name, path in report["files"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
