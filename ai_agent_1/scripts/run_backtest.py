#!/usr/bin/env python3
"""Run a single backtest.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --config config/default.yaml
    python scripts/run_backtest.py --strategy circle --granularity M5 --atr-factor 3.5
"""

import argparse
from pathlib import Path

import yaml

from src.data.data_manager import load_candles
from src.backtest.engine import BacktestEngine
from src.backtest.portfolio import Portfolio
from src.backtest.report import generate_report
from src.strategy.ppst_signal import PPSTSignalStrategy
from src.strategy.ppst_circle import PPSTCircleStrategy


def main():
    parser = argparse.ArgumentParser(description="Run PPST backtest")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--strategy", choices=["signal", "circle"], default="signal")
    parser.add_argument("--instrument", default=None, help="Override instrument")
    parser.add_argument("--granularity", default=None, help="Override timeframe")
    parser.add_argument("--atr-factor", type=float, default=None, help="Override ATR factor")
    parser.add_argument("--rr-ratio", type=float, default=None, help="Override RR ratio")
    parser.add_argument("--risk", type=float, default=None, help="Override risk percent")
    parser.add_argument("--label", default=None, help="Result label")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    instrument = args.instrument or config["instrument"]
    granularity = args.granularity or config["timeframes"][0]

    # PPST params
    ppst_cfg = config["ppst"]
    ppst_params = {
        "pivot_period": ppst_cfg["pivot_period"],
        "atr_factor": args.atr_factor or ppst_cfg["atr_factor"],
        "atr_period": ppst_cfg["atr_period"],
    }

    # Time filter params
    tf_cfg = config["time_filter"]
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

    # Sizing params
    sizing_cfg = config["sizing"]
    risk_percent = args.risk if args.risk is not None else sizing_cfg["risk_percent"]
    portfolio = Portfolio(
        initial_capital=sizing_cfg["initial_capital"],
        risk_percent=risk_percent,
        max_position_units=sizing_cfg["max_position_units"],
        spread_pips=config["spread_pips"],
    )

    # Strategy
    if args.strategy == "signal":
        sig_cfg = config["signal_strategy"]
        rr = args.rr_ratio or sig_cfg["rr_ratio"]
        strategy = PPSTSignalStrategy(
            use_rr_tp=sig_cfg["use_rr_tp"],
            rr_ratio=rr,
            trail_with_supertrend=sig_cfg["trail_with_supertrend"],
            only_long=sig_cfg["only_long"],
        )
        use_circles = False
    else:
        circ_cfg = config["circle_strategy"]
        rr = args.rr_ratio or circ_cfg["tp_ratio"]
        strategy = PPSTCircleStrategy(
            entry_mode=circ_cfg["entry_mode"],
            entry_window_min=circ_cfg["entry_window_min"],
            entry_circle_num=circ_cfg["entry_circle_num"],
            max_entry_dist_pips=circ_cfg["max_entry_dist_pips"],
            sl_buffer_pips=circ_cfg["sl_buffer_pips"],
            tp_type=circ_cfg["tp_type"],
            tp_ratio=rr,
            tp_fixed_pips=circ_cfg["tp_fixed_pips"],
            be_enabled=circ_cfg["be_enabled"],
            be_trigger_pips=circ_cfg["be_trigger_pips"],
            be_offset_pips=circ_cfg["be_offset_pips"],
        )
        use_circles = True

    # Load data
    print(f"Loading {instrument} {granularity} data...")
    df = load_candles(instrument, granularity)
    print(f"Loaded {len(df)} candles: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Run backtest
    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        ppst_params=ppst_params,
        time_filter_params=time_filter_params,
        conservative_fills=config["backtest"]["conservative_fills"],
        use_circles=use_circles,
    )

    print("Running backtest...")
    result = engine.run(df)

    # Generate report
    label = args.label or f"{args.strategy}_{granularity}_atr{ppst_params['atr_factor']}_rr{rr}"
    report = generate_report(
        result["trade_log"],
        result["equity_curve"],
        df["time"],
        portfolio.initial_capital,
        label,
    )

    print(f"\nFiles saved to: results/")
    for name, path in report["files"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
