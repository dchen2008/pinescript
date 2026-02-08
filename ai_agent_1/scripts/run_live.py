#!/usr/bin/env python3
"""Run live paper trading on OANDA demo account.

Usage:
    python -m scripts.run_live
    python -m scripts.run_live --config config/default.yaml
    python -m scripts.run_live --risk 5.0
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

import yaml

from src.data.oanda_client import OandaClient
from src.backtest.portfolio import Portfolio
from src.strategy.ppst_circle import PPSTCircleStrategy
from src.live.paper_trader import PaperTrader
from src.live.performance_monitor import PerformanceMonitor


def setup_logging(logs_dir: str) -> None:
    """Configure logging to console and file."""
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "live_trading.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler — INFO level
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(console)

    # File handler — DEBUG level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)


def main():
    parser = argparse.ArgumentParser(description="Run PPST live paper trading")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--risk", type=float, default=None, help="Override risk percent")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    live_cfg = config.get("live", {})
    logs_dir = live_cfg.get("logs_dir", "logs")

    # Set up logging
    setup_logging(logs_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PPST Live Paper Trading")
    logger.info("=" * 60)

    # Initialize OANDA client
    client = OandaClient()
    logger.info(f"OANDA API: {client.api_url}")
    logger.info(f"Account ID: {client.account_id}")

    # Fetch current account balance
    try:
        account_data = client.get_account()
        account_info = account_data.get("account", {})
        balance = float(account_info.get("balance", 0))
        currency = account_info.get("currency", "USD")
        logger.info(f"Account balance: {balance:,.2f} {currency}")
    except Exception as e:
        logger.error(f"Failed to connect to OANDA: {e}")
        sys.exit(1)

    # Risk percent: CLI flag > live config > backtest config
    risk_percent = args.risk or live_cfg.get("risk_percent", config["sizing"]["risk_percent"])

    # PPST params
    ppst_cfg = config["ppst"]
    ppst_params = {
        "pivot_period": ppst_cfg["pivot_period"],
        "atr_factor": ppst_cfg["atr_factor"],
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

    # Portfolio with OANDA balance and live risk
    sizing_cfg = config["sizing"]
    portfolio = Portfolio(
        initial_capital=balance,
        risk_percent=risk_percent,
        max_position_units=sizing_cfg["max_position_units"],
        spread_pips=config["spread_pips"],
    )

    # Circle strategy (best performing)
    circ_cfg = config["circle_strategy"]
    strategy = PPSTCircleStrategy(
        entry_mode=circ_cfg["entry_mode"],
        entry_window_min=circ_cfg["entry_window_min"],
        entry_circle_num=circ_cfg["entry_circle_num"],
        max_entry_dist_pips=circ_cfg["max_entry_dist_pips"],
        sl_buffer_pips=circ_cfg["sl_buffer_pips"],
        tp_type=circ_cfg["tp_type"],
        tp_ratio=circ_cfg["tp_ratio"],
        tp_fixed_pips=circ_cfg["tp_fixed_pips"],
        be_enabled=circ_cfg["be_enabled"],
        be_trigger_pips=circ_cfg["be_trigger_pips"],
        be_offset_pips=circ_cfg["be_offset_pips"],
    )

    # Performance monitor
    monitor_cfg = config.get("monitor", {})
    monitor = None
    if monitor_cfg.get("enabled", True):
        monitor = PerformanceMonitor(
            baseline_risk=monitor_cfg.get("baseline_risk", risk_percent),
            min_risk=monitor_cfg.get("min_risk", 0.5),
            dd_caution_threshold=monitor_cfg.get("dd_caution_threshold", 0.10),
            dd_danger_threshold=monitor_cfg.get("dd_danger_threshold", 0.20),
            gain_protect_threshold=monitor_cfg.get("gain_protect_threshold", 8.0),
            gain_protect_risk=monitor_cfg.get("gain_protect_risk", 0.75),
            logs_dir=logs_dir,
        )
        logger.info(f"Performance monitor: ENABLED (baseline={risk_percent}%, min={monitor.min_risk}%)")
    else:
        logger.info("Performance monitor: DISABLED")

    instrument = config["instrument"]
    granularity = config["timeframes"][0]  # M1

    logger.info(f"Strategy: {circ_cfg['entry_mode']}, Circle #{circ_cfg['entry_circle_num']}")
    logger.info(f"ATR factor: {ppst_cfg['atr_factor']}, RR: {circ_cfg['tp_ratio']}")
    logger.info(f"Risk: {risk_percent}% per trade")
    logger.info(f"Instrument: {instrument}, Timeframe: {granularity}")

    # Create paper trader
    trader = PaperTrader(
        client=client,
        strategy=strategy,
        portfolio=portfolio,
        instrument=instrument,
        granularity=granularity,
        ppst_params=ppst_params,
        time_filter_params=time_filter_params,
        poll_interval=live_cfg.get("poll_interval_seconds", 5),
        max_candles=live_cfg.get("max_candles_buffer", 500),
        use_circles=True,
        logs_dir=logs_dir,
        performance_monitor=monitor,
    )

    # Graceful shutdown on SIGINT
    def shutdown(signum, frame):
        logger.info("Shutdown signal received...")
        trader.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Run
    logger.info("Starting trading loop... (Ctrl+C to stop)")
    trader.run()

    # Print session summary
    summary = trader.get_session_summary()
    logger.info("=" * 60)
    logger.info("Session Summary")
    logger.info("=" * 60)
    logger.info(f"Trades taken: {summary['trades']}")
    logger.info(f"Session PnL: ${summary['pnl']:+,.2f}")
    logger.info(f"Start balance: ${summary['start_balance']:,.2f}")
    logger.info(f"End balance: ${summary['end_balance']:,.2f}")
    logger.info(f"Return: {summary['return_pct']:+.2f}%")
    if monitor is not None:
        logger.info(f"Final risk level: {monitor.current_risk_percent:.2f}%")
        logger.info(f"Week PnL: ${monitor.week_pnl:+,.2f}")


if __name__ == "__main__":
    main()
