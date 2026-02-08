"""Parameter grid sweep optimizer using multiprocessing."""

import itertools
from multiprocessing import Pool
from typing import Optional

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.portfolio import Portfolio
from src.backtest.metrics import compute_metrics, compute_weekly_returns
from src.strategy.ppst_signal import PPSTSignalStrategy
from src.strategy.ppst_circle import PPSTCircleStrategy
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy


def _run_single_backtest(args: tuple) -> dict:
    """Run a single backtest with given parameters. Designed for multiprocessing."""
    params, df_dict, strategy_type = args

    # Reconstruct DataFrame from dict (can't pickle DataFrame across processes easily)
    df = pd.DataFrame(df_dict)
    df["time"] = pd.to_datetime(df["time"])

    # Build strategy
    if strategy_type == "vwap":
        strategy = VWAPMeanReversionStrategy(
            rsi_oversold=params.get("rsi_oversold", 30.0),
            rsi_overbought=params.get("rsi_overbought", 70.0),
            sl_pips=params.get("sl_pips", 15.0),
            tp_type=params.get("vwap_tp_type", "vwap"),
            tp_rr=params.get("tp_rr", 1.5),
        )
        use_circles = False
    elif strategy_type == "signal":
        strategy = PPSTSignalStrategy(
            use_rr_tp=params.get("use_rr_tp", True),
            rr_ratio=params.get("rr_ratio", 1.5),
            trail_with_supertrend=True,
        )
        use_circles = False
    else:
        strategy = PPSTCircleStrategy(
            entry_mode=params.get("entry_mode", "Nth Circle"),
            entry_window_min=params.get("entry_window_min", 30),
            entry_circle_num=params.get("entry_circle_num", 1),
            max_entry_dist_pips=params.get("max_entry_dist_pips", 20.0),
            tp_type=params.get("tp_type_str", "Risk Ratio"),
            tp_ratio=params.get("rr_ratio", 1.5),
            sl_buffer_pips=params.get("sl_buffer_pips", 1.0),
            be_enabled=params.get("be_enabled", True),
            be_trigger_pips=params.get("be_trigger_pips", 10.0),
            be_offset_pips=params.get("be_offset_pips", 1.0),
            volume_threshold=params.get("volume_threshold", 0.0),
        )
        use_circles = True

    portfolio = Portfolio(
        initial_capital=params.get("initial_capital", 10000.0),
        risk_percent=params.get("risk_percent", 1.0),
        spread_pips=params.get("spread_pips", 1.5),
    )

    ppst_params = {
        "atr_factor": params.get("atr_factor", 3.0),
        "pivot_period": params.get("pivot_period", 2),
        "atr_period": params.get("atr_period", 10),
        "volume_filter_period": params.get("volume_filter_period", 20),
    }

    vwap_params = {
        "session_start_utc": params.get("session_start_utc", 8),
        "vwap_band_mult": params.get("vwap_band_mult", 1.5),
        "rsi_period": params.get("rsi_period", 14),
    }

    time_filter_params = params.get("time_filter_params", {})

    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        ppst_params=ppst_params,
        time_filter_params=time_filter_params,
        use_circles=use_circles,
        strategy_type=strategy_type if strategy_type == "vwap" else "ppst",
        vwap_params=vwap_params,
    )

    result = engine.run(df)
    metrics = compute_metrics(result["trade_log"], result["equity_curve"], portfolio.initial_capital)

    weekly = compute_weekly_returns(result["trade_log"], df["time"], portfolio.initial_capital)
    avg_weekly_return = weekly["return_pct"].mean() if not weekly.empty else 0.0

    return {
        "params": params,
        "metrics": metrics,
        "avg_weekly_return": round(avg_weekly_return, 2),
        "total_trades": metrics["total_trades"],
        "net_profit_pct": metrics["net_profit_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
    }


def _build_circle_param_combos(
    atr_factors: list,
    rr_ratios: list,
    tp_types: list,
    entry_modes: list,
    entry_window_mins: Optional[list],
    entry_circle_nums: Optional[list],
    tp_type_map: dict,
    base_params: dict,
    volume_thresholds: Optional[list] = None,
) -> list:
    """Build parameter combos for circle strategy with mode-specific sub-params."""
    if volume_thresholds is None:
        volume_thresholds = [0.0]

    param_combos = []

    for atr_f, rr, tp, em, vt in itertools.product(
        atr_factors, rr_ratios, tp_types, entry_modes, volume_thresholds
    ):
        tp_cfg = tp_type_map.get(tp, tp_type_map["rr"])

        # Determine sub-parameter grid based on entry mode
        if em == "Circle within Time Window" and entry_window_mins:
            sub_params_list = [{"entry_window_min": w} for w in entry_window_mins]
        elif em == "Nth Circle" and entry_circle_nums:
            sub_params_list = [{"entry_circle_num": c} for c in entry_circle_nums]
        else:
            sub_params_list = [{}]

        for sub_params in sub_params_list:
            params = {
                "atr_factor": atr_f,
                "rr_ratio": rr,
                "tp_type": tp,
                "entry_mode": em,
                "volume_threshold": vt,
                **tp_cfg,
                **base_params,
                **sub_params,
            }
            param_combos.append(params)

    return param_combos


def _build_vwap_param_combos(
    session_starts: list,
    vwap_band_mults: list,
    rsi_periods: list,
    rsi_thresholds: list,
    sl_pips_list: list,
    tp_types: list,
    tp_rrs: list,
    base_params: dict,
) -> list:
    """Build parameter combos for VWAP mean reversion strategy."""
    param_combos = []

    for sess, band, rsi_p, (rsi_os, rsi_ob), sl, tp_type in itertools.product(
        session_starts, vwap_band_mults, rsi_periods, rsi_thresholds, sl_pips_list, tp_types,
    ):
        if tp_type == "rr":
            for rr in tp_rrs:
                params = {
                    "session_start_utc": sess,
                    "vwap_band_mult": band,
                    "rsi_period": rsi_p,
                    "rsi_oversold": rsi_os,
                    "rsi_overbought": rsi_ob,
                    "sl_pips": sl,
                    "vwap_tp_type": tp_type,
                    "tp_rr": rr,
                    **base_params,
                }
                param_combos.append(params)
        else:
            params = {
                "session_start_utc": sess,
                "vwap_band_mult": band,
                "rsi_period": rsi_p,
                "rsi_oversold": rsi_os,
                "rsi_overbought": rsi_ob,
                "sl_pips": sl,
                "vwap_tp_type": tp_type,
                "tp_rr": 1.5,
                **base_params,
            }
            param_combos.append(params)

    return param_combos


def run_optimization(
    df: pd.DataFrame,
    strategy_type: str = "signal",
    atr_factors: Optional[list] = None,
    rr_ratios: Optional[list] = None,
    tp_types: Optional[list] = None,
    entry_modes: Optional[list] = None,
    entry_window_mins: Optional[list] = None,
    entry_circle_nums: Optional[list] = None,
    time_filter_params: Optional[dict] = None,
    initial_capital: float = 10000.0,
    risk_percent: float = 1.0,
    spread_pips: float = 1.5,
    max_workers: int = 4,
    vwap_grid: Optional[dict] = None,
    volume_thresholds: Optional[list] = None,
) -> pd.DataFrame:
    """Run parameter grid sweep.

    Args:
        df: Candle data
        strategy_type: "signal", "circle", or "vwap"
        atr_factors: List of ATR factors to test (PPST strategies)
        rr_ratios: List of RR ratios to test (PPST strategies)
        tp_types: List of TP types to test
        entry_modes: List of entry modes for circle strategy
        entry_window_mins: List of window sizes for "Circle within Time Window" mode
        entry_circle_nums: List of circle numbers for "Nth Circle" mode
        time_filter_params: Time filter config
        initial_capital: Starting capital
        risk_percent: Risk percentage
        spread_pips: Spread in pips
        max_workers: Number of parallel workers
        vwap_grid: Dict of VWAP-specific grid params (for strategy_type="vwap")

    Returns:
        DataFrame with results sorted by avg_weekly_return
    """
    if atr_factors is None:
        atr_factors = [3.0, 3.5, 3.75, 4.0, 4.5, 5.0]
    if rr_ratios is None:
        rr_ratios = [1.0, 1.5, 2.0, 2.5]
    if tp_types is None:
        tp_types = ["rr", "none"]
    if time_filter_params is None:
        time_filter_params = {}

    # Map tp_type strings to strategy params
    tp_type_map = {
        "rr": {"use_rr_tp": True, "tp_type_str": "Risk Ratio"},
        "trailing": {"use_rr_tp": False, "tp_type_str": "None"},
        "none": {"use_rr_tp": False, "tp_type_str": "None"},
    }

    base_params = {
        "initial_capital": initial_capital,
        "risk_percent": risk_percent,
        "spread_pips": spread_pips,
        "time_filter_params": time_filter_params,
    }

    # Build parameter grid
    if strategy_type == "vwap":
        grid = vwap_grid or {}
        param_combos = _build_vwap_param_combos(
            session_starts=grid.get("session_starts", [8, 13]),
            vwap_band_mults=grid.get("vwap_band_mults", [1.0, 1.5, 2.0]),
            rsi_periods=grid.get("rsi_periods", [7, 14]),
            rsi_thresholds=grid.get("rsi_thresholds", [(30, 70), (25, 75)]),
            sl_pips_list=grid.get("sl_pips", [10, 15, 20]),
            tp_types=grid.get("tp_types", ["vwap", "rr"]),
            tp_rrs=grid.get("tp_rrs", [1.0, 1.5, 2.0]),
            base_params=base_params,
        )
    elif strategy_type == "circle":
        if entry_modes is None:
            entry_modes = [
                "Signal + Circle Same Bar",
                "Circle within Time Window",
                "Nth Circle",
                "Signal Swing",
            ]
        if entry_window_mins is None:
            entry_window_mins = [10, 15, 20, 25, 30, 35, 40, 45, 60]
        if entry_circle_nums is None:
            entry_circle_nums = [1, 2, 3]

        param_combos = _build_circle_param_combos(
            atr_factors=atr_factors,
            rr_ratios=rr_ratios,
            tp_types=tp_types,
            entry_modes=entry_modes,
            entry_window_mins=entry_window_mins,
            entry_circle_nums=entry_circle_nums,
            tp_type_map=tp_type_map,
            base_params=base_params,
            volume_thresholds=volume_thresholds,
        )
    else:
        param_combos = []
        for atr_f, rr, tp in itertools.product(atr_factors, rr_ratios, tp_types):
            tp_cfg = tp_type_map.get(tp, tp_type_map["rr"])
            params = {
                "atr_factor": atr_f,
                "rr_ratio": rr,
                "tp_type": tp,
                **tp_cfg,
                **base_params,
            }
            param_combos.append(params)

    print(f"Running {len(param_combos)} parameter combinations with {max_workers} workers...")

    # Convert df to dict for pickling
    df_dict = df.to_dict(orient="list")

    # Run in parallel
    args_list = [(p, df_dict, strategy_type) for p in param_combos]

    with Pool(processes=max_workers) as pool:
        results = pool.map(_run_single_backtest, args_list)

    # Build results DataFrame
    rows = []
    for r in results:
        if strategy_type == "vwap":
            row = {
                "session_start_utc": r["params"].get("session_start_utc", ""),
                "vwap_band_mult": r["params"].get("vwap_band_mult", ""),
                "rsi_period": r["params"].get("rsi_period", ""),
                "rsi_oversold": r["params"].get("rsi_oversold", ""),
                "rsi_overbought": r["params"].get("rsi_overbought", ""),
                "sl_pips": r["params"].get("sl_pips", ""),
                "tp_type": r["params"].get("vwap_tp_type", ""),
                "tp_rr": r["params"].get("tp_rr", ""),
                "total_trades": r["total_trades"],
                "win_rate": r["win_rate"],
                "profit_factor": r["profit_factor"],
                "net_profit_pct": r["net_profit_pct"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "avg_weekly_return": r["avg_weekly_return"],
            }
        else:
            row = {
                "entry_mode": r["params"].get("entry_mode", "N/A"),
                "entry_window_min": r["params"].get("entry_window_min", ""),
                "entry_circle_num": r["params"].get("entry_circle_num", ""),
                "atr_factor": r["params"]["atr_factor"],
                "rr_ratio": r["params"]["rr_ratio"],
                "tp_type": r["params"]["tp_type"],
                "volume_threshold": r["params"].get("volume_threshold", 0.0),
                "total_trades": r["total_trades"],
                "win_rate": r["win_rate"],
                "profit_factor": r["profit_factor"],
                "net_profit_pct": r["net_profit_pct"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "avg_weekly_return": r["avg_weekly_return"],
            }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("avg_weekly_return", ascending=False)
    return results_df
