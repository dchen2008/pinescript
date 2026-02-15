#!/usr/bin/env python3
"""PPST Volume Filter Backtest.

Port of ppst_vol_filter_bt.pine.

Filters: Volume (SMA ratio), Quiet Hours (configurable TZ).
Features: Vol Recovery, Filtered Exit, SL Buffer Trailing, Compound Sizing.

Usage:
    cd ai_agent_1
    python3 -m src.vol
    python3 -m src.vol --granularity M5 --atr-factor 5.0 --vol-threshold 1.0
    python3 -m src.vol --granularity M5 --tp-type rr --tp-ratio 2.0 --risk 15
"""

import argparse
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytz

import yaml

from src.data.data_manager import load_candles
from src.indicators.ppst import compute_ppst
from src.indicators.volume_filter import compute_relative_volume
from src.backtest.portfolio import Portfolio
from src.backtest.report import generate_report
from src.strategy.position import Position
from src.strategy.wick_cross import (
    WickCrossState,
    detect_wick_cross_long,
    detect_wick_cross_short,
    detect_engulfing_near_st,
    detect_wcse1_setup,
    check_stop_fill,
)
from src.utils.forex_utils import pips_to_price


# ---------------------------------------------------------------------------
# Time window helpers (shared by vol.py and vol_tf.py)
# ---------------------------------------------------------------------------

def compute_time_window(times: pd.Series, tz_name: str,
                        start_hour: int, start_min: int,
                        end_hour: int, end_min: int,
                        enabled: bool, invert: bool = False) -> np.ndarray:
    """Compute a time-of-day boolean mask in the given timezone.

    Args:
        times: Series of UTC timestamps.
        tz_name: IANA timezone name (e.g. "America/Los_Angeles").
        start_hour/min, end_hour/min: Window boundaries.
        enabled: If False, returns all-False (for blacklists) or all-True (for whitelists).
        invert: If False (blacklist/quiet), returns True when INSIDE window.
                If True (whitelist/entry), returns True when INSIDE window.

    Returns:
        Boolean ndarray, True when bar is INSIDE the window (and enabled).
    """
    n = len(times)
    if not enabled:
        return np.zeros(n, dtype=bool) if not invert else np.ones(n, dtype=bool)

    tz = pytz.timezone(tz_name)
    utc_times = times.dt.tz_localize("UTC") if times.dt.tz is None else times.dt.tz_convert("UTC")
    local_times = utc_times.dt.tz_convert(tz)

    current_min = (local_times.dt.hour * 60 + local_times.dt.minute).values
    start_total = start_hour * 60 + start_min
    end_total = end_hour * 60 + end_min

    if start_total <= end_total:
        # Normal range (e.g. 12:00 - 14:30)
        inside = (current_min >= start_total) & (current_min < end_total)
    else:
        # Overnight range (e.g. 22:00 - 06:00)
        inside = (current_min >= start_total) | (current_min < end_total)

    return inside


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_vol_backtest(
    df: pd.DataFrame,
    ppst_params: dict,
    vol_params: dict,
    quiet_params: dict,
    strategy_params: dict,
    sizing_params: dict,
    entry_window_params: Optional[dict] = None,
    wcse_params: Optional[dict] = None,
    wcse2_params: Optional[dict] = None,
) -> dict:
    """Run the volume filter backtest.

    This function is used by both vol.py (no entry window) and vol_tf.py
    (with entry window).

    Returns:
        Dict with trade_log, equity_curve, stats, total_bars, warmup_bars.
    """
    # --- Compute indicators (vectorized) ---
    data = compute_ppst(df, **ppst_params)

    vol_period = vol_params["vol_sma_period"]
    data["rel_volume"] = compute_relative_volume(df["volume"].values, vol_period)

    # Quiet hours mask (True = bar is inside quiet window)
    is_quiet_arr = compute_time_window(
        data["time"], quiet_params["tz"],
        quiet_params["start_hour"], quiet_params["start_min"],
        quiet_params["end_hour"], quiet_params["end_min"],
        quiet_params["enabled"],
    )

    # Entry window mask (True = bar is inside entry window)
    use_etw = entry_window_params is not None and entry_window_params.get("enabled", False)
    if use_etw:
        is_in_etw_arr = compute_time_window(
            data["time"], entry_window_params["tz"],
            entry_window_params["start_hour"], entry_window_params["start_min"],
            entry_window_params["end_hour"], entry_window_params["end_min"],
            True,
        )
    else:
        is_in_etw_arr = np.ones(len(data), dtype=bool)

    # --- Setup ---
    portfolio = Portfolio(
        initial_capital=sizing_params["initial_capital"],
        risk_percent=sizing_params["risk_percent"],
        max_position_units=sizing_params["max_position_units"],
        spread_pips=sizing_params["spread_pips"],
    )

    sl_buffer_price = pips_to_price(strategy_params["sl_buffer_pips"])
    warmup = max(ppst_params["pivot_period"] * 2 + 1,
                 ppst_params["atr_period"] + 1,
                 vol_period + 1, 20)
    n = len(data)

    # Optionally start trading after a specific date (for matching TV range)
    start_date = strategy_params.get("start_date")
    if start_date is not None:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        for j in range(warmup, n):
            if data.iloc[j]["time"] >= start_ts:
                warmup = max(warmup, j)
                break

    position = None  # type: Optional[Position]
    trade_log: list = []

    # Deferred entry state (volume recovery)
    deferred_dir = 0   # 1=deferred buy, -1=deferred sell, 0=none
    deferred_age = 0

    # Filtered exit state
    filtered_exit_countdown = 0
    filtered_exit_dir = 0

    stats = {
        "total_signals": 0,
        "vol_filtered": 0,
        "qh_filtered": 0,
        "etw_filtered": 0,
        "vol_recovered": 0,
        "filtered_exits": 0,
        "traded": 0,
        "wcse_armed": 0,
        "wcse_patterns": 0,
        "wcse_entries": 0,
        "wcse_skip_limit": 0,
        "wcse_skip_vol": 0,
        "wcse_stop_gap_fills": 0,
        "wcse_stop_normal_fills": 0,
        "wcse_stop_unfilled": 0,
        "wcse2_armed": 0,
        "wcse2_patterns": 0,
        "wcse2_entries": 0,
        "wcse2_skip_limit": 0,
        "wcse2_skip_vol": 0,
    }

    # WCSE state
    use_wcse = wcse_params is not None and wcse_params.get("enabled", False)
    wcse = WickCrossState()
    wcse_entry_info: list = []  # (entry_bar, entry_price, sl_price, direction, sl_dist)

    # WCSE2 state
    use_wcse2 = wcse2_params is not None and wcse2_params.get("enabled", False)
    wcse2 = WickCrossState()
    wcse2_entry_info: list = []  # (entry_bar, entry_price, sl_price, direction, sl_dist)

    # WCSE1 pending stop order (for stop order entry mode)
    wcse1_pending_stop = None  # {direction, stop_price, sl, tp, units, placed_bar}

    # --- Bar-by-bar loop ---
    for i in range(warmup, n):
        row = data.iloc[i]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        open_price = row["open"]
        tup = row["tup"]
        tdown = row["tdown"]
        trend = int(row.get("trend", 0))

        buy_raw = bool(row.get("buy_signal", False))
        sell_raw = bool(row.get("sell_signal", False))

        is_quiet_bar = bool(is_quiet_arr[i])
        is_in_etw = bool(is_in_etw_arr[i])
        rel_vol = row["rel_volume"]
        is_low_vol = (vol_params["use_vol_filter"]
                      and not np.isnan(rel_vol)
                      and rel_vol < vol_params["vol_threshold"])

        # ── Check SL/TP fills (skip in swing-close mode) ──
        swing_close = strategy_params.get("swing_close", False)
        if position is not None and not swing_close:
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
                # Conservative: assume SL hit first
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
                if use_wcse and position.entry_source == "wcse":
                    wcse.record_tp_hit()
                if use_wcse2 and position.entry_source == "wcse2":
                    wcse2.record_tp_hit()
                pnl = position.close_position(tp_fill, "TP Hit", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None

        # ── Trail SL with SuperTrend + buffer (skip in swing-close mode) ──
        if position is not None and not swing_close:
            if position.is_long and not math.isnan(tup):
                position.trail_sl_with_supertrend(
                    tup, sl_buffer_pips=strategy_params["sl_buffer_pips"])
            elif position.is_short and not math.isnan(tdown):
                position.trail_sl_with_supertrend(
                    tdown, sl_buffer_pips=strategy_params["sl_buffer_pips"])

        # ── Filter logic ──
        # Priority: QH > ETW > Vol (matches PineScript label priority)
        if buy_raw or sell_raw:
            stats["total_signals"] += 1

        is_vol_filtered = ((buy_raw or sell_raw) and is_low_vol
                           and not is_quiet_bar and is_in_etw)
        is_qh_filtered = (buy_raw or sell_raw) and is_quiet_bar
        is_etw_filtered = ((buy_raw or sell_raw) and use_etw
                           and not is_in_etw and not is_quiet_bar)
        is_filtered = is_quiet_bar or is_low_vol or (use_etw and not is_in_etw)

        if is_vol_filtered:
            stats["vol_filtered"] += 1
        if is_qh_filtered:
            stats["qh_filtered"] += 1
        if is_etw_filtered:
            stats["etw_filtered"] += 1

        # ── Volume Recovery (deferred entry) ──
        if buy_raw and is_vol_filtered:
            deferred_dir = 1
            deferred_age = 0
        if sell_raw and is_vol_filtered:
            deferred_dir = -1
            deferred_age = 0

        # Opposite raw signal resets deferral
        if buy_raw and deferred_dir == -1:
            deferred_dir = 0
        if sell_raw and deferred_dir == 1:
            deferred_dir = 0

        # Age deferral (only when no new signal this bar)
        if deferred_dir != 0 and not buy_raw and not sell_raw:
            deferred_age += 1

        # Expire if too old or trend reversed
        if deferred_dir == 1 and (trend != 1 or deferred_age > vol_params["vol_recovery_bars"]):
            deferred_dir = 0
        if deferred_dir == -1 and (trend != -1 or deferred_age > vol_params["vol_recovery_bars"]):
            deferred_dir = 0

        # Volume recovered? (skip if quiet or outside entry window)
        vol_recovered_ok = not is_low_vol and not is_quiet_bar and is_in_etw
        deferred_buy = (deferred_dir == 1 and vol_recovered_ok
                        and deferred_age > 0 and vol_params["vol_recovery_bars"] > 0)
        deferred_sell = (deferred_dir == -1 and vol_recovered_ok
                         and deferred_age > 0 and vol_params["vol_recovery_bars"] > 0)

        if deferred_buy or deferred_sell:
            deferred_dir = 0
            stats["vol_recovered"] += 1

        # ── Filtered Exit (delayed close of opposite position) ──
        has_short = position is not None and position.is_short
        has_long = position is not None and position.is_long

        if buy_raw and is_vol_filtered and has_short and vol_params["filtered_exit_bars"] > 0:
            filtered_exit_countdown = vol_params["filtered_exit_bars"]
            filtered_exit_dir = 1
        if sell_raw and is_vol_filtered and has_long and vol_params["filtered_exit_bars"] > 0:
            filtered_exit_countdown = vol_params["filtered_exit_bars"]
            filtered_exit_dir = -1

        # Actionable signals
        buy_signal = (buy_raw and not is_filtered) or deferred_buy
        sell_signal = (sell_raw and not is_filtered) or deferred_sell

        # Actionable signal cancels filtered exit countdown
        if buy_signal or sell_signal:
            filtered_exit_countdown = 0
            filtered_exit_dir = 0

        # Countdown
        filtered_exit_now = False
        if filtered_exit_countdown > 0:
            filtered_exit_countdown -= 1
            if filtered_exit_countdown == 0:
                filtered_exit_now = True

        # ── Execute filtered exit ──
        if filtered_exit_now and position is not None:
            should_exit = ((filtered_exit_dir == 1 and position.is_short) or
                           (filtered_exit_dir == -1 and position.is_long))
            if should_exit:
                pnl = position.close_position(close, "Filtered Exit", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None
                stats["filtered_exits"] += 1
            filtered_exit_dir = 0

        # ── WCSE2 arm/disarm ──
        wcse2_entry = False
        if use_wcse2:
            # Arm on vol-filtered signal during matching trend
            if buy_raw and is_vol_filtered and trend == 1:
                wcse2.arm(1)
                stats["wcse2_armed"] += 1
            elif sell_raw and is_vol_filtered and trend == -1:
                wcse2.arm(-1)
                stats["wcse2_armed"] += 1
            # Disarm on actionable signal
            if (buy_raw and not is_filtered) or (sell_raw and not is_filtered):
                wcse2.disarm()
            # Disarm on trend mismatch
            if wcse2.armed_dir == 1 and trend != 1:
                wcse2.disarm()
            if wcse2.armed_dir == -1 and trend != -1:
                wcse2.disarm()

        # ── WCSE2 pattern detection + entry ──
        if use_wcse2 and i >= warmup + 2 and wcse2.is_active(position is None):
            wcse2_entry_times = wcse2_params.get("entry_times", 0)
            if not wcse2.can_enter(wcse2_entry_times):
                stats["wcse2_skip_limit"] += 1
            elif is_quiet_bar or not is_in_etw:
                pass  # skip during quiet hours / outside ETW
            else:
                # Volume filter for WCSE2: skip or use alt threshold
                wcse2_vol_ok = True
                if not wcse2_params.get("vf_skip", False):
                    wcse2_vf_threshold = wcse2_params.get("vf_threshold", 3.0)
                    if (vol_params["use_vol_filter"]
                            and not np.isnan(rel_vol)
                            and rel_vol < wcse2_vf_threshold):
                        wcse2_vol_ok = False
                        stats["wcse2_skip_vol"] += 1

                if wcse2_vol_ok:
                    # Extract 3 bars: c1 (i-2), c2 (i-1), c3 (i)
                    bars_3 = []
                    st_vals_3 = []
                    for k in range(2, -1, -1):  # i-2, i-1, i
                        r = data.iloc[i - k]
                        bars_3.append((r["open"], r["high"], r["low"], r["close"]))
                        if wcse2.armed_dir == 1:
                            st_vals_3.append(r["tup"])
                        else:
                            st_vals_3.append(r["tdown"])

                    pattern_match = detect_engulfing_near_st(
                        bars_3, st_vals_3, wcse2.armed_dir,
                        c_body_pips=wcse2_params.get("c_body_pips", 0.2),
                        c_wick_pips=wcse2_params.get("c_wick_pips", 5.0),
                        c_close_pips=wcse2_params.get("c_close_pips", 5.0),
                        engulf_body_pips=wcse2_params.get("engulf_body_pips", 1.0),
                    )

                    if pattern_match:
                        stats["wcse2_patterns"] += 1
                        w2_dir = wcse2.armed_dir
                        w2_sl_buffer = pips_to_price(wcse2_params.get("sl_buffer_pips", 0.0))
                        w2_tp_rr = wcse2_params.get("tp_rr", 4.0)

                        if w2_dir == 1:
                            # LONG: entry at close, SL at TUp
                            spread_entry = close + portfolio.spread_price
                            sl = tup - w2_sl_buffer
                            sl_dist = spread_entry - sl
                        else:
                            # SHORT: entry at close, SL at TDown
                            spread_entry = close - portfolio.spread_price
                            sl = tdown + w2_sl_buffer
                            sl_dist = sl - spread_entry

                        if sl_dist > 0:
                            tp = spread_entry + w2_dir * sl_dist * w2_tp_rr
                            units = portfolio.compute_position_size(sl_dist)
                            if units > 0:
                                position = Position(
                                    direction=w2_dir,
                                    entry_price=close,
                                    units=units,
                                    sl_price=sl,
                                    tp_price=tp,
                                    entry_bar=i,
                                    entry_time=str(row["time"]),
                                    entry_source="wcse2",
                                )
                                wcse2_entry = True
                                stats["wcse2_entries"] += 1
                                stats["traded"] += 1
                                wcse2_entry_info.append((i, close, sl, w2_dir, sl_dist))

        # ── WCSE1 pending stop fill ──
        wcse_entry = False
        if wcse1_pending_stop is not None and position is None:
            ps = wcse1_pending_stop
            filled, fill_price = check_stop_fill(
                (open_price, high, low, close),
                ps["stop_price"],
                ps["direction"],
            )
            if filled:
                if fill_price == open_price and fill_price != ps["stop_price"]:
                    stats["wcse_stop_gap_fills"] += 1
                else:
                    stats["wcse_stop_normal_fills"] += 1
                # Apply spread to fill price
                if ps["direction"] == 1:
                    spread_entry = fill_price + portfolio.spread_price
                else:
                    spread_entry = fill_price - portfolio.spread_price
                # Recompute SL distance from actual fill price (may differ for gap fills)
                sl_dist_actual = abs(spread_entry - ps["sl"])
                if sl_dist_actual > 0:
                    # Recompute TP from actual fill
                    wcse_tp_rr = wcse_params.get("tp_rr", 2.0)
                    tp_actual = spread_entry + ps["direction"] * sl_dist_actual * wcse_tp_rr
                    # Recompute position size from actual SL distance
                    units_actual = portfolio.compute_position_size(sl_dist_actual)
                    if units_actual > 0:
                        position = Position(
                            direction=ps["direction"],
                            entry_price=fill_price,
                            units=units_actual,
                            sl_price=ps["sl"],
                            tp_price=tp_actual,
                            entry_bar=i,
                            entry_time=str(row["time"]),
                            entry_source="wcse",
                        )
                        wcse_entry = True
                        stats["wcse_entries"] += 1
                        stats["traded"] += 1
                        wcse_entry_info.append((i, fill_price, ps["sl"], ps["direction"], sl_dist_actual))

                        # Same-bar SL/TP check (stop filled mid-bar)
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
            if not filled:
                stats["wcse_stop_unfilled"] += 1
            wcse1_pending_stop = None  # 1-bar validity: always clear

        # ── WCSE arm/disarm ──
        if use_wcse:
            # Arm on vol-filtered signal during matching trend
            if buy_raw and is_vol_filtered and trend == 1:
                wcse.arm(1)
                stats["wcse_armed"] += 1
            elif sell_raw and is_vol_filtered and trend == -1:
                wcse.arm(-1)
                stats["wcse_armed"] += 1
            # Disarm on actionable signal
            if (buy_raw and not is_filtered) or (sell_raw and not is_filtered):
                wcse.disarm()
            # Disarm on trend mismatch
            if wcse.armed_dir == 1 and trend != 1:
                wcse.disarm()
            if wcse.armed_dir == -1 and trend != -1:
                wcse.disarm()

        # ── WCSE1 setup detection → set pending stop for next bar ──
        if not wcse2_entry and use_wcse and i >= warmup + 1 and wcse.is_active(position is None):
            wcse_entry_times = wcse_params.get("entry_times", 0)
            if not wcse.can_enter(wcse_entry_times):
                stats["wcse_skip_limit"] += 1
            elif is_quiet_bar or not is_in_etw:
                pass  # skip during quiet hours / outside ETW
            else:
                # Volume filter for WCSE: skip or use alt threshold
                wcse_vol_ok = True
                if not wcse_params.get("vf_skip", False):
                    wcse_vf_threshold = wcse_params.get("vf_threshold", 3.0)
                    if (vol_params["use_vol_filter"]
                            and not np.isnan(rel_vol)
                            and rel_vol < wcse_vf_threshold):
                        wcse_vol_ok = False
                        stats["wcse_skip_vol"] += 1

                if wcse_vol_ok:
                    row_m1 = data.iloc[i - 1]  # c1
                    # c2 is current bar (fully closed in backtest)
                    wcse_dir = wcse.armed_dir

                    if wcse_dir == 1:
                        c1_st = row_m1["tup"]
                        c2_st = tup
                    else:
                        c1_st = row_m1["tdown"]
                        c2_st = tdown

                    valid, breakout_level = detect_wcse1_setup(
                        (row_m1["open"], row_m1["high"], row_m1["low"], row_m1["close"]),
                        (open_price, high, low, close),
                        c1_st, c2_st, wcse_dir,
                        c1_body_pips=wcse_params.get("c1_body_pips", 3.0),
                        c1_wick_pips=wcse_params.get("c1_wick_pips", 0.5),
                        c1_close_pips=wcse_params.get("c1_close_pips", 1.0),
                        c2_body_pips=wcse_params.get("c2_body_pips", 2.0),
                        c2_wick_pips=wcse_params.get("c2_wick_pips", 0.5),
                        c2_close_pips=wcse_params.get("c2_close_pips", 1.0),
                    )

                    if valid:
                        stats["wcse_patterns"] += 1
                        stop_buffer = pips_to_price(wcse_params.get("stop_buffer_pips", 0.3))
                        stop_price = breakout_level + wcse_dir * stop_buffer

                        # Compute SL/TP using c2's SuperTrend (current bar)
                        wcse_sl_buffer = pips_to_price(wcse_params.get("sl_buffer_pips", 0.0))
                        wcse_tp_rr = wcse_params.get("tp_rr", 2.0)

                        if wcse_dir == 1:
                            sl = tup - wcse_sl_buffer
                            spread_entry = stop_price + portfolio.spread_price
                            sl_dist = spread_entry - sl
                        else:
                            sl = tdown + wcse_sl_buffer
                            spread_entry = stop_price - portfolio.spread_price
                            sl_dist = sl - spread_entry

                        if sl_dist > 0:
                            tp = spread_entry + wcse_dir * sl_dist * wcse_tp_rr
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
        if not wcse2_entry and not wcse_entry and buy_signal and (position is None or (position is not None and position.is_short)):
            stats["traded"] += 1

            # Close existing short (reversal)
            if position is not None and position.is_short:
                pnl = position.close_position(close, "Reversal", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None

            # Open long
            sl_base = tup if not math.isnan(tup) else np.nan
            if not math.isnan(sl_base):
                # TV fills at close (process_orders_on_close=true).
                # Spread only affects sizing and TP computation.
                spread_entry = close + portfolio.spread_price
                sl = sl_base - sl_buffer_price
                sl_dist = spread_entry - sl
                if sl_dist > 0:
                    tp = _compute_tp(spread_entry, sl_dist, 1, strategy_params)
                    units = portfolio.compute_position_size(sl_dist)
                    if units > 0:
                        position = Position(
                            direction=1, entry_price=close, units=units,
                            sl_price=sl, tp_price=tp,
                            entry_bar=i, entry_time=str(row["time"]),
                        )

        # ── Execute sell signal ──
        elif not wcse2_entry and not wcse_entry and sell_signal and (position is None or (position is not None and position.is_long)):
            stats["traded"] += 1

            # Close existing long (reversal)
            if position is not None and position.is_long:
                pnl = position.close_position(close, "Reversal", i)
                portfolio.record_trade(pnl)
                trade_log.append(position.to_dict())
                position = None

            # Open short
            sl_base = tdown if not math.isnan(tdown) else np.nan
            if not math.isnan(sl_base):
                # TV fills at close (process_orders_on_close=true).
                spread_entry = close - portfolio.spread_price
                sl = sl_base + sl_buffer_price
                sl_dist = sl - spread_entry
                if sl_dist > 0:
                    tp = _compute_tp(spread_entry, sl_dist, -1, strategy_params)
                    units = portfolio.compute_position_size(sl_dist)
                    if units > 0:
                        position = Position(
                            direction=-1, entry_price=close, units=units,
                            sl_price=sl, tp_price=tp,
                            entry_bar=i, entry_time=str(row["time"]),
                        )

        # ── Record equity ──
        unrealized = 0.0
        if position is not None:
            if position.is_long:
                unrealized = (close - position.entry_price) * position.units
            else:
                unrealized = (position.entry_price - close) * position.units
        portfolio.record_equity(i, unrealized)

    # Close remaining position at end of data
    if position is not None:
        last_close = data.iloc[-1]["close"]
        pnl = position.close_position(last_close, "End of Data", n - 1)
        portfolio.record_trade(pnl)
        trade_log.append(position.to_dict())

    # --- WCSE MFE analysis (theoretical max RR without TP) ---
    highs = data["high"].values
    lows = data["low"].values

    wcse_mfe_rrs: list = []
    if use_wcse and wcse_entry_info:
        for entry_bar, entry_price, sl_price, direction, sl_dist in wcse_entry_info:
            max_rr = 0.0
            for j in range(entry_bar, n):
                if direction == 1:
                    favorable = highs[j] - entry_price
                    if lows[j] <= sl_price:
                        break
                else:
                    favorable = entry_price - lows[j]
                    if highs[j] >= sl_price:
                        break
                rr = favorable / sl_dist if sl_dist > 0 else 0.0
                if rr > max_rr:
                    max_rr = rr
            wcse_mfe_rrs.append(max_rr)

    # --- WCSE2 MFE analysis ---
    wcse2_mfe_rrs: list = []
    if use_wcse2 and wcse2_entry_info:
        for entry_bar, entry_price, sl_price, direction, sl_dist in wcse2_entry_info:
            max_rr = 0.0
            for j in range(entry_bar, n):
                if direction == 1:
                    favorable = highs[j] - entry_price
                    if lows[j] <= sl_price:
                        break
                else:
                    favorable = entry_price - lows[j]
                    if highs[j] >= sl_price:
                        break
                rr = favorable / sl_dist if sl_dist > 0 else 0.0
                if rr > max_rr:
                    max_rr = rr
            wcse2_mfe_rrs.append(max_rr)

    return {
        "trade_log": trade_log,
        "equity_curve": portfolio.equity_curve,
        "stats": stats,
        "total_bars": n,
        "warmup_bars": warmup,
        "initial_capital": sizing_params["initial_capital"],
        "wcse_mfe_rrs": wcse_mfe_rrs,
        "wcse2_mfe_rrs": wcse2_mfe_rrs,
    }


def _compute_tp(entry: float, sl_dist: float, direction: int,
                strategy_params: dict) -> Optional[float]:
    """Compute TP price based on strategy params."""
    tp_type = strategy_params["tp_type"]
    if tp_type == "rr":
        return entry + direction * sl_dist * strategy_params["tp_ratio"]
    elif tp_type == "fixed":
        return entry + direction * pips_to_price(strategy_params["tp_fixed_pips"])
    return None


def print_filter_stats(stats: dict, vol_params: dict, quiet_params: dict,
                       entry_window_params: Optional[dict] = None,
                       wcse_params: Optional[dict] = None,
                       wcse2_params: Optional[dict] = None) -> None:
    """Print filter statistics."""
    print(f"\n  Filter Statistics:")
    print(f"  Total Signals:   {stats['total_signals']}")
    print(f"  Vol Filtered:    {stats['vol_filtered']}")
    print(f"  Vol Recovered:   {stats['vol_recovered']}")
    print(f"  Filtered Exits:  {stats['filtered_exits']}")
    print(f"  QH Filtered:     {stats['qh_filtered']}")
    if entry_window_params and entry_window_params.get("enabled"):
        print(f"  ETW Filtered:    {stats['etw_filtered']}")
    total = stats["total_signals"]
    traded = stats["traded"]
    pct = ((1.0 - traded / total) * 100) if total > 0 else 0
    print(f"  Traded:          {traded} ({pct:.1f}% filtered)")
    if wcse2_params and wcse2_params.get("enabled"):
        print(f"\n  WCSE2 Statistics:")
        print(f"  WCSE2 Armed:      {stats.get('wcse2_armed', 0)}")
        print(f"  WCSE2 Patterns:   {stats.get('wcse2_patterns', 0)}")
        print(f"  WCSE2 Entries:    {stats.get('wcse2_entries', 0)}")
        print(f"  WCSE2 Skip Limit: {stats.get('wcse2_skip_limit', 0)}")
        print(f"  WCSE2 Skip Vol:   {stats.get('wcse2_skip_vol', 0)}")
    if wcse_params and wcse_params.get("enabled"):
        print(f"\n  WCSE Statistics:")
        print(f"  WCSE Armed:      {stats.get('wcse_armed', 0)}")
        print(f"  WCSE Patterns:   {stats.get('wcse_patterns', 0)}")
        print(f"  WCSE Entries:    {stats.get('wcse_entries', 0)}")
        print(f"  WCSE Skip Limit: {stats.get('wcse_skip_limit', 0)}")
        print(f"  WCSE Skip Vol:   {stats.get('wcse_skip_vol', 0)}")
        gap = stats.get('wcse_stop_gap_fills', 0)
        normal = stats.get('wcse_stop_normal_fills', 0)
        unfilled = stats.get('wcse_stop_unfilled', 0)
        print(f"  WCSE Stop Fills: {gap + normal} (gap: {gap}, normal: {normal}, unfilled: {unfilled})")


def print_wcse_mfe(wcse_mfe_rrs: list) -> None:
    """Print WCSE MFE RR distribution (theoretical max RR without TP)."""
    if not wcse_mfe_rrs:
        return
    total = len(wcse_mfe_rrs)
    print(f"\n  WCSE MFE Analysis ({total} entries, no-TP theoretical max):")
    print(f"  {'RR Level':<12} {'Count':>6} {'% Reached':>10}  Bar")
    print(f"  {'-'*42}")
    for rr_level in [0.5, 1, 2, 3, 4, 5, 6, 8, 10]:
        count = sum(1 for rr in wcse_mfe_rrs if rr >= rr_level)
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {rr_level:>5.1f}R      {count:>6} {pct:>9.1f}%  {bar}")
    avg_rr = sum(wcse_mfe_rrs) / total
    max_rr = max(wcse_mfe_rrs)
    med_rr = sorted(wcse_mfe_rrs)[total // 2]
    print(f"  {'-'*42}")
    print(f"  Avg MFE: {avg_rr:.2f}R  |  Median: {med_rr:.2f}R  |  Max: {max_rr:.1f}R")


def print_wcse2_mfe(wcse2_mfe_rrs: list) -> None:
    """Print WCSE2 MFE RR distribution (theoretical max RR without TP)."""
    if not wcse2_mfe_rrs:
        return
    total = len(wcse2_mfe_rrs)
    print(f"\n  WCSE2 MFE Analysis ({total} entries, no-TP theoretical max):")
    print(f"  {'RR Level':<12} {'Count':>6} {'% Reached':>10}  Bar")
    print(f"  {'-'*42}")
    for rr_level in [0.5, 1, 2, 3, 4, 5, 6, 8, 10]:
        count = sum(1 for rr in wcse2_mfe_rrs if rr >= rr_level)
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {rr_level:>5.1f}R      {count:>6} {pct:>9.1f}%  {bar}")
    avg_rr = sum(wcse2_mfe_rrs) / total
    max_rr = max(wcse2_mfe_rrs)
    med_rr = sorted(wcse2_mfe_rrs)[total // 2]
    print(f"  {'-'*42}")
    print(f"  Avg MFE: {avg_rr:.2f}R  |  Median: {med_rr:.2f}R  |  Max: {max_rr:.1f}R")


def build_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments shared by vol.py and vol_tf.py."""
    # PPST
    parser.add_argument("--pivot-period", type=int, default=2)
    parser.add_argument("--atr-factor", type=float, default=5.0)
    parser.add_argument("--atr-period", type=int, default=10)

    # Volume filter
    parser.add_argument("--vol-sma", type=int, default=None, help="Volume SMA period")
    parser.add_argument("--vol-threshold", type=float, default=None, help="Min vol ratio")
    parser.add_argument("--vol-recovery", type=int, default=0, help="Vol recovery bars (0=off)")
    parser.add_argument("--filtered-exit", type=int, default=0, help="Filtered exit bars (0=off)")
    parser.add_argument("--no-vol-filter", action="store_true", help="Disable volume filter")

    # Quiet hours
    parser.add_argument("--no-quiet", action="store_true", help="Disable quiet hours")
    parser.add_argument("--quiet-start", default="12:00", help="Quiet start HH:MM")
    parser.add_argument("--quiet-end", default="14:30", help="Quiet end HH:MM")
    parser.add_argument("--quiet-tz", default="America/Los_Angeles")

    # Strategy
    parser.add_argument("--swing-close", action="store_true",
                        help="Signal swing only: no SL/TP exits, close only on reversal signal")
    parser.add_argument("--sl-buffer", type=float, default=3.0, help="SL buffer pips")
    parser.add_argument("--tp-type", choices=["rr", "fixed", "none"], default="none")
    parser.add_argument("--tp-ratio", type=float, default=1.5, help="TP risk:reward ratio")
    parser.add_argument("--tp-pips", type=float, default=30.0, help="TP fixed pips")

    # Sizing
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade (pct)")
    parser.add_argument("--max-position", type=int, default=500000)
    parser.add_argument("--spread", type=float, default=1.5, help="Spread in pips")

    # Data
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--start-date", default=None,
                        help="Start trading from this UTC date (YYYY-MM-DD or YYYY-MM-DDTHH:MM)")

    # Config file
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")

    # WCSE overrides (CLI takes priority over config YAML)
    parser.add_argument("--wcse", action="store_true", help="Enable WCSE")
    parser.add_argument("--wcse-entry-times", type=int, default=None, help="Max TP exits per swing")
    parser.add_argument("--wcse-c1-body", type=float, default=None, help="1st candle min body pips")
    parser.add_argument("--wcse-c1-wick", type=float, default=None, help="1st candle wick proximity pips")
    parser.add_argument("--wcse-c1-close", type=float, default=None, help="1st candle close proximity pips")
    parser.add_argument("--wcse-c2-body", type=float, default=None, help="2nd candle min body pips")
    parser.add_argument("--wcse-c2-wick", type=float, default=None, help="2nd candle wick proximity pips")
    parser.add_argument("--wcse-c2-close", type=float, default=None, help="2nd candle close proximity pips")
    parser.add_argument("--wcse-tp-rr", type=float, default=None, help="TP risk:reward ratio")
    parser.add_argument("--wcse-sl-buffer", type=float, default=None, help="SL buffer pips")
    parser.add_argument("--wcse-stop-buffer", type=float, default=None, help="Stop order buffer pips above/below c2 extreme")
    parser.add_argument("--wcse-vf-skip", action="store_true", help="Skip vol filter for WCSE")
    parser.add_argument("--wcse-vf-threshold", type=float, default=None, help="Alt vol threshold")

    # WCSE2 overrides (CLI takes priority over config YAML)
    parser.add_argument("--wcse2", action="store_true", help="Enable WCSE2")
    parser.add_argument("--wcse2-entry-times", type=int, default=None, help="Max TP exits per swing")
    parser.add_argument("--wcse2-c-body", type=float, default=None, help="Pre-engulfing min body pips")
    parser.add_argument("--wcse2-c-wick", type=float, default=None, help="c1 wick proximity pips")
    parser.add_argument("--wcse2-c-close", type=float, default=None, help="c1 close proximity pips")
    parser.add_argument("--wcse2-engulf-body", type=float, default=None, help="Engulfing candle min body pips")
    parser.add_argument("--wcse2-tp-rr", type=float, default=None, help="TP risk:reward ratio")
    parser.add_argument("--wcse2-sl-buffer", type=float, default=None, help="SL buffer pips")
    parser.add_argument("--wcse2-vf-skip", action="store_true", help="Skip vol filter for WCSE2")
    parser.add_argument("--wcse2-vf-threshold", type=float, default=None, help="Alt vol threshold")

    # Output
    parser.add_argument("--label", default=None, help="Result label")


def parse_time(s: str) -> Tuple[int, int]:
    """Parse 'HH:MM' to (hour, minute)."""
    parts = s.split(":")
    return int(parts[0]), int(parts[1])


def build_params(args) -> Tuple[dict, dict, dict, dict, dict, dict, dict]:
    """Build parameter dicts from parsed args."""
    ppst_params = {
        "pivot_period": args.pivot_period,
        "atr_factor": args.atr_factor,
        "atr_period": args.atr_period,
    }

    # Load config yaml for volume filter defaults
    config = {}
    config_path = getattr(args, 'config', 'config/default.yaml')
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass
    vf_cfg = config.get('volume_filter', {})

    vol_params = {
        "use_vol_filter": not args.no_vol_filter,
        "vol_sma_period": args.vol_sma if args.vol_sma is not None else vf_cfg.get("sma_period", 20),
        "vol_threshold": args.vol_threshold if args.vol_threshold is not None else vf_cfg.get("threshold", 1.6),
        "vol_recovery_bars": args.vol_recovery,
        "filtered_exit_bars": args.filtered_exit,
    }

    qh_start = parse_time(args.quiet_start)
    qh_end = parse_time(args.quiet_end)
    quiet_params = {
        "enabled": not args.no_quiet,
        "start_hour": qh_start[0], "start_min": qh_start[1],
        "end_hour": qh_end[0], "end_min": qh_end[1],
        "tz": args.quiet_tz,
    }

    strategy_params = {
        "swing_close": args.swing_close,
        "sl_buffer_pips": args.sl_buffer,
        "tp_type": args.tp_type,
        "tp_ratio": args.tp_ratio,
        "tp_fixed_pips": args.tp_pips,
        "start_date": args.start_date,
    }

    sizing_params = {
        "initial_capital": args.capital,
        "risk_percent": args.risk,
        "max_position_units": args.max_position,
        "spread_pips": args.spread,
    }

    # Load WCSE from config yaml (reuse config loaded above), override with CLI args
    wcse_cfg = config.get('wcse', {})

    wcse_params = {
        "enabled": args.wcse if args.wcse else wcse_cfg.get("enabled", False),
        "entry_times": args.wcse_entry_times if args.wcse_entry_times is not None else wcse_cfg.get("entry_times", 0),
        "c1_body_pips": args.wcse_c1_body if args.wcse_c1_body is not None else wcse_cfg.get("c1_body_pips", 3.0),
        "c1_wick_pips": args.wcse_c1_wick if args.wcse_c1_wick is not None else wcse_cfg.get("c1_wick_pips", 0.5),
        "c1_close_pips": args.wcse_c1_close if args.wcse_c1_close is not None else wcse_cfg.get("c1_close_pips", 1.0),
        "c2_body_pips": args.wcse_c2_body if args.wcse_c2_body is not None else wcse_cfg.get("c2_body_pips", 2.0),
        "c2_wick_pips": args.wcse_c2_wick if args.wcse_c2_wick is not None else wcse_cfg.get("c2_wick_pips", 0.5),
        "c2_close_pips": args.wcse_c2_close if args.wcse_c2_close is not None else wcse_cfg.get("c2_close_pips", 1.0),
        "tp_rr": args.wcse_tp_rr if args.wcse_tp_rr is not None else wcse_cfg.get("tp_rr", 2.0),
        "sl_buffer_pips": args.wcse_sl_buffer if args.wcse_sl_buffer is not None else wcse_cfg.get("sl_buffer_pips", 0.0),
        "stop_buffer_pips": args.wcse_stop_buffer if args.wcse_stop_buffer is not None else wcse_cfg.get("stop_buffer_pips", 0.3),
        "vf_skip": args.wcse_vf_skip if args.wcse_vf_skip else wcse_cfg.get("vf_skip", False),
        "vf_threshold": args.wcse_vf_threshold if args.wcse_vf_threshold is not None else wcse_cfg.get("vf_threshold", 3.0),
    }

    # Load WCSE2 from config yaml, override with CLI args
    wcse2_cfg = config.get('wcse2', {})

    wcse2_params = {
        "enabled": args.wcse2 if args.wcse2 else wcse2_cfg.get("enabled", False),
        "entry_times": args.wcse2_entry_times if args.wcse2_entry_times is not None else wcse2_cfg.get("entry_times", 0),
        "c_body_pips": args.wcse2_c_body if args.wcse2_c_body is not None else wcse2_cfg.get("c_body_pips", 0.2),
        "c_wick_pips": args.wcse2_c_wick if args.wcse2_c_wick is not None else wcse2_cfg.get("c_wick_pips", 5.0),
        "c_close_pips": args.wcse2_c_close if args.wcse2_c_close is not None else wcse2_cfg.get("c_close_pips", 5.0),
        "engulf_body_pips": args.wcse2_engulf_body if args.wcse2_engulf_body is not None else wcse2_cfg.get("engulf_body_pips", 1.0),
        "tp_rr": args.wcse2_tp_rr if args.wcse2_tp_rr is not None else wcse2_cfg.get("tp_rr", 4.0),
        "sl_buffer_pips": args.wcse2_sl_buffer if args.wcse2_sl_buffer is not None else wcse2_cfg.get("sl_buffer_pips", 0.0),
        "vf_skip": args.wcse2_vf_skip if args.wcse2_vf_skip else wcse2_cfg.get("vf_skip", False),
        "vf_threshold": args.wcse2_vf_threshold if args.wcse2_vf_threshold is not None else wcse2_cfg.get("vf_threshold", 3.0),
    }

    return ppst_params, vol_params, quiet_params, strategy_params, sizing_params, wcse_params, wcse2_params


def main():
    parser = argparse.ArgumentParser(description="PPST Volume Filter Backtest")
    build_common_args(parser)
    args = parser.parse_args()

    ppst_params, vol_params, quiet_params, strategy_params, sizing_params, wcse_params, wcse2_params = build_params(args)

    # Load data
    print(f"Loading {args.instrument} {args.granularity} data...")
    df = load_candles(args.instrument, args.granularity)
    print(f"Loaded {len(df)} candles: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Run backtest
    print("Running vol filter backtest...")
    result = run_vol_backtest(
        df, ppst_params, vol_params, quiet_params, strategy_params, sizing_params,
        wcse_params=wcse_params,
        wcse2_params=wcse2_params,
    )

    # Generate report
    close_mode = "swing" if args.swing_close else f"tp{args.tp_type}"
    label = args.label or (
        f"vol_{args.granularity}_atr{args.atr_factor}"
        f"_sma{vol_params['vol_sma_period']}_vf{vol_params['vol_threshold']}"
        f"_sl{args.sl_buffer}_r{args.risk}"
        f"_{close_mode}"
    )
    report = generate_report(
        result["trade_log"], result["equity_curve"],
        df["time"], sizing_params["initial_capital"], label,
    )

    print_filter_stats(result["stats"], vol_params, quiet_params,
                       wcse_params=wcse_params, wcse2_params=wcse2_params)
    print_wcse2_mfe(result.get("wcse2_mfe_rrs", []))
    print_wcse_mfe(result.get("wcse_mfe_rrs", []))

    print(f"\nFiles saved to: results/")
    for name, path in report["files"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
