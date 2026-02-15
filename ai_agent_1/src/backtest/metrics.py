"""Backtest metrics: win rate, profit factor, weekly returns, drawdown, Sharpe."""

import numpy as np
import pandas as pd

from src.utils.timezone_utils import PT


def compute_metrics(trade_log: list, equity_curve: list, initial_capital: float = 10000.0) -> dict:
    """Compute comprehensive backtest metrics.

    Args:
        trade_log: List of trade dicts from BacktestEngine
        equity_curve: List of equity dicts from Portfolio
        initial_capital: Starting capital

    Returns:
        Dict with all computed metrics
    """
    if not trade_log:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "net_profit_pct": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "avg_bars_in_trade": 0.0,
        }

    trades_df = pd.DataFrame(trade_log)
    pnls = trades_df["pnl"].values

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_trades = len(pnls)
    win_count = len(wins)
    loss_count = len(losses)

    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0

    net_profit = pnls.sum()
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    # Drawdown from equity curve
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        equity = eq_df["equity"].values
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_drawdown = drawdown.max()
        # Compute percentage drawdown directly at each bar to avoid index issues
        drawdown_pct = np.where(peak > 0, (peak - equity) / peak * 100, 0.0)
        max_drawdown_pct = drawdown_pct.max()
    else:
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

    # Sharpe ratio (annualized from trade returns)
    if equity_curve and len(equity_curve) > 1:
        eq_df_sharpe = pd.DataFrame(equity_curve)
        eq_vals = eq_df_sharpe["equity"].values
        # Compute bar-to-bar returns, skip zero-equity bars
        prev = eq_vals[:-1]
        curr = eq_vals[1:]
        mask = prev > 0
        if mask.sum() > 1:
            bar_returns = (curr[mask] - prev[mask]) / prev[mask]
            sharpe = (np.mean(bar_returns) / np.std(bar_returns)) * np.sqrt(252 * 1440) if np.std(bar_returns) > 0 else 0.0
        else:
            sharpe = 0.0
    elif len(pnls) > 1:
        returns = pnls / initial_capital
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    # Trade duration
    if "entry_bar" in trades_df.columns and "close_bar" in trades_df.columns:
        durations = trades_df["close_bar"] - trades_df["entry_bar"]
        avg_bars = durations.mean()
    else:
        avg_bars = 0.0

    long_trades = (trades_df["direction"] == "LONG").sum() if "direction" in trades_df.columns else 0
    short_trades = (trades_df["direction"] == "SHORT").sum() if "direction" in trades_df.columns else 0

    return {
        "total_trades": total_trades,
        "win_trades": win_count,
        "loss_trades": loss_count,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "net_profit": round(net_profit, 2),
        "net_profit_pct": round(net_profit / initial_capital * 100, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "avg_win": round(wins.mean(), 2) if len(wins) > 0 else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) > 0 else 0.0,
        "largest_win": round(wins.max(), 2) if len(wins) > 0 else 0.0,
        "largest_loss": round(losses.min(), 2) if len(losses) > 0 else 0.0,
        "long_trades": int(long_trades),
        "short_trades": int(short_trades),
        "avg_bars_in_trade": round(avg_bars, 1),
    }


def compute_weekly_returns(
    trade_log: list,
    df_times: pd.Series,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """Compute weekly returns (Sun 2pm PT to Fri 2pm PT forex weeks).

    Args:
        trade_log: List of trade dicts
        df_times: Full time series from backtest data
        initial_capital: Starting capital

    Returns:
        DataFrame with columns: week_start, week_end, start_balance, end_balance,
        pnl, return_pct, trade_count
    """
    if not trade_log:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trade_log)

    # Convert bar indices to times
    if "entry_bar" not in trades_df.columns:
        return pd.DataFrame()

    # We use close_bar to assign trades to weeks
    trades_df["close_time"] = trades_df["close_bar"].apply(
        lambda b: df_times.iloc[min(b, len(df_times) - 1)] if b < len(df_times) else df_times.iloc[-1]
    )

    # Convert to PT for week boundaries
    close_times_utc = pd.to_datetime(trades_df["close_time"])
    if close_times_utc.dt.tz is None:
        close_times_utc = close_times_utc.dt.tz_localize("UTC")
    close_times_pt = close_times_utc.dt.tz_convert(PT)

    # Create forex week boundaries (Sunday 2pm PT)
    # Group trades by which forex week they close in
    trades_df["close_pt"] = close_times_pt

    # Find the Sunday 2pm boundary for each trade
    def get_forex_week_start(dt):
        """Get the Sunday 2pm PT that starts this forex week."""
        # If it's Sunday before 2pm, belongs to previous week
        wd = dt.weekday()  # Mon=0..Sun=6
        if wd == 6:  # Sunday
            if dt.hour < 14:
                # Previous week's Sunday
                return (dt - pd.Timedelta(days=7)).replace(hour=14, minute=0, second=0, microsecond=0)
            else:
                return dt.replace(hour=14, minute=0, second=0, microsecond=0)
        else:
            # Go back to most recent Sunday
            days_since_sunday = (wd + 1) % 7
            sunday = dt - pd.Timedelta(days=days_since_sunday)
            return sunday.replace(hour=14, minute=0, second=0, microsecond=0)

    trades_df["forex_week"] = trades_df["close_pt"].apply(get_forex_week_start)

    # Group by forex week
    weekly = []
    balance = initial_capital
    for week, group in trades_df.groupby("forex_week"):
        start_balance = balance
        week_pnl = group["pnl"].sum()
        balance += week_pnl
        weekly.append({
            "week_start": week,
            "start_balance": round(start_balance, 2),
            "end_balance": round(balance, 2),
            "pnl": round(week_pnl, 2),
            "return_pct": round(week_pnl / start_balance * 100, 2) if start_balance > 0 else 0.0,
            "trade_count": len(group),
        })

    return pd.DataFrame(weekly)
