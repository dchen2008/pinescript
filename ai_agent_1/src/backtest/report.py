"""Backtest reporting: CSV export, equity curve plot, results summary."""

from pathlib import Path

import pandas as pd
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtest.metrics import compute_metrics, compute_weekly_returns


RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def generate_report(
    trade_log: list,
    equity_curve: list,
    df_times: pd.Series,
    initial_capital: float = 10000.0,
    label: str = "backtest",
) -> dict:
    """Generate full backtest report.

    Args:
        trade_log: List of trade dicts
        equity_curve: List of equity dicts
        df_times: Full time series from data
        initial_capital: Starting capital
        label: Name prefix for output files

    Returns:
        Dict with metrics and file paths
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    metrics = compute_metrics(trade_log, equity_curve, initial_capital)

    # Weekly returns
    weekly = compute_weekly_returns(trade_log, df_times, initial_capital)

    # Save trade log CSV
    trades_path = RESULTS_DIR / f"{label}_trades.csv"
    if trade_log:
        trades_df = pd.DataFrame(trade_log)
        # Add entry_time / close_time in PT timezone
        pt = pytz.timezone("America/Los_Angeles")
        # Convert full time series to PT once
        times_pt = df_times.dt.tz_convert(pt)
        def _bar_to_pt(b):
            if pd.isna(b):
                return ""
            idx = int(b)
            if idx < 0 or idx >= len(times_pt):
                return ""
            return times_pt.iloc[idx].strftime("%Y-%m-%d %H:%M")
        trades_df.insert(0, "entry_time", trades_df["entry_bar"].map(_bar_to_pt))
        close_col_idx = trades_df.columns.get_loc("close_bar") + 1
        trades_df.insert(close_col_idx, "close_time", trades_df["close_bar"].map(_bar_to_pt))
        trades_df.to_csv(trades_path, index=False)

    # Save weekly returns CSV
    weekly_path = RESULTS_DIR / f"{label}_weekly.csv"
    if not weekly.empty:
        weekly.to_csv(weekly_path, index=False)

    # Save equity curve plot
    equity_path = RESULTS_DIR / f"{label}_equity.png"
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(eq_df["bar"], eq_df["equity"], linewidth=0.5, color="blue")
        ax.axhline(y=initial_capital, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(f"Equity Curve - {label}")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(equity_path, dpi=150)
        plt.close(fig)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS: {label}")
    print(f"{'='*60}")
    print(f"  Total Trades:    {metrics['total_trades']}")
    print(f"  Win Rate:        {metrics['win_rate']}%")
    print(f"  Profit Factor:   {metrics['profit_factor']}")
    print(f"  Net Profit:      ${metrics['net_profit']} ({metrics['net_profit_pct']}%)")
    print(f"  Max Drawdown:    ${metrics['max_drawdown']} ({metrics['max_drawdown_pct']}%)")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']}")
    print(f"  Long Trades:     {metrics['long_trades']}")
    print(f"  Short Trades:    {metrics['short_trades']}")
    print(f"  Avg Win:         ${metrics['avg_win']}")
    print(f"  Avg Loss:        ${metrics['avg_loss']}")
    print(f"  Largest Win:     ${metrics['largest_win']}")
    print(f"  Largest Loss:    ${metrics['largest_loss']}")
    print(f"  Avg Bars/Trade:  {metrics['avg_bars_in_trade']}")
    print(f"{'='*60}")

    if not weekly.empty:
        avg_weekly = weekly["return_pct"].mean()
        print(f"\n  Weekly Returns Summary:")
        print(f"  Avg Weekly Return: {avg_weekly:.2f}%")
        print(f"  Best Week:         {weekly['return_pct'].max():.2f}%")
        print(f"  Worst Week:        {weekly['return_pct'].min():.2f}%")
        print(f"  Weeks with 25%+:   {(weekly['return_pct'] >= 25).sum()}/{len(weekly)}")
        print(f"{'='*60}\n")

    return {
        "metrics": metrics,
        "weekly": weekly,
        "files": {
            "trades": str(trades_path),
            "weekly": str(weekly_path),
            "equity": str(equity_path),
        },
    }


def compare_results(results: list) -> pd.DataFrame:
    """Compare multiple backtest results in a table.

    Args:
        results: List of dicts with 'label' and 'metrics' keys

    Returns:
        DataFrame comparison table
    """
    rows = []
    for r in results:
        row = {"label": r["label"]}
        row.update(r["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save comparison
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    return df
