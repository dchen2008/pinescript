"""Time filter: quiet window and market hours.

Port of ppst_official_bt.pine lines 37-84.

Quiet window: 13:30-16:30 PT daily (no new entries, close existing)
Market hours: Sunday 2pm PT to Friday 2pm PT

PineScript dayofweek: 1=Sun, 2=Mon, ..., 7=Sat
Python weekday(): 0=Mon, 1=Tue, ..., 6=Sun
"""

import numpy as np
import pandas as pd

from src.utils.timezone_utils import utc_to_pt_series


def compute_time_filter(
    times: pd.Series,
    use_quiet_window: bool = True,
    quiet_start_hour: int = 13,
    quiet_start_min: int = 30,
    quiet_end_hour: int = 16,
    quiet_end_min: int = 30,
    use_market_window: bool = True,
    market_open_day: int = 6,     # Sunday (Python weekday)
    market_open_hour: int = 14,
    market_close_day: int = 4,    # Friday (Python weekday)
    market_close_hour: int = 14,
) -> pd.DataFrame:
    """Compute trading permission for each bar.

    Args:
        times: Series of UTC timestamps
        Other args match config/default.yaml time_filter section

    Returns:
        DataFrame with columns: can_trade, not_in_quiet, is_market_open, entering_quiet
    """
    pt = utc_to_pt_series(times)
    n = len(times)

    # Quiet window check (minute precision)
    current_time_min = pt["pt_hour"] * 60 + pt["pt_min"]
    quiet_start_min_total = quiet_start_hour * 60 + quiet_start_min
    quiet_end_min_total = quiet_end_hour * 60 + quiet_end_min

    if not use_quiet_window:
        not_in_quiet = pd.Series(np.ones(n, dtype=bool))
    elif quiet_start_min_total < quiet_end_min_total:
        # Normal range: e.g., 13:30-16:30
        not_in_quiet = (current_time_min < quiet_start_min_total) | (current_time_min >= quiet_end_min_total)
    else:
        # Wrapping range (crosses midnight)
        not_in_quiet = (current_time_min >= quiet_end_min_total) & (current_time_min < quiet_start_min_total)

    # Market hours check
    # Sunday 2pm PT to Friday 2pm PT
    # Python weekday: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    if not use_market_window:
        is_market_open = pd.Series(np.ones(n, dtype=bool))
    else:
        weekday = pt["pt_weekday"]
        hour = pt["pt_hour"]

        # Market is open from Sunday 14:00 PT to Friday 14:00 PT
        # Check each day explicitly:
        # Sunday: open if hour >= 14
        # Monday-Thursday: always open
        # Friday: open if hour < 14
        # Saturday: closed
        is_sunday_open = (weekday == market_open_day) & (hour >= market_open_hour)
        is_weekday = (weekday >= 0) & (weekday <= 3)  # Mon-Thu
        is_friday_open = (weekday == market_close_day) & (hour < market_close_hour)

        is_market_open = is_sunday_open | is_weekday | is_friday_open

    can_trade = not_in_quiet & is_market_open

    # Detect entering quiet window: transition from can_trade to in_quiet
    # "First bar where quiet window becomes active"
    in_quiet = ~not_in_quiet
    prev_not_in_quiet = pd.Series(np.ones(n, dtype=bool))
    prev_not_in_quiet.iloc[1:] = not_in_quiet.values[:-1]
    entering_quiet = in_quiet & prev_not_in_quiet

    return pd.DataFrame({
        "can_trade": can_trade,
        "not_in_quiet": not_in_quiet,
        "is_market_open": is_market_open,
        "entering_quiet": entering_quiet,
    })
