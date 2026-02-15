"""PT timezone helpers for forex market hours."""

import pytz
from datetime import datetime
import pandas as pd

PT = pytz.timezone("America/Los_Angeles")


def to_pt(dt: datetime) -> datetime:
    """Convert a datetime to Pacific Time."""
    if dt.tzinfo is None:
        # Assume UTC if naive
        dt = pytz.utc.localize(dt)
    return dt.astimezone(PT)


def utc_to_pt_series(times: pd.Series) -> pd.DataFrame:
    """Convert a Series of UTC timestamps to PT components.

    Returns DataFrame with columns: pt_hour, pt_min, pt_weekday (Mon=0..Sun=6)
    """
    # Ensure UTC
    if times.dt.tz is None:
        utc_times = times.dt.tz_localize("UTC")
    else:
        utc_times = times.dt.tz_convert("UTC")

    pt_times = utc_times.dt.tz_convert(PT)
    return pd.DataFrame({
        "pt_hour": pt_times.dt.hour,
        "pt_min": pt_times.dt.minute,
        "pt_weekday": pt_times.dt.weekday,  # Mon=0..Sun=6
    })
