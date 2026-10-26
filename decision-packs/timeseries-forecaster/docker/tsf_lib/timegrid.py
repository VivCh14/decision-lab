"""Uniform-frequency time grid helpers.

Series are handled on an integer "step" axis (one step per period) so that
lagging, gap detection and window arithmetic are exact for any supported
frequency, independent of pandas offset-alias quirks.
"""

import calendar

import numpy as np
import pandas as pd

FREQUENCIES: dict[str, int] = {
    "hourly": 24,
    "daily": 7,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

_MEDIAN_DAY_RANGES: list[tuple[str, float, float]] = [
    ("hourly", 0.0, 0.5),
    ("daily", 0.5, 3.0),
    ("weekly", 3.0, 14.0),
    ("monthly", 14.0, 45.0),
    ("quarterly", 45.0, 135.0),
    ("yearly", 135.0, 1000.0),
]


class FrequencyError(ValueError):
    """Raised when a frequency cannot be inferred or dates are not on a grid."""


def classify_frequency(dates: pd.DatetimeIndex) -> str:
    """Infer the sampling frequency name from the median spacing of dates.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Sorted, unique timestamps (at least 3).

    Returns
    -------
    str
        One of the keys of ``FREQUENCIES``.
    """
    if len(dates) < 3:
        raise FrequencyError(f"Need at least 3 dates to infer frequency, got {len(dates)}.")
    median_days = float(np.median(np.diff(dates.values).astype("timedelta64[s]").astype(float)) / 86400.0)
    for name, low, high in _MEDIAN_DAY_RANGES:
        if low <= median_days < high:
            return name
    raise FrequencyError(f"Median spacing of {median_days:.2f} days matches no supported frequency.")


def date_to_step(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Map timestamps to integer period indices for ``freq``.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Timestamps to convert.
    freq : str
        Frequency name from ``FREQUENCIES``.

    Returns
    -------
    np.ndarray
        Integer step per date. Consecutive periods differ by exactly 1.
    """
    if freq == "monthly":
        return (dates.year * 12 + dates.month - 1).to_numpy(dtype=np.int64)
    if freq == "quarterly":
        return (dates.year * 4 + (dates.month - 1) // 3).to_numpy(dtype=np.int64)
    if freq == "yearly":
        return np.asarray(dates.year, dtype=np.int64)
    epoch = pd.Timestamp("1970-01-01")
    seconds = ((dates - epoch) / pd.Timedelta(seconds=1)).to_numpy()
    if freq == "hourly":
        return np.floor(seconds / 3600.0).astype(np.int64)
    if freq == "daily":
        return np.floor(seconds / 86400.0).astype(np.int64)
    if freq == "weekly":
        return np.floor(seconds / (7 * 86400.0)).astype(np.int64)
    raise FrequencyError(f"Unsupported frequency '{freq}'.")


def _month_date(year: int, month: int, template: pd.Timestamp) -> pd.Timestamp:
    last_day = calendar.monthrange(year, month)[1]
    day = last_day if template.is_month_end else min(template.day, last_day)
    return pd.Timestamp(year=year, month=month, day=day, hour=template.hour)


def step_to_date(steps: np.ndarray, freq: str, template: pd.Timestamp) -> pd.DatetimeIndex:
    """Map integer steps back to timestamps, mimicking ``template``'s anchor.

    Parameters
    ----------
    steps : np.ndarray
        Integer period indices (as produced by ``date_to_step``).
    freq : str
        Frequency name from ``FREQUENCIES``.
    template : pd.Timestamp
        A real timestamp of the series; its day-of-month / weekday / hour
        anchor is reused for the generated dates.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps for each step.
    """
    steps = np.asarray(steps, dtype=np.int64)
    template_step = int(date_to_step(pd.DatetimeIndex([template]), freq)[0])
    delta = steps - template_step
    if freq == "hourly":
        return pd.DatetimeIndex([template + pd.Timedelta(hours=int(d)) for d in delta])
    if freq == "daily":
        return pd.DatetimeIndex([template + pd.Timedelta(days=int(d)) for d in delta])
    if freq == "weekly":
        return pd.DatetimeIndex([template + pd.Timedelta(weeks=int(d)) for d in delta])
    if freq == "monthly":
        return pd.DatetimeIndex([_month_date(int(s) // 12, int(s) % 12 + 1, template) for s in steps])
    if freq == "quarterly":
        offset = (template.month - 1) % 3
        return pd.DatetimeIndex(
            [_month_date(int(s) // 4, (int(s) % 4) * 3 + offset + 1, template) for s in steps]
        )
    if freq == "yearly":
        return pd.DatetimeIndex([_month_date(int(s), template.month, template) for s in steps])
    raise FrequencyError(f"Unsupported frequency '{freq}'.")


def regularize(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Reindex a (date, value) frame onto the full step grid.

    Parameters
    ----------
    df : pd.DataFrame
        Columns ``date`` (datetime64) and ``value``; sorted, unique dates.
    freq : str
        Frequency name from ``FREQUENCIES``.

    Returns
    -------
    pd.DataFrame
        Columns ``step``, ``date``, ``value`` covering every period between
        the first and last observation; missing periods have NaN values.
    """
    steps = date_to_step(pd.DatetimeIndex(df["date"]), freq)
    if len(np.unique(steps)) != len(steps):
        raise FrequencyError("Two observations fall into the same period; dates are not unique per period.")
    full_steps = np.arange(steps.min(), steps.max() + 1)
    values = pd.Series(df["value"].to_numpy(dtype=float), index=steps).reindex(full_steps)
    known_dates = pd.Series(pd.DatetimeIndex(df["date"]), index=steps)
    template = pd.Timestamp(df["date"].iloc[-1])
    dates = step_to_date(full_steps, freq, template)
    dates = pd.DatetimeIndex(
        [known_dates[s] if s in known_dates.index else d for s, d in zip(full_steps, dates)]
    )
    return pd.DataFrame({"step": full_steps, "date": dates, "value": values.to_numpy()})


def contiguous_tail(values: np.ndarray) -> int:
    """Index of the first element of the trailing run without NaN.

    Parameters
    ----------
    values : np.ndarray
        Series values, possibly containing NaN gaps.

    Returns
    -------
    int
        Start index of the longest trailing NaN-free run (0 if no NaN).
    """
    nan_positions = np.where(np.isnan(values))[0]
    return 0 if len(nan_positions) == 0 else int(nan_positions[-1]) + 1
