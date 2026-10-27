"""Data-quality checks for one regularised series (outliers, level shifts, gaps, short history)."""

from typing import Any

import numpy as np
import pandas as pd

SHIFT_SIGMA = 5.0
MIN_POINTS = 8


def _robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def detect_jumps(y: np.ndarray, threshold: float = SHIFT_SIGMA) -> tuple[list[int], list[int]]:
    """Separate one-off spikes from persistent level shifts using first differences.

    A spike produces two large opposite-signed differences in a row; a level
    shift produces one large difference that is not reversed.

    Parameters
    ----------
    y : np.ndarray
        Contiguous values of the series (no NaN).
    threshold : float
        Number of robust standard deviations of the differences that counts as large.

    Returns
    -------
    tuple of (list of int, list of int)
        Indices of outlier points and of the first index after each level shift.
    """
    if len(y) < MIN_POINTS:
        return [], []
    diff = np.diff(y)
    sigma = _robust_sigma(diff)
    if sigma <= 0:
        sigma = max(float(np.std(diff)), 1e-12)
    big = np.abs(diff - np.median(diff)) > threshold * sigma
    outliers: list[int] = []
    shifts: list[int] = []
    j = 0
    while j < len(diff):
        if not big[j]:
            j += 1
            continue
        if j + 1 < len(diff) and big[j + 1] and np.sign(diff[j]) != np.sign(diff[j + 1]):
            outliers.append(j + 1)
            j += 2
        else:
            shifts.append(j + 1)
            j += 1
    return outliers, shifts


def detect_quality_issues(grid: pd.DataFrame, season_length: int, n_missing: int) -> dict[str, Any]:
    """Collect data-quality issues of one regularised series.

    Parameters
    ----------
    grid : pd.DataFrame
        Columns ``date`` and ``value`` of the contiguous analysis window.
    season_length : int
        Nominal season length of the frequency (for the short-history rule).
    n_missing : int
        Number of missing periods in the full regularised grid.

    Returns
    -------
    dict
        ``issues`` (list of ``{"type", "detail"}``), ``n_outliers``,
        ``outlier_dates`` and ``level_shift_date`` (latest shift or ``None``).
    """
    y = grid["value"].to_numpy(dtype=float)
    dates = pd.to_datetime(grid["date"]).reset_index(drop=True)
    outliers, shifts = detect_jumps(y)
    issues: list[dict[str, Any]] = []
    if outliers:
        issues.append({"type": "outlier", "detail": f"{len(outliers)} spike(s)"})
    if shifts:
        issues.append({"type": "level_shift", "detail": f"{len(shifts)} shift(s), latest at {dates[shifts[-1]].date()}"})
    if n_missing:
        issues.append({"type": "gaps", "detail": f"{n_missing} missing period(s)"})
    if len(y) < 2 * season_length + 1:
        issues.append({"type": "short_history", "detail": f"{len(y)} points < 2 seasons of {season_length}"})
    return {
        "issues": issues,
        "n_outliers": len(outliers),
        "outlier_dates": [str(dates[i].date()) for i in outliers],
        "level_shift_date": str(dates[shifts[-1]].date()) if shifts else None,
    }


def trim_to_level_shift(train: pd.DataFrame, recommended: dict[str, Any]) -> pd.DataFrame:
    """Drop history before the latest level shift when enough data remains.

    Parameters
    ----------
    train : pd.DataFrame
        Columns ``date`` and ``value``.
    recommended : dict
        The series' ``analysis["recommended"]`` (``train_from_date``,
        ``min_train_size``).

    Returns
    -------
    pd.DataFrame
        The trimmed frame, or ``train`` unchanged when there is no shift inside
        the window or the post-shift data would be shorter than ``min_train_size``.
    """
    start = recommended.get("train_from_date")
    if not start:
        return train
    kept = train[pd.to_datetime(train["date"]) >= pd.Timestamp(start)]
    return kept.reset_index(drop=True) if len(kept) >= int(recommended.get("min_train_size") or MIN_POINTS) else train
