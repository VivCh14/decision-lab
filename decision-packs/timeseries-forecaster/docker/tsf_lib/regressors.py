"""Regressor lag analysis and lagged-feature construction.

Regressors arrive as a long table (``series_id, regressor_name, date, value``)
of any history length. For each (series, regressor) pair this module measures
the association between the series and the regressor shifted by 0..max_lag
periods and recommends which lags, if any, are worth modeling.

Levels of trending series correlate spuriously, so correlations are computed
both on levels and on first differences; ``prefer_differenced`` (set when the
series is non-stationary) selects which one drives the recommendation.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from .analysis import AnalysisError, prepare_series
from .timegrid import FREQUENCIES, FrequencyError, classify_frequency, date_to_step

REGRESSOR_COLUMNS: dict[str, str] = {
    "series_id": "string",
    "regressor_name": "string",
    "basis": "string",
    "n_overlap": "int",
    "max_lag_used": "int",
    "best_lag": "int",
    "best_corr": "double",
    "recommended_lags": "string",
    "corr_level": "string",
    "corr_diff": "string",
    "n_by_lag": "string",
    "regressor_start": "string",
    "regressor_end": "string",
    "periods_beyond_series_end": "int",
    "recommendation": "string",
    "error_category": "string",
    "error_message": "string",
}


def _error_row(series_id: str, regressor_name: str, category: str, message: str) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in REGRESSOR_COLUMNS}
    row.update(
        series_id=str(series_id),
        regressor_name=str(regressor_name),
        recommendation="unusable",
        error_category=category,
        error_message=message,
    )
    return row


def _regressor_by_step(reg_df: pd.DataFrame, freq: str) -> pd.Series:
    clean = reg_df[["date", "value"]].dropna(subset=["date"]).copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean = clean.sort_values("date")
    if len(clean) < 3:
        raise FrequencyError(f"Regressor has only {len(clean)} observations.")
    if classify_frequency(pd.DatetimeIndex(clean["date"])) != freq:
        raise FrequencyError("Regressor frequency differs from the series frequency.")
    steps = date_to_step(pd.DatetimeIndex(clean["date"]), freq)
    if len(np.unique(steps)) != len(steps):
        raise FrequencyError("Regressor has several observations in one period.")
    series = pd.Series(clean["value"].to_numpy(dtype=float), index=steps)
    return series.reindex(np.arange(steps.min(), steps.max() + 1))


def analyze_regressor_lags(
    series_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    series_id: str,
    regressor_name: str,
    prefer_differenced: bool = False,
    max_lag: int | None = None,
    method: str = "spearman",
    min_overlap: int = 12,
    max_recommended: int = 3,
) -> dict[str, Any]:
    """Measure association between a series and lagged copies of one regressor.

    Lag ``k`` pairs the series at period ``t`` with the regressor at ``t - k``
    (the regressor leads). The regressor may start earlier or end later than
    the series; only the overlapping periods are used and ``n_by_lag`` reports
    how many pairs back each correlation.

    Parameters
    ----------
    series_df : pd.DataFrame
        Columns ``date`` and ``value`` of one series.
    reg_df : pd.DataFrame
        Columns ``date`` and ``value`` of one regressor for that series.
    series_id, regressor_name : str
        Identifiers copied into the result.
    prefer_differenced : bool
        Base the recommendation on first-difference correlations (use for
        non-stationary series).
    max_lag : int, optional
        Largest lag; defaults to the nominal season length, capped at one third
        of the series length.
    method : str
        Correlation method: ``spearman``, ``pearson`` or ``kendall``.
    min_overlap : int
        Minimum number of pairs for a lag to be evaluated.
    max_recommended : int
        Maximum number of lags to recommend.

    Returns
    -------
    dict
        One value per key of ``REGRESSOR_COLUMNS``; failures are reported in
        ``error_category`` / ``error_message`` instead of raising.
    """
    try:
        grid, freq = prepare_series(series_df)
        reg_by_step = _regressor_by_step(reg_df, freq)
    except (AnalysisError, FrequencyError) as exc:
        return _error_row(series_id, regressor_name, getattr(exc, "category", "frequency_mismatch"), str(exc))

    y = pd.Series(grid["value"].to_numpy(dtype=float), index=grid["step"].to_numpy())
    dy = y.diff()
    dreg = reg_by_step.diff()
    lag_limit = min(max_lag if max_lag is not None else FREQUENCIES[freq], max(1, int(y.notna().sum()) // 3))

    corr_level: list[float | None] = []
    corr_diff: list[float | None] = []
    n_by_lag: list[int] = []
    for lag in range(lag_limit + 1):
        shifted_index = y.index.to_numpy() - lag
        x_level = pd.Series(reg_by_step.reindex(shifted_index).to_numpy(), index=y.index)
        x_diff = pd.Series(dreg.reindex(shifted_index).to_numpy(), index=y.index)
        valid_level = y.notna() & x_level.notna()
        valid_diff = dy.notna() & x_diff.notna()
        n_by_lag.append(int(valid_level.sum()))
        for valid, out, a, b in (
            (valid_level, corr_level, y, x_level),
            (valid_diff, corr_diff, dy, x_diff),
        ):
            if valid.sum() >= min_overlap and a[valid].nunique() > 1 and b[valid].nunique() > 1:
                out.append(float(a[valid].corr(b[valid], method=method)))
            else:
                out.append(None)

    if max(n_by_lag) < min_overlap:
        return _error_row(
            series_id,
            regressor_name,
            "insufficient_overlap",
            f"At most {max(n_by_lag)} overlapping periods at any lag; need {min_overlap}.",
        )

    basis = "diff" if prefer_differenced else "level"
    chosen = corr_diff if prefer_differenced else corr_level
    bounds = [2.0 / np.sqrt(max(n, 1)) for n in n_by_lag]
    significant = [
        (lag, abs(c)) for lag, (c, b) in enumerate(zip(chosen, bounds)) if c is not None and abs(c) >= b
    ]
    significant.sort(key=lambda item: item[1], reverse=True)
    recommended = sorted(lag for lag, _ in significant[:max_recommended])
    finite = [(lag, abs(c)) for lag, c in enumerate(chosen) if c is not None]
    best_lag = max(finite, key=lambda item: item[1])[0] if finite else None

    reg_dates = pd.to_datetime(reg_df["date"])
    series_end_step = int(grid["step"].iloc[-1])
    return {
        "series_id": str(series_id),
        "regressor_name": str(regressor_name),
        "basis": basis,
        "n_overlap": int(max(n_by_lag)),
        "max_lag_used": int(lag_limit),
        "best_lag": best_lag,
        "best_corr": chosen[best_lag] if best_lag is not None else None,
        "recommended_lags": json.dumps(recommended),
        "corr_level": json.dumps(corr_level),
        "corr_diff": json.dumps(corr_diff),
        "n_by_lag": json.dumps(n_by_lag),
        "regressor_start": str(reg_dates.min().date()),
        "regressor_end": str(reg_dates.max().date()),
        "periods_beyond_series_end": int(reg_by_step.index.max() - series_end_step),
        "recommendation": "use_lags" if recommended else "no_useful_lag",
        "error_category": None,
        "error_message": None,
    }


def lagged_regressor_matrix(
    regressors: pd.DataFrame, dates: pd.DatetimeIndex, lags: dict[str, list[int]], freq: str
) -> pd.DataFrame:
    """Build lagged regressor features aligned to ``dates``.

    Parameters
    ----------
    regressors : pd.DataFrame
        Long table with columns ``regressor_name``, ``date``, ``value``,
        already truncated to what is known at the forecast origin.
    dates : pd.DatetimeIndex
        Target dates (training or future) to build features for.
    lags : dict of str to list of int
        Lags to build per regressor name.
    freq : str
        Frequency name of the series (``analysis["frequency"]``).

    Returns
    -------
    pd.DataFrame
        Indexed by ``dates`` with one column ``<name>_lag<k>`` per requested
        lag; entries are NaN where the regressor is not known (e.g. a lag
        shorter than the forecast horizon).
    """
    target_steps = date_to_step(pd.DatetimeIndex(dates), freq)
    columns: dict[str, np.ndarray] = {}
    for name, name_lags in lags.items():
        subset = regressors[regressors["regressor_name"] == name]
        by_step = pd.Series(
            subset["value"].to_numpy(dtype=float), index=date_to_step(pd.DatetimeIndex(pd.to_datetime(subset["date"])), freq)
        )
        by_step = by_step[~by_step.index.duplicated(keep="last")]
        for lag in name_lags:
            columns[f"{name}_lag{lag}"] = by_step.reindex(target_steps - lag).to_numpy()
    return pd.DataFrame(columns, index=pd.DatetimeIndex(dates))
