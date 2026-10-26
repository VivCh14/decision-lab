"""Rolling-origin backtesting and forecasting for arbitrary model functions.

A model is any function with the signature::

    fit_forecast(train, future_dates, regressors, analysis, params) -> pd.DataFrame

where ``train`` has columns ``date`` and ``value`` (uniform grid, no gaps),
``future_dates`` is a ``pd.DatetimeIndex`` of the ``horizon`` target dates,
``regressors`` is a long table (``regressor_name, date, value``) truncated to
dates not after the forecast origin (or ``None``), ``analysis`` is the
series' analysis row (dict) and ``params`` is a free-form dict. It must return
a frame with ``horizon`` rows and a ``forecast_mean`` column, plus optional
``forecast_median``, ``forecast_lower`` and ``forecast_upper``.

The output tables hold forecasts next to actuals (backtest) so any metric can
be computed from them afterwards; ``smape`` is provided as the default.
"""

import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from .analysis import prepare_series
from .timegrid import contiguous_tail, step_to_date

ModelFn = Callable[[pd.DataFrame, pd.DatetimeIndex, "pd.DataFrame | None", dict, dict], pd.DataFrame]

BACKTEST_COLUMNS: dict[str, str] = {
    "series_id": "string",
    "model": "string",
    "fold": "int",
    "origin_date": "timestamp",
    "target_date": "timestamp",
    "horizon_step": "int",
    "actual": "double",
    "forecast_mean": "double",
    "forecast_median": "double",
    "forecast_lower": "double",
    "forecast_upper": "double",
    "interval_level": "double",
    "error": "string",
}

FORECAST_COLUMNS: dict[str, str] = {
    name: dtype for name, dtype in BACKTEST_COLUMNS.items() if name not in ("fold", "actual")
}

_FORECAST_FIELDS = ("forecast_mean", "forecast_median", "forecast_lower", "forecast_upper")


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE in percent, ``mean(200 * |a - f| / (|a| + |f|))``.

    Periods where both actual and forecast are zero count as a perfect (0)
    error; periods with a zero actual and a non-zero forecast score the maximum
    of 200, so sMAPE is harsh on intermittent series.

    Parameters
    ----------
    actual, forecast : np.ndarray
        Arrays of equal length.

    Returns
    -------
    float
        sMAPE in [0, 200]; NaN pairs are ignored.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(forecast)
    a, f = actual[valid], forecast[valid]
    denom = np.abs(a) + np.abs(f)
    terms = np.where(denom == 0, 0.0, 200.0 * np.abs(a - f) / np.where(denom == 0, 1.0, denom))
    return float(terms.mean()) if len(terms) else float("nan")


def rolling_origins(n: int, horizon: int, n_folds: int, step: int, min_train: int) -> list[int]:
    """Training-window sizes for a rolling-origin backtest, oldest first.

    The last fold's test window ends at the final observation; earlier origins
    are ``step`` periods apart. Origins that leave fewer than ``min_train``
    training points are dropped.

    Parameters
    ----------
    n : int
        Number of observations.
    horizon : int
        Forecast horizon per fold.
    n_folds : int
        Maximum number of folds.
    step : int
        Periods between consecutive origins.
    min_train : int
        Smallest allowed training size.

    Returns
    -------
    list of int
        Training sizes (origin = first ``size`` observations).
    """
    if horizon < 1 or n_folds < 1 or step < 1:
        raise ValueError("horizon, n_folds and step must all be >= 1.")
    sizes = [n - horizon - i * step for i in range(n_folds)]
    return sorted(size for size in sizes if size >= min_train)


def _validate_forecast(result: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if "forecast_mean" not in result.columns:
        raise ValueError("Model output must contain a 'forecast_mean' column.")
    if len(result) != horizon:
        raise ValueError(f"Model returned {len(result)} rows for horizon {horizon}.")
    out = pd.DataFrame(index=range(horizon))
    for name in _FORECAST_FIELDS:
        out[name] = result[name].to_numpy(dtype=float) if name in result.columns else np.nan
    return out


def _error_frame(
    series_id: str, model_name: str, message: str, backtest: bool, interval_level: float
) -> pd.DataFrame:
    columns = BACKTEST_COLUMNS if backtest else FORECAST_COLUMNS
    row: dict[str, Any] = {name: None for name in columns}
    row.update(series_id=str(series_id), model=model_name, interval_level=interval_level, error=message)
    return pd.DataFrame([row])


def _run_model(
    model_fn: ModelFn,
    train: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    regressors: pd.DataFrame | None,
    analysis: dict,
    params: dict,
) -> tuple[pd.DataFrame | None, str | None]:
    origin = train["date"].iloc[-1]
    known = None if regressors is None else regressors[pd.to_datetime(regressors["date"]) <= origin]
    try:
        return _validate_forecast(model_fn(train, future_dates, known, analysis, params), len(future_dates)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def backtest_series(
    series_df: pd.DataFrame,
    regressors: pd.DataFrame | None,
    analysis: dict,
    model_fn: ModelFn,
    model_name: str,
    series_id: str,
    horizon: int,
    n_folds: int,
    step: int,
    params: dict | None = None,
    min_train: int | None = None,
) -> pd.DataFrame:
    """Rolling-origin backtest of one series.

    Each fold trains on data up to its origin only (regressors are truncated
    the same way, so lags shorter than the horizon are NaN in the future).
    Model failures are recorded in the ``error`` column of that fold's rows
    rather than aborting the series.

    Parameters
    ----------
    series_df : pd.DataFrame
        Columns ``date`` and ``value`` of the series.
    regressors : pd.DataFrame or None
        Long table of this series' regressors.
    analysis : dict
        The series' analysis row (``recommended_params`` already parsed is not
        required; the raw row is passed through to the model).
    model_fn : callable
        Model function; see the module docstring.
    model_name, series_id : str
        Identifiers copied into the result.
    horizon, n_folds, step : int
        Backtest geometry.
    params : dict, optional
        Free-form model parameters; ``interval_level`` (default 0.8) is
        recorded in the result.
    min_train : int, optional
        Smallest training window; defaults to the analysis recommendation.

    Returns
    -------
    pd.DataFrame
        One row per fold and horizon step with columns ``BACKTEST_COLUMNS``.
    """
    params = params or {}
    interval_level = float(params.get("interval_level", 0.8))
    grid, _ = prepare_series(series_df)
    window = grid.iloc[contiguous_tail(grid["value"].to_numpy(dtype=float)) :].reset_index(drop=True)
    if min_train is None:
        recommended = json.loads(analysis.get("recommended_params") or "{}")
        min_train = int(recommended.get("min_train_size") or 10)

    origins = rolling_origins(len(window), horizon, n_folds, step, min_train)
    if not origins:
        return _error_frame(
            series_id,
            model_name,
            f"insufficient_history: {len(window)} usable points for horizon={horizon}, min_train={min_train}",
            True,
            interval_level,
        )

    frames = []
    for fold, size in enumerate(origins):
        train = window.iloc[:size][["date", "value"]].reset_index(drop=True)
        test = window.iloc[size : size + horizon]
        forecast, error = _run_model(model_fn, train, pd.DatetimeIndex(test["date"]), regressors, analysis, params)
        frame = pd.DataFrame(
            {
                "series_id": str(series_id),
                "model": model_name,
                "fold": fold,
                "origin_date": train["date"].iloc[-1],
                "target_date": test["date"].to_numpy(),
                "horizon_step": np.arange(1, horizon + 1),
                "actual": test["value"].to_numpy(dtype=float),
                "interval_level": interval_level,
                "error": error,
            }
        )
        for name in _FORECAST_FIELDS:
            frame[name] = forecast[name].to_numpy() if forecast is not None else np.nan
        frames.append(frame[list(BACKTEST_COLUMNS)])
    return pd.concat(frames, ignore_index=True)


def forecast_series(
    series_df: pd.DataFrame,
    regressors: pd.DataFrame | None,
    analysis: dict,
    model_fn: ModelFn,
    model_name: str,
    series_id: str,
    horizon: int,
    params: dict | None = None,
) -> pd.DataFrame:
    """Fit on the full series and forecast ``horizon`` periods past its end.

    Parameters
    ----------
    series_df, regressors, analysis, model_fn, model_name, series_id, horizon, params
        As in ``backtest_series``.

    Returns
    -------
    pd.DataFrame
        One row per horizon step with columns ``FORECAST_COLUMNS``.
    """
    params = params or {}
    interval_level = float(params.get("interval_level", 0.8))
    grid, freq = prepare_series(series_df)
    window = grid.iloc[contiguous_tail(grid["value"].to_numpy(dtype=float)) :].reset_index(drop=True)
    train = window[["date", "value"]]
    last_step = int(window["step"].iloc[-1])
    future_dates = step_to_date(np.arange(last_step + 1, last_step + 1 + horizon), freq, train["date"].iloc[-1])

    forecast, error = _run_model(model_fn, train, future_dates, regressors, analysis, params)
    if forecast is None:
        return _error_frame(series_id, model_name, str(error), False, interval_level)
    frame = pd.DataFrame(
        {
            "series_id": str(series_id),
            "model": model_name,
            "origin_date": train["date"].iloc[-1],
            "target_date": future_dates,
            "horizon_step": np.arange(1, horizon + 1),
            "interval_level": interval_level,
            "error": None,
        }
    )
    for name in _FORECAST_FIELDS:
        frame[name] = forecast[name].to_numpy()
    return frame[list(FORECAST_COLUMNS)]


def score_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a backtest table into sMAPE per series and model.

    Parameters
    ----------
    backtest : pd.DataFrame
        A table with ``BACKTEST_COLUMNS``, possibly stacked over many models.

    Returns
    -------
    pd.DataFrame
        Columns ``series_id``, ``model``, ``smape``, ``n_points``,
        ``n_failed_folds``.
    """
    rows = []
    for (series_id, model), group in backtest.groupby(["series_id", "model"], dropna=False):
        scored = group.dropna(subset=["actual", "forecast_mean"])
        rows.append(
            {
                "series_id": series_id,
                "model": model,
                "smape": smape(scored["actual"].to_numpy(), scored["forecast_mean"].to_numpy()),
                "n_points": int(len(scored)),
                "n_failed_folds": int(group.loc[group["error"].notna(), "fold"].nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)
