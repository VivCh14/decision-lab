"""Per-group functions executed on Spark workers (pure pandas, no Spark imports).

Each ``make_*`` factory returns a closure for ``applyInPandas``. Keeping this
module free of ``pyspark`` / ``databricks.connect`` imports means workers only
need the packages the analysis itself uses.
"""

import importlib
import json
from typing import Any, Callable

import pandas as pd

from .analysis import ANALYSIS_COLUMNS, analyze_series
from .backtest import BACKTEST_COLUMNS, FORECAST_COLUMNS, backtest_series, forecast_series
from .regressors import REGRESSOR_COLUMNS, analyze_regressor_lags
from .schema import typed_frame

MODEL_PACKAGE = "tsf_models"
MODEL_ENTRYPOINT = "fit_forecast"


def load_model_fn(module_name: str) -> Callable:
    """Import ``tsf_models.<module_name>`` and return its ``fit_forecast`` function."""
    module = importlib.import_module(f"{MODEL_PACKAGE}.{module_name}")
    return getattr(module, MODEL_ENTRYPOINT)


def parse_analysis(analysis_json: str) -> dict[str, Any]:
    """Parse the per-series analysis JSON handed to models.

    ``regressor_lags`` becomes ``{regressor_name: [lags]}`` and
    ``recommended_params`` is exposed as a dict under ``recommended``.

    Parameters
    ----------
    analysis_json : str
        JSON of one analysis row, optionally with a ``regressor_lags`` list of
        ``{regressor_name, recommended_lags}`` entries.

    Returns
    -------
    dict
        The analysis row with parsed nested fields.
    """
    analysis = json.loads(analysis_json)
    analysis["recommended"] = json.loads(analysis.get("recommended_params") or "{}")
    analysis["regressor_lags"] = {
        entry["regressor_name"]: json.loads(entry["recommended_lags"] or "[]")
        for entry in (analysis.get("regressor_lags") or [])
    }
    return analysis


def make_analysis_fn(alpha: float, max_lag: int | None) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Closure running ``analyze_series`` on one ``series_id`` group."""

    def analyze_group(pdf: pd.DataFrame) -> pd.DataFrame:
        row = analyze_series(pdf[["date", "value"]], str(pdf["series_id"].iloc[0]), alpha, max_lag)
        return typed_frame([row], ANALYSIS_COLUMNS)

    return analyze_group


def make_regressor_lags_fn(
    max_lag: int | None, method: str, min_overlap: int
) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Closure running ``analyze_regressor_lags`` for every regressor of one series."""

    def lags_group(series_pdf: pd.DataFrame, reg_pdf: pd.DataFrame) -> pd.DataFrame:
        if series_pdf.empty or reg_pdf.empty:
            return typed_frame([], REGRESSOR_COLUMNS)
        series_id = str(series_pdf["series_id"].iloc[0])
        prefer_diff = series_pdf["stationarity_decision"].iloc[0] == "non_stationary"
        rows = [
            analyze_regressor_lags(
                series_pdf[["date", "value"]],
                group[["date", "value"]],
                series_id,
                str(name),
                prefer_differenced=bool(prefer_diff),
                max_lag=max_lag,
                method=method,
                min_overlap=min_overlap,
            )
            for name, group in reg_pdf.groupby("regressor_name")
        ]
        return typed_frame(rows, REGRESSOR_COLUMNS)

    return lags_group


def make_model_fn(
    module_name: str,
    model_name: str,
    horizon: int,
    n_folds: int,
    step: int,
    params: dict[str, Any],
    forecast: bool,
) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Closure running a backtest (or the final forecast) for one series."""

    def model_group(series_pdf: pd.DataFrame, reg_pdf: pd.DataFrame) -> pd.DataFrame:
        columns = FORECAST_COLUMNS if forecast else BACKTEST_COLUMNS
        if series_pdf.empty:
            return typed_frame([], columns)
        series_id = str(series_pdf["series_id"].iloc[0])
        analysis = parse_analysis(series_pdf["analysis_json"].iloc[0])
        regressors = reg_pdf[["regressor_name", "date", "value"]] if not reg_pdf.empty else None
        model_fn = load_model_fn(module_name)
        data = series_pdf[["date", "value"]]
        if forecast:
            result = forecast_series(data, regressors, analysis, model_fn, model_name, series_id, horizon, params)
        else:
            result = backtest_series(
                data, regressors, analysis, model_fn, model_name, series_id, horizon, n_folds, step, params
            )
        return typed_frame(result.to_dict("records"), columns)

    return model_group


def environment_report(pdf: pd.DataFrame) -> pd.DataFrame:
    """Report package versions on a worker (used by the connectivity check)."""
    report: dict[str, str] = {}
    for module_name in ("numpy", "pandas", "scipy", "statsmodels", "pymc", "arviz", "nutpie", "pyarrow"):
        try:
            report[module_name] = str(importlib.import_module(module_name).__version__)
        except ImportError:
            report[module_name] = "MISSING"
    return pd.DataFrame({"versions": [json.dumps(report)]})
