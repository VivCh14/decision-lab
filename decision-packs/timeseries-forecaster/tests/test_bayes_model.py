"""Run the PyMC template model for real (small sampler settings)."""

import json

import numpy as np
import pandas as pd
import pytest

import bayes_fourier_regression
import intermittent_hurdle
from tsf_lib.analysis import analyze_series
from tsf_lib.backtest import backtest_series, smape
from tsf_lib.worker import parse_analysis

pytest.importorskip("pymc")
pytest.importorskip("nutpie")

PARAMS = {"draws": 100, "tune": 100, "chains": 2}


def _seasonal(kind: str = "continuous") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    t = np.arange(60)
    dates = pd.date_range("2019-01-31", periods=60, freq="ME")
    if kind == "count":
        return pd.DataFrame({"date": dates, "value": rng.poisson(np.exp(2.5 + 0.4 * np.sin(2 * np.pi * t / 12))).astype(float)})
    return pd.DataFrame({"date": dates, "value": 100 + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 60)})


@pytest.mark.slow
def test_recovers_seasonal_pattern() -> None:
    df = _seasonal()
    analysis = parse_analysis(json.dumps(analyze_series(df, "s"), default=str))
    out = bayes_fourier_regression.fit_forecast(
        df.iloc[:54], pd.DatetimeIndex(df["date"].iloc[54:]), None, analysis, PARAMS
    )
    assert len(out) == 6 and np.isfinite(out["forecast_mean"]).all()
    assert (out["forecast_lower"] <= out["forecast_median"]).all() and (out["forecast_median"] <= out["forecast_upper"]).all()
    assert smape(df["value"].iloc[54:].to_numpy(), out["forecast_mean"].to_numpy()) < 10
    assert {"rhat_max", "ess_min", "divergence_pct"} <= set(out.columns)
    assert 0.99 < out["rhat_max"].iloc[0] < 1.5 and out["ess_min"].iloc[0] > 10


@pytest.mark.slow
def test_count_series_through_backtest() -> None:
    df = _seasonal("count")
    analysis = parse_analysis(json.dumps(analyze_series(df, "c"), default=str))
    out = backtest_series(
        df, None, analysis, bayes_fourier_regression.fit_forecast, "bayes", "c", 6, 1, 6, {**PARAMS, "likelihood": "negative_binomial"}
    )
    assert out["error"].isna().all() and (out["forecast_mean"] > 0).all()


@pytest.mark.slow
def test_no_design_columns() -> None:
    rng = np.random.default_rng(2)
    dates = pd.date_range("2022-01-31", periods=8, freq="ME")
    df = pd.DataFrame({"date": dates, "value": 50 + rng.normal(0, 2, 8)})
    analysis = parse_analysis(json.dumps(analyze_series(df, "short"), default=str))
    out = bayes_fourier_regression.fit_forecast(
        df.iloc[:6], pd.DatetimeIndex(df["date"].iloc[6:]), None, {**analysis, "seasonal_periods": []}, PARAMS
    )
    assert len(out) == 2 and np.isfinite(out["forecast_mean"]).all()


@pytest.mark.slow
def test_robust_likelihood_handles_outlier_and_level_shift() -> None:
    df = _seasonal()
    df.loc[10, "value"] *= 6
    df.loc[36:, "value"] += 80
    analysis = parse_analysis(json.dumps(analyze_series(df, "q"), default=str))
    assert analysis["recommended"]["robust_likelihood"] and analysis["recommended"]["train_from_date"]
    out = bayes_fourier_regression.fit_forecast(
        df.iloc[:54], pd.DatetimeIndex(df["date"].iloc[54:]), None, analysis, PARAMS
    )
    assert np.isfinite(out["forecast_mean"]).all()
    assert smape(df["value"].iloc[54:].to_numpy(), out["forecast_mean"].to_numpy()) < 15


@pytest.mark.slow
def test_hurdle_model_on_intermittent_counts() -> None:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2019-01-31", periods=60, freq="ME")
    values = np.where(rng.random(60) < 0.25, rng.poisson(12, 60), 0).astype(float)
    df = pd.DataFrame({"date": dates, "value": values})
    analysis = parse_analysis(json.dumps(analyze_series(df, "i"), default=str))
    out = intermittent_hurdle.fit_forecast(df.iloc[:54], pd.DatetimeIndex(df["date"].iloc[54:]), None, analysis, PARAMS)
    assert len(out) == 6 and np.isfinite(out["forecast_mean"]).all()
    assert (out["forecast_median"] <= out["forecast_mean"] + 1e-9).all()
    assert (out["forecast_lower"] >= 0).all() and out["forecast_mean"].iloc[0] > 0
    assert {"rhat_max", "ess_min", "divergence_pct"} <= set(out.columns)
