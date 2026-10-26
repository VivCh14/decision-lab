"""Run the PyMC template model for real (small sampler settings)."""

import json

import numpy as np
import pandas as pd
import pytest

import bayes_fourier_regression
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


@pytest.mark.slow
def test_count_series_through_backtest() -> None:
    df = _seasonal("count")
    analysis = parse_analysis(json.dumps(analyze_series(df, "c"), default=str))
    out = backtest_series(
        df, None, analysis, bayes_fourier_regression.fit_forecast, "bayes", "c", 6, 1, 6, {**PARAMS, "likelihood": "negative_binomial"}
    )
    assert out["error"].isna().all() and (out["forecast_mean"] > 0).all()
