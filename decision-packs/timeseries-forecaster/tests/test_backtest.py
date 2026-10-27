"""Tests for sMAPE, rolling origins, leakage control and score aggregation."""

import json

import numpy as np
import pandas as pd
import pytest

import seasonal_naive
from tsf_lib.analysis import analyze_series
from tsf_lib.backtest import BACKTEST_COLUMNS, backtest_series, forecast_series, rolling_origins, score_backtest, smape
from tsf_lib.worker import parse_analysis


def _analysis(series: pd.DataFrame) -> dict:
    return parse_analysis(json.dumps(analyze_series(series, "s1"), default=str))


def test_smape_values() -> None:
    assert smape(np.array([100.0]), np.array([100.0])) == 0.0
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0
    assert smape(np.array([0.0]), np.array([5.0])) == 200.0
    assert smape(np.array([100.0]), np.array([50.0])) == pytest.approx(200 * 50 / 150)


def test_smape_ignores_nan_pairs() -> None:
    assert smape(np.array([1.0, np.nan]), np.array([1.0, 2.0])) == 0.0


def test_rolling_origins() -> None:
    assert rolling_origins(n=30, horizon=6, n_folds=3, step=3, min_train=10) == [18, 21, 24]
    assert rolling_origins(n=30, horizon=6, n_folds=5, step=10, min_train=10) == [14, 24]
    with pytest.raises(ValueError):
        rolling_origins(30, 0, 1, 1, 5)


def test_backtest_last_fold_ends_at_series_end(monthly_series: pd.DataFrame) -> None:
    analysis = _analysis(monthly_series)
    out = backtest_series(monthly_series, None, analysis, seasonal_naive.fit_forecast, "snaive", "s1", 6, 3, 6, {})
    assert out["error"].isna().all()
    assert out["target_date"].max() == monthly_series["date"].iloc[-1]
    assert out["fold"].nunique() == 3
    # seasonal naive over a yearly season beats a wildly wrong constant
    assert smape(out["actual"].to_numpy(), out["forecast_mean"].to_numpy()) < 30


def test_regressors_are_truncated_at_origin(monthly_series: pd.DataFrame) -> None:
    reg = pd.DataFrame({"regressor_name": "x", "date": monthly_series["date"], "value": 1.0})
    analysis = _analysis(monthly_series)
    seen: list[pd.Timestamp] = []

    def spy(train, future_dates, regressors, analysis, params):  # noqa: ANN001
        seen.append(pd.to_datetime(regressors["date"]).max() - train["date"].iloc[-1])
        return seasonal_naive.fit_forecast(train, future_dates, regressors, analysis, params)

    backtest_series(monthly_series, reg, analysis, spy, "spy", "s1", 6, 2, 6, {})
    assert all(delta <= pd.Timedelta(0) for delta in seen)


def test_model_failure_is_recorded(monthly_series: pd.DataFrame) -> None:
    def broken(train, future_dates, regressors, analysis, params):  # noqa: ANN001
        raise RuntimeError("boom")

    out = backtest_series(monthly_series, None, _analysis(monthly_series), broken, "broken", "s1", 6, 2, 6, {})
    assert out["error"].str.contains("boom").all()
    assert out["forecast_mean"].isna().all()


def test_too_short_history_gives_error_row() -> None:
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    df = pd.DataFrame({"date": dates, "value": np.arange(8, dtype=float) + 1})
    out = backtest_series(df, None, {"recommended_params": "{}"}, seasonal_naive.fit_forecast, "m", "s", 6, 2, 1, {})
    assert len(out) == 1 and out["error"].iloc[0].startswith("insufficient_history")


def test_forecast_extends_past_series_end(monthly_series: pd.DataFrame) -> None:
    out = forecast_series(monthly_series, None, _analysis(monthly_series), seasonal_naive.fit_forecast, "snaive", "s1", 12, {})
    assert len(out) == 12
    assert out["target_date"].min() > monthly_series["date"].iloc[-1]
    assert out["target_date"].is_monotonic_increasing


def test_score_backtest_counts_failed_folds(monthly_series: pd.DataFrame) -> None:
    analysis = _analysis(monthly_series)
    good = backtest_series(monthly_series, None, analysis, seasonal_naive.fit_forecast, "good", "s1", 6, 2, 6, {})

    def broken(train, future_dates, regressors, analysis, params):  # noqa: ANN001
        raise RuntimeError("boom")

    bad = backtest_series(monthly_series, None, analysis, broken, "bad", "s1", 6, 2, 6, {})
    scores = score_backtest(pd.concat([good, bad])).set_index("model")
    assert scores.loc["good", "n_failed_folds"] == 0
    assert scores.loc["bad", "n_failed_folds"] == 2
    assert np.isnan(scores.loc["bad", "smape"])


def test_median_score_falls_back_to_mean_for_point_models() -> None:
    row = {c: None for c in BACKTEST_COLUMNS}
    row.update(series_id="a", model="naive", fold=0, actual=10.0, forecast_mean=8.0, forecast_median=None)
    scores = score_backtest(pd.DataFrame([row], columns=BACKTEST_COLUMNS))
    assert scores.loc[0, "smape_median"] == scores.loc[0, "smape"]
