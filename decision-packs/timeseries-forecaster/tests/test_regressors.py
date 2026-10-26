"""Tests for regressor lag analysis and lagged feature construction."""

import pandas as pd

from tsf_lib.regressors import analyze_regressor_lags, lagged_regressor_matrix


def test_true_lag_is_recommended(lagged_regressor: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    series, regressor = lagged_regressor
    row = analyze_regressor_lags(series, regressor, "s1", "x", max_lag=6, min_overlap=12)
    assert row["error_category"] is None
    assert row["best_lag"] == 2
    assert "2" in row["recommended_lags"]


def test_regressor_beyond_series_end_is_reported(lagged_regressor: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    series, regressor = lagged_regressor
    row = analyze_regressor_lags(series, regressor, "s1", "x", max_lag=6)
    assert row["periods_beyond_series_end"] > 0


def test_short_overlap_is_reported_not_raised(monthly_series: pd.DataFrame) -> None:
    tiny = monthly_series.tail(4).copy()
    row = analyze_regressor_lags(monthly_series, tiny, "s1", "x")
    assert row["recommendation"] == "unusable"
    assert row["error_category"] is not None


def test_lagged_matrix_uses_period_shift() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    reg = pd.DataFrame({"regressor_name": "x", "date": dates, "value": [1.0, 2, 3, 4, 5, 6]})
    out = lagged_regressor_matrix(reg, dates[2:], {"x": [0, 2]}, "monthly")
    assert out["x_lag0"].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert out["x_lag2"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_lagged_matrix_is_nan_where_unknown() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    reg = pd.DataFrame({"regressor_name": "x", "date": dates[:4], "value": [1.0, 2, 3, 4]})
    out = lagged_regressor_matrix(reg, dates[4:], {"x": [1]}, "monthly")
    assert out["x_lag1"].isna().tolist() == [False, True]
