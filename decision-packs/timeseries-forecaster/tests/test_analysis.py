"""Tests for the per-series analysis and its column contract."""

import numpy as np
import pandas as pd

from tsf_lib.analysis import ANALYSIS_COLUMNS, analyze_series


def test_row_matches_column_spec(monthly_series: pd.DataFrame) -> None:
    row = analyze_series(monthly_series, "s1")
    assert list(row) == list(ANALYSIS_COLUMNS) or set(row) == set(ANALYSIS_COLUMNS)
    assert row["series_id"] == "s1"
    assert row["frequency"] == "monthly"
    assert row["n_obs"] == 72


def test_yearly_season_is_found(monthly_series: pd.DataFrame) -> None:
    row = analyze_series(monthly_series, "s1")
    assert '"season_length": 12' in row["recommended_params"]


def test_value_semantics_count_vs_continuous() -> None:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2019-01-31", periods=60, freq="ME")
    counts = pd.DataFrame({"date": dates, "value": rng.poisson(5, 60).astype(float)})
    continuous = pd.DataFrame({"date": dates, "value": rng.gamma(2.0, 3.0, 60)})
    assert analyze_series(counts, "c")["value_kind"] == "count"
    assert analyze_series(continuous, "x")["value_kind"] == "nonnegative_continuous"


def test_intermittent_series_is_classified() -> None:
    rng = np.random.default_rng(4)
    dates = pd.date_range("2019-01-31", periods=60, freq="ME")
    values = np.where(rng.random(60) < 0.2, rng.integers(1, 4, 60), 0).astype(float)
    row = analyze_series(pd.DataFrame({"date": dates, "value": values}), "i")
    assert row["demand_class"] in ("intermittent", "lumpy")
    assert row["pct_zero"] > 0.5


def test_short_series_does_not_raise() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    row = analyze_series(pd.DataFrame({"date": dates, "value": [1.0, 2.0, 3.0, 4.0]}), "short")
    assert row["series_id"] == "short"
    assert set(row) == set(ANALYSIS_COLUMNS)


def test_seasonal_period_is_not_off_by_one() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(60)
    df = pd.DataFrame(
        {"date": pd.date_range("2019-01-31", periods=60, freq="ME"), "value": 100 + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 60)}
    )
    row = analyze_series(df, "s")
    assert '"season_length": 12' in row["recommended_params"]
