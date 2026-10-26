"""Tests for frequency inference and the integer period grid."""

import numpy as np
import pandas as pd
import pytest

from tsf_lib.timegrid import FrequencyError, classify_frequency, contiguous_tail, date_to_step, regularize, step_to_date


@pytest.mark.parametrize(
    "freq, expected",
    [("D", "daily"), ("W", "weekly"), ("ME", "monthly"), ("QE", "quarterly"), ("YE", "yearly"), ("h", "hourly")],
)
def test_classify_frequency(freq: str, expected: str) -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq=freq)
    assert classify_frequency(dates) == expected


def test_step_roundtrip_monthly() -> None:
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    steps = date_to_step(dates, "monthly")
    assert np.all(np.diff(steps) == 1)
    back = step_to_date(steps, "monthly", dates[-1])
    assert list(back) == list(dates)


def test_regularize_fills_gap_with_nan() -> None:
    dates = pd.date_range("2020-01-31", periods=12, freq="ME").delete(5)
    df = pd.DataFrame({"date": dates, "value": np.arange(11, dtype=float)})
    grid = regularize(df, "monthly")
    assert len(grid) == 12
    assert grid["value"].isna().sum() == 1


def test_contiguous_tail_skips_earlier_gap() -> None:
    values = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    assert contiguous_tail(values) == 2


def test_too_few_dates_raise() -> None:
    with pytest.raises(FrequencyError):
        classify_frequency(pd.DatetimeIndex(["2020-01-01", "2020-01-02"]))


def test_unsupported_spacing_raises() -> None:
    dates = pd.date_range("1990-01-01", periods=5, freq="5YE")
    with pytest.raises(FrequencyError):
        classify_frequency(dates)
