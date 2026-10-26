"""Pytest fixtures for the timeseries-forecaster pack library."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "docker"))
sys.path.insert(0, str(PACK_DIR / "opencode" / "skills" / "timeseries-forecasting" / "references"))


@pytest.fixture
def monthly_series() -> pd.DataFrame:
    """Monthly series with trend, yearly seasonality and noise (72 points)."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2018-01-31", periods=72, freq="ME")
    t = np.arange(72)
    value = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 72)
    return pd.DataFrame({"date": dates, "value": value})


@pytest.fixture
def lagged_regressor(monthly_series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A series driven by an i.i.d. regressor two periods earlier, plus that regressor."""
    rng = np.random.default_rng(2)
    n = len(monthly_series) + 12
    dates = pd.date_range("2017-12-31", periods=n, freq="ME")
    x = rng.normal(size=n)
    y_dates = pd.date_range("2018-02-28", periods=60, freq="ME")
    y = 50 + 5 * x[: 60] + rng.normal(0, 0.5, 60)
    series = pd.DataFrame({"date": y_dates, "value": y})
    regressor = pd.DataFrame({"date": dates, "value": x})
    return series, regressor
