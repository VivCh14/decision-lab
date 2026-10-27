"""Tests for per-segment model selection and forecast assembly."""

import numpy as np
import pandas as pd
import pytest

from tsf_lib.selection import assemble_forecast, select_models


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["a", "a", "b", "b"],
            "model": ["base", "bayes", "base", "bayes"],
            "smape": [80.0, 160.0, 10.0, 5.0],
            "smape_median": [80.0, 40.0, 10.0, 6.0],
        }
    )


def _analysis() -> pd.DataFrame:
    return pd.DataFrame({"series_id": ["a", "b"], "demand_class": ["intermittent", "smooth"]})


def test_default_metric_uses_mean_forecast() -> None:
    sel = select_models(_scores(), _analysis(), "demand_class")
    chosen = dict(zip(sel["series_id"], sel["model"]))
    assert chosen == {"a": "base", "b": "bayes"}
    assert set(sel["metric"]) == {"smape"}


def test_per_segment_metric_switches_choice() -> None:
    sel = select_models(_scores(), _analysis(), "demand_class", {"intermittent": "smape_median"})
    chosen = dict(zip(sel["series_id"], sel["model"]))
    assert chosen == {"a": "bayes", "b": "bayes"}
    assert dict(zip(sel["series_id"], sel["metric"])) == {"a": "smape_median", "b": "smape"}


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        select_models(_scores(), _analysis(), "demand_class", "mase")


def test_assemble_adds_point_forecast_from_basis() -> None:
    sel = select_models(_scores(), _analysis(), "demand_class", {"intermittent": "smape_median"})
    frame = pd.DataFrame(
        {
            "series_id": ["a", "b"],
            "forecast_mean": [3.0, 7.0],
            "forecast_median": [0.0, np.nan],
        }
    )
    out = assemble_forecast({"bayes": frame}, sel).set_index("series_id")
    assert out.loc["a", "point_forecast"] == 0.0
    assert out.loc["a", "point_basis"] == "forecast_median"
    assert out.loc["b", "point_forecast"] == 7.0
    assert out.loc["b", "point_basis"] == "forecast_mean"
