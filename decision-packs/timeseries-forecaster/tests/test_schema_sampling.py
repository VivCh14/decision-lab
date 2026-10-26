"""Tests for typed frames, column specs and stratified sampling."""

import pandas as pd

from tsf_lib.analysis import ANALYSIS_COLUMNS
from tsf_lib.backtest import BACKTEST_COLUMNS, FORECAST_COLUMNS
from tsf_lib.sampling import select_sample
from tsf_lib.schema import typed_frame


def test_typed_frame_keeps_nulls_and_dtypes() -> None:
    cols = {"a": "string", "b": "int", "c": "double", "d": "boolean", "e": "timestamp"}
    out = typed_frame([{"a": "x", "b": 1, "c": 1.5, "d": True, "e": "2020-01-01"}, {}], cols)
    assert list(out.columns) == list(cols)
    assert str(out["b"].dtype) == "Int64" and out["b"].isna().tolist() == [False, True]
    assert str(out["d"].dtype) == "boolean"
    assert out["c"].isna().tolist() == [False, True]
    assert pd.api.types.is_datetime64_any_dtype(out["e"])


def test_typed_frame_empty() -> None:
    out = typed_frame([], BACKTEST_COLUMNS)
    assert out.empty and list(out.columns) == list(BACKTEST_COLUMNS)


def test_forecast_columns_are_backtest_columns_without_actuals() -> None:
    assert set(BACKTEST_COLUMNS) - set(FORECAST_COLUMNS) == {"fold", "actual"}


def test_column_types_are_known() -> None:
    known = {"string", "int", "double", "boolean", "timestamp"}
    assert set(ANALYSIS_COLUMNS.values()) <= known


def test_sample_covers_rare_strata_first() -> None:
    analysis = pd.DataFrame(
        {
            "series_id": [f"s{i}" for i in range(20)],
            "data_quality_flag": ["ok"] * 18 + ["short"] * 2,
            "demand_class": ["smooth"] * 20,
            "value_kind": ["count"] * 20,
        }
    )
    chosen = select_sample(analysis, 4, seed=1)
    assert len(chosen) == len(set(chosen)) == 4
    assert any(s in ("s18", "s19") for s in chosen[:2])
