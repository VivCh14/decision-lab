"""Data-quality detection: outliers, level shifts, gaps and short histories."""

import json

import numpy as np
import pandas as pd

from tsf_lib.analysis import analyze_series
from tsf_lib.quality import detect_jumps, detect_quality_issues


def _grid(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2019-01-31", periods=len(values), freq="ME"), "value": values})


def _base(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)


def test_clean_series_has_no_jumps() -> None:
    assert detect_jumps(_base()) == ([], [])


def test_spike_is_outlier_not_shift() -> None:
    y = _base()
    y[20] *= 6
    outliers, shifts = detect_jumps(y)
    assert outliers == [20] and shifts == []


def test_persistent_jump_is_level_shift() -> None:
    y = _base()
    y[36:] += 80
    outliers, shifts = detect_jumps(y)
    assert shifts == [36] and outliers == []


def test_issue_list() -> None:
    y = _base(18)
    out = detect_quality_issues(_grid(y), 12, n_missing=3)
    types = {i["type"] for i in out["issues"]}
    assert types == {"gaps", "short_history"}
    assert out["level_shift_date"] is None


def test_analysis_row_carries_quality_columns() -> None:
    y = _base()
    y[36:] += 80
    y[10] *= 8
    row = analyze_series(_grid(y).assign(series_id="s"), "s")
    assert row["n_outliers"] == 1
    assert row["level_shift_date"] == "2022-01-31"
    types = {i["type"] for i in json.loads(row["quality_issues"])}
    assert types == {"outlier", "level_shift"}
    rec = json.loads(row["recommended_params"])
    assert rec["robust_likelihood"] is True and rec["train_from_date"] == "2022-01-31"


def test_trim_to_level_shift() -> None:
    from tsf_lib.quality import trim_to_level_shift

    grid = _grid(_base())
    rec = {"train_from_date": "2022-01-31", "min_train_size": 10}
    assert len(trim_to_level_shift(grid, rec)) == 24
    assert len(trim_to_level_shift(grid, {**rec, "min_train_size": 30})) == 60
    assert len(trim_to_level_shift(grid.iloc[:30], rec)) == 30
    assert len(trim_to_level_shift(grid, {})) == 60
