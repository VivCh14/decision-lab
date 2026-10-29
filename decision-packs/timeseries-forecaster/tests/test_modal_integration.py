"""Modal batch execution; requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET and a deployed ``tsf-runner`` app."""

import os
from pathlib import Path

import pandas as pd
import pytest

from tsf_lib.modal_runner import execute_batches, should_run_locally
from tsf_lib.runner import load_series, make_batches

REFERENCES = Path(__file__).resolve().parents[1] / "opencode" / "skills" / "timeseries-forecasting" / "references"
EXAMPLE = Path(__file__).resolve().parents[1] / "example-data"

pytestmark = pytest.mark.skipif(
    not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")),
    reason="Modal credentials not set",
)


@pytest.mark.slow
def test_modal_matches_local(monkeypatch: pytest.MonkeyPatch) -> None:
    series = load_series(EXAMPLE / "series.csv", "series_id", "date", "value")
    ids = sorted(series["series_id"].unique())[:2]
    spec = {
        "task": "analysis", "alpha": 0.05, "max_lag": None, "lag_max": None,
        "lag_method": "spearman", "min_overlap": 12,
    }
    batches = make_batches(series, None, ids, 1)
    for batch in batches:
        batch["spec"] = spec

    monkeypatch.setenv("DLAB_FIT_MODEL_LOCALLY", "0")
    assert not should_run_locally()
    remote, backend = execute_batches(batches)
    assert backend == "modal"

    monkeypatch.setenv("DLAB_FIT_MODEL_LOCALLY", "1")
    local, _ = execute_batches(batches, workers=1)
    pd.testing.assert_frame_equal(
        pd.concat([r["analysis"] for r in remote], ignore_index=True),
        pd.concat([r["analysis"] for r in local], ignore_index=True),
    )
