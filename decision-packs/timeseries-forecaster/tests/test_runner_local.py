"""File-based pipeline and local batch execution (real process pool, no Modal)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tsf_lib import cli
from tsf_lib.modal_runner import execute_batches, should_run_locally
from tsf_lib.runner import default_batch_size, load_series, make_batches, split_batches

REFERENCES = Path(__file__).resolve().parents[1] / "opencode" / "skills" / "timeseries-forecasting" / "references"
EXAMPLE = Path(__file__).resolve().parents[1] / "example-data"


@pytest.fixture(autouse=True)
def local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DLAB_FIT_MODEL_LOCALLY", "1")
    monkeypatch.setenv("DLAB_TSF_WORKERS", "2")


@pytest.fixture
def small_series(tmp_path: Path) -> Path:
    series = pd.read_csv(EXAMPLE / "series.csv")
    ids = sorted(series["series_id"].unique())[:4]
    path = tmp_path / "series.csv"
    series[series["series_id"].isin(ids)].to_csv(path, index=False)
    return path


def run(command: str, work: Path, **args: object) -> dict:
    return cli.COMMANDS[command]({"work_dir": str(work), "output_dir": str(work), **args})


def test_split_batches_covers_all_ids_once() -> None:
    ids = [str(i) for i in range(7)]
    batches = split_batches(ids, 3)
    assert [len(b) for b in batches] == [3, 3, 1]
    assert [i for b in batches for i in b] == ids
    with pytest.raises(ValueError):
        split_batches(ids, 0)


def test_default_batch_size_gives_waves() -> None:
    assert default_batch_size(100, 4) == 13
    assert default_batch_size(1, 8) == 1


def test_should_run_locally_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    assert should_run_locally()
    monkeypatch.setenv("DLAB_FIT_MODEL_LOCALLY", "0")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    assert should_run_locally()
    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "y")
    assert not should_run_locally()


def test_make_batches_self_contained(small_series: Path) -> None:
    series = load_series(small_series, "series_id", "date", "value")
    ids = sorted(series["series_id"].unique())
    batches = make_batches(series, None, ids, 3)
    assert [len(b["items"]) for b in batches] == [3, 1]
    assert all(set(i) >= {"series_id", "data", "regressors", "analysis_json"} for b in batches for i in b["items"])


def test_full_pipeline_with_process_pool(small_series: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    inspected = run("inspect", work, series_file=str(small_series))
    assert inspected["series"]["n_series"] == 4

    analysed = run("analyze", work, series_file=str(small_series), batch_size=2)
    assert analysed["backend"] == "local" and analysed["n_series"] == 4
    assert (work / "analysis.parquet").exists() and (work / "inputs.json").exists()
    assert len(pd.read_parquet(work / "analysis.parquet")) == 4

    sample = run("sample", work, n=2)
    assert len(sample["series_ids"]) == 2

    model_file = REFERENCES / "seasonal_naive.py"
    common = dict(model_file=str(model_file), model_name="snaive", horizon=6)
    estimate = run("estimate", work, **common, n_folds=2)
    assert estimate["backend"] == "local" and estimate["estimated_total_minutes"] >= 0

    backtest = run("backtest", work, **common, n_folds=2, step=3, batch_size=2)
    assert backtest["n_series"] == 4 and backtest["n_rows"] > 0
    scores = run("score", work, backtest_files=[str(work / "backtest_snaive.parquet")])
    assert scores["by_model"][0]["count"] == 4

    forecast = run("forecast", work, **common)
    assert (work / "forecast_snaive.parquet").exists() and forecast["n_rows"] == 4 * 6

    selected = run("select", work, scores_file=str(work / "scores.parquet"), forecast_files=[str(work / "forecast_snaive.parquet")], by="demand_class")
    assert selected["final_forecast_rows"] == 4 * 6
    fetched = run("fetch", work, file="scores.parquet", limit=2)
    assert len(fetched["rows"]) == 2


def test_pool_matches_serial(small_series: Path, tmp_path: Path) -> None:
    run("analyze", tmp_path / "w", series_file=str(small_series))
    common = dict(model_file=str(REFERENCES / "seasonal_naive.py"), model_name="snaive", horizon=3)
    batches, _ = cli._model_batches({"work_dir": str(tmp_path / "w"), **common}, forecast=True)
    one, _ = execute_batches(batches, workers=1)
    two, _ = execute_batches(batches, workers=2)
    pd.testing.assert_frame_equal(
        pd.concat([r["result"] for r in one], ignore_index=True),
        pd.concat([r["result"] for r in two], ignore_index=True),
    )


def test_missing_series_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run("inspect", tmp_path)


def test_regressors_flow_into_analysis_json(small_series: Path, tmp_path: Path) -> None:
    ids = pd.read_csv(small_series)["series_id"].unique()
    regs = pd.read_csv(EXAMPLE / "regressors.csv")
    regs_path = tmp_path / "regressors.csv"
    regs[regs["series_id"].isin(ids)].to_csv(regs_path, index=False)
    work = tmp_path / "work"
    result = run("analyze", work, series_file=str(small_series), regressor_file=str(regs_path))
    assert result["n_regressor_rows"] > 0
    lags = pd.read_parquet(work / "regressor_lags.parquet")
    batches, _ = cli._model_batches(
        {"work_dir": str(work), "model_file": str(REFERENCES / "seasonal_naive.py"), "model_name": "snaive", "horizon": 3},
        forecast=True,
    )
    payload = json.loads(batches[0]["items"][0]["analysis_json"])
    assert {e["regressor_name"] for e in payload["regressor_lags"]} <= set(lags["regressor_name"])
    assert payload["regressor_lags"]


def test_cli_returns_absolute_paths(tmp_path, monkeypatch):
    """Every path a tool hands back must be absolute.

    Agents copy these straight into the next call or into summary.md. When they
    were relative, one modeler joined ``backtests/...`` to the shared work_dir
    instead of its own instance directory and reported a file the orchestrator
    could not find, so a clean plan looked like a failure.
    """
    import json
    import pandas as pd
    from tsf_lib import cli
    from tsf_lib.analysis import analyze_series
    from tsf_lib.regressors import REGRESSOR_COLUMNS

    def build_analysis(frame):
        rows = [analyze_series(g, sid) for sid, g in frame.groupby("series_id")]
        return pd.DataFrame(rows)

    dates = pd.date_range("2020-01-31", periods=40, freq="ME")
    series = pd.DataFrame([
        {"series_id": sid, "date": d, "value": 100.0 + i}
        for sid in ("s0", "s1") for i, d in enumerate(dates)
    ])
    series_file = tmp_path / "series.parquet"
    series.to_parquet(series_file, index=False)

    model = tmp_path / "m.py"
    model.write_text(
        "import pandas as pd\n"
        "def fit_forecast(train, future_dates, regressors, analysis, params):\n"
        "    last = float(train['value'].iloc[-1])\n"
        "    return pd.DataFrame({'forecast_mean': [last] * len(future_dates)})\n"
    )

    # cmd_run_model reads these from work_dir (the cwd here).
    analysis = build_analysis(series)
    analysis.to_parquet(tmp_path / "analysis.parquet", index=False)
    pd.DataFrame(columns=list(REGRESSOR_COLUMNS)).to_parquet(
        tmp_path / "regressor_lags.parquet", index=False
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DLAB_FIT_MODEL_LOCALLY", "1")
    monkeypatch.setenv("DLAB_TSF_WORKERS", "1")

    args = {
        "series_file": str(series_file), "series_id_col": "series_id",
        "date_col": "date", "value_col": "value",
        "model_file": str(model), "model_name": "m",
        "horizon": 3, "n_folds": 2, "step": 1,
    }
    backtest = cli.cmd_run_model(dict(args), forecast=False)
    assert Path(backtest["file"]).is_absolute(), backtest["file"]
    assert Path(backtest["file"]).exists()
    # Default lands in backtests/, not the working directory root.
    assert Path(backtest["file"]).parent.name == "backtests"

    scored = cli.cmd_score({"backtest_files": [backtest["file"]], "baseline": "m"})
    assert Path(scored["scores_file"]).is_absolute(), scored["scores_file"]
    assert Path(scored["scores_file"]).parent.name == "scoring"

    forecast = cli.cmd_run_model(dict(args), forecast=True)
    assert Path(forecast["file"]).is_absolute(), forecast["file"]
    assert Path(forecast["file"]).parent.name == "forecasts"

    # The JSON the tool actually prints must carry the absolute path too.
    assert str(tmp_path) in json.dumps(backtest, default=str)
