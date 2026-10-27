"""File-based batch runner: series files in, parquet tables out.

Everything here is pure pandas. A *batch* is a plain dict holding the data for
several series, so the same ``run_batch`` executes in a local process pool or
inside a Modal container.
"""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import ANALYSIS_COLUMNS, analyze_series
from .backtest import BACKTEST_COLUMNS, FORECAST_COLUMNS, backtest_series, forecast_series
from .regressors import REGRESSOR_COLUMNS, analyze_regressor_lags
from .schema import typed_frame
from .worker import load_model_fn, parse_analysis


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a ``.parquet`` or ``.csv`` file."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{path.suffix}' (use .parquet or .csv).")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a parquet file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_series(path: str | Path, series_id_col: str, date_col: str, value_col: str) -> pd.DataFrame:
    """Load the series file as ``series_id``, ``date``, ``value`` sorted by series and date."""
    raw = read_table(path)
    missing = [c for c in (series_id_col, date_col, value_col) if c not in raw.columns]
    if missing:
        raise ValueError(f"Series file lacks columns {missing}; found {list(raw.columns)}.")
    out = pd.DataFrame(
        {
            "series_id": raw[series_id_col].astype(str),
            "date": pd.to_datetime(raw[date_col]),
            "value": pd.to_numeric(raw[value_col]),
        }
    )
    return out.sort_values(["series_id", "date"]).reset_index(drop=True)


def load_regressors(
    path: str | Path, series_id_col: str, name_col: str, date_col: str, value_col: str
) -> pd.DataFrame:
    """Load the long regressor file as ``series_id``, ``regressor_name``, ``date``, ``value``."""
    raw = read_table(path)
    missing = [c for c in (series_id_col, name_col, date_col, value_col) if c not in raw.columns]
    if missing:
        raise ValueError(f"Regressor file lacks columns {missing}; found {list(raw.columns)}.")
    out = pd.DataFrame(
        {
            "series_id": raw[series_id_col].astype(str),
            "regressor_name": raw[name_col].astype(str),
            "date": pd.to_datetime(raw[date_col]),
            "value": pd.to_numeric(raw[value_col]),
        }
    )
    return out.sort_values(["series_id", "regressor_name", "date"]).reset_index(drop=True)


def split_batches(series_ids: list[str], batch_size: int) -> list[list[str]]:
    """Split ids into consecutive batches of at most ``batch_size``."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    return [series_ids[i : i + batch_size] for i in range(0, len(series_ids), batch_size)]


def default_batch_size(n_series: int, workers: int, per_worker_waves: int = 2) -> int:
    """Batch size giving each worker about ``per_worker_waves`` batches."""
    return max(1, math.ceil(n_series / max(1, workers * per_worker_waves)))


def build_analysis_json(analysis_row: dict[str, Any], lags: pd.DataFrame | None) -> str:
    """Serialise one analysis row plus its regressor-lag rows for a model batch."""
    row = {k: v for k, v in analysis_row.items()}
    row = {k: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for k, v in row.items()}
    row["regressor_lags"] = (
        []
        if lags is None or lags.empty
        else lags[["regressor_name", "recommended_lags"]].to_dict("records")
    )
    return json.dumps(row, default=str)


def make_batches(
    series: pd.DataFrame,
    regressors: pd.DataFrame | None,
    series_ids: list[str],
    batch_size: int,
    analysis: pd.DataFrame | None = None,
    lags: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Cut the data into self-contained batches (one dict per batch).

    Each batch maps ``series_id`` to its series frame, regressor frame and
    (when ``analysis`` is given) analysis JSON, so a worker needs nothing else.
    """
    by_series = {k: v for k, v in series.groupby("series_id")}
    by_reg = {k: v for k, v in regressors.groupby("series_id")} if regressors is not None else {}
    analysis_rows = {r["series_id"]: r for r in analysis.to_dict("records")} if analysis is not None else {}
    lag_groups = {k: v for k, v in lags.groupby("series_id")} if lags is not None and not lags.empty else {}
    batches = []
    for ids in split_batches(series_ids, batch_size):
        items = []
        for sid in ids:
            items.append(
                {
                    "series_id": sid,
                    "data": by_series[sid][["date", "value"]].reset_index(drop=True),
                    "regressors": (
                        by_reg[sid][["regressor_name", "date", "value"]].reset_index(drop=True)
                        if sid in by_reg
                        else None
                    ),
                    "analysis_json": (
                        build_analysis_json(analysis_rows[sid], lag_groups.get(sid)) if sid in analysis_rows else None
                    ),
                }
            )
        batches.append({"items": items})
    return batches


def _analysis_task(batch: dict[str, Any]) -> dict[str, pd.DataFrame]:
    spec = batch["spec"]
    analysis_rows, lag_rows = [], []
    for item in batch["items"]:
        row = analyze_series(item["data"], item["series_id"], spec["alpha"], spec.get("max_lag"))
        analysis_rows.append(row)
        if item["regressors"] is not None and not item["regressors"].empty:
            prefer_diff = row.get("stationarity_decision") == "non_stationary"
            for name, group in item["regressors"].groupby("regressor_name"):
                lag_rows.append(
                    analyze_regressor_lags(
                        item["data"],
                        group[["date", "value"]],
                        item["series_id"],
                        str(name),
                        prefer_differenced=bool(prefer_diff),
                        max_lag=spec.get("lag_max"),
                        method=spec["lag_method"],
                        min_overlap=spec["min_overlap"],
                    )
                )
    return {
        "analysis": typed_frame(analysis_rows, ANALYSIS_COLUMNS),
        "regressor_lags": typed_frame(lag_rows, REGRESSOR_COLUMNS),
    }


def _model_task(batch: dict[str, Any]) -> dict[str, pd.DataFrame]:
    spec = batch["spec"]
    forecast = spec["forecast"]
    model_fn = load_model_fn(spec["model_source"], spec["model_name"])
    frames = []
    for item in batch["items"]:
        analysis = parse_analysis(item["analysis_json"])
        if forecast:
            result = forecast_series(
                item["data"], item["regressors"], analysis, model_fn, spec["model_name"], item["series_id"],
                spec["horizon"], spec["params"],
            )
        else:
            result = backtest_series(
                item["data"], item["regressors"], analysis, model_fn, spec["model_name"], item["series_id"],
                spec["horizon"], spec["n_folds"], spec["step"], spec["params"],
            )
        frames.append(result)
    columns = FORECAST_COLUMNS if forecast else BACKTEST_COLUMNS
    rows = pd.concat(frames, ignore_index=True).to_dict("records") if frames else []
    return {"result": typed_frame(rows, columns)}


def run_batch(batch: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Execute one batch; ``batch["spec"]["task"]`` is ``analysis`` or ``model``."""
    task = batch["spec"]["task"]
    if task == "analysis":
        return _analysis_task(batch)
    if task == "model":
        return _model_task(batch)
    raise ValueError(f"Unknown task '{task}'.")


def combine(results: list[dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    """Concatenate one result table across batches."""
    frames = [r[key] for r in results if not r[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else results[0][key]
