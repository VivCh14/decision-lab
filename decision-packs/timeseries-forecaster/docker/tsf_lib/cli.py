"""Command line entry point used by the agent tools: ``python -m tsf_lib.cli <command> --args '<json>'``.

Every command prints one JSON document to stdout and reads/writes files in
``work_dir`` (default: current directory):

    analysis.parquet, regressor_lags.parquet, backtest_<model>.parquet,
    forecast_<model>.parquet, scores.parquet, metrics.parquet, selection.parquet,
    final_forecast.parquet, inputs.json (remembered data-file settings).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import (
    DIVERGENCE_PCT_MAX,
    ESS_MIN,
    NONCONVERGED_SHARE_MAX,
    RHAT_MAX,
    score_backtest,
)
from .metrics import compute_metrics, nonconverged_folds
from .modal_runner import estimate_seconds, execute_batches, local_workers, should_run_locally
from .runner import (
    combine,
    default_batch_size,
    load_regressors,
    load_series,
    make_batches,
    read_table,
    write_table,
)
from .sampling import select_sample
from .selection import assemble_forecast, select_models

DEFAULT_LIMIT = 50
PROBE_SERIES = 2
MODAL_BATCH_TARGET = 20
INPUT_KEYS = (
    "series_file", "series_id_col", "date_col", "value_col", "regressor_file",
    "regressor_id_col", "regressor_name_col", "regressor_date_col", "regressor_value_col",
)


def _work_dir(args: dict[str, Any]) -> Path:
    return Path(args.get("work_dir", "."))


def _inputs(args: dict[str, Any]) -> dict[str, Any]:
    """Merge call arguments over the settings remembered in ``inputs.json``."""
    saved_path = _work_dir(args) / "inputs.json"
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    merged = {**saved, **{k: args[k] for k in INPUT_KEYS if k in args}}
    if "series_file" not in merged:
        raise ValueError("series_file is required (give it once; it is remembered in inputs.json).")
    return merged


def _load(inputs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    sid = inputs.get("series_id_col", "series_id")
    date = inputs.get("date_col", "date")
    series = load_series(inputs["series_file"], sid, date, inputs.get("value_col", "value"))
    regressors = None
    if inputs.get("regressor_file"):
        regressors = load_regressors(
            inputs["regressor_file"],
            inputs.get("regressor_id_col", sid),
            inputs.get("regressor_name_col", "regressor_name"),
            inputs.get("regressor_date_col", date),
            inputs.get("regressor_value_col", "value"),
        )
    return series, regressors


def _records(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    return json.loads(df.head(limit).to_json(orient="records", date_format="iso"))


def _batch_size(n_series: int, args: dict[str, Any]) -> int:
    if "batch_size" in args:
        return int(args["batch_size"])
    if should_run_locally():
        return default_batch_size(n_series, local_workers())
    return max(1, min(MODAL_BATCH_TARGET, n_series))


def cmd_inspect(args: dict[str, Any]) -> dict[str, Any]:
    inputs = _inputs(args)
    series, regressors = _load(inputs)
    out: dict[str, Any] = {
        "series": {
            "n_series": int(series["series_id"].nunique()),
            "n_rows": len(series),
            "date_min": str(series["date"].min()),
            "date_max": str(series["date"].max()),
            "head": _records(series, 5),
        }
    }
    if regressors is not None:
        out["regressors"] = {
            "names": sorted(regressors["regressor_name"].unique().tolist()),
            "n_series": int(regressors["series_id"].nunique()),
            "n_rows": len(regressors),
            "date_max": str(regressors["date"].max()),
        }
    return out


def cmd_analyze(args: dict[str, Any]) -> dict[str, Any]:
    inputs = _inputs(args)
    work = _work_dir(args)
    series, regressors = _load(inputs)
    ids = sorted(series["series_id"].unique().tolist())
    spec = {
        "task": "analysis",
        "alpha": args.get("alpha", 0.05),
        "max_lag": args.get("max_lag"),
        "lag_max": args.get("lag_max"),
        "lag_method": args.get("lag_method", "spearman"),
        "min_overlap": args.get("min_overlap", 12),
    }
    batches = make_batches(series, regressors, ids, _batch_size(len(ids), args))
    for batch in batches:
        batch["spec"] = spec
    results, backend = execute_batches(batches)
    analysis, lags = combine(results, "analysis"), combine(results, "regressor_lags")
    write_table(analysis, work / "analysis.parquet")
    write_table(lags, work / "regressor_lags.parquet")
    (work / "inputs.json").write_text(json.dumps(inputs, indent=2))
    return {"backend": backend, "n_series": len(analysis), "n_regressor_rows": len(lags), **_summary(analysis)}


def _summary(analysis: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in ("frequency", "value_kind", "demand_class", "data_quality_flag", "stationarity_decision", "seasonality_evidence"):
        if column in analysis.columns:
            out[column] = analysis[column].value_counts(dropna=False).astype(int).to_dict()
    if "quality_issues" in analysis.columns:
        issues = analysis["quality_issues"].dropna().map(json.loads).explode().dropna()
        out["quality_issue_types"] = issues.map(lambda i: i["type"]).value_counts().astype(int).to_dict()
        flagged = analysis[analysis["quality_issues"].fillna("[]") != "[]"]
        out["series_with_issues"] = flagged["series_id"].tolist()[:50]
    for column in ("n_obs", "pct_zero"):
        if column in analysis.columns:
            out[f"{column}_quantiles"] = analysis[column].quantile([0, 0.25, 0.5, 0.75, 1]).round(3).to_dict()
    return out


def cmd_summarize(args: dict[str, Any]) -> dict[str, Any]:
    analysis = read_table(_work_dir(args) / "analysis.parquet")
    return {"n_series": len(analysis), **_summary(analysis)}


def cmd_sample(args: dict[str, Any]) -> dict[str, Any]:
    analysis = read_table(_work_dir(args) / "analysis.parquet")
    return {"series_ids": select_sample(analysis, int(args["n"]), int(args.get("seed", 0)))}


def _model_batches(args: dict[str, Any], forecast: bool) -> tuple[list[dict[str, Any]], int]:
    work = _work_dir(args)
    series, regressors = _load(_inputs(args))
    analysis = read_table(work / "analysis.parquet")
    lags = read_table(work / "regressor_lags.parquet")
    ids = args.get("series_ids") or sorted(analysis["series_id"].astype(str).tolist())
    spec = {
        "task": "model",
        "forecast": forecast,
        "model_name": args["model_name"],
        "model_source": Path(args["model_file"]).read_text(),
        "horizon": int(args["horizon"]),
        "n_folds": int(args.get("n_folds", 1)),
        "step": int(args.get("step", 1)),
        "params": args.get("params", {}),
    }
    batches = make_batches(series, regressors, ids, _batch_size(len(ids), args), analysis, lags)
    for batch in batches:
        batch["spec"] = spec
    return batches, len(ids)


def cmd_run_model(args: dict[str, Any], forecast: bool) -> dict[str, Any]:
    work = _work_dir(args)
    batches, n_series = _model_batches(args, forecast)
    results, backend = execute_batches(batches)
    result = combine(results, "result")
    kind = "forecast" if forecast else "backtest"
    suffix = "_dev" if args.get("dev") else ""
    path = work / f"{kind}_{args['model_name']}{suffix}.parquet"
    write_table(result, path)
    out: dict[str, Any] = {"backend": backend, "file": str(path), "n_series": n_series, "n_rows": len(result)}
    if "error" in result.columns:
        errors = result[result["error"].notna()]
        out["n_error_rows"] = len(errors)
        out["errors"] = errors["error"].astype(str).str.slice(0, 200).value_counts().head(5).to_dict()
    return out


def cmd_estimate(args: dict[str, Any]) -> dict[str, Any]:
    batches, n_series = _model_batches(args, bool(args.get("forecast", False)))
    probe_ids = [item["series_id"] for batch in batches for item in batch["items"]][:PROBE_SERIES]
    probe_items = [item for batch in batches for item in batch["items"] if item["series_id"] in probe_ids]
    probe = {"items": probe_items, "spec": batches[0]["spec"]}
    return estimate_seconds(probe, n_series, len(batches))


def cmd_score(args: dict[str, Any]) -> dict[str, Any]:
    work = _work_dir(args)
    suffix = "_dev" if args.get("dev") else ""
    backtest = pd.concat([read_table(work / f"backtest_{m}{suffix}.parquet") for m in args["models"]], ignore_index=True)
    scores = score_backtest(backtest)
    path = work / f"scores{suffix}.parquet"
    write_table(scores, path)
    by_model = scores.groupby("model")["smape"].agg(["count", "mean", "median"]).round(3).reset_index()
    by_model["median_smape_on_median_forecast"] = scores.groupby("model")["smape_median"].median().round(3).to_numpy()
    folds = scores.groupby("model")[["n_folds", "n_nonconverged_folds", "n_failed_folds"]].sum()
    folds["nonconverged_share"] = (folds["n_nonconverged_folds"] / folds["n_folds"].clip(lower=1)).round(3)
    folds["needs_retry"] = folds["nonconverged_share"] > NONCONVERGED_SHARE_MAX
    baseline = args.get("baseline", "seasonal_naive")
    vs_baseline: list[dict[str, Any]] = []
    if baseline in set(scores["model"]):
        base = scores[scores["model"] == baseline].set_index("series_id")
        for model, group in scores[scores["model"] != baseline].groupby("model"):
            common = group.set_index("series_id").join(base, rsuffix="_base", how="inner")
            vs_baseline.append(
                {
                    "model": model,
                    "n_common_series": int(len(common)),
                    "smape_median": round(float(common["smape"].median()), 3),
                    "smape_median_baseline": round(float(common["smape_base"].median()), 3),
                    "smape_median_forecast_median": round(float(common["smape_median"].median()), 3),
                    "smape_median_forecast_median_baseline": round(float(common["smape_median_base"].median()), 3),
                }
            )
    worst = scores.sort_values("smape", ascending=False)
    return {
        "scores_file": str(path),
        "vs_baseline_same_series": vs_baseline,
        "by_model": by_model.to_dict(orient="records"),
        "folds_by_model": folds.reset_index().to_dict(orient="records"),
        "convergence_limits": {
            "rhat_max": RHAT_MAX,
            "ess_min": ESS_MIN,
            "divergence_pct_max": DIVERGENCE_PCT_MAX,
            "nonconverged_share_max": NONCONVERGED_SHARE_MAX,
        },
        "worst": _records(worst, 10),
    }


def cmd_metrics(args: dict[str, Any]) -> dict[str, Any]:
    work = _work_dir(args)
    suffix = "_dev" if args.get("dev") else ""
    backtest = pd.concat([read_table(work / f"backtest_{m}{suffix}.parquet") for m in args["models"]], ignore_index=True)
    series, _ = _load(_inputs(args))
    analysis = read_table(work / "analysis.parquet") if (work / "analysis.parquet").exists() else None
    metrics = compute_metrics(backtest, series, analysis)
    path = work / f"metrics{suffix}.parquet"
    write_table(metrics, path)
    columns = [c for c in ("wape", "bias", "mase", "coverage", "interval_score") if c in metrics.columns]
    by_model = metrics.groupby("model")[columns].median().round(4).reset_index().to_dict(orient="records")
    bad = nonconverged_folds(backtest)
    return {
        "metrics_file": str(path),
        "median_by_model": by_model,
        "nonconverged_folds_by_model": bad.groupby("model").size().to_dict(),
    }


def cmd_select(args: dict[str, Any]) -> dict[str, Any]:
    work = _work_dir(args)
    scores = read_table(work / "scores.parquet")
    analysis = read_table(work / "analysis.parquet")
    selection = select_models(scores, analysis, args.get("by", "demand_class"), args.get("metric", "smape"))
    write_table(selection, work / "selection.parquet")
    out: dict[str, Any] = {
        "selection_file": str(work / "selection.parquet"),
        "by_segment": _records(
            selection.drop_duplicates("segment")[["segment", "model", "metric", "segment_median_smape"]], 100
        ),
    }
    forecasts = {m: read_table(work / f"forecast_{m}.parquet") for m in args.get("models", []) if (work / f"forecast_{m}.parquet").exists()}
    if forecasts:
        final = assemble_forecast(forecasts, selection)
        write_table(final, work / "final_forecast.parquet")
        out["final_forecast_file"] = str(work / "final_forecast.parquet")
        out["final_forecast_rows"] = len(final)
    return out


def cmd_fetch(args: dict[str, Any]) -> dict[str, Any]:
    df = read_table(_work_dir(args) / args["file"])
    if args.get("where"):
        df = df.query(args["where"])
    if args.get("columns"):
        df = df[args["columns"]]
    limit = int(args.get("limit", DEFAULT_LIMIT))
    return {"file": args["file"], "n_rows_total": len(df), "rows": _records(df, limit)}


COMMANDS = {
    "inspect": cmd_inspect,
    "analyze": cmd_analyze,
    "summarize": cmd_summarize,
    "sample": cmd_sample,
    "estimate": cmd_estimate,
    "backtest": lambda a: cmd_run_model(a, forecast=False),
    "forecast": lambda a: cmd_run_model(a, forecast=True),
    "score": cmd_score,
    "metrics": cmd_metrics,
    "select": cmd_select,
    "fetch": cmd_fetch,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsf_lib.cli")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--args", default="{}", help="JSON object with the command arguments.")
    parsed = parser.parse_args(argv)
    print(json.dumps(COMMANDS[parsed.command](json.loads(parsed.args)), default=str, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
