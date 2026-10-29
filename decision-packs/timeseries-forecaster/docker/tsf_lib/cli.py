"""Command line entry point used by the agent tools: ``python -m tsf_lib.cli <command> --args '<json>'``.

Every command prints one JSON document to stdout. Shared inputs
(``analysis.parquet``, ``regressor_lags.parquet``, ``inputs.json``) are read from
``work_dir`` (default: current directory). Outputs are written to ``output_dir``,
which defaults to a purpose-named subdirectory so a session root stays readable:

==================  ====================================================
``output_dir``      Files
==================  ====================================================
``backtests/``      ``backtest_<model>.parquet``
``forecasts/``      ``forecast_<model>.parquet``
``scoring/``        ``scores.parquet``, ``metrics.parquet``
``selection/``      ``selection.parquet``, ``final_forecast.parquet``
==================  ====================================================

Every returned path is **absolute**. Agents must be able to copy a path straight
into the next call or into ``summary.md``; when the paths were relative, one modeler
joined ``backtests/...`` to the shared ``work_dir`` instead of its own instance
directory and reported a file the orchestrator could not find.

Parallel instances pass ``output_dir`` explicitly to keep their own files.
Commands that consume model outputs take explicit file paths.
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
    implausible_rows,
    RHAT_MAX,
    score_backtest,
)
from .metrics import compute_metrics, nonconverged_folds
from .modal_runner import estimate_seconds, execute_batches, local_workers, should_run_locally
from .report import build_report
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


def _out_dir(args: dict[str, Any], default: str) -> Path:
    """Resolve the output directory, defaulting to a purpose-named subdirectory.

    Parameters
    ----------
    args : dict[str, Any]
        Call arguments; ``output_dir`` overrides the default when given.
    default : str
        Subdirectory used when the caller does not pass ``output_dir``. Keeping
        the default a named directory rather than ``"."`` is what stops a session
        from piling every parquet file into the working directory root.

    Returns
    -------
    Path
        The directory, created if missing.
    """
    path = Path(args.get("output_dir", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_backtests(args: dict[str, Any]) -> pd.DataFrame:
    files = args.get("backtest_files")
    if not files:
        raise ValueError("backtest_files is required: a list of backtest parquet paths.")
    return pd.concat([read_table(Path(f)) for f in files], ignore_index=True)


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
    out_dir = _out_dir(args, "forecasts" if forecast else "backtests")
    batches, n_series = _model_batches(args, forecast)
    results, backend = execute_batches(batches)
    result = combine(results, "result")
    kind = "forecast" if forecast else "backtest"
    suffix = "_dev" if args.get("dev") else ""
    path = out_dir / f"{kind}_{args['model_name']}{suffix}.parquet"
    write_table(result, path)
    out: dict[str, Any] = {"backend": backend, "file": str(path.resolve()), "n_series": n_series, "n_rows": len(result)}
    bad = implausible_rows(result)
    if len(bad):
        out["implausible_mean_rows"] = len(bad)
        out["implausible_mean_series"] = sorted(bad["series_id"].astype(str).unique())[:20]
    if "error" in result.columns:
        errors = result[result["error"].notna()]
        out["n_error_rows"] = len(errors)
        out["errors"] = errors["error"].astype(str).str.slice(0, 200).value_counts().head(5).to_dict()
        if len(result) and len(errors) == len(result):
            out["failed"] = (
                f"every row failed ({len(errors)}/{len(result)}): the {kind} produced no usable "
                "output. Fix the model and re-run; do not score this file."
            )
    return out


def cmd_estimate(args: dict[str, Any]) -> dict[str, Any]:
    batches, n_series = _model_batches(args, bool(args.get("forecast", False)))
    probe_ids = [item["series_id"] for batch in batches for item in batch["items"]][:PROBE_SERIES]
    probe_items = [item for batch in batches for item in batch["items"] if item["series_id"] in probe_ids]
    probe = {"items": probe_items, "spec": batches[0]["spec"]}
    return estimate_seconds(probe, n_series, len(batches))


def cmd_score(args: dict[str, Any]) -> dict[str, Any]:
    suffix = "_dev" if args.get("dev") else ""
    scores = score_backtest(_read_backtests(args))
    path = _out_dir(args, "scoring") / f"scores{suffix}.parquet"
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
        "scores_file": str(path.resolve()),
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
    backtest = _read_backtests(args)
    series, _ = _load(_inputs(args))
    analysis = read_table(work / "analysis.parquet") if (work / "analysis.parquet").exists() else None
    metrics = compute_metrics(backtest, series, analysis)
    path = _out_dir(args, "scoring") / f"metrics{suffix}.parquet"
    write_table(metrics, path)
    columns = [c for c in ("wape", "bias", "mase", "coverage", "interval_score") if c in metrics.columns]
    by_model = metrics.groupby("model")[columns].median().round(4).reset_index().to_dict(orient="records")
    bad = nonconverged_folds(backtest)
    return {
        "metrics_file": str(path.resolve()),
        "median_by_model": by_model,
        "nonconverged_folds_by_model": bad.groupby("model").size().to_dict(),
    }


def cmd_select(args: dict[str, Any]) -> dict[str, Any]:
    work, out_dir = _work_dir(args), _out_dir(args, "selection")
    scores = read_table(Path(args["scores_file"]))
    analysis = read_table(work / "analysis.parquet")
    selection = select_models(scores, analysis, args.get("by", "demand_class"), args.get("metric", "smape"))
    write_table(selection, out_dir / "selection.parquet")
    out: dict[str, Any] = {
        "selection_file": str((out_dir / "selection.parquet").resolve()),
        "by_segment": _records(
            selection.drop_duplicates("segment")[["segment", "model", "metric", "segment_median_smape"]], 100
        ),
    }
    if args.get("forecast_files"):
        frames = [read_table(Path(f)) for f in args["forecast_files"]]
        forecasts = {str(frame["model"].iloc[0]): frame for frame in frames}
        final = assemble_forecast(forecasts, selection)
        write_table(final, out_dir / "final_forecast.parquet")
        out["final_forecast_file"] = str((out_dir / "final_forecast.parquet").resolve())
        out["final_forecast_rows"] = len(final)
        bad = implausible_rows(final)
        if len(bad):
            out["implausible_mean_series"] = sorted(bad["series_id"].astype(str).unique())[:20]
    return out


def cmd_fetch(args: dict[str, Any]) -> dict[str, Any]:
    df = read_table(_work_dir(args) / args["file"])
    if args.get("where"):
        df = df.query(args["where"])
    if args.get("columns"):
        df = df[args["columns"]]
    limit = int(args.get("limit", DEFAULT_LIMIT))
    return {"file": args["file"], "n_rows_total": len(df), "rows": _records(df, limit)}


def cmd_report(args: dict[str, Any]) -> dict[str, Any]:
    """Build the numbered report figures and ``report_summary.json``."""
    work = _work_dir(args)

    def _resolve(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() or path.exists() else work / path)

    summary = build_report(
        selection_file=_resolve(args["selection_file"]),
        scores_file=_resolve(args["scores_file"]),
        metrics_file=_resolve(args["metrics_file"]),
        series_file=_resolve(args["series_file"]),
        backtest_files=[_resolve(f) for f in args.get("backtest_files", [])],
        forecast_files=[_resolve(f) for f in args.get("forecast_files", [])],
        output_dir=args.get("output_dir", "report_figures"),
        top_n=int(args.get("top_n", 3)),
        baseline=args.get("baseline", "seasonal_naive"),
    )
    # Keep the tool output small: the per-series records live in the JSON file.
    return {
        "summary_file": summary["summary_file"],
        "figures": summary["figures"],
        "segments": summary["segments"],
        "models": summary["models"],
        "n_series_beating_baseline": summary.get("n_series_beating_baseline"),
        "n_series_compared_to_baseline": summary.get("n_series_compared_to_baseline"),
        "pooled_accuracy_by_segment": summary["pooled_accuracy_by_segment"],
    }


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
    "report": cmd_report,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsf_lib.cli")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--args", default="{}", help="JSON object with the command arguments.")
    parsed = parser.parse_args(argv)
    result = COMMANDS[parsed.command](json.loads(parsed.args))
    print(json.dumps(result, default=str, indent=2))
    # A command that produced nothing usable must not look like success. The JSON
    # is still printed so the caller can read the error histogram.
    return 1 if isinstance(result, dict) and result.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
