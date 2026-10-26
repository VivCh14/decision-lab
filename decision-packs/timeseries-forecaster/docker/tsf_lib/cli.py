"""Command line entry point used by the agent tools: ``python -m tsf_lib.cli <command> --args '<json>'``.

Every command prints one JSON document to stdout. Table-valued results are
returned as a bounded number of rows (``limit``) so output stays readable.
"""

import argparse
import json
import sys
from typing import Any

import pandas as pd

from .sampling import select_sample
from .spark_batch import (
    check_environment,
    get_spark,
    read_regressors,
    read_series,
    run_analysis,
    run_model,
    score_tables,
    table_name,
)

DEFAULT_LIMIT = 50


def _inputs(spark: Any, args: dict[str, Any]) -> tuple[Any, Any]:
    series = read_series(
        spark, args["series_table"], args.get("series_id_col", "series_id"), args.get("date_col", "date"), args.get("value_col", "value")
    )
    regressors = None
    if args.get("regressor_table"):
        regressors = read_regressors(
            spark,
            args["regressor_table"],
            args.get("regressor_id_col", args.get("series_id_col", "series_id")),
            args.get("regressor_name_col", "regressor_name"),
            args.get("regressor_date_col", args.get("date_col", "date")),
            args.get("regressor_value_col", "value"),
        )
    return series, regressors


def _records(pdf: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    return json.loads(pdf.head(limit).to_json(orient="records", date_format="iso"))


def cmd_check(args: dict[str, Any]) -> dict[str, Any]:
    return check_environment(get_spark())


def cmd_inspect(args: dict[str, Any]) -> dict[str, Any]:
    spark = get_spark()
    series, regressors = _inputs(spark, args)
    out: dict[str, Any] = {
        "series": {
            "n_series": series.select("series_id").distinct().count(),
            "n_rows": series.count(),
            "date_min": str(series.agg({"date": "min"}).collect()[0][0]),
            "date_max": str(series.agg({"date": "max"}).collect()[0][0]),
            "head": _records(series.limit(5).toPandas(), 5),
        }
    }
    if regressors is not None:
        out["regressors"] = {
            "names": [r[0] for r in regressors.select("regressor_name").distinct().collect()],
            "n_series": regressors.select("series_id").distinct().count(),
            "n_rows": regressors.count(),
        }
    return out


def cmd_analyze(args: dict[str, Any]) -> dict[str, Any]:
    spark = get_spark()
    series, regressors = _inputs(spark, args)
    return run_analysis(
        spark,
        series,
        regressors,
        args["output_schema"],
        args["run_id"],
        alpha=args.get("alpha", 0.05),
        max_lag=args.get("max_lag"),
        lag_max=args.get("lag_max"),
        lag_method=args.get("lag_method", "spearman"),
        min_overlap=args.get("min_overlap", 12),
    )


def cmd_fetch(args: dict[str, Any]) -> dict[str, Any]:
    spark = get_spark()
    table = table_name(args["output_schema"], args["run_id"], args["kind"], args.get("model"), args.get("dev", False))
    df = spark.table(table)
    if args.get("where"):
        df = df.where(args["where"])
    if args.get("columns"):
        df = df.select(*args["columns"])
    limit = int(args.get("limit", DEFAULT_LIMIT))
    return {"table": table, "rows": _records(df.limit(limit).toPandas(), limit)}


def cmd_summarize(args: dict[str, Any]) -> dict[str, Any]:
    spark = get_spark()
    analysis = spark.table(table_name(args["output_schema"], args["run_id"], "analysis")).toPandas()
    out: dict[str, Any] = {"n_series": len(analysis)}
    for column in ("frequency", "value_kind", "demand_class", "data_quality_flag", "stationarity_decision", "seasonality_evidence"):
        if column in analysis.columns:
            out[column] = analysis[column].value_counts(dropna=False).astype(int).to_dict()
    for column in ("n_obs", "pct_zero"):
        if column in analysis.columns:
            out[f"{column}_quantiles"] = analysis[column].quantile([0, 0.25, 0.5, 0.75, 1]).round(3).to_dict()
    return out


def cmd_sample(args: dict[str, Any]) -> dict[str, Any]:
    spark = get_spark()
    analysis = spark.table(table_name(args["output_schema"], args["run_id"], "analysis")).toPandas()
    return {"series_ids": select_sample(analysis, int(args["n"]), int(args.get("seed", 0)))}


def cmd_run_model(args: dict[str, Any], forecast: bool) -> dict[str, Any]:
    spark = get_spark()
    series, regressors = _inputs(spark, args)
    return run_model(
        spark,
        series,
        regressors,
        args["output_schema"],
        args["run_id"],
        args["model_file"],
        args["model_name"],
        int(args["horizon"]),
        int(args.get("n_folds", 1)),
        int(args.get("step", 1)),
        args.get("params", {}),
        forecast,
        series_ids=args.get("series_ids"),
        dev=bool(args.get("dev", False)),
    )


def cmd_score(args: dict[str, Any]) -> dict[str, Any]:
    spark = get_spark()
    dev = bool(args.get("dev", False))
    tables = [table_name(args["output_schema"], args["run_id"], "backtest", m, dev) for m in args["models"]]
    output = table_name(args["output_schema"], args["run_id"], "scores", dev=dev)
    scores = score_tables(spark, tables, output).toPandas()
    summary = (
        scores.groupby("model")["smape"]
        .agg(["count", "mean", "median"])
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )
    return {"scores_table": output, "by_model": summary, "worst": _records(scores.sort_values("smape", ascending=False), 10)}


COMMANDS = {
    "check": cmd_check,
    "inspect": cmd_inspect,
    "analyze": cmd_analyze,
    "summarize": cmd_summarize,
    "fetch": cmd_fetch,
    "sample": cmd_sample,
    "backtest": lambda a: cmd_run_model(a, forecast=False),
    "forecast": lambda a: cmd_run_model(a, forecast=True),
    "score": cmd_score,
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
