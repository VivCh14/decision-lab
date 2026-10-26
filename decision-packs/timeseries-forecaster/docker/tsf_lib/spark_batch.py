"""Databricks execution layer: connection, code shipping and batch runs.

All heavy work runs on a Databricks cluster through ``databricks-connect`` and
Spark ``applyInPandas``; inputs and outputs are Unity Catalog tables.

Environment: ``DATABRICKS_CLUSTER_ID`` plus either ``DATABRICKS_HOST`` and
``DATABRICKS_TOKEN`` (containers) or a ``~/.databrickscfg`` profile selected
with ``DATABRICKS_CONFIG_PROFILE`` (local runs, supports OAuth).
"""

import hashlib
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from .analysis import ANALYSIS_COLUMNS
from .backtest import BACKTEST_COLUMNS, FORECAST_COLUMNS
from .regressors import REGRESSOR_COLUMNS
from .worker import (
    MODEL_PACKAGE,
    environment_report,
    make_analysis_fn,
    make_model_fn,
    make_regressor_lags_fn,
)

LIB_DIR = Path(__file__).resolve().parent
_SPARK_TYPES = {
    "string": StringType(),
    "int": IntegerType(),
    "double": DoubleType(),
    "boolean": BooleanType(),
    "timestamp": TimestampType(),
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


def spark_schema(columns: dict[str, str]) -> StructType:
    """Convert a column spec (name -> type name) into a Spark ``StructType``."""
    return StructType([StructField(name, _SPARK_TYPES[dtype], True) for name, dtype in columns.items()])


def get_spark() -> SparkSession:
    """Open a ``databricks-connect`` session on the configured cluster.

    Uses ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` when set (containers);
    otherwise the ``DATABRICKS_CONFIG_PROFILE`` profile (or ``DEFAULT``) from
    ``~/.databrickscfg``, which also supports OAuth logins (local runs).
    """
    cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID")
    if not cluster_id:
        raise ValueError("DATABRICKS_CLUSTER_ID is not set (add it to the .env file).")
    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        builder = DatabricksSession.builder.remote(
            host=os.environ["DATABRICKS_HOST"], token=os.environ["DATABRICKS_TOKEN"], cluster_id=cluster_id
        )
    else:
        builder = DatabricksSession.builder.profile(os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")).clusterId(
            cluster_id
        )
    return builder.getOrCreate()


def table_name(output_schema: str, run_id: str, kind: str, model: str | None = None, dev: bool = False) -> str:
    """Fully-qualified name of an output table.

    Parameters
    ----------
    output_schema : str
        ``catalog.schema`` to write into.
    run_id : str
        Run identifier; keeps reruns from overwriting each other.
    kind : str
        ``analysis``, ``regressor_lags``, ``backtest``, ``forecast`` or ``scores``.
    model : str, optional
        Model name for ``backtest`` / ``forecast`` tables.
    dev : bool
        Add a ``_dev`` suffix for sample (development) runs.

    Returns
    -------
    str
        ``<output_schema>.tsf_<run_id>_<kind>[_<model>][_dev]``.
    """
    parts = [run_id, kind] + ([model] if model else [])
    for part in parts:
        if not _IDENTIFIER.match(part):
            raise ValueError(f"'{part}' must contain only letters, digits and underscores.")
    return f"{output_schema}.tsf_{'_'.join(parts)}{'_dev' if dev else ''}"


def ship_code(spark: SparkSession, model_file: str | None = None) -> str | None:
    """Upload ``tsf_lib`` (and optionally a model file) to the cluster.

    The artifact name is content-addressed so an edited model file is never
    shadowed by an earlier upload in the same session.

    Parameters
    ----------
    spark : SparkSession
        Active session.
    model_file : str, optional
        Path to a Python file defining ``fit_forecast``.

    Returns
    -------
    str or None
        Module name to import as ``tsf_models.<name>``, or ``None`` when no
        model file was given.
    """
    digest = hashlib.sha256()
    lib_files = sorted(LIB_DIR.glob("*.py"))
    for path in lib_files:
        digest.update(path.read_bytes())
    module_name = None
    if model_file:
        model_bytes = Path(model_file).read_bytes()
        digest.update(model_bytes)
        module_name = f"m_{hashlib.sha256(model_bytes).hexdigest()[:10]}"
    tag = digest.hexdigest()[:12]

    zip_path = Path(tempfile.gettempdir()) / f"tsf_code_{tag}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in lib_files:
            archive.write(path, arcname=f"tsf_lib/{path.name}")
        if module_name:
            archive.writestr(f"{MODEL_PACKAGE}/__init__.py", "")
            archive.writestr(f"{MODEL_PACKAGE}/{module_name}.py", model_bytes)
    spark.addArtifacts(str(zip_path), pyfile=True)
    return module_name


def read_series(spark: SparkSession, table: str, id_col: str, date_col: str, value_col: str) -> DataFrame:
    """Read a series table as ``series_id, date, value``."""
    return spark.table(table).select(
        F.col(id_col).cast("string").alias("series_id"),
        F.to_timestamp(F.col(date_col)).alias("date"),
        F.col(value_col).cast("double").alias("value"),
    )


def read_regressors(
    spark: SparkSession, table: str, id_col: str, name_col: str, date_col: str, value_col: str
) -> DataFrame:
    """Read a long regressor table as ``series_id, regressor_name, date, value``."""
    return spark.table(table).select(
        F.col(id_col).cast("string").alias("series_id"),
        F.col(name_col).cast("string").alias("regressor_name"),
        F.to_timestamp(F.col(date_col)).alias("date"),
        F.col(value_col).cast("double").alias("value"),
    )


def _write(df: DataFrame, table: str) -> int:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    return df.sparkSession.table(table).count()


def check_environment(spark: SparkSession) -> dict[str, Any]:
    """Verify the connection and report package versions seen by a worker."""
    ship_code(spark)
    report = (
        spark.range(1)
        .withColumn("series_id", F.lit("x"))
        .groupBy("series_id")
        .applyInPandas(environment_report, schema="versions string")
        .collect()
    )
    return {"spark_version": spark.version, "worker_packages": report[0]["versions"]}


def run_analysis(
    spark: SparkSession,
    series: DataFrame,
    regressors: DataFrame | None,
    output_schema: str,
    run_id: str,
    alpha: float = 0.05,
    max_lag: int | None = None,
    lag_max: int | None = None,
    lag_method: str = "spearman",
    min_overlap: int = 12,
) -> dict[str, Any]:
    """Analyse every series (and every series/regressor pair) in parallel.

    Parameters
    ----------
    spark : SparkSession
        Active session.
    series : DataFrame
        ``series_id, date, value``.
    regressors : DataFrame or None
        ``series_id, regressor_name, date, value``.
    output_schema, run_id : str
        Where and under which run id to write results.
    alpha : float
        Significance level of the tests.
    max_lag : int, optional
        Largest ACF lag; defaults to the nominal season length per series.
    lag_max, lag_method, min_overlap
        Regressor lag search settings (see ``analyze_regressor_lags``).

    Returns
    -------
    dict
        Output table names and row counts.
    """
    ship_code(spark)
    analysis_table = table_name(output_schema, run_id, "analysis")
    result: dict[str, Any] = {"analysis_table": analysis_table}

    analysis = series.groupBy("series_id").applyInPandas(make_analysis_fn(alpha, max_lag), spark_schema(ANALYSIS_COLUMNS))
    result["analysis_rows"] = _write(analysis, analysis_table)

    if regressors is not None:
        lags_table = table_name(output_schema, run_id, "regressor_lags")
        decisions = spark.table(analysis_table).select("series_id", "stationarity_decision")
        series_with_decision = series.join(decisions, on="series_id", how="left")
        lags = (
            series_with_decision.groupBy("series_id")
            .cogroup(regressors.groupBy("series_id"))
            .applyInPandas(make_regressor_lags_fn(lag_max, lag_method, min_overlap), spark_schema(REGRESSOR_COLUMNS))
        )
        result["regressor_lags_table"] = lags_table
        result["regressor_lags_rows"] = _write(lags, lags_table)
    return result


def _with_analysis_json(spark: SparkSession, series: DataFrame, output_schema: str, run_id: str) -> DataFrame:
    analysis = spark.table(table_name(output_schema, run_id, "analysis"))
    fields = [F.col(c) for c in analysis.columns]
    lags_table = table_name(output_schema, run_id, "regressor_lags")
    if spark.catalog.tableExists(lags_table):
        lags = (
            spark.table(lags_table)
            .where(F.col("recommended_lags").isNotNull())
            .groupBy("series_id")
            .agg(F.collect_list(F.struct("regressor_name", "recommended_lags", "best_lag", "basis")).alias("regressor_lags"))
        )
        analysis = analysis.join(lags, on="series_id", how="left")
        fields.append(F.col("regressor_lags"))
    payload = analysis.select("series_id", F.to_json(F.struct(*fields)).alias("analysis_json"))
    return series.join(payload, on="series_id", how="left")


def run_model(
    spark: SparkSession,
    series: DataFrame,
    regressors: DataFrame | None,
    output_schema: str,
    run_id: str,
    model_file: str,
    model_name: str,
    horizon: int,
    n_folds: int,
    step: int,
    params: dict[str, Any],
    forecast: bool,
    series_ids: list[str] | None = None,
    dev: bool = False,
) -> dict[str, Any]:
    """Run a model's backtest (``forecast=False``) or final forecast on Databricks.

    Parameters
    ----------
    spark : SparkSession
        Active session.
    series, regressors : DataFrame
        As in ``run_analysis``.
    output_schema, run_id : str
        Output location; the analysis tables of the same run must exist.
    model_file : str
        Python file defining ``fit_forecast`` (see ``backtest`` module).
    model_name : str
        Name stored in the ``model`` column and the table name.
    horizon, n_folds, step : int
        Backtest geometry; ``n_folds`` and ``step`` are ignored for forecasts.
    params : dict
        Free-form parameters handed to the model.
    forecast : bool
        Write the final forecast instead of the backtest.
    series_ids : list of str, optional
        Restrict the run to these series (development sample).
    dev : bool
        Write to ``_dev`` tables so samples never overwrite full runs.

    Returns
    -------
    dict
        Output table name and row count.
    """
    module_name = ship_code(spark, model_file)
    if series_ids is not None:
        series = series.where(F.col("series_id").isin(series_ids))
        if regressors is not None:
            regressors = regressors.where(F.col("series_id").isin(series_ids))
    series = _with_analysis_json(spark, series, output_schema, run_id)
    if regressors is None:
        regressors = spark.createDataFrame(
            [], "series_id string, regressor_name string, date timestamp, value double"
        )

    columns = FORECAST_COLUMNS if forecast else BACKTEST_COLUMNS
    fn = make_model_fn(module_name, model_name, horizon, n_folds, step, params, forecast)
    output = series.groupBy("series_id").cogroup(regressors.groupBy("series_id")).applyInPandas(fn, spark_schema(columns))
    table = table_name(output_schema, run_id, "forecast" if forecast else "backtest", model_name, dev)
    return {"table": table, "rows": _write(output, table)}


def score_tables(spark: SparkSession, tables: list[str], output_table: str) -> DataFrame:
    """Compute sMAPE per series and model from backtest tables and write it.

    Parameters
    ----------
    spark : SparkSession
        Active session.
    tables : list of str
        Backtest tables (each may hold a different model).
    output_table : str
        Destination of the per-series scores.

    Returns
    -------
    DataFrame
        Columns ``series_id, model, smape, n_points, n_failed_folds``.
    """
    backtest = spark.table(tables[0])
    for table in tables[1:]:
        backtest = backtest.unionByName(spark.table(table))
    term = F.when(
        F.abs(F.col("actual")) + F.abs(F.col("forecast_mean")) == 0, F.lit(0.0)
    ).otherwise(200.0 * F.abs(F.col("actual") - F.col("forecast_mean")) / (F.abs(F.col("actual")) + F.abs(F.col("forecast_mean"))))
    scored = backtest.where(F.col("actual").isNotNull() & F.col("forecast_mean").isNotNull() & ~F.isnan("forecast_mean"))
    scores = scored.groupBy("series_id", "model").agg(F.avg(term).alias("smape"), F.count("*").alias("n_points"))
    failed = (
        backtest.where(F.col("error").isNotNull())
        .groupBy("series_id", "model")
        .agg(F.countDistinct("fold").alias("n_failed_folds"))
    )
    result = scores.join(failed, on=["series_id", "model"], how="full_outer").fillna({"n_failed_folds": 0})
    _write(result, output_table)
    return result
