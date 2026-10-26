# timeseries-forecaster

Batch time series analysis, Bayesian backtesting and forecasting on Databricks.

- **Input**: Unity Catalog tables passed by name. Series table `(series_id, date, value)`;
  optional long regressor table `(series_id, regressor_name, date, value)` of any length.
- **Compute**: a Databricks classic cluster (`DATABRICKS_CLUSTER_ID`) through `databricks-connect`
  and Spark `applyInPandas`. Nothing is copied into the work directory. Install `pymc`,
  `nutpie`, `statsmodels`, `scipy` on the cluster; the `databricks-connect` version must
  match the cluster runtime (`--build-arg DATABRICKS_CONNECT_VERSION=...`).
- **Output**: tables `<catalog.schema>.tsf_<run_id>_<kind>[_<model>][_dev]`: `analysis`,
  `regressor_lags`, `backtest_<model>` (forecast next to actual), `forecast_<model>`, `scores`.
- **Evaluation**: sMAPE; the backtest table supports any other metric.

## Run

```bash
cp .env.example .env   # fill in Databricks and model-provider credentials
dlab run --dpack decision-packs/timeseries-forecaster --env-file .env \
  --prompt "Series table main.demo.series (series_id, date, value), regressors main.demo.regressors, \
outputs to main.demo_out, run id r1, horizon 6, rolling backtest."
```

`example-data/make_example.py` writes a synthetic series/regressor dataset to upload for a dry run.

## Tests

```bash
python -m pytest decision-packs/timeseries-forecaster/tests -v
```

The Spark layer (`spark_batch.py`, `cli.py`) needs a live cluster and is not covered by the
local suite; the analysis, regressor, backtest and sampling logic is pure pandas/numpy.
