# timeseries-forecaster

Batch time series analysis, Bayesian backtesting, model selection and forecasting. Series batches run on Modal, or in a local process pool without Modal credentials (same pattern as the MMM pack).

- **Input** (`--data`): a series file `(series_id, date, value)` and, optionally, a long regressor file `(series_id, regressor_name, date, value)`; `.csv` or `.parquet`. Column names are given in the prompt.
- **Compute**: `tsf_lib.modal_runner` splits series into batches and runs them in ONE call per tool invocation: `modal.Function.map` on the `tsf-runner` app (`docker/modal_app/tsf_runner.py`), or a local process pool when `DLAB_FIT_MODEL_LOCALLY=1` or `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` are unset. `DLAB_TSF_WORKERS` sets the local pool size. The `deploy_modal.sh` pre-run hook deploys the app.
- **Workflow** (MMM-style): the orchestrator explores the data, runs 2 `data-preparer` instances in parallel (different models; they must not transform the value column), analyses the cleaned data, runs the baseline itself, then fans out `modeler` instances (one model family each) in parallel. Modelers keep their model file, backtest/forecast parquet files and `summary.md` in their own instance directory; the orchestrator reads them by absolute path.
- **Output**: orchestrator directory: `data_summary.md`, `data_preparation_decision.md`, `analysis.parquet`, `regressor_lags.parquet`, `scores.parquet`, `metrics.parquet`, `selection.parquet`, `final_forecast.parquet`, reports and figures. Per modeler instance: `<model>.py`, `backtest_<model>.parquet`, `forecast_<model>.parquet`, `summary.md`.
- **Tools** take explicit file paths (`backtest_files`, `scores_file`, `forecast_files`); `output_dir` defaults to the current directory. They return JSON objects (`{error, stderr, stdout}` on failure).
- **Evaluation**: sMAPE; the backtest file supports any other metric.

## Run

```bash
cp .env.example .env   # optional: Modal tokens; model-provider credentials
dlab run --dpack decision-packs/timeseries-forecaster --env-file .env \
  --data example-data/series.csv --data example-data/regressors.csv \
  --prompt "Series file series.csv (series_id, date, value), regressors regressors.csv, horizon 6, rolling backtest."
```

Add `DLAB_FIT_MODEL_LOCALLY=1` to force the local pool. `example-data/make_example.py` regenerates the synthetic dataset.

## Tests

```bash
see tests/README.md
```

The suite covers analysis, regressors, backtesting, sampling, batching and the local process-pool pipeline end to end. The Modal path (`modal_runner._run_modal`, `modal_app/tsf_runner.py`) is not exercised without a Modal account.

## First live Modal run (checklist)

1. Put `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in `.env`; unset `DLAB_FIT_MODEL_LOCALLY`.
2. `modal deploy docker/modal_app/tsf_runner.py` (or let the hook do it); confirm the image builds with the pinned conda/pip versions.
3. Run the example data and compare `analysis.parquet` / `scores.parquet` against a `DLAB_FIT_MODEL_LOCALLY=1` run.
4. Check `estimate-run-time` against the measured wall time and tune `cpu`/`memory` in the app.
