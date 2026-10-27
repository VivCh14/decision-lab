# timeseries-forecaster

Batch time series analysis, Bayesian backtesting, model selection and forecasting. Series batches run on Modal, or in a local process pool without Modal credentials (same pattern as the MMM pack).

- **Input** (`--data`): a series file `(series_id, date, value)` and, optionally, a long regressor file `(series_id, regressor_name, date, value)`; `.csv` or `.parquet`. Column names are given in the prompt.
- **Compute**: `tsf_lib.modal_runner` splits series into batches and runs them in ONE call per tool invocation: `modal.Function.map` on the `tsf-runner` app (`docker/modal_app/tsf_runner.py`), or a local process pool when `DLAB_FIT_MODEL_LOCALLY=1` or `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` are unset. `DLAB_TSF_WORKERS` sets the local pool size. The `deploy_modal.sh` pre-run hook deploys the app.
- **Output** (in the work dir): `analysis.parquet`, `regressor_lags.parquet`, `backtest_<model>.parquet` (forecast next to actual), `forecast_<model>.parquet`, `scores.parquet`, `selection.parquet`, `final_forecast.parquet`.
- **Evaluation**: sMAPE; the backtest file supports any other metric.

## Run

```bash
cp .env.example .env   # optional: Modal tokens; model-provider credentials
dlab run --dpack decision-packs/timeseries-forecaster --env-file .env \
  --data example-data/series.csv example-data/regressors.csv \
  --prompt "Series file series.csv (series_id, date, value), regressors regressors.csv, horizon 6, rolling backtest."
```

Add `DLAB_FIT_MODEL_LOCALLY=1` to force the local pool. `example-data/make_example.py` regenerates the synthetic dataset.

## Tests

```bash
python -m pytest decision-packs/timeseries-forecaster/tests -v
```

The suite covers analysis, regressors, backtesting, sampling, batching and the local process-pool pipeline end to end. The Modal path (`modal_runner._run_modal`, `modal_app/tsf_runner.py`) is not exercised without a Modal account.

## First live Modal run (checklist)

1. Put `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in `.env`; unset `DLAB_FIT_MODEL_LOCALLY`.
2. `modal deploy docker/modal_app/tsf_runner.py` (or let the hook do it); confirm the image builds with the pinned conda/pip versions.
3. Run the example data and compare `analysis.parquet` / `scores.parquet` against a `DLAB_FIT_MODEL_LOCALLY=1` run.
4. Check `estimate-run-time` against the measured wall time and tune `cpu`/`memory` in the app.
