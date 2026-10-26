---
name: timeseries-forecasting
description: How to analyse many time series on Databricks, write model functions whose structure and parameters come from the analysis, backtest them with a rolling origin, and score with sMAPE. Includes the model menu.
---

# Time series forecasting on Databricks

All heavy work runs on a Databricks cluster through the `tsf` tool. Inputs and
outputs are Unity Catalog tables. You never copy data into the work directory.

## Workflow

1. `inspect` the input tables (counts, date range, regressor names).
2. `analyze` all series once per run (`run_id`). Writes `tsf_<run_id>_analysis`
   and, when regressors exist, `tsf_<run_id>_regressor_lags`.
3. `summarize` and `fetch` the analysis. Decide value semantics, seasonality,
   stationarity and intermittency from the analysis columns, not from assumptions.
4. Develop on a sample: `sample` returns stratified `series_ids`; run
   `backtest` with `series_ids` and `dev=true`, small `n_folds`, small draws.
5. Fix failures (the backtest table has an `error` column per fold), compare
   models with `score`, then scale the SAME model file to all series
   (`series_ids` omitted, `dev=false`).
6. Run `forecast` for the chosen model, then `score` for the report.

Table names: `<output_schema>.tsf_<run_id>_<kind>[_<model>][_dev]`. Reruns with a
new `run_id` never overwrite earlier results.

## Model function contract

A model file defines:

```python
def fit_forecast(train, future_dates, regressors, analysis, params) -> pd.DataFrame
```

- `train`: columns `date`, `value`; uniform grid, no gaps (latest contiguous run).
- `future_dates`: `pd.DatetimeIndex` of the `horizon` target dates.
- `regressors`: long table `regressor_name, date, value` limited to dates up to
  the forecast origin (or `None`). Never use values after the origin.
- `analysis`: the series' analysis row as a dict. `analysis["recommended"]` is the
  parsed recommendation dict and `analysis["regressor_lags"]` is
  `{regressor_name: [lags]}`.
- `params`: free-form dict you pass in the `backtest`/`forecast` call.
- Return `horizon` rows with `forecast_mean` (required) and optionally
  `forecast_median`, `forecast_lower`, `forecast_upper`.

The file runs on Spark workers: import only packages installed on the cluster,
keep it self-contained (plus `tsf_lib`), no file or network access, no global
state, no fixed random seeds shared across series. A failing fold is recorded in
`error` instead of aborting; a model that fails often is a bug to fix.

## From analysis to model parameters

Read these columns of the analysis table and set the model from them:

| Analysis field | Drives |
|---|---|
| `value_kind` (count, nonnegative_continuous, signed_*) | likelihood family, link, whether a transform is allowed |
| `demand_class` (smooth, erratic, intermittent, lumpy), `pct_zero` | intermittent/zero-inflated structure vs. standard |
| `recommended.season_length`, `seasonality_evidence` | seasonal period; no seasonal terms when evidence is weak |
| `recommended.fourier_terms` | number of harmonics |
| `recommended.ar_order`, `differencing`, `stationarity_decision` | error structure, differencing |
| `recommended.trend` | trend term |
| `recommended.level_prior`, `positive_scale`, `noise_prior_sd` | prior locations and scales |
| `recommended.transform` | e.g. log1p for positive-skewed continuous data |
| `recommended.min_train_size`, `data_quality_flag` | fold eligibility; short series need simpler models |
| regressor lags (`analysis["regressor_lags"]`) | which lagged regressors enter; lags shorter than the horizon are unknown at the origin and must be left out or replaced by a forecast |

Per-series differences are the point: the same model file reads each series'
analysis row, so structure and priors adapt per series. Use `params` for choices
shared across series (draws, fallback model, interval level).

## Model menu

Pick methods that fit the analysis; run several in parallel and compare by sMAPE.

| Model | Use when | Template |
|---|---|---|
| Seasonal naive / last value | always run as the baseline every model must beat | `references/seasonal_naive.py` |
| Bayesian regression (trend + Fourier + lagged regressors; Normal, Poisson or NegBin) | default Bayesian model, counts or continuous | `references/bayes_fourier_regression.py` |
| Intermittent models (Croston/SBA style, zero-inflated NegBin) | `demand_class` intermittent or lumpy, high `pct_zero` | write from the pieces above |
| Local level / seasonal state-space (statsmodels or PyMC) | evolving level or seasonality, non-stationary series | write from the pieces above |
| Hierarchical pooling over similar series | many short series | only if per-series fits are too weak |

Copy a template into your work directory, then change it. Per-series MCMC runs
inside Spark tasks: keep draws, chains and tuning small, and measure the
runtime on the sample before scaling.

## Backtest and metrics

- Rolling origin: `horizon`, `n_folds`, `step`. The last fold ends at the last
  observation. Each fold trains only on data up to its origin.
- The backtest table holds `actual` next to `forecast_*` per series, fold and
  horizon step, so any metric can be derived from it later.
- sMAPE (percent, 0-200) is the evaluation metric. Zero actual with non-zero
  forecast scores 200; both zero scores 0. Report it with the number of failed
  folds so failures are never hidden by averaging.
- Rank models per series and overall; always show the baseline next to them.
