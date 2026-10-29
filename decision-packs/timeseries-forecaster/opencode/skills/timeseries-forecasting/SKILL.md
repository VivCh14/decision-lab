---
name: timeseries-forecasting
description: How to analyse many time series in batches, write model functions whose structure and parameters come from the analysis, backtest them with a rolling origin, and score with sMAPE. Includes the model menu.
---

# Time series forecasting

Batches of series run on Modal (or a local process pool) through custom tools.
Inputs are a series file and an optional long regressor file; outputs are
parquet files in the work directory.

## Workflow

1. `inspect-data` the input files (counts, date range, regressor names).
2. `analyze-series` once. Writes `analysis.parquet` and, when regressors exist,
   `regressor_lags.parquet`; the data-file settings are remembered in `inputs.json`.
3. `summarize-analysis` and `fetch-table` the analysis. Decide value semantics,
   seasonality, stationarity and intermittency from the analysis columns, not
   from assumptions.
4. Develop on a sample: `sample-series` returns stratified `series_ids`; run
   `run-models` with `series_ids` and `dev=true`, small `n_folds`, small draws.
5. Fix failures (the backtest file has an `error` column per fold), compare
   models with `score-models`, then scale the SAME model file to all series
   (`series_ids` omitted, `dev=false`). Use `estimate-run-time` first.
6. `select-models` picks a model per segment; `run-forecast` produces forecasts.

Files: `backtest_<model>[_dev].parquet`, `forecast_<model>.parquet`,
`scores[_dev].parquet`, `metrics[_dev].parquet`, `selection.parquet`, `final_forecast.parquet`.

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
- `params`: free-form dict you pass in the `run-models`/`run-forecast` call.
- Return `horizon` rows with `forecast_mean` (required) and optionally
  `forecast_median`, `forecast_lower`, `forecast_upper`. Sampling-based models
  also add the constant columns from `tsf_lib.diagnostics.sampler_diagnostics(idata)`
  (`rhat_max`, `ess_min`, `divergence_pct`); `score-models` counts folds above
  R-hat 1.1, below ESS 200 or above 1% divergences as non-converged.
- `tsf_lib.quality.bound_draws(draws, history)` clips predictive draws to a plausible range; use it after exp/expm1 back-transforms and heavy-tailed likelihoods. `run-models`, `run-forecast` and `select-models` report `implausible_mean_series` (mean outside its own interval).

### MANDATORY: guard the posterior before stacking

Under the installed arviz, `idata.posterior` can be a `DataTree`, which has no
`.stack`. Code that stacks it directly dies with
`'DataTree' object has no attribute 'stack'` on every fold, and because
`run-models` reports that per row you get a backtest file where every row is an
error. Always convert first:

```python
posterior = idata.posterior
posterior = posterior.to_dataset() if hasattr(posterior, "to_dataset") else posterior
post = posterior.stack(sample=("chain", "draw"))
```

The same guard applies to `posterior_predictive` and `sample_stats`. Every
reference template in `references/` already does this; copy it, do not drop it.

The file runs inside worker containers (Modal or local processes): import only
packages installed in the image, keep it self-contained (plus `tsf_lib`), no file or network access, no global
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
| `recommended.robust_likelihood` (outliers found) | heavy-tailed likelihood (Student-t) instead of Normal |
| `recommended.train_from_date` (level shift found) | `tsf_lib.quality.trim_to_level_shift(train, rec)` drops pre-shift history when enough remains |
| `quality_issues`, `n_outliers`, `level_shift_date` | per-series data-quality findings from the analysis |
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
| Hurdle (Bernoulli occurrence x log-normal size) | intermittent or zero-heavy series where a mean forecast is rarely zero; score it with `smape_median` | `references/intermittent_hurdle.py` |
| Intermittent models (Croston/SBA style, zero-inflated NegBin) | `demand_class` intermittent or lumpy, high `pct_zero` | write from the pieces above |
| Local level / seasonal state-space (statsmodels or PyMC) | evolving level or seasonality, non-stationary series | write from the pieces above |
| Hierarchical pooling over similar series | many short series | only if per-series fits are too weak |

Copy a template into your work directory, then change it. Per-series MCMC runs
inside batch workers: keep draws, chains and tuning small, and measure the
runtime on the sample before scaling.

## Backtest and metrics

- Rolling origin: `horizon`, `n_folds`, `step`. The last fold ends at the last
  observation. Each fold trains only on data up to its origin.
- The backtest file holds `actual` next to `forecast_*` per series, fold and
  horizon step, so any metric can be derived from it later.
- `compute-metrics` derives WAPE, bias, MAE, RMSE, MASE, interval coverage and
  interval score from the stored actuals and forecasts; use them to explain
  results (especially on intermittent series), not to replace the selection metric.
- sMAPE (percent, 0-200) is the evaluation metric. Zero actual with non-zero
  forecast scores 200; both zero scores 0. Report it with the number of failed
  folds so failures are never hidden by averaging.
- Rank models per series and overall; always show the baseline next to them.
