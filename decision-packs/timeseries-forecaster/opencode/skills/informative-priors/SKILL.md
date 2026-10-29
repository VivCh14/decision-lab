---
name: Informative Priors for Time Series
description: Guide to calculating weakly informative priors for time series models (level, trend, seasonality, regressors, noise) from each series' own history. Use when configuring priors in a model file or when folds fail to converge.
---

# Informative Priors for Time Series Models

Derive priors from each series' training window and the `analysis` row, never from hard-coded numbers. Every model runs once per series and fold, so the priors must adapt to scale.

## When to use

- Setting priors for level, trend, seasonal amplitude, regressor effects and noise
- Folds that do not converge or forecasts that blow up (priors too wide for the series scale)
- Short series, where priors carry most of the information

## Step 1: Standardise the scale

Work on a standardised copy of the training values (after your own transform, e.g. log1p):

```python
mu_y, sd_y = float(np.mean(y_train)), float(np.std(y_train)) or 1.0
z = (y_train - mu_y) / sd_y
```

Priors are then set in units of `sd_y`, and forecasts are mapped back with `mu_y + sd_y * draws`.

## Step 2: Reason per component

| Component | Prior rule |
|---|---|
| Level / intercept | `Normal(mean of last season of z, <LEVEL_SIGMA>)`; centre on the recent level, not on 0, for trending series |
| Trend slope | `Normal(0, <SLOPE_SIGMA>)` with `<SLOPE_SIGMA>` from the regression slope's standard error on z, floored at a small value |
| Level/trend innovations (state-space) | `HalfNormal(<INNOV_SIGMA>)` with a small scale (changes per period are a fraction of 1 in z units) |
| Seasonal amplitude | `Normal(0, <AMP_SIGMA>)` per Fourier term, shrinking with harmonic order |
| Regressor effect | `Normal(0, <BETA_SIGMA>)` on a standardised, lagged regressor; centre on 0 unless the sign is known |
| Noise | `HalfNormal(<NOISE_SIGMA>)` with `<NOISE_SIGMA>` near the residual sd of a simple fit |
| Degrees of freedom (robust) | `Gamma(2, 0.1)` or `Exponential` shifted above 2 |
| Occurrence probability (hurdle) | `Beta` centred on the training share of non-zero periods |

## Step 3: Check the prior predictive

Sample the prior predictive and confirm the prior forecast range covers the observed range without reaching orders of magnitude beyond it (a prior that allows 100x the history causes the mean-forecast blow-ups `bound_draws` guards against). Tighten when it does.

```python
with model:
    prior = pm.sample_prior_predictive(draws=<N>, random_seed=None)
```

## Step 4: Short series

With few observations, shrink: smaller `<AMP_SIGMA>`, fewer Fourier terms, no regressor effects unless lagged evidence exists. If the prior predictive is the only information, mark the series `unvalidated`.

## Rules

- Every scale comes from the data or the `analysis` row; do not copy values from examples.
- Never fix a random seed.
- Record the prior choices in `summary.md` so the orchestrator can explain them.
