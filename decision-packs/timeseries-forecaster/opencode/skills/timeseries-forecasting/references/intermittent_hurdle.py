"""Hurdle model for intermittent demand (PyMC): occurrence x size.

Zeros are modelled by a Bernoulli occurrence process (optionally seasonal on the
logit scale), non-zero values by a log-normal size distribution. Forecast draws are
``occurrence * size`` (rounded for count series), so the median forecast is the typical
period (often zero) and the mean forecast is the expected demand. Structure follows
``analysis["recommended"]``; override through ``params``. Starting point for the
modeling agents: copy it, then adapt it to what the backtest shows.
"""

import numpy as np
import pandas as pd
import pymc as pm

from tsf_lib.diagnostics import sampler_diagnostics
from tsf_lib.quality import bound_draws, trim_to_level_shift


def _fourier(t: np.ndarray, period: int, terms: int) -> np.ndarray:
    cols = []
    for k in range(1, terms + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def fit_forecast(
    train: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    regressors: "pd.DataFrame | None",
    analysis: dict,
    params: dict,
) -> pd.DataFrame:
    rec = analysis.get("recommended") or {}
    level = float(params.get("interval_level", 0.8))
    train = trim_to_level_shift(train, rec) if params.get("trim_level_shift", True) else train
    y = train["value"].to_numpy(dtype=float)
    n, horizon = len(y), len(future_dates)
    is_count = analysis.get("value_kind") == "count"
    period = int(params.get("season_length") or rec.get("season_length") or 1)
    terms = int(params.get("occurrence_fourier_terms", 0)) if period > 1 else 0

    occurred = (y > 0).astype(int)
    positive = np.log(y[y > 0])
    size_center = float(positive.mean()) if positive.size else 0.0
    size_spread = float(positive.std()) if positive.size > 1 else 1.0

    t_all = np.arange(n + horizon)
    season = _fourier(t_all, period, terms)
    season_train, season_future = season[:n], season[n:]
    k = season.shape[1]

    with pm.Model():
        logit_base = pm.Normal("logit_base", 0.0, 1.5)
        logit = logit_base + (pm.math.dot(season_train, pm.Normal("beta", 0.0, 0.5, shape=k)) if k else 0.0)
        pm.Bernoulli("occurrence", logit_p=logit, observed=occurred)
        if positive.size:
            size_mu = pm.Normal("size_mu", size_center, 1.0)
            size_sigma = pm.HalfNormal("size_sigma", max(size_spread, 0.1))
            pm.Normal("size", size_mu, size_sigma, observed=positive)
        idata = pm.sample(
            draws=int(params.get("draws", 500)),
            tune=int(params.get("tune", 500)),
            chains=int(params.get("chains", 2)),
            nuts_sampler="nutpie",
            progressbar=False,
            compute_convergence_checks=False,
        )

    diagnostics = sampler_diagnostics(idata)
    posterior = idata.posterior
    posterior = posterior.to_dataset() if hasattr(posterior, "to_dataset") else posterior
    post = posterior.stack(sample=("chain", "draw"))
    n_samples = post["logit_base"].shape[0]
    eta = post["logit_base"].to_numpy()[None, :] + (
        season_future @ post["beta"].to_numpy() if k else np.zeros((horizon, n_samples))
    )
    rng = np.random.default_rng()
    occurs = rng.random(eta.shape) < 1.0 / (1.0 + np.exp(-eta))
    if positive.size:
        sizes = np.exp(rng.normal(post["size_mu"].to_numpy()[None, :], post["size_sigma"].to_numpy()[None, :], eta.shape))
    else:
        sizes = np.zeros(eta.shape)
    draws = bound_draws(np.where(occurs, sizes, 0.0), y)
    if is_count:
        draws = np.round(draws)

    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return pd.DataFrame(
        {
            "forecast_mean": draws.mean(axis=1),
            "forecast_median": np.median(draws, axis=1),
            "forecast_lower": np.quantile(draws, lo, axis=1),
            "forecast_upper": np.quantile(draws, hi, axis=1),
            **diagnostics,
        }
    )
