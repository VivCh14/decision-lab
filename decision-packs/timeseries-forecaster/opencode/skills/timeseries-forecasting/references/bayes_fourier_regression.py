"""Bayesian regression with trend, Fourier seasonality and lagged regressors (PyMC).

Structure and priors are driven by the series analysis (``analysis["recommended"]``)
and can be overridden through ``params``. Starting point for the modeling agents:
copy it, then adapt the structure to what the analysis and the backtest show.
"""

import numpy as np
import pandas as pd
import pymc as pm

from tsf_lib.regressors import lagged_regressor_matrix


def _fourier(t: np.ndarray, period: int, terms: int) -> np.ndarray:
    cols = []
    for k in range(1, terms + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def _regressor_features(
    regressors: "pd.DataFrame | None", analysis: dict, train_dates: pd.DatetimeIndex, future_dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lags = analysis.get("regressor_lags") or {}
    if regressors is None or not lags:
        return np.zeros((len(train_dates), 0)), np.zeros((len(future_dates), 0)), []
    freq = analysis["frequency"]
    x_train = lagged_regressor_matrix(regressors, train_dates, lags, freq)
    x_future = lagged_regressor_matrix(regressors, future_dates, lags, freq)
    # a lag shorter than the horizon is unknown in the future: leave it out (no leakage)
    keep = [c for c in x_future.columns if x_future[c].notna().all() and x_train[c].notna().mean() > 0.8]
    mean, std = x_train[keep].mean(), x_train[keep].std().replace(0, 1.0)
    x_train = ((x_train[keep] - mean) / std).fillna(0.0)
    x_future = (x_future[keep] - mean) / std
    return x_train.to_numpy(), x_future.to_numpy(), keep


def fit_forecast(
    train: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    regressors: "pd.DataFrame | None",
    analysis: dict,
    params: dict,
) -> pd.DataFrame:
    rec = analysis["recommended"]
    period = int(params.get("season_length") or rec.get("season_length") or 1)
    terms = int(params.get("fourier_terms", rec.get("fourier_terms") or 0)) if period > 1 else 0
    likelihood = params.get("likelihood") or (rec.get("likelihood_candidates") or ["normal"])[0]
    level = float(params.get("interval_level", 0.8))

    y = train["value"].to_numpy(dtype=float)
    n, horizon = len(y), len(future_dates)
    t = np.arange(n + horizon)
    season = _fourier(t, period, terms)
    x_train, x_future, _ = _regressor_features(regressors, analysis, pd.DatetimeIndex(train["date"]), future_dates)
    design_train = np.column_stack([season[:n], x_train])
    design_future = np.column_stack([season[n:], x_future])
    t_train, t_future = t[:n] / n, t[n:] / n

    count_model = likelihood in ("poisson", "negative_binomial")
    transform = rec.get("transform")
    target = np.log1p(y) if (not count_model and transform == "log1p") else y
    center, scale = (0.0, 1.0) if count_model else (target.mean(), target.std() or 1.0)
    z = target if count_model else (target - center) / scale

    with pm.Model():
        intercept = pm.Normal("intercept", np.log(y.mean() + 1.0) if count_model else 0.0, 1.0)
        slope = pm.Normal("slope", 0.0, 0.5 if count_model else 1.0) if rec.get("trend") != "none" else 0.0
        beta = pm.Normal("beta", 0.0, 0.5, shape=design_train.shape[1])
        eta = intercept + slope * t_train + pm.math.dot(design_train, beta)
        if count_model:
            mu = pm.math.exp(eta)
            if likelihood == "poisson":
                pm.Poisson("obs", mu, observed=z)
            else:
                phi = pm.Exponential("phi", 0.5)
                pm.NegativeBinomial("obs", mu=mu, alpha=phi, observed=z)
        else:
            sigma = pm.HalfNormal("sigma", float(rec.get("noise_prior_sd") or 1.0))
            pm.Normal("obs", eta, sigma, observed=z)
        idata = pm.sample(
            draws=int(params.get("draws", 300)),
            tune=int(params.get("tune", 300)),
            chains=int(params.get("chains", 2)),
            nuts_sampler="nutpie",
            progressbar=False,
            compute_convergence_checks=False,
        )

    posterior = idata.posterior
    posterior = posterior.to_dataset() if hasattr(posterior, "to_dataset") else posterior
    post = posterior.stack(sample=("chain", "draw"))
    b0 = post["intercept"].to_numpy()
    b1 = post["slope"].to_numpy() if "slope" in post else np.zeros_like(b0)
    bs = post["beta"].to_numpy()  # (k, samples)
    eta_future = b0[None, :] + b1[None, :] * t_future[:, None] + design_future @ bs
    rng = np.random.default_rng()
    if count_model:
        mu_future = np.exp(eta_future)
        if likelihood == "poisson":
            draws = rng.poisson(mu_future)
        else:
            phi_s = post["phi"].to_numpy()[None, :]
            draws = rng.negative_binomial(phi_s, phi_s / (phi_s + mu_future))
    else:
        draws = eta_future + rng.normal(size=eta_future.shape) * post["sigma"].to_numpy()[None, :]
        draws = draws * scale + center
        if transform == "log1p":
            draws = np.expm1(draws)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return pd.DataFrame(
        {
            "forecast_mean": draws.mean(axis=1),
            "forecast_median": np.median(draws, axis=1),
            "forecast_lower": np.quantile(draws, lo, axis=1),
            "forecast_upper": np.quantile(draws, hi, axis=1),
        }
    )
