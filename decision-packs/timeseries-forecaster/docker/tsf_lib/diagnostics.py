"""Sampler convergence diagnostics shared by Bayesian model files."""

from typing import Any

import arviz as az
import numpy as np


def _to_dataset(group: Any) -> Any:
    return group.to_dataset() if hasattr(group, "to_dataset") else group


def sampler_diagnostics(idata: Any) -> dict[str, float]:
    """Worst-case R-hat, smallest ESS and percent of divergent draws of a fit.

    Parameters
    ----------
    idata : arviz InferenceData or DataTree
        Result of ``pm.sample``.

    Returns
    -------
    dict
        ``rhat_max``, ``ess_min`` (bulk and tail) and ``divergence_pct``; add
        these as constant columns of the model output so ``score_backtest``
        can flag non-converged folds.
    """
    rhat = _to_dataset(az.rhat(idata))
    ess = _to_dataset(az.ess(idata))
    rhat_values = np.concatenate([np.ravel(rhat[v].to_numpy()) for v in rhat.data_vars])
    ess_values = np.concatenate([np.ravel(ess[v].to_numpy()) for v in ess.data_vars])
    stats = _to_dataset(idata.sample_stats)
    divergent = float(np.mean(stats["diverging"].to_numpy())) * 100.0 if "diverging" in stats else float("nan")
    return {
        "rhat_max": float(np.nanmax(rhat_values)),
        "ess_min": float(np.nanmin(ess_values)),
        "divergence_pct": divergent,
    }
