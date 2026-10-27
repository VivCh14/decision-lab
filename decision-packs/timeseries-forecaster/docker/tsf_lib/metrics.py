"""Post-processing metrics computed from backtest tables (actuals next to forecasts).

``sMAPE`` stays the selection metric; these complement it, especially for
intermittent series where sMAPE punishes zero actuals.
"""

import json

import numpy as np
import pandas as pd

from .backtest import DIVERGENCE_PCT_MAX, ESS_MIN, RHAT_MAX


def _season(analysis_row: pd.Series | None) -> int:
    if analysis_row is None:
        return 1
    recommended = json.loads(analysis_row.get("recommended_params") or "{}")
    return max(1, int(recommended.get("season_length") or 1))


def _naive_scale(train: np.ndarray, season: int) -> float:
    lag = season if len(train) > season else 1
    return float(np.mean(np.abs(train[lag:] - train[:-lag]))) if len(train) > lag else np.nan


def series_metrics(group: pd.DataFrame, values: pd.DataFrame, season: int) -> dict[str, float]:
    """Metrics for one series and model over all successful backtest folds.

    Parameters
    ----------
    group : pd.DataFrame
        Backtest rows of one series and model.
    values : pd.DataFrame
        Columns ``date`` and ``value`` of the series (for the MASE scale).
    season : int
        Season length used for the seasonal-naive scale of MASE.
    """
    scored = group.dropna(subset=["actual", "forecast_mean"])
    if scored.empty:
        return {}
    err = scored["forecast_mean"] - scored["actual"]
    out: dict[str, float] = {
        "wape": float(err.abs().sum() / max(scored["actual"].abs().sum(), 1e-12)),
        "bias": float(err.sum() / max(scored["actual"].abs().sum(), 1e-12)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
    }
    scales = []
    for origin, fold in scored.groupby("origin_date"):
        train = values.loc[values["date"] <= origin, "value"].to_numpy(dtype=float)
        scales.append(_naive_scale(train, season))
    scale = np.nanmean(scales) if scales else np.nan
    out["mase"] = float(err.abs().mean() / scale) if np.isfinite(scale) and scale > 0 else np.nan
    interval = scored.dropna(subset=["forecast_lower", "forecast_upper"])
    if not interval.empty:
        level = float(interval["interval_level"].iloc[0])
        alpha = 1.0 - level
        a, lo, hi = interval["actual"], interval["forecast_lower"], interval["forecast_upper"]
        out["coverage"] = float(((a >= lo) & (a <= hi)).mean())
        out["nominal_level"] = level
        out["interval_width"] = float((hi - lo).mean())
        out["interval_score"] = float(
            ((hi - lo) + 2 / alpha * (lo - a).clip(lower=0) + 2 / alpha * (a - hi).clip(lower=0)).mean()
        )
    return out


def compute_metrics(backtest: pd.DataFrame, series: pd.DataFrame, analysis: pd.DataFrame | None) -> pd.DataFrame:
    """Per series and model metrics: WAPE, bias, MAE, RMSE, MASE, interval coverage and score.

    Parameters
    ----------
    backtest : pd.DataFrame
        Backtest table, possibly stacked over models.
    series : pd.DataFrame
        Long table with ``series_id``, ``date``, ``value``.
    analysis : pd.DataFrame or None
        Analysis table (for the season length used by MASE).
    """
    by_id = {sid: grp[["date", "value"]].sort_values("date") for sid, grp in series.groupby("series_id")}
    rows = []
    for (series_id, model), group in backtest.groupby(["series_id", "model"], dropna=False):
        analysis_row = None
        if analysis is not None and (analysis["series_id"] == series_id).any():
            analysis_row = analysis[analysis["series_id"] == series_id].iloc[0]
        values = by_id.get(series_id)
        if values is None:
            raise ValueError(f"Series '{series_id}' from the backtest is not in the series table.")
        rows.append({"series_id": series_id, "model": model, **series_metrics(group, values, _season(analysis_row))})
    return pd.DataFrame(rows)


def nonconverged_folds(backtest: pd.DataFrame) -> pd.DataFrame:
    """Folds whose sampler diagnostics breach the convergence standards."""
    diag = backtest.dropna(subset=["fold"]).groupby(["series_id", "model", "fold"])[
        ["rhat_max", "ess_min", "divergence_pct"]
    ].first().reset_index()
    bad = (diag["rhat_max"] > RHAT_MAX) | (diag["ess_min"] < ESS_MIN) | (diag["divergence_pct"] > DIVERGENCE_PCT_MAX)
    return diag[bad]
