"""Map analysis results to recommended model structure and prior scales.

The recommendations are defaults for modeling agents, not decisions: every
entry carries the evidence it was derived from so an agent can override it.
"""

import json
import math
from typing import Any


def _finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _season_length(row: dict[str, Any]) -> tuple[int | None, str]:
    agreement = row.get("seasonality_agreement")
    strength = row.get("stl_strength") or 0.0
    if agreement == "agree":
        return int(row["stl_period"]), "strong"
    if agreement == "stl_only" and strength >= 0.5:
        return int(row["stl_period"]), "moderate"
    if agreement == "fft_only":
        return int(round(row["fft_period"])), "moderate"
    return None, "none"


def _ar_order(row: dict[str, Any], max_order: int = 3) -> int:
    pacf_raw = row.get("pacf_values")
    n = row.get("analysis_n") or 0
    if not pacf_raw or n < 2:
        return 0
    pacf_values = json.loads(pacf_raw)
    bound = 1.96 / math.sqrt(n)
    significant = [lag for lag in range(1, min(max_order, len(pacf_values) - 1) + 1) if abs(pacf_values[lag]) > bound]
    return max(significant) if significant else 0


def _likelihood_candidates(row: dict[str, Any]) -> list[str]:
    kind = row.get("value_kind")
    pct_zero = row.get("pct_zero") or 0.0
    dispersion = row.get("dispersion_index")
    if kind == "count":
        overdispersed = _finite(dispersion) and dispersion > 1.5
        candidates = ["negative_binomial"] if overdispersed else ["poisson", "negative_binomial"]
        if pct_zero > 0.3:
            candidates.append("zero_inflated_negative_binomial")
        return candidates
    if kind == "nonnegative_continuous":
        return ["hurdle_gamma", "gamma_on_shifted"] if pct_zero > 0 else ["lognormal", "gamma"]
    if kind in ("signed_integer", "signed_continuous"):
        return ["student_t", "normal"]
    return []


def recommend_params(row: dict[str, Any]) -> dict[str, Any]:
    """Derive recommended structure and prior scales from one analysis row.

    Parameters
    ----------
    row : dict
        Flat analysis result (keys of ``ANALYSIS_COLUMNS``); stage columns may
        be ``None`` when that stage failed.

    Returns
    -------
    dict
        Recommended ``season_length``, ``fourier_terms``, ``ar_order``,
        ``differencing``, ``trend``, ``likelihood_candidates``, ``transform``,
        prior scales for level / noise, ``min_train_size`` and ``evidence``.
    """
    season_length, season_evidence = _season_length(row)
    strength = row.get("stl_strength") or 0.0
    if season_length is None:
        fourier_terms = 0
    else:
        fourier_terms = max(1, min(season_length // 2, 2 if strength < 0.5 else 3 if strength < 0.7 else 4))

    nonnegative = row.get("value_kind") in ("count", "nonnegative_continuous")
    mean = row.get("mean") or 0.0
    std = row.get("std") or 0.0
    positive_scale = row.get("mean_nonzero") if _finite(row.get("mean_nonzero")) else abs(mean)

    min_train = max(
        2 * (season_length or 1) + 1,
        (row.get("max_lag_used") or 1) + 3,
        10,
    )

    return {
        "season_length": season_length,
        "seasonality_evidence": season_evidence,
        "fourier_terms": fourier_terms,
        "ar_order": _ar_order(row),
        "differencing": 1 if row.get("stationarity_decision") == "non_stationary" else 0,
        "trend": "linear" if row.get("trend_direction") in ("upward", "downward") else "none",
        "likelihood_candidates": _likelihood_candidates(row),
        "transform": "log1p" if nonnegative and (row.get("demand_class") in ("erratic", "lumpy")) else "none",
        "zero_inflated": bool((row.get("pct_zero") or 0.0) > 0.3),
        "level_prior": {"mean": mean, "sd": max(std, abs(mean) * 0.5, 1e-6)},
        "positive_scale": float(positive_scale) if _finite(positive_scale) else None,
        "noise_prior_sd": max(std, 1e-6),
        "min_train_size": int(min_train),
        "evidence": {
            "seasonality_agreement": row.get("seasonality_agreement"),
            "stl_strength": row.get("stl_strength"),
            "stationarity_decision": row.get("stationarity_decision"),
            "demand_class": row.get("demand_class"),
            "dispersion_index": row.get("dispersion_index"),
            "data_quality_flag": row.get("data_quality_flag"),
        },
    }
