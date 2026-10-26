"""Generic per-series analysis: value semantics, frequency, autocorrelation,
seasonality (STL + FFT cross-check), stationarity and trend.

Ported and generalised from the spare-part demand analysis scripts. Every
function works on a plain ``(date, value)`` series; nothing is specific to a
domain or to monthly data. Results from ``analyze_series`` are flat dicts whose
keys are listed in ``ANALYSIS_COLUMNS`` so they can be written as one table row
per series.
"""

import json
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from .recommend import recommend_params
from .timegrid import (
    FREQUENCIES,
    FrequencyError,
    classify_frequency,
    contiguous_tail,
    regularize,
)

logger = logging.getLogger("tsf_analysis")

ANALYSIS_COLUMNS: dict[str, str] = {
    "series_id": "string",
    "frequency": "string",
    "season_length_nominal": "int",
    "n_obs": "int",
    "n_missing_periods": "int",
    "start_date": "string",
    "end_date": "string",
    "analysis_n": "int",
    "value_kind": "string",
    "is_integer": "boolean",
    "pct_zero": "double",
    "pct_negative": "double",
    "mean": "double",
    "std": "double",
    "min": "double",
    "max": "double",
    "mean_nonzero": "double",
    "dispersion_index": "double",
    "demand_class": "string",
    "adi": "double",
    "cv2": "double",
    "dominant_cycle": "int",
    "cycle_strength": "double",
    "candidate_cycles": "string",
    "significant_lags": "string",
    "ljung_box_pvalue": "double",
    "has_autocorrelation": "boolean",
    "acf_values": "string",
    "pacf_values": "string",
    "max_lag_used": "int",
    "stl_period": "int",
    "stl_strength": "double",
    "fft_period": "double",
    "fft_pvalue": "double",
    "seasonality_agreement": "string",
    "stationarity_decision": "string",
    "adf_pvalue": "double",
    "kpss_pvalue": "double",
    "trend_direction": "string",
    "trend_slope": "double",
    "trend_pvalue": "double",
    "trend_total_pct_change": "double",
    "data_quality_flag": "string",
    "recommended_params": "string",
    "stage_errors": "string",
    "warnings": "string",
}


class AnalysisError(Exception):
    """Classified analysis failure; ``category`` groups failures by root cause."""

    category: str = "internal_error"


class InsufficientSampleError(AnalysisError):
    category = "insufficient_sample"


class ConstantSeriesError(AnalysisError):
    category = "constant_series"


class DuplicateDatesError(AnalysisError):
    category = "duplicate_dates"


class NonUniformFrequencyError(AnalysisError):
    category = "non_uniform_frequency"


def classify_data_quality(sample_size: int, pct_zero: float) -> str:
    """Bucket a series into ``good`` / ``marginal`` / ``poor``.

    Parameters
    ----------
    sample_size : int
        Number of usable observations.
    pct_zero : float
        Fraction of observations equal to zero.

    Returns
    -------
    str
        Reliability bucket.
    """
    if sample_size < 10 or pct_zero > 0.5:
        return "poor"
    if sample_size > 25 and pct_zero < 0.3:
        return "good"
    return "marginal"


def describe_values(y: np.ndarray) -> dict[str, Any]:
    """Describe what kind of numbers a series holds.

    Parameters
    ----------
    y : np.ndarray
        Series values without NaN.

    Returns
    -------
    dict
        ``value_kind`` (count / nonnegative_continuous / signed_integer /
        signed_continuous), zero and negative shares, moments, dispersion index
        and the Syntetos-Boylan intermittency class (``demand_class``).
    """
    n = len(y)
    is_integer = bool(np.all(np.isclose(y, np.round(y))))
    has_negative = bool((y < 0).any())
    if has_negative:
        value_kind = "signed_integer" if is_integer else "signed_continuous"
    else:
        value_kind = "count" if is_integer else "nonnegative_continuous"

    nonzero = y[y != 0]
    mean = float(y.mean())
    std = float(y.std(ddof=1)) if n > 1 else 0.0
    mean_nonzero = float(nonzero.mean()) if len(nonzero) else float("nan")
    adi = float(n / len(nonzero)) if len(nonzero) else float("inf")
    cv2 = (
        float((nonzero.std(ddof=0) / abs(nonzero.mean())) ** 2)
        if len(nonzero) and nonzero.mean() != 0
        else float("nan")
    )
    if len(nonzero) == 0:
        demand_class = "all_zero"
    elif adi < 1.32:
        demand_class = "smooth" if cv2 < 0.49 else "erratic"
    else:
        demand_class = "intermittent" if cv2 < 0.49 else "lumpy"

    return {
        "value_kind": value_kind,
        "is_integer": is_integer,
        "pct_zero": float((y == 0).mean()),
        "pct_negative": float((y < 0).mean()),
        "mean": mean,
        "std": std,
        "min": float(y.min()),
        "max": float(y.max()),
        "mean_nonzero": mean_nonzero,
        "dispersion_index": float(std**2 / mean) if mean > 0 else float("nan"),
        "demand_class": demand_class,
        "adi": adi,
        "cv2": cv2,
    }


def analyze_autocorrelation(
    y: np.ndarray, max_lag: int, min_cycle_lag: int = 2, alpha: float = 0.05
) -> dict[str, Any]:
    """ACF/PACF cycle detection with Ljung-Box significance.

    A cycle is a local ACF peak that is significant and at least
    ``min_cycle_lag`` long; ``confirmed_by_pacf`` marks peaks also supported by
    the PACF. ``max_lag`` is reduced to ``n // 3`` when the sample is short.

    Parameters
    ----------
    y : np.ndarray
        Series values without NaN.
    max_lag : int
        Largest lag to evaluate.
    min_cycle_lag : int
        Smallest lag accepted as a cycle.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Dominant cycle, strength, candidate cycles, significant lags, Ljung-Box
        p-value and the ACF/PACF values.
    """
    from scipy.signal import find_peaks
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, pacf

    if len(np.unique(y)) <= 1:
        raise ConstantSeriesError("Series is constant; ACF is undefined.")
    safe_max_lag = max(1, min(max_lag, len(y) // 3))
    if safe_max_lag < 2:
        raise InsufficientSampleError(f"Only {len(y)} points; need at least 6 for a 2-lag ACF.")

    acf_values, confint = acf(y, nlags=safe_max_lag, fft=False, alpha=alpha)
    pacf_values = pacf(y, nlags=safe_max_lag)
    ljung_box_pvalue = float(acorr_ljungbox(y, lags=[safe_max_lag], return_df=True)["lb_pvalue"].iloc[0])

    significant_mask = (confint[:, 0] > 0) | (confint[:, 1] < 0)
    significant_lags = (np.where(significant_mask[1:])[0] + 1).tolist()
    peak_lags = set((find_peaks(acf_values[1:])[0] + 1).tolist())
    pacf_bound = 1.96 / np.sqrt(len(y))
    candidates = [
        {
            "lag": int(lag),
            "acf_value": float(acf_values[lag]),
            "confirmed_by_pacf": bool(abs(pacf_values[lag]) > pacf_bound),
        }
        for lag in sorted(peak_lags & set(significant_lags))
        if lag >= min_cycle_lag
    ]
    candidates.sort(key=lambda c: c["acf_value"], reverse=True)

    return {
        "dominant_cycle": candidates[0]["lag"] if candidates else None,
        "cycle_strength": candidates[0]["acf_value"] if candidates else 0.0,
        "candidate_cycles": candidates,
        "significant_lags": significant_lags,
        "ljung_box_pvalue": ljung_box_pvalue,
        "has_autocorrelation": ljung_box_pvalue < alpha,
        "acf_values": acf_values.tolist(),
        "pacf_values": pacf_values.tolist(),
        "max_lag_used": safe_max_lag,
    }


def stl_seasonal_strength(y: np.ndarray, period: int) -> float:
    """Seasonal strength ``max(0, 1 - Var(resid) / Var(seasonal + resid))`` from a robust STL fit."""
    from statsmodels.tsa.seasonal import STL

    min_cycles = 3
    if len(np.unique(y)) <= 1:
        raise ConstantSeriesError("Series is constant; STL is undefined.")
    if len(y) < min_cycles * period:
        raise InsufficientSampleError(f"Need {min_cycles * period} points for STL period {period}, got {len(y)}.")
    fit = STL(y, period=period, robust=True).fit()
    resid_var = float(np.var(fit.resid))
    denom = float(np.var(fit.seasonal + fit.resid))
    return max(0.0, 1 - resid_var / denom) if denom > 0 else 0.0


def find_best_seasonal_period(
    y: np.ndarray, min_period: int, max_period: int, fundamental_tolerance: float = 0.02
) -> dict[str, Any]:
    """Search ``[min_period, max_period]`` for the period with the highest STL seasonal strength.

    If the best period is a multiple of a smaller period whose strength is within
    ``fundamental_tolerance``, the smallest such divisor is returned instead.
    """
    results: list[dict[str, Any]] = []
    for period in range(min_period, max_period + 1):
        try:
            results.append({"period": period, "strength": stl_seasonal_strength(y, period)})
        except AnalysisError:
            continue
    best: dict[str, Any] | None = None
    if results:
        best = max(results, key=lambda r: r["strength"])
        # a multiple of the true period scores almost as high; prefer its divisors
        divisors = [
            r
            for r in results
            if r["period"] < best["period"]
            and best["period"] % r["period"] == 0
            and r["strength"] >= best["strength"] - fundamental_tolerance
        ]
        if divisors:
            best = divisors[0]
    if best is None:
        raise InsufficientSampleError(
            f"Not enough points ({len(y)}) for STL at any period in [{min_period}, {max_period}]."
        )
    return best


def detect_seasonality_fft(
    y: np.ndarray,
    min_period: int = 2,
    max_period: int | None = None,
    alpha: float = 0.01,
    n_bootstrap: int = 200,
    seed: int | None = 0,
) -> dict[str, Any]:
    """FFT seasonality test with an AR(1)-whitened bootstrap significance check.

    Independent of STL, so agreement between the two is evidence that a
    detected period is real rather than an artefact of a short sample.
    The AR(1) whitening prevents high-persistence noise (a steep "red"
    spectrum) from being flagged as periodic.

    Parameters
    ----------
    y : np.ndarray
        Series values without NaN.
    min_period, max_period : int
        Period search range; ``max_period`` defaults to ``n // 2``.
    alpha : float
        Bootstrap significance level (stricter than elsewhere on purpose).
    n_bootstrap : int
        Number of AR(1) surrogate series.
    seed : int or None
        Seed of the bootstrap generator.

    Returns
    -------
    dict
        ``has_seasonality``, ``dominant_period``, ``p_value`` and the top peaks.
    """
    heuristic_ratio = 10.0
    min_prominence_ratio = 2.0
    top_k = 5
    noise_exclusion_bins = 3
    boundary_ratio = 0.9
    min_sample = 16

    n_raw = len(y)
    if n_raw < min_sample:
        raise InsufficientSampleError(f"Need at least {min_sample} points for FFT seasonality, got {n_raw}.")
    max_period = min(max_period or n_raw // 2, n_raw // 2)
    min_period = max(min_period, 2)
    if min_period >= max_period:
        raise InsufficientSampleError(f"Invalid FFT period range: {min_period}..{max_period}.")

    x = np.arange(n_raw)
    y = y - np.polyval(np.polyfit(x, y, 2), x)
    std = np.std(y)
    if std < 1e-10:
        raise ConstantSeriesError("Series is near-constant after detrending; FFT is undefined.")
    y = (y - np.mean(y)) / std

    window = signal.windows.hann(n_raw)
    windowed = y * window / np.sqrt(np.mean(window**2))
    psd = np.abs(np.fft.rfft(windowed)) ** 2 / n_raw
    freqs = np.fft.rfftfreq(n_raw, d=1.0)
    valid = (freqs > 0) & (freqs >= 1.0 / max_period) & (freqs <= min(1.0 / min_period, 0.5))

    none_found = {"has_seasonality": False, "dominant_period": None, "p_value": 1.0, "peaks": []}
    if not valid.any():
        return none_found

    valid_idx = np.where(valid)[0]
    psd_valid = psd[valid]
    median_psd = np.median(psd_valid)
    if median_psd <= 0:
        median_psd = np.mean(psd_valid) + 1e-10
    peak_indices, _ = signal.find_peaks(psd_valid, prominence=median_psd * min_prominence_ratio, distance=2)
    if len(peak_indices) == 0:
        return none_found

    candidates = []
    for local in peak_indices:
        noise_mask = np.ones(len(psd_valid), dtype=bool)
        noise_mask[max(0, local - noise_exclusion_bins) : local + noise_exclusion_bins + 1] = False
        noise = np.median(psd_valid[noise_mask]) if noise_mask.any() else median_psd
        noise = noise if noise > 0 else 1e-10
        freq = freqs[valid_idx[local]]
        candidates.append(
            {"period": float(1.0 / freq), "power": float(psd[valid_idx[local]]), "ratio": float(psd[valid_idx[local]] / noise)}
        )
    candidates.sort(key=lambda c: c["ratio"], reverse=True)
    candidates = candidates[:top_k]
    dominant = candidates[0]

    heuristic_pass = dominant["ratio"] >= heuristic_ratio and dominant["period"] <= max_period * boundary_ratio
    if not heuristic_pass:
        return {**none_found, "peaks": candidates}

    rng = np.random.default_rng(seed)
    n_bootstrap = max(n_bootstrap, int(np.ceil(1.0 / alpha)))
    phi = float(np.clip(np.nan_to_num(np.corrcoef(y[:-1], y[1:])[0, 1]), -0.99, 0.99))
    ar1_valid = (1.0 / np.abs(1 - phi * np.exp(-2j * np.pi * freqs)) ** 2)[valid]

    def whitened_ratio(spectrum: np.ndarray) -> float:
        whitened = spectrum / ar1_valid
        med = np.median(whitened)
        return float(np.max(whitened) / med) if med > 0 else 1.0

    observed = whitened_ratio(psd_valid)
    innovation_std = np.sqrt(1 - phi**2)
    exceed = 0
    for _ in range(n_bootstrap):
        noise = rng.normal(0, innovation_std, n_raw)
        surrogate = np.empty(n_raw)
        surrogate[0] = rng.normal(0, 1.0)
        for i in range(1, n_raw):
            surrogate[i] = phi * surrogate[i - 1] + noise[i]
        surrogate = (surrogate - surrogate.mean()) / surrogate.std()
        surrogate_windowed = surrogate * window / np.sqrt(np.mean(window**2))
        surrogate_psd = np.abs(np.fft.rfft(surrogate_windowed)) ** 2 / n_raw
        exceed += whitened_ratio(surrogate_psd[valid]) >= observed
    p_value = (exceed + 1) / (n_bootstrap + 1)

    significant = p_value < alpha
    return {
        "has_seasonality": bool(significant),
        "dominant_period": dominant["period"] if significant else None,
        "p_value": float(p_value),
        "peaks": candidates,
    }


def test_stationarity(y: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """ADF + KPSS combined into ``stationary`` / ``non_stationary`` / ``mixed``."""
    from statsmodels.tsa.stattools import adfuller, kpss

    min_sample = 20
    if len(y) < min_sample:
        raise InsufficientSampleError(f"Need at least {min_sample} points for stationarity tests, got {len(y)}.")
    if len(np.unique(y)) <= 1:
        raise ConstantSeriesError("Series is constant; stationarity is undefined.")
    adf_pvalue = float(adfuller(y, autolag="AIC")[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_pvalue = float(kpss(y, regression="c", nlags="auto")[1])
    adf_stationary = adf_pvalue < alpha
    kpss_stationary = kpss_pvalue >= alpha
    if adf_stationary and kpss_stationary:
        decision = "stationary"
    elif not adf_stationary and not kpss_stationary:
        decision = "non_stationary"
    else:
        decision = "mixed"
    return {"stationarity_decision": decision, "adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue}


def analyze_trend(y: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Linear trend slope, significance and implied total percent change."""
    from scipy import stats as scipy_stats

    min_sample = 10
    if len(y) < min_sample:
        raise InsufficientSampleError(f"Need at least {min_sample} points for trend analysis, got {len(y)}.")
    if len(np.unique(y)) <= 1:
        raise ConstantSeriesError("Series is constant; trend is undefined.")
    slope, intercept, _, p_value, _ = scipy_stats.linregress(np.arange(len(y)), y)
    direction = ("upward" if slope > 0 else "downward") if p_value < alpha else "no_significant_trend"
    end_value = intercept + slope * (len(y) - 1)
    pct_change = float((end_value - intercept) / abs(intercept) * 100) if intercept != 0 else float("nan")
    return {
        "trend_direction": direction,
        "trend_slope": float(slope),
        "trend_pvalue": float(p_value),
        "trend_total_pct_change": pct_change,
    }


def prepare_series(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Validate and regularise one series onto its inferred uniform grid.

    Parameters
    ----------
    df : pd.DataFrame
        Columns ``date`` and ``value`` for a single series.

    Returns
    -------
    tuple of (pd.DataFrame, str)
        Frame with columns ``step``, ``date``, ``value`` (missing periods are
        NaN) and the inferred frequency name.
    """
    clean = df[["date", "value"]].dropna(subset=["date"]).copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean = clean.sort_values("date")
    if clean["date"].duplicated().any():
        raise DuplicateDatesError(f"{int(clean['date'].duplicated().sum())} duplicated dates in series.")
    try:
        freq = classify_frequency(pd.DatetimeIndex(clean["date"]))
        return regularize(clean.reset_index(drop=True), freq), freq
    except FrequencyError as exc:
        raise NonUniformFrequencyError(str(exc)) from exc


def _empty_row() -> dict[str, Any]:
    return {name: None for name in ANALYSIS_COLUMNS}


def analyze_series(df: pd.DataFrame, series_id: str, alpha: float = 0.05, max_lag: int | None = None) -> dict[str, Any]:
    """Run the full analysis suite on one series and return a flat result row.

    Failures inside one stage are recorded in ``stage_errors`` and leave that
    stage's columns empty, so one inapplicable test never hides the others.

    Parameters
    ----------
    df : pd.DataFrame
        Columns ``date`` and ``value`` for one series.
    series_id : str
        Identifier copied into the result.
    alpha : float
        Significance level for the tests.
    max_lag : int, optional
        Largest ACF lag; defaults to the nominal season length of the
        inferred frequency.

    Returns
    -------
    dict
        One value per key of ``ANALYSIS_COLUMNS``.
    """
    row = _empty_row()
    row["series_id"] = str(series_id)
    stage_errors: dict[str, dict[str, str]] = {}
    notes: list[str] = []

    try:
        grid, freq = prepare_series(df)
    except AnalysisError as exc:
        row["stage_errors"] = json.dumps({"prepare": {"category": exc.category, "message": str(exc)}})
        row["warnings"] = "[]"
        return row

    values = grid["value"].to_numpy(dtype=float)
    n_missing = int(np.isnan(values).sum())
    tail_start = contiguous_tail(values)
    window = grid.iloc[tail_start:]
    y = window["value"].to_numpy(dtype=float)
    if tail_start > 0:
        notes.append(f"Series has gaps; analysis uses the latest contiguous run of {len(y)} of {len(grid)} periods.")

    nominal = FREQUENCIES[freq]
    row.update(
        frequency=freq,
        season_length_nominal=nominal,
        n_obs=int(len(df)),
        n_missing_periods=n_missing,
        start_date=str(grid["date"].iloc[0].date()),
        end_date=str(grid["date"].iloc[-1].date()),
        analysis_n=int(len(y)),
    )

    if len(y) < 3:
        stage_errors["describe"] = {"category": "insufficient_sample", "message": f"Only {len(y)} usable points."}
        row["stage_errors"] = json.dumps(stage_errors)
        row["warnings"] = json.dumps(notes)
        return row

    row.update(describe_values(y))
    row["data_quality_flag"] = classify_data_quality(len(y), row["pct_zero"])

    def run_stage(name: str, func: Any) -> dict[str, Any] | None:
        try:
            return func()
        except AnalysisError as exc:
            stage_errors[name] = {"category": exc.category, "message": str(exc)}
            return None

    acf_result = run_stage(
        "autocorrelation", lambda: analyze_autocorrelation(y, max_lag or max(nominal, 2), alpha=alpha)
    )
    if acf_result:
        for key in ("dominant_cycle", "cycle_strength", "ljung_box_pvalue", "has_autocorrelation", "max_lag_used"):
            row[key] = acf_result[key]
        for key in ("candidate_cycles", "significant_lags", "acf_values", "pacf_values"):
            row[key] = json.dumps(acf_result[key])

    max_period = max(2, min(2 * nominal, len(y) // 3))
    stl_result = run_stage("stl", lambda: find_best_seasonal_period(y, 2, max_period))
    fft_result = run_stage("fft", lambda: detect_seasonality_fft(y, max_period=max(3, 2 * nominal)))
    stl_period = stl_result["period"] if stl_result else None
    fft_period = fft_result["dominant_period"] if fft_result and fft_result["has_seasonality"] else None
    row["stl_period"] = stl_period
    row["stl_strength"] = stl_result["strength"] if stl_result else None
    row["fft_period"] = fft_period
    row["fft_pvalue"] = fft_result["p_value"] if fft_result else None
    if stl_period is not None and fft_period is not None:
        row["seasonality_agreement"] = "agree" if abs(stl_period - fft_period) <= 1 else "disagree"
    elif stl_period is not None:
        row["seasonality_agreement"] = "stl_only"
    elif fft_period is not None:
        row["seasonality_agreement"] = "fft_only"
    else:
        row["seasonality_agreement"] = "neither"

    stationarity = run_stage("stationarity", lambda: test_stationarity(y, alpha))
    if stationarity:
        row.update(stationarity)
    trend = run_stage("trend", lambda: analyze_trend(y, alpha))
    if trend:
        row.update(trend)

    row["recommended_params"] = json.dumps(recommend_params(row), default=float)
    row["stage_errors"] = json.dumps(stage_errors)
    row["warnings"] = json.dumps(notes)
    return row
