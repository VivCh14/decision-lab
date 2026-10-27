"""Model selection per segment from backtest scores."""

import pandas as pd

METRICS = {"smape": "forecast_mean", "smape_median": "forecast_median"}


def select_models(
    scores: pd.DataFrame, analysis: pd.DataFrame, by: str, metric: "str | dict[str, str]" = "smape"
) -> pd.DataFrame:
    """Choose, for each segment, the model with the lowest median score.

    Parameters
    ----------
    scores : pd.DataFrame
        Output of ``score_backtest`` (``series_id``, ``model``, ``smape``,
        ``smape_median``).
    analysis : pd.DataFrame
        Analysis table holding the segment column ``by``.
    by : str
        Analysis column defining segments (e.g. ``demand_class``).
    metric : str or dict
        ``smape`` (mean forecast) or ``smape_median`` (median forecast), or a
        mapping from segment name to metric (unlisted segments use ``smape``).
        Median-forecast scoring is the fair choice for intermittent demand,
        where the mean is rarely zero but the typical month is.

    Returns
    -------
    pd.DataFrame
        One row per series: ``series_id``, ``segment``, ``model``, ``metric``,
        ``segment_median_smape`` (the median of the chosen metric).
    """
    if by not in analysis.columns:
        raise ValueError(f"Segment column '{by}' not in analysis; found {list(analysis.columns)}.")
    metrics = {metric: None} if isinstance(metric, str) else metric
    unknown = [m for m in ([metric] if isinstance(metric, str) else metric.values()) if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metric {unknown}; use one of {sorted(METRICS)}.")
    segments = analysis[["series_id", by]].rename(columns={by: "segment"})
    segments["segment"] = segments["segment"].fillna("unknown").astype(str)
    joined = scores.merge(segments, on="series_id")
    best_rows = []
    for segment, group in joined.groupby("segment"):
        chosen = metric if isinstance(metric, str) else metric.get(segment, "smape")
        medians = group.dropna(subset=[chosen]).groupby("model")[chosen].median().reset_index(name="score")
        if medians.empty:
            continue
        top = medians.sort_values(["score", "model"]).iloc[0]
        best_rows.append(
            {"segment": segment, "model": top["model"], "metric": chosen, "segment_median_smape": top["score"]}
        )
    best = pd.DataFrame(best_rows, columns=["segment", "model", "metric", "segment_median_smape"])
    return segments.merge(best, on="segment", how="left")[
        ["series_id", "segment", "model", "metric", "segment_median_smape"]
    ]


def assemble_forecast(forecasts: dict[str, pd.DataFrame], selection: pd.DataFrame) -> pd.DataFrame:
    """Keep each series' forecast from its selected model and add the point forecast.

    ``point_forecast`` is the mean forecast, or the median forecast for series
    whose segment was selected by ``smape_median``; ``point_basis`` names which.

    Parameters
    ----------
    forecasts : dict of str to pd.DataFrame
        Forecast table per model name.
    selection : pd.DataFrame
        Output of ``select_models``.
    """
    parts = []
    for (model, metric), group in selection.dropna(subset=["model"]).groupby(["model", "metric"]):
        if model not in forecasts:
            raise ValueError(f"No forecast table for selected model '{model}'.")
        frame = forecasts[model]
        frame = frame[frame["series_id"].isin(group["series_id"])].copy()
        basis = METRICS[metric]
        frame["point_forecast"] = frame[basis].where(frame[basis].notna(), frame["forecast_mean"])
        frame["point_basis"] = basis
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
