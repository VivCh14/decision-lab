"""Baseline model: seasonal naive (repeat the last season), flat for non-seasonal series."""

import numpy as np
import pandas as pd


def fit_forecast(
    train: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    regressors: "pd.DataFrame | None",
    analysis: dict,
    params: dict,
) -> pd.DataFrame:
    """Forecast by repeating the last ``season_length`` observations.

    ``season_length`` comes from ``params`` or, if absent, from the analysis
    recommendation (1 means a flat last-value forecast).
    """
    y = train["value"].to_numpy(dtype=float)
    period = int(params.get("season_length") or analysis["recommended"].get("season_length") or 1)
    period = max(1, min(period, len(y)))
    last_season = y[-period:]
    horizon = len(future_dates)
    forecast = np.array([last_season[i % period] for i in range(horizon)])
    return pd.DataFrame({"forecast_mean": forecast, "forecast_median": forecast})
