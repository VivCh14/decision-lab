"""Generate a synthetic series + regressor dataset with known structure.

Writes ``series.csv`` (series_id, date, value) and ``regressors.csv``
(series_id, regressor_name, date, value) next to this file; upload them to
Unity Catalog tables to try the pack end to end.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
N_SERIES = 30
N_PERIODS = 60


def main() -> None:
    """Create a mix of seasonal, trending, count and intermittent monthly series."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-01-31", periods=N_PERIODS, freq="ME")
    series_rows: list[pd.DataFrame] = []
    regressor_rows: list[pd.DataFrame] = []
    for i in range(N_SERIES):
        kind = ["seasonal", "trend", "count", "intermittent"][i % 4]
        t = np.arange(N_PERIODS)
        driver = rng.normal(size=N_PERIODS + 3)
        if kind == "seasonal":
            value = 200 + 30 * np.sin(2 * np.pi * t / 12) + 4 * driver[: N_PERIODS] + rng.normal(0, 3, N_PERIODS)
        elif kind == "trend":
            value = 50 + 1.5 * t + rng.normal(0, 5, N_PERIODS)
        elif kind == "count":
            value = rng.poisson(np.exp(2 + 0.3 * np.sin(2 * np.pi * t / 12))).astype(float)
        else:
            value = np.where(rng.random(N_PERIODS) < 0.25, rng.integers(1, 6, N_PERIODS), 0).astype(float)
        sid = f"series_{i:03d}"
        series_rows.append(pd.DataFrame({"series_id": sid, "date": dates, "value": value}))
        reg_dates = pd.date_range(dates[0] - pd.offsets.MonthEnd(3), periods=N_PERIODS + 3, freq="ME")
        regressor_rows.append(
            pd.DataFrame({"series_id": sid, "regressor_name": "driver", "date": reg_dates, "value": driver})
        )
    pd.concat(series_rows).to_csv(OUT_DIR / "series.csv", index=False)
    pd.concat(regressor_rows).to_csv(OUT_DIR / "regressors.csv", index=False)


if __name__ == "__main__":
    main()
