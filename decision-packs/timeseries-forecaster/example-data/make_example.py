"""Generate a synthetic series + regressor dataset with known structure.

Writes ``series.csv`` (series_id, date, value) and ``regressors.csv``
(series_id, regressor_name, date, value) next to this file. It also writes
``series_hard.csv`` / ``regressors_hard.csv``: the same 30 series plus 10
difficult ones (level shifts, outliers, short histories, gaps, late starts)
that exercise the data-quality gate and the robustness of the models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
N_SERIES = 30
N_PERIODS = 60


def hard_series(rng: np.random.Generator, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ten difficult series, ids ``series_030`` .. ``series_039``."""
    series_rows: list[pd.DataFrame] = []
    regressor_rows: list[pd.DataFrame] = []
    t = np.arange(N_PERIODS)
    for i in range(30, 40):
        kind = ["level_shift", "outlier", "short", "gap", "late_start"][(i - 30) % 5]
        driver = rng.normal(size=N_PERIODS + 3)
        base = 120 + 15 * np.sin(2 * np.pi * t / 12) + 3 * driver[:N_PERIODS] + rng.normal(0, 3, N_PERIODS)
        frame = pd.DataFrame({"date": dates, "value": base})
        if kind == "level_shift":
            frame["value"] = base + np.where(t >= 36, 60.0, 0.0)
        elif kind == "outlier":
            frame.loc[[14, 41], "value"] = [base[14] * 6, base[41] * 0.05]
        elif kind == "short":
            frame = frame.iloc[-20:]
        elif kind == "gap":
            frame = frame.drop(index=[20, 21, 22])
        else:
            frame = frame.iloc[30:]
        sid = f"series_{i:03d}"
        series_rows.append(frame.assign(series_id=sid)[["series_id", "date", "value"]])
        reg_dates = pd.date_range(dates[0] - pd.offsets.MonthEnd(3), periods=N_PERIODS + 3, freq="ME")
        regressor_rows.append(
            pd.DataFrame({"series_id": sid, "regressor_name": "driver", "date": reg_dates, "value": driver})
        )
    return pd.concat(series_rows), pd.concat(regressor_rows)


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
    extra_series, extra_regressors = hard_series(np.random.default_rng(1), dates)
    pd.concat(series_rows + [extra_series]).to_csv(OUT_DIR / "series_hard.csv", index=False)
    pd.concat(regressor_rows + [extra_regressors]).to_csv(OUT_DIR / "regressors_hard.csv", index=False)


if __name__ == "__main__":
    main()
