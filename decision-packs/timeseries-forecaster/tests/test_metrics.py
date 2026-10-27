"""Post-processing metrics and convergence flags computed from backtest tables."""

import numpy as np
import pandas as pd
import pytest

from tsf_lib.backtest import BACKTEST_COLUMNS, score_backtest
from tsf_lib.metrics import compute_metrics, nonconverged_folds


def _backtest(rhat: float = 1.0, ess: float = 500.0, div: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    actual = np.arange(10.0, 22.0)
    rows = []
    for fold, (origin, targets) in enumerate([(dates[5], dates[6:9]), (dates[8], dates[9:12])]):
        for step, target in enumerate(targets, start=1):
            a = float(actual[list(dates).index(target)])
            rows.append(
                {
                    "series_id": "s1", "model": "m", "fold": fold, "origin_date": origin, "target_date": target,
                    "horizon_step": step, "actual": a, "forecast_mean": a + 2.0, "forecast_median": a + 2.0,
                    "forecast_lower": a + 1.0, "forecast_upper": a + 3.0, "interval_level": 0.8,
                    "rhat_max": rhat, "ess_min": ess, "divergence_pct": div, "error": None,
                }
            )
    return pd.DataFrame(rows)[list(BACKTEST_COLUMNS)]


def _series() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    return pd.DataFrame({"series_id": "s1", "date": dates, "value": np.arange(10.0, 22.0)})


def test_metrics_values() -> None:
    out = compute_metrics(_backtest(), _series(), None).iloc[0]
    assert out["mae"] == pytest.approx(2.0)
    assert out["bias"] == pytest.approx(12.0 / 111.0)
    assert out["wape"] == pytest.approx(12.0 / 111.0)
    assert out["mase"] == pytest.approx(2.0)
    assert out["coverage"] == 0.0 and out["nominal_level"] == 0.8
    assert out["interval_width"] == pytest.approx(2.0)
    assert out["interval_score"] == pytest.approx(2.0 + 2 / 0.2 * 1.0)


def test_convergence_flags_in_scores() -> None:
    good = score_backtest(_backtest()).iloc[0]
    assert good["n_nonconverged_folds"] == 0 and good["max_rhat"] == pytest.approx(1.0)
    for kwargs in ({"rhat": 1.5}, {"ess": 50.0}, {"div": 5.0}):
        bad = _backtest(**kwargs)
        assert score_backtest(bad).iloc[0]["n_nonconverged_folds"] == 2
        assert len(nonconverged_folds(bad)) == 2


def test_non_bayesian_models_have_no_diagnostics() -> None:
    frame = _backtest(rhat=np.nan, ess=np.nan, div=np.nan)
    row = score_backtest(frame).iloc[0]
    assert row["n_nonconverged_folds"] == 0 and np.isnan(row["max_rhat"])


def test_unknown_series_raises() -> None:
    other = _series().assign(series_id="other")
    with pytest.raises(ValueError, match="not in the series table"):
        compute_metrics(_backtest(), other, None)
