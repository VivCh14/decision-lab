"""Contract tests for the report figure module.

The figures are produced by ``tsf_lib.report`` rather than by agent-written
plotting code, so their names, ranking rules and summary keys are a property of
the pack and are pinned here.
"""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from tsf_lib.report import (  # noqa: E402
    BASELINE_MODEL,
    accuracy_from_wape,
    accuracy_by_segment,
    baseline_comparison,
    build_report,
    forecast_grid,
    pooled_wape,
    savefig,
)

SEGMENTS = {"a": ["s0", "s1", "s2", "s3"], "b": ["s4", "s5"]}


def _selection() -> pd.DataFrame:
    rows = []
    for segment, ids in SEGMENTS.items():
        for sid in ids:
            rows.append({"series_id": sid, "segment": segment, "model": "good",
                         "metric": "smape", "segment_median_smape": 10.0})
    return pd.DataFrame(rows)


def _scores() -> pd.DataFrame:
    """Scores with a known ordering: s0 best in 'a', s3 worst."""
    smape = {"s0": 1.0, "s1": 2.0, "s2": 3.0, "s3": 4.0, "s4": 5.0, "s5": 6.0}
    rows = []
    for sid, value in smape.items():
        rows.append({"series_id": sid, "model": "good", "smape": value,
                     "smape_median": 10.0 - value})
        rows.append({"series_id": sid, "model": BASELINE_MODEL, "smape": value + 1.0,
                     "smape_median": 11.0 - value})
    return pd.DataFrame(rows)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_ids = [sid for ids in SEGMENTS.values() for sid in ids]
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    series = pd.DataFrame([
        {"series_id": sid, "date": d, "value": 100.0 + i}
        for sid in all_ids for i, d in enumerate(dates)
    ])
    bt_rows, fc_rows, met_rows = [], [], []
    for sid in all_ids:
        for model in ("good", BASELINE_MODEL):
            for fold, origin in enumerate(dates[17:20]):
                for step, target in enumerate(dates[18 + fold:21 + fold], start=1):
                    bt_rows.append({
                        "series_id": sid, "model": model, "fold": fold,
                        "origin_date": origin, "target_date": target, "horizon_step": step,
                        "actual": 110.0, "forecast_mean": 105.0, "forecast_median": 105.0,
                        "forecast_lower": 100.0, "forecast_upper": 115.0,
                        "interval_level": 0.8, "rhat_max": 1.0, "ess_min": 500.0,
                        "divergence_pct": 0.0, "error": None,
                    })
            for step, target in enumerate(dates[-3:], start=1):
                fc_rows.append({
                    "series_id": sid, "model": model, "origin_date": dates[-4],
                    "target_date": target, "horizon_step": step,
                    "forecast_mean": 106.0, "forecast_median": 106.0,
                    "forecast_lower": 101.0, "forecast_upper": 116.0,
                    "interval_level": 0.8, "rhat_max": 1.0, "ess_min": 500.0,
                    "divergence_pct": 0.0, "error": None,
                })
            met_rows.append({"series_id": sid, "model": model, "wape": 0.25, "bias": 0.0,
                             "mae": 5.0, "rmse": 5.0, "mase": 0.5, "coverage": 0.8,
                             "nominal_level": 0.8, "interval_width": 15.0,
                             "interval_score": 20.0})
    return series, pd.DataFrame(bt_rows), pd.DataFrame(fc_rows), pd.DataFrame(met_rows)


def test_accuracy_is_unbounded_below():
    """Accuracy is 1 - WAPE and must never be clipped at zero."""
    assert accuracy_from_wape(0.25) == pytest.approx(0.75)
    assert accuracy_from_wape(5.0) == pytest.approx(-4.0)
    assert np.isnan(accuracy_from_wape(float("nan")))


def test_pooled_wape_weights_by_volume():
    frame = pd.DataFrame({"actual": [100.0, 1.0], "forecast_mean": [110.0, 2.0]})
    # 11 absolute error over 101 actual, not the mean of the two series' rates.
    assert pooled_wape(frame) == pytest.approx(11.0 / 101.0)


def test_pooled_wape_is_nan_without_scored_rows():
    frame = pd.DataFrame({"actual": [np.nan], "forecast_mean": [np.nan]})
    assert np.isnan(pooled_wape(frame))


def test_savefig_writes_and_closes(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = savefig(fig, tmp_path / "out", "01_thing")
    assert path.exists() and path.name == "01_thing.png"
    assert not plt.get_fignums(), "savefig must close the figure"


def test_grids_rank_on_the_requested_basis(tmp_path):
    series, backtests, forecasts, metrics = _frames()
    selection, scores = _selection(), _scores()

    _, best = forecast_grid(selection, scores, metrics, series, backtests, forecasts,
                            tmp_path, worst=False, top_n=3)
    _, worst = forecast_grid(selection, scores, metrics, series, backtests, forecasts,
                             tmp_path, worst=True, top_n=3)

    best_a = [r["series_id"] for r in best if r["segment"] == "a"]
    worst_a = [r["series_id"] for r in worst if r["segment"] == "a"]
    assert best_a == ["s0", "s1", "s2"], "best grid must rank ascending on smape"
    assert worst_a == ["s3", "s2", "s1"], "worst grid must rank descending on smape"


def test_grid_segment_smaller_than_grid_is_flagged(tmp_path):
    """Segment 'b' has 2 series and 3 columns: it fills 2 and says 'all 2'."""
    series, backtests, forecasts, metrics = _frames()
    _, records = forecast_grid(_selection(), _scores(), metrics, series, backtests,
                               forecasts, tmp_path, worst=False, top_n=3)
    assert len([r for r in records if r["segment"] == "b"]) == 2


def test_grid_reports_accuracy_from_wape(tmp_path):
    series, backtests, forecasts, metrics = _frames()
    _, records = forecast_grid(_selection(), _scores(), metrics, series, backtests,
                               forecasts, tmp_path, worst=False, top_n=3)
    assert all(r["accuracy"] == pytest.approx(0.75) for r in records)


def test_absent_model_gets_no_bar(tmp_path):
    """A model that never scored a segment must be omitted, not drawn as zero."""
    _, backtests, _, _ = _frames()
    # Drop the "good" model from segment b only, leaving the baseline there.
    backtests = backtests[~(backtests["series_id"].isin(SEGMENTS["b"])
                            & (backtests["model"] == "good"))]
    _, records = accuracy_by_segment(_selection(), backtests, tmp_path)
    pairs = {(r["segment"], r["model"]) for r in records}
    assert ("b", "good") not in pairs
    assert ("b", BASELINE_MODEL) in pairs


def test_baseline_comparison_marks_wins(tmp_path):
    """The fixture gives 'good' a lower smape than the baseline on every series."""
    _, records = baseline_comparison(_selection(), _scores(), tmp_path)
    assert records and all(r["beats_baseline"] for r in records)


def test_build_report_writes_every_figure_and_summary(tmp_path):
    series, backtests, forecasts, metrics = _frames()
    paths = {}
    for name, frame in (("selection", _selection()), ("scores", _scores()),
                        ("metrics", metrics), ("series", series)):
        paths[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(paths[name], index=False)
    bt_path, fc_path = tmp_path / "bt.parquet", tmp_path / "fc.parquet"
    backtests.to_parquet(bt_path, index=False)
    forecasts.to_parquet(fc_path, index=False)

    out = tmp_path / "report_figures"
    summary = build_report(
        selection_file=str(paths["selection"]), scores_file=str(paths["scores"]),
        metrics_file=str(paths["metrics"]), series_file=str(paths["series"]),
        backtest_files=[str(bt_path)], forecast_files=[str(fc_path)],
        output_dir=str(out),
    )

    expected = {
        "01_accuracy_by_segment_and_model", "02_best_forecasts_by_segment",
        "03_worst_forecasts_by_segment", "04_wape_by_horizon_step",
        "05_interval_coverage_vs_nominal", "06_baseline_comparison_per_series",
    }
    assert expected <= set(summary["figures"])
    for name in expected:
        assert (out / f"{name}.png").exists()

    written = json.loads((out / "report_summary.json").read_text())
    assert written["segments"] == ["a", "b"]
    assert written["best_series"] and written["worst_series"]
    assert written["baseline"] == BASELINE_MODEL


def test_build_report_rejects_missing_columns(tmp_path):
    series, backtests, forecasts, metrics = _frames()
    bad = _selection().drop(columns=["segment"])
    bad_path = tmp_path / "selection.parquet"
    bad.to_parquet(bad_path, index=False)
    for name, frame in (("scores", _scores()), ("metrics", metrics), ("series", series)):
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    backtests.to_parquet(tmp_path / "bt.parquet", index=False)
    forecasts.to_parquet(tmp_path / "fc.parquet", index=False)

    with pytest.raises(ValueError, match="selection is missing required columns"):
        build_report(
            selection_file=str(bad_path), scores_file=str(tmp_path / "scores.parquet"),
            metrics_file=str(tmp_path / "metrics.parquet"),
            series_file=str(tmp_path / "series.parquet"),
            backtest_files=[str(tmp_path / "bt.parquet")],
            forecast_files=[str(tmp_path / "fc.parquet")],
            output_dir=str(tmp_path / "figs"),
        )
