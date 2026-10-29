"""Report figures for a finished forecasting session.

All figures are produced by this module rather than by agent-written plotting
code, so their layout and naming are a property of the pack and not of whichever
model happened to run. Figures land in ``output_dir`` with a numbered prefix
giving reading order, alongside ``report_summary.json`` holding the same numbers
in machine-readable form.

The house style is applied through ``MATPLOTLIBRC`` and ``dlab_plotstyle``; this
module never sets a style or hard-codes a colour outside the cycle.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .runner import read_table  # noqa: E402

BASELINE_MODEL = "seasonal_naive"
TOP_N = 3
FIGURE_DPI = 150

# Badge colours: the best grid reads as neutral/positive, the worst as a warning.
# Applied as text colour, not a box fill - the house style drops text bboxes.
BADGE_BEST = "#3B4AB8"
BADGE_WORST = "#C1352B"


def savefig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    """Save a figure as PNG and close it.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    output_dir : Path
        Destination directory, created if missing.
    name : str
        File stem, including the numbered prefix.

    Returns
    -------
    Path
        The written path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = (output_dir / f"{name}.png").resolve()
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def accuracy_from_wape(wape: float) -> float:
    """Business-readable accuracy for a WAPE.

    Accuracy is ``1 - WAPE`` and is deliberately unbounded below: a model far
    worse than the actuals scores strongly negative. Never clip it to zero and
    never average it; a clipped or averaged accuracy hides exactly the failures
    the worst-case grid exists to show.

    Parameters
    ----------
    wape : float
        Weighted absolute percentage error.

    Returns
    -------
    float
        Accuracy as a fraction, or NaN when ``wape`` is not finite.
    """
    if wape is None or not np.isfinite(wape):
        return float("nan")
    return 1.0 - float(wape)


def pooled_wape(backtest: pd.DataFrame) -> float:
    """WAPE pooled over every scored row, weighting series by volume.

    Parameters
    ----------
    backtest : pd.DataFrame
        Backtest rows with ``actual`` and ``forecast_mean``.

    Returns
    -------
    float
        Pooled WAPE, or NaN when nothing scored.
    """
    scored = backtest.dropna(subset=["actual", "forecast_mean"])
    if scored.empty:
        return float("nan")
    denom = scored["actual"].abs().sum()
    if denom <= 0:
        return float("nan")
    return float((scored["forecast_mean"] - scored["actual"]).abs().sum() / denom)


def _segment_order(selection: pd.DataFrame) -> list[str]:
    return sorted(selection["segment"].dropna().astype(str).unique())


def _rank_series(
    selection: pd.DataFrame, scores: pd.DataFrame, segment: str, worst: bool, top_n: int
) -> pd.DataFrame:
    """Rank a segment's series by its selected model's score.

    Ranking uses the metric the segment was actually selected on, so the figure
    agrees with ``select_models`` instead of imposing a different basis.
    """
    seg = selection[selection["segment"].astype(str) == segment]
    if seg.empty or seg["model"].isna().all():
        return pd.DataFrame()
    model = seg["model"].dropna().iloc[0]
    basis = seg["metric"].dropna().iloc[0] if seg["metric"].notna().any() else "smape"
    if basis not in scores.columns:
        basis = "smape"
    joined = seg[["series_id"]].merge(
        scores[scores["model"] == model][["series_id", basis]], on="series_id", how="inner"
    )
    joined = joined.dropna(subset=[basis]).sort_values(basis, ascending=not worst)
    joined["model"] = model
    joined["basis"] = basis
    return joined.head(top_n)


def _panel(
    ax: plt.Axes,
    history: pd.DataFrame,
    backtest: pd.DataFrame,
    forecast: pd.DataFrame,
    segment: str,
    badge_color: str,
    badge_suffix: str = "",
) -> None:
    """Draw one series panel: history, backtest folds, and the forecast."""
    ax.plot(history["date"], history["value"], marker="o", markersize=2.5,
            linewidth=1.0, label="Historical")

    scored = backtest.dropna(subset=["forecast_mean"])
    if not scored.empty:
        # One shaded span over the whole backtest region. Shading each fold
        # separately would overlap into mud; the folds are carried by their
        # individual prediction traces instead.
        lo, hi = scored["target_date"].min(), scored["target_date"].max()
        span_months = max(int(round((hi - lo).days / 30.4)) + 1, 1)
        ax.axvspan(lo, hi, alpha=0.18, color="#E8A33D",
                   label=f"Backtest ({span_months}m)", zorder=0)
        actuals = scored.dropna(subset=["actual"])
        if not actuals.empty:
            ax.plot(actuals["target_date"], actuals["actual"], linestyle="none",
                    marker="o", markersize=5, label="Actual (in backtest)")
        # Every fold is the same series in the legend's eyes, so they must share
        # one colour; letting the cycle advance per fold would contradict the
        # single "Backtest pred" entry.
        fold_color = None
        for _, fold in scored.groupby("fold", dropna=False):
            fold = fold.sort_values("target_date")
            line, = ax.plot(fold["target_date"], fold["forecast_mean"], linestyle="--",
                            marker="^", markersize=3.5, linewidth=1.0,
                            color=fold_color,
                            label="Backtest pred" if fold_color is None else None)
            fold_color = line.get_color()

    if not forecast.empty:
        fc = forecast.sort_values("target_date")
        basis = "forecast_median" if fc["forecast_median"].notna().any() else "forecast_mean"
        line, = ax.plot(fc["target_date"], fc[basis], marker="s", markersize=4,
                        linewidth=1.4, label="Forecast")
        fc_color = line.get_color()
        if not history.empty:
            # Dashed connector so there is no visual gap at the forecast origin.
            ax.plot([history["date"].iloc[-1], fc["target_date"].iloc[0]],
                    [history["value"].iloc[-1], fc[basis].iloc[0]],
                    linestyle="--", linewidth=1.0, color=fc_color)
        if {"forecast_lower", "forecast_upper"} <= set(fc.columns) and fc["forecast_lower"].notna().any():
            # Band over the out-of-sample region only: bands on the backtest
            # folds as well made low-count panels read as filled bars.
            ax.fill_between(fc["target_date"], fc["forecast_lower"], fc["forecast_upper"],
                            alpha=0.18, color=fc_color)

    # Coloured bold text, NOT a boxed label: dlab_plotstyle deliberately drops the
    # `bbox` kwarg from text ("the box competes with the data"), which would leave
    # a white-on-nothing badge. The colour alone carries best-vs-worst.
    ax.text(0.99, 0.97, segment.replace("_", " ").upper() + badge_suffix,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold", color=badge_color)
    ax.set_ylabel("Qty")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=6, loc="upper left")


def forecast_grid(
    selection: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: pd.DataFrame,
    series: pd.DataFrame,
    backtests: pd.DataFrame,
    forecasts: pd.DataFrame,
    output_dir: Path,
    worst: bool,
    top_n: int = TOP_N,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Grid of per-series panels: rows are segments, columns are rank 1..top_n.

    Parameters
    ----------
    selection : pd.DataFrame
        ``selection.parquet``: ``series_id``, ``segment``, ``model``, ``metric``.
    scores : pd.DataFrame
        ``scores.parquet``: per series and model ``smape`` / ``smape_median``.
    metrics : pd.DataFrame
        ``metrics.parquet``: per series and model ``wape`` and friends.
    series : pd.DataFrame
        Cleaned long series: ``series_id``, ``date``, ``value``.
    backtests, forecasts : pd.DataFrame
        Stacked backtest and forecast tables over every model.
    output_dir : Path
        Destination directory.
    worst : bool
        Rank ascending (best) or descending (worst).
    top_n : int
        Panels per segment.

    Returns
    -------
    tuple
        The written path (None when nothing could be ranked) and the per-panel
        records that go into ``report_summary.json``.
    """
    segments = _segment_order(selection)
    picked = {s: _rank_series(selection, scores, s, worst, top_n) for s in segments}
    segments = [s for s in segments if not picked[s].empty]
    if not segments:
        return None, []

    n_rows, n_cols = len(segments), top_n
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 3.4 * n_rows), squeeze=False)
    badge = BADGE_WORST if worst else BADGE_BEST
    records: list[dict[str, Any]] = []

    for r, segment in enumerate(segments):
        rows = picked[segment]
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(rows):
                ax.set_visible(False)
                continue
            row = rows.iloc[c]
            sid, model, basis = row["series_id"], row["model"], row["basis"]
            hist = series[series["series_id"] == sid].sort_values("date")
            bt = backtests[(backtests["series_id"] == sid) & (backtests["model"] == model)]
            fc = forecasts[(forecasts["series_id"] == sid) & (forecasts["model"] == model)]
            # When a segment has no more series than the grid has columns, the
            # best and worst grids show the same set and the ranking is not
            # selective. Say so on the badge rather than implying a choice.
            n_in_segment = int((selection["segment"].astype(str) == segment).sum())
            suffix = f" (all {n_in_segment})" if n_in_segment <= top_n else ""
            _panel(ax, hist, bt, fc, segment, badge, suffix)

            met = metrics[(metrics["series_id"] == sid) & (metrics["model"] == model)]
            wape = float(met["wape"].iloc[0]) if len(met) and "wape" in met else float("nan")
            acc = accuracy_from_wape(wape)
            title = f"{sid}\n{model}  WAPE={wape:.3f}  acc={acc:.1%}" if np.isfinite(wape) \
                else f"{sid}\n{model}  {basis}={row[basis]:.2f}"
            if model == BASELINE_MODEL:
                title += "  [baseline retained]"
            ax.set_title(title, fontsize=9)
            records.append({
                "segment": segment, "rank": c + 1, "series_id": str(sid), "model": str(model),
                "ranking_basis": str(basis), "score": float(row[basis]),
                "wape": None if not np.isfinite(wape) else wape,
                "accuracy": None if not np.isfinite(acc) else acc,
                "baseline_retained": bool(model == BASELINE_MODEL),
            })

    which = "worst" if worst else "top"
    fig.suptitle(
        f"Historical + Best-Model Forecast ({which} {top_n} accuracy per segment)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    name = "03_worst_forecasts_by_segment" if worst else "02_best_forecasts_by_segment"
    return savefig(fig, output_dir, name), records


def accuracy_by_segment(
    selection: pd.DataFrame, backtests: pd.DataFrame, output_dir: Path
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Grouped bars of pooled accuracy: x is segment, one bar per model.

    A model that never scored a segment gets no bar at all rather than a
    zero-height one, which would read as "perfectly wrong" instead of "not run".
    """
    seg_map = selection[["series_id", "segment"]].drop_duplicates()
    joined = backtests.merge(seg_map, on="series_id", how="inner")
    if joined.empty:
        return None, []

    segments = sorted(joined["segment"].astype(str).unique())
    # Baseline first so every group starts from the reference it is judged against.
    others = sorted(m for m in joined["model"].astype(str).unique() if m != BASELINE_MODEL)
    models = ([BASELINE_MODEL] if BASELINE_MODEL in set(joined["model"].astype(str)) else []) + others
    records: list[dict[str, Any]] = []
    table: dict[str, dict[str, float]] = {}
    for segment in segments:
        table[segment] = {}
        for model in models:
            part = joined[(joined["segment"].astype(str) == segment) & (joined["model"].astype(str) == model)]
            if part.empty:
                continue
            acc = accuracy_from_wape(pooled_wape(part))
            if not np.isfinite(acc):
                continue
            table[segment][model] = acc
            records.append({
                "segment": segment, "model": model, "pooled_accuracy": acc,
                "n_series": int(part["series_id"].nunique()),
            })

    fig, ax = plt.subplots(figsize=(max(1.8 * len(segments) + 3.0, 7.0), 4.5))
    # Bars are packed within each segment rather than held at a fixed per-model
    # offset: a model absent from a segment leaves no gap, so each group reads as
    # a dense comparison and the legend carries the identity.
    colors = {model: f"C{i}" for i, model in enumerate(models)}
    width_for = {s: 0.8 / max(len(table[s]), 1) for s in segments}
    for j, segment in enumerate(segments):
        present = [m for m in models if m in table[segment]]
        w = width_for[segment]
        for k, model in enumerate(present):
            ax.bar(j - 0.4 + (k + 0.5) * w, table[segment][model], width=w,
                   color=colors[model],
                   label=model if j == next(i for i, sg in enumerate(segments)
                                            if model in table[sg]) else None)
    ax.axhline(0.0, linewidth=1.0, color="0.3", zorder=1)
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels(segments)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Pooled accuracy (1 - WAPE)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Pooled accuracy by segment", fontsize=12, fontweight="bold")
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    return savefig(fig, output_dir, "01_accuracy_by_segment_and_model"), records


def smape_by_horizon(
    selection: pd.DataFrame, backtests: pd.DataFrame, output_dir: Path
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Error growth across horizon steps, one line per segment.

    Accuracy at step 1 and at the final step are different claims; a single
    headline number hides which one a reader is relying on.
    """
    seg_map = selection[["series_id", "segment", "model"]].drop_duplicates()
    joined = backtests.merge(seg_map, on=["series_id", "model"], how="inner")
    joined = joined.dropna(subset=["actual", "forecast_mean", "horizon_step"])
    if joined.empty:
        return None, []

    records: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for segment, group in joined.groupby("segment"):
        rows = []
        for step, part in group.groupby("horizon_step"):
            value = pooled_wape(part)
            if np.isfinite(value):
                rows.append((int(step), value))
                records.append({"segment": str(segment), "horizon_step": int(step), "pooled_wape": value})
        if rows:
            rows.sort()
            ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label=str(segment))
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Pooled WAPE")
    ax.set_title("Error by horizon step", fontsize=12, fontweight="bold")
    ax.legend(title="Segment", fontsize=8)
    fig.tight_layout()
    return savefig(fig, output_dir, "04_wape_by_horizon_step"), records


def coverage_vs_nominal(
    selection: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Realised interval coverage against the nominal level, per segment."""
    if "coverage" not in metrics.columns:
        return None, []
    seg_map = selection[["series_id", "segment", "model"]].drop_duplicates()
    joined = metrics.merge(seg_map, on=["series_id", "model"], how="inner").dropna(subset=["coverage"])
    if joined.empty:
        return None, []

    segments = sorted(joined["segment"].astype(str).unique())
    data = [joined.loc[joined["segment"].astype(str) == s, "coverage"].to_numpy() for s in segments]
    nominal = float(joined["nominal_level"].dropna().iloc[0]) if joined["nominal_level"].notna().any() else np.nan

    fig, ax = plt.subplots(figsize=(1.4 * max(len(segments), 1) + 4, 4.5))
    ax.boxplot(data, tick_labels=segments)
    if np.isfinite(nominal):
        ax.axhline(nominal, linestyle="--", linewidth=1.2, color="0.3",
                   label=f"Nominal {nominal:.0%}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Realised coverage")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Interval coverage vs nominal", fontsize=12, fontweight="bold")
    fig.tight_layout()
    records = [
        {"segment": s, "median_coverage": float(np.nanmedian(d)), "nominal_level": nominal, "n_series": int(len(d))}
        for s, d in zip(segments, data) if len(d)
    ]
    return savefig(fig, output_dir, "05_interval_coverage_vs_nominal"), records


def baseline_comparison(
    selection: pd.DataFrame, scores: pd.DataFrame, output_dir: Path,
    baseline: str = BASELINE_MODEL,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Per-series selected model against the baseline, on the same series.

    Points below the diagonal are series where the selected model beats the
    baseline. This is the ACCEPT/REJECT evidence in one picture.
    """
    if baseline not in set(scores["model"].astype(str)):
        return None, []
    base = scores[scores["model"].astype(str) == baseline][["series_id", "smape"]].rename(
        columns={"smape": "baseline_smape"}
    )
    chosen = selection.dropna(subset=["model"])[["series_id", "segment", "model", "metric"]]
    sel = chosen.merge(scores, on=["series_id", "model"], how="inner")
    if sel.empty:
        return None, []
    joined = sel.merge(base, on="series_id", how="inner").dropna(subset=["smape", "baseline_smape"])
    joined = joined[joined["model"].astype(str) != baseline]
    if joined.empty:
        return None, []

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    for segment, group in joined.groupby("segment"):
        ax.scatter(group["baseline_smape"], group["smape"], s=28, label=str(segment))
    limit = float(np.nanmax([joined["smape"].max(), joined["baseline_smape"].max()])) * 1.05
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1.0, color="0.3")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel(f"{baseline} sMAPE")
    ax.set_ylabel("Selected model sMAPE")
    ax.set_title("Selected model vs baseline (below the line = better)",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Segment", fontsize=8)
    fig.tight_layout()
    records = [
        {"series_id": str(r.series_id), "segment": str(r.segment), "model": str(r.model),
         "selected_smape": float(r.smape), "baseline_smape": float(r.baseline_smape),
         "beats_baseline": bool(r.smape < r.baseline_smape)}
        for r in joined.itertuples()
    ]
    return savefig(fig, output_dir, "06_baseline_comparison_per_series"), records


def _stack(paths: list[str]) -> pd.DataFrame:
    """Read and concatenate parquet tables, tolerating an empty list."""
    frames = [read_table(Path(p)) for p in paths or []]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_report(
    selection_file: str,
    scores_file: str,
    metrics_file: str,
    series_file: str,
    backtest_files: list[str],
    forecast_files: list[str],
    output_dir: str = "report_figures",
    top_n: int = TOP_N,
    baseline: str = BASELINE_MODEL,
) -> dict[str, Any]:
    """Build every report figure and the machine-readable summary.

    Parameters
    ----------
    selection_file : str
        ``selection.parquet`` from ``select-models``; supplies the segment and
        the selected model and metric basis per series.
    scores_file, metrics_file : str
        ``scores.parquet`` and ``metrics.parquet`` from ``score-models`` and
        ``compute-metrics``.
    series_file : str
        Cleaned long series (``series_id``, ``date``, ``value``) for history.
    backtest_files, forecast_files : list[str]
        Every model's backtest and forecast parquet, including the baseline's.
    output_dir : str
        Destination directory for figures and ``report_summary.json``.
    top_n : int
        Panels per segment in the best and worst grids.
    baseline : str
        Baseline model name, kept in every comparison.

    Returns
    -------
    dict[str, Any]
        The summary document, also written to ``report_summary.json``.
    """
    out = Path(output_dir)
    selection = read_table(Path(selection_file))
    scores = read_table(Path(scores_file))
    metrics = read_table(Path(metrics_file))
    series = read_table(Path(series_file))
    backtests = _stack(backtest_files)
    forecasts = _stack(forecast_files)

    for frame, name, cols in (
        (selection, "selection", {"series_id", "segment", "model"}),
        (scores, "scores", {"series_id", "model", "smape"}),
        (series, "series", {"series_id", "date", "value"}),
        (backtests, "backtests", {"series_id", "model", "target_date"}),
    ):
        missing = cols - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    summary: dict[str, Any] = {
        "baseline": baseline,
        "top_n": top_n,
        "segments": _segment_order(selection),
        "models": sorted(scores["model"].astype(str).unique()),
        "figures": {},
    }

    acc_path, acc_records = accuracy_by_segment(selection, backtests, out)
    summary["pooled_accuracy_by_segment"] = acc_records
    if acc_path:
        summary["figures"]["01_accuracy_by_segment_and_model"] = str(acc_path)

    best_path, best_records = forecast_grid(
        selection, scores, metrics, series, backtests, forecasts, out, worst=False, top_n=top_n
    )
    summary["best_series"] = best_records
    if best_path:
        summary["figures"]["02_best_forecasts_by_segment"] = str(best_path)

    worst_path, worst_records = forecast_grid(
        selection, scores, metrics, series, backtests, forecasts, out, worst=True, top_n=top_n
    )
    summary["worst_series"] = worst_records
    if worst_path:
        summary["figures"]["03_worst_forecasts_by_segment"] = str(worst_path)

    horizon_path, horizon_records = smape_by_horizon(selection, backtests, out)
    summary["wape_by_horizon_step"] = horizon_records
    if horizon_path:
        summary["figures"]["04_wape_by_horizon_step"] = str(horizon_path)

    cov_path, cov_records = coverage_vs_nominal(selection, metrics, out)
    summary["coverage_by_segment"] = cov_records
    if cov_path:
        summary["figures"]["05_interval_coverage_vs_nominal"] = str(cov_path)

    base_path, base_records = baseline_comparison(selection, scores, out, baseline)
    summary["baseline_comparison"] = base_records
    if base_path:
        summary["figures"]["06_baseline_comparison_per_series"] = str(base_path)
        beats = [r for r in base_records if r["beats_baseline"]]
        summary["n_series_beating_baseline"] = len(beats)
        summary["n_series_compared_to_baseline"] = len(base_records)

    out.mkdir(parents=True, exist_ok=True)
    summary_path = (out / "report_summary.json").resolve()
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Saved {summary_path}")
    summary["summary_file"] = str(summary_path)
    return summary
