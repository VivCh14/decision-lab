"""Stratified sampling of series for the development loop."""

import numpy as np
import pandas as pd

STRATA_COLUMNS = ("data_quality_flag", "demand_class", "value_kind")


def select_sample(analysis: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    """Pick ``n`` series ids covering the variety found in the analysis.

    Series are grouped by ``data_quality_flag`` x ``demand_class`` x
    ``value_kind`` (columns missing from ``analysis`` are ignored) and drawn
    round-robin from the groups, so rare kinds of series are represented
    before common ones are repeated.

    Parameters
    ----------
    analysis : pd.DataFrame
        Analysis table with a ``series_id`` column.
    n : int
        Number of series wanted.
    seed : int
        Random seed.

    Returns
    -------
    list of str
        Up to ``n`` distinct series ids.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    rng = np.random.default_rng(seed)
    strata = [c for c in STRATA_COLUMNS if c in analysis.columns]
    usable = analysis[analysis["error_category"].isna()] if "error_category" in analysis.columns else analysis
    groups = [
        list(rng.permutation(group["series_id"].astype(str).to_numpy()))
        for _, group in (usable.groupby(strata, dropna=False) if strata else [(None, usable)])
    ]
    groups.sort(key=len)
    chosen: list[str] = []
    while len(chosen) < n and any(groups):
        for group in groups:
            if group and len(chosen) < n:
                chosen.append(group.pop())
    return chosen
