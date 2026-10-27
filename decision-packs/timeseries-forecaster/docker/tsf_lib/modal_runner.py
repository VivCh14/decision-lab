"""Batch executor: Modal cloud by default, local process pool as fallback.

Environment variables:
    DLAB_FIT_MODEL_LOCALLY: "1" runs every batch in a local process pool.
    MODAL_TOKEN_ID / MODAL_TOKEN_SECRET: required for Modal; when missing the
        local pool is used instead.
    DLAB_TSF_WORKERS: worker processes for local runs (default: CPU count).
"""

import multiprocessing
import os
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .runner import run_batch

MODAL_APP_PATH = "/opt/modal_app/tsf_runner.py"
MODAL_APP_NAME = "tsf-runner"
MODAL_FUNCTION = "run_series_batch"
MODAL_ESTIMATE_FUNCTION = "estimate_run_time"
MODAL_PARALLELISM = 50


def should_run_locally() -> bool:
    """True when ``DLAB_FIT_MODEL_LOCALLY=1`` or Modal credentials are missing."""
    if os.environ.get("DLAB_FIT_MODEL_LOCALLY", "0") == "1":
        return True
    return not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))


def local_workers() -> int:
    """Worker process count for local runs."""
    return max(1, int(os.environ.get("DLAB_TSF_WORKERS", os.cpu_count() or 1)))


def ensure_modal_deployed() -> None:
    """Deploy the Modal app when the ``tsf-runner`` functions are not found."""
    import modal

    try:
        modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION)
        return
    except modal.exception.NotFoundError:
        print("Modal app not deployed, deploying now...")
    if not Path(MODAL_APP_PATH).exists():
        raise FileNotFoundError(f"Modal app not found at {MODAL_APP_PATH}; run inside the decision-pack container.")
    result = subprocess.run(["modal", "deploy", MODAL_APP_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to deploy Modal app:\n{result.stderr}\n{result.stdout}")


def _run_local(batches: list[dict[str, Any]], workers: int) -> list[dict[str, pd.DataFrame]]:
    if workers == 1 or len(batches) == 1:
        return [run_batch(b) for b in batches]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(batches)), mp_context=context) as pool:
        return list(pool.map(run_batch, batches))


def _run_modal(batches: list[dict[str, Any]]) -> list[dict[str, pd.DataFrame]]:
    import modal

    ensure_modal_deployed()
    fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION)
    payloads = [pickle.dumps(b) for b in batches]
    return [pickle.loads(out) for out in fn.map(payloads, order_outputs=True)]


def execute_batches(batches: list[dict[str, Any]], workers: int | None = None) -> tuple[list[dict[str, pd.DataFrame]], str]:
    """Run all batches in one call and return ``(results, backend)``.

    Results keep the batch order. ``backend`` is ``"modal"`` or ``"local"``.
    """
    if should_run_locally():
        return _run_local(batches, workers or local_workers()), "local"
    return _run_modal(batches), "modal"


def estimate_seconds(probe: dict[str, Any], n_series: int, n_batches: int) -> dict[str, Any]:
    """Time a small probe batch and extrapolate to the full run.

    Parameters
    ----------
    probe : dict
        A batch holding a few series, built with the same spec as the full run.
    n_series : int
        Series in the full run.
    n_batches : int
        Batches the full run will use.

    Returns
    -------
    dict
        ``seconds_per_series``, ``estimated_total_minutes`` and ``backend``.
    """
    n_probe = len(probe["items"])
    if should_run_locally():
        start = time.time()
        run_batch(probe)
        elapsed, backend, parallel = time.time() - start, "local", local_workers()
    else:
        import modal

        ensure_modal_deployed()
        fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_ESTIMATE_FUNCTION)
        elapsed = pickle.loads(fn.remote(pickle.dumps(probe)))["elapsed_seconds"]
        backend, parallel = "modal", MODAL_PARALLELISM
    per_series = elapsed / n_probe
    waves = max(1, -(-n_batches // parallel))
    batch_seconds = per_series * (n_series / max(1, n_batches))
    return {
        "backend": backend,
        "seconds_per_series": round(per_series, 3),
        "estimated_total_minutes": round(waves * batch_seconds / 60, 2),
        "n_series": n_series,
        "n_batches": n_batches,
    }
