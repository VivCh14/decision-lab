"""Modal app running batches of series (analysis, backtest, forecast) in parallel.

Each call receives a pickled batch (pandas frames plus model source) and
returns pickled result frames. The ``tsf_lib`` package is copied into the
image, so containers run exactly the code of the local fallback.

Package versions are pinned to match ``docker/Dockerfile``; keep them in sync.

Setup:
    1. Create a Modal account at https://modal.com
    2. Run ``modal setup`` (or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)
    3. ``modal deploy tsf_runner.py`` (dlab's pre-run hook does this for you)
"""

import hashlib
import pickle
import time
from pathlib import Path

import modal

_modal_app_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

# Inside the pack container the library lives in /opt/tsf_lib; from a checkout it is a sibling directory.
_LIB_CANDIDATES = [Path("/opt/tsf_lib"), Path(__file__).resolve().parents[1] / "tsf_lib"]
LIB_DIR = next(p for p in _LIB_CANDIDATES if p.exists())
REMOTE_LIB_DIR = "/root/tsf_lib"

CONDA_PACKAGES = [
    "python=3.12",
    "numpy=2.1.3",
    "pandas=2.2.3",
]

# Packages installed with pip (versions must match docker/Dockerfile)
PIP_PACKAGES = [
    "scipy==1.15.3",
    "statsmodels==0.15.0",
    "pymc==6.3.2",
    "pytensor==3.3.4",
    "arviz==1.3.0",
    "nutpie==0.16.11",
    "pyarrow==21.0.0",
]

runner_image = (
    modal.Image.micromamba(python_version="3.12")
    .run_commands(f"echo 'modal_app hash: {_modal_app_hash}'")  # Cache buster
    .micromamba_install(*CONDA_PACKAGES, channels=["conda-forge"])
    .pip_install(*PIP_PACKAGES)
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(str(LIB_DIR), REMOTE_LIB_DIR, copy=True, ignore=["__pycache__", "*.pyc"])
)

app = modal.App("tsf-runner")


@app.function(image=runner_image, cpu=4, memory=8192, timeout=7200)
def run_series_batch(batch_pickle: bytes) -> bytes:
    """Run one batch and return its pickled result frames.

    Parameters
    ----------
    batch_pickle : bytes
        Pickled batch dict (``items`` plus ``spec``) as built by
        ``tsf_lib.runner.make_batches``.

    Returns
    -------
    bytes
        Pickled ``{name: DataFrame}`` result of ``tsf_lib.runner.run_batch``.
    """
    from tsf_lib.runner import run_batch

    return pickle.dumps(run_batch(pickle.loads(batch_pickle)))


@app.function(image=runner_image, cpu=4, memory=8192, timeout=1800)
def estimate_run_time(batch_pickle: bytes) -> bytes:
    """Time a small probe batch and return ``{"elapsed_seconds", "n_series"}`` pickled."""
    from tsf_lib.runner import run_batch

    batch = pickle.loads(batch_pickle)
    start = time.time()
    run_batch(batch)
    return pickle.dumps({"elapsed_seconds": time.time() - start, "n_series": len(batch["items"])})
