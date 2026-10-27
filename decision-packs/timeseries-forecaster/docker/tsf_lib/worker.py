"""Model loading and analysis parsing shared by the local and Modal executors."""

import json
import types
from typing import Any, Callable

MODEL_ENTRYPOINT = "fit_forecast"


def load_model_fn(source: str, model_name: str) -> Callable:
    """Execute model source code and return its ``fit_forecast`` function.

    The source travels with each batch, so remote workers need no model files.

    Parameters
    ----------
    source : str
        Python source of a model module defining ``fit_forecast``.
    model_name : str
        Used only to name the in-memory module.

    Returns
    -------
    Callable
        The ``fit_forecast`` function.
    """
    module = types.ModuleType(f"tsf_model_{model_name}")
    exec(compile(source, f"<model:{model_name}>", "exec"), module.__dict__)
    if not hasattr(module, MODEL_ENTRYPOINT):
        raise ValueError(f"Model '{model_name}' does not define {MODEL_ENTRYPOINT}().")
    return getattr(module, MODEL_ENTRYPOINT)


def parse_analysis(analysis_json: str) -> dict[str, Any]:
    """Parse the per-series analysis JSON handed to models.

    ``regressor_lags`` becomes ``{regressor_name: [lags]}`` and
    ``recommended_params`` is exposed as a dict under ``recommended``.

    Parameters
    ----------
    analysis_json : str
        JSON of one analysis row, optionally with a ``regressor_lags`` list of
        ``{regressor_name, recommended_lags}`` entries.

    Returns
    -------
    dict
        The analysis row with parsed nested fields.
    """
    analysis = json.loads(analysis_json)
    analysis["recommended"] = json.loads(analysis.get("recommended_params") or "{}")
    analysis["regressor_lags"] = {
        entry["regressor_name"]: json.loads(entry["recommended_lags"] or "[]")
        for entry in (analysis.get("regressor_lags") or [])
    }
    return analysis
