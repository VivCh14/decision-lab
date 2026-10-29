"""Time series analysis, backtesting and forecasting for the timeseries-forecaster pack.

Model files written by the ``modeler`` agent import from this package. The
helpers most of them need are re-exported here, so a model file can do::

    from tsf_lib import bound_draws, sampler_diagnostics

rather than reaching into submodules. Batches are executed either in a local
process pool or on Modal; see :mod:`tsf_lib.modal_runner`.
"""

# In dlab sessions the decision-lab figure style module is on PYTHONPATH; it
# enforces the parts of the house style an rc file cannot (band edges, legend
# frames, colormaps). Standalone use of tsf_lib works without it.
import importlib.util

if importlib.util.find_spec("dlab_plotstyle") is not None:
    import dlab_plotstyle  # noqa: F401

from .analysis import (
    ANALYSIS_COLUMNS,
    AnalysisError,
    ConstantSeriesError,
    DuplicateDatesError,
    InsufficientSampleError,
    NonUniformFrequencyError,
    analyze_autocorrelation,
    analyze_series,
    analyze_trend,
    classify_data_quality,
    describe_values,
    detect_seasonality_fft,
    find_best_seasonal_period,
    prepare_series,
    stl_seasonal_strength,
    test_stationarity,
)
from .backtest import (
    BACKTEST_COLUMNS,
    DIVERGENCE_PCT_MAX,
    ESS_MIN,
    FORECAST_COLUMNS,
    NONCONVERGED_SHARE_MAX,
    RHAT_MAX,
    backtest_series,
    forecast_series,
    implausible_rows,
    rolling_origins,
    score_backtest,
    smape,
)
from .diagnostics import sampler_diagnostics
from .metrics import compute_metrics, nonconverged_folds, series_metrics
from .quality import bound_draws, detect_jumps, detect_quality_issues, trim_to_level_shift
from .recommend import recommend_params
from .regressors import REGRESSOR_COLUMNS, analyze_regressor_lags, lagged_regressor_matrix
from .sampling import select_sample
from .schema import typed_frame
from .selection import METRICS, assemble_forecast, select_models
from .timegrid import (
    FREQUENCIES,
    FrequencyError,
    classify_frequency,
    contiguous_tail,
    date_to_step,
    regularize,
    step_to_date,
)

__version__ = "0.1.0"

__all__ = [
    # Series analysis
    "ANALYSIS_COLUMNS", "analyze_series", "prepare_series", "describe_values",
    "classify_data_quality", "analyze_autocorrelation", "analyze_trend",
    "stl_seasonal_strength", "find_best_seasonal_period", "detect_seasonality_fft",
    "test_stationarity",
    "AnalysisError", "ConstantSeriesError", "DuplicateDatesError",
    "InsufficientSampleError", "NonUniformFrequencyError",
    # Backtesting and scoring
    "BACKTEST_COLUMNS", "FORECAST_COLUMNS", "backtest_series", "forecast_series",
    "rolling_origins", "score_backtest", "smape", "implausible_rows",
    # Convergence limits (prompts and reports quote these; do not duplicate the values)
    "RHAT_MAX", "ESS_MIN", "DIVERGENCE_PCT_MAX", "NONCONVERGED_SHARE_MAX",
    # Diagnostics and metrics
    "sampler_diagnostics", "compute_metrics", "series_metrics", "nonconverged_folds",
    # Data quality (model files use bound_draws and trim_to_level_shift)
    "bound_draws", "trim_to_level_shift", "detect_jumps", "detect_quality_issues",
    # Recommended parameters per series
    "recommend_params",
    # Regressors
    "REGRESSOR_COLUMNS", "analyze_regressor_lags", "lagged_regressor_matrix",
    # Selection
    "METRICS", "select_models", "assemble_forecast",
    # Calendars
    "FREQUENCIES", "FrequencyError", "classify_frequency", "regularize",
    "date_to_step", "step_to_date", "contiguous_tail",
    # Sampling and schema
    "select_sample", "typed_frame",
]
