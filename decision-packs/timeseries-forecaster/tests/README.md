# Timeseries-forecaster tests

Real implementations only (no mocks). `tsf_lib` and the Bayesian templates need the pack image, so run the suite in it from the pack directory:

```bash
docker run --rm -v "$PWD":/pack -w /pack \
  -e PYTHONPATH=/pack/docker:/pack/opencode/skills/timeseries-forecasting/references \
  dlab-timeseries-forecaster:latest sh -c "pip install -q pytest; python -m pytest tests -q -W ignore"
```

Skip the slow MCMC tests with `-m "not slow"`.

| File | Covers |
|---|---|
| `test_analysis.py`, `test_timegrid.py`, `test_schema_sampling.py`, `test_regressors.py` | analysis, calendar, sampling, lag detection |
| `test_backtest.py`, `test_metrics.py`, `test_selection.py`, `test_quality.py` | scoring, metrics, model selection, quality gates |
| `test_bayes_model.py` | the reference model templates (slow, MCMC) |
| `test_runner_local.py` | file-based pipeline with the local process pool |
| `test_agent_prompts.py` | static contract of agent prompts, parallel-agent configs and tools |
| `test_figure_style_integration.py` | `tsf_lib` does not restyle matplotlib on import |
| `test_modal_integration.py` | Modal vs local batch results; skipped without `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` |

## Modal

Set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, deploy with `modal deploy docker/modal_app/tsf_runner.py`, then run `test_modal_integration.py`.
