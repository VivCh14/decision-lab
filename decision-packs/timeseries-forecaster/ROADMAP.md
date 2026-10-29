# timeseries-forecaster roadmap and status

Last updated: 2026-10-29 (after run 012 triage). Branch `feat/tsf-modal` (last commit 744b8b0, unpushed; everything below is uncommitted).
Rule: follow the MMM pack's patterns exactly; no new patterns without agreement. No commit/push without the user's go-ahead (run the redaction scan first).

## Status

Run 011 (2026-10-29, `github-copilot-enterprise/claude-sonnet-5`, `DLAB_FIT_MODEL_LOCALLY=1`) was the pack's **first complete end-to-end session**, through Step 9: `evaluation.md`, four forecasts, `final_forecast.parquet`, both reports and both figures. Archived at `~/dlab-run-archive/tsf-run-011-reference` as the pre-change reference.

An earlier run 011 (2026-10-27) was deleted when the directory name was reused, taking the evidence for the old Issues 6 and 7 with it. **Never reuse a work-dir name.**

## Done

Phases 1-5 (alignment with the MMM pack): per-instance outputs; `data-preparer` agent and `parallel_agents/data-preparer.yaml`; skills (`data`, `distributions`, `informative-priors`, `samplers`) and the `intermittent_hurdle.py` reference; all custom tools return strings (opencode 1.2.10 throws `text4.split is not a function` on objects); MMM-style pytest suite.

MMM parity sweep (2026-10-29), all uncommitted:

- **Output structure** — every tool now defaults `output_dir` to a purpose-named subdirectory (`backtests/`, `forecasts/`, `scoring/`, `selection/`) instead of `.`. This was the whole reason run 011's root was a flat heap of 20 files; MMM's layout comes from tool defaults, not prompt discipline.
- **`tsf_lib/report.py` + `build-report-figures`** — the pack had *no* figure-producing code at all, so the orchestrator hand-wrote matplotlib. Now six numbered figures plus `report_summary.json`, built on `mmm_lib/analyze_model.py`'s pattern (shared `savefig`, tool-owned `report_figures/`). The best/worst series grids follow the user's reference design: rows are segments, columns are rank 1-3, each panel carrying history, a shaded backtest span labelled with its length, in-backtest actuals, one dashed trace per fold, and the forecast with an interval band over the out-of-sample region only.
- **Failures are no longer silent** — `run-models`/`run-forecast` exit non-zero when every row errored (`cli.py` returns a `failed` verdict; `main()` exits 1). Previously `n_error_rows == n_rows` still exited 0.
- **`tsf_lib/__init__.py`** — stale Spark/Databricks docstring replaced; flat re-export façade with `__all__`; `find_spec`-guarded `dlab_plotstyle` import at the library boundary, matching `mmm_lib`.
- **`DataTree` guard promoted from example to rule** — stated in `SKILL.md` and `modeler.md`, with a test asserting every sampling reference carries it.
- **Prompts rebuilt to MMM depth** — orchestrator 162 → 476 lines, modeler 51 → 261, data-preparer 69 → 271. Added: conflict-detection checks with decision-logic pseudocode, a 3-round retry protocol with the epistemic-cost paragraph, 7 stop conditions, a "Not Validatable - Baseline Retained" failure report template, WRONG/CORRECT pairs, subagent personalities and full `summary.md` schemas, and an Expected Outputs directory tree.
- **Reports renamed MMM-style** — `report.md` (business, Executive Summary inside) + `technical_report.md`; `executive_summary.md` is no longer written.
- **`data-explorer.md` deleted** (referenced by nothing), stray 858 MB in-pack `venv/` removed, `environment.yaml` added.
- Tests: 128 passed, 2 skipped. New `tests/test_report_figures.py` (11 tests) and 19 new assertions in `tests/test_agent_prompts.py`.

## Run 012 (2026-10-29, horizon 6) — stopped early, root cause found

Killed after ~45 min with the orchestrator blocked behind its modeler `parallel-agents` barrier.

**Root cause: modeler instances had no custom tools.** `copyAllowedTools` (`dlab/js/parallel-agents.ts:189-198`) copies a tool into a subagent only when that tool is `true` in the subagent's frontmatter — the MMM modeler declares its three tools for exactly this reason. tsf's `modeler.md` declared only `read`/`edit`/`bash`, so every instance's `.opencode/tools/` was empty and **no modeler ever called `run-models`, `score-models`, `run-forecast`, `compute-metrics` or `estimate-run-time`**. Each burned 5-11 turns reverse-engineering `/opt/tsf_lib/cli.py`; one wrote a `driver.py` against `tsf_lib.runner` whose outputs `score-models` cannot find; another hand-passed `output_dir` and created `out/`. Pre-existing bug — the old prompt's comment "no custom tools listed — they come from the global tool dir" was simply false. **Fixed**, with a test that fails if any instructed tool is left undeclared.

Instance outcomes: instance-1 complete (31 series, 558 rows, 0 errors, schema-compliant summary); instance-2 and instance-4 killed by **non-retryable HTTP 400s from the GHE Copilot proxy** (`isRetryable: false`, plain-text `Bad Request` from `server: github.com`, 61 s apart) — instance-4 losing an already-complete clean backtest because `summary.md` is written last; instance-3 stalled 21 min after six dev rounds chasing `implausible_mean_rows` by escalating sampler settings, then a rewrite that made a 37 s job take 14 m 12 s.

Also found: the `unvalidated` set was **wrong**. Step 2.5 applied a self-chosen `min_train_size=24` while the harness enforces the per-series `recommended.min_train_size` (10 for the short series); all four "infeasible" series produced full folds with real scores, and a modeler had to catch it.

Confirmed working: coverage reconciliation (complete `series_ids`, union 40, Plan 1/4 overlap declared as "a comparison, not a partition"), plan diversity (four distinct families incl. a non-MCMC ETS), fold-feasibility reporting, output layout and promotion, the `summary.md` schema, 0 `task` calls, and the `DataTree` guard (present in all logs only as the skill's warning — no instance hit the error).

Fixes applied, all prompt/yaml only (no rebuild needed): modeler and data-preparer tool declarations; "USE THEM, DO NOT WRITE A SCRIPT" with a FORBIDDEN driver example; summary-written-early plus a failure-only schema; `output_dir` prohibition restated; dev-iteration cap with "implausible means is a parameterisation problem, not a sampling-budget problem"; re-time after a rewrite; `timeout_minutes` 120 → 30; orchestrator per-plan accounting (returned / salvaged / failed) and per-series `min_train_size`. Tests: **144 passed, 2 skipped**.

## Closed issues

| # | Was | Resolution |
|---|---|---|
| 1 | Orchestrator bypassed `parallel-agents` | `task: false` holds; run 011 logged 0 `task` calls |
| 2 | Run never reached Step 9 | Run 011 completed, figures produced |
| 3 | Consolidator claimed but never runs at 2 instances | Orchestrator now reads both preparer summaries itself; yaml notes the `>= 3` threshold |
| 6 | Model artifacts in the orchestrator root | Does not reproduce — all four instances kept their own outputs |

## Open issues

1. **Coverage is unaccounted for (was Issue 5, sharpened).** Run 011's four plan scopes summed to 50 slots across 40 series: Plan A's 24 and Plan D's 10 overlapped, and the consolidator said it could not reconcile them. Root cause: target sets written as ellipsis-terminated examples. Fixed in the prompt (complete `series_ids`, `n_target_series`, declared overlaps) — **needs confirming in run 012.**
2. **Invalid cross-scope comparison (was Issue 7).** `evaluation.md` compared Plan A's median over 24 series against Plan D's over a different 12, in a sentence that contradicted itself mid-clause, while a correct per-series table sat unused directly above. Fixed in the prompt — **needs confirming in run 012.**
3. **Plan diversity** was fine in run 011 (harmonic regression / count GLM / ZINB hurdle / robust Student-t). Weaker than previously recorded; the residual risk is coverage, not diversity.
4. **Backtest limits.** Series with 20-30 observations get no fold at long horizons; gap series get one fold after the post-gap trim. The feasibility inequality is now stated in Step 2.5 and in the data-preparer, but a shorter horizon for short series is still not implemented.
5. **Intermittent accuracy poor** — run 011's ZINB scored a negative pooled accuracy on the intermittent segment, worse than the baseline on the mean basis. Croston/TSB still deferred.
6. **uid 100999 ownership.** Container-created run dirs are owned by uid 100999, so `--continue-dir` fails and cleanup needs sudo. Decision: leave it, use fresh run dirs only.
7. **Stale untracked run artifacts at the repo root** (`tsf-run-001..004`, `010`, `011`, logs, `mmm-claude-sonnet-5-run/`, `pyproject.toml.bak`). Not for commit.
8. **Docs not yet updated** — `README.md` (36 lines vs MMM's 247) and `tests/README.md` (25 vs 232) still need expanding; `docs/decision-packs.md` should document the consolidator's `>= 3` threshold and the run-directory layout convention.

## Next steps

1. Launch run 013 (command below), detached, into a fresh directory name. No `--rebuild` needed — the run-012 fixes are prompt/yaml only.
2. Review against the acceptance checks: complete per-plan `series_ids` with reconciled counts; no cross-scope aggregate comparison; every `evaluation.md` number traced to a file; `n_error_rows` treated as failure; no `DataTree` error; the documented directory tree; figures from `build-report-figures` only; `report.md` + `technical_report.md` and no `executive_summary.md`.
3. Finish issue 8 (docs).
4. Redaction scan, then commit on the user's go-ahead.
5. Update the memory file `project_tsf_modal_status.md`.

Launch (host has no pandas; `dlab` is not on PATH; `--rebuild` is required because `tsf_lib` changed):
```
P=decision-packs/timeseries-forecaster/example-data
DLAB_FIT_MODEL_LOCALLY=1 nohup venv/bin/dlab run --dpack decision-packs/timeseries-forecaster \
  --data $P/series_hard.csv --data $P/regressors_hard.csv --work-dir tsf-run-013 --yes \
  --model github-copilot-enterprise/claude-sonnet-5 \
  --prompt "Forecast the next 6 months for all series. Use the regressors where they help. Segment model selection by demand class." \
  > tsf-run-013.log 2>&1 &
```
Tests (in docker, host lacks pandas):
```
docker run --rm -v "$PWD":/pack -w /pack -e PYTHONPATH=/pack/docker:/pack/opencode/skills/timeseries-forecasting/references dlab-timeseries-forecaster:latest sh -c "pip install -q pytest 2>/dev/null; python -m pytest tests -q -W ignore"
```
(run from the pack directory; `tsf_lib` changes need `dlab --rebuild`).

## Deferred (Tier 2/3)

Partial pooling, ensemble, Croston/TSB, scenario tool and quantile output, run manifest, per-batch retry/resume, uv instead of conda, verify the Modal path once an account exists.

## Reminders

Rotate the exposed NVIDIA/OpenAI/Databricks tokens and revoke the rejected `GH_TOKEN`.
