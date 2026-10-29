"""Static checks of the agent prompts, parallel-agent configs and tools (the pack's contract)."""

import re
from pathlib import Path

import pytest
import yaml

OPENCODE = Path(__file__).resolve().parents[1] / "opencode"
AGENTS = sorted((OPENCODE / "agents").glob("*.md"))
TOOLS = sorted(p.stem for p in (OPENCODE / "tools").glob("*.ts"))


def frontmatter(path: Path) -> dict:
    return yaml.safe_load(path.read_text().split("---")[1])


def test_orchestrator_lists_parallel_agents_and_all_tools() -> None:
    tools = frontmatter(OPENCODE / "agents" / "orchestrator.md")["tools"]
    assert tools["parallel-agents"] is True
    assert tools["task"] is False
    assert set(TOOLS) <= set(tools)


def test_orchestrator_mandates_parallel_agents() -> None:
    text = (OPENCODE / "agents" / "orchestrator.md").read_text()
    assert "Use the `parallel-agents` tool to spawn `modeler`" in text
    assert "data-preparer" in text


@pytest.mark.parametrize("agent", AGENTS, ids=lambda p: p.stem)
def test_declared_skills_exist(agent: Path) -> None:
    for skill in frontmatter(agent).get("skills", []):
        assert (OPENCODE / "skills" / skill / "SKILL.md").exists(), skill


@pytest.mark.parametrize("config", sorted((OPENCODE / "parallel_agents").glob("*.yaml")), ids=lambda p: p.stem)
def test_parallel_agent_configs_name_existing_agents(config: Path) -> None:
    spec = yaml.safe_load(config.read_text())
    assert (OPENCODE / "agents" / f"{spec['name']}.md").exists()
    assert "summarizer_prompt" in spec


@pytest.mark.parametrize("agent", AGENTS, ids=lambda p: p.stem)
def test_prompts_have_no_fixed_sampler_values_or_seeds(agent: Path) -> None:
    text = agent.read_text()
    assert not re.search(r"\b(draws|tune|chains|n_folds|horizon|random_seed|seed)\s*=\s*\d+", text)


def test_data_preparer_forbids_value_transforms() -> None:
    text = (OPENCODE / "agents" / "data-preparer.md").read_text()
    assert "Any transform of the value column" in text


@pytest.mark.parametrize("tool", TOOLS)
def test_tools_return_strings(tool: str) -> None:
    text = (OPENCODE / "tools" / f"{tool}.ts").read_text()
    assert "async execute(args)" in text
    assert "ERROR (exit code" in text
    assert "return {" not in text
    assert "JSON.parse" not in text


def test_orchestrator_mandates_two_data_preparers() -> None:
    text = (OPENCODE / "agents" / "orchestrator.md").read_text()
    assert "Parallel data preparation (CRITICAL, MANDATORY)" in text
    assert '"agent": "data-preparer"' in text
    assert "never copies in `/workspace`" in text


def test_orchestrator_forbids_task_tool() -> None:
    text = (OPENCODE / "agents" / "orchestrator.md").read_text()
    assert "Never use the built-in `task` tool" in text
    assert "never write, backtest or forecast a model in your own working directory" in text


def test_orchestrator_disables_task_tool() -> None:
    text = (OPENCODE / "agents" / "orchestrator.md").read_text()
    frontmatter = text.split("---")[1]
    assert "  task: false" in frontmatter
    assert "Use the `parallel-agents` tool to spawn 2 data-preparer agents" in text
    assert "Use the `parallel-agents` tool to spawn `modeler` instances" in text


REFERENCES = sorted(
    (OPENCODE / "skills" / "timeseries-forecasting" / "references").glob("*.py")
)
SKILL = OPENCODE / "skills" / "timeseries-forecasting" / "SKILL.md"
DATATREE_GUARD = 'to_dataset() if hasattr(posterior, "to_dataset")'


def test_skill_mandates_the_datatree_guard() -> None:
    """Agent-written model code kept omitting this and failing every fold."""
    text = SKILL.read_text()
    assert "DataTree" in text
    assert DATATREE_GUARD in text


@pytest.mark.parametrize("reference", REFERENCES, ids=lambda p: p.name)
def test_sampling_references_guard_the_posterior(reference: Path) -> None:
    """Any template that stacks a posterior must convert it first.

    Deterministic templates have no posterior and are exempt.
    """
    text = reference.read_text()
    if ".stack(" not in text or "posterior" not in text:
        pytest.skip("no posterior to stack")
    assert DATATREE_GUARD in text


def test_no_data_explorer_agent() -> None:
    """data-explorer was referenced by no yaml and no prompt; it is gone."""
    assert not (OPENCODE / "agents" / "data-explorer.md").exists()
    for path in AGENTS + sorted((OPENCODE / "parallel_agents").glob("*.yaml")):
        assert "data-explorer" not in path.read_text()


@pytest.mark.parametrize("config", sorted((OPENCODE / "parallel_agents").glob("*.yaml")),
                         ids=lambda p: p.name)
def test_no_stray_slash_on_suffix_prompt(config: Path) -> None:
    """Guards against copying the mmm pack's `/subagent_suffix_prompt` typo."""
    assert "/subagent_suffix_prompt" not in config.read_text()


def test_tool_output_dir_defaults_are_named_directories() -> None:
    """Tool help must match the CLI defaults, so prompt and code cannot drift."""
    expected = {
        "run-models": "backtests/",
        "run-forecast": "forecasts/",
        "score-models": "scoring/",
        "compute-metrics": "scoring/",
        "select-models": "selection/",
    }
    for tool, directory in expected.items():
        text = (OPENCODE / "tools" / f"{tool}.ts").read_text()
        line = next(ln for ln in text.splitlines() if ln.strip().startswith("output_dir:"))
        assert f"default: {directory}" in line, f"{tool} should document {directory}"
        # work_dir may still default to the current directory; output_dir may not.
        assert "default: current directory)" not in line


ORCH = (OPENCODE / "agents" / "orchestrator.md").read_text()
MODELER = (OPENCODE / "agents" / "modeler.md").read_text()


def test_reports_use_the_mmm_naming() -> None:
    """report.md is the business report; the executive summary is a section of it."""
    assert "`report.md`" in ORCH
    assert "`technical_report.md`" in ORCH
    assert "**Executive Summary**" in ORCH or "## Executive Summary" in ORCH
    assert "executive_summary.md" in ORCH, "the prompt must say not to write it"
    assert "Do **not** write `executive_summary.md`" in ORCH


def test_prompts_treat_error_rows_as_failure() -> None:
    """run-models records failures per row; exit code 0 is not success."""
    for text in (ORCH, MODELER):
        assert "n_error_rows" in text
    assert "THE EXIT CODE IS NOT THE VERDICT" in MODELER
    assert "must be 0" in ORCH


def test_modeler_mandates_the_datatree_guard() -> None:
    assert DATATREE_GUARD in MODELER
    assert "DataTree" in MODELER


def test_prompts_require_absolute_paths_with_the_reason() -> None:
    """A bare filename is ambiguous because instance dirs are seeded copies."""
    for text in (MODELER, (OPENCODE / "agents" / "data-preparer.md").read_text()):
        assert "ABSOLUTE" in text or "absolute" in text
    assert "byte-identical" in MODELER


def test_orchestrator_requires_complete_series_ids_per_plan() -> None:
    """Ellipsis-terminated target lists are how plan coverage goes unchecked."""
    assert "n_target_series" in ORCH
    assert "series_ids" in ORCH
    assert "overlap" in ORCH.lower()


def test_orchestrator_requires_structurally_diverse_plans() -> None:
    assert "structurally diverse" in ORCH
    assert "different family from the skill's model menu" in ORCH
    assert "parameter tweak" in ORCH


def test_orchestrator_forbids_cross_scope_aggregate_comparison() -> None:
    """Comparing medians over different series sets is invalid, caveat or not."""
    assert "different series sets" in ORCH
    assert "per-series comparison exists, it governs" in ORCH


def test_orchestrator_caps_modeling_rounds() -> None:
    assert "MAXIMUM 3 MODELING ROUNDS TOTAL" in ORCH
    assert "Never rerun an unchanged plan" in ORCH


def test_orchestrator_has_seven_stop_conditions() -> None:
    section = ORCH.split("### When to Stop Without Recommendations")[1].split("###")[0]
    numbered = re.findall(r"^\d+\. \*\*", section, flags=re.MULTILINE)
    assert len(numbered) == 7, f"expected 7 stop conditions, found {len(numbered)}"


def test_orchestrator_has_failure_report_template() -> None:
    for heading in ("Not Validatable", "What We CAN Say", "What We CANNOT Say",
                    "Data Needed to Resolve"):
        assert heading in ORCH


def test_orchestrator_has_conflict_checks_and_decision_logic() -> None:
    assert "CONFLICTING" in ORCH
    assert "FOR each segment:" in ORCH
    for check in ("Check 1", "Check 2", "Check 3"):
        assert check in ORCH
    assert "adding caveats does not fix a conflict" in ORCH.lower()


def test_prompts_reuse_library_convergence_limits() -> None:
    """The prompts must quote the library's limits, not invent their own."""
    from tsf_lib.backtest import DIVERGENCE_PCT_MAX, ESS_MIN, NONCONVERGED_SHARE_MAX, RHAT_MAX

    for text in (ORCH, MODELER):
        assert str(RHAT_MAX) in text
        assert str(int(ESS_MIN)) in text
        assert str(NONCONVERGED_SHARE_MAX) in text
        assert f"{DIVERGENCE_PCT_MAX:g}%" in text


def test_orchestrator_delegates_figures_to_the_tool() -> None:
    """Hand-written plotting is what produced the run-011 layout defects."""
    assert "build-report-figures" in ORCH
    assert "DO NOT WRITE PLOTTING CODE" in ORCH
    assert "savefig" not in ORCH


def test_orchestrator_keeps_smape_as_the_selection_metric() -> None:
    """WAPE and accuracy are for reporting and must not override selection."""
    assert "sMAPE is the selection metric" in ORCH
    assert "unbounded below" in ORCH


@pytest.mark.parametrize("path", AGENTS, ids=lambda p: p.name)
def test_prompts_are_not_truncated(path: Path) -> None:
    """Guards against the prompts being silently cut back toward their old size."""
    assert len(path.read_text().splitlines()) >= 200


PREPARER = (OPENCODE / "agents" / "data-preparer.md").read_text()

# Tools the modeler prompt tells the agent to call. parallel-agents copies a tool
# into an instance ONLY when the frontmatter marks it true (copyAllowedTools in
# dlab/js/parallel-agents.ts), so an instruction without a declaration is a silent
# no-op: run 012 had four modelers with an empty .opencode/tools/ directory.
MODELER_REQUIRED_TOOLS = (
    "estimate-run-time", "run-models", "run-forecast",
    "score-models", "compute-metrics", "fetch-table",
)


@pytest.mark.parametrize("tool", MODELER_REQUIRED_TOOLS)
def test_modeler_declares_every_tool_it_is_told_to_use(tool: str) -> None:
    tools = frontmatter(OPENCODE / "agents" / "modeler.md")["tools"]
    assert tools.get(tool) is True, (
        f"modeler.md instructs the agent to use `{tool}` but does not declare it true "
        "in frontmatter, so parallel-agents will not copy it into the instance"
    )
    assert tool in MODELER, f"`{tool}` is declared but never mentioned in the prompt"


def test_subagent_tool_declarations_exist_as_tools() -> None:
    """A declared tool must correspond to a real .ts file, or nothing is copied."""
    for agent in ("modeler.md", "data-preparer.md"):
        tools = frontmatter(OPENCODE / "agents" / agent)["tools"]
        custom = [t for t, v in tools.items()
                  if v is True and t not in ("read", "edit", "bash", "parallel-agents")]
        for tool in custom:
            assert tool in TOOLS, f"{agent} declares unknown tool `{tool}`"


def test_orchestrator_only_tools_stay_off_subagents() -> None:
    """Orchestrator-only tools must not be copied into modeler instances."""
    tools = frontmatter(OPENCODE / "agents" / "modeler.md")["tools"]
    for tool in ("build-report-figures", "select-models", "analyze-series",
                 "summarize-analysis", "sample-series", "inspect-data"):
        assert tools.get(tool) is not True, f"modeler should not receive `{tool}`"


def test_modeler_forbids_hand_rolled_runners() -> None:
    """instance-4 wrote a driver.py against tsf_lib.runner and became unscoreable."""
    assert "DO NOT WRITE A SCRIPT" in MODELER
    assert "tsf_lib.runner" in MODELER
    assert "python -m tsf_lib.cli" in MODELER


def test_modeler_prohibits_output_dir_twice() -> None:
    """Stated once in a numbered list was not enough to stop an `out/` directory."""
    assert MODELER.count("output_dir") >= 2
    assert "NEVER set `output_dir`" in MODELER


def test_modeler_writes_summary_early() -> None:
    """A summary written only at the end is lost to a single API failure."""
    assert "WRITE THE SUMMARY EARLY" in MODELER
    assert "skeleton `summary.md`" in MODELER
    assert "failure_class" in MODELER, "a failure-only schema must be given"


def test_modeler_caps_dev_iterations() -> None:
    assert "at most 3 dev runs" in MODELER
    assert "never a sampling-budget problem" in MODELER
    assert "Re-time after any structural rewrite" in MODELER


def test_modeler_timeout_is_bounded() -> None:
    """One stalled instance blocks the orchestrator for the whole timeout."""
    spec = yaml.safe_load((OPENCODE / "parallel_agents" / "modeler.yaml").read_text())
    assert spec["timeout_minutes"] <= 30


def test_orchestrator_accounts_for_missing_summaries() -> None:
    """Step 6 must not assume every modeler returned."""
    assert "account for every plan" in ORCH
    assert "Salvage it" in ORCH
    assert "Never report a segment as covered when its plan returned nothing" in ORCH


def test_orchestrator_uses_per_series_min_train_size() -> None:
    """A self-chosen floor produced a wrong `unvalidated` list in run 012."""
    assert "recommended.min_train_size" in ORCH
    assert "provisional" in ORCH
    assert "reconcile your list against what the modelers actually achieved" in ORCH


def test_data_preparer_declares_its_inspection_tools() -> None:
    tools = frontmatter(OPENCODE / "agents" / "data-preparer.md")["tools"]
    for tool in ("inspect-data", "fetch-table"):
        assert tools.get(tool) is True
        assert tool in PREPARER
