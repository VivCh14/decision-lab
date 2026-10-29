import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Score backtests by sMAPE per series and model. Writes scores.parquet (scores_dev.parquet with dev=true) to output_dir and returns per-model count/mean/median, fold and convergence counts, a same-series comparison against the baseline model (vs_baseline_same_series, on both mean- and median-forecast sMAPE) and the worst series.`,

  args: {
    backtest_files: tool.schema.array(tool.schema.string()).describe("Paths of the backtest parquet files to use (one per model, e.g. from run-models)"),
    baseline: tool.schema.string().optional().describe("Baseline model name for the same-series comparison (default seasonal_naive)"),
    dev: tool.schema.boolean().optional().describe("Mark as development scoring (scores_dev.parquet)"),
    output_dir: tool.schema.string().optional().describe("Directory for output files (default: scoring/ under the current directory). Parallel instances should pass their own instance directory."),
  },

  async execute(args) {
    const cmd = `python -m tsf_lib.cli score --args '${JSON.stringify(args).replace(/'/g, "'\\''")}'`
    const result = await Bun.$`sh -c ${cmd}`.nothrow()
    const stdout = result.stdout.toString().trim()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr}\n\nStdout:\n${stdout}`
    }

    return stdout
  },
})
