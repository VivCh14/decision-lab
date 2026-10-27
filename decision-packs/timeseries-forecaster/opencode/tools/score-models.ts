import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Score backtests by sMAPE per series and model. Writes scores.parquet (scores_dev.parquet with dev=true) and returns per-model count/mean/median, fold and convergence counts, a same-series comparison against the baseline model (vs_baseline_same_series, on both mean- and median-forecast sMAPE) and the worst series.`,

  args: {
    models: tool.schema.array(tool.schema.string()).describe("Model names whose backtest files to score"),
    baseline: tool.schema.string().optional().describe("Baseline model name for the same-series comparison (default seasonal_naive)"),
    dev: tool.schema.boolean().optional().describe("Score the _dev backtest files"),
    work_dir: tool.schema.string().optional().describe("Directory for the output parquet files (default: current directory)"),
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
