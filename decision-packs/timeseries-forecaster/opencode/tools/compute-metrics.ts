import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Post-process backtests into per-series, per-model metrics from the stored actuals and forecasts: WAPE, bias, MAE, RMSE, MASE, interval coverage and interval score. Writes metrics.parquet (metrics_dev.parquet with dev=true), returns per-model medians and the count of non-converged folds. sMAPE remains the selection metric; use these to explain results, especially on intermittent series.`,

  args: {
    backtest_files: tool.schema.array(tool.schema.string()).describe("Paths of the backtest parquet files to use (one per model, e.g. from run-models)"),
    dev: tool.schema.boolean().optional().describe("Mark as development metrics (metrics_dev.parquet)"),
    output_dir: tool.schema.string().optional().describe("Directory for output files (default: scoring/ under the current directory). Parallel instances should pass their own instance directory."),
    work_dir: tool.schema.string().optional().describe("Directory holding analysis.parquet and inputs.json (default: current directory)"),
  },

  async execute(args) {
    const cmd = `python -m tsf_lib.cli metrics --args '${JSON.stringify(args).replace(/'/g, "'\\''")}'`
    const result = await Bun.$`sh -c ${cmd}`.nothrow()
    const stdout = result.stdout.toString().trim()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr}\n\nStdout:\n${stdout}`
    }

    return stdout
  },
})
