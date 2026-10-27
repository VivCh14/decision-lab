import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Post-process backtests into per-series, per-model metrics from the stored actuals and forecasts: WAPE, bias, MAE, RMSE, MASE, interval coverage and interval score. Writes metrics.parquet (metrics_dev.parquet with dev=true), returns per-model medians and the count of non-converged folds. sMAPE remains the selection metric; use these to explain results, especially on intermittent series.`,

  args: {
    models: tool.schema.array(tool.schema.string()).describe("Model names whose backtest files to use"),
    dev: tool.schema.boolean().optional().describe("Use the _dev backtest files"),
    work_dir: tool.schema.string().optional().describe("Directory for the output parquet files (default: current directory)"),
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
