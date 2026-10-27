import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Estimate wall-clock time of a backtest or forecast run by timing a small probe (2 series) and extrapolating. Use BEFORE run-models on the full set of series. Same arguments as run-models.`,

  args: {
    model_file: tool.schema.string().describe("Path to the model .py file defining fit_forecast"),
    model_name: tool.schema.string().describe("Short model name used in output file names"),
    horizon: tool.schema.number().describe("Forecast horizon in periods"),
    n_folds: tool.schema.number().optional().describe("Rolling-origin folds for backtests (default 1)"),
    step: tool.schema.number().optional().describe("Periods between fold origins (default 1)"),
    params: tool.schema.record(tool.schema.string(), tool.schema.any()).optional().describe("Extra parameters passed to fit_forecast"),
    series_ids: tool.schema.array(tool.schema.string()).optional().describe("Restrict to these series (default: all)"),
    batch_size: tool.schema.number().optional().describe("Series per batch (default: chosen automatically)"),
    forecast: tool.schema.boolean().optional().describe("Estimate a forecast run instead of a backtest"),
    work_dir: tool.schema.string().optional().describe("Directory for the output parquet files (default: current directory)"),
  },

  async execute(args) {
    const cmd = `python -m tsf_lib.cli estimate --args '${JSON.stringify(args).replace(/'/g, "'\\''")}'`
    const result = await Bun.$`sh -c ${cmd}`.nothrow()
    const stdout = result.stdout.toString().trim()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr}\n\nStdout:\n${stdout}`
    }

    return stdout
  },
})
