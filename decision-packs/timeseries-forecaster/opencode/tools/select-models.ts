import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Pick, per segment of an analysis column, the model with the lowest median sMAPE (selection.parquet); with forecast files present for the listed models, also writes final_forecast.parquet.`,

  args: {
    models: tool.schema.array(tool.schema.string()).optional().describe("Model names with forecast_<model>.parquet files to assemble"),
    by: tool.schema.string().optional().describe("Analysis column defining segments (default demand_class)"),
    metric: tool.schema.union([tool.schema.string(), tool.schema.record(tool.schema.string(), tool.schema.string())]).optional().describe("smape (mean forecast, default) or smape_median (median forecast); a string for all segments or an object mapping segment name to metric"),
    work_dir: tool.schema.string().optional().describe("Directory for the output parquet files (default: current directory)"),
  },

  async execute(args) {
    const cmd = `python -m tsf_lib.cli select --args '${JSON.stringify(args).replace(/'/g, "'\\''")}'`
    const result = await Bun.$`sh -c ${cmd}`.nothrow()
    const stdout = result.stdout.toString().trim()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr}\n\nStdout:\n${stdout}`
    }

    return stdout
  },
})
