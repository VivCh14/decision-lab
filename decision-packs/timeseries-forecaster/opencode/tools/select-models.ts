import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Pick, per segment of an analysis column, the model with the lowest median sMAPE (selection.parquet); with forecast_files given, also writes final_forecast.parquet.`,

  args: {
    scores_file: tool.schema.string().describe("Path of the scores parquet file (from score-models) covering all candidate models"),
    forecast_files: tool.schema.array(tool.schema.string()).optional().describe("Paths of the forecast parquet files (from run-forecast) to assemble into final_forecast.parquet"),
    by: tool.schema.string().optional().describe("Analysis column defining segments (default demand_class)"),
    metric: tool.schema.union([tool.schema.string(), tool.schema.record(tool.schema.string(), tool.schema.string())]).optional().describe("smape (mean forecast, default) or smape_median (median forecast); a string for all segments or an object mapping segment name to metric"),
    output_dir: tool.schema.string().optional().describe("Directory for output files (default: selection/ under the current directory). Parallel instances should pass their own instance directory."),
    work_dir: tool.schema.string().optional().describe("Directory holding analysis.parquet and inputs.json (default: current directory)"),
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
