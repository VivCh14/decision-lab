import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Inspect the series (and regressor) files: number of series, date range, a few rows. Run first, before analyze-series.`,

  args: {
    series_file: tool.schema.string().optional().describe("Series file (.csv/.parquet); remembered in inputs.json after analyze-series"),
    series_id_col: tool.schema.string().optional().describe("Series id column (default series_id)"),
    date_col: tool.schema.string().optional().describe("Date column (default date)"),
    value_col: tool.schema.string().optional().describe("Value column (default value)"),
    regressor_file: tool.schema.string().optional().describe("Long regressor file with series id, regressor name, date and value columns"),
    regressor_id_col: tool.schema.string().optional().describe("Regressor series id column (default: series id column)"),
    regressor_name_col: tool.schema.string().optional().describe("Regressor name column (default regressor_name)"),
    regressor_date_col: tool.schema.string().optional().describe("Regressor date column (default: date column)"),
    regressor_value_col: tool.schema.string().optional().describe("Regressor value column (default value)"),
    work_dir: tool.schema.string().optional().describe("Directory for the output parquet files (default: current directory)"),
  },

  async execute(args) {
    const cmd = `python -m tsf_lib.cli inspect --args '${JSON.stringify(args).replace(/'/g, "'\\''")}'`
    const result = await Bun.$`sh -c ${cmd}`.nothrow()
    const stdout = result.stdout.toString().trim()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr}\n\nStdout:\n${stdout}`
    }

    return stdout
  },
})
