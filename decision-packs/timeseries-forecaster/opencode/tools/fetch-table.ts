import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Read rows of an output parquet file in work_dir, optionally filtered with a pandas query.`,

  args: {
    file: tool.schema.string().describe("File name, e.g. scores.parquet"),
    where: tool.schema.string().optional().describe("pandas query string"),
    columns: tool.schema.array(tool.schema.string()).optional().describe("Columns to return"),
    limit: tool.schema.number().optional().describe("Max rows (default 50)"),
    work_dir: tool.schema.string().optional().describe("Directory for the output parquet files (default: current directory)"),
  },

  async execute(args) {
    const cmd = `python -m tsf_lib.cli fetch --args '${JSON.stringify(args).replace(/'/g, "'\\''")}'`
    const result = await Bun.$`sh -c ${cmd}`.nothrow()
    const stdout = result.stdout.toString().trim()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr}\n\nStdout:\n${stdout}`
    }

    return stdout
  },
})
