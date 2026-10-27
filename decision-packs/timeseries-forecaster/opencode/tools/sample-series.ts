import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Stratified sample of series ids (by data quality, demand class and value kind) for the development loop.`,

  args: {
    n: tool.schema.number().describe("Number of series wanted"),
    seed: tool.schema.number().optional().describe("Random seed (default 0)"),
    work_dir: tool.schema.string().optional().describe("Directory for the output parquet files (default: current directory)"),
  },

  async execute(args) {
    const cmd = `python -m tsf_lib.cli sample --args '${JSON.stringify(args).replace(/'/g, "'\\''")}'`
    const result = await Bun.$`sh -c ${cmd}`.nothrow()
    const stdout = result.stdout.toString().trim()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr}\n\nStdout:\n${stdout}`
    }

    return stdout
  },
})
