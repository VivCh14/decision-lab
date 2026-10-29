import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Build every report figure for a finished session. DO NOT WRITE PLOTTING CODE - use this tool.

Writes to output_dir (default report_figures/):
- 01_accuracy_by_segment_and_model.png - grouped bars, pooled accuracy per segment and model
- 02_best_forecasts_by_segment.png - grid of the top-N series per segment: history, shaded backtest window, per-fold predictions, forecast with interval
- 03_worst_forecasts_by_segment.png - the same grid for the worst-N series per segment
- 04_wape_by_horizon_step.png - error growth across horizon steps
- 05_interval_coverage_vs_nominal.png - realised coverage against the nominal level
- 06_baseline_comparison_per_series.png - selected model against the baseline on the same series
- report_summary.json - the same numbers in machine-readable form

Series are ranked on the metric each segment was selected with (from selection.parquet), and labelled with WAPE and accuracy (1 - WAPE). Accuracy is unbounded below: a model far worse than the actuals scores strongly negative. Returns the figure paths and the pooled accuracy table; read report_summary.json for the per-series records, and cite those numbers in your reports rather than re-deriving them.`,

  args: {
    selection_file: tool.schema.string().describe("Path to selection.parquet from select-models (supplies segment, selected model and metric basis per series)"),
    scores_file: tool.schema.string().describe("Path to scores.parquet from score-models"),
    metrics_file: tool.schema.string().describe("Path to metrics.parquet from compute-metrics"),
    series_file: tool.schema.string().describe("Path to the cleaned long series parquet (series_id, date, value) used for history"),
    backtest_files: tool.schema.array(tool.schema.string()).describe("Backtest parquet paths for every model, including the baseline"),
    forecast_files: tool.schema.array(tool.schema.string()).optional().describe("Forecast parquet paths for every selected model"),
    output_dir: tool.schema.string().optional().describe("Directory for figures and report_summary.json (default: report_figures/ under the current directory)"),
    top_n: tool.schema.number().optional().describe("Panels per segment in the best and worst grids (default: 3)"),
    baseline: tool.schema.string().optional().describe("Baseline model name kept in every comparison (default: seasonal_naive)"),
    work_dir: tool.schema.string().optional().describe("Directory that relative input paths resolve against (default: current directory)"),
  },

  async execute(args) {
    const cmd = `python -m tsf_lib.cli report --args '${JSON.stringify(args).replace(/'/g, "'\\''")}'`
    const result = await Bun.$`sh -c ${cmd}`.nothrow()
    const stdout = result.stdout.toString().trim()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr}\n\nStdout:\n${stdout}`
    }

    return stdout
  },
})
