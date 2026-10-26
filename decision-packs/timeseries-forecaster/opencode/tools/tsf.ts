import { tool } from "@opencode-ai/plugin"

export default tool({
  description: `Run a time series batch command on Databricks (databricks-connect). Heavy work runs on the cluster; this returns small JSON results.

Commands (args is a JSON object):
- inspect:   {series_table, regressor_table?, series_id_col?, date_col?, value_col?, regressor_name_col?, regressor_value_col?} -> row counts, date range, regressor names
- analyze:   inspect args + {output_schema, run_id, alpha?, max_lag?, lag_max?, lag_method?, min_overlap?} -> writes <run_id> analysis and regressor-lag tables
- summarize: {output_schema, run_id} -> distribution of frequency / value kind / intermittency / data quality / stationarity / seasonality
- fetch:     {output_schema, run_id, kind: analysis|regressor_lags|backtest|forecast|scores, model?, dev?, where?, columns?, limit?} -> rows of an output table
- sample:    {output_schema, run_id, n, seed?} -> stratified series ids for development
- backtest:  inspect args + {output_schema, run_id, model_file, model_name, horizon, n_folds, step, params?, series_ids?, dev?} -> writes the backtest table
- forecast:  same as backtest but fits on the full series and forecasts horizon periods ahead (n_folds/step unused)
- score:     {output_schema, run_id, models: [model_name...], dev?} -> sMAPE per series and model, written to the scores table

model_file is a local Python file defining fit_forecast(train, future_dates, regressors, analysis, params); see the timeseries-forecasting skill. Use dev=true together with series_ids while developing so full-run tables are never overwritten.`,

  args: {
    command: tool.schema
      .enum(["check", "inspect", "analyze", "summarize", "fetch", "sample", "backtest", "forecast", "score"])
      .describe("Command to run"),
    args: tool.schema.string().optional().describe("JSON object with the command arguments"),
  },

  async execute({ command, args }) {
    const json = args ?? "{}"
    const result = await Bun.$`python -m tsf_lib.cli ${command} --args ${json}`.nothrow()
    const stdout = result.stdout.toString()
    const stderr = result.stderr.toString()

    if (result.exitCode !== 0) {
      return `ERROR (exit code ${result.exitCode}):\n${stderr.slice(-4000)}\n${stdout.slice(-1000)}`
    }
    return stdout.trim()
  },
})
