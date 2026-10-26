#!/usr/bin/env bash
# Pre-run hook: verify Databricks credentials, connectivity and worker packages.
set -e

echo "=== check_databricks.sh: pre-run hook ==="
for var in DATABRICKS_HOST DATABRICKS_TOKEN DATABRICKS_CLUSTER_ID; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set (add it to the .env file)" >&2
        exit 1
    fi
done

python -m tsf_lib.cli check

echo "=== check_databricks.sh: done ==="
