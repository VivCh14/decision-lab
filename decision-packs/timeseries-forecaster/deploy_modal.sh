#!/bin/bash
# timeseries-forecaster decision-pack prerun: deploy the Modal app that runs series batches
set -e

if [ "${DLAB_FIT_MODEL_LOCALLY}" = "1" ]; then
    echo "DLAB_FIT_MODEL_LOCALLY=1. Skipping Modal deployment."
    exit 0
fi

if [ -z "$MODAL_TOKEN_ID" ] || [ -z "$MODAL_TOKEN_SECRET" ]; then
    echo "Modal tokens not set. Series batches will run locally."
    echo "For cloud fan-out, add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to your .env"
    exit 0
fi

MODAL_APP="/opt/modal_app/tsf_runner.py"

if [ -f "$MODAL_APP" ]; then
    echo "Deploying Modal app..."
    modal deploy "$MODAL_APP"
    echo "Modal app deployed."
else
    echo "Warning: Modal app not found at $MODAL_APP, skipping deploy"
fi
