#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "03_build_or_prepare_runtime.sh is deprecated."

if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
  model_slug="meta-llama__Llama-3.1-8B-Instruct"
  if [[ ! -d "$ARTIFACTS_DIR/export/$model_slug/onnx" ]]; then
    workflow_log "The Thor runtime build now depends on a packaged host export bundle."
    workflow_log "No imported ONNX export was found under $ARTIFACTS_DIR/export/$model_slug/onnx, so this compatibility step is a no-op."
    workflow_log "Run target_02_fetch_export.sh --source user@host:/abs/path/<bundle>.tar.gz when the x86 host export is ready."
    exit 0
  fi
fi

workflow_log "Dispatching to target_03_build_engine.sh."
exec bash "$WORKFLOW_ROOT/target_03_build_engine.sh" "$@"
