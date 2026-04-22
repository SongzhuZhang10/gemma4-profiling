#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

workflow_log "Exporting meta-llama/Llama-3.1-8B-Instruct to TensorRT Edge-LLM ONNX in FP16."
"$PYTHON_BIN" "$WORKFLOW_ROOT/profiling_workflow.py" host-export "$@"
