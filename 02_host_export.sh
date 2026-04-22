#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

workflow_log "Exporting the configured Llama 3.1 8B Instruct model to TensorRT Edge-LLM ONNX in FP16."
"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" host-export "$@"
