#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

workflow_log "Packaging the exported TensorRT Edge-LLM ONNX bundle for transfer to Thor."
"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" host-package-export "$@"
