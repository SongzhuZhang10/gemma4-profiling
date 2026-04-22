#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

workflow_log "Packaging the exported TensorRT Edge-LLM ONNX bundle for transfer to Thor."
"$PYTHON_BIN" "$WORKFLOW_ROOT/profiling_workflow.py" host-package-export "$@"
