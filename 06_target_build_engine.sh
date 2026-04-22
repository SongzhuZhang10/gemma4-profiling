#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

workflow_log "Building the FP16 TensorRT Edge-LLM engine on the Thor target."
"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" build-engine "$@"
