#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

ensure_selected_python_env
require_python_env
ensure_workflow_dirs
ensure_vscode_settings

workflow_log "Installing target-side Python helpers for the Thor profiling workflow."
"$PIP_BIN" install --upgrade pip 'setuptools<80' 'wheel<=0.45.1'
"$PIP_BIN" install --upgrade \
  huggingface_hub==0.36.2 \
  transformers==4.56.0 \
  sentencepiece==0.2.1 \
  PyYAML==6.0.3

if ! command -v cmake >/dev/null 2>&1; then
  workflow_log "cmake was not found in PATH. Installing a user-space cmake wheel into $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade 'cmake>=3.28,<4'
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" target-preflight "$@"
