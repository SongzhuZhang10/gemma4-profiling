#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

ensure_selected_python_env
require_python_env
ensure_workflow_dirs
ensure_vscode_settings

workflow_log "Installing host-side Python dependencies for the TensorRT Edge-LLM export workflow."
"$PIP_BIN" install --upgrade pip 'setuptools<80' 'wheel<=0.45.1'
"$PIP_BIN" install --upgrade \
  'numpy<2' \
  'pandas>=2.1,<2.2' \
  huggingface_hub==0.36.2 \
  transformers==4.56.0 \
  datasets==3.1.0 \
  nvidia-modelopt==0.37.0 \
  peft==0.19.1 \
  onnx==1.19.1 \
  sentencepiece==0.2.1 \
  PyYAML==6.0.3

if [[ -d "$EDGE_LLM_REPO_DIR" ]]; then
  workflow_log "Installing TensorRT Edge-LLM export tools from $EDGE_LLM_REPO_DIR."
  "$PIP_BIN" install --upgrade --no-deps -e "$EDGE_LLM_REPO_DIR"
else
  workflow_log "TensorRT Edge-LLM repo not found at $EDGE_LLM_REPO_DIR."
  workflow_log "If the export tools are already available in PATH, the workflow will reuse them."
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" host-preflight "$@"
