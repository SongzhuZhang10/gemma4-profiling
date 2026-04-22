#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

workflow_log "Fetching the packaged Edge-LLM export bundle onto the Thor target."
"$PYTHON_BIN" "$WORKFLOW_ROOT/profiling_workflow.py" target-fetch-export "$@"
