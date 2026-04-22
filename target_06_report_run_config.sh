#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

report_output="$REPORTS_DIR/run_config_summary.txt"

"$PYTHON_BIN" "$WORKFLOW_ROOT/profiling_workflow.py" report-config --output "$report_output"

workflow_log "Saved run configuration summary to $report_output"
