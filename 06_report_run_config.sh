#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "06_report_run_config.sh is deprecated. Dispatching to target_06_report_run_config.sh."
exec bash "$WORKFLOW_ROOT/target_06_report_run_config.sh" "$@"
