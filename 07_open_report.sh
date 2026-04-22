#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "07_open_report.sh is deprecated. Dispatching to target_07_open_report.sh."
exec bash "$WORKFLOW_ROOT/target_07_open_report.sh" "$@"
