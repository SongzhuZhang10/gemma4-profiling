#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "08_inspect_ncu.sh is deprecated. Dispatching to target_08_inspect_ncu.sh."
exec bash "$WORKFLOW_ROOT/target_08_inspect_ncu.sh" "$@"
