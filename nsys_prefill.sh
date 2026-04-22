#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "nsys_prefill.sh is deprecated. Dispatching to target_nsys_prefill.sh."
exec bash "$WORKFLOW_ROOT/target_nsys_prefill.sh" "$@"
