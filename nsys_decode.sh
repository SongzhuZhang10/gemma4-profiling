#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "nsys_decode.sh is deprecated. Dispatching to target_nsys_decode.sh."
exec bash "$WORKFLOW_ROOT/target_nsys_decode.sh" "$@"
