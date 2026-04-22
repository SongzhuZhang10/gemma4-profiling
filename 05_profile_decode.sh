#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "05_profile_decode.sh is deprecated. Dispatching to target_05_profile_decode.sh."
exec bash "$WORKFLOW_ROOT/target_05_profile_decode.sh" "$@"
