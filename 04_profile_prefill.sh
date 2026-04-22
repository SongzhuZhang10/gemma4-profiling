#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "04_profile_prefill.sh is deprecated. Dispatching to target_04_profile_prefill.sh."
exec bash "$WORKFLOW_ROOT/target_04_profile_prefill.sh" "$@"
