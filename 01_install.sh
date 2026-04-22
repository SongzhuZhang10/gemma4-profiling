#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "01_install.sh is deprecated. Use host_01_install.sh on the x86 export host or target_01_install.sh on the Thor target."

case "$(uname -m)" in
  x86_64|amd64)
    exec bash "$WORKFLOW_ROOT/host_01_install.sh" "$@"
    ;;
  aarch64|arm64)
    exec bash "$WORKFLOW_ROOT/target_01_install.sh" "$@"
    ;;
  *)
    workflow_log "Could not infer host vs target from architecture $(uname -m)."
    workflow_log "Run host_01_install.sh or target_01_install.sh explicitly."
    exit 1
    ;;
esac
