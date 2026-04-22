#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

workflow_log "02_prepare_model.sh is deprecated."

case "$(uname -m)" in
  x86_64|amd64)
    workflow_log "Dispatching to host_02_export_model.sh on the x86 export host."
    exec bash "$WORKFLOW_ROOT/host_02_export_model.sh" "$@"
    ;;
  aarch64|arm64)
    workflow_log "Model export now happens on the x86 host. This target-side compatibility step is a no-op."
    workflow_log "Use target_02_fetch_export.sh --source user@host:/abs/path/<bundle>.tar.gz to import the packaged export onto Thor."
    exit 0
    ;;
  *)
    workflow_log "Could not infer host vs target from architecture $(uname -m)."
    workflow_log "Run host_02_export_model.sh on the export host or target_02_fetch_export.sh on the Thor target explicitly."
    exit 1
    ;;
esac
