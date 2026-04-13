#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# Nsight Systems validates the phase marker first so the Nsight Compute filter is anchored to a real prefill step.
bash "$WORKFLOW_ROOT/06_nsys_prefill.sh"

prefill_nvtx_filter="$("$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" nvtx-filter --phase prefill)"
prefill_ncu_stem="$REPORTS_DIR/prefill"
prefill_metadata_json="$REPORTS_DIR/prefill_ncu_run_metadata.json"

workflow_log "Profiling the prefill phase with Nsight Compute."
ncu \
  --nvtx \
  --nvtx-include "$prefill_nvtx_filter" \
  --target-processes all \
  --set full \
  --replay-mode kernel \
  --force-overwrite \
  -o "$prefill_ncu_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference \
  --max-new-tokens 128 \
  --metadata-output "$prefill_metadata_json"

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" register-ncu \
  --phase prefill \
  --report "$REPORTS_DIR/prefill.ncu-rep"

workflow_log "Saved prefill Nsight Compute report to $REPORTS_DIR/prefill.ncu-rep"

