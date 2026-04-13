#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# The decode profile reuses an Nsight Systems trace so we can target a middle generation-only step instead of startup or drain.
bash "$WORKFLOW_ROOT/07_nsys_decode.sh"

decode_nvtx_filter="$("$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" nvtx-filter --phase decode)"
decode_ncu_stem="$REPORTS_DIR/decode"
decode_metadata_json="$REPORTS_DIR/decode_ncu_run_metadata.json"

workflow_log "Profiling the decode phase with Nsight Compute."
ncu \
  --nvtx \
  --nvtx-include "$decode_nvtx_filter" \
  --target-processes all \
  --set full \
  --replay-mode kernel \
  --force-overwrite \
  -o "$decode_ncu_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference \
  --max-new-tokens 128 \
  --metadata-output "$decode_metadata_json"

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" register-ncu \
  --phase decode \
  --report "$REPORTS_DIR/decode.ncu-rep"

workflow_log "Saved decode Nsight Compute report to $REPORTS_DIR/decode.ncu-rep"

