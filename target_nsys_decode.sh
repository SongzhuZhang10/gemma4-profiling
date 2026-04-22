#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

bash "$WORKFLOW_ROOT/target_03_build_engine.sh"

decode_nsys_stem="$REPORTS_DIR/decode"
decode_metadata_json="$REPORTS_DIR/decode_nsys_run_metadata.json"

workflow_log "Capturing Nsight Systems timeline for the decode workload."
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output "$decode_nsys_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/profiling_workflow.py" run-inference \
  --phase decode \
  --metadata-output "$decode_metadata_json"

"$PYTHON_BIN" "$WORKFLOW_ROOT/profiling_workflow.py" summarize-nsys \
  --phase decode \
  --report "$REPORTS_DIR/decode.nsys-rep"

workflow_log "Saved decode Nsight Systems report to $REPORTS_DIR/decode.nsys-rep"
