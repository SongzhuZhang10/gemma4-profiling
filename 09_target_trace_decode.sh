#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

decode_nsys_stem="$REPORTS_DIR/decode"
decode_metadata_json="$REPORTS_DIR/decode_nsys_run_metadata.json"
decode_runtime_profile_json="$REPORTS_DIR/decode_nsys_runtime_profile.json"

rm -f "$decode_metadata_json" "$decode_runtime_profile_json"

workflow_log "Capturing Nsight Systems timeline for the decode workload."
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --capture-range=nvtx \
  --nvtx-capture 'LLM_GENERATION@*' \
  --capture-range-end=stop \
  --sample=none \
  --force-overwrite=true \
  --output "$decode_nsys_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" run-inference \
  --phase decode \
  --metadata-output "$decode_metadata_json" \
  --runtime-profile-output "$decode_runtime_profile_json"

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" summarize-nsys \
  --phase decode \
  --report "$REPORTS_DIR/decode.nsys-rep"

workflow_log "Saved decode Nsight Systems report to $REPORTS_DIR/decode.nsys-rep"
