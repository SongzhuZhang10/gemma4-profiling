#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

prefill_nsys_stem="$REPORTS_DIR/prefill"
prefill_metadata_json="$REPORTS_DIR/prefill_nsys_run_metadata.json"

workflow_log "Capturing Nsight Systems timeline for the prefill workload."
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output "$prefill_nsys_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" run-inference \
  --phase prefill \
  --metadata-output "$prefill_metadata_json"

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" summarize-nsys \
  --phase prefill \
  --report "$REPORTS_DIR/prefill.nsys-rep"

workflow_log "Saved prefill Nsight Systems report to $REPORTS_DIR/prefill.nsys-rep"
