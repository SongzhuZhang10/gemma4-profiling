#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# Runtime preparation performs the unprofiled warmup and selects the profiled Gemma 4 configuration.
bash "$WORKFLOW_ROOT/03_build_or_prepare_runtime.sh"

prefill_nsys_stem="$REPORTS_DIR/prefill"
prefill_metadata_json="$REPORTS_DIR/prefill_nsys_run_metadata.json"

workflow_log "Capturing a full CUDA+NVTX timeline for prefill phase selection."
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output "$prefill_nsys_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference \
  --max-new-tokens 128 \
  --metadata-output "$prefill_metadata_json"

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" summarize-nsys \
  --phase prefill \
  --report "$REPORTS_DIR/prefill.nsys-rep"

workflow_log "Saved prefill Nsight Systems report to $REPORTS_DIR/prefill.nsys-rep"

