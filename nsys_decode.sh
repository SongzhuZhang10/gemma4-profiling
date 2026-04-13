#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# Reuse the prepared runtime and capture a fresh timeline so decode-step selection reflects the full 128-token run.
bash "$WORKFLOW_ROOT/03_build_or_prepare_runtime.sh"

decode_nsys_stem="$REPORTS_DIR/decode"
decode_metadata_json="$REPORTS_DIR/decode_nsys_run_metadata.json"

workflow_log "Capturing a full CUDA+NVTX timeline for decode phase selection."
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output "$decode_nsys_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference \
  --max-new-tokens 128 \
  --metadata-output "$decode_metadata_json"

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" summarize-nsys \
  --phase decode \
  --report "$REPORTS_DIR/decode.nsys-rep"

workflow_log "Saved decode Nsight Systems report to $REPORTS_DIR/decode.nsys-rep"

