#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

prefill_ncu_stem="$REPORTS_DIR/prefill"
prefill_ncu_report="$REPORTS_DIR/prefill.ncu-rep"
prefill_metadata_json="$REPORTS_DIR/prefill_ncu_run_metadata.json"
prefill_runtime_profile_json="$REPORTS_DIR/prefill_ncu_runtime_profile.json"

rm -f "$prefill_ncu_report" "$prefill_metadata_json" "$prefill_runtime_profile_json"

workflow_log "Capturing Nsight Compute profile for the prefill workload."
# LLM_PREFILL is the confirmed stable outer prefill range in the current
# TensorRT-Edge-LLM C++ binary.
ncu \
  --target-processes all \
  --nvtx \
  --nvtx-include "LLM_PREFILL/" \
  --replay-mode kernel \
  --clock-control none \
  --set basic \
  --disable-extra-suffixes \
  --force-overwrite \
  -o "$prefill_ncu_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" run-inference \
  --phase prefill \
  --max-generate-length 1 \
  --metadata-output "$prefill_metadata_json" \
  --runtime-profile-output "$prefill_runtime_profile_json"

if [[ ! -f "$prefill_ncu_report" ]]; then
  workflow_log "Expected Nsight Compute report was not produced: $prefill_ncu_report"
  exit 1
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" register-ncu \
  --phase prefill \
  --report "$prefill_ncu_report" \
  --metadata "$prefill_metadata_json" \
  --collection-backend tensorrt_edge_llm \
  --replay-mode kernel \
  --collection-profile basic \
  --phase-filter LLM_PREFILL/ \
  --runtime-profile "$prefill_runtime_profile_json" \
  --requested-max-new-tokens 1

workflow_log "Saved prefill Nsight Compute report to $prefill_ncu_report"
