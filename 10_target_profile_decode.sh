#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

readonly DECODE_NCU_MINIMAL_METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

decode_ncu_stem="$REPORTS_DIR/decode"
decode_ncu_report="$REPORTS_DIR/decode.ncu-rep"
decode_metadata_json="$REPORTS_DIR/decode_ncu_run_metadata.json"
decode_runtime_profile_json="$REPORTS_DIR/decode_ncu_runtime_profile.json"

rm -f "$decode_ncu_report" "$decode_metadata_json" "$decode_runtime_profile_json"

workflow_log "Capturing Nsight Compute profile for the decode workload."
# Use --replay-mode application because the 128-token decode loop depends on KV-cache and RNG state.
ncu \
  --target-processes all \
  --nvtx \
  --nvtx-include "LLM_GENERATION/" \
  --clock-control none \
  --cache-control=all \
  --replay-mode application \
  --metrics "$DECODE_NCU_MINIMAL_METRICS" \
  --disable-extra-suffixes \
  --force-overwrite \
  -o "$decode_ncu_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" run-inference \
  --phase decode \
  --max-generate-length 128 \
  --metadata-output "$decode_metadata_json" \
  --runtime-profile-output "$decode_runtime_profile_json"

if [[ ! -f "$decode_ncu_report" ]]; then
  workflow_log "Expected Nsight Compute report was not produced: $decode_ncu_report"
  exit 1
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" register-ncu \
  --phase decode \
  --report "$decode_ncu_report" \
  --metadata "$decode_metadata_json" \
  --collection-backend tensorrt_edge_llm \
  --replay-mode application \
  --collection-profile minimal \
  --phase-filter LLM_GENERATION/ \
  --runtime-profile "$decode_runtime_profile_json" \
  --requested-max-new-tokens 128

workflow_log "Saved decode Nsight Compute report to $decode_ncu_report"
