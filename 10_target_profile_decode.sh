#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

readonly DECODE_NCU_MINIMAL_METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

bash "$WORKFLOW_ROOT/09_target_trace_decode.sh"

decode_ncu_stem="$REPORTS_DIR/decode"
decode_ncu_report="$REPORTS_DIR/decode.ncu-rep"
decode_metadata_json="$REPORTS_DIR/decode_ncu_run_metadata.json"

rm -f "$decode_ncu_report" "$decode_metadata_json"

workflow_log "Capturing Nsight Compute profile for the decode workload."
ncu \
  --target-processes all \
  --replay-mode kernel \
  --clock-control none \
  --metrics "$DECODE_NCU_MINIMAL_METRICS" \
  --disable-extra-suffixes \
  --force-overwrite \
  -o "$decode_ncu_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" run-inference \
  --phase decode \
  --metadata-output "$decode_metadata_json"

if [[ ! -f "$decode_ncu_report" ]]; then
  workflow_log "Expected Nsight Compute report was not produced: $decode_ncu_report"
  exit 1
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" register-ncu \
  --phase decode \
  --report "$decode_ncu_report" \
  --metadata "$decode_metadata_json" \
  --collection-backend tensorrt_edge_llm \
  --replay-mode kernel \
  --collection-profile minimal \
  --requested-max-new-tokens 128

workflow_log "Saved decode Nsight Compute report to $decode_ncu_report"
