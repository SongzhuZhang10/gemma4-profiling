#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

readonly PREFILL_NCU_MINIMAL_METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

bash "$WORKFLOW_ROOT/07_target_trace_prefill.sh"

prefill_ncu_stem="$REPORTS_DIR/prefill"
prefill_ncu_report="$REPORTS_DIR/prefill.ncu-rep"
prefill_metadata_json="$REPORTS_DIR/prefill_ncu_run_metadata.json"

rm -f "$prefill_ncu_report" "$prefill_metadata_json"

workflow_log "Capturing Nsight Compute profile for the prefill workload."
ncu \
  --target-processes all \
  --replay-mode kernel \
  --clock-control none \
  --metrics "$PREFILL_NCU_MINIMAL_METRICS" \
  --disable-extra-suffixes \
  --force-overwrite \
  -o "$prefill_ncu_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" run-inference \
  --phase prefill \
  --metadata-output "$prefill_metadata_json"

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
  --collection-profile minimal \
  --requested-max-new-tokens 1

workflow_log "Saved prefill Nsight Compute report to $prefill_ncu_report"
