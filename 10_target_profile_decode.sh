#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

readonly DECODE_NCU_MINIMAL_METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

decode_probe_metadata_json="$REPORTS_DIR/decode_ncu_probe_metadata.json"
decode_probe_runtime_profile_json="$REPORTS_DIR/decode_ncu_probe_runtime_profile.json"
decode_summary_json="$REPORTS_DIR/decode_ncu_steady_state_summary.json"

rm -f \
  "$REPORTS_DIR/decode.ncu-rep" \
  "$REPORTS_DIR/decode.ncu-slice.json" \
  "$REPORTS_DIR/decode_ncu_run_metadata.json" \
  "$REPORTS_DIR/decode_ncu_runtime_profile.json" \
  "$decode_probe_metadata_json" \
  "$decode_probe_runtime_profile_json" \
  "$decode_summary_json" \
  "$REPORTS_DIR"/decode_iter*.ncu-rep \
  "$REPORTS_DIR"/decode_iter*.ncu-slice.json \
  "$REPORTS_DIR"/decode_iter*_ncu_run_metadata.json \
  "$REPORTS_DIR"/decode_iter*_ncu_runtime_profile.json

workflow_log "Running a decode probe to resolve the steady-state decode iterations."
"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" run-inference \
  --phase decode \
  --max-generate-length 128 \
  --metadata-output "$decode_probe_metadata_json" \
  --runtime-profile-output "$decode_probe_runtime_profile_json"

mapfile -t decode_iterations < <("$PYTHON_BIN" - "$decode_probe_metadata_json" <<'PY'
import json
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
payload = json.loads(metadata_path.read_text())
selection = payload.get("decode_steady_state_iteration_selection", {})
iterations = selection.get("iterations", [])
for iteration in iterations:
    print(int(iteration))
PY
)

representative_iteration="$("$PYTHON_BIN" - "$decode_probe_metadata_json" <<'PY'
import json
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
payload = json.loads(metadata_path.read_text())
selection = payload.get("decode_steady_state_iteration_selection", {})
representative_iteration = selection.get("representative_iteration")
if representative_iteration is None:
    raise SystemExit(1)
print(int(representative_iteration))
PY
)"

if [[ "${#decode_iterations[@]}" -eq 0 ]]; then
  workflow_log "Decode probe did not resolve any steady-state iterations."
  exit 1
fi

iteration_report_args=()
iteration_metadata_args=()
iteration_runtime_profile_args=()

for iteration in "${decode_iterations[@]}"; do
  iteration_padded="$(printf "%03d" "$iteration")"
  decode_ncu_stem="$REPORTS_DIR/decode_iter${iteration_padded}"
  decode_ncu_report="${decode_ncu_stem}.ncu-rep"
  decode_metadata_json="$REPORTS_DIR/decode_iter${iteration_padded}_ncu_run_metadata.json"
  decode_runtime_profile_json="$REPORTS_DIR/decode_iter${iteration_padded}_ncu_runtime_profile.json"
  decode_phase_filter="LLM_GENERATION/DECODE_ITER_${iteration_padded}/"

  workflow_log "Capturing Nsight Compute profile for steady-state decode iteration ${iteration}."
  # Use --replay-mode application because the 128-token decode loop depends on KV-cache and RNG state.
  ncu \
    --target-processes all \
    --nvtx \
    --nvtx-include "$decode_phase_filter" \
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

  iteration_report_args+=(--iteration-report "${iteration}=${decode_ncu_report}")
  iteration_metadata_args+=(--iteration-metadata "${iteration}=${decode_metadata_json}")
  iteration_runtime_profile_args+=(--iteration-runtime-profile "${iteration}=${decode_runtime_profile_json}")
done

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" register-decode-steady-state-ncu \
  --selection-metadata "$decode_probe_metadata_json" \
  "${iteration_report_args[@]}" \
  "${iteration_metadata_args[@]}" \
  "${iteration_runtime_profile_args[@]}" \
  --summary-output "$decode_summary_json" \
  --collection-backend tensorrt_edge_llm \
  --replay-mode application \
  --collection-profile minimal

workflow_log \
  "Saved steady-state decode Nsight Compute reports for iterations ${decode_iterations[*]} (representative: ${representative_iteration})"
