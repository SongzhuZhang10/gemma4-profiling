#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

decode_probe_metadata_json="$REPORTS_DIR/decode_ncu_probe_metadata.json"
decode_probe_runtime_profile_json="$REPORTS_DIR/decode_ncu_probe_runtime_profile.json"
decode_summary_json="$REPORTS_DIR/decode_ncu_steady_state_summary.json"
decode_phase_summary_json="$REPORTS_DIR/decode_phase_summary.json"
decode_launcher_metadata_json="$REPORTS_DIR/decode_ncu_launch_metadata.json"
decode_launcher_script="$REPORTS_DIR/decode_ncu_launch.sh"

rm -f \
  "$REPORTS_DIR/decode.ncu-rep" \
  "$REPORTS_DIR/decode.ncu-slice.json" \
  "$REPORTS_DIR/decode_ncu_run_metadata.json" \
  "$REPORTS_DIR/decode_ncu_runtime_profile.json" \
  "$decode_probe_metadata_json" \
  "$decode_probe_runtime_profile_json" \
  "$decode_summary_json" \
  "$decode_launcher_metadata_json" \
  "$decode_launcher_script" \
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

if [[ "${#decode_iterations[@]}" -eq 0 ]]; then
  workflow_log "Decode probe did not resolve any steady-state iterations."
  exit 1
fi

if [[ ! -f "$decode_phase_summary_json" ]]; then
  workflow_log "Decode phase summary is required before decode NCU profiling: $decode_phase_summary_json"
  workflow_log "Run bash 09_target_trace_decode.sh first to refresh the phase summary."
  exit 1
fi

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

iteration_report_args=()
iteration_metadata_args=()

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" emit-inference-launch \
  --phase decode \
  --max-generate-length 128 \
  --output "$decode_launcher_metadata_json" \
  --runtime-profile-output "$REPORTS_DIR/decode_ncu_killed_runtime_profile.json"

"$PYTHON_BIN" - "$decode_launcher_metadata_json" "$decode_launcher_script" <<'PY'
import json
import pathlib
import shlex
import sys

metadata_path = pathlib.Path(sys.argv[1])
launcher_path = pathlib.Path(sys.argv[2])
payload = json.loads(metadata_path.read_text())
env_overrides = payload.get("env_overrides", {})
run_command = payload["run_command"]
cwd = payload.get("cwd")

lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
for key, value in sorted(env_overrides.items()):
    lines.append(f"export {key}={shlex.quote(str(value))}")
if cwd:
    lines.append(f"cd {shlex.quote(str(cwd))}")
lines.append("exec " + " ".join(shlex.quote(str(part)) for part in run_command))
launcher_path.write_text("\n".join(lines) + "\n")
launcher_path.chmod(0o755)
PY

for iteration in "${decode_iterations[@]}"; do
  iteration_padded="$(printf "%03d" "$iteration")"
  decode_ncu_stem="$REPORTS_DIR/decode_iter${iteration_padded}"
  decode_ncu_report="${decode_ncu_stem}.ncu-rep"
  decode_metadata_json="$REPORTS_DIR/decode_iter${iteration_padded}_ncu_run_metadata.json"
  read -r decode_phase_filter decode_launch_skip decode_launch_count < <("$PYTHON_BIN" - "$decode_probe_metadata_json" "$decode_phase_summary_json" "$iteration" <<'PY'
import json
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
phase_summary_path = pathlib.Path(sys.argv[2])
iteration = int(sys.argv[3])
payload = json.loads(metadata_path.read_text())
selection = payload.get("decode_steady_state_iteration_selection", {})
phase_summary = json.loads(phase_summary_path.read_text())
gpu_activity = phase_summary.get("nsys_activity_summary", {}).get("gpu_activity", {})
kernel_launch_count = int(gpu_activity.get("kernel_launch_count") or 0)
actual_output_token_count = selection.get("actual_output_token_count")
if actual_output_token_count is None:
    actual_output_token_count = payload.get("actual_output_token_count")
decode_loop_count = max(0, int(actual_output_token_count) - 1)
if decode_loop_count <= 0:
    raise SystemExit(1)
if kernel_launch_count <= 0 or kernel_launch_count % decode_loop_count != 0:
    raise SystemExit(1)
launches_per_iteration = kernel_launch_count // decode_loop_count
launch_skip = max(0, (iteration - 1) * launches_per_iteration)
print("LLM_GENERATION/", launch_skip, launches_per_iteration)
PY
  )

  "$PYTHON_BIN" - "$decode_probe_metadata_json" "$decode_metadata_json" "$iteration" "$decode_phase_filter" "$decode_launch_skip" "$decode_launch_count" <<'PY'
import json
import pathlib
import sys

probe_metadata_path = pathlib.Path(sys.argv[1])
iteration_metadata_path = pathlib.Path(sys.argv[2])
iteration = int(sys.argv[3])
phase_filter = str(sys.argv[4])
launch_skip = int(sys.argv[5])
launch_count = int(sys.argv[6])

payload = json.loads(probe_metadata_path.read_text())
payload["ncu_iteration"] = iteration
payload["phase_filter"] = phase_filter
payload["ncu_launch_skip"] = launch_skip
payload["ncu_launch_count"] = launch_count
payload["ncu_selection_strategy"] = "outer_generation_launch_window"
payload["runtime_profile_json"] = None
iteration_metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

  workflow_log \
    "Capturing Nsight Compute profile for steady-state decode iteration ${iteration} " \
    "(launch-skip=${decode_launch_skip}, launch-count=${decode_launch_count})."
  # The per-iteration NVTX subranges do not match Nsight Compute's kernel-context filters reliably
  # with this runtime, so we select the steady-state iteration by launch window inside LLM_GENERATION.
  ncu \
    --target-processes all \
    --nvtx \
    --nvtx-include "$decode_phase_filter" \
    --clock-control none \
    --cache-control=all \
    --replay-mode application \
    --launch-skip "$decode_launch_skip" \
    --launch-count "$decode_launch_count" \
    --kill on \
    --check-exit-code 0 \
    --set basic \
    --disable-extra-suffixes \
    --force-overwrite \
    -o "$decode_ncu_stem" \
    "$decode_launcher_script"

  if [[ ! -f "$decode_ncu_report" ]]; then
    workflow_log "Expected Nsight Compute report was not produced: $decode_ncu_report"
    exit 1
  fi

  iteration_report_args+=(--iteration-report "${iteration}=${decode_ncu_report}")
  iteration_metadata_args+=(--iteration-metadata "${iteration}=${decode_metadata_json}")
done

rm -f "$decode_launcher_script"

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" register-decode-steady-state-ncu \
  --selection-metadata "$decode_probe_metadata_json" \
  "${iteration_report_args[@]}" \
  "${iteration_metadata_args[@]}" \
  --summary-output "$decode_summary_json" \
  --collection-backend tensorrt_edge_llm \
  --replay-mode application \
  --collection-profile basic

workflow_log \
  "Saved steady-state decode Nsight Compute reports for iterations ${decode_iterations[*]} (representative: ${representative_iteration})"
