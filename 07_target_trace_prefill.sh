#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

prefill_nsys_stem="$REPORTS_DIR/prefill_trace"
prefill_nsys_report="$REPORTS_DIR/prefill.nsys-rep"
prefill_metadata_json="$REPORTS_DIR/prefill_nsys_run_metadata.json"
prefill_runtime_profile_json="$REPORTS_DIR/prefill_nsys_runtime_profile.json"
prefill_launcher_script="$REPORTS_DIR/prefill_nsys_launch.sh"

rm -f \
  "$prefill_metadata_json" \
  "$prefill_runtime_profile_json" \
  "$prefill_launcher_script" \
  "$prefill_nsys_stem.nsys-rep" \
  "$prefill_nsys_stem.sqlite" \
  "$prefill_nsys_report" \
  "$REPORTS_DIR/prefill.sqlite"

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" emit-inference-launch \
  --phase prefill \
  --max-generate-length 1 \
  --output "$prefill_metadata_json" \
  --runtime-profile-output "$prefill_runtime_profile_json"

"$PYTHON_BIN" - "$prefill_metadata_json" "$prefill_launcher_script" <<'PY'
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

workflow_log "Capturing Nsight Systems timeline for the prefill workload."
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --sample=none \
  --force-overwrite=true \
  --output "$prefill_nsys_stem" \
  "$prefill_launcher_script"

if [[ ! -f "$prefill_nsys_stem.nsys-rep" ]]; then
  workflow_log "Expected Nsight Systems report was not produced: $prefill_nsys_stem.nsys-rep"
  exit 1
fi

mv -f "$prefill_nsys_stem.nsys-rep" "$prefill_nsys_report"
rm -f "$prefill_launcher_script"

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" summarize-nsys \
  --phase prefill \
  --report "$prefill_nsys_report"

workflow_log "Saved prefill Nsight Systems report to $prefill_nsys_report"
