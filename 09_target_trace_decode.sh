#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

decode_nsys_stem="$REPORTS_DIR/decode_trace"
decode_nsys_report="$REPORTS_DIR/decode.nsys-rep"
decode_metadata_json="$REPORTS_DIR/decode_nsys_run_metadata.json"
decode_runtime_profile_json="$REPORTS_DIR/decode_nsys_runtime_profile.json"
decode_launcher_script="$REPORTS_DIR/decode_nsys_launch.sh"

rm -f \
  "$decode_metadata_json" \
  "$decode_runtime_profile_json" \
  "$decode_launcher_script" \
  "$decode_nsys_stem.nsys-rep" \
  "$decode_nsys_stem.sqlite" \
  "$decode_nsys_report" \
  "$REPORTS_DIR/decode.sqlite"

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" emit-inference-launch \
  --phase decode \
  --max-generate-length 128 \
  --output "$decode_metadata_json" \
  --runtime-profile-output "$decode_runtime_profile_json"

"$PYTHON_BIN" - "$decode_metadata_json" "$decode_launcher_script" <<'PY'
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

workflow_log "Capturing Nsight Systems timeline for the decode workload."
# The useful application-owned phase ranges are emitted in the default NVTX
# domain. Restricting NVTX tracing to that domain drops the non-default
# TensorRT/myelin NVTX domains while phase scoping still comes from LLM_GENERATION.
nsys profile \
  --trace=cuda,nvtx \
  --nvtx-domain-include default \
  --cuda-graph-trace=node \
  --sample=none \
  --force-overwrite=true \
  --output "$decode_nsys_stem" \
  "$decode_launcher_script"

if [[ ! -f "$decode_nsys_stem.nsys-rep" ]]; then
  workflow_log "Expected Nsight Systems report was not produced: $decode_nsys_stem.nsys-rep"
  exit 1
fi

mv -f "$decode_nsys_stem.nsys-rep" "$decode_nsys_report"
rm -f "$decode_launcher_script"

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" summarize-nsys \
  --phase decode \
  --report "$decode_nsys_report"

workflow_log "Saved decode Nsight Systems report to $decode_nsys_report"
