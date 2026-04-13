#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# The decode profile reuses an Nsight Systems trace so we can target a middle generation-only step instead of startup or drain.
bash "$WORKFLOW_ROOT/07_nsys_decode.sh"

decode_ncu_stem="$REPORTS_DIR/decode"
decode_ncu_report="$REPORTS_DIR/decode.ncu-rep"
decode_metadata_json="$REPORTS_DIR/decode_ncu_run_metadata.json"
decode_step_number="$(
  WORKFLOW_ROOT="$WORKFLOW_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.environ["WORKFLOW_ROOT"]) / "artifacts" / "run_config.json"
config = json.loads(config_path.read_text())
step = config.get("profiles", {}).get("decode", {}).get("selected_step_number")
if step is None:
    raise SystemExit(1)
print(step)
PY
)"

workflow_log "Profiling the decode phase with Nsight Compute."
workflow_log "Targeting decode executor step $decode_step_number with worker-side CUDA profiler control."
rm -f /tmp/nsight-compute-lock
rm -f "$decode_ncu_report"
# TensorRT-LLM's decode kernels are launched by the internal executor worker,
# so parent-side cudaProfilerStart/Stop calls miss the actual kernels. Use the
# built-in PyExecutor worker-side profiler window instead: TLLM_PROFILE_START_STOP
# keys off the same `_forward_step N` iteration counter exposed in the Nsight
# Systems NVTX labels, letting us target one decode-only executor step directly.
TLLM_PROFILE_START_STOP="${decode_step_number}-${decode_step_number}" ncu \
  --target-processes all \
  --profile-from-start off \
  --replay-mode kernel \
  --launch-count 1 \
  --metrics \
  gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  --disable-extra-suffixes \
  --force-overwrite \
  -o "$decode_ncu_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference \
  --max-new-tokens 128 \
  --metadata-output "$decode_metadata_json"

if [[ ! -f "$decode_ncu_report" ]]; then
  workflow_log "ERROR: ncu completed without producing $decode_ncu_report"
  workflow_log "  Decode step $decode_step_number was targeted via TLLM_PROFILE_START_STOP."
  workflow_log "  If ncu reported ResourceUnavailable or a driver resource error,"
  workflow_log "  the failure is now in Nsight Compute counter collection itself,"
  workflow_log "  not in decode-step selection."
  exit 1
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" register-ncu \
  --phase decode \
  --report "$decode_ncu_report"

workflow_log "Saved decode Nsight Compute report to $decode_ncu_report"
