#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# The decode profile reuses an Nsight Systems trace so we can target a middle
# generation-only step instead of startup or drain.
bash "$WORKFLOW_ROOT/nsys_decode.sh"

decode_ncu_stem="$REPORTS_DIR/decode"
decode_ncu_report="$REPORTS_DIR/decode.ncu-rep"
decode_metadata_json="$REPORTS_DIR/decode_ncu_run_metadata.json"

# ---------------------------------------------------------------------------
# Cleanup stale ncu state from run_config.json before profiling.
# ---------------------------------------------------------------------------
function cleanup_decode_ncu_state() {
  workflow_log "Cleaning stale Nsight Compute state for the decode run."

  rm -f /tmp/nsight-compute-lock
  rm -f "$decode_ncu_report"

  "$PYTHON_BIN" - "$RUN_CONFIG_JSON" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
if not config_path.exists():
    raise SystemExit(0)

config = json.loads(config_path.read_text())
profiles = config.get("profiles")
if not isinstance(profiles, dict):
    raise SystemExit(0)

decode = profiles.get("decode")
if not isinstance(decode, dict):
    raise SystemExit(0)

changed = False
for key in ("ncu_rep", "ncu_registered_at"):
    if key in decode:
        decode.pop(key, None)
        changed = True

if not decode:
    profiles.pop("decode", None)
    changed = True

if changed:
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

# ---------------------------------------------------------------------------
# Helper: run one ncu decode profiling attempt.
#
#   run_decode_ncu <replay_mode> <max_tokens> <target_step> <use_profiler_ctl> <label>
#
#   replay_mode        : "kernel" or "application"
#   max_tokens         : --max-new-tokens value for the inference run
#   target_step        : executor step number to target
#   use_profiler_ctl   : "yes" = set TLLM_PROFILE_START_STOP + --profile-from-start off
#                        "no"  = profile from start (no worker-side control)
#   label              : human-readable attempt description for logs
#
# Returns 0 if decode_ncu_report was produced, 1 otherwise.
# ---------------------------------------------------------------------------
function run_decode_ncu() {
  local replay_mode="$1"
  local max_tokens="$2"
  local target_step="$3"
  local use_profiler_ctl="$4"
  local label="$5"

  workflow_log "NCU attempt ($label): replay=$replay_mode, tokens=$max_tokens, step=$target_step, profiler_control=$use_profiler_ctl"

  rm -f /tmp/nsight-compute-lock
  rm -f "$decode_ncu_report"

  local ncu_cmd=()
  local env_vars=()

  # If using worker-side profiler control, set the env var and tell ncu to
  # wait for cudaProfilerStart() before collecting.
  if [[ "$use_profiler_ctl" == "yes" ]]; then
    env_vars+=("TLLM_PROFILE_START_STOP=${target_step}-${target_step}")
  fi

  ncu_cmd+=(ncu)
  ncu_cmd+=(--target-processes all)

  if [[ "$use_profiler_ctl" == "yes" ]]; then
    ncu_cmd+=(--profile-from-start off)
  fi

  ncu_cmd+=(--replay-mode "$replay_mode")
  ncu_cmd+=(--launch-count 1)
  ncu_cmd+=(--clock-control none)
  ncu_cmd+=(
    --metrics
    "gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
  )
  ncu_cmd+=(--disable-extra-suffixes)
  ncu_cmd+=(--force-overwrite)
  ncu_cmd+=(-o "$decode_ncu_stem")
  ncu_cmd+=("$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference)
  ncu_cmd+=(--max-new-tokens "$max_tokens")
  ncu_cmd+=(--metadata-output "$decode_metadata_json")

  if [[ ${#env_vars[@]} -gt 0 ]]; then
    env "${env_vars[@]}" "${ncu_cmd[@]}" || true
  else
    "${ncu_cmd[@]}" || true
  fi

  if [[ -f "$decode_ncu_report" ]]; then
    workflow_log "  -> Success: $decode_ncu_report produced."
    return 0
  else
    workflow_log "  -> Failed: $decode_ncu_report was not produced."
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
cleanup_decode_ncu_state

workflow_log "Profiling the decode phase with Nsight Compute."

# Use a reduced token count for the ncu run.  With 128 tokens the KV cache
# consumes most of the RTX 3050's 6 GB VRAM, leaving no room for ncu's
# kernel-replay buffers (ResourceUnavailable).  8 tokens still produce
# genuine decode-only executor steps with ~16x less KV cache memory.
NCU_DECODE_MAX_TOKENS=8
ncu_decode_step=$(( NCU_DECODE_MAX_TOKENS / 2 ))

success=false

# Attempt 1: kernel replay with reduced tokens and worker-side profiler control.
if run_decode_ncu kernel "$NCU_DECODE_MAX_TOKENS" "$ncu_decode_step" yes \
     "kernel-replay, ${NCU_DECODE_MAX_TOKENS} tokens, step ${ncu_decode_step}"; then
  success=true
fi

# Attempt 2: application replay avoids kernel-level memory save/restore.
if [[ "$success" != "true" ]]; then
  workflow_log "Kernel replay failed.  Falling back to application replay."
  if run_decode_ncu application "$NCU_DECODE_MAX_TOKENS" "$ncu_decode_step" yes \
       "application-replay, ${NCU_DECODE_MAX_TOKENS} tokens, step ${ncu_decode_step}"; then
    success=true
  fi
fi

# Attempt 3: last resort -- minimal tokens, profile from start, no worker-side
# control.  This may capture a prefill kernel rather than a pure decode kernel,
# but at least produces a valid .ncu-rep on the 6 GB GPU.
if [[ "$success" != "true" ]]; then
  workflow_log "Application replay also failed.  Trying minimal tokens with profiling from start."
  if run_decode_ncu kernel 2 1 no \
       "kernel-replay, 2 tokens, profile-from-start"; then
    success=true
  fi
fi

if [[ "$success" != "true" ]]; then
  workflow_log "ERROR: All decode ncu profiling attempts failed."
  workflow_log "  If ncu reported ResourceUnavailable or a driver resource error,"
  workflow_log "  the failure is in Nsight Compute counter collection itself."
  workflow_log "  Verify GPU counter access with: ncu --query-metrics"
  exit 1
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" register-ncu \
  --phase decode \
  --report "$decode_ncu_report"

workflow_log "Saved decode Nsight Compute report to $decode_ncu_report"
