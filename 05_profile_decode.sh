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
readonly DECODE_NCU_MINIMAL_METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
readonly DECODE_TRT_PROBE_LAUNCH_COUNT=256
readonly DECODE_NCU_RESEARCH_SECTIONS=(
  ComputeWorkloadAnalysis
  InstructionStats
  LaunchStats
  MemoryWorkloadAnalysis
  Occupancy
  SchedulerStats
  SourceCounters
  SpeedOfLight
  SpeedOfLight_RooflineChart
  WarpStateStats
  WorkloadDistribution
)

NCU_COLLECTION_ARGS=()
DECODE_NCU_SELECTED_BACKEND=""
DECODE_NCU_SELECTED_REPLAY_MODE=""
DECODE_NCU_SELECTED_COLLECTION_PROFILE=""
DECODE_NCU_SELECTED_REQUESTED_MAX_NEW_TOKENS=""

# ---------------------------------------------------------------------------
# Cleanup stale ncu state from run_config.json before profiling.
# ---------------------------------------------------------------------------
function cleanup_decode_ncu_state() {
  workflow_log "Cleaning stale Nsight Compute state for the decode run."

  rm -f /tmp/nsight-compute-lock
  rm -f "$decode_ncu_report" "$decode_metadata_json"

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
for key in list(decode):
    if key.startswith("ncu_"):
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
#   run_decode_ncu <replay_mode> <collection_profile> <max_tokens> <label>
#
#   replay_mode        : "kernel" or "application"
#   collection_profile : "research" for the expanded section bundle,
#                        "minimal" for the original four-counter fallback
#   max_tokens         : --max-new-tokens value for the inference run
#   label              : human-readable attempt description for logs
#
# Profiler control (TLLM_PROFILE_START_STOP + --profile-from-start off) is
# intentionally omitted: on WSL2, that combination triggers ERR_NVGPUCTRPERM
# on _fused_gather_scatter_kernel for every attempt.  The default
# profile-from-start avoids the permission error and lets NCU collect all
# kernels in the run.  Enough output tokens are requested so that decode steps
# account for the statistical majority of all captured kernel launches.
#
# Returns 0 if decode_ncu_report was produced, 1 otherwise.
# ---------------------------------------------------------------------------
function build_decode_ncu_collection_args() {
  local collection_profile="$1"
  local section=""

  NCU_COLLECTION_ARGS=()

  case "$collection_profile" in
    research)
      for section in "${DECODE_NCU_RESEARCH_SECTIONS[@]}"; do
        NCU_COLLECTION_ARGS+=(--section "$section")
      done
      ;;
    minimal)
      NCU_COLLECTION_ARGS+=(--metrics "$DECODE_NCU_MINIMAL_METRICS")
      ;;
    *)
      workflow_log "ERROR: Unknown decode Nsight Compute collection profile: $collection_profile"
      return 1
      ;;
  esac
}

function run_decode_ncu() {
  local replay_mode="$1"
  local collection_profile="$2"
  local max_tokens="$3"
  local label="$4"
  local launch_count="${5:-}"

  workflow_log "NCU attempt ($label): replay=$replay_mode, collection=$collection_profile, tokens=$max_tokens"

  rm -f /tmp/nsight-compute-lock
  rm -f "$decode_ncu_report" "$decode_metadata_json"

  if ! build_decode_ncu_collection_args "$collection_profile"; then
    return 1
  fi

  # Keep clocks unlocked here and in prefill so phase-to-phase comparisons
  # reflect the same runtime clock policy.
  local ncu_cmd=(
    ncu
    --target-processes all
    --replay-mode "$replay_mode"
    --clock-control none
  )

  ncu_cmd+=("${NCU_COLLECTION_ARGS[@]}")
  if [[ -n "$launch_count" ]]; then
    ncu_cmd+=(--launch-count "$launch_count")
  fi
  ncu_cmd+=(
    --disable-extra-suffixes
    --force-overwrite
    -o "$decode_ncu_stem"
    "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference
    --max-new-tokens "$max_tokens"
    --metadata-output "$decode_metadata_json"
  )

  "${ncu_cmd[@]}" || true

  if [[ -f "$decode_ncu_report" ]]; then
    DECODE_NCU_SELECTED_BACKEND="trtllm_autodeploy"
    DECODE_NCU_SELECTED_REPLAY_MODE="$replay_mode"
    DECODE_NCU_SELECTED_COLLECTION_PROFILE="$collection_profile"
    DECODE_NCU_SELECTED_REQUESTED_MAX_NEW_TOKENS="$max_tokens"
    workflow_log "  -> Success: $decode_ncu_report produced."
    return 0
  else
    workflow_log "  -> Failed: $decode_ncu_report was not produced."
    return 1
  fi
}

readonly DECODE_VALID_GPU_TIME_NS=50000000

function validate_ncu_report() {
  local report="$1"
  local collection_backend="$2"
  local metadata_json="$3"
  local threshold="${4:-$DECODE_VALID_GPU_TIME_NS}"
  local parse_output
  local total_ns
  local is_valid
  local reason
  local family_counts

  parse_output=$(
    "$PYTHON_BIN" - "$report" "$collection_backend" "$metadata_json" "$threshold" <<'PY'
import json
from pathlib import Path
import sys

from gemma4_workflow import summarize_ncu_report

report_path = Path(sys.argv[1])
collection_backend = sys.argv[2]
metadata_path = Path(sys.argv[3])
threshold = int(sys.argv[4])

metadata = {}
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())

summary = summarize_ncu_report(report_path)
family_counts = {
    "attention_like": int(summary["family_counts"].get("attention_like", 0)),
    "copy_like": int(summary["family_counts"].get("copy_like", 0)),
    "gather_like": int(summary["family_counts"].get("gather_like", 0)),
    "gemm_like": int(summary["family_counts"].get("gemm_like", 0)),
    "other": int(summary["family_counts"].get("other", 0)),
    "sampling_like": int(summary["family_counts"].get("sampling_like", 0)),
}
total = int(summary["total_gpu_time_ns"])
inference_like = int(summary["inference_like_kernel_count"])
rng_copy_only = bool(summary["is_rng_copy_only"])
proxy_fallback = bool(metadata.get("proxy_fallback"))

if total < threshold:
    valid = False
    reason = f"only {int(total)} ns GPU time (< {threshold} ns)"
elif collection_backend == "trtllm_autodeploy":
    valid = inference_like > 0 and not rng_copy_only
    reason = (
        "TRT-LLM report contains only RNG/copy kernels"
        if not valid
        else "ok"
    )
elif collection_backend == "direct_hf":
    valid = (
        family_counts["gemm_like"] > 0
        if proxy_fallback
        else inference_like > 0
    ) and not rng_copy_only
    reason = (
        "direct fallback report is missing compute-relevant kernels"
        if not valid
        else "ok"
    )
else:
    valid = inference_like > 0 and not rng_copy_only
    reason = "report is missing inference-like kernels" if not valid else "ok"

print(int(total))
print(int(valid))
print(reason)
print(json.dumps(family_counts, sort_keys=True))
PY
  ) || parse_output=$'0\n0\nvalidation command failed\n{}'

  total_ns="$(sed -n '1p' <<<"$parse_output")"
  is_valid="$(sed -n '2p' <<<"$parse_output")"
  reason="$(sed -n '3p' <<<"$parse_output")"
  family_counts="$(sed -n '4p' <<<"$parse_output")"

  if [[ "${is_valid:-0}" == "1" ]]; then
    workflow_log "  -> Report validated: ${total_ns} ns GPU time captured. Families: $family_counts"
    return 0
  fi

  workflow_log "  -> Report invalid: $reason. Families: $family_counts"
  if [[ "$collection_backend" == "trtllm_autodeploy" ]]; then
    workflow_log "     TRT-LLM worker subprocess was not profiled representatively."
  fi
  return 1
}

function run_decode_ncu_direct() {
  local collection_profile="$1"
  local label="$2"

  workflow_log "NCU attempt ($label): replay=application, direct HF inference, tokens=64"

  rm -f /tmp/nsight-compute-lock
  rm -f "$decode_ncu_report" "$decode_metadata_json"

  if ! build_decode_ncu_collection_args "$collection_profile"; then
    return 1
  fi

  local ncu_cmd=(
    ncu
    --replay-mode application
    --clock-control none
    "${NCU_COLLECTION_ARGS[@]}"
    --disable-extra-suffixes
    --force-overwrite
    -o "$decode_ncu_stem"
    "$PYTHON_BIN" "$WORKFLOW_ROOT/ncu_direct_inference.py"
    --phase decode
    --max-new-tokens 64
    --metadata-output "$decode_metadata_json"
  )

  "${ncu_cmd[@]}" || true

  if [[ -f "$decode_ncu_report" ]]; then
    DECODE_NCU_SELECTED_BACKEND="direct_hf"
    DECODE_NCU_SELECTED_REPLAY_MODE="application"
    DECODE_NCU_SELECTED_COLLECTION_PROFILE="$collection_profile"
    DECODE_NCU_SELECTED_REQUESTED_MAX_NEW_TOKENS="64"
    workflow_log "  -> Success: $decode_ncu_report produced."
    return 0
  fi

  workflow_log "  -> Failed: $decode_ncu_report was not produced."
  return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
cleanup_decode_ncu_state

workflow_log "Profiling the decode phase with Nsight Compute."

# Profile a full inference run with enough output tokens so decode steps
# dominate total GPU execution time.  Based on nsys step timing
# (prefill ≈ 1813 ms, decode ≈ 74 ms/step), 64 output tokens yield roughly
# 72 % decode-dominated wall time and ~98 % of all kernel launches (64 decode
# forward passes vs. 1 prefill forward pass through the same layer stack).
# Profiler control is dropped; profile-from-start is used throughout.
NCU_DECODE_MAX_TOKENS=64

success=false
trt_probe_valid=false

# Probe TRT-LLM cheaply before attempting a full replay-heavy report. On this
# setup the invalid path is dominated by sampling/copy kernels, so a short
# application-replay probe is enough to decide whether a full TRT collection
# is worth attempting.
if run_decode_ncu application minimal "$NCU_DECODE_MAX_TOKENS" \
     "TRT-LLM representativeness probe, application-replay, launch-count ${DECODE_TRT_PROBE_LAUNCH_COUNT}, ${NCU_DECODE_MAX_TOKENS} tokens" \
     "$DECODE_TRT_PROBE_LAUNCH_COUNT"; then
  if validate_ncu_report "$decode_ncu_report" "trtllm_autodeploy" "$decode_metadata_json"; then
    trt_probe_valid=true
    workflow_log "TRT-LLM probe captured inference-like kernels; continuing to full collection."
  else
    workflow_log "TRT-LLM probe was not representative; skipping full TRT-LLM replay on this machine."
  fi
else
  workflow_log "TRT-LLM probe failed to produce a report; skipping full TRT-LLM replay."
fi

if [[ "$trt_probe_valid" == "true" ]]; then
  # Attempt 1: kernel replay with the expanded section bundle.
  if run_decode_ncu kernel research "$NCU_DECODE_MAX_TOKENS" \
       "kernel-replay, expanded bottleneck profile, ${NCU_DECODE_MAX_TOKENS} tokens"; then
    success=true
  fi

  # Attempt 2: application replay avoids kernel-level memory save/restore.
  if [[ "$success" != "true" ]]; then
    workflow_log "Expanded kernel replay failed. Falling back to application replay."
    if run_decode_ncu application research "$NCU_DECODE_MAX_TOKENS" \
         "application-replay, expanded bottleneck profile, ${NCU_DECODE_MAX_TOKENS} tokens"; then
      success=true
    fi
  fi

  # Attempt 3: research sections too heavy; try minimal metrics with kernel replay.
  if [[ "$success" != "true" ]]; then
    workflow_log "Expanded application replay also failed. Trying minimal kernel replay."
    if run_decode_ncu kernel minimal "$NCU_DECODE_MAX_TOKENS" \
         "kernel-replay, minimal metric quartet, ${NCU_DECODE_MAX_TOKENS} tokens"; then
      success=true
    fi
  fi

  # Attempt 4: application replay with minimal metrics.
  if [[ "$success" != "true" ]]; then
    workflow_log "Minimal kernel replay failed. Falling back to minimal application replay."
    if run_decode_ncu application minimal "$NCU_DECODE_MAX_TOKENS" \
         "application-replay, minimal metric quartet, ${NCU_DECODE_MAX_TOKENS} tokens"; then
      success=true
    fi
  fi

  # Attempt 5: reduce token count to relieve GPU memory pressure.
  if [[ "$success" != "true" ]]; then
    workflow_log "Minimal application replay failed. Reducing token count to 32."
    if run_decode_ncu kernel minimal 32 \
         "kernel-replay, minimal metric quartet, 32 tokens"; then
      success=true
    fi
  fi

  # Attempt 6: last resort — 32 tokens with application replay.
  if [[ "$success" != "true" ]]; then
    workflow_log "Last resort: application replay with 32 tokens."
    if run_decode_ncu application minimal 32 \
         "application-replay, minimal metric quartet, 32 tokens"; then
      success=true
    fi
  fi

  if [[ "$success" == "true" ]]; then
    if ! validate_ncu_report "$decode_ncu_report" "trtllm_autodeploy" "$decode_metadata_json"; then
      success=false
    fi
  fi
fi

if [[ "$success" != "true" ]]; then
  workflow_log "Falling back to direct HuggingFace inference (single process, no MPI worker)."
  if run_decode_ncu_direct minimal \
       "direct HF inference, application-replay, minimal metric quartet, 64 tokens"; then
    if validate_ncu_report "$decode_ncu_report" "direct_hf" "$decode_metadata_json"; then
      success=true
    fi
  fi
fi

if [[ "$success" != "true" ]]; then
  workflow_log "Minimal direct fallback failed. Retrying with expanded bottleneck sections."
  if run_decode_ncu_direct research \
       "direct HF inference, application-replay, expanded bottleneck profile, 64 tokens"; then
    if validate_ncu_report "$decode_ncu_report" "direct_hf" "$decode_metadata_json"; then
      success=true
    fi
  fi
fi

if [[ "$success" != "true" ]]; then
  workflow_log "ERROR: All decode ncu profiling attempts failed."
  if [[ -f "$decode_ncu_report" ]]; then
    workflow_log "  A decode report file was produced, but no profiling attempt passed validation."
    workflow_log "  Inspect the validation message above before discarding $decode_ncu_report."
  fi
  workflow_log "  If ncu reported ResourceUnavailable or a driver resource error,"
  workflow_log "  the failure is in Nsight Compute counter collection itself."
  workflow_log "  Verify GPU counter access with: ncu --query-metrics"
  exit 1
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" register-ncu \
  --phase decode \
  --report "$decode_ncu_report" \
  --metadata "$decode_metadata_json" \
  --collection-backend "$DECODE_NCU_SELECTED_BACKEND" \
  --replay-mode "$DECODE_NCU_SELECTED_REPLAY_MODE" \
  --collection-profile "$DECODE_NCU_SELECTED_COLLECTION_PROFILE" \
  --requested-max-new-tokens "$DECODE_NCU_SELECTED_REQUESTED_MAX_NEW_TOKENS"

workflow_log "Saved decode Nsight Compute report to $decode_ncu_report"
