#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# Nsight Systems validates the phase marker first so the workflow confirms a
# real prefill step before Nsight Compute runs.
bash "$WORKFLOW_ROOT/nsys_prefill.sh"

prefill_ncu_stem="$REPORTS_DIR/prefill"
prefill_ncu_report="$REPORTS_DIR/prefill.ncu-rep"
prefill_metadata_json="$REPORTS_DIR/prefill_ncu_run_metadata.json"
readonly PREFILL_NCU_MINIMAL_METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
readonly PREFILL_TRT_PROBE_LAUNCH_COUNT=192
readonly PREFILL_NCU_RESEARCH_SECTIONS=(
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
PREFILL_NCU_SELECTED_BACKEND=""
PREFILL_NCU_SELECTED_REPLAY_MODE=""
PREFILL_NCU_SELECTED_COLLECTION_PROFILE=""
PREFILL_NCU_SELECTED_REQUESTED_MAX_NEW_TOKENS=""

function fail_prefill_ncu() {
  workflow_log "ERROR: $*"
  exit 1
}

function cleanup_prefill_ncu_state() {
  workflow_log "Cleaning stale Nsight Compute state for the prefill run."

  rm -f /tmp/nsight-compute-lock
  rm -f "$prefill_ncu_report" "$prefill_metadata_json"

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

prefill = profiles.get("prefill")
if not isinstance(prefill, dict):
    raise SystemExit(0)

changed = False
for key in list(prefill):
    if key.startswith("ncu_"):
        prefill.pop(key, None)
        changed = True

if not prefill:
    profiles.pop("prefill", None)
    changed = True

if changed:
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

function probe_windows_interop() {
  local output=""

  if ! command -v powershell.exe &>/dev/null; then
    return 1
  fi

  if output="$(powershell.exe -NoProfile -Command "[Console]::Out.Write('ok')" 2>&1)"; then
    return 0
  fi

  workflow_log "WARNING: powershell.exe is present but Windows interop is not runnable from this shell."
  while IFS= read -r line; do
    [[ -n "$line" ]] && workflow_log "  $line"
  done <<<"$output"
  return 1
}

function query_ncu_metrics() {
  local output=""

  rm -f /tmp/nsight-compute-lock
  if output="$(ncu --query-metrics 2>&1)"; then
    printf '%s\n' "$output"
    return 0
  fi

  printf '%s\n' "$output"
  return 1
}

function enable_gpu_counters_via_registry() {
  workflow_log "Attempting to enable GPU performance counters via Windows registry."
  powershell.exe -NoProfile -Command \
    'Start-Process reg -Verb RunAs -Wait -ArgumentList "add `"HKLM\SOFTWARE\NVIDIA Corporation\Global\NVTweak`" /v NVreg_RestrictProfilingToAdminUsers /t REG_DWORD /d 0 /f"' \
    2>/dev/null || true
}

function build_prefill_ncu_collection_args() {
  local collection_profile="$1"
  local section=""

  NCU_COLLECTION_ARGS=()

  case "$collection_profile" in
    research)
      for section in "${PREFILL_NCU_RESEARCH_SECTIONS[@]}"; do
        NCU_COLLECTION_ARGS+=(--section "$section")
      done
      ;;
    minimal)
      NCU_COLLECTION_ARGS+=(--metrics "$PREFILL_NCU_MINIMAL_METRICS")
      ;;
    *)
      fail_prefill_ncu "Unknown prefill Nsight Compute collection profile: $collection_profile"
      ;;
  esac
}

function run_prefill_ncu() {
  local replay_mode="$1"
  local collection_profile="$2"
  local label="$3"
  local launch_count="${4:-}"

  workflow_log "NCU attempt ($label): replay=$replay_mode, collection=$collection_profile"

  rm -f /tmp/nsight-compute-lock
  rm -f "$prefill_ncu_report" "$prefill_metadata_json"

  build_prefill_ncu_collection_args "$collection_profile"

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
    -o "$prefill_ncu_stem"
    "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference
    --max-new-tokens 1
    --metadata-output "$prefill_metadata_json"
  )

  "${ncu_cmd[@]}" || true

  if [[ -f "$prefill_ncu_report" ]]; then
    PREFILL_NCU_SELECTED_BACKEND="trtllm_autodeploy"
    PREFILL_NCU_SELECTED_REPLAY_MODE="$replay_mode"
    PREFILL_NCU_SELECTED_COLLECTION_PROFILE="$collection_profile"
    PREFILL_NCU_SELECTED_REQUESTED_MAX_NEW_TOKENS="1"
    workflow_log "  -> Success: $prefill_ncu_report produced."
    return 0
  fi

  workflow_log "  -> Failed: $prefill_ncu_report was not produced."
  return 1
}

# Returns 0 if the report captured at least MIN_GPU_TIME_NS of GPU work.
# A report under this threshold contains only process-init kernels, not LLM
# inference kernels (TRT-LLM worker subprocess was not profiled).
readonly PREFILL_VALID_GPU_TIME_NS=50000000

function validate_ncu_report() {
  local report="$1"
  local collection_backend="$2"
  local metadata_json="$3"
  local threshold="${4:-$PREFILL_VALID_GPU_TIME_NS}"
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

function run_prefill_ncu_direct() {
  local collection_profile="$1"
  local label="$2"

  workflow_log "NCU attempt ($label): replay=application, direct HF inference, T=512"

  rm -f /tmp/nsight-compute-lock
  rm -f "$prefill_ncu_report" "$prefill_metadata_json"

  build_prefill_ncu_collection_args "$collection_profile"

  local ncu_cmd=(
    ncu
    --replay-mode application
    --clock-control none
    "${NCU_COLLECTION_ARGS[@]}"
    --disable-extra-suffixes
    --force-overwrite
    -o "$prefill_ncu_stem"
    "$PYTHON_BIN" "$WORKFLOW_ROOT/ncu_direct_inference.py"
    --phase prefill
    --metadata-output "$prefill_metadata_json"
  )

  "${ncu_cmd[@]}" || true

  if [[ -f "$prefill_ncu_report" ]]; then
    PREFILL_NCU_SELECTED_BACKEND="direct_hf"
    PREFILL_NCU_SELECTED_REPLAY_MODE="application"
    PREFILL_NCU_SELECTED_COLLECTION_PROFILE="$collection_profile"
    PREFILL_NCU_SELECTED_REQUESTED_MAX_NEW_TOKENS="0"
    workflow_log "  -> Success: $prefill_ncu_report produced."
    return 0
  fi

  workflow_log "  -> Failed: $prefill_ncu_report was not produced."
  return 1
}

# ---------------------------------------------------------------------------
# GPU performance-counter access check (WSL2-specific).
#
# On WSL2, GPU counters are gated by the Windows-side NVIDIA driver.  The
# standard Linux fix (modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0)
# does not apply; instead we must set the equivalent Windows registry value
# and reload the driver (typically via a Windows reboot or WSL restart).
#
# NVIDIA's official guidance for ERR_NVGPUCTRPERM on Windows/WDDM:
#   1. Open NVIDIA Control Panel as Administrator.
#   2. Turn on Desktop -> Enable Developer Settings.
#   3. Go to Developer -> Manage GPU Performance Counters.
#   4. Select "Allow access to the GPU performance counter to all users".
#   5. Reboot Windows again.
#   6. In WSL, rerun: ncu --query-metrics | sed -n '1,20p'
#   7. If the error is gone, rerun: bash 04_profile_prefill.sh
#
# Reference: https://developer.nvidia.com/ERR_NVGPUCTRPERM
# ---------------------------------------------------------------------------
cleanup_prefill_ncu_state

WINDOWS_INTEROP_OK=false
if probe_windows_interop; then
  WINDOWS_INTEROP_OK=true
fi

ncu_query_output=""
if ncu_query_output="$(query_ncu_metrics)"; then
  ncu_query_status=0
else
  ncu_query_status=$?
fi

if grep -q "ERR_NVGPUCTRPERM" <<<"$ncu_query_output"; then
  if [[ "$WINDOWS_INTEROP_OK" == "true" ]]; then
    enable_gpu_counters_via_registry
    if ncu_query_output="$(query_ncu_metrics)"; then
      ncu_query_status=0
    else
      ncu_query_status=$?
    fi
  else
    workflow_log "WARNING: GPU performance counters are blocked (ERR_NVGPUCTRPERM)."
    workflow_log "  Cannot attempt the Windows registry self-heal because Windows interop"
    workflow_log "  commands are not runnable from this shell."
  fi

  if grep -q "ERR_NVGPUCTRPERM" <<<"$ncu_query_output"; then
    workflow_log "WARNING: GPU performance counters are still blocked."
    workflow_log "  Run 'wsl --shutdown' from a Windows command prompt, relaunch WSL,"
    workflow_log "  and retry this script. A full Windows reboot also works."
    fail_prefill_ncu "Nsight Compute profiling requires NVIDIA GPU performance counter access."
  fi
elif [[ "${ncu_query_status:-0}" -ne 0 ]]; then
  workflow_log "ncu --query-metrics failed unexpectedly:"
  while IFS= read -r line; do
    [[ -n "$line" ]] && workflow_log "  $line"
  done <<<"$ncu_query_output"
  fail_prefill_ncu "Unable to verify NVIDIA Nsight Compute metric access."
elif [[ -z "$ncu_query_output" ]]; then
  fail_prefill_ncu "ncu --query-metrics returned no output."
elif ! grep -q "^Device " <<<"$ncu_query_output"; then
  workflow_log "Unexpected ncu --query-metrics output:"
  while IFS= read -r line; do
    [[ -n "$line" ]] && workflow_log "  $line"
  done <<<"$ncu_query_output"
  fail_prefill_ncu "Unable to interpret NVIDIA Nsight Compute metric availability."
fi

# ---------------------------------------------------------------------------
# TensorRT-LLM's executor emits NVTX start/end ranges (not push/pop) and
# launches kernels from internal threads, so ncu --nvtx-include cannot isolate
# individual forward steps.  Instead we profile the entire inference with only
# 1 generated token so that the captured kernels are dominated by the prefill
# step.  Start with a research-oriented section bundle that adds launch stats,
# occupancy, cache/memory analysis, instruction mix, scheduler + warp stall
# breakdowns, source counters, and roofline views.  Use the same unlocked
# clock policy as decode so cross-phase timing and throughput comparisons stay
# fair.  If the richer collection proves too heavy for the RTX 3050 6 GB GPU,
# fall back in order: application replay with the same research sections, then
# application replay with the minimal quartet, then kernel replay with the
# minimal quartet. Reports that capture only init kernels are treated as
# invalid and retried through a single-process direct HuggingFace fallback.
# ---------------------------------------------------------------------------
workflow_log "Profiling the prefill phase with Nsight Compute."

success=false
trt_probe_valid=false

# The TRT-LLM AutoDeploy path has repeatedly produced long-running replay
# sessions dominated by sampling/copy kernels on this setup. Probe that path
# cheaply first; only pay for a full report if the probe captures
# inference-relevant compute kernels.
if run_prefill_ncu application minimal \
     "TRT-LLM representativeness probe, application-replay, launch-count ${PREFILL_TRT_PROBE_LAUNCH_COUNT}" \
     "$PREFILL_TRT_PROBE_LAUNCH_COUNT"; then
  if validate_ncu_report "$prefill_ncu_report" "trtllm_autodeploy" "$prefill_metadata_json"; then
    trt_probe_valid=true
    workflow_log "TRT-LLM probe captured inference-like kernels; continuing to full collection."
  else
    workflow_log "TRT-LLM probe was not representative; skipping full TRT-LLM replay on this machine."
  fi
else
  workflow_log "TRT-LLM probe failed to produce a report; skipping full TRT-LLM replay."
fi

if [[ "$trt_probe_valid" == "true" ]]; then
  if run_prefill_ncu kernel research "kernel-replay, expanded bottleneck profile"; then
    success=true
  fi

  if [[ "$success" != "true" ]]; then
    workflow_log "Expanded prefill collection failed. Falling back to application replay."
    if run_prefill_ncu application research \
         "application-replay, expanded bottleneck profile"; then
      success=true
    fi
  fi

  if [[ "$success" != "true" ]]; then
    workflow_log "Expanded application replay also failed. Trying minimal application replay."
    if run_prefill_ncu application minimal \
         "application-replay, minimal metric quartet"; then
      success=true
    fi
  fi

  if [[ "$success" != "true" ]]; then
    workflow_log "Minimal application replay failed. Falling back to minimal kernel replay."
    if run_prefill_ncu kernel minimal "kernel-replay, minimal metric quartet"; then
      success=true
    fi
  fi

  if [[ "$success" == "true" ]]; then
    if ! validate_ncu_report "$prefill_ncu_report" "trtllm_autodeploy" "$prefill_metadata_json"; then
      success=false
    fi
  fi
fi

if [[ "$success" != "true" ]]; then
  workflow_log "Falling back to direct HuggingFace inference (single process, no MPI worker)."
  if run_prefill_ncu_direct minimal \
       "direct HF inference, application-replay, minimal metric quartet, T=512"; then
    if validate_ncu_report "$prefill_ncu_report" "direct_hf" "$prefill_metadata_json"; then
      success=true
    fi
  fi
fi

if [[ "$success" != "true" ]]; then
  workflow_log "Minimal direct fallback failed. Retrying with expanded bottleneck sections."
  if run_prefill_ncu_direct research \
       "direct HF inference, application-replay, expanded bottleneck profile, T=512"; then
    if validate_ncu_report "$prefill_ncu_report" "direct_hf" "$prefill_metadata_json"; then
      success=true
    fi
  fi
fi

if [[ "$success" != "true" ]]; then
  if [[ -f "$prefill_ncu_report" ]]; then
    fail_prefill_ncu \
      "ncu produced $prefill_ncu_report, but no profiling attempt passed validation. Inspect the validation message above."
  fi
  fail_prefill_ncu "ncu completed without producing $prefill_ncu_report"
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" register-ncu \
  --phase prefill \
  --report "$prefill_ncu_report" \
  --metadata "$prefill_metadata_json" \
  --collection-backend "$PREFILL_NCU_SELECTED_BACKEND" \
  --replay-mode "$PREFILL_NCU_SELECTED_REPLAY_MODE" \
  --collection-profile "$PREFILL_NCU_SELECTED_COLLECTION_PROFILE" \
  --requested-max-new-tokens "$PREFILL_NCU_SELECTED_REQUESTED_MAX_NEW_TOKENS"

workflow_log "Saved prefill Nsight Compute report to $prefill_ncu_report"
