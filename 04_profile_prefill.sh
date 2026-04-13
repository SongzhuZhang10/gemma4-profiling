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

function fail_prefill_ncu() {
  workflow_log "ERROR: $*"
  exit 1
}

function cleanup_prefill_ncu_state() {
  workflow_log "Cleaning stale Nsight Compute state for the prefill run."

  rm -f /tmp/nsight-compute-lock
  rm -f "$prefill_ncu_report"

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
for key in ("ncu_rep", "ncu_registered_at"):
    if key in prefill:
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
# step.  To keep replay time manageable on the RTX 3050 6 GB GPU, collect only
# a minimal metric set for a single matched kernel result instead of the full
# "basic" section bundle.
# ---------------------------------------------------------------------------
workflow_log "Profiling the prefill phase with Nsight Compute."

rm -f /tmp/nsight-compute-lock
ncu \
  --target-processes all \
  --replay-mode kernel \
  --launch-count 1 \
  --metrics \
  gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  --disable-extra-suffixes \
  --force-overwrite \
  -o "$prefill_ncu_stem" \
  "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" run-inference \
  --max-new-tokens 1 \
  --metadata-output "$prefill_metadata_json"

if [[ ! -f "$prefill_ncu_report" ]]; then
  fail_prefill_ncu "ncu completed without producing $prefill_ncu_report"
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" register-ncu \
  --phase prefill \
  --report "$prefill_ncu_report"

workflow_log "Saved prefill Nsight Compute report to $prefill_ncu_report"
