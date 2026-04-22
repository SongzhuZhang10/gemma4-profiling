#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

ensure_workflow_dirs

if [[ $# -ne 1 ]]; then
  workflow_log "Usage: bash 12_target_open_report.sh <report-path>"
  workflow_log "Examples:"
  workflow_log "  bash 12_target_open_report.sh artifacts/reports/prefill.ncu-rep"
  workflow_log "  bash 12_target_open_report.sh artifacts/reports/decode.nsys-rep"
  exit 1
fi

report_path="$1"

if [[ ! -f "$report_path" ]]; then
  workflow_log "Report not found: $report_path"
  exit 1
fi

function find_windows_ncu_ui() {
  if ! command -v powershell.exe >/dev/null 2>&1; then
    return 1
  fi

  local windows_path
  windows_path="$(
    powershell.exe -NoProfile -Command \
      '[Console]::Out.Write((Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Recurse -Filter ncu-ui.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName))' \
      | tr -d '\r'
  )"

  [[ -n "$windows_path" ]] || return 1
  printf '%s\n' "$windows_path"
}

viewer=""
report_kind=""
case "$report_path" in
  *.ncu-rep)
    viewer="ncu-ui"
    report_kind="Nsight Compute"
    ;;
  *.nsys-rep)
    viewer="nsys-ui"
    report_kind="Nsight Systems"
    ;;
  *)
    workflow_log "Unsupported report type: $report_path"
    workflow_log "Expected a .ncu-rep or .nsys-rep file."
    exit 1
    ;;
esac

if ! command -v "$viewer" >/dev/null 2>&1; then
  workflow_log "Required viewer not found in PATH: $viewer"
  exit 1
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  workflow_log "No GUI display detected."
  workflow_log "Run this from a WSLg session or start an X server, then retry."
  exit 1
fi

viewer_args=()
viewer_env=()
if [[ "$viewer" == "ncu-ui" ]]; then
  if [[ -n "${WSL_DISTRO_NAME:-}" ]] && command -v wslpath >/dev/null 2>&1; then
    windows_ncu_ui="$(find_windows_ncu_ui || true)"
    if [[ -n "${windows_ncu_ui:-}" ]]; then
      windows_report_path="$(wslpath -w "$(realpath "$report_path")")"
      staged_report_base="$(basename "$report_path" .ncu-rep)"
      workflow_log "Staging Nsight Compute report into Windows temp storage."
      workflow_log "Opening Nsight Compute report with Windows ncu-ui to avoid WSL memory limits."
      exec powershell.exe -NoProfile -Command \
        "\$src = '$windows_report_path'; \
         \$exe = '$windows_ncu_ui'; \
         \$stagingDir = Join-Path \$env:TEMP 'codex-ncu'; \
         New-Item -ItemType Directory -Force -Path \$stagingDir | Out-Null; \
         \$srcInfo = Get-Item -LiteralPath \$src; \
         \$stamp = [DateTimeOffset]::new(\$srcInfo.LastWriteTimeUtc).ToUnixTimeSeconds(); \
         \$dst = Join-Path \$stagingDir ('$staged_report_base' + '-' + \$srcInfo.Length + '-' + \$stamp + '.ncu-rep'); \
         \$needsCopy = !(Test-Path \$dst); \
         if (-not \$needsCopy) { \
           \$dstInfo = Get-Item -LiteralPath \$dst; \
           \$needsCopy = (\$dstInfo.Length -ne \$srcInfo.Length); \
         }; \
         if (\$needsCopy -and (Test-Path \$dst)) { Remove-Item -LiteralPath \$dst -Force; }; \
         if (\$needsCopy) { Copy-Item -LiteralPath \$src -Destination \$dst -Force; }; \
         Start-Process -FilePath \$exe -ArgumentList @(\$dst)"
    fi
  fi

  viewer_args+=(--shared-instance off --no-splash)

  if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    viewer_env+=(QT_QPA_PLATFORM=xcb)
  fi

  report_size_bytes="$(stat -c '%s' "$report_path")"
  if (( report_size_bytes >= 1000000000 )); then
    workflow_log "Large Nsight Compute report detected ($(numfmt --to=iec "$report_size_bytes"))."
    workflow_log "The GUI may take around a minute to become responsive while it parses the report."
  fi
fi

workflow_log "Opening $report_kind report with $viewer: $report_path"
if [[ ${#viewer_env[@]} -gt 0 ]]; then
  exec env "${viewer_env[@]}" "$viewer" "${viewer_args[@]}" "$report_path"
fi
exec "$viewer" "${viewer_args[@]}" "$report_path"
