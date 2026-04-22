#!/usr/bin/env bash

# Shared environment for the TensorRT Edge-LLM profiling workflow.
# This file is sourced by both the explicit host/target wrappers and the
# deprecated numbered compatibility entrypoints.

set -euo pipefail

readonly WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LEGACY_DL_ENV_ROOT="/home/songzhu/Desktop/dl_env"
readonly FALLBACK_DL_ENV_ROOT="$WORKFLOW_ROOT/.venv"

if [[ -d "$LEGACY_DL_ENV_ROOT" ]]; then
  readonly DL_ENV_ROOT="$LEGACY_DL_ENV_ROOT"
else
  readonly DL_ENV_ROOT="$FALLBACK_DL_ENV_ROOT"
fi

readonly PYTHON_BIN="$DL_ENV_ROOT/bin/python"
readonly PIP_BIN="$DL_ENV_ROOT/bin/pip"

readonly ARTIFACTS_DIR="$WORKFLOW_ROOT/artifacts"
readonly CACHE_DIR="$ARTIFACTS_DIR/cache"
readonly EXPORT_DIR="$ARTIFACTS_DIR/export"
readonly EXPORT_BUNDLE_DIR="$ARTIFACTS_DIR/export_bundle"
readonly RUNTIME_DIR="$ARTIFACTS_DIR/runtime"
readonly INPUTS_DIR="$RUNTIME_DIR/inputs"
readonly REPORTS_DIR="$ARTIFACTS_DIR/reports"
readonly RUN_CONFIG_JSON="$ARTIFACTS_DIR/run_config.json"

readonly EDGE_LLM_REPO_DIR="${TENSORRT_EDGE_LLM_ROOT:-${EDGE_LLM_REPO_DIR:-$HOME/TensorRT-Edge-LLM}}"

export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_DIR/hf/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"
export PYTHONPATH="$WORKFLOW_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -d "$DL_ENV_ROOT/bin" && ":$PATH:" != *":$DL_ENV_ROOT/bin:"* ]]; then
  export PATH="$DL_ENV_ROOT/bin:$PATH"
fi

for cuda_bin_dir in /usr/local/cuda/bin /usr/local/cuda-13/bin /usr/local/cuda-13.0/bin /usr/local/cuda-12.8/bin; do
  if [[ -d "$cuda_bin_dir" && ":$PATH:" != *":$cuda_bin_dir:"* ]]; then
    export PATH="$cuda_bin_dir:$PATH"
  fi
done

for cuda_lib_dir in /usr/local/cuda/lib64 /usr/local/cuda-13/lib64 /usr/local/cuda-13.0/lib64 /usr/local/cuda-12.8/lib64; do
  if [[ -d "$cuda_lib_dir" && ":${LD_LIBRARY_PATH:-}:" != *":$cuda_lib_dir:"* ]]; then
    export LD_LIBRARY_PATH="$cuda_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done

function workflow_log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

function python_env_is_python312() {
  local python_bin="${1:-$PYTHON_BIN}"

  "$python_bin" - <<'PY' >/dev/null 2>&1
import sys

raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)
PY
}

function env_has_required_binaries() {
  [[ -x "$DL_ENV_ROOT/bin/python" && -x "$DL_ENV_ROOT/bin/pip" ]]
}

function env_is_usable() {
  if ! env_has_required_binaries; then
    return 1
  fi

  python_env_is_python312 "$DL_ENV_ROOT/bin/python"
}

function ensure_selected_python_env() {
  if [[ "$DL_ENV_ROOT" == "$LEGACY_DL_ENV_ROOT" ]]; then
    if ! env_has_required_binaries; then
      workflow_log "Legacy Python environment detected at $DL_ENV_ROOT, but bin/python or bin/pip is missing."
      workflow_log "Please repair or remove that directory so the workflow can create and use $FALLBACK_DL_ENV_ROOT instead."
      exit 1
    fi

    if ! env_is_usable; then
      workflow_log "Legacy Python environment at $DL_ENV_ROOT is not a usable Python 3.12 virtual environment."
      workflow_log "Please repair or remove that directory so the workflow can create and use $FALLBACK_DL_ENV_ROOT instead."
      exit 1
    fi

    workflow_log "Using existing legacy Python environment at $DL_ENV_ROOT."
    return 0
  fi

  if env_is_usable; then
    workflow_log "Using existing portable Python environment at $DL_ENV_ROOT."
    return 0
  fi

  if ! command -v python3.12 >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    workflow_log "python3.12 was not found in PATH."
    workflow_log "Install Python 3.12 on this machine first, then rerun the install wrapper."
    exit 1
  fi

  local python312_bin
  python312_bin="$(command -v python3.12 2>/dev/null || command -v python3)"

  if ! python_env_is_python312 "$python312_bin"; then
    workflow_log "The python3 command does not report Python 3.12 as expected (found: $("$python312_bin" --version 2>&1))."
    exit 1
  fi

  workflow_log "Creating or repairing the portable Python 3.12 virtual environment at $DL_ENV_ROOT."
  mkdir -p "$(dirname "$DL_ENV_ROOT")"
  "$python312_bin" -m venv --without-pip "$DL_ENV_ROOT"
  curl -sS https://bootstrap.pypa.io/get-pip.py | "$DL_ENV_ROOT/bin/python3"

  if ! env_is_usable; then
    workflow_log "Python 3.12 virtual environment creation completed, but $DL_ENV_ROOT is still not usable."
    exit 1
  fi
}

function ensure_vscode_settings() {
  local interpreter_path="$PYTHON_BIN"

  if [[ "$DL_ENV_ROOT" == "$FALLBACK_DL_ENV_ROOT" ]]; then
    interpreter_path='${workspaceFolder}/.venv/bin/python'
  fi

  mkdir -p "$WORKFLOW_ROOT/.vscode"

  VSCODE_SETTINGS_PATH="$WORKFLOW_ROOT/.vscode/settings.json" \
  VSCODE_INTERPRETER_PATH="$interpreter_path" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os
import sys

settings_path = Path(os.environ["VSCODE_SETTINGS_PATH"])
interpreter_path = os.environ["VSCODE_INTERPRETER_PATH"]

if settings_path.exists():
    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"{settings_path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(1)
else:
    settings = {}

if not isinstance(settings, dict):
    print(f"{settings_path} must contain a JSON object.", file=sys.stderr)
    raise SystemExit(1)

settings["python.defaultInterpreterPath"] = interpreter_path
settings_path.write_text(json.dumps(settings, indent=4) + "\n")
PY

  workflow_log "Updated VS Code interpreter setting at $WORKFLOW_ROOT/.vscode/settings.json."
}

function require_python_env() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    workflow_log "Expected Python interpreter not found at $PYTHON_BIN"
    exit 1
  fi

  if [[ ! -x "$PIP_BIN" ]]; then
    workflow_log "Expected pip executable not found at $PIP_BIN"
    exit 1
  fi

  if ! python_env_is_python312 "$PYTHON_BIN"; then
    workflow_log "Expected Python 3.12 interpreter at $PYTHON_BIN"
    exit 1
  fi
}

function ensure_workflow_dirs() {
  mkdir -p \
    "$ARTIFACTS_DIR" \
    "$CACHE_DIR" \
    "$EXPORT_DIR" \
    "$EXPORT_BUNDLE_DIR" \
    "$RUNTIME_DIR" \
    "$INPUTS_DIR" \
    "$REPORTS_DIR"
  mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME"
}
