#!/usr/bin/env bash

# Shared environment for the Gemma 4 WSL2 profiling workflow.
# This file is sourced by the numbered entrypoint scripts.

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
readonly RUNTIME_DIR="$ARTIFACTS_DIR/runtime"
readonly REPORTS_DIR="$ARTIFACTS_DIR/reports"
readonly RUN_CONFIG_JSON="$ARTIFACTS_DIR/run_config.json"

# Keep all Python-side caches inside the repo-local artifacts tree so the run is
# reproducible and we avoid depending on per-user global cache locations.
export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_DIR/hf/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_DIR/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_DIR/triton}"
export TLLM_LLMAPI_BUILD_CACHE=1
export TLLM_LLMAPI_BUILD_CACHE_ROOT="${TLLM_LLMAPI_BUILD_CACHE_ROOT:-$CACHE_DIR/tllm_build_cache}"

# Suppress modelopt.torch's UserWarning about transformers >= 4.57.
# TRT-LLM 1.2.0 pins transformers==4.57.3 and nvidia-modelopt~=0.37.0, but
# modelopt 0.37.0 only accepts transformers < 4.57. The two constraints are
# irreconcilable within TRT-LLM 1.2.0; the warning is noise, not an actionable
# error for this workflow.
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}ignore::UserWarning:modelopt.torch"

# WSL2 CUDA user-space tools are often installed under one of these prefixes,
# but they are not always exported into PATH by default shell startup files.
for cuda_bin_dir in /usr/local/cuda/bin /usr/local/cuda-12.8/bin; do
  if [[ -d "$cuda_bin_dir" && ":$PATH:" != *":$cuda_bin_dir:"* ]]; then
    export PATH="$cuda_bin_dir:$PATH"
  fi
done

# TensorRT-LLM in this workflow relies on native libraries installed into the
# shared dl_env prefix. Export these paths centrally so every numbered script
# inherits the same runtime view after 01_install.sh provisions them.
if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_SITE_PACKAGES="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
else
  PYTHON_SITE_PACKAGES="$DL_ENV_ROOT/lib/python3.10/site-packages"
fi

if [[ -d "$DL_ENV_ROOT/share/openmpi" ]]; then
  export OPAL_PREFIX="${OPAL_PREFIX:-$DL_ENV_ROOT}"
fi

for native_dir in \
  "$DL_ENV_ROOT/bin" \
  "$DL_ENV_ROOT/lib" \
  "$DL_ENV_ROOT/lib/openmpi" \
  "$PYTHON_SITE_PACKAGES/nvidia/cu13/lib" \
  "$PYTHON_SITE_PACKAGES/nvidia/nccl/lib" \
  "$PYTHON_SITE_PACKAGES/tensorrt_libs"; do
  if [[ -d "$native_dir" ]]; then
    if [[ "$native_dir" == *"/bin" ]]; then
      if [[ ":$PATH:" != *":$native_dir:"* ]]; then
        export PATH="$native_dir:$PATH"
      fi
    else
      if [[ ":${LD_LIBRARY_PATH:-}:" != *":$native_dir:"* ]]; then
        export LD_LIBRARY_PATH="$native_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      fi
    fi
  fi
done

export PYTHONPATH="$WORKFLOW_ROOT${PYTHONPATH:+:$PYTHONPATH}"

function workflow_log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

function python_env_is_python310() {
  local python_bin="${1:-$PYTHON_BIN}"

  "$python_bin" - <<'PY' >/dev/null 2>&1
import sys

raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
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

  if ! python_env_is_python310 "$PYTHON_BIN"; then
    workflow_log "Expected Python 3.10 interpreter at $PYTHON_BIN"
    exit 1
  fi
}

function ensure_workflow_dirs() {
  mkdir -p "$ARTIFACTS_DIR" "$CACHE_DIR" "$RUNTIME_DIR" "$REPORTS_DIR"
  mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"
  mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  mkdir -p "$TLLM_LLMAPI_BUILD_CACHE_ROOT"
}
