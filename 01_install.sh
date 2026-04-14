#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

function env_has_required_binaries() {
  [[ -x "$DL_ENV_ROOT/bin/python" && -x "$DL_ENV_ROOT/bin/pip" ]]
}

function env_is_usable() {
  if ! env_has_required_binaries; then
    return 1
  fi

  python_env_is_python310 "$DL_ENV_ROOT/bin/python"
}

function ensure_selected_python_env() {
  if [[ "$DL_ENV_ROOT" == "$LEGACY_DL_ENV_ROOT" ]]; then
    if ! env_has_required_binaries; then
      workflow_log "Legacy Python environment detected at $DL_ENV_ROOT, but bin/python or bin/pip is missing."
      workflow_log "Please repair or remove that directory so the workflow can create and use $FALLBACK_DL_ENV_ROOT instead."
      exit 1
    fi

    if ! env_is_usable; then
      workflow_log "Legacy Python environment at $DL_ENV_ROOT is not a usable Python 3.10 virtual environment."
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

  if ! command -v python3.10 >/dev/null 2>&1; then
    workflow_log "python3.10 was not found in PATH."
    workflow_log "Install Python 3.10 on this machine first, then rerun 01_install.sh."
    exit 1
  fi

  if ! python_env_is_python310 "$(command -v python3.10)"; then
    workflow_log "The python3.10 command does not report Python 3.10 as expected."
    exit 1
  fi

  workflow_log "Creating or repairing the portable Python 3.10 virtual environment at $DL_ENV_ROOT."
  mkdir -p "$(dirname "$DL_ENV_ROOT")"
  python3.10 -m venv "$DL_ENV_ROOT"

  if ! env_is_usable; then
    workflow_log "Python 3.10 virtual environment creation completed, but $DL_ENV_ROOT is still not usable."
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

function ensure_pinned_python_stack() {
  workflow_log "Installing or updating packaging tools inside $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade pip 'setuptools<80' wheel

  workflow_log "Installing pinned PyTorch packages for CUDA 12.8 inside $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    torch==2.9.1+cu128 \
    torchvision==0.24.1+cu128

  workflow_log "Installing pinned workflow packages inside $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade \
    tensorrt-llm==1.2.0 \
    transformers==4.57.3 \
    PyYAML==6.0.3 \
    sentencepiece==0.2.1 \
    Pillow==12.1.1 \
    accelerate==1.13.0 \
    huggingface_hub==0.36.2
}

ensure_selected_python_env
require_python_env
ensure_workflow_dirs
ensure_vscode_settings
ensure_pinned_python_stack

site_packages_dir="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

function has_local_openmpi_runtime() {
  [[ -f "$DL_ENV_ROOT/lib/libmpi.so.40" && -d "$DL_ENV_ROOT/share/openmpi" ]]
}

function has_cuda13_cublas_runtime() {
  [[ -f "$site_packages_dir/nvidia/cu13/lib/libcublasLt.so.13" && -f "$site_packages_dir/nvidia/cu13/lib/libcublas.so.13" ]]
}

function refresh_native_runtime_env() {
  export PATH="$DL_ENV_ROOT/bin:$PATH"
  export OPAL_PREFIX="${OPAL_PREFIX:-$DL_ENV_ROOT}"

  for native_dir in \
    "$DL_ENV_ROOT/lib" \
    "$DL_ENV_ROOT/lib/openmpi" \
    "$site_packages_dir/nvidia/cu13/lib" \
    "$site_packages_dir/nvidia/nccl/lib" \
    "$site_packages_dir/tensorrt_libs"; do
    if [[ -d "$native_dir" && ":${LD_LIBRARY_PATH:-}:" != *":$native_dir:"* ]]; then
      export LD_LIBRARY_PATH="$native_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  done
}

function ensure_openmpi_runtime() {
  if has_local_openmpi_runtime; then
    refresh_native_runtime_env
    return 0
  fi

  workflow_log "Installing user-space OpenMPI runtime into $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade openmpi==4.1.8
  refresh_native_runtime_env

  if ! has_local_openmpi_runtime; then
    workflow_log "OpenMPI installation finished, but the expected local runtime files were not created."
    exit 1
  fi
}

function ensure_cuda13_cublas_runtime() {
  if has_cuda13_cublas_runtime; then
    refresh_native_runtime_env
    return 0
  fi

  workflow_log "Installing CUDA 13 CUBLAS runtime into $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade nvidia-cublas==13.3.0.5
  refresh_native_runtime_env

  if ! has_cuda13_cublas_runtime; then
    workflow_log "CUDA 13 CUBLAS installation finished, but libcublas.so.13 was not found."
    exit 1
  fi
}

function try_import_tensorrt_llm() {
  "$PYTHON_BIN" - <<'PY'
import tensorrt_llm
PY
}

function tensorrt_llm_has_gemma4_autodeploy() {
  "$PYTHON_BIN" - <<'PY'
from tensorrt_llm._torch.auto_deploy import LLM as _  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.custom import Gemma4ForConditionalGeneration  # noqa: F401
PY
}

function ensure_tensorrt_llm_importable() {
  local import_error_log
  import_error_log="$(mktemp)"

  refresh_native_runtime_env

  if try_import_tensorrt_llm >/dev/null 2>"$import_error_log"; then
    rm -f "$import_error_log"
    return 0
  fi

  for attempt in 1 2 3; do
    if grep -Eq 'libmpi\.so\.40|cannot load MPI library|orte_init:startup:internal-failure|mpi_init:startup:internal-failure|/opt/openmpi/share/openmpi' "$import_error_log"; then
      workflow_log "TensorRT-LLM is installed, but its native bindings need the OpenMPI runtime configured."
      ensure_openmpi_runtime
    elif grep -Eq 'libcublasLt\.so\.13|libcublas\.so\.13' "$import_error_log"; then
      workflow_log "TensorRT-LLM is installed, but its native bindings need CUDA 13 CUBLAS libraries."
      ensure_cuda13_cublas_runtime
    else
      break
    fi

    if try_import_tensorrt_llm >/dev/null 2>"$import_error_log"; then
      rm -f "$import_error_log"
      return 0
    fi
  done

  cat "$import_error_log" >&2
  rm -f "$import_error_log"
  workflow_log "TensorRT-LLM is installed but not importable in $DL_ENV_ROOT."
  exit 1
}

function ensure_gemma4_autodeploy_support() {
  local support_error_log
  support_error_log="$(mktemp)"

  refresh_native_runtime_env

  if tensorrt_llm_has_gemma4_autodeploy >/dev/null 2>"$support_error_log"; then
    rm -f "$support_error_log"
    return 0
  fi

  workflow_log "The installed TensorRT-LLM build lacks Gemma 4 AutoDeploy support."
  workflow_log "Applying the upstream Gemma 4 AutoDeploy Python patch into the installed package."
  GEMMA4_PATCH_URL="https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/26a28ea4f045bb5e6c5da1ce8fd7051e99a66eb2/tensorrt_llm/_torch/auto_deploy/models/custom/modeling_gemma4.py" \
  CUSTOM_MODELS_DIR="$site_packages_dir/tensorrt_llm/_torch/auto_deploy/models/custom" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import os

custom_dir = Path(os.environ["CUSTOM_MODELS_DIR"])
custom_dir.mkdir(parents=True, exist_ok=True)

modeling_path = custom_dir / "modeling_gemma4.py"
init_path = custom_dir / "__init__.py"
patch_url = os.environ["GEMMA4_PATCH_URL"]

modeling_path.write_text(urlopen(patch_url).read().decode("utf-8"))

init_text = init_path.read_text() if init_path.exists() else ""
import_line = "from .modeling_gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration\n"
if import_line not in init_text:
    init_text = import_line + init_text

if "__all__ = (" not in init_text:
    init_text += "\n__all__ = (\n    \"Gemma4ForCausalLM\",\n    \"Gemma4ForConditionalGeneration\",\n)\n"
else:
    if '"Gemma4ForCausalLM",' not in init_text:
        init_text = init_text.replace(
            "__all__ = (\n",
            "__all__ = (\n    \"Gemma4ForCausalLM\",\n    \"Gemma4ForConditionalGeneration\",\n",
        )

init_path.write_text(init_text)
PY

  refresh_native_runtime_env
  ensure_tensorrt_llm_importable

  if ! tensorrt_llm_has_gemma4_autodeploy >/dev/null 2>"$support_error_log"; then
    cat "$support_error_log" >&2
    rm -f "$support_error_log"
    workflow_log "TensorRT-LLM still does not expose Gemma 4 AutoDeploy support after the source upgrade."
    exit 1
  fi

  rm -f "$support_error_log"
}

workflow_log "Verifying WSL2 CUDA and profiler tooling before Python-side installs."

for required_cmd in nvidia-smi ncu nsys; do
  if ! command -v "$required_cmd" >/dev/null 2>&1; then
    workflow_log "Required command '$required_cmd' was not found in PATH."
    workflow_log "This workflow avoids system-wide installation by default, so please install the missing Linux-side tool manually."
    exit 1
  fi
done

# Print the exact GPU and profiler details that the rest of the workflow will rely on.
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
ncu --version
nsys --version

workflow_log "Verified the pinned Python dependency set inside $DL_ENV_ROOT."

# TensorRT-LLM ships native bindings that require OpenMPI at runtime even after pip install succeeds.
ensure_tensorrt_llm_importable
ensure_gemma4_autodeploy_support

# Show the effective interpreter and package versions that will be used for every later script.
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import sys

print(f"python_executable={sys.executable}")
for package_name in ("torch", "transformers", "tensorrt_llm", "PyYAML", "huggingface_hub"):
    try:
        print(f"{package_name}={md.version(package_name)}")
    except md.PackageNotFoundError:
        print(f"{package_name}=not-installed")
PY

workflow_log "Running compatibility preflight."
"$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" preflight
