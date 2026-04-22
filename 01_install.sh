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
    workflow_log "Install Python 3.12 on this machine first, then rerun 01_install.sh."
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
  "$python312_bin" -m venv --without-pip --system-site-packages "$DL_ENV_ROOT"
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

function ensure_pinned_python_stack() {
  local torch_constraints_file
  local torch_cuda_index_url="https://download.pytorch.org/whl/cu130"

  workflow_log "Installing or updating packaging tools inside $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade pip 'setuptools<80' 'wheel<=0.45.1'

  workflow_log "Installing pinned CUDA 13.0 PyTorch packages for aarch64 inside $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade \
    --index-url "$torch_cuda_index_url" \
    --extra-index-url https://pypi.org/simple \
    'torch==2.9.0+cu130' \
    torchvision==0.24.0

  # TensorRT-LLM can cause pip to replace the CUDA-enabled torch wheel with the
  # CPU-only build unless torch is constrained during dependency resolution.
  torch_constraints_file="$(mktemp)"
  cat >"$torch_constraints_file" <<'EOF'
torch==2.9.0+cu130
torchvision==0.24.0
EOF

  workflow_log "Installing pinned workflow packages inside $DL_ENV_ROOT."
  "$PIP_BIN" install --upgrade \
    --extra-index-url "$torch_cuda_index_url" \
    --extra-index-url https://pypi.nvidia.com \
    -c "$torch_constraints_file" \
    tensorrt-llm==1.1.0 \
    transformers==4.56.0 \
    PyYAML==6.0.3 \
    sentencepiece==0.2.1 \
    Pillow==12.1.1 \
    accelerate==1.13.0 \
    huggingface_hub==0.36.2

  rm -f "$torch_constraints_file"
}

ensure_selected_python_env
require_python_env
ensure_workflow_dirs

workflow_log "Verifying CUDA and profiler tooling before Python-side installs."

for required_cmd in nvidia-smi ncu nsys; do
  if ! command -v "$required_cmd" >/dev/null 2>&1; then
    workflow_log "Required command '$required_cmd' was not found in PATH."
    workflow_log "Please install the missing tool (e.g. via apt or the NVIDIA SDK Manager) and rerun 01_install.sh."
    exit 1
  fi
done

# Print the exact GPU and profiler details that the rest of the workflow will rely on.
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
ncu --version
nsys --version

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

function refresh_native_runtime_env() {
  export PATH="$DL_ENV_ROOT/bin:$PATH"
  export OPAL_PREFIX="${OPAL_PREFIX:-$DL_ENV_ROOT}"

  for native_dir in \
    "$DL_ENV_ROOT/lib" \
    "$DL_ENV_ROOT/lib/openmpi" \
    "/usr/local/cuda/lib64" \
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

modeling_text = urlopen(patch_url).read().decode("utf-8")
# The pinned TensorRT-LLM 1.1.0 torch_moe op does not expose the newer
# ActivationType-based act_fn argument expected by this upstream Gemma 4 patch.
modeling_text = modeling_text.replace(
    "from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizerFast\n",
    "from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedTokenizerFast\n",
)
modeling_text = modeling_text.replace(
    "from tensorrt_llm._torch.utils import ActivationType\n",
    "",
)
modeling_text = modeling_text.replace(
    "            is_gated_mlp=True,\n            act_fn=int(ActivationType.Gelu),\n",
    "            is_gated_mlp=True,\n",
)
modeling_text = modeling_text.replace(
    'AutoModelForCausalLMFactory.register_custom_model_cls("Gemma4TextConfig", Gemma4ForCausalLM)\n'
    "Gemma4ForConditionalGenerationFactory.register_custom_model_cls(\n"
    '    "Gemma4Config", Gemma4ForConditionalGeneration\n'
    ")\n",
    'AutoModelForCausalLM.register(Gemma4TextConfig, Gemma4ForCausalLM, exist_ok=True)\n'
    "AutoModelForCausalLM.register(\n"
    "    Gemma4Config, Gemma4ForConditionalGeneration, exist_ok=True\n"
    ")\n",
)
modeling_path.write_text(modeling_text)

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
