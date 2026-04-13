#!/home/songzhu/Desktop/dl_env/bin/python
"""Helper entrypoints for the Gemma 4 WSL2 profiling workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
RUNTIME_DIR = ARTIFACTS_DIR / "runtime"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RUN_CONFIG_PATH = ARTIFACTS_DIR / "run_config.json"

DEFAULT_PROMPT = "How does a large language model work?"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_BATCH_SIZE = 1

REQUESTED_MODEL_ID = "google/gemma-4-E2B-it"
MODEL_CANDIDATES: List[Dict[str, Any]] = [
    {
        "model_id": "google/gemma-4-E2B-it",
        "template_kind": "e2b_local_generic",
        "free_gpu_memory_fraction": 0.10,
    },
    {
        "model_id": "google/gemma-4-26B-A4B-it",
        "template_kind": "registry_moe",
        "free_gpu_memory_fraction": 0.40,
    },
    {
        "model_id": "google/gemma-4-31B-it",
        "template_kind": "registry_dense",
        "free_gpu_memory_fraction": 0.80,
    },
]

PRECISION_ORDER = ["FP16", "FP32", "BF16", "INT8"]
PRECISION_TO_DTYPE = {
    "FP16": "float16",
    "FP32": "float32",
    "BF16": "bfloat16",
}

FORWARD_STEP_PATTERN = re.compile(
    r"_forward_step\s+(\d+):\s+(\d+)\s+ctx reqs,\s+(\d+)\s+gen reqs"
)


class WorkflowError(RuntimeError):
    """Raised when a workflow step fails in an expected way."""


@dataclass
class StepRecord:
    text: str
    step_number: int
    ctx_requests: int
    gen_requests: int
    start_ns: int
    end_ns: int

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "step_number": self.step_number,
            "ctx_requests": self.ctx_requests,
            "gen_requests": self.gen_requests,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_us": self.duration_us,
        }


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


def fail(message: str) -> None:
    raise WorkflowError(message)


def ensure_dirs() -> None:
    for path in (ARTIFACTS_DIR, CACHE_DIR, RUNTIME_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)

    cache_env_dirs = [
        Path(os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))),
        Path(
            os.environ.setdefault(
                "HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hf" / "hub")
            )
        ),
        Path(
            os.environ.setdefault(
                "TRANSFORMERS_CACHE", str(CACHE_DIR / "hf" / "transformers")
            )
        ),
        Path(os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))),
        Path(
            os.environ.setdefault(
                "TORCHINDUCTOR_CACHE_DIR", str(CACHE_DIR / "torchinductor")
            )
        ),
        Path(os.environ.setdefault("TRITON_CACHE_DIR", str(CACHE_DIR / "triton"))),
        Path(
            os.environ.setdefault(
                "TLLM_LLMAPI_BUILD_CACHE_ROOT",
                str(CACHE_DIR / "tllm_build_cache"),
            )
        ),
    ]
    os.environ.setdefault("TLLM_LLMAPI_BUILD_CACHE", "1")
    for path in cache_env_dirs:
        path.mkdir(parents=True, exist_ok=True)


def run(
    cmd: Sequence[str],
    *,
    capture_output: bool = False,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    log(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        list(cmd),
        check=check,
        text=True,
        capture_output=capture_output,
        env=env or os.environ.copy(),
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def default_run_config() -> Dict[str, Any]:
    return {
        "requested_model_id": REQUESTED_MODEL_ID,
        "workload": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "prompt": DEFAULT_PROMPT,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        },
        "precision_probe_order": PRECISION_ORDER,
        "candidate_models": MODEL_CANDIDATES,
        "artifacts": {
            "artifacts_dir": str(ARTIFACTS_DIR),
            "cache_dir": str(CACHE_DIR),
            "runtime_dir": str(RUNTIME_DIR),
            "reports_dir": str(REPORTS_DIR),
            "run_config_json": str(RUN_CONFIG_PATH),
        },
    }


def load_run_config() -> Dict[str, Any]:
    config = default_run_config()
    if RUN_CONFIG_PATH.exists():
        existing = read_json(RUN_CONFIG_PATH)
        config.update(existing)
    return config


def save_run_config(config: Dict[str, Any]) -> None:
    write_json(RUN_CONFIG_PATH, config)


def slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def fetch_json(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_hf_config(model_id: str) -> Dict[str, Any]:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    return fetch_json(url)


def command_path(name: str) -> Optional[str]:
    return shutil.which(name)


def collect_gpu_info() -> Dict[str, Any]:
    if not command_path("nvidia-smi"):
        fail("nvidia-smi is not available in PATH.")

    result = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
    )
    line = result.stdout.strip().splitlines()[0]
    name, driver_version, memory_total, compute_cap = [part.strip() for part in line.split(",")]
    return {
        "name": name,
        "driver_version": driver_version,
        "memory_total": memory_total,
        "compute_capability": compute_cap,
    }


def collect_host_info() -> Dict[str, Any]:
    uname = run(["uname", "-a"], capture_output=True).stdout.strip()
    os_release = {}
    if Path("/etc/os-release").exists():
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')

    return {
        "uname": uname,
        "os_release": os_release,
        "is_wsl2": "WSL2" in uname or "microsoft-standard-WSL2" in uname,
        "ncu_path": command_path("ncu"),
        "nsys_path": command_path("nsys"),
    }


def collect_python_stack_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_bf16_supported"] = bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        info["torch_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception as exc:  # pragma: no cover - runtime dependent
        info["transformers_error"] = f"{type(exc).__name__}: {exc}"

    return info


def detect_tensorrt_llm_support() -> Dict[str, Any]:
    support: Dict[str, Any] = {
        "installed": False,
        "importable": False,
        "gemma4_autodeploy_supported": False,
        "missing_openmpi_runtime": False,
        "errors": [],
    }
    try:
        import importlib.metadata as importlib_metadata

        support["version"] = importlib_metadata.version("tensorrt_llm")
        support["installed"] = True
    except Exception:
        support["errors"].append(
            "Package metadata for tensorrt_llm was not found in /home/songzhu/Desktop/dl_env."
        )
        return support

    try:
        import tensorrt_llm

        support["importable"] = True
        support["version"] = getattr(tensorrt_llm, "__version__", support["version"])
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        support["errors"].append(f"Failed to import tensorrt_llm: {message}")
        if "libmpi.so.40" in str(exc) or "cannot load MPI library" in str(exc):
            support["missing_openmpi_runtime"] = True
        return support

    try:
        from tensorrt_llm._torch.auto_deploy import LLM as _  # noqa: F401

        support["autodeploy_llm_class"] = True
    except Exception as exc:
        support["errors"].append(f"Failed to import AutoDeploy LLM: {exc}")
        support["autodeploy_llm_class"] = False

    try:
        from tensorrt_llm._torch.auto_deploy.models.custom import (  # noqa: F401
            Gemma4ForConditionalGeneration,
        )

        support["gemma4_custom_model"] = True
    except Exception as exc:
        support["errors"].append(f"Failed to import Gemma4 custom model: {exc}")
        support["gemma4_custom_model"] = False

    support["gemma4_autodeploy_supported"] = bool(
        support.get("autodeploy_llm_class") and support.get("gemma4_custom_model")
    )
    return support


def gather_precision_support() -> Dict[str, Dict[str, Any]]:
    support: Dict[str, Dict[str, Any]] = {}
    torch_info = collect_python_stack_info()
    compute_capability = collect_gpu_info()["compute_capability"]

    support["FP16"] = {
        "supported": True,
        "dtype": "float16",
        "reason": "Ampere GPUs support FP16 Tensor Core execution.",
    }
    support["FP32"] = {
        "supported": True,
        "dtype": "float32",
        "reason": "FP32 is always available as a fallback precision.",
    }
    support["BF16"] = {
        "supported": bool(torch_info.get("torch_bf16_supported")),
        "dtype": "bfloat16",
        "reason": "BF16 requires backend and GPU support; torch.cuda reports availability."
        if torch_info.get("torch_bf16_supported")
        else "torch.cuda does not report BF16 support in this environment.",
    }
    support["INT8"] = {
        "supported": False,
        "dtype": None,
        "reason": (
            "No official INT8 Gemma 4 checkpoint or explicit INT8 workflow is configured "
            "for this profiling workflow."
        ),
    }
    support["metadata"] = {
        "gpu_compute_capability": compute_capability,
    }
    return support


def embedded_yaml_template(
    model_id: str, prompt_tokens: int, max_new_tokens: int
) -> Tuple[str, Dict[str, Any]]:
    total_tokens = prompt_tokens + max_new_tokens
    candidate = next(item for item in MODEL_CANDIDATES if item["model_id"] == model_id)
    template_kind = candidate["template_kind"]

    base = {
        "model_factory": "Gemma4ForConditionalGeneration",
        "tokenizer": model_id,
        "attn_backend": "triton_paged",
        "compile_backend": "torch-cudagraph",
        "cuda_graph_config": {"batch_sizes": [1]},
        "max_num_tokens": total_tokens,
        "max_batch_size": 1,
        "max_seq_len": total_tokens,
        "enable_chunked_prefill": False,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": candidate["free_gpu_memory_fraction"],
        },
        "transforms": {
            "compile_model": {"piecewise_enabled": True},
            "mlir_elementwise_fusion": {"enabled": True},
            "gather_logits_before_lm_head": {"enabled": True},
            "fuse_gemms": {"enabled": True},
        },
    }

    if template_kind == "registry_moe":
        base["transforms"]["multi_stream_moe"] = {"enabled": False}
        origin = "embedded-registry-moe-template"
    elif template_kind == "registry_dense":
        origin = "embedded-registry-dense-template"
    else:
        origin = "local-e2b-template"

    return origin, base


def locate_prompt_metadata(config: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    prepared = config.get("prepared_models", {}).get(model_id)
    if not prepared:
        fail(
            f"Prompt metadata for {model_id} is missing. Run 02_prepare_model.sh first."
        )
    return prepared


def processor_bundle_for_model(model_id: str) -> Tuple[str, Any]:
    from transformers import AutoProcessor, AutoTokenizer

    errors: List[str] = []
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return "processor", processor
    except Exception as exc:
        errors.append(f"AutoProcessor failed: {exc}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return "tokenizer", tokenizer
    except Exception as exc:
        errors.append(f"AutoTokenizer failed: {exc}")

    fail(f"Failed to load text templating assets for {model_id}: {'; '.join(errors)}")


def render_chat_prompt(bundle_type: str, bundle: Any) -> Tuple[str, int]:
    messages = [{"role": "user", "content": DEFAULT_PROMPT}]

    if not hasattr(bundle, "apply_chat_template"):
        fail("Loaded processor/tokenizer does not expose apply_chat_template().")

    try:
        templated = bundle.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        templated = bundle.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if bundle_type == "processor":
        try:
            tokenized = bundle(text=templated, return_tensors="pt")
        except TypeError:
            tokenizer = getattr(bundle, "tokenizer", None)
            if tokenizer is None:
                fail("AutoProcessor tokenization fallback failed: tokenizer attribute missing.")
            tokenized = tokenizer(templated, return_tensors="pt")
    else:
        tokenized = bundle(templated, return_tensors="pt")

    input_ids = tokenized["input_ids"]
    token_count = int(input_ids.shape[-1])
    return templated, token_count


def yaml_path_for(model_id: str, precision: str) -> Path:
    return RUNTIME_DIR / f"{slugify_model_id(model_id)}__{precision.lower()}.yaml"


def dtype_for_precision(precision: str) -> Optional[str]:
    return PRECISION_TO_DTYPE.get(precision)


def record_failure(
    failures: List[Dict[str, Any]],
    model_id: str,
    precision: str,
    reason: str,
) -> None:
    failures.append(
        {
            "model_id": model_id,
            "precision": precision,
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def extract_output_metadata(result: Any) -> Dict[str, Any]:
    request = result[0] if isinstance(result, list) else result
    metadata: Dict[str, Any] = {
        "finished": getattr(request, "finished", None),
    }
    outputs = getattr(request, "outputs", None)
    if outputs:
        first = outputs[0]
        metadata["text"] = getattr(first, "text", None)
        token_ids = getattr(first, "token_ids", None)
        if token_ids is not None:
            metadata["output_token_count"] = len(token_ids)
    return metadata


def attempt_runtime_probe(
    model_id: str,
    precision: str,
    prompt_text: str,
    yaml_path: Path,
    max_new_tokens: int,
) -> Dict[str, Any]:
    import torch
    from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
    from tensorrt_llm.llmapi import SamplingParams

    dtype = dtype_for_precision(precision)
    if dtype is None:
        fail(f"No runtime dtype mapping is defined for {precision}.")

    torch.cuda.empty_cache()
    start = time.time()
    with AutoDeployLLM(
        model=model_id,
        tokenizer=model_id,
        dtype=dtype,
        yaml_extra=[str(yaml_path)],
        trust_remote_code=True,
    ) as llm:
        result = llm.generate(
            prompt_text,
            sampling_params=SamplingParams(max_tokens=max_new_tokens),
            use_tqdm=False,
        )
        metadata = extract_output_metadata(result)

        runtime_info = {
            "backend": "_autodeploy",
            "dtype": dtype,
            "compile_backend": "torch-cudagraph",
            "attention_backend": "triton_paged",
            "tensor_parallel_size": 1,
        }
        if hasattr(llm, "args"):
            runtime_info["llm_args_backend"] = getattr(llm.args, "backend", None)

    elapsed = time.time() - start
    torch.cuda.empty_cache()

    metadata.update(runtime_info)
    metadata["probe_elapsed_seconds"] = round(elapsed, 3)
    return metadata


def preflight_command(_: argparse.Namespace) -> int:
    ensure_dirs()
    config = load_run_config()

    host_info = collect_host_info()
    gpu_info = collect_gpu_info()
    python_stack = collect_python_stack_info()
    tllm_support = detect_tensorrt_llm_support()
    precision_support = gather_precision_support()

    if not host_info["is_wsl2"]:
        fail("This workflow expects Ubuntu running on WSL2.")
    if not host_info["ncu_path"] or not host_info["nsys_path"]:
        fail("Both ncu and nsys must be available before profiling.")
    if not python_stack.get("torch_cuda_available"):
        fail("torch.cuda.is_available() is false in /home/songzhu/Desktop/dl_env.")
    if not tllm_support["installed"]:
        fail("tensorrt_llm is not installed in /home/songzhu/Desktop/dl_env.")
    if not tllm_support.get("importable"):
        if tllm_support.get("missing_openmpi_runtime"):
            fail(
                "TensorRT-LLM is installed but cannot be imported because libmpi.so.40 "
                "(the OpenMPI runtime library) is missing. Install libopenmpi3 and "
                "openmpi-bin system-wide, then rerun 01_install.sh."
            )
        detail = "; ".join(tllm_support.get("errors", [])) or "unknown import error"
        fail(
            "TensorRT-LLM is installed but cannot be imported in /home/songzhu/Desktop/dl_env: "
            f"{detail}"
        )
    if not tllm_support["gemma4_autodeploy_supported"]:
        detail = "; ".join(tllm_support.get("errors", []))
        fail(
            "The installed TensorRT-LLM release does not expose Gemma 4 AutoDeploy support."
            + (f" Details: {detail}" if detail else "")
        )

    hf_models: List[Dict[str, Any]] = []
    for candidate in MODEL_CANDIDATES:
        model_id = candidate["model_id"]
        hf_config = fetch_hf_config(model_id)
        architectures = hf_config.get("architectures", [])
        hf_models.append(
            {
                "model_id": model_id,
                "architectures": architectures,
                "model_type": hf_config.get("model_type"),
                "text_model_type": (hf_config.get("text_config") or {}).get("model_type"),
                "workflow_supported": "Gemma4ForConditionalGeneration" in architectures,
            }
        )

    config["environment"] = {
        "host": host_info,
        "gpu": gpu_info,
        "python_stack": python_stack,
        "tensorrt_llm": tllm_support,
        "precision_support": precision_support,
        "preflight_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config["hf_models"] = hf_models
    config["workflow"] = {
        "framework": "TensorRT-LLM AutoDeploy",
        "runtime_backend": "trtllm",
        "compile_backend": "torch-cudagraph",
        "attention_backend": "triton_paged",
        "phase_marker_source": "[Executor] _forward_step NVTX ranges",
    }
    config.setdefault("fallback_decisions", [])
    config.setdefault("probe_failures", [])
    save_run_config(config)

    log("Preflight completed successfully.")
    return 0


def prepare_model_command(_: argparse.Namespace) -> int:
    ensure_dirs()
    config = load_run_config()
    if "environment" not in config:
        preflight_command(argparse.Namespace())
        config = load_run_config()

    prepared = config.setdefault("prepared_models", {})

    for candidate in MODEL_CANDIDATES:
        model_id = candidate["model_id"]
        if model_id in prepared and Path(prepared[model_id]["templated_prompt_path"]).exists():
            continue

        log(f"Preparing prompt assets for {model_id}")
        bundle_type, bundle = processor_bundle_for_model(model_id)
        templated_prompt, token_count = render_chat_prompt(bundle_type, bundle)

        model_dir = RUNTIME_DIR / slugify_model_id(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = model_dir / "templated_prompt.txt"
        prompt_meta_path = model_dir / "templated_prompt.json"

        prompt_path.write_text(templated_prompt)
        prompt_meta = {
            "model_id": model_id,
            "bundle_type": bundle_type,
            "input_token_count": token_count,
            "templated_prompt_path": str(prompt_path),
            "prompt_text": DEFAULT_PROMPT,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        }
        write_json(prompt_meta_path, prompt_meta)
        prompt_meta["templated_prompt_metadata_path"] = str(prompt_meta_path)
        prepared[model_id] = prompt_meta

    config["prepared_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_run_config(config)
    log("Model preparation completed successfully.")
    return 0


def prepare_runtime_command(args: argparse.Namespace) -> int:
    ensure_dirs()
    config = load_run_config()
    if "prepared_models" not in config:
        prepare_model_command(argparse.Namespace())
        config = load_run_config()

    selected = config.get("selected_runtime")
    if (
        selected
        and not args.force
        and Path(selected.get("yaml_path", "")).exists()
        and Path(selected.get("prompt_path", "")).exists()
    ):
        log("Runtime preparation already completed; reusing existing selection.")
        return 0

    failures: List[Dict[str, Any]] = []
    precision_support = config["environment"]["precision_support"]

    for candidate in MODEL_CANDIDATES:
        model_id = candidate["model_id"]
        prompt_meta = locate_prompt_metadata(config, model_id)
        prompt_path = Path(prompt_meta["templated_prompt_path"])
        prompt_text = prompt_path.read_text()
        prompt_tokens = int(prompt_meta["input_token_count"])

        for precision in PRECISION_ORDER:
            support = precision_support.get(precision, {})
            if not support.get("supported"):
                record_failure(failures, model_id, precision, support.get("reason", "Unsupported precision."))
                continue

            yaml_origin, yaml_payload = embedded_yaml_template(
                model_id, prompt_tokens, DEFAULT_MAX_NEW_TOKENS
            )
            yaml_file = yaml_path_for(model_id, precision)
            yaml_file.write_text(yaml.safe_dump(yaml_payload, sort_keys=False))

            try:
                probe = attempt_runtime_probe(
                    model_id=model_id,
                    precision=precision,
                    prompt_text=prompt_text,
                    yaml_path=yaml_file,
                    max_new_tokens=1,
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                record_failure(failures, model_id, precision, tb)
                continue

            selected_runtime = {
                "requested_model_id": REQUESTED_MODEL_ID,
                "actual_model_id": model_id,
                "precision": precision,
                "dtype": dtype_for_precision(precision),
                "workflow": "TensorRT-LLM AutoDeploy text-only",
                "compile_backend": "torch-cudagraph",
                "attention_backend": "triton_paged",
                "yaml_path": str(yaml_file),
                "yaml_origin": yaml_origin,
                "prompt_path": str(prompt_path),
                "input_token_count": prompt_tokens,
                "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                "batch_size": DEFAULT_BATCH_SIZE,
                "probe": probe,
                "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cache_paths": {
                    "hf_home": os.environ["HF_HOME"],
                    "huggingface_hub_cache": os.environ["HUGGINGFACE_HUB_CACHE"],
                    "transformers_cache": os.environ["TRANSFORMERS_CACHE"],
                    "torchinductor_cache_dir": os.environ["TORCHINDUCTOR_CACHE_DIR"],
                    "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
                    "tllm_llmapi_build_cache_root": os.environ["TLLM_LLMAPI_BUILD_CACHE_ROOT"],
                },
            }

            if model_id != REQUESTED_MODEL_ID:
                config.setdefault("fallback_decisions", []).append(
                    {
                        "from_model_id": REQUESTED_MODEL_ID,
                        "to_model_id": model_id,
                        "reason": (
                            "The requested model did not complete a successful support/feasibility "
                            "probe before this family variant."
                        ),
                    }
                )

            config["selected_runtime"] = selected_runtime
            config["probe_failures"] = failures
            save_run_config(config)
            log(
                f"Selected {model_id} with {precision} after a successful AutoDeploy probe."
            )
            return 0

    config["probe_failures"] = failures
    save_run_config(config)
    fail("No supported Gemma 4 model/precision combination completed a successful probe.")


def load_selected_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = config.get("selected_runtime")
    if not runtime:
        fail(
            "No selected_runtime is present in artifacts/run_config.json. Run 03_build_or_prepare_runtime.sh first."
        )
    return runtime


def run_inference_command(args: argparse.Namespace) -> int:
    ensure_dirs()
    config = load_run_config()
    runtime = load_selected_runtime(config)

    from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
    from tensorrt_llm.llmapi import SamplingParams

    prompt_path = Path(runtime["prompt_path"])
    prompt_text = prompt_path.read_text()
    max_new_tokens = args.max_new_tokens or runtime["max_new_tokens"]

    started = time.time()
    with AutoDeployLLM(
        model=runtime["actual_model_id"],
        tokenizer=runtime["actual_model_id"],
        dtype=runtime["dtype"],
        yaml_extra=[runtime["yaml_path"]],
        trust_remote_code=True,
    ) as llm:
        result = llm.generate(
            prompt_text,
            sampling_params=SamplingParams(max_tokens=max_new_tokens),
            use_tqdm=False,
        )
        metadata = extract_output_metadata(result)

    metadata.update(
        {
            "model_id": runtime["actual_model_id"],
            "requested_model_id": runtime["requested_model_id"],
            "precision": runtime["precision"],
            "dtype": runtime["dtype"],
            "max_new_tokens": max_new_tokens,
            "prompt_path": str(prompt_path),
            "elapsed_seconds": round(time.time() - started, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    if args.metadata_output:
        write_json(Path(args.metadata_output), metadata)
    else:
        print(json.dumps(metadata, indent=2, sort_keys=True))

    return 0


def export_nsys_sqlite(report_path: Path) -> Path:
    sqlite_prefix = report_path.with_suffix("")
    run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_prefix),
            str(report_path),
        ]
    )
    sqlite_path = sqlite_prefix.with_suffix(".sqlite")
    if not sqlite_path.exists():
        fail(f"Expected exported SQLite report was not created at {sqlite_path}")
    return sqlite_path


def parse_forward_steps(sqlite_path: Path) -> List[StepRecord]:
    query = """
        SELECT s.value, n.start, n.end
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE n.end > 0 AND s.value LIKE '%[Executor] _forward_step%'
        ORDER BY n.start
    """
    connection = sqlite3.connect(str(sqlite_path))
    try:
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()

    steps: List[StepRecord] = []
    for text, start_ns, end_ns in rows:
        match = FORWARD_STEP_PATTERN.search(text)
        if not match:
            continue
        step_number, ctx, gen = [int(group) for group in match.groups()]
        candidate = StepRecord(
            text=text,
            step_number=step_number,
            ctx_requests=ctx,
            gen_requests=gen,
            start_ns=int(start_ns),
            end_ns=int(end_ns),
        )

        # NVTX ranges can appear more than once in TP settings; collapse exact duplicates.
        if steps and steps[-1].text == candidate.text and abs(steps[-1].start_ns - candidate.start_ns) <= 100_000:
            continue
        steps.append(candidate)
    return steps


def choose_phase_step(steps: Sequence[StepRecord], phase: str) -> StepRecord:
    if phase == "prefill":
        for step in steps:
            if step.ctx_requests > 0:
                return step
        fail("No prefill step with ctx reqs > 0 was found in the Nsight Systems trace.")

    generation_only = [step for step in steps if step.ctx_requests == 0 and step.gen_requests > 0]
    if not generation_only:
        fail(
            "No generation-only step with ctx reqs = 0 and gen reqs > 0 was found in the Nsight Systems trace."
        )
    return generation_only[len(generation_only) // 2]


def nvtx_filter_for_text(step_text: str) -> str:
    return f"regex:{re.escape(step_text)}/"


def summarize_nsys_command(args: argparse.Namespace) -> int:
    ensure_dirs()
    config = load_run_config()
    runtime = load_selected_runtime(config)
    report_path = Path(args.report).resolve()
    if not report_path.exists():
        fail(f"Nsight Systems report not found: {report_path}")

    sqlite_path = export_nsys_sqlite(report_path)
    steps = parse_forward_steps(sqlite_path)
    if not steps:
        fail("No TensorRT-LLM _forward_step NVTX ranges were found in the exported SQLite report.")

    selected_step = choose_phase_step(steps, args.phase)
    summary = {
        "phase": args.phase,
        "requested_model_id": runtime["requested_model_id"],
        "actual_model_id": runtime["actual_model_id"],
        "precision": runtime["precision"],
        "report_path": str(report_path),
        "sqlite_path": str(sqlite_path),
        "selected_step": selected_step.to_dict(),
        "selected_nvtx_include_filter": nvtx_filter_for_text(selected_step.text),
        "steps": [step.to_dict() for step in steps],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = REPORTS_DIR / f"{args.phase}_phase_summary.json"
    write_json(summary_path, summary)

    profiles = config.setdefault("profiles", {})
    phase_profile = profiles.setdefault(args.phase, {})
    phase_profile.update(
        {
            "nsys_rep": str(report_path),
            "nsys_sqlite": str(sqlite_path),
            "summary_json": str(summary_path),
            "selected_step_text": selected_step.text,
            "selected_step_number": selected_step.step_number,
            "selected_nvtx_include_filter": summary["selected_nvtx_include_filter"],
        }
    )
    save_run_config(config)

    log(f"Saved {args.phase} phase summary to {summary_path}")
    return 0


def nvtx_filter_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    phase_profile = config.get("profiles", {}).get(args.phase)
    if not phase_profile or not phase_profile.get("selected_nvtx_include_filter"):
        fail(
            f"No NVTX filter is recorded for phase '{args.phase}'. Run the corresponding Nsight Systems script first."
        )
    print(phase_profile["selected_nvtx_include_filter"])
    return 0


def register_ncu_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    profiles = config.setdefault("profiles", {})
    phase_profile = profiles.setdefault(args.phase, {})
    phase_profile["ncu_rep"] = str(Path(args.report).resolve())
    phase_profile["ncu_registered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_run_config(config)
    return 0


def report_config_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    runtime = config.get("selected_runtime")

    lines = [
        "Gemma 4 Profiling Run Configuration",
        f"Requested model: {config.get('requested_model_id', REQUESTED_MODEL_ID)}",
    ]

    if runtime:
        lines.extend(
            [
                f"Actual model: {runtime.get('actual_model_id')}",
                f"TensorRT-LLM version: {config.get('environment', {}).get('tensorrt_llm', {}).get('version', 'unknown')}",
                f"Workflow: {runtime.get('workflow')}",
                f"Precision: {runtime.get('precision')}",
                f"Batch size: {runtime.get('batch_size')}",
                f"Input token count: {runtime.get('input_token_count')}",
                f"Output token cap: {runtime.get('max_new_tokens')}",
                f"YAML path: {runtime.get('yaml_path')}",
            ]
        )
    else:
        lines.append("Actual model: not selected yet")

    fallback_decisions = config.get("fallback_decisions", [])
    if fallback_decisions:
        lines.append("Fallback decisions:")
        for entry in fallback_decisions:
            lines.append(
                f"  - {entry['from_model_id']} -> {entry['to_model_id']}: {entry['reason']}"
            )

    probe_failures = config.get("probe_failures", [])
    if probe_failures:
        lines.append("Rejected model/precision probes:")
        for failure in probe_failures:
            lines.append(
                f"  - {failure['model_id']} / {failure['precision']}: {failure['reason']}"
            )

    for phase in ("prefill", "decode"):
        phase_profile = config.get("profiles", {}).get(phase, {})
        if phase_profile:
            lines.append(f"{phase.title()} artifacts:")
            for key in ("nsys_rep", "nsys_sqlite", "summary_json", "ncu_rep"):
                if phase_profile.get(key):
                    lines.append(f"  - {key}: {phase_profile[key]}")

    report_text = "\n".join(lines) + "\n"
    print(report_text, end="")

    if args.output:
        Path(args.output).write_text(report_text)

    config.setdefault("reports", {})
    if args.output:
        config["reports"]["human_summary"] = str(Path(args.output).resolve())
        save_run_config(config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="Run environment and compatibility preflight checks.")
    subparsers.add_parser(
        "prepare-model",
        help="Prepare prompt templating metadata and cache model-side text assets.",
    )

    prepare_runtime = subparsers.add_parser(
        "prepare-runtime",
        help="Probe candidate Gemma 4 models and select the first runnable precision path.",
    )
    prepare_runtime.add_argument(
        "--force",
        action="store_true",
        help="Force a new runtime selection probe even if a previous selection exists.",
    )

    run_inference = subparsers.add_parser(
        "run-inference",
        help="Run one inference request with the selected runtime configuration.",
    )
    run_inference.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the configured max_new_tokens value for this invocation.",
    )
    run_inference.add_argument(
        "--metadata-output",
        type=str,
        default=None,
        help="Optional JSON path for inference metadata.",
    )

    summarize_nsys = subparsers.add_parser(
        "summarize-nsys",
        help="Export a .nsys-rep file to SQLite and derive the target NVTX phase marker.",
    )
    summarize_nsys.add_argument("--phase", choices=["prefill", "decode"], required=True)
    summarize_nsys.add_argument("--report", required=True)

    nvtx_filter = subparsers.add_parser(
        "nvtx-filter",
        help="Print the ncu --nvtx-include filter for a selected profiling phase.",
    )
    nvtx_filter.add_argument("--phase", choices=["prefill", "decode"], required=True)

    register_ncu = subparsers.add_parser(
        "register-ncu",
        help="Record an .ncu-rep path in artifacts/run_config.json.",
    )
    register_ncu.add_argument("--phase", choices=["prefill", "decode"], required=True)
    register_ncu.add_argument("--report", required=True)

    report_config = subparsers.add_parser(
        "report-config",
        help="Print and optionally save a human-readable run summary.",
    )
    report_config.add_argument("--output", default=None)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "preflight":
            return preflight_command(args)
        if args.command == "prepare-model":
            return prepare_model_command(args)
        if args.command == "prepare-runtime":
            return prepare_runtime_command(args)
        if args.command == "run-inference":
            return run_inference_command(args)
        if args.command == "summarize-nsys":
            return summarize_nsys_command(args)
        if args.command == "nvtx-filter":
            return nvtx_filter_command(args)
        if args.command == "register-ncu":
            return register_ncu_command(args)
        if args.command == "report-config":
            return report_config_command(args)
    except WorkflowError as exc:
        log(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        log(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        return 1

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
