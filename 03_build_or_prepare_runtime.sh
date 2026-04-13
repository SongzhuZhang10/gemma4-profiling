#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# The Gemma 4 Python patch applied in 01_install.sh comes from a newer upstream
# commit than the local TensorRT-LLM 1.2.0 custom op bundle. On this machine,
# the older auto_deploy::torch_attention op does not accept the trailing
# layer_idx argument, so patch the installed Gemma 4 model file back to the
# local runtime signature before probing.
"$PYTHON_BIN" - <<'PY'
import sysconfig
from pathlib import Path

import tensorrt_llm

site_packages = Path(sysconfig.get_paths()["purelib"])
gemma4_path = (
    site_packages / "tensorrt_llm" / "_torch" / "auto_deploy" / "models" / "custom" / "modeling_gemma4.py"
)
factory_path = (
    site_packages / "tensorrt_llm" / "_torch" / "auto_deploy" / "models" / "factory.py"
)
hf_models_path = site_packages / "tensorrt_llm" / "_torch" / "auto_deploy" / "models" / "hf.py"
triton_attn_path = (
    site_packages / "tensorrt_llm" / "_torch" / "auto_deploy" / "custom_ops" / "triton_attention.py"
)
torch_attn_path = (
    site_packages
    / "tensorrt_llm"
    / "_torch"
    / "auto_deploy"
    / "custom_ops"
    / "torch_backend_attention.py"
)

if gemma4_path.exists() and tensorrt_llm.__version__ == "1.2.0":
    text = gemma4_path.read_text()
    legacy_call = "            self.layer_idx,\n        )"
    if legacy_call in text:
        text = text.replace(legacy_call, "        )", 1)
    legacy_mlp = """class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
"""
    widened_mlp = """class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
"""
    if legacy_mlp in text:
        text = text.replace(legacy_mlp, widened_mlp, 1)
    legacy_mlp_ctor = "        self.mlp = Gemma4TextMLP(config)\n"
    if legacy_mlp_ctor in text:
        text = text.replace(legacy_mlp_ctor, "        self.mlp = Gemma4TextMLP(config, layer_idx)\n", 1)
    gemma4_path.write_text(text)
    print(f"Patched {gemma4_path} for TensorRT-LLM 1.2.0 Gemma 4 compatibility.")

if factory_path.exists() and tensorrt_llm.__version__ == "1.2.0":
    text = factory_path.read_text()
    legacy_dynamic = """    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        batch_size_dyn = Dim.DYNAMIC
        seq_len_dyn = Dim.DYNAMIC
        return {
            "input_ids": {0: batch_size_dyn, 1: seq_len_dyn},
            "position_ids": {0: batch_size_dyn, 1: seq_len_dyn},
        }
"""
    static_batch = """    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        seq_len_dyn = Dim.DYNAMIC
        return {
            "input_ids": {1: seq_len_dyn},
            "position_ids": {1: seq_len_dyn},
        }
"""
    if legacy_dynamic in text:
        factory_path.write_text(text.replace(legacy_dynamic, static_batch, 1))
        print(f"Patched {factory_path} for static batch-1 export hints.")

if hf_models_path.exists() and tensorrt_llm.__version__ == "1.2.0":
    text = hf_models_path.read_text()
    legacy_dynamic = """    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        batch_size_dynamic = Dim.DYNAMIC
        seq_len_dynamic = Dim.DYNAMIC
        return {
            "input_ids": {0: batch_size_dynamic, 1: seq_len_dynamic},
            "inputs_embeds": {0: batch_size_dynamic, 1: seq_len_dynamic},
            "position_ids": {0: batch_size_dynamic, 1: seq_len_dynamic},
        }
"""
    static_batch = """    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        seq_len_dynamic = Dim.DYNAMIC
        return {
            "input_ids": {1: seq_len_dynamic},
            "inputs_embeds": {1: seq_len_dynamic},
            "position_ids": {1: seq_len_dynamic},
        }
"""
    if legacy_dynamic in text:
        hf_models_path.write_text(text.replace(legacy_dynamic, static_batch, 1))
        print(f"Patched {hf_models_path} for static batch-1 export hints.")

if tensorrt_llm.__version__ == "1.2.0":
    legacy_attn_meta = """        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        v_fake: FakeTensor = source_attn_node.args[2].meta["val"]
        num_kv_heads = k_fake.shape[2]
        k_head_dim = k_fake.shape[3]
        v_head_dim = v_fake.shape[3]
"""
    broken_attn_meta = """        k_info = source_attn_node.args[1].meta.get("val") or source_attn_node.args[1].meta.get("tensor_meta")
        v_info = source_attn_node.args[2].meta.get("val") or source_attn_node.args[2].meta.get("tensor_meta")
        if k_info is None or v_info is None:
            raise KeyError("val")
        num_kv_heads = k_info.shape[2]
        k_head_dim = k_info.shape[3]
        v_head_dim = v_info.shape[3]
"""
    updated_attn_meta = """        k_info = source_attn_node.args[1].meta.get("val")
        if k_info is None:
            k_info = source_attn_node.args[1].meta.get("tensor_meta")
        v_info = source_attn_node.args[2].meta.get("val")
        if v_info is None:
            v_info = source_attn_node.args[2].meta.get("tensor_meta")
        if k_info is None or v_info is None:
            raise KeyError("val")
        num_kv_heads = k_info.shape[2]
        k_head_dim = k_info.shape[3]
        v_head_dim = v_info.shape[3]
"""
    recursive_attn_meta_v0 = """        def _resolve_tensor_info(node):
            stack = [node]
            visited = set()
            while stack:
                candidate = stack.pop()
                candidate_id = id(candidate)
                if candidate_id in visited:
                    continue
                visited.add(candidate_id)
                meta = getattr(candidate, "meta", {})
                info = meta.get("val")
                if info is not None and hasattr(info, "shape"):
                    return info
                info = meta.get("tensor_meta")
                if info is not None and hasattr(info, "shape"):
                    return info
                stack.extend(getattr(candidate, "all_input_nodes", ()))
            return None

        k_info = _resolve_tensor_info(source_attn_node.args[1])
        v_info = _resolve_tensor_info(source_attn_node.args[2])
        if k_info is None or v_info is None:
            raise KeyError("val")
        num_kv_heads = k_info.shape[2]
        k_head_dim = k_info.shape[3]
        v_head_dim = v_info.shape[3]
"""
    recursive_attn_meta_v1 = """        def _resolve_tensor_info(node):
            stack = [node]
            visited = set()
            while stack:
                candidate = stack.pop()
                candidate_id = id(candidate)
                if candidate_id in visited:
                    continue
                visited.add(candidate_id)
                meta = getattr(candidate, "meta", {})
                info = meta.get("val")
                if info is not None and hasattr(info, "shape") and len(info.shape) >= 4:
                    return info
                info = meta.get("tensor_meta")
                if info is not None and hasattr(info, "shape") and len(info.shape) >= 4:
                    return info
                stack.extend(getattr(candidate, "all_input_nodes", ()))
            return None

        k_info = _resolve_tensor_info(source_attn_node.args[1])
        v_info = _resolve_tensor_info(source_attn_node.args[2])
        if k_info is None or v_info is None:
            raise KeyError("val")
        num_kv_heads = k_info.shape[2]
        k_head_dim = k_info.shape[3]
        v_head_dim = v_info.shape[3]
"""
    recursive_attn_meta = """        def _resolve_tensor_info(node, min_rank=4):
            stack = [node]
            visited = set()
            while stack:
                candidate = stack.pop()
                candidate_id = id(candidate)
                if candidate_id in visited:
                    continue
                visited.add(candidate_id)
                meta = getattr(candidate, "meta", {})
                info = meta.get("val")
                if info is not None and hasattr(info, "shape") and len(info.shape) >= min_rank:
                    return info
                info = meta.get("tensor_meta")
                if info is not None and hasattr(info, "shape") and len(info.shape) >= min_rank:
                    return info
                stack.extend(getattr(candidate, "all_input_nodes", ()))
            return None

        k_info = _resolve_tensor_info(source_attn_node.args[1])
        v_info = _resolve_tensor_info(source_attn_node.args[2])
        if k_info is None:
            k_info = v_info
        if v_info is None:
            v_info = k_info
        if k_info is None or v_info is None:
            raise KeyError("val")
        num_kv_heads = k_info.shape[2]
        k_head_dim = k_info.shape[3]
        v_head_dim = v_info.shape[3]
"""
    for attn_path in (triton_attn_path, torch_attn_path):
        if not attn_path.exists():
            continue
        text = attn_path.read_text()
        updated_text = text
        for source_snippet in (
            legacy_attn_meta,
            broken_attn_meta,
            updated_attn_meta,
            recursive_attn_meta_v0,
            recursive_attn_meta_v1,
        ):
            if source_snippet in updated_text:
                updated_text = updated_text.replace(source_snippet, recursive_attn_meta, 1)
        updated_text = updated_text.replace("k_fake.dtype", "k_info.dtype")
        updated_text = updated_text.replace("v_fake.dtype", "v_info.dtype")
        if updated_text != text:
            attn_path.write_text(updated_text)
            print(f"Patched {attn_path} to tolerate tensor_meta-only attention nodes.")
PY

# Model prep generates the templated prompt and token counts that drive the runtime YAML budget.
bash "$WORKFLOW_ROOT/02_prepare_model.sh"

workflow_log "Selecting and warming the first feasible Gemma 4 AutoDeploy runtime."

runtime_prepare_log="$(mktemp)"

# Prefer the shared helper first so this script stays aligned with the rest of
# the workflow when the installed TensorRT-LLM build accepts the generated YAML.
if "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" prepare-runtime "$@" 2>&1 | tee "$runtime_prepare_log"; then
  rm -f "$runtime_prepare_log"
  exit 0
fi

if ! grep -q 'cuda_graph_config' "$runtime_prepare_log"; then
  workflow_log "Runtime preparation failed for a reason other than the known AutoDeploy YAML schema mismatch."
  rm -f "$runtime_prepare_log"
  exit 1
fi

workflow_log "Installed AutoDeploy YAML rejects cuda_graph_config. Retrying with sanitized runtime YAML files."

# The installed TensorRT-LLM 1.2.0 AutoDeploy path parses a narrower YAML
# schema than the current helper expects. Retry runtime selection with the same
# probing logic, but remove the unsupported cuda_graph_config block before the
# warmup probe.
fallback_runtime_script="$(mktemp "$RUNTIME_DIR/prepare_runtime_fallback.XXXXXX.py")"

cat >"$fallback_runtime_script" <<'PY'
import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path

import yaml

import gemma4_workflow as gw


def main() -> int:
    force = "--force" in sys.argv[1:]

    gw.ensure_dirs()
    config = gw.load_run_config()

    selected = config.get("selected_runtime")
    if (
        selected
        and not force
        and Path(selected.get("yaml_path", "")).exists()
        and Path(selected.get("prompt_path", "")).exists()
    ):
        gw.log("Runtime preparation already completed; reusing existing selection.")
        return 0

    if "prepared_models" not in config:
        gw.prepare_model_command(argparse.Namespace())
        config = gw.load_run_config()

    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

    failures = []
    precision_support = config["environment"]["precision_support"]
    gpu_memory_text = str(config.get("environment", {}).get("gpu", {}).get("memory_total", ""))
    gpu_memory_match = re.search(r"(\d+)", gpu_memory_text)
    gpu_memory_mib = int(gpu_memory_match.group(1)) if gpu_memory_match else 0
    backend_descriptors = {}
    available_backends = []
    for candidate_backend in ("triton_paged", "flashinfer", "triton", "torch"):
        if not AttentionRegistry.has(candidate_backend):
            continue
        descriptor = AttentionRegistry.get(candidate_backend)()
        backend_descriptors[candidate_backend] = descriptor
        available_backends.append(
            {"name": candidate_backend, "is_paged": bool(descriptor.is_paged())}
        )

    attention_strategies = []
    for candidate_backend in ("torch", "triton"):
        if candidate_backend in backend_descriptors:
            attention_strategies.append(
                {"attn_backend": candidate_backend, "mode": "non_paged"}
            )
    for candidate_backend in ("triton_paged", "flashinfer"):
        descriptor = backend_descriptors.get(candidate_backend)
        if descriptor is not None and descriptor.is_paged():
            attention_strategies.append(
                {"attn_backend": candidate_backend, "mode": "paged"}
            )

    if not attention_strategies:
        config["probe_failures"] = failures
        config["prepare_runtime_fallback"] = {
            "used": True,
            "reason": "Installed AutoDeploy YAML parser rejected cuda_graph_config, and no usable attention backend was available for sanitized probing.",
            "available_attention_backends": available_backends,
            "attention_strategies": attention_strategies,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        gw.save_run_config(config)
        gw.fail(
            "No usable attention backend is available in this TensorRT-LLM build. "
            f"Observed backends: {available_backends}"
        )

    def compile_strategies_for(attention_strategy):
        backend_name = attention_strategy["attn_backend"]
        if backend_name == "torch":
            return [
                {
                    "compile_backend": "torch-simple",
                    "cuda_graph_batch_sizes": None,
                },
                {
                    "compile_backend": "torch-cudagraph",
                    "cuda_graph_batch_sizes": [gw.DEFAULT_BATCH_SIZE],
                },
            ]
        return [
            {
                "compile_backend": "torch-simple",
                "cuda_graph_batch_sizes": None,
            }
        ]

    for candidate in gw.MODEL_CANDIDATES:
        model_id = candidate["model_id"]
        if model_id != gw.REQUESTED_MODEL_ID and gpu_memory_mib and gpu_memory_mib < 48 * 1024:
            for precision in gw.PRECISION_ORDER:
                gw.record_failure(
                    failures,
                    model_id,
                    precision,
                    (
                        f"Skipped on this {gpu_memory_mib} MiB GPU because the larger Gemma 4 family "
                        "fallback models are not feasible to warm in this non-quantized workflow."
                    ),
                )
            continue

        prompt_meta = gw.locate_prompt_metadata(config, model_id)
        prompt_path = Path(prompt_meta["templated_prompt_path"])
        prompt_text = prompt_path.read_text()
        prompt_tokens = int(prompt_meta["input_token_count"])

        for precision in gw.PRECISION_ORDER:
            support = precision_support.get(precision, {})
            if not support.get("supported"):
                gw.record_failure(
                    failures,
                    model_id,
                    precision,
                    support.get("reason", "Unsupported precision."),
                )
                continue

            for attention_strategy in attention_strategies:
                for compile_strategy in compile_strategies_for(attention_strategy):
                    yaml_origin, yaml_payload = gw.embedded_yaml_template(
                        model_id, prompt_tokens, gw.DEFAULT_MAX_NEW_TOKENS
                    )
                    yaml_payload.pop("cuda_graph_config", None)
                    yaml_payload.pop("transforms", None)
                    yaml_payload["compile_backend"] = compile_strategy["compile_backend"]
                    if compile_strategy["cuda_graph_batch_sizes"] is None:
                        yaml_payload.pop("cuda_graph_batch_sizes", None)
                    else:
                        yaml_payload["cuda_graph_batch_sizes"] = compile_strategy[
                            "cuda_graph_batch_sizes"
                        ]
                    yaml_payload["attn_backend"] = attention_strategy["attn_backend"]
                    if attention_strategy["mode"] == "non_paged":
                        yaml_payload["attn_page_size"] = yaml_payload["max_seq_len"]
                    yaml_payload["transforms"] = {
                        "fuse_rmsnorm": {
                            "stage": "post_load_fusion",
                            "rmsnorm_backend": "triton",
                            "gated_rmsnorm_backend": "triton",
                        },
                        "fuse_add_rms_norm": {
                            "stage": "post_load_fusion",
                            "enabled": False,
                        },
                    }
                    yaml_file = gw.yaml_path_for(
                        model_id,
                        (
                            f"{precision.lower()}__{attention_strategy['attn_backend']}_{attention_strategy['mode']}"
                            f"__{compile_strategy['compile_backend'].replace('-', '_')}"
                        ),
                    )
                    yaml_file.write_text(yaml.safe_dump(yaml_payload, sort_keys=False))

                    gw.log(
                        "Trying sanitized runtime probe for "
                        f"{model_id} / {precision} / {attention_strategy['attn_backend']} "
                        f"({attention_strategy['mode']}) / {compile_strategy['compile_backend']}"
                    )

                    try:
                        probe = gw.attempt_runtime_probe(
                            model_id=model_id,
                            precision=precision,
                            prompt_text=prompt_text,
                            yaml_path=yaml_file,
                            max_new_tokens=1,
                        )
                    except Exception as exc:
                        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                        gw.record_failure(
                            failures,
                            model_id,
                            precision,
                            (
                                f"[attn_backend={attention_strategy['attn_backend']}, "
                                f"mode={attention_strategy['mode']}, "
                                f"compile_backend={compile_strategy['compile_backend']}] {tb}"
                            ),
                        )
                        continue

                    selected_runtime = {
                        "requested_model_id": gw.REQUESTED_MODEL_ID,
                        "actual_model_id": model_id,
                        "precision": precision,
                        "dtype": gw.dtype_for_precision(precision),
                        "workflow": "TensorRT-LLM AutoDeploy text-only",
                        "compile_backend": compile_strategy["compile_backend"],
                        "attention_backend": attention_strategy["attn_backend"],
                        "attention_mode": attention_strategy["mode"],
                        "attn_page_size": yaml_payload.get("attn_page_size"),
                        "yaml_path": str(yaml_file),
                        "yaml_origin": f"{yaml_origin}-without-cuda-graph-config-or-modern-transform-overrides",
                        "prompt_path": str(prompt_path),
                        "input_token_count": prompt_tokens,
                        "max_new_tokens": gw.DEFAULT_MAX_NEW_TOKENS,
                        "batch_size": gw.DEFAULT_BATCH_SIZE,
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

                    if model_id != gw.REQUESTED_MODEL_ID:
                        config.setdefault("fallback_decisions", []).append(
                            {
                                "from_model_id": gw.REQUESTED_MODEL_ID,
                                "to_model_id": model_id,
                                "reason": (
                                    "The requested model did not complete a successful support/feasibility "
                                    "probe before this family variant."
                                ),
                            }
                        )

                    config["selected_runtime"] = selected_runtime
                    config["probe_failures"] = failures
                    config["prepare_runtime_fallback"] = {
                        "used": True,
                        "reason": "Installed AutoDeploy YAML parser rejected cuda_graph_config.",
                        "attention_backend": attention_strategy["attn_backend"],
                        "attention_mode": attention_strategy["mode"],
                        "compile_backend": compile_strategy["compile_backend"],
                        "available_attention_backends": available_backends,
                        "attention_strategies": attention_strategies,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    gw.save_run_config(config)
                    gw.log(
                        f"Selected {model_id} with {precision} using "
                        f"{attention_strategy['attn_backend']} ({attention_strategy['mode']}) "
                        f"and {compile_strategy['compile_backend']} after a sanitized AutoDeploy probe."
                    )
                    return 0

    config["probe_failures"] = failures
    config["prepare_runtime_fallback"] = {
        "used": True,
        "reason": "Installed AutoDeploy YAML parser rejected cuda_graph_config, and all sanitized probes failed.",
        "available_attention_backends": available_backends,
        "attention_strategies": attention_strategies,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    gw.save_run_config(config)
    gw.fail("No supported Gemma 4 model/precision combination completed a successful sanitized probe.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
PY

"$PYTHON_BIN" "$fallback_runtime_script" "$@"

rm -f "$fallback_runtime_script"
rm -f "$runtime_prepare_log"
