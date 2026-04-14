#!/usr/bin/env python3
"""Single-process HuggingFace inference fallback for Nsight Compute."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parent
RUN_CONFIG_PATH = ROOT_DIR / "artifacts" / "run_config.json"

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_PROMPT = "How does a large language model work?"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_HIDDEN_SIZE = 1536
DEFAULT_INTERMEDIATE_SIZE = 6144
DEFAULT_LAYER_COUNT = 35
DEFAULT_DECODE_INPUT_TOKENS = 17
PREFILL_TOKEN_TARGET = 512
PREFILL_WARMUP_PASSES = 3
PREFILL_PROFILED_PASSES = 3
DECODE_WARMUP_PASSES = 1
DECODE_PROFILED_PASSES = 1


class GemmaFFNProxy(nn.Module):
    """Memory-light structural proxy that preserves Gemma FFN dimensions."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).split(self.down.in_features, dim=-1)
        return self.down(F.silu(gate) * up)


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_run_settings() -> Tuple[Dict[str, Any], str, str]:
    config = read_json(RUN_CONFIG_PATH)
    model_id = str(config.get("requested_model_id") or DEFAULT_MODEL_ID)

    workload = config.get("workload")
    prompt = workload.get("prompt") if isinstance(workload, dict) else None
    if not prompt:
        prepared = config.get("prepared_models", {})
        prepared_metadata = prepared.get(model_id) if isinstance(prepared, dict) else None
        if isinstance(prepared_metadata, dict):
            prompt = prepared_metadata.get("prompt_text")

    return config, model_id, str(prompt or DEFAULT_PROMPT)


def load_text_config(model_id: str) -> Tuple[int, int, int]:
    try:
        from huggingface_hub import hf_hub_download

        config_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="config.json",
            )
        )
        config = read_json(config_path)
        text_config = config.get("text_config") if isinstance(config, dict) else None
        if not isinstance(text_config, dict):
            raise ValueError("config.json is missing a text_config object")

        hidden_size = int(text_config.get("hidden_size", DEFAULT_HIDDEN_SIZE))
        intermediate_size = int(
            text_config.get("intermediate_size", DEFAULT_INTERMEDIATE_SIZE)
        )
        layer_count = int(text_config.get("num_hidden_layers", DEFAULT_LAYER_COUNT))
        return hidden_size, intermediate_size, layer_count
    except Exception as exc:
        log(
            "Falling back to default Gemma-4-E2B dimensions because config.json "
            f"could not be loaded: {exc}"
        )
        return DEFAULT_HIDDEN_SIZE, DEFAULT_INTERMEDIATE_SIZE, DEFAULT_LAYER_COUNT


def prepared_prompt_token_count(config: Dict[str, Any], model_id: str) -> int:
    prepared = config.get("prepared_models", {})
    prepared_metadata = prepared.get(model_id) if isinstance(prepared, dict) else None
    if not isinstance(prepared_metadata, dict):
        return DEFAULT_DECODE_INPUT_TOKENS

    try:
        token_count = int(prepared_metadata.get("input_token_count", DEFAULT_DECODE_INPUT_TOKENS))
    except (TypeError, ValueError):
        return DEFAULT_DECODE_INPUT_TOKENS
    return token_count if token_count > 0 else DEFAULT_DECODE_INPUT_TOKENS


def try_load_text_bundle(model_id: str) -> Tuple[str | None, Any | None]:
    from transformers import AutoProcessor, AutoTokenizer

    errors = []
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

    log(
        "Falling back to structural proxy because text assets could not be loaded: "
        + "; ".join(errors)
    )
    return None, None


def try_load_real_model(model_id: str):
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        log(f"Loaded real HuggingFace model for direct inference: {model_id}")
        return model
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"Falling back to structural proxy after model load failed: {exc}")
        return None


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Nsight Compute direct inference requires a CUDA GPU.")


def model_input_device(model: nn.Module) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
            if isinstance(device, torch.device) and device.type not in {"cpu", "meta"}:
                return device

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0")


def move_tensor_inputs(
    tokenized: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in tokenized.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def tokenize_text_bundle(
    bundle_type: str,
    bundle: Any,
    text: str,
    *,
    max_length: int | None = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})

    if bundle_type == "processor":
        try:
            return bundle(text=text, **kwargs)
        except TypeError:
            tokenizer = getattr(bundle, "tokenizer", None)
            if tokenizer is None:
                raise
            return tokenizer(text, **kwargs)

    return bundle(text, **kwargs)


def build_prefill_text(bundle_type: str, bundle: Any, prompt: str) -> str:
    base_prompt = prompt.strip() or DEFAULT_PROMPT
    repeated = base_prompt

    while True:
        tokenized = tokenize_text_bundle(bundle_type, bundle, repeated)
        token_count = int(tokenized["input_ids"].shape[-1])
        if token_count >= PREFILL_TOKEN_TARGET:
            return repeated
        repeated = f"{repeated} {base_prompt}"


def run_proxy_prefill(
    input_token_count: int,
    hidden_size: int,
    intermediate_size: int,
    layer_count: int,
) -> None:
    device = torch.device("cuda:0")
    layer = GemmaFFNProxy(hidden_size, intermediate_size).to(device=device, dtype=torch.float16)
    layer.eval()
    x_seed = torch.randn(1, input_token_count, hidden_size, device=device, dtype=torch.float16)

    with torch.inference_mode():
        for _ in range(PREFILL_WARMUP_PASSES):
            x = x_seed
            for _ in range(layer_count):
                x = layer(x)
        synchronize_cuda()

        for _ in range(PREFILL_PROFILED_PASSES):
            x = x_seed
            for _ in range(layer_count):
                x = layer(x)
        synchronize_cuda()


def run_proxy_decode(
    max_new_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    layer_count: int,
) -> None:
    device = torch.device("cuda:0")
    layer = GemmaFFNProxy(hidden_size, intermediate_size).to(device=device, dtype=torch.float16)
    layer.eval()
    x_single_token = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float16)

    with torch.inference_mode():
        for _ in range(DECODE_WARMUP_PASSES):
            for _step in range(max_new_tokens):
                x = x_single_token
                for _ in range(layer_count):
                    x = layer(x)
        synchronize_cuda()

        for _ in range(DECODE_PROFILED_PASSES):
            for _step in range(max_new_tokens):
                x = x_single_token
                for _ in range(layer_count):
                    x = layer(x)
        synchronize_cuda()


def run_real_prefill(model: nn.Module, inputs: Dict[str, Any]) -> None:
    with torch.inference_mode():
        for _ in range(PREFILL_WARMUP_PASSES):
            model(**inputs, use_cache=False)
        synchronize_cuda()

        for _ in range(PREFILL_PROFILED_PASSES):
            model(**inputs, use_cache=False)
        synchronize_cuda()


def run_real_decode(model: nn.Module, inputs: Dict[str, Any], max_new_tokens: int) -> None:
    with torch.inference_mode():
        for _ in range(DECODE_WARMUP_PASSES):
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        synchronize_cuda()

        for _ in range(DECODE_PROFILED_PASSES):
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        synchronize_cuda()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-process direct inference for Nsight Compute fallback profiling."
    )
    parser.add_argument("--phase", choices=["prefill", "decode"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--metadata-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    ensure_cuda_available()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    config, model_id, prompt = load_run_settings()
    hidden_size, intermediate_size, layer_count = load_text_config(model_id)
    model = try_load_real_model(model_id)
    bundle_type, bundle = try_load_text_bundle(model_id)
    using_proxy = model is None or bundle is None
    started_at = time.time()
    actual_output_token_count = 0

    if using_proxy and model is not None:
        del model
        torch.cuda.empty_cache()
        model = None

    if args.phase == "prefill":
        if bundle is not None and bundle_type is not None:
            prefill_text = build_prefill_text(bundle_type, bundle, prompt)
            tokenized = tokenize_text_bundle(
                bundle_type,
                bundle,
                prefill_text,
                max_length=PREFILL_TOKEN_TARGET,
            )
            input_token_count = int(tokenized["input_ids"].shape[-1])
        else:
            tokenized = None
            input_token_count = PREFILL_TOKEN_TARGET

        if using_proxy:
            log(
                "Running structural proxy for prefill "
                f"(T={input_token_count}, layers={layer_count}, d={hidden_size}, ffn={intermediate_size})."
            )
            run_proxy_prefill(input_token_count, hidden_size, intermediate_size, layer_count)
        else:
            log(f"Running real-model prefill with {input_token_count} input tokens.")
            inputs = move_tensor_inputs(tokenized, model_input_device(model))
            run_real_prefill(model, inputs)
    else:
        decode_max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS
        if bundle is not None and bundle_type is not None:
            tokenized = tokenize_text_bundle(bundle_type, bundle, prompt)
            input_token_count = int(tokenized["input_ids"].shape[-1])
        else:
            tokenized = None
            input_token_count = prepared_prompt_token_count(config, model_id)

        if using_proxy:
            log(
                "Running structural proxy for decode "
                f"(T={input_token_count}, steps={decode_max_new_tokens}, layers={layer_count})."
            )
            run_proxy_decode(decode_max_new_tokens, hidden_size, intermediate_size, layer_count)
        else:
            log(
                "Running real-model decode with "
                f"{input_token_count} input tokens and {decode_max_new_tokens} generated tokens."
            )
            inputs = move_tensor_inputs(tokenized, model_input_device(model))
            run_real_decode(model, inputs, decode_max_new_tokens)
        actual_output_token_count = decode_max_new_tokens

    if args.metadata_output:
        requested_max_new_tokens = 0 if args.phase == "prefill" else decode_max_new_tokens
        write_json(
            Path(args.metadata_output),
            {
                "actual_output_token_count": actual_output_token_count,
                "collection_backend": "direct_hf",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "phase": args.phase,
                "model_id": model_id,
                "input_token_count": input_token_count,
                "max_new_tokens": requested_max_new_tokens,
                "requested_max_new_tokens": requested_max_new_tokens,
                "direct_inference": True,
                "proxy_fallback": using_proxy,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        log(f"Wrote direct-inference metadata to {args.metadata_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
