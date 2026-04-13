#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# Reuse the install/preflight step so prompt preparation always runs against a validated environment.
bash "$WORKFLOW_ROOT/01_install.sh"

workflow_log "Preparing templated prompt artifacts for all Gemma 4 candidates."

prepare_model_log="$(mktemp)"

# Prefer the shared helper implementation first; it keeps this script aligned
# with the rest of the workflow when the installed transformers stack can load
# the Gemma 4 processor/tokenizer classes directly.
if "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" prepare-model 2>&1 | tee "$prepare_model_log"; then
  rm -f "$prepare_model_log"
  exit 0
fi

workflow_log "Default prompt-preparation path failed. Falling back to a tokenizer-only Gemma 4 prompt builder."

# Gemma 4's tokenizer metadata currently trips AutoTokenizer/AutoProcessor in
# this environment, so we build the templated prompt directly from
# tokenizer.json plus chat_template.jinja and write the same artifacts that the
# shared helper normally produces.
"$PYTHON_BIN" - <<'PY'
import argparse
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

import gemma4_workflow as gw


def unique_preserving_order(values):
    result = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_tokenizer(model_id: str) -> PreTrainedTokenizerFast:
    tokenizer_json = hf_hub_download(model_id, "tokenizer.json", repo_type="model")
    tokenizer_config = hf_hub_download(model_id, "tokenizer_config.json", repo_type="model")
    chat_template_path = hf_hub_download(model_id, "chat_template.jinja", repo_type="model")

    config = json.loads(Path(tokenizer_config).read_text())
    chat_template = Path(chat_template_path).read_text()

    special_token_candidates = list(config.get("extra_special_tokens") or [])
    special_token_candidates.extend(
        [
            config.get("image_token"),
            config.get("audio_token"),
            config.get("boi_token"),
            config.get("eoi_token"),
            config.get("boa_token"),
            config.get("eoa_token"),
            config.get("sot_token"),
            config.get("eot_token"),
            config.get("soc_token"),
            config.get("eoc_token"),
            config.get("stc_token"),
            config.get("std_token"),
            config.get("str_token"),
            config.get("etc_token"),
            config.get("etd_token"),
            config.get("etr_token"),
            config.get("think_token"),
        ]
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        bos_token=config.get("bos_token"),
        eos_token=config.get("eos_token"),
        unk_token=config.get("unk_token"),
        pad_token=config.get("pad_token"),
        mask_token=config.get("mask_token"),
        padding_side=config.get("padding_side", "left"),
        chat_template=chat_template,
        additional_special_tokens=unique_preserving_order(special_token_candidates),
    )
    return tokenizer


def render_prompt(tokenizer: PreTrainedTokenizerFast) -> tuple[str, int]:
    messages = [{"role": "user", "content": gw.DEFAULT_PROMPT}]
    try:
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    tokenized = tokenizer(templated, return_tensors="pt")
    token_count = int(tokenized["input_ids"].shape[-1])
    return templated, token_count


gw.ensure_dirs()
config = gw.load_run_config()
if "environment" not in config:
    gw.preflight_command(argparse.Namespace())
    config = gw.load_run_config()

prepared = config.setdefault("prepared_models", {})
candidate_models = config.get("candidate_models", gw.MODEL_CANDIDATES)

for candidate in candidate_models:
    model_id = candidate["model_id"]
    existing = prepared.get(model_id)
    if existing and Path(existing["templated_prompt_path"]).exists():
        continue

    gw.log(f"Fallback prompt preparation for {model_id}")
    tokenizer = build_tokenizer(model_id)
    templated_prompt, token_count = render_prompt(tokenizer)

    model_dir = gw.RUNTIME_DIR / gw.slugify_model_id(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = model_dir / "templated_prompt.txt"
    prompt_meta_path = model_dir / "templated_prompt.json"

    prompt_path.write_text(templated_prompt)
    prompt_meta = {
        "model_id": model_id,
        "bundle_type": "tokenizer_fast_fallback",
        "input_token_count": token_count,
        "templated_prompt_path": str(prompt_path),
        "prompt_text": gw.DEFAULT_PROMPT,
        "max_new_tokens": gw.DEFAULT_MAX_NEW_TOKENS,
        "templated_prompt_metadata_path": str(prompt_meta_path),
    }
    gw.write_json(prompt_meta_path, prompt_meta)
    prepared[model_id] = prompt_meta

config["prepared_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
config["prepare_model_fallback"] = {
    "used": True,
    "reason": "AutoProcessor/AutoTokenizer could not load Gemma 4 templating assets in the current transformers build.",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
gw.save_run_config(config)
gw.log("Fallback model preparation completed successfully.")
PY

rm -f "$prepare_model_log"
