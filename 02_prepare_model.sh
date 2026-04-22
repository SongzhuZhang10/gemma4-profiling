#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

# Step 01 is responsible for provisioning the Python environment. Step 02 only
# validates that preflight metadata exists, then prepares prompt artifacts in
# the repo-local runtime directory.
if ! WORKFLOW_RUN_CONFIG_JSON="$RUN_CONFIG_JSON" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

run_config_path = Path(os.environ["WORKFLOW_RUN_CONFIG_JSON"])
if not run_config_path.exists():
    raise SystemExit(1)

try:
    payload = json.loads(run_config_path.read_text())
except json.JSONDecodeError:
    raise SystemExit(1)

raise SystemExit(0 if "environment" in payload else 1)
PY
then
  workflow_log "Run configuration is missing environment preflight data. Rebuilding it with gemma4_workflow.py preflight."
  "$PYTHON_BIN" "$WORKFLOW_ROOT/gemma4_workflow.py" preflight
else
  workflow_log "Reusing preflight environment metadata from $RUN_CONFIG_JSON."
fi

workflow_log "Preparing templated prompt artifacts for all Gemma 4 candidates on the DGX server."

# On this DGX stack, Gemma 4 prompt templating works reliably when we render
# directly from tokenizer.json plus chat_template.jinja. AutoProcessor and
# AutoTokenizer do not currently load these assets correctly for the configured
# Gemma 4 repositories, so this is the primary preparation path.
"$PYTHON_BIN" - <<'PY'
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

import gemma4_workflow as gw

PRIMARY_BUNDLE_TYPE = "tokenizer_fast_repo_assets"


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


def count_tokens(tokenizer: PreTrainedTokenizerFast, prompt_text: str) -> int:
    tokenized = tokenizer(prompt_text, return_tensors="pt")
    return int(tokenized["input_ids"].shape[-1])


def first_non_empty(*values):
    for value in values:
        if value is not None:
            return value
    return None


def normalize_prompt_metadata(
    *,
    model_id: str,
    prompt_path: Path,
    prompt_meta_path: Path,
    input_token_count: int,
    prompt_text: str,
    max_new_tokens: int,
) -> dict:
    return {
        "model_id": model_id,
        "bundle_type": PRIMARY_BUNDLE_TYPE,
        "input_token_count": int(input_token_count),
        "templated_prompt_path": str(prompt_path),
        "prompt_text": prompt_text,
        "max_new_tokens": int(max_new_tokens),
        "templated_prompt_metadata_path": str(prompt_meta_path),
    }


gw.ensure_dirs()
config = gw.load_run_config()

prepared = config.setdefault("prepared_models", {})
candidate_models = config.get("candidate_models", gw.MODEL_CANDIDATES)
prepared_at = time.strftime("%Y-%m-%d %H:%M:%S")

for candidate in candidate_models:
    model_id = candidate["model_id"]
    existing = prepared.get(model_id)

    model_dir = gw.RUNTIME_DIR / gw.slugify_model_id(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = Path(
        first_non_empty(
            (existing or {}).get("templated_prompt_path"),
            model_dir / "templated_prompt.txt",
        )
    )
    prompt_meta_path = Path(
        first_non_empty(
            (existing or {}).get("templated_prompt_metadata_path"),
            model_dir / "templated_prompt.json",
        )
    )

    persisted_meta = gw.read_json(prompt_meta_path)
    prompt_text = first_non_empty(
        (existing or {}).get("prompt_text"),
        persisted_meta.get("prompt_text"),
        gw.DEFAULT_PROMPT,
    )
    max_new_tokens = first_non_empty(
        (existing or {}).get("max_new_tokens"),
        persisted_meta.get("max_new_tokens"),
        gw.DEFAULT_MAX_NEW_TOKENS,
    )

    if prompt_path.exists():
        gw.log(f"Reusing existing templated prompt artifact for {model_id}")
        token_count = first_non_empty(
            (existing or {}).get("input_token_count"),
            persisted_meta.get("input_token_count"),
        )
        if token_count is None:
            gw.log(
                f"Existing prompt metadata for {model_id} is missing input_token_count; "
                "reconstructing it with the repository tokenizer assets."
            )
            tokenizer = build_tokenizer(model_id)
            token_count = count_tokens(tokenizer, prompt_path.read_text())
    else:
        gw.log(f"Preparing prompt assets for {model_id} from repository tokenizer files")
        tokenizer = build_tokenizer(model_id)
        templated_prompt, token_count = render_prompt(tokenizer)
        prompt_path.write_text(templated_prompt)

    prompt_meta = normalize_prompt_metadata(
        model_id=model_id,
        prompt_path=prompt_path,
        prompt_meta_path=prompt_meta_path,
        input_token_count=token_count,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
    )
    gw.write_json(prompt_meta_path, prompt_meta)
    prepared[model_id] = prompt_meta

config["prepared_at"] = prepared_at
config["prepare_model_strategy"] = {
    "bundle_type": PRIMARY_BUNDLE_TYPE,
    "mode": "dgx_tokenizer_repo_assets",
    "reason": (
        "Gemma 4 prompt preparation on the DGX server renders chat prompts from "
        "tokenizer.json, tokenizer_config.json, and chat_template.jinja because "
        "AutoProcessor and AutoTokenizer do not currently load these assets "
        "reliably for the configured repositories."
    ),
    "reuses_existing_prompt_artifacts": True,
    "timestamp": prepared_at,
}
config.pop("prepare_model_fallback", None)
gw.save_run_config(config)
gw.log("Model preparation completed successfully with the DGX tokenizer-asset strategy.")
PY
