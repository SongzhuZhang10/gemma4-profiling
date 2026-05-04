#!/usr/bin/env python3
"""TensorRT Edge-LLM workflow entrypoints for Llama 3.1 8B on DGX Spark / Thor."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
EXPORT_DIR = ARTIFACTS_DIR / "export"
EXPORT_BUNDLE_DIR = ARTIFACTS_DIR / "export_bundle"
RUNTIME_DIR = ARTIFACTS_DIR / "runtime"
INPUTS_DIR = RUNTIME_DIR / "inputs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RUN_CONFIG_PATH = ARTIFACTS_DIR / "run_config.json"

WORKFLOW_BACKEND = "tensorrt_edge_llm"
REQUESTED_MODEL_ID = os.environ.get(
    "EDGE_LLM_MODEL_ID",
    "RedHatAI/Llama-3.1-8B-Instruct",
)
WORKFLOW_ID = f"{REQUESTED_MODEL_ID.replace('/', '__').replace('.', '').lower()}_edgellm_dgxspark_fp16_v1"
EXPORT_PRECISION = "FP16"
DEFAULT_PROMPT = "How does a large language model work?"
MODEL_SLUG = REQUESTED_MODEL_ID.replace("/", "__")
EDGE_LLM_GIT_URL = "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"

ENGINE_BUILD_DEFAULTS = {
    "maxBatchSize": 1,
    "maxInputLen": 1280,
    "maxKVCacheCapacity": 4096,
    "precision": EXPORT_PRECISION,
}

PHASE_WORKLOAD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "prefill": {
        "batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_generate_length": 1,
        "target_input_tokens": 1024,
        "prompt_kind": "long_prefill",
    },
    "decode": {
        "batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_generate_length": 128,
        "target_input_tokens": 1024,
        "prompt_kind": "long_decode_context",
    },
}

BENCHMARK_REPEATS_DEFAULT = 5
BENCHMARK_WARMUP_RUNS_DEFAULT = 1
RUNTIME_PROFILE_STAGE_IDS = ("llm_prefill", "llm_generation")

REQUIRED_TOKENIZER_FILES = ("config.json", "tokenizer_config.json")
OPTIONAL_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
)

GEMM_KERNEL_TOKENS = ("gemm", "gemv", "xmma", "mma", "cublas", "cutlass")
ATTENTION_KERNEL_TOKENS = ("attention", "flash")
GATHER_KERNEL_TOKENS = ("gather", "scatter")
SAMPLING_KERNEL_TOKENS = (
    "distribution_elementwise_grid_stride_kernel",
    "distribution_nullary_kernel",
    "philox",
    "curand",
    "normal_kernel",
)
COPY_KERNEL_TOKENS = (
    "vectorized_elementwise_kernel",
    "copy_kernel",
    "copy_kernel_cuda",
    "bfloat16_copy_kernel_cuda",
)

NCU_SLICE_METRICS = ",".join(
    (
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    )
)

NCU_METRIC_ALIASES = {
    "gpu__time_duration.sum": "duration_ns",
    "Duration": "duration_ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "Compute (SM) Throughput": "sm_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "DRAM Throughput": "dram_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "warps_pct",
    "Achieved Occupancy": "warps_pct",
}

NCU_FAMILIES = (
    "gemm_like",
    "attention_like",
    "gather_like",
    "copy_like",
    "sampling_like",
    "other",
)


class WorkflowError(RuntimeError):
    """Raised when a workflow step fails in an expected way."""


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


def fail(message: str) -> None:
    raise WorkflowError(message)


def ensure_dirs() -> None:
    for path in (
        ARTIFACTS_DIR,
        CACHE_DIR,
        EXPORT_DIR,
        EXPORT_BUNDLE_DIR,
        RUNTIME_DIR,
        INPUTS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    cache_env_dirs = [
        Path(os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))),
        Path(
            os.environ.setdefault(
                "HUGGINGFACE_HUB_CACHE",
                str(CACHE_DIR / "hf" / "hub"),
            )
        ),
        Path(os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))),
    ]

    for path in cache_env_dirs:
        path.mkdir(parents=True, exist_ok=True)


def run(
    cmd: Sequence[str],
    *,
    capture_output: bool = False,
    check: bool = True,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    log(
        "Running: "
        + " ".join(cmd)
        + (f" (cwd={cwd})" if cwd is not None else "")
    )
    return subprocess.run(
        list(cmd),
        check=check,
        text=True,
        capture_output=capture_output,
        cwd=str(cwd) if cwd is not None else None,
        env=env or os.environ.copy(),
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_path(name: str) -> Optional[str]:
    return shutil.which(name)


def default_edge_llm_repo_root() -> Path:
    env_value = os.environ.get("TENSORRT_EDGE_LLM_ROOT") or os.environ.get(
        "EDGE_LLM_REPO_DIR"
    )
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path.home() / "TensorRT-Edge-LLM").resolve()


def deep_merge_in_place(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overlay.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge_in_place(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def default_run_config() -> Dict[str, Any]:
    return {
        "workflow_backend": WORKFLOW_BACKEND,
        "workflow_id": WORKFLOW_ID,
        "requested_model_id": REQUESTED_MODEL_ID,
        "export_precision": EXPORT_PRECISION,
        "workload": {
            "prompt": DEFAULT_PROMPT,
        },
        "phase_workloads": copy.deepcopy(PHASE_WORKLOAD_DEFAULTS),
        "engine_build_config": copy.deepcopy(ENGINE_BUILD_DEFAULTS),
        "artifacts": {
            "artifacts_dir": str(ARTIFACTS_DIR),
            "cache_dir": str(CACHE_DIR),
            "export_dir": str(EXPORT_DIR),
            "export_bundle_dir": str(EXPORT_BUNDLE_DIR),
            "runtime_dir": str(RUNTIME_DIR),
            "inputs_dir": str(INPUTS_DIR),
            "reports_dir": str(REPORTS_DIR),
            "run_config_json": str(RUN_CONFIG_PATH),
        },
        "edge_llm": {
            "repo_root": str(default_edge_llm_repo_root()),
        },
    }


def stale_workflow_state(existing: Dict[str, Any]) -> bool:
    if not existing:
        return False

    stored_backend = existing.get("workflow_backend")
    stored_workflow_id = existing.get("workflow_id")
    stored_model_id = existing.get("requested_model_id")

    return bool(
        stored_backend != WORKFLOW_BACKEND
        or stored_workflow_id != WORKFLOW_ID
        or stored_model_id != REQUESTED_MODEL_ID
    )


def normalize_phase_workloads(config: Dict[str, Any]) -> None:
    phase_workloads = config.get("phase_workloads")
    if not isinstance(phase_workloads, dict):
        phase_workloads = {}

    normalized = copy.deepcopy(PHASE_WORKLOAD_DEFAULTS)
    for phase, payload in phase_workloads.items():
        if phase not in normalized or not isinstance(payload, dict):
            continue
        deep_merge_in_place(normalized[phase], payload)

    decode_payload = normalized.get("decode", {})
    if (
        isinstance(decode_payload, dict)
        and int(decode_payload.get("target_input_tokens", 0) or 0) > 0
        and decode_payload.get("prompt_kind") == "prepared_prompt"
    ):
        decode_payload["prompt_kind"] = "long_decode_context"
    config["phase_workloads"] = normalized


def load_run_config() -> Dict[str, Any]:
    ensure_dirs()
    existing = read_json(RUN_CONFIG_PATH)
    base = default_run_config()

    if stale_workflow_state(existing):
        preserved_workload = existing.get("workload") if isinstance(existing.get("workload"), dict) else {}
        preserved_phase_workloads = (
            existing.get("phase_workloads")
            if isinstance(existing.get("phase_workloads"), dict)
            else {}
        )
        preserved_artifacts = (
            existing.get("artifacts") if isinstance(existing.get("artifacts"), dict) else {}
        )
        preserved_edge_llm = (
            existing.get("edge_llm") if isinstance(existing.get("edge_llm"), dict) else {}
        )

        deep_merge_in_place(base["workload"], preserved_workload)
        deep_merge_in_place(base["phase_workloads"], preserved_phase_workloads)
        deep_merge_in_place(base["artifacts"], preserved_artifacts)
        deep_merge_in_place(base["edge_llm"], preserved_edge_llm)
        base["migration"] = {
            "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "previous_workflow_backend": existing.get("workflow_backend"),
            "previous_workflow_id": existing.get("workflow_id"),
            "previous_requested_model_id": existing.get("requested_model_id"),
        }
        config = base
    else:
        config = copy.deepcopy(base)
        deep_merge_in_place(config, existing)

    config["workflow_backend"] = WORKFLOW_BACKEND
    config["workflow_id"] = WORKFLOW_ID
    config["requested_model_id"] = REQUESTED_MODEL_ID
    config["export_precision"] = EXPORT_PRECISION

    deep_merge_in_place(config["engine_build_config"], ENGINE_BUILD_DEFAULTS)
    config["engine_build_config"]["precision"] = EXPORT_PRECISION

    normalize_phase_workloads(config)
    return config


def save_run_config(config: Dict[str, Any]) -> None:
    config["workflow_backend"] = WORKFLOW_BACKEND
    config["workflow_id"] = WORKFLOW_ID
    config["requested_model_id"] = REQUESTED_MODEL_ID
    config["export_precision"] = EXPORT_PRECISION
    normalize_phase_workloads(config)
    write_json(RUN_CONFIG_PATH, config)


def export_root(config: Dict[str, Any]) -> Path:
    base = Path(config["artifacts"]["export_dir"])
    return base / MODEL_SLUG


def bundle_path(config: Dict[str, Any]) -> Path:
    base = Path(config["artifacts"]["export_bundle_dir"])
    return base / f"{MODEL_SLUG}__fp16.tar"


def onnx_dir(config: Dict[str, Any]) -> Path:
    return export_root(config) / "onnx"


def hf_assets_dir(config: Dict[str, Any]) -> Path:
    return export_root(config) / "hf_assets"


def bundle_manifest_path(config: Dict[str, Any]) -> Path:
    return export_root(config) / "bundle_manifest.json"


def engine_dir(config: Dict[str, Any]) -> Path:
    return Path(config["artifacts"]["runtime_dir"]) / MODEL_SLUG / "engines"


def runtime_output_path(config: Dict[str, Any], phase: str) -> Path:
    return Path(config["artifacts"]["runtime_dir"]) / MODEL_SLUG / f"{phase}_output.json"


def runtime_metadata_path(config: Dict[str, Any], phase: str, suffix: str) -> Path:
    return Path(config["artifacts"]["reports_dir"]) / f"{phase}_{suffix}.json"


def input_json_path(config: Dict[str, Any], phase: str) -> Path:
    return Path(config["artifacts"]["inputs_dir"]) / f"{phase}_input.json"


def metric_value_to_ns(value: str, unit: str) -> Optional[int]:
    try:
        numeric_value = float((value or "").replace(",", ""))
    except ValueError:
        return None

    scale = {
        "": 1.0,
        "ns": 1.0,
        "us": 1_000.0,
        "ms": 1_000_000.0,
        "s": 1_000_000_000.0,
    }.get((unit or "").strip().lower())
    if scale is None:
        return None
    return int(numeric_value * scale)


def metric_value_to_float(value: str) -> Optional[float]:
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def average_or_none(total: float, count: int) -> Optional[float]:
    if count <= 0:
        return None
    return total / count


def classify_kernel_family(kernel_name: str) -> str:
    lowered = kernel_name.lower()
    if any(token in lowered for token in SAMPLING_KERNEL_TOKENS):
        return "sampling_like"
    if any(token in lowered for token in COPY_KERNEL_TOKENS):
        return "copy_like"
    if any(token in lowered for token in GEMM_KERNEL_TOKENS):
        return "gemm_like"
    if any(token in lowered for token in ATTENTION_KERNEL_TOKENS):
        return "attention_like"
    if any(token in lowered for token in GATHER_KERNEL_TOKENS):
        return "gather_like"
    return "other"


def summarize_ncu_report(report_path: Path) -> Dict[str, Any]:
    result = run(
        [
            "ncu",
            "--import",
            str(report_path),
            "--csv",
            "--metrics",
            "gpu__time_duration.sum",
        ],
        capture_output=True,
    )

    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    per_kernel: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        metric_name = row.get("Metric Name")
        if metric_name not in {"gpu__time_duration.sum", "Duration"}:
            continue

        kernel_id = row.get("ID")
        if kernel_id is None:
            continue

        duration_ns = metric_value_to_ns(
            row.get("Metric Value", ""),
            row.get("Metric Unit", ""),
        )
        if duration_ns is None:
            continue

        per_kernel[kernel_id] = {
            "kernel_name": row.get("Kernel Name", ""),
            "duration_ns": duration_ns,
        }

    family_counts: Dict[str, int] = defaultdict(int)
    family_time_ns: Dict[str, int] = defaultdict(int)
    kernel_counter: Counter[str] = Counter()

    for kernel in per_kernel.values():
        family = classify_kernel_family(kernel["kernel_name"])
        family_counts[family] += 1
        family_time_ns[family] += kernel["duration_ns"]
        kernel_counter[kernel["kernel_name"]] += 1

    total_gpu_time_ns = sum(kernel["duration_ns"] for kernel in per_kernel.values())
    top_kernels = [
        {
            "kernel_name": kernel_name,
            "count": count,
            "family": classify_kernel_family(kernel_name),
        }
        for kernel_name, count in kernel_counter.most_common(5)
    ]

    inference_like_kernel_count = sum(
        family_counts.get(family, 0)
        for family in ("gemm_like", "attention_like", "gather_like")
    )
    rng_copy_kernel_count = sum(
        family_counts.get(family, 0)
        for family in ("sampling_like", "copy_like")
    )

    return {
        "kernel_id_count": len(per_kernel),
        "total_gpu_time_ns": total_gpu_time_ns,
        "family_counts": dict(sorted(family_counts.items())),
        "family_time_ns": dict(sorted(family_time_ns.items())),
        "inference_like_kernel_count": inference_like_kernel_count,
        "rng_copy_kernel_count": rng_copy_kernel_count,
        "is_rng_copy_only": (
            len(per_kernel) > 0
            and inference_like_kernel_count == 0
            and family_counts.get("other", 0) == 0
            and rng_copy_kernel_count == len(per_kernel)
        ),
        "top_kernels": top_kernels,
    }


def ncu_slice_cache_path(report_path: Path) -> Path:
    if report_path.name.endswith(".ncu-rep"):
        return report_path.with_name(report_path.name[:-8] + ".ncu-slice.json")
    return report_path.with_suffix(report_path.suffix + ".ncu-slice.json")


def build_ncu_slice_cache(report_path: Path) -> Dict[str, Any]:
    result = run(
        [
            "ncu",
            "--import",
            str(report_path),
            "--csv",
            "--metrics",
            NCU_SLICE_METRICS,
        ],
        capture_output=True,
    )

    launches: Dict[str, Dict[str, Any]] = {}
    for row in csv.DictReader(io.StringIO(result.stdout)):
        kernel_id = row.get("ID")
        kernel_name = row.get("Kernel Name", "")
        if not kernel_id or not kernel_name:
            continue

        metric_key = NCU_METRIC_ALIASES.get(row.get("Metric Name", ""))
        if metric_key is None:
            continue

        launch = launches.setdefault(
            kernel_id,
            {
                "id": kernel_id,
                "kernel_name": kernel_name,
                "family": classify_kernel_family(kernel_name),
                "process_id": row.get("Process ID", ""),
                "process_name": row.get("Process Name", ""),
                "context": row.get("Context", ""),
                "stream": row.get("Stream", ""),
                "device": row.get("Device", ""),
                "cc": row.get("CC", ""),
                "section_name": row.get("Section Name", ""),
                "block_size": row.get("Block Size", ""),
                "grid_size": row.get("Grid Size", ""),
                "duration_ns": None,
                "sm_pct": None,
                "dram_pct": None,
                "warps_pct": None,
            },
        )

        if metric_key == "duration_ns":
            duration_ns = metric_value_to_ns(
                row.get("Metric Value", ""),
                row.get("Metric Unit", ""),
            )
            if duration_ns is not None:
                launch["duration_ns"] = duration_ns
            continue

        numeric_value = metric_value_to_float(row.get("Metric Value", ""))
        if numeric_value is not None:
            launch[metric_key] = numeric_value

    launch_rows = sorted(
        launches.values(),
        key=lambda row: (
            -(row.get("duration_ns") or 0),
            row["kernel_name"],
            row["id"],
        ),
    )

    family_counts: Dict[str, int] = defaultdict(int)
    family_time_ns: Dict[str, int] = defaultdict(int)
    kernel_groups: Dict[str, Dict[str, Any]] = {}

    for launch in launch_rows:
        family = str(launch["family"])
        duration_ns = int(launch.get("duration_ns") or 0)
        family_counts[family] += 1
        family_time_ns[family] += duration_ns

        group = kernel_groups.setdefault(
            launch["kernel_name"],
            {
                "kernel_name": launch["kernel_name"],
                "family": family,
                "count": 0,
                "total_duration_ns": 0,
                "duration_sample_count": 0,
                "sm_total": 0.0,
                "sm_count": 0,
                "dram_total": 0.0,
                "dram_count": 0,
                "warps_total": 0.0,
                "warps_count": 0,
            },
        )
        group["count"] += 1
        if duration_ns > 0:
            group["total_duration_ns"] += duration_ns
            group["duration_sample_count"] += 1
        if launch.get("sm_pct") is not None:
            group["sm_total"] += float(launch["sm_pct"])
            group["sm_count"] += 1
        if launch.get("dram_pct") is not None:
            group["dram_total"] += float(launch["dram_pct"])
            group["dram_count"] += 1
        if launch.get("warps_pct") is not None:
            group["warps_total"] += float(launch["warps_pct"])
            group["warps_count"] += 1

    kernel_rows: List[Dict[str, Any]] = []
    for kernel_name, group in kernel_groups.items():
        avg_duration_ns = average_or_none(
            float(group["total_duration_ns"]),
            int(group["duration_sample_count"]),
        )
        kernel_rows.append(
            {
                "kernel_name": kernel_name,
                "family": str(group["family"]),
                "count": int(group["count"]),
                "total_duration_ns": int(group["total_duration_ns"]),
                "avg_duration_ns": int(avg_duration_ns) if avg_duration_ns is not None else None,
                "avg_sm_pct": average_or_none(group["sm_total"], group["sm_count"]),
                "avg_dram_pct": average_or_none(group["dram_total"], group["dram_count"]),
                "avg_warps_pct": average_or_none(group["warps_total"], group["warps_count"]),
            }
        )

    kernel_rows.sort(
        key=lambda row: (
            -(row.get("total_duration_ns") or 0),
            -(row.get("count") or 0),
            row["kernel_name"],
        )
    )

    total_gpu_time_ns = sum(int(row.get("duration_ns") or 0) for row in launch_rows)
    report_stat = report_path.stat()
    return {
        "report_path": str(report_path),
        "report_size_bytes": report_stat.st_size,
        "report_mtime_ns": report_stat.st_mtime_ns,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": NCU_SLICE_METRICS.split(","),
        "summary": {
            "kernel_id_count": len(launch_rows),
            "kernel_name_count": len(kernel_rows),
            "total_gpu_time_ns": total_gpu_time_ns,
            "family_counts": dict(sorted(family_counts.items())),
            "family_time_ns": dict(sorted(family_time_ns.items())),
            "top_kernels": kernel_rows[:20],
        },
        "launches": launch_rows,
        "kernels": kernel_rows,
    }


def load_ncu_slice_cache(report_path: Path, refresh: bool = False) -> Dict[str, Any]:
    cache_path = ncu_slice_cache_path(report_path)
    report_stat = report_path.stat()

    if not refresh and cache_path.exists():
        cache = read_json(cache_path)
        if (
            cache.get("report_path") == str(report_path)
            and int(cache.get("report_size_bytes", -1)) == report_stat.st_size
            and int(cache.get("report_mtime_ns", -1)) == report_stat.st_mtime_ns
        ):
            return cache

    cache = build_ncu_slice_cache(report_path)
    write_json(cache_path, cache)
    return cache


def ns_to_ms_text(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return f"{value / 1_000_000.0:.3f}"


def pct_text(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def truncate_text(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def print_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> None:
    widths = []
    for key, header in columns:
        width = len(header)
        for row in rows:
            width = max(width, len(str(row.get(key, ""))))
        widths.append(width)

    header_line = "  ".join(header.ljust(width) for (_, header), width in zip(columns, widths))
    print(header_line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                str(row.get(key, "")).ljust(width)
                for (key, _), width in zip(columns, widths)
            )
        )


def inspect_ncu_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    report_path: Optional[Path] = None

    if args.report:
        report_path = Path(args.report).resolve()
    else:
        phase_profile = config.get("profiles", {}).get(args.phase or "")
        if isinstance(phase_profile, dict) and phase_profile.get("ncu_rep"):
            report_path = Path(str(phase_profile["ncu_rep"])).resolve()

    if report_path is None:
        if args.phase:
            fail(
                f"No registered Nsight Compute report is recorded for phase '{args.phase}'. "
                "Pass --report explicitly or rerun the corresponding profiling step."
            )
        fail("Provide --report or --phase for a registered Nsight Compute report.")
    if not report_path.exists():
        fail(f"Nsight Compute report not found: {report_path}")

    cache = load_ncu_slice_cache(report_path, refresh=args.refresh_cache)

    family_filter = None if args.family == "all" else args.family
    name_filter = (args.name_contains or "").lower()

    def include_row(row: Dict[str, Any]) -> bool:
        if family_filter and row.get("family") != family_filter:
            return False
        if name_filter and name_filter not in str(row.get("kernel_name", "")).lower():
            return False
        return True

    if args.view == "summary":
        summary = cache["summary"]
        family_counts = summary.get("family_counts", {})
        family_time_ns = summary.get("family_time_ns", {})
        rows = []
        for family in NCU_FAMILIES:
            if family_filter and family != family_filter:
                continue
            count = int(family_counts.get(family, 0))
            total_ns = int(family_time_ns.get(family, 0))
            if count == 0 and total_ns == 0:
                continue
            rows.append(
                {
                    "family": family,
                    "count": count,
                    "total_ms": ns_to_ms_text(total_ns),
                }
            )

        if args.format == "json":
            print(
                json.dumps(
                    {
                        "report_path": str(report_path),
                        "cache_path": str(ncu_slice_cache_path(report_path)),
                        "summary": summary,
                    },
                    indent=2,
                )
            )
            return 0

        print(f"report: {report_path}")
        print(f"cache:  {ncu_slice_cache_path(report_path)}")
        print(f"launches: {summary.get('kernel_id_count', 0)}")
        print(f"kernel names: {summary.get('kernel_name_count', 0)}")
        print(f"total gpu time (ms): {ns_to_ms_text(summary.get('total_gpu_time_ns'))}")
        print("")
        if rows:
            print("family breakdown:")
            print_table(rows, (("family", "family"), ("count", "count"), ("total_ms", "total_ms")))
            print("")

        top_rows = []
        for row in cache["kernels"]:
            if not include_row(row):
                continue
            top_rows.append(
                {
                    "family": row["family"],
                    "count": row["count"],
                    "total_ms": ns_to_ms_text(row.get("total_duration_ns")),
                    "avg_ms": ns_to_ms_text(row.get("avg_duration_ns")),
                    "name": truncate_text(str(row["kernel_name"]), 90),
                }
            )
            if len(top_rows) >= args.limit:
                break

        if top_rows:
            print("top kernels by total gpu time:")
            print_table(
                top_rows,
                (
                    ("family", "family"),
                    ("count", "count"),
                    ("total_ms", "total_ms"),
                    ("avg_ms", "avg_ms"),
                    ("name", "kernel_name"),
                ),
            )
        return 0

    items = cache["kernels"] if args.view == "kernels" else cache["launches"]
    filtered = [row for row in items if include_row(row)]

    sort_key = args.sort

    def sort_value(row: Dict[str, Any]) -> Any:
        metric_value: Optional[float]
        if sort_key == "count":
            return -(row.get("count") or 0)
        if sort_key == "total_ms":
            return -(row.get("total_duration_ns") or 0)
        if sort_key == "avg_ms":
            return -(row.get("avg_duration_ns") or row.get("duration_ns") or 0)
        if sort_key == "sm_pct":
            metric_value = row.get("avg_sm_pct") if args.view == "kernels" else row.get("sm_pct")
            return -(metric_value if metric_value is not None else -1)
        if sort_key == "dram_pct":
            metric_value = row.get("avg_dram_pct") if args.view == "kernels" else row.get("dram_pct")
            return -(metric_value if metric_value is not None else -1)
        if sort_key == "warps_pct":
            metric_value = row.get("avg_warps_pct") if args.view == "kernels" else row.get("warps_pct")
            return -(metric_value if metric_value is not None else -1)
        if sort_key == "id":
            return int(row.get("id") or 0)
        return -(row.get("total_duration_ns") or row.get("duration_ns") or 0)

    filtered.sort(key=lambda row: (sort_value(row), str(row.get("kernel_name", "")), str(row.get("id", ""))))
    filtered = filtered[: args.limit]

    if args.format == "json":
        print(json.dumps(filtered, indent=2))
        return 0

    if args.view == "kernels":
        rows = [
            {
                "family": row["family"],
                "count": row["count"],
                "total_ms": ns_to_ms_text(row.get("total_duration_ns")),
                "avg_ms": ns_to_ms_text(row.get("avg_duration_ns")),
                "sm_pct": pct_text(row.get("avg_sm_pct")),
                "dram_pct": pct_text(row.get("avg_dram_pct")),
                "warps_pct": pct_text(row.get("avg_warps_pct")),
                "kernel_name": truncate_text(str(row["kernel_name"]), 90),
            }
            for row in filtered
        ]
        print_table(
            rows,
            (
                ("family", "family"),
                ("count", "count"),
                ("total_ms", "total_ms"),
                ("avg_ms", "avg_ms"),
                ("sm_pct", "sm_pct"),
                ("dram_pct", "dram_pct"),
                ("warps_pct", "warps_pct"),
                ("kernel_name", "kernel_name"),
            ),
        )
        return 0

    rows = [
        {
            "id": row["id"],
            "family": row["family"],
            "duration_ms": ns_to_ms_text(row.get("duration_ns")),
            "sm_pct": pct_text(row.get("sm_pct")),
            "dram_pct": pct_text(row.get("dram_pct")),
            "warps_pct": pct_text(row.get("warps_pct")),
            "kernel_name": truncate_text(str(row["kernel_name"]), 80),
        }
        for row in filtered
    ]
    print_table(
        rows,
        (
            ("id", "id"),
            ("family", "family"),
            ("duration_ms", "duration_ms"),
            ("sm_pct", "sm_pct"),
            ("dram_pct", "dram_pct"),
            ("warps_pct", "warps_pct"),
            ("kernel_name", "kernel_name"),
        ),
    )
    return 0


def collect_gpu_info() -> Dict[str, Any]:
    if not command_path("nvidia-smi"):
        return {"available": False}

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
        "available": True,
        "name": name,
        "driver_version": driver_version,
        "memory_total": memory_total,
        "compute_capability": compute_cap,
    }


def collect_python_info() -> Dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def ensure_host_export_platform() -> None:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
        fail(
            "Host export requires Linux on a supported CPU architecture "
            "(x86_64 or aarch64). "
            f"Detected machine architecture: {platform.machine()}"
        )


def ensure_aarch64_target() -> None:
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        fail(
            "Target runtime preparation is intended for DGX Spark / Thor-class aarch64 systems. "
            f"Detected machine architecture: {platform.machine()}"
        )


def detect_embedded_target() -> str:
    gb10_target_dir = Path("/usr/local/cuda/n1/targets/aarch64-linux")
    if gb10_target_dir.exists():
        return "gb10"
    return "jetson-thor"


def ensure_module_import(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        fail(f"Missing Python dependency '{module_name}': {exc}. {install_hint}")


def preferred_export_device() -> str:
    try:
        torch = ensure_module_import(
            "torch",
            "Install torch in the workflow virtual environment first.",
        )
    except WorkflowError:
        return "cpu"

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        try:
            if bool(cuda.is_available()):
                return "cuda"
        except Exception:
            pass
    return "cpu"


def ensure_hf_access(model_id: str) -> Dict[str, Any]:
    ensure_module_import(
        "huggingface_hub",
        "Install huggingface_hub in the workflow virtual environment first.",
    )
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, HfHubHTTPError

    try:
        config_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                repo_type="model",
            )
        )
    except GatedRepoError as exc:
        fail(
            f"Hugging Face access to {model_id} is required. "
            "Request access to the gated repo and authenticate with `huggingface-cli login` "
            f"before retrying. Original error: {exc}"
        )
    except EntryNotFoundError as exc:
        fail(f"{model_id} is missing config.json on Hugging Face: {exc}")
    except HfHubHTTPError as exc:
        fail(
            f"Failed to access {model_id} on Hugging Face. "
            "Confirm your token has read access and that the model terms were accepted. "
            f"Original error: {exc}"
        )
    except Exception as exc:
        fail(f"Failed to validate Hugging Face access for {model_id}: {type(exc).__name__}: {exc}")

    payload = read_json(config_path)
    return {
        "config_path": str(config_path),
        "config": payload,
    }


def export_tool_path() -> str:
    path = command_path("tensorrt-edgellm-export-llm")
    if not path:
        fail(
            "The TensorRT Edge-LLM export tool `tensorrt-edgellm-export-llm` was not found in PATH. "
            "Install the TensorRT Edge-LLM Python export pipeline in the workflow environment first."
        )
    return path


def target_repo_root(config: Dict[str, Any], override: Optional[str] = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    repo_root = config.get("edge_llm", {}).get("repo_root")
    if repo_root:
        return Path(str(repo_root)).expanduser().resolve()
    return default_edge_llm_repo_root()


def ensure_edge_llm_repo(repo_root: Path) -> Path:
    git_path = command_path("git")
    if repo_root.exists():
        if not git_path:
            return repo_root
        run([git_path, "submodule", "update", "--init", "--recursive"], cwd=repo_root)
        return repo_root

    if not git_path:
        fail(
            f"TensorRT Edge-LLM repository was not found at {repo_root}, and `git` is not available "
            "to clone it automatically."
        )

    repo_root.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            git_path,
            "clone",
            "--depth",
            "1",
            "--recurse-submodules",
            EDGE_LLM_GIT_URL,
            str(repo_root),
        ]
    )
    return repo_root


def llm_build_path(repo_root: Path) -> Path:
    return repo_root / "build" / "examples" / "llm" / "llm_build"


def llm_inference_path(repo_root: Path) -> Path:
    return repo_root / "build" / "examples" / "llm" / "llm_inference"


def check_required_paths(paths: Iterable[Path], description: str) -> None:
    for path in paths:
        if not path.exists():
            fail(f"Missing {description}: {path}")


def copy_hf_file(model_id: str, filename: str, destination_dir: Path) -> Optional[Path]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, HfHubHTTPError

    try:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                repo_type="model",
            )
        )
    except EntryNotFoundError:
        return None
    except GatedRepoError as exc:
        fail(
            f"Hugging Face access to {model_id} is required to fetch {filename}. "
            f"Original error: {exc}"
        )
    except HfHubHTTPError as exc:
        fail(f"Failed to download {filename} from {model_id}: {exc}")

    destination = destination_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, destination)
    return destination


def materialize_hf_assets(model_id: str, asset_dir: Path) -> Dict[str, Any]:
    copied: Dict[str, str] = {}
    asset_dir.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_TOKENIZER_FILES + OPTIONAL_TOKENIZER_FILES:
        copied_path = copy_hf_file(model_id, filename, asset_dir)
        if copied_path is not None:
            copied[filename] = str(copied_path)

    if not (asset_dir / "tokenizer.json").exists() and not (asset_dir / "tokenizer.model").exists():
        fail(
            f"Expected tokenizer.json or tokenizer.model in {asset_dir}, but neither file was available."
        )

    tokenizer_config = read_json(asset_dir / "tokenizer_config.json")
    has_chat_template = (asset_dir / "chat_template.jinja").exists() or bool(
        tokenizer_config.get("chat_template")
    )
    if not has_chat_template:
        fail(
            "The export bundle is missing chat-template metadata. "
            "Expected chat_template.jinja or tokenizer_config.json['chat_template']."
        )

    return copied


def validate_export_tree(config: Dict[str, Any]) -> Dict[str, Any]:
    model_root = export_root(config)
    model_onnx_dir = onnx_dir(config)
    model_assets_dir = hf_assets_dir(config)

    if not model_onnx_dir.exists():
        fail(f"Expected ONNX export directory was not found: {model_onnx_dir}")
    if not model_assets_dir.exists():
        fail(f"Expected Hugging Face asset directory was not found: {model_assets_dir}")

    onnx_files = sorted(str(path.relative_to(model_root)) for path in model_onnx_dir.rglob("*.onnx"))
    if not onnx_files:
        fail(f"No ONNX files were found under {model_onnx_dir}")

    check_required_paths(
        [model_assets_dir / filename for filename in REQUIRED_TOKENIZER_FILES],
        "required tokenizer asset",
    )

    tokenizer_exists = (model_assets_dir / "tokenizer.json").exists() or (
        model_assets_dir / "tokenizer.model"
    ).exists()
    if not tokenizer_exists:
        fail(
            f"Tokenizer assets are incomplete in {model_assets_dir}. "
            "Expected tokenizer.json or tokenizer.model."
        )

    tokenizer_config = read_json(model_assets_dir / "tokenizer_config.json")
    has_chat_template = (model_assets_dir / "chat_template.jinja").exists() or bool(
        tokenizer_config.get("chat_template")
    )
    if not has_chat_template:
        fail(
            f"Chat-template metadata is incomplete in {model_assets_dir}. "
            "Expected chat_template.jinja or tokenizer_config.json['chat_template']."
        )

    hf_files = sorted(
        str(path.relative_to(model_root))
        for path in model_assets_dir.rglob("*")
        if path.is_file()
    )

    manifest = {
        "model_id": REQUESTED_MODEL_ID,
        "model_slug": MODEL_SLUG,
        "workflow_backend": WORKFLOW_BACKEND,
        "workflow_id": WORKFLOW_ID,
        "export_precision": EXPORT_PRECISION,
        "export_root": str(model_root),
        "onnx_dir": str(model_onnx_dir),
        "hf_assets_dir": str(model_assets_dir),
        "onnx_files": onnx_files,
        "hf_asset_files": hf_files,
        "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(bundle_manifest_path(config), manifest)
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not member_path.is_relative_to(destination):
            fail(f"Refusing to extract bundle member outside destination: {member.name}")
    tar.extractall(destination)


def prune_generated_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def host_preflight_command(_: argparse.Namespace) -> int:
    config = load_run_config()
    ensure_host_export_platform()

    hf_access = ensure_hf_access(REQUESTED_MODEL_ID)
    gpu_info = collect_gpu_info()

    tool_path = export_tool_path()

    result = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_model_id": REQUESTED_MODEL_ID,
        "export_precision": EXPORT_PRECISION,
        "machine": platform.machine(),
        "python": collect_python_info(),
        "gpu": gpu_info,
        "tensorrt_edgellm_export_llm": tool_path,
        "hugging_face_access": {
            "config_path": hf_access["config_path"],
            "model_type": hf_access["config"].get("model_type"),
        },
    }

    config["host_preflight"] = result
    save_run_config(config)
    print(json.dumps(result, indent=2))
    return 0


def host_export_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    ensure_host_export_platform()

    hf_access = ensure_hf_access(REQUESTED_MODEL_ID)
    tool = export_tool_path()
    model_root = export_root(config)
    model_onnx_dir = onnx_dir(config)
    model_assets_dir = hf_assets_dir(config)

    if args.force:
        prune_generated_tree(model_onnx_dir)
        prune_generated_tree(model_assets_dir)

    model_root.mkdir(parents=True, exist_ok=True)
    materialize_hf_assets(REQUESTED_MODEL_ID, model_assets_dir)

    export_cmd = [
        tool,
        "--model_dir",
        REQUESTED_MODEL_ID,
        "--output_dir",
        str(model_onnx_dir),
        "--device",
        preferred_export_device(),
    ]
    run(export_cmd)
    manifest = validate_export_tree(config)

    metadata = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_model_id": REQUESTED_MODEL_ID,
        "actual_model_id": REQUESTED_MODEL_ID,
        "export_precision": EXPORT_PRECISION,
        "export_command": export_cmd,
        "export_root": str(model_root),
        "onnx_dir": str(model_onnx_dir),
        "hf_assets_dir": str(model_assets_dir),
        "hugging_face_config_path": hf_access["config_path"],
        "bundle_manifest_path": str(bundle_manifest_path(config)),
        "manifest": manifest,
    }

    config["host_preflight"] = {
        **config.get("host_preflight", {}),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_model_id": REQUESTED_MODEL_ID,
        "export_precision": EXPORT_PRECISION,
    }
    config["host_export"] = metadata
    save_run_config(config)
    print(json.dumps(metadata, indent=2))
    return 0


def host_package_export_command(_: argparse.Namespace) -> int:
    config = load_run_config()
    model_root = export_root(config)
    manifest = validate_export_tree(config)
    output_bundle = bundle_path(config)
    output_bundle.parent.mkdir(parents=True, exist_ok=True)

    if output_bundle.exists():
        output_bundle.unlink()

    with tarfile.open(output_bundle, "w") as archive:
        archive.add(onnx_dir(config), arcname="onnx")
        archive.add(hf_assets_dir(config), arcname="hf_assets")
        archive.add(bundle_manifest_path(config), arcname="bundle_manifest.json")

    checksum = sha256_file(output_bundle)
    payload = {
        **config.get("host_export", {}),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_model_id": REQUESTED_MODEL_ID,
        "actual_model_id": REQUESTED_MODEL_ID,
        "export_precision": EXPORT_PRECISION,
        "export_root": str(model_root),
        "onnx_dir": str(onnx_dir(config)),
        "hf_assets_dir": str(hf_assets_dir(config)),
        "bundle_manifest_path": str(bundle_manifest_path(config)),
        "bundle_path": str(output_bundle),
        "bundle_sha256": checksum,
        "manifest": manifest,
    }
    config["host_export"] = payload
    save_run_config(config)
    print(json.dumps(payload, indent=2))
    return 0


def default_bundle_source(config: Dict[str, Any]) -> str:
    existing_bundle = config.get("host_export", {}).get("bundle_path")
    if existing_bundle:
        return str(existing_bundle)
    return str(bundle_path(config))


def target_preflight_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    ensure_aarch64_target()

    repo_root = ensure_edge_llm_repo(target_repo_root(config, args.repo_root))

    required_commands = {}
    for command in ("cmake", "make", "nvcc"):
        resolved = command_path(command)
        if not resolved:
            fail(f"Required target command `{command}` was not found in PATH.")
        required_commands[command] = resolved

    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cache = build_dir / "CMakeCache.txt"
    toolchain_file = repo_root / "cmake" / "aarch64_linux_toolchain.cmake"
    trt_package_dir = os.environ.get("TRT_PACKAGE_DIR", "/usr")
    embedded_target = detect_embedded_target()

    if args.force_reconfigure or not cmake_cache.exists():
        run(
            [
                "cmake",
                str(repo_root),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DTRT_PACKAGE_DIR={trt_package_dir}",
                f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
                f"-DEMBEDDED_TARGET={embedded_target}",
            ],
            cwd=build_dir,
        )

    build_jobs = max(1, args.jobs or (os.cpu_count() or 1))
    build_binary_paths = [llm_build_path(repo_root), llm_inference_path(repo_root)]
    if args.force_rebuild or any(not path.exists() for path in build_binary_paths):
        run(["make", f"-j{build_jobs}"], cwd=build_dir)

    check_required_paths(build_binary_paths, "Edge-LLM runtime binary")

    nvcc_version = run(["nvcc", "--version"], capture_output=True).stdout.strip()
    dpkg_output = run(["dpkg", "-l"], capture_output=True).stdout
    tensorrt_lines = [
        line
        for line in dpkg_output.splitlines()
        if "tensorrt" in line.lower() or "nvinfer" in line.lower()
    ]

    tegra_release = Path("/etc/nv_tegra_release")
    target_info = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.machine(),
        "repo_root": str(repo_root),
        "build_dir": str(build_dir),
        "llm_build_path": str(llm_build_path(repo_root)),
        "llm_inference_path": str(llm_inference_path(repo_root)),
        "required_commands": required_commands,
        "trt_package_dir": trt_package_dir,
        "embedded_target": embedded_target,
        "nvcc_version": nvcc_version,
        "tensorrt_packages": tensorrt_lines,
        "nv_tegra_release": tegra_release.read_text().strip() if tegra_release.exists() else None,
    }

    config.setdefault("edge_llm", {})
    config["edge_llm"]["repo_root"] = str(repo_root)
    config["target_preflight"] = target_info
    save_run_config(config)
    print(json.dumps(target_info, indent=2))
    return 0


def fetch_bundle_to_local_path(source: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(source).expanduser()

    if source_path.exists():
        if source_path.resolve() == destination.resolve():
            return destination
        shutil.copy2(source_path, destination)
        return destination

    if ":" in source:
        if not command_path("scp"):
            fail("The bundle source looks remote, but `scp` was not found in PATH.")
        run(["scp", source, str(destination)])
        return destination

    fail(f"Bundle source does not exist and is not a valid remote SCP source: {source}")


def target_fetch_export_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    model_root = export_root(config)
    bundle_copy = bundle_path(config)
    source = args.source or default_bundle_source(config)
    source_path = Path(source).expanduser()

    if source_path.exists():
        try:
            manifest = validate_export_tree(config)
        except WorkflowError:
            manifest = None
        else:
            checksum = sha256_file(source_path)
            config["host_export"] = {
                **config.get("host_export", {}),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "requested_model_id": REQUESTED_MODEL_ID,
                "actual_model_id": REQUESTED_MODEL_ID,
                "export_precision": EXPORT_PRECISION,
                "source": source,
                "bundle_path": str(source_path.resolve()),
                "bundle_sha256": checksum,
                "export_root": str(model_root),
                "onnx_dir": str(onnx_dir(config)),
                "hf_assets_dir": str(hf_assets_dir(config)),
                "bundle_manifest_path": str(bundle_manifest_path(config)),
                "manifest": manifest,
                "reuse_existing_export_tree": True,
            }
            config.pop("engine_build", None)
            config.pop("selected_runtime", None)
            profiles = config.get("profiles")
            if isinstance(profiles, dict):
                profiles.clear()
            reports = config.get("reports")
            if isinstance(reports, dict):
                reports.clear()
            save_run_config(config)
            print(json.dumps(config["host_export"], indent=2))
            return 0

    if args.force:
        prune_generated_tree(model_root)
    model_root.mkdir(parents=True, exist_ok=True)

    fetched_bundle = fetch_bundle_to_local_path(source, bundle_copy)

    for child in ("onnx", "hf_assets", "bundle_manifest.json"):
        target = model_root / child
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    with tarfile.open(fetched_bundle, "r:*") as archive:
        safe_extract_tar(archive, model_root)

    manifest = validate_export_tree(config)
    checksum = sha256_file(fetched_bundle)

    config["host_export"] = {
        **config.get("host_export", {}),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_model_id": REQUESTED_MODEL_ID,
        "actual_model_id": REQUESTED_MODEL_ID,
        "export_precision": EXPORT_PRECISION,
        "source": source,
        "bundle_path": str(fetched_bundle),
        "bundle_sha256": checksum,
        "export_root": str(model_root),
        "onnx_dir": str(onnx_dir(config)),
        "hf_assets_dir": str(hf_assets_dir(config)),
        "bundle_manifest_path": str(bundle_manifest_path(config)),
        "manifest": manifest,
    }
    config.pop("engine_build", None)
    config.pop("selected_runtime", None)
    profiles = config.get("profiles")
    if isinstance(profiles, dict):
        profiles.clear()
    reports = config.get("reports")
    if isinstance(reports, dict):
        reports.clear()
    save_run_config(config)
    print(json.dumps(config["host_export"], indent=2))
    return 0


def ensure_target_binaries(config: Dict[str, Any], repo_override: Optional[str] = None) -> Dict[str, str]:
    repo_root = target_repo_root(config, repo_override)
    build_path = llm_build_path(repo_root)
    inference_path = llm_inference_path(repo_root)

    if build_path.exists() and inference_path.exists():
        return {
            "repo_root": str(repo_root),
            "llm_build_path": str(build_path),
            "llm_inference_path": str(inference_path),
        }

    target_preflight_command(
        argparse.Namespace(
            repo_root=repo_override,
            force_reconfigure=False,
            force_rebuild=False,
            jobs=None,
        )
    )
    return {
        "repo_root": str(repo_root),
        "llm_build_path": str(build_path),
        "llm_inference_path": str(inference_path),
    }


def build_engine_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    runtime_paths = ensure_target_binaries(config, args.repo_root)
    manifest = validate_export_tree(config)
    repo_root = Path(runtime_paths["repo_root"])
    plugin_path = repo_root / "build" / "libNvInfer_edgellm_plugin.so"

    build_config = copy.deepcopy(config["engine_build_config"])
    target_engine_dir = engine_dir(config)
    if args.force:
        prune_generated_tree(target_engine_dir)
    target_engine_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        runtime_paths["llm_build_path"],
        "--onnxDir",
        str(onnx_dir(config)),
        "--engineDir",
        str(target_engine_dir),
        "--maxBatchSize",
        str(build_config["maxBatchSize"]),
        "--maxInputLen",
        str(build_config["maxInputLen"]),
        "--maxKVCacheCapacity",
        str(build_config["maxKVCacheCapacity"]),
    ]
    env = os.environ.copy()
    if plugin_path.exists():
        env["EDGELLM_PLUGIN_PATH"] = str(plugin_path)
    run(cmd, cwd=repo_root, env=env)

    payload = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_model_id": REQUESTED_MODEL_ID,
        "actual_model_id": REQUESTED_MODEL_ID,
        "precision": EXPORT_PRECISION,
        "engine_dir": str(target_engine_dir),
        "onnx_dir": str(onnx_dir(config)),
        "llm_build_path": runtime_paths["llm_build_path"],
        "llm_inference_path": runtime_paths["llm_inference_path"],
        "edgellm_plugin_path": str(plugin_path) if plugin_path.exists() else None,
        "build_command": cmd,
        "engine_build_config": build_config,
        "manifest": manifest,
    }

    config["engine_build"] = payload
    config["selected_runtime"] = {
        "workflow_backend": WORKFLOW_BACKEND,
        "workflow_id": WORKFLOW_ID,
        "requested_model_id": REQUESTED_MODEL_ID,
        "actual_model_id": REQUESTED_MODEL_ID,
        "precision": EXPORT_PRECISION,
        "engine_dir": str(target_engine_dir),
        "llm_inference_path": runtime_paths["llm_inference_path"],
        "llm_build_path": runtime_paths["llm_build_path"],
        "repo_root": str(repo_root),
        "edgellm_plugin_path": str(plugin_path) if plugin_path.exists() else None,
        "max_input_len": build_config["maxInputLen"],
    }
    save_run_config(config)
    print(json.dumps(payload, indent=2))
    return 0


def tokenizer_source(config: Dict[str, Any]) -> Tuple[str, bool]:
    asset_root = hf_assets_dir(config)
    if asset_root.exists():
        return str(asset_root), True
    return REQUESTED_MODEL_ID, False


def load_tokenizer(config: Dict[str, Any]) -> Any:
    ensure_module_import(
        "transformers",
        "Install transformers and sentencepiece in the workflow virtual environment first.",
    )
    from transformers import AutoTokenizer

    source, local_only = tokenizer_source(config)
    return AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        local_files_only=local_only,
    )


def token_count_for_messages(tokenizer: Any, messages: List[Dict[str, Any]]) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            templated = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            if isinstance(templated, dict):
                input_ids = templated.get("input_ids", [])
                return len(input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids)
            if hasattr(templated, "shape"):
                shape = getattr(templated, "shape")
                if len(shape) == 1:
                    return int(shape[0])
                if len(shape) >= 2:
                    return int(shape[-1])
            return len(templated)
        except TypeError:
            pass

    text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    tokenized = tokenizer(text, return_tensors=None)
    input_ids = tokenized["input_ids"]
    return len(input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids)


def build_prefill_prompt(tokenizer: Any, base_prompt: str, target_tokens: int) -> Tuple[List[Dict[str, Any]], int]:
    expanded_sections = [
        base_prompt.strip(),
        "Explain the tokenizer boundary conditions, special tokens, and chat-template expansion.",
        "Describe how embeddings, self-attention, MLP blocks, and the KV cache interact during autoregressive inference.",
        "Include practical notes about prefill bandwidth, decode latency, batching, sampling, and memory pressure.",
        "Summarize how export, engine build, and runtime execution differ between x86 host preparation and Jetson Thor deployment.",
    ]

    content = "\n\n".join(section for section in expanded_sections if section)
    messages = [{"role": "user", "content": content}]
    token_count = token_count_for_messages(tokenizer, messages)

    while token_count < target_tokens:
        content = content + "\n\n" + "\n\n".join(expanded_sections[1:])
        messages = [{"role": "user", "content": content}]
        token_count = token_count_for_messages(tokenizer, messages)

    return messages, token_count


def build_phase_payload(
    config: Dict[str, Any],
    phase: str,
    *,
    prompt_override: Optional[str] = None,
    max_generate_length_override: Optional[int] = None,
) -> Dict[str, Any]:
    phase_config = copy.deepcopy(config["phase_workloads"][phase])
    tokenizer = load_tokenizer(config)
    base_prompt = prompt_override or config.get("workload", {}).get("prompt") or DEFAULT_PROMPT

    target_tokens = phase_config.get("target_input_tokens")
    if target_tokens is not None and int(target_tokens) > 0:
        messages, input_tokens = build_prefill_prompt(tokenizer, base_prompt, int(target_tokens))
    else:
        messages = [{"role": "user", "content": base_prompt}]
        input_tokens = token_count_for_messages(tokenizer, messages)

    max_generate_length = (
        max_generate_length_override
        if max_generate_length_override is not None
        else int(phase_config["max_generate_length"])
    )

    payload = {
        "batch_size": int(phase_config.get("batch_size", 1)),
        "temperature": float(phase_config.get("temperature", 1.0)),
        "top_p": float(phase_config.get("top_p", 1.0)),
        "top_k": int(phase_config.get("top_k", 50)),
        "max_generate_length": int(max_generate_length),
        "requests": [
            {
                "messages": messages,
            }
        ],
    }

    phase_config.update(
        {
            "input_json": str(input_json_path(config, phase)),
            "input_token_count": int(input_tokens),
            "resolved_prompt": messages[0]["content"],
            "resolved_prompt_kind": phase_config.get("prompt_kind"),
            "max_generate_length": int(max_generate_length),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    config["phase_workloads"][phase] = phase_config
    return payload


def validate_phase_input_token_count(
    runtime: Dict[str, Any],
    phase: str,
    input_metadata: Dict[str, Any],
) -> None:
    input_token_count = int(input_metadata.get("input_token_count", 0) or 0)
    max_input_len = int(runtime.get("max_input_len", 0) or 0)
    if max_input_len > 0 and input_token_count > max_input_len:
        fail(
            f"{phase.title()} input token count ({input_token_count}) exceeds the engine's "
            f"maxInputLen ({max_input_len}). Rerun 06_target_build_engine.sh to rebuild "
            "the engine with a larger input length."
        )


def load_selected_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = config.get("selected_runtime")
    if not isinstance(runtime, dict):
        fail(
            "No selected Edge-LLM runtime is recorded. Run target-fetch-export and build-engine first."
        )

    runtime_engine_dir = Path(str(runtime.get("engine_dir", "")))
    runtime_binary = Path(str(runtime.get("llm_inference_path", "")))
    if not runtime_engine_dir.exists():
        fail(f"Selected engine directory does not exist: {runtime_engine_dir}")
    if not runtime_binary.exists():
        fail(f"Selected llm_inference binary does not exist: {runtime_binary}")
    return runtime


def resolve_inference_input(
    config: Dict[str, Any],
    runtime: Dict[str, Any],
    phase: str,
    *,
    input_file: Optional[str] = None,
    prompt_override: Optional[str] = None,
    max_generate_length_override: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    if input_file:
        input_path = Path(input_file).resolve()
        if not input_path.exists():
            fail(f"Input JSON file was not found: {input_path}")
        payload = read_json(input_path)
        return input_path, {
            "input_json": str(input_path),
            "resolved_prompt_kind": "external_input_file",
            "max_generate_length": payload.get("max_generate_length"),
        }

    payload = build_phase_payload(
        config,
        phase,
        prompt_override=prompt_override,
        max_generate_length_override=max_generate_length_override,
    )
    input_path = input_json_path(config, phase)
    write_json(input_path, payload)
    input_metadata = copy.deepcopy(config["phase_workloads"][phase])
    save_run_config(config)
    validate_phase_input_token_count(runtime, phase, input_metadata)
    return input_path, input_metadata


def extract_output_metadata(output_path: Path) -> Dict[str, Any]:
    if not output_path.exists():
        return {"output_exists": False}

    payload: Dict[str, Any] = {"output_exists": True, "output_file": str(output_path)}
    try:
        parsed = read_json(output_path)
    except json.JSONDecodeError:
        payload["output_json_valid"] = False
        return payload

    payload["output_json_valid"] = True
    if isinstance(parsed, dict):
        payload["output_top_level_keys"] = sorted(parsed.keys())
        requests = parsed.get("requests")
        if isinstance(requests, list):
            payload["request_count"] = len(requests)
        responses = parsed.get("responses")
        if isinstance(responses, list):
            payload["response_count"] = len(responses)
    return payload


def runtime_profile_stage_timings(runtime_profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stages = runtime_profile.get("stages")
    if not isinstance(stages, list):
        return {}

    stage_timings: Dict[str, Dict[str, Any]] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_id = stage.get("stage_id")
        if not stage_id:
            continue
        stage_timings[str(stage_id)] = {
            "total_runs": stage.get("total_runs"),
            "total_gpu_time_ms": stage.get("total_gpu_time_ms"),
            "average_time_per_run_ms": stage.get("average_time_per_run_ms"),
            "gpu_time_stats": copy.deepcopy(stage.get("gpu_time_stats", {})),
        }
    return stage_timings


def runtime_profile_summary(runtime_profile: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in ("prefill", "generation", "eagle_generation", "multimodal"):
        section = runtime_profile.get(key)
        if isinstance(section, dict):
            summary[key] = copy.deepcopy(section)

    for key in (
        "peak_unified_memory_bytes",
        "peak_unified_memory_mb",
        "peak_gpu_memory_bytes",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_bytes",
        "peak_cpu_memory_mb",
    ):
        if key in runtime_profile:
            summary[key] = runtime_profile.get(key)
    return summary


def infer_actual_output_token_count(
    runtime_profile: Dict[str, Any],
    requested_max_generate_length: Optional[Any],
    output_exists: bool,
) -> Optional[int]:
    if not output_exists:
        return None

    generation = runtime_profile.get("generation")
    generated_after_prefill = 0
    if isinstance(generation, dict):
        generated_after_prefill = int(generation.get("generated_tokens", 0) or 0)

    if generated_after_prefill > 0:
        return generated_after_prefill + 1

    try:
        max_generate_length = int(requested_max_generate_length)
    except (TypeError, ValueError):
        return None

    if max_generate_length <= 0:
        return 0
    return 1


def extract_runtime_profile_metadata(
    runtime_profile_path: Optional[Path],
    requested_max_generate_length: Optional[Any],
    output_exists: bool,
) -> Dict[str, Any]:
    if runtime_profile_path is None:
        return {}

    metadata: Dict[str, Any] = {
        "runtime_profile_json": str(runtime_profile_path),
        "runtime_profile_exists": runtime_profile_path.exists(),
    }
    if not runtime_profile_path.exists():
        return metadata

    try:
        runtime_profile = read_json(runtime_profile_path)
    except json.JSONDecodeError:
        metadata["runtime_profile_valid"] = False
        return metadata

    if not isinstance(runtime_profile, dict):
        metadata["runtime_profile_valid"] = False
        return metadata

    metadata["runtime_profile_valid"] = True
    metadata["runtime_profile_summary"] = runtime_profile_summary(runtime_profile)
    metadata["runtime_profile_stage_timings"] = runtime_profile_stage_timings(runtime_profile)
    metadata["actual_output_token_count"] = infer_actual_output_token_count(
        runtime_profile,
        requested_max_generate_length,
        output_exists,
    )
    return metadata


def execute_inference_run(
    runtime: Dict[str, Any],
    phase: str,
    input_path: Path,
    output_path: Path,
    input_metadata: Dict[str, Any],
    *,
    warmup_runs: int = 0,
    runtime_profile_output: Optional[Path] = None,
) -> Dict[str, Any]:
    if warmup_runs < 0:
        fail("--warmup-runs must be non-negative.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(runtime["llm_inference_path"]),
        "--engineDir",
        str(runtime["engine_dir"]),
        "--inputFile",
        str(input_path),
        "--outputFile",
        str(output_path),
    ]
    if warmup_runs > 0:
        cmd.extend(["--warmup", str(warmup_runs)])
    if runtime_profile_output is not None:
        runtime_profile_output.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(
            [
                "--dumpProfile",
                "--profileOutputFile",
                str(runtime_profile_output),
            ]
        )

    repo_root_str = runtime.get("repo_root")
    cwd = Path(str(repo_root_str)) if repo_root_str else None
    plugin_path_str = runtime.get("edgellm_plugin_path")
    env = os.environ.copy()
    if plugin_path_str:
        env["EDGELLM_PLUGIN_PATH"] = str(plugin_path_str)
        log(f"Resolved Edge-LLM plugin path: {plugin_path_str}")

    started = time.time()
    run(cmd, cwd=cwd, env=env)
    elapsed = round(time.time() - started, 3)

    metadata = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase,
        "workflow_backend": WORKFLOW_BACKEND,
        "requested_model_id": str(runtime.get("requested_model_id", REQUESTED_MODEL_ID)),
        "actual_model_id": str(runtime.get("actual_model_id", REQUESTED_MODEL_ID)),
        "precision": str(runtime.get("precision", EXPORT_PRECISION)),
        "engine_dir": str(runtime["engine_dir"]),
        "llm_inference_path": str(runtime["llm_inference_path"]),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "run_command": cmd,
        "elapsed_seconds": elapsed,
        "requested_max_generate_length": input_metadata.get("max_generate_length"),
        "phase_workload": input_metadata,
        "warmup_runs": int(warmup_runs),
    }
    metadata.update(extract_output_metadata(output_path))
    metadata.update(
        extract_runtime_profile_metadata(
            runtime_profile_output,
            input_metadata.get("max_generate_length"),
            bool(metadata.get("output_exists")),
        )
    )
    return metadata


def run_inference_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    runtime = load_selected_runtime(config)
    phase = args.phase
    input_path, input_metadata = resolve_inference_input(
        config,
        runtime,
        phase,
        input_file=args.input_file,
        prompt_override=args.prompt,
        max_generate_length_override=args.max_generate_length,
    )
    output_path = Path(args.output_file).resolve() if args.output_file else runtime_output_path(config, phase)
    runtime_profile_output = (
        Path(args.runtime_profile_output).resolve() if args.runtime_profile_output else None
    )
    metadata = execute_inference_run(
        runtime,
        phase,
        input_path,
        output_path,
        input_metadata,
        warmup_runs=args.warmup_runs,
        runtime_profile_output=runtime_profile_output,
    )

    if args.metadata_output:
        write_json(Path(args.metadata_output).resolve(), metadata)
    else:
        print(json.dumps(metadata, indent=2))

    profiles = config.setdefault("profiles", {})
    phase_profile = profiles.setdefault(phase, {})
    phase_profile["last_output_json"] = str(output_path)
    phase_profile["last_runtime_metadata"] = metadata
    if metadata.get("runtime_profile_json"):
        phase_profile["last_runtime_profile_json"] = str(metadata["runtime_profile_json"])
    save_run_config(config)
    return 0


def percentile_value(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    index = min(int(len(sorted_values) * percentile), len(sorted_values) - 1)
    return sorted_values[index]


def numeric_series_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}

    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "min": min(normalized),
        "max": max(normalized),
        "mean": statistics.fmean(normalized),
        "median": statistics.median(normalized),
        "p95": percentile_value(normalized, 0.95),
        "p99": percentile_value(normalized, 0.99),
        "stddev": statistics.pstdev(normalized) if len(normalized) > 1 else 0.0,
    }


def benchmark_output_path(config: Dict[str, Any], phase: str, run_index: int) -> Path:
    return Path(config["artifacts"]["runtime_dir"]) / MODEL_SLUG / f"{phase}_benchmark_run_{run_index:02d}_output.json"


def benchmark_runtime_profile_path(config: Dict[str, Any], phase: str, run_index: int) -> Path:
    return runtime_metadata_path(config, phase, f"benchmark_run_{run_index:02d}_runtime_profile")


def benchmark_metadata_path(config: Dict[str, Any], phase: str, run_index: int) -> Path:
    return runtime_metadata_path(config, phase, f"benchmark_run_{run_index:02d}_metadata")


def benchmark_summary_path(config: Dict[str, Any], phase: str) -> Path:
    return runtime_metadata_path(config, phase, "benchmark_summary")


def runtime_profile_section_from_metadata(
    metadata: Dict[str, Any],
    section_name: str,
) -> Optional[Dict[str, Any]]:
    runtime_profile_summary_payload = metadata.get("runtime_profile_summary")
    if not isinstance(runtime_profile_summary_payload, dict):
        return None
    section = runtime_profile_summary_payload.get(section_name)
    if not isinstance(section, dict):
        return None
    return section


def runtime_stage_from_metadata(
    metadata: Dict[str, Any],
    stage_id: str,
) -> Optional[Dict[str, Any]]:
    stage_timings = metadata.get("runtime_profile_stage_timings")
    if not isinstance(stage_timings, dict):
        return None
    stage = stage_timings.get(stage_id)
    if not isinstance(stage, dict):
        return None
    return stage


def build_benchmark_section_summary(
    run_metadata: Sequence[Dict[str, Any]],
    section_name: str,
    stage_id: str,
    metric_names: Sequence[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "present_in_runs": 0,
        "metrics": {},
    }

    for metadata in run_metadata:
        if runtime_profile_section_from_metadata(metadata, section_name):
            summary["present_in_runs"] += 1

    for metric_name in metric_names:
        values: List[float] = []
        for metadata in run_metadata:
            section = runtime_profile_section_from_metadata(metadata, section_name)
            if not section:
                continue
            value = section.get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            summary["metrics"][metric_name] = numeric_series_stats(values)

    for metric_name in ("total_gpu_time_ms", "average_time_per_run_ms"):
        values = []
        for metadata in run_metadata:
            stage = runtime_stage_from_metadata(metadata, stage_id)
            if not stage:
                continue
            value = stage.get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            summary["metrics"][f"stage_{metric_name}"] = numeric_series_stats(values)

    return summary


def benchmark_phase_command(args: argparse.Namespace) -> int:
    if args.repeats <= 0:
        fail("--repeats must be positive.")
    if args.warmup_runs < 0:
        fail("--warmup-runs must be non-negative.")

    config = load_run_config()
    runtime = load_selected_runtime(config)
    phase = args.phase
    input_path, input_metadata = resolve_inference_input(
        config,
        runtime,
        phase,
        prompt_override=args.prompt,
        max_generate_length_override=args.max_generate_length,
    )

    collected_metadata: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []
    for run_index in range(1, args.repeats + 1):
        output_path = benchmark_output_path(config, phase, run_index)
        runtime_profile_path = benchmark_runtime_profile_path(config, phase, run_index)
        metadata_path = benchmark_metadata_path(config, phase, run_index)

        metadata = execute_inference_run(
            runtime,
            phase,
            input_path,
            output_path,
            input_metadata,
            warmup_runs=args.warmup_runs,
            runtime_profile_output=runtime_profile_path,
        )
        if not metadata.get("runtime_profile_valid"):
            fail(
                f"Benchmark run {run_index} did not produce a valid runtime profile at "
                f"{runtime_profile_path}."
            )

        write_json(metadata_path, metadata)
        collected_metadata.append(metadata)
        run_summaries.append(
            {
                "run_index": run_index,
                "metadata_json": str(metadata_path),
                "runtime_profile_json": metadata.get("runtime_profile_json"),
                "output_file": metadata.get("output_file"),
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "actual_output_token_count": metadata.get("actual_output_token_count"),
                "prefill": copy.deepcopy(runtime_profile_section_from_metadata(metadata, "prefill")),
                "generation": copy.deepcopy(runtime_profile_section_from_metadata(metadata, "generation")),
                "runtime_profile_stage_timings": copy.deepcopy(
                    metadata.get("runtime_profile_stage_timings", {})
                ),
            }
        )

    benchmark_summary = {
        "phase": phase,
        "workflow_backend": WORKFLOW_BACKEND,
        "workflow_id": WORKFLOW_ID,
        "requested_model_id": str(runtime.get("requested_model_id", REQUESTED_MODEL_ID)),
        "actual_model_id": str(runtime.get("actual_model_id", REQUESTED_MODEL_ID)),
        "precision": str(runtime.get("precision", EXPORT_PRECISION)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repeats": int(args.repeats),
        "warmup_runs": int(args.warmup_runs),
        "input_file": str(input_path),
        "phase_workload": copy.deepcopy(input_metadata),
        "runs": run_summaries,
        "summary": {
            "prefill": build_benchmark_section_summary(
                collected_metadata,
                "prefill",
                "llm_prefill",
                (
                    "reused_tokens",
                    "computed_tokens",
                    "average_tokens_per_run",
                    "average_time_per_run_ms",
                    "tokens_per_second",
                    "average_time_per_token_ms",
                ),
            ),
            "generation": build_benchmark_section_summary(
                collected_metadata,
                "generation",
                "llm_generation",
                (
                    "generated_tokens",
                    "average_tokens_per_run",
                    "tokens_per_second",
                    "average_time_per_token_ms",
                ),
            ),
            "actual_output_token_count": numeric_series_stats(
                [
                    float(metadata["actual_output_token_count"])
                    for metadata in collected_metadata
                    if metadata.get("actual_output_token_count") is not None
                ]
            ),
        },
    }

    summary_path = benchmark_summary_path(config, phase)
    write_json(summary_path, benchmark_summary)

    profiles = config.setdefault("profiles", {})
    phase_profile = profiles.setdefault(phase, {})
    phase_profile["benchmark_summary_json"] = str(summary_path)
    phase_profile["benchmark_repeats"] = int(args.repeats)
    phase_profile["benchmark_warmup_runs"] = int(args.warmup_runs)
    phase_profile["benchmark_runtime_profile_jsons"] = [
        str(metadata.get("runtime_profile_json"))
        for metadata in collected_metadata
        if metadata.get("runtime_profile_json")
    ]
    phase_profile["benchmark_metadata_jsons"] = [str(run["metadata_json"]) for run in run_summaries]
    phase_profile["last_runtime_metadata"] = collected_metadata[-1]
    phase_profile["last_output_json"] = str(collected_metadata[-1]["output_file"])
    if collected_metadata[-1].get("runtime_profile_json"):
        phase_profile["last_runtime_profile_json"] = str(collected_metadata[-1]["runtime_profile_json"])
    save_run_config(config)

    log(f"Saved {phase} benchmark summary to {summary_path}")
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
    if sqlite_path.exists():
        return sqlite_path
    if sqlite_prefix.exists() and sqlite_prefix.suffix != ".sqlite":
        return sqlite_prefix
    fail(f"Expected exported SQLite report was not created at {sqlite_path} or {sqlite_prefix}")


def sqlite_table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def summarize_nsys_sqlite(sqlite_path: Path, top_limit: int = 20) -> Dict[str, Any]:
    with sqlite3.connect(str(sqlite_path)) as connection:
        connection.row_factory = sqlite3.Row

        def sum_duration_ns(table_name: str) -> int:
            if not sqlite_table_exists(connection, table_name):
                return 0
            row = connection.execute(
                f"SELECT COALESCE(SUM(end - start), 0) AS total_ns FROM {table_name}"
            ).fetchone()
            return int(row["total_ns"] or 0) if row else 0

        def count_rows(table_name: str) -> int:
            if not sqlite_table_exists(connection, table_name):
                return 0
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
            return int(row["count"] or 0) if row else 0

        top_kernels: List[Dict[str, Any]] = []
        if sqlite_table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
            rows = connection.execute(
                """
                SELECT
                    COALESCE(demangled.value, short_name.value, '<unknown>') AS kernel_name,
                    COUNT(*) AS launch_count,
                    SUM(kernel.end - kernel.start) AS total_duration_ns,
                    AVG(kernel.end - kernel.start) AS avg_duration_ns,
                    MAX(kernel.end - kernel.start) AS max_duration_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
                LEFT JOIN StringIds AS demangled ON kernel.demangledName = demangled.id
                LEFT JOIN StringIds AS short_name ON kernel.shortName = short_name.id
                GROUP BY kernel_name
                ORDER BY total_duration_ns DESC
                LIMIT ?
                """,
                (top_limit,),
            ).fetchall()
            for row in rows:
                total_duration_ns = int(row["total_duration_ns"] or 0)
                avg_duration_ns = float(row["avg_duration_ns"] or 0.0)
                max_duration_ns = int(row["max_duration_ns"] or 0)
                top_kernels.append(
                    {
                        "kernel_name": str(row["kernel_name"]),
                        "launch_count": int(row["launch_count"] or 0),
                        "total_duration_ns": total_duration_ns,
                        "total_duration_ms": total_duration_ns / 1_000_000.0,
                        "avg_duration_ns": avg_duration_ns,
                        "avg_duration_ms": avg_duration_ns / 1_000_000.0,
                        "max_duration_ns": max_duration_ns,
                        "max_duration_ms": max_duration_ns / 1_000_000.0,
                    }
                )

        top_memcpy: List[Dict[str, Any]] = []
        if sqlite_table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            rows = connection.execute(
                """
                SELECT
                    memcpy.start AS start_ns,
                    memcpy.end AS end_ns,
                    (memcpy.end - memcpy.start) AS duration_ns,
                    memcpy.bytes AS bytes,
                    memcpy.streamId AS stream_id,
                    memcpy.contextId AS context_id,
                    COALESCE(copy_kind.label, CAST(memcpy.copyKind AS TEXT)) AS copy_kind
                FROM CUPTI_ACTIVITY_KIND_MEMCPY AS memcpy
                LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS copy_kind ON memcpy.copyKind = copy_kind.id
                ORDER BY duration_ns DESC
                LIMIT ?
                """,
                (top_limit,),
            ).fetchall()
            for row in rows:
                duration_ns = int(row["duration_ns"] or 0)
                byte_count = int(row["bytes"] or 0)
                throughput_mb_per_s = (
                    (byte_count / 1_000_000.0) / (duration_ns / 1_000_000_000.0)
                    if duration_ns > 0
                    else 0.0
                )
                top_memcpy.append(
                    {
                        "copy_kind": str(row["copy_kind"]),
                        "bytes": byte_count,
                        "duration_ns": duration_ns,
                        "duration_ms": duration_ns / 1_000_000.0,
                        "throughput_mb_per_s": throughput_mb_per_s,
                        "stream_id": int(row["stream_id"] or 0),
                        "context_id": int(row["context_id"] or 0),
                        "start_ns": int(row["start_ns"] or 0),
                        "end_ns": int(row["end_ns"] or 0),
                    }
                )

        top_memset: List[Dict[str, Any]] = []
        if sqlite_table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMSET"):
            rows = connection.execute(
                """
                SELECT
                    memset.start AS start_ns,
                    memset.end AS end_ns,
                    (memset.end - memset.start) AS duration_ns,
                    memset.bytes AS bytes,
                    memset.value AS value,
                    memset.streamId AS stream_id,
                    memset.contextId AS context_id
                FROM CUPTI_ACTIVITY_KIND_MEMSET AS memset
                ORDER BY duration_ns DESC
                LIMIT ?
                """,
                (top_limit,),
            ).fetchall()
            for row in rows:
                duration_ns = int(row["duration_ns"] or 0)
                byte_count = int(row["bytes"] or 0)
                throughput_mb_per_s = (
                    (byte_count / 1_000_000.0) / (duration_ns / 1_000_000_000.0)
                    if duration_ns > 0
                    else 0.0
                )
                top_memset.append(
                    {
                        "bytes": byte_count,
                        "value": int(row["value"] or 0),
                        "duration_ns": duration_ns,
                        "duration_ms": duration_ns / 1_000_000.0,
                        "throughput_mb_per_s": throughput_mb_per_s,
                        "stream_id": int(row["stream_id"] or 0),
                        "context_id": int(row["context_id"] or 0),
                        "start_ns": int(row["start_ns"] or 0),
                        "end_ns": int(row["end_ns"] or 0),
                    }
                )

        kernel_total_ns = sum_duration_ns("CUPTI_ACTIVITY_KIND_KERNEL")
        memcpy_total_ns = sum_duration_ns("CUPTI_ACTIVITY_KIND_MEMCPY")
        memset_total_ns = sum_duration_ns("CUPTI_ACTIVITY_KIND_MEMSET")
        total_gpu_activity_ns = kernel_total_ns + memcpy_total_ns + memset_total_ns

        return {
            "gpu_activity": {
                "kernel_launch_count": count_rows("CUPTI_ACTIVITY_KIND_KERNEL"),
                "memcpy_count": count_rows("CUPTI_ACTIVITY_KIND_MEMCPY"),
                "memset_count": count_rows("CUPTI_ACTIVITY_KIND_MEMSET"),
                "total_kernel_gpu_time_ns": kernel_total_ns,
                "total_kernel_gpu_time_ms": kernel_total_ns / 1_000_000.0,
                "total_memcpy_gpu_time_ns": memcpy_total_ns,
                "total_memcpy_gpu_time_ms": memcpy_total_ns / 1_000_000.0,
                "total_memset_gpu_time_ns": memset_total_ns,
                "total_memset_gpu_time_ms": memset_total_ns / 1_000_000.0,
                "total_captured_gpu_time_ns": total_gpu_activity_ns,
                "total_captured_gpu_time_ms": total_gpu_activity_ns / 1_000_000.0,
            },
            "top_kernels": top_kernels,
            "top_memcpy": top_memcpy,
            "top_memset": top_memset,
        }


def summarize_nsys_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    runtime = load_selected_runtime(config)
    report_path = Path(args.report).resolve()
    if not report_path.exists():
        fail(f"Nsight Systems report not found: {report_path}")

    sqlite_path = export_nsys_sqlite(report_path)
    phase_workload = copy.deepcopy(config["phase_workloads"][args.phase])
    nsys_activity_summary = summarize_nsys_sqlite(sqlite_path)
    summary = {
        "phase": args.phase,
        "workflow_backend": WORKFLOW_BACKEND,
        "workflow_id": WORKFLOW_ID,
        "requested_model_id": runtime["requested_model_id"],
        "actual_model_id": runtime["actual_model_id"],
        "precision": runtime["precision"],
        "report_path": str(report_path),
        "sqlite_path": str(sqlite_path),
        "phase_workload": phase_workload,
        "nsys_activity_summary": nsys_activity_summary,
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
            "phase_definition": phase_workload,
            "nsys_activity_summary": nsys_activity_summary,
        }
    )
    save_run_config(config)

    log(f"Saved {args.phase} phase summary to {summary_path}")
    return 0


def register_ncu_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    profiles = config.setdefault("profiles", {})
    phase_profile = profiles.setdefault(args.phase, {})
    report_path = Path(args.report).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else None
    runtime_profile_path = Path(args.runtime_profile).resolve() if args.runtime_profile else None

    if not report_path.exists():
        fail(f"Nsight Compute report was not found: {report_path}")

    ncu_summary = summarize_ncu_report(report_path)
    registered_at = time.strftime("%Y-%m-%d %H:%M:%S")

    metadata_payload: Dict[str, Any] = {}
    if metadata_path and metadata_path.exists():
        metadata_payload = read_json(metadata_path)

    requested_max_new_tokens = args.requested_max_new_tokens
    if requested_max_new_tokens is None:
        requested_max_new_tokens = metadata_payload.get("requested_max_generate_length")
    if requested_max_new_tokens is None:
        requested_max_new_tokens = config["phase_workloads"][args.phase].get("max_generate_length")

    actual_output_token_count = metadata_payload.get("actual_output_token_count")

    if metadata_path:
        metadata_payload["collection_backend"] = args.collection_backend
        metadata_payload["replay_mode"] = args.replay_mode
        metadata_payload["collection_profile"] = args.collection_profile
        metadata_payload["phase_filter"] = args.phase_filter
        metadata_payload["requested_max_generate_length"] = requested_max_new_tokens
        metadata_payload["actual_output_token_count"] = actual_output_token_count
        metadata_payload["report_created_at"] = registered_at
        metadata_payload["ncu_summary"] = ncu_summary
        if runtime_profile_path is not None:
            metadata_payload["runtime_profile_json"] = str(runtime_profile_path)
        write_json(metadata_path, metadata_payload)

    phase_profile["ncu_rep"] = str(report_path)
    phase_profile["ncu_registered_at"] = registered_at
    phase_profile["ncu_collection_backend"] = args.collection_backend
    phase_profile["ncu_replay_mode"] = args.replay_mode
    phase_profile["ncu_collection_profile"] = args.collection_profile
    phase_profile["ncu_phase_filter"] = args.phase_filter
    phase_profile["ncu_requested_max_generate_length"] = requested_max_new_tokens
    phase_profile["ncu_actual_output_token_count"] = actual_output_token_count
    phase_profile["ncu_summary"] = ncu_summary
    if metadata_path:
        phase_profile["ncu_metadata_json"] = str(metadata_path)
    if runtime_profile_path is not None:
        phase_profile["ncu_runtime_profile_json"] = str(runtime_profile_path)

    save_run_config(config)
    return 0


def report_config_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    runtime = config.get("selected_runtime", {})
    host_export = config.get("host_export", {})
    engine_build = config.get("engine_build", {})

    lines = [
        "TensorRT Edge-LLM Profiling Run Configuration",
        f"Backend: {config.get('workflow_backend')}",
        f"Workflow ID: {config.get('workflow_id')}",
        f"Requested model: {config.get('requested_model_id', REQUESTED_MODEL_ID)}",
        f"Export precision: {config.get('export_precision', EXPORT_PRECISION)}",
        f"Edge-LLM repo: {config.get('edge_llm', {}).get('repo_root', default_edge_llm_repo_root())}",
    ]

    if host_export:
        lines.extend(
            [
                "",
                "Host export:",
                f"  bundle: {host_export.get('bundle_path', 'not packaged yet')}",
                f"  onnx_dir: {host_export.get('onnx_dir', 'not exported yet')}",
                f"  hf_assets_dir: {host_export.get('hf_assets_dir', 'not exported yet')}",
            ]
        )

    if engine_build:
        build_cfg = engine_build.get("engine_build_config", {})
        lines.extend(
            [
                "",
                "Engine build:",
                f"  engine_dir: {engine_build.get('engine_dir')}",
                f"  llm_build: {engine_build.get('llm_build_path')}",
                f"  maxBatchSize: {build_cfg.get('maxBatchSize')}",
                f"  maxInputLen: {build_cfg.get('maxInputLen')}",
                f"  maxKVCacheCapacity: {build_cfg.get('maxKVCacheCapacity')}",
                f"  precision: {build_cfg.get('precision')}",
            ]
        )

    if runtime:
        lines.extend(
            [
                "",
                "Selected runtime:",
                f"  engine_dir: {runtime.get('engine_dir')}",
                f"  llm_inference: {runtime.get('llm_inference_path')}",
                f"  actual_model_id: {runtime.get('actual_model_id')}",
            ]
        )

    lines.append("")
    lines.append("Phase workloads:")
    for phase in ("prefill", "decode"):
        phase_payload = config["phase_workloads"].get(phase, {})
        lines.extend(
            [
                f"  {phase}:",
                f"    input_json: {phase_payload.get('input_json', 'not generated yet')}",
                f"    prompt_kind: {phase_payload.get('resolved_prompt_kind', phase_payload.get('prompt_kind'))}",
                f"    input_token_count: {phase_payload.get('input_token_count', 'unknown')}",
                f"    max_generate_length: {phase_payload.get('max_generate_length')}",
            ]
        )

    profiles = config.get("profiles", {})
    for phase in ("prefill", "decode"):
        phase_profile = profiles.get(phase, {})
        if not phase_profile:
            continue
        lines.extend(
            [
                "",
                f"{phase.title()} artifacts:",
                f"  nsys_rep: {phase_profile.get('nsys_rep', '-')}",
                f"  ncu_rep: {phase_profile.get('ncu_rep', '-')}",
                f"  ncu_metadata_json: {phase_profile.get('ncu_metadata_json', '-')}",
            ]
        )
        ncu_summary = phase_profile.get("ncu_summary")
        if isinstance(ncu_summary, dict):
            lines.extend(
                [
                    f"  ncu_kernel_ids: {ncu_summary.get('kernel_id_count')}",
                    f"  ncu_total_gpu_time_ms: {round(ncu_summary.get('total_gpu_time_ns', 0) / 1_000_000.0, 3)}",
                    f"  ncu_family_counts: {ncu_summary.get('family_counts')}",
                ]
            )

    report_text = "\n".join(lines) + "\n"
    print(report_text, end="")

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.write_text(report_text)
        config.setdefault("reports", {})
        config["reports"]["human_summary"] = str(output_path)
        save_run_config(config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "host-preflight",
        help="Validate export prerequisites and gated model access on the current Linux machine.",
    )

    host_export = subparsers.add_parser(
        "host-export",
        help="Export Llama 3.1 8B Instruct to ONNX with TensorRT Edge-LLM in FP16.",
    )
    host_export.add_argument(
        "--force",
        action="store_true",
        help="Clear the generated ONNX and tokenizer assets before exporting again.",
    )

    subparsers.add_parser(
        "host-package-export",
        help="Package the exported ONNX bundle and tokenizer assets into a tarball.",
    )

    target_preflight = subparsers.add_parser(
        "target-preflight",
        help="Validate the DGX Spark / Thor target environment and build the local Edge-LLM C++ runtime.",
    )
    target_preflight.add_argument("--repo-root", default=None)
    target_preflight.add_argument("--force-reconfigure", action="store_true")
    target_preflight.add_argument("--force-rebuild", action="store_true")
    target_preflight.add_argument("--jobs", type=int, default=None)

    target_fetch = subparsers.add_parser(
        "target-fetch-export",
        help="Fetch an exported ONNX bundle from an explicit local path or user@host:/abs/path source.",
    )
    target_fetch.add_argument("--source", default=None)
    target_fetch.add_argument("--force", action="store_true")

    build_engine = subparsers.add_parser(
        "build-engine",
        help="Build a TensorRT Edge-LLM engine on the Thor target from the imported ONNX bundle.",
    )
    build_engine.add_argument("--repo-root", default=None)
    build_engine.add_argument("--force", action="store_true")

    run_inference = subparsers.add_parser(
        "run-inference",
        help="Run llm_inference using the selected Edge-LLM engine and a generated phase workload.",
    )
    run_inference.add_argument("--phase", choices=["prefill", "decode"], default="decode")
    run_inference.add_argument("--input-file", default=None)
    run_inference.add_argument("--output-file", default=None)
    run_inference.add_argument("--metadata-output", default=None)
    run_inference.add_argument("--max-generate-length", type=int, default=None)
    run_inference.add_argument("--prompt", default=None)
    run_inference.add_argument("--warmup-runs", type=int, default=0)
    run_inference.add_argument("--runtime-profile-output", default=None)

    benchmark_phase = subparsers.add_parser(
        "benchmark-phase",
        help="Run repeated llm_inference measurements with native phase metrics.",
    )
    benchmark_phase.add_argument("--phase", choices=["prefill", "decode"], default="decode")
    benchmark_phase.add_argument("--repeats", type=int, default=BENCHMARK_REPEATS_DEFAULT)
    benchmark_phase.add_argument("--warmup-runs", type=int, default=BENCHMARK_WARMUP_RUNS_DEFAULT)
    benchmark_phase.add_argument("--max-generate-length", type=int, default=None)
    benchmark_phase.add_argument("--prompt", default=None)

    summarize_nsys = subparsers.add_parser(
        "summarize-nsys",
        help="Export a .nsys-rep file to SQLite and attach workload-defined phase metadata.",
    )
    summarize_nsys.add_argument("--phase", choices=["prefill", "decode"], required=True)
    summarize_nsys.add_argument("--report", required=True)

    register_ncu = subparsers.add_parser(
        "register-ncu",
        help="Record an .ncu-rep path and summary in artifacts/run_config.json.",
    )
    register_ncu.add_argument("--phase", choices=["prefill", "decode"], required=True)
    register_ncu.add_argument("--report", required=True)
    register_ncu.add_argument("--metadata", default=None)
    register_ncu.add_argument("--collection-backend", required=True)
    register_ncu.add_argument("--replay-mode", required=True)
    register_ncu.add_argument("--collection-profile", required=True)
    register_ncu.add_argument("--requested-max-new-tokens", type=int, default=None)
    register_ncu.add_argument("--phase-filter", default=None)
    register_ncu.add_argument("--runtime-profile", default=None)

    inspect_ncu = subparsers.add_parser(
        "inspect-ncu",
        help="Query an .ncu-rep report through a cache-backed CLI instead of the GUI.",
    )
    inspect_ncu.add_argument("--report", default=None)
    inspect_ncu.add_argument("--phase", choices=["prefill", "decode"], default=None)
    inspect_ncu.add_argument(
        "--view",
        choices=["summary", "kernels", "launches"],
        default="summary",
    )
    inspect_ncu.add_argument("--family", choices=["all", *NCU_FAMILIES], default="all")
    inspect_ncu.add_argument("--name-contains", default=None)
    inspect_ncu.add_argument(
        "--sort",
        choices=["total_ms", "avg_ms", "count", "sm_pct", "dram_pct", "warps_pct", "id"],
        default="total_ms",
    )
    inspect_ncu.add_argument("--limit", type=int, default=10)
    inspect_ncu.add_argument("--refresh-cache", action="store_true")
    inspect_ncu.add_argument("--format", choices=["table", "json"], default="table")

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
        if args.command == "host-preflight":
            return host_preflight_command(args)
        if args.command == "host-export":
            return host_export_command(args)
        if args.command == "host-package-export":
            return host_package_export_command(args)
        if args.command == "target-preflight":
            return target_preflight_command(args)
        if args.command == "target-fetch-export":
            return target_fetch_export_command(args)
        if args.command == "build-engine":
            return build_engine_command(args)
        if args.command == "run-inference":
            return run_inference_command(args)
        if args.command == "benchmark-phase":
            return benchmark_phase_command(args)
        if args.command == "summarize-nsys":
            return summarize_nsys_command(args)
        if args.command == "register-ncu":
            return register_ncu_command(args)
        if args.command == "inspect-ncu":
            return inspect_ncu_command(args)
        if args.command == "report-config":
            return report_config_command(args)
    except WorkflowError as exc:
        log(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        log(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        return 1

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
