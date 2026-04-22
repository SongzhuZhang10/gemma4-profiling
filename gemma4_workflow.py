#!/usr/bin/env python3
"""Helper entrypoints for the Gemma 4 GPU profiling workflow."""

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
from collections import Counter, defaultdict
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

PRECISION_ORDER = ["FP16", "BF16"]
DISABLED_PRECISIONS = {"FP32", "INT8"}
PRECISION_TO_DTYPE = {
    "FP16": "float16",
    "FP32": "float32",
    "BF16": "bfloat16",
}


def sanitize_precision_order(precision_order: Optional[Iterable[str]]) -> List[str]:
    requested_order = list(precision_order) if precision_order is not None else list(PRECISION_ORDER)
    sanitized: List[str] = []

    for precision in requested_order:
        if precision in DISABLED_PRECISIONS or precision not in PRECISION_TO_DTYPE:
            continue
        if precision not in sanitized:
            sanitized.append(precision)

    return sanitized or list(PRECISION_ORDER)

FORWARD_STEP_PATTERN = re.compile(
    r"_forward_step\s+(\d+):\s+(\d+)\s+ctx reqs,\s+(\d+)\s+gen reqs"
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

    import csv
    import io

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


def ncu_slice_cache_path(report_path: Path) -> Path:
    if report_path.name.endswith(".ncu-rep"):
        return report_path.with_name(report_path.name[:-8] + ".ncu-slice.json")
    return report_path.with_suffix(report_path.suffix + ".ncu-slice.json")


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

    import csv
    import io

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


def default_run_config() -> Dict[str, Any]:
    return {
        "requested_model_id": REQUESTED_MODEL_ID,
        "workload": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "prompt": DEFAULT_PROMPT,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        },
        "precision_probe_order": sanitize_precision_order(None),
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
    config["precision_probe_order"] = sanitize_precision_order(
        config.get("precision_probe_order")
    )
    return config


def save_run_config(config: Dict[str, Any]) -> None:
    config["precision_probe_order"] = sanitize_precision_order(
        config.get("precision_probe_order")
    )
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
            f"Package metadata for tensorrt_llm was not found for interpreter {sys.executable}."
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

    if not host_info["ncu_path"] or not host_info["nsys_path"]:
        fail("Both ncu and nsys must be available before profiling.")
    if not python_stack.get("torch_cuda_available"):
        fail(f"torch.cuda.is_available() is false for interpreter {sys.executable}.")
    if not tllm_support["installed"]:
        fail(f"tensorrt_llm is not installed for interpreter {sys.executable}.")
    if not tllm_support.get("importable"):
        if tllm_support.get("missing_openmpi_runtime"):
            fail(
                "TensorRT-LLM is installed but cannot be imported because libmpi.so.40 "
                "(the OpenMPI runtime library) is missing. Install libopenmpi3 and "
                "openmpi-bin system-wide, then rerun 01_install.sh."
            )
        detail = "; ".join(tllm_support.get("errors", [])) or "unknown import error"
        fail(
            f"TensorRT-LLM is installed but cannot be imported for interpreter {sys.executable}: "
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
    precision_probe_order = sanitize_precision_order(config.get("precision_probe_order"))
    config["precision_probe_order"] = precision_probe_order
    if "prepared_models" not in config:
        prepare_model_command(argparse.Namespace())
        config = load_run_config()
        precision_probe_order = sanitize_precision_order(config.get("precision_probe_order"))
        config["precision_probe_order"] = precision_probe_order

    selected = config.get("selected_runtime")
    if selected and selected.get("precision") not in precision_probe_order:
        log(
            "Existing runtime selection uses disabled precision "
            f"{selected.get('precision')}; re-probing allowed precisions."
        )
        config.pop("selected_runtime", None)
        selected = None
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

        for precision in precision_probe_order:
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
    precision_probe_order = sanitize_precision_order(config.get("precision_probe_order"))
    if runtime.get("precision") not in precision_probe_order:
        fail(
            "selected_runtime uses disabled precision "
            f"{runtime.get('precision')}. Run 03_build_or_prepare_runtime.sh again "
            f"to select one of {precision_probe_order}."
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

    nvtx_tag = getattr(args, "nvtx_tag", None)

    started = time.time()
    with AutoDeployLLM(
        model=runtime["actual_model_id"],
        tokenizer=runtime["actual_model_id"],
        dtype=runtime["dtype"],
        yaml_extra=[runtime["yaml_path"]],
        trust_remote_code=True,
    ) as llm:
        if nvtx_tag:
            import torch.cuda.nvtx as cuda_nvtx
            cuda_nvtx.range_push(nvtx_tag)

        result = llm.generate(
            prompt_text,
            sampling_params=SamplingParams(max_tokens=max_new_tokens),
            use_tqdm=False,
        )

        if nvtx_tag:
            cuda_nvtx.range_pop()
        metadata = extract_output_metadata(result)

    metadata.update(
        {
            "actual_output_token_count": metadata.get("output_token_count"),
            "model_id": runtime["actual_model_id"],
            "requested_model_id": runtime["requested_model_id"],
            "requested_max_new_tokens": max_new_tokens,
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
    # Newer nsys versions (2024.6+) write to the exact --output path without
    # appending .sqlite, while older versions append .sqlite automatically.
    sqlite_path = sqlite_prefix.with_suffix(".sqlite")
    if sqlite_path.exists():
        return sqlite_path
    if sqlite_prefix.exists() and sqlite_prefix.suffix != ".sqlite":
        return sqlite_prefix
    fail(f"Expected exported SQLite report was not created at {sqlite_path} or {sqlite_prefix}")


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
    # Nsight Compute expects start/end range names to be passed literally, but
    # filter-quantifier characters in the range text itself must be escaped.
    # TensorRT-LLM emits labels like "[Executor] _forward_step ... , ..." which
    # otherwise get misparsed as push/pop syntax or range delimiters.
    escaped = step_text.replace("\\", "\\\\")
    for char in ("@", ",", "[", "]", "/", "*", "+"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


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
    if not phase_profile:
        fail(
            f"No NVTX filter is recorded for phase '{args.phase}'. Run the corresponding Nsight Systems script first."
        )

    selected_step_text = phase_profile.get("selected_step_text")
    if selected_step_text:
        print(nvtx_filter_for_text(selected_step_text))
        return 0

    stored_filter = phase_profile.get("selected_nvtx_include_filter")
    if not stored_filter:
        fail(
            f"No NVTX filter is recorded for phase '{args.phase}'. Run the corresponding Nsight Systems script first."
        )
    print(stored_filter)
    return 0


def register_ncu_command(args: argparse.Namespace) -> int:
    config = load_run_config()
    profiles = config.setdefault("profiles", {})
    phase_profile = profiles.setdefault(args.phase, {})
    report_path = Path(args.report).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else None
    ncu_summary = summarize_ncu_report(report_path)
    registered_at = time.strftime("%Y-%m-%d %H:%M:%S")

    metadata_payload: Dict[str, Any] = {}
    if metadata_path and metadata_path.exists():
        metadata_payload = read_json(metadata_path)

    requested_max_new_tokens = args.requested_max_new_tokens
    if requested_max_new_tokens is None:
        requested_max_new_tokens = metadata_payload.get("requested_max_new_tokens")
    if requested_max_new_tokens is None:
        requested_max_new_tokens = metadata_payload.get("max_new_tokens")

    if requested_max_new_tokens is None:
        requested_max_new_tokens = 0 if args.phase == "prefill" else None

    actual_output_token_count = metadata_payload.get("actual_output_token_count")
    if actual_output_token_count is None:
        actual_output_token_count = metadata_payload.get("output_token_count")
    if actual_output_token_count is None and metadata_payload.get("direct_inference"):
        actual_output_token_count = 0 if args.phase == "prefill" else requested_max_new_tokens

    if metadata_path:
        metadata_payload["collection_backend"] = args.collection_backend
        metadata_payload["replay_mode"] = args.replay_mode
        metadata_payload["collection_profile"] = args.collection_profile
        metadata_payload["requested_max_new_tokens"] = requested_max_new_tokens
        metadata_payload["actual_output_token_count"] = actual_output_token_count
        metadata_payload["report_created_at"] = registered_at
        metadata_payload["ncu_summary"] = ncu_summary
        if metadata_path.exists() or metadata_payload:
            write_json(metadata_path, metadata_payload)

    phase_profile["ncu_rep"] = str(report_path)
    phase_profile["ncu_registered_at"] = registered_at
    phase_profile["ncu_collection_backend"] = args.collection_backend
    phase_profile["ncu_replay_mode"] = args.replay_mode
    phase_profile["ncu_collection_profile"] = args.collection_profile
    phase_profile["ncu_requested_max_new_tokens"] = requested_max_new_tokens
    phase_profile["ncu_actual_output_token_count"] = actual_output_token_count
    phase_profile["ncu_summary"] = ncu_summary
    if metadata_path:
        phase_profile["ncu_metadata_json"] = str(metadata_path)
    if "proxy_fallback" in metadata_payload:
        phase_profile["ncu_proxy_fallback"] = metadata_payload["proxy_fallback"]
    if "direct_inference" in metadata_payload:
        phase_profile["ncu_direct_inference"] = metadata_payload["direct_inference"]
    if "execution_path" in metadata_payload:
        phase_profile["ncu_execution_path"] = metadata_payload["execution_path"]
    if "execution_path_detail" in metadata_payload:
        phase_profile["ncu_execution_path_detail"] = metadata_payload["execution_path_detail"]
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
            for key in ("nsys_rep", "nsys_sqlite", "summary_json", "ncu_rep", "ncu_metadata_json"):
                if phase_profile.get(key):
                    lines.append(f"  - {key}: {phase_profile[key]}")
            if phase_profile.get("ncu_collection_backend"):
                lines.append(f"  - ncu_backend: {phase_profile['ncu_collection_backend']}")
            if phase_profile.get("ncu_replay_mode"):
                lines.append(f"  - ncu_replay_mode: {phase_profile['ncu_replay_mode']}")
            if phase_profile.get("ncu_collection_profile"):
                lines.append(f"  - ncu_collection_profile: {phase_profile['ncu_collection_profile']}")
            if "ncu_proxy_fallback" in phase_profile:
                lines.append(f"  - ncu_proxy_fallback: {phase_profile['ncu_proxy_fallback']}")
            if phase_profile.get("ncu_execution_path"):
                lines.append(f"  - ncu_execution_path: {phase_profile['ncu_execution_path']}")
            if phase_profile.get("ncu_execution_path_detail"):
                lines.append(
                    f"  - ncu_execution_path_detail: {phase_profile['ncu_execution_path_detail']}"
                )
            if phase_profile.get("ncu_requested_max_new_tokens") is not None:
                lines.append(
                    f"  - ncu_requested_max_new_tokens: {phase_profile['ncu_requested_max_new_tokens']}"
                )
            if phase_profile.get("ncu_actual_output_token_count") is not None:
                lines.append(
                    f"  - ncu_actual_output_token_count: {phase_profile['ncu_actual_output_token_count']}"
                )
            ncu_summary = phase_profile.get("ncu_summary")
            if isinstance(ncu_summary, dict):
                lines.append(
                    f"  - ncu_kernel_ids: {ncu_summary.get('kernel_id_count')}"
                )
                lines.append(
                    f"  - ncu_total_gpu_time_ms: "
                    f"{round(ncu_summary.get('total_gpu_time_ns', 0) / 1_000_000.0, 3)}"
                )
                family_counts = ncu_summary.get("family_counts")
                if isinstance(family_counts, dict) and family_counts:
                    lines.append(f"  - ncu_family_counts: {family_counts}")

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
    run_inference.add_argument(
        "--nvtx-tag",
        type=str,
        default=None,
        help="Wrap llm.generate() in an NVTX push/pop range with this name for ncu filtering.",
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
    register_ncu.add_argument("--metadata", default=None)
    register_ncu.add_argument("--collection-backend", required=True)
    register_ncu.add_argument("--replay-mode", required=True)
    register_ncu.add_argument("--collection-profile", required=True)
    register_ncu.add_argument("--requested-max-new-tokens", type=int, default=None)

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
    inspect_ncu.add_argument(
        "--family",
        choices=["all", *NCU_FAMILIES],
        default="all",
    )
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
        if args.command == "inspect-ncu":
            return inspect_ncu_command(args)
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
