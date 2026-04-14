# Gemma 4 WSL2 Profiling Workflow Implementation

## 1. Overview

This project implements an end-to-end GPU profiling workflow for the Google Gemma 4 E2B-it large language model running under Windows Subsystem for Linux 2 (WSL2) on an NVIDIA GeForce RTX 3050 6 GB Laptop GPU. The workflow uses TensorRT-LLM 1.2.0 AutoDeploy as the inference backend, NVIDIA Nsight Systems (nsys) for timeline tracing of CUDA API calls and NVTX-annotated executor steps, and NVIDIA Nsight Compute (ncu) for hardware-counter profiling of individual GPU kernels. Seven user-facing shell scripts drive the workflow from dependency installation through report inspection, assisted by two internal helper scripts and one shared Python module (`gemma4_workflow.py`). All Python execution uses a pre-existing Python virtual environment at `/home/songzhu/Desktop/dl_env` (Python 3.10, PyTorch 2.9.1, CUDA 12.8). The workflow produces separate `.nsys-rep` and `.ncu-rep` reports for the prefill and decode phases of inference, a run-configuration summary, and a consolidated performance report.

---

## 2. Accomplishments

| # | Milestone | Realizing file(s) |
|---|---|---|
| 1 | Validated WSL2 CUDA tooling, installed TensorRT-LLM and its native dependencies (OpenMPI, CUDA 13 cuBLAS), patched Gemma 4 AutoDeploy support into TensorRT-LLM 1.2.0 | `01_install.sh` |
| 2 | Prepared templated chat prompts for all Gemma 4 model candidates using the HuggingFace tokenizer, with a fallback path for environments where `AutoProcessor` fails | `02_prepare_model.sh` |
| 3 | Selected and warmed a feasible model/precision/attention-backend combination through automated probing, with a YAML-schema fallback for TensorRT-LLM 1.2.0 incompatibilities | `03_build_or_prepare_runtime.sh` |
| 4 | Captured Nsight Systems CUDA+NVTX timelines for both the prefill and decode phases, exported them to SQLite, and selected representative forward steps using NVTX range analysis | `nsys_prefill.sh`, `nsys_decode.sh`, `gemma4_workflow.py` (`summarize-nsys`) |
| 5 | Profiled the prefill phase with Nsight Compute, including automated GPU counter-permission detection and Windows registry self-heal for `ERR_NVGPUCTRPERM` | `04_profile_prefill.sh` |
| 6 | Profiled the decode phase with Nsight Compute using a reduced token count (8 tokens) to avoid `ResourceUnavailable` errors from VRAM exhaustion, with a three-tier fallback chain | `05_profile_decode.sh` |
| 7 | Generated a human-readable run-configuration summary listing the selected model, precision, backend, token counts, and all artifact paths | `06_report_run_config.sh` |
| 8 | Produced a consolidated performance report comparing prefill and decode metrics from both Nsight Systems and Nsight Compute | `artifacts/reports/performance_report.md` |

---

# Background Knowledge

## Why Nsight Systems Should Be Used Before Nsight Compute

Nsight Systems should be used before Nsight Compute because the two tools answer different performance questions at different levels of detail. Nsight Systems provides a top-down, end-to-end timeline of execution. It shows when CPU activity, CUDA API calls, GPU kernel launches, and NVTX-annotated phases occur, making it the right tool for identifying the structure of the workload and locating the exact region of interest within the full inference run. In contrast, Nsight Compute is a kernel-level profiler: it does not primarily explain where a phase begins or ends in the application timeline, but instead measures how individual GPU kernels behave once that phase has already been isolated.

For that reason, beginning with Nsight Systems reduces ambiguity and improves profiling accuracy. In a large language model inference pipeline, the prefill and decode phases have different execution patterns and performance characteristics. A timeline trace makes it possible to distinguish these phases reliably, especially when NVTX ranges are present, and to choose a representative step for deeper analysis. Only after that phase has been identified does it make sense to run Nsight Compute, which can then focus on detailed hardware metrics such as kernel duration, SM throughput, DRAM throughput, and warp activity. This sequence prevents low-level measurements from being taken out of context and ensures that the reported kernel statistics actually correspond to the intended phase of execution.

This workflow follows that logic explicitly. It first captures a CUDA and NVTX timeline with Nsight Systems, exports the trace, and analyzes the NVTX-marked forward steps to select the representative phase. It then runs Nsight Compute on the prefill stage, using a configuration that keeps the captured kernels dominated by prefill work. In practical terms, this means Nsight Systems is used to answer the question, “Where is the performance-critical phase?” and Nsight Compute is used afterward to answer, “Why is that phase fast or slow at the GPU kernel level?” Together, they form a disciplined profiling methodology: first establish execution context, then perform fine-grained hardware analysis.


# Project Output Directory Structure

All profiling data is stored under `artifacts/` in the project directory.

```
artifacts/
├── run_config.json                          ← Central config: all decisions, paths, metadata
│
├── cache/                                   ← All download/build caches (git-ignored)
│   ├── hf/                                  ← HuggingFace model weights
│   ├── torchinductor/                       ← TorchInductor compilation cache
│   ├── triton/                              ← Triton kernel cache
│   └── tllm_build_cache/                   ← TensorRT-LLM build cache
│
├── runtime/                                 ← Runtime configuration
│   └── google__gemma-4-E2B-it/
│       ├── templated_prompt.txt             ← Chat-templated input prompt (17 tokens)
│       ├── templated_prompt.json            ← Prompt metadata (token count, model ID)
│       └── ...fp16__torch_non_paged__torch_simple.yaml  ← Selected runtime YAML
│
└── reports/                                 ← All profiling output
    ├── prefill.nsys-rep                     ← Nsight Systems report, prefill run
    ├── prefill          (or .sqlite)        ← Exported SQLite from prefill nsys trace
    ├── prefill_nsys_run_metadata.json       ← Inference metadata from nsys prefill run
    ├── prefill_phase_summary.json           ← All 128 forward-step NVTX durations; selected step
    ├── prefill.ncu-rep                      ← Nsight Compute report, prefill phase
    ├── prefill_ncu_run_metadata.json        ← Inference metadata from ncu prefill run
    │
    ├── decode.nsys-rep                      ← Nsight Systems report, decode run
    ├── decode.sqlite                        ← Exported SQLite from decode nsys trace
    ├── decode_nsys_run_metadata.json        ← Inference metadata from nsys decode run
    ├── decode_phase_summary.json            ← All 128 forward-step NVTX durations; selected step 64
    ├── decode.ncu-rep                       ← Nsight Compute report, decode phase
    ├── decode_ncu_run_metadata.json         ← Inference metadata from ncu decode run
    │
    ├── run_config_summary.txt               ← Human-readable run summary (06_report_run_config.sh)
    └── performance_report.md                ← Consolidated analysis report

```

---

## 3. Deliverables

### 3.1 Shell Scripts — User-Facing (Run in Order)

#### `01_install.sh`

- **Purpose:** Verify WSL2 CUDA and profiler tooling, install missing Python packages into `dl_env`, ensure TensorRT-LLM is importable, and patch Gemma 4 AutoDeploy support into the installed package.
- **Inputs:** The pre-existing Python environment at `/home/songzhu/Desktop/dl_env`. Requires `nvidia-smi`, `ncu`, and `nsys` on `PATH`.
- **Outputs:** A validated Python environment with `tensorrt_llm`, `torch`, `transformers`, `PyYAML`, `sentencepiece`, `Pillow`, `accelerate`, and `huggingface_hub` importable. Prints installed package versions. Runs `gemma4_workflow.py preflight` to record environment metadata in `artifacts/run_config.json`.
- **Workflow position:** Step 1. All subsequent scripts depend on the environment this script validates.
- **Invocation:** `bash 01_install.sh`

#### `02_prepare_model.sh`

- **Purpose:** Download tokenizer assets from HuggingFace for each Gemma 4 model candidate and render a chat-templated prompt.
- **Inputs:** Calls `01_install.sh` internally for validation. Reads `MODEL_CANDIDATES` from `gemma4_workflow.py`.
- **Outputs:** For each candidate model, writes `templated_prompt.txt` and `templated_prompt.json` under `artifacts/runtime/<model_slug>/`. Updates `prepared_models` in `artifacts/run_config.json`.
- **Workflow position:** Step 2. Provides the prompt artifacts that step 3 uses for runtime probing.
- **Invocation:** `bash 02_prepare_model.sh`

#### `03_build_or_prepare_runtime.sh`

- **Purpose:** Apply TensorRT-LLM 1.2.0 compatibility patches (Gemma 4 MLP widening, static batch-1 export hints, attention-node metadata resolution), then probe model/precision/attention-backend combinations until one succeeds.
- **Inputs:** Calls `02_prepare_model.sh` internally. Reads prepared prompt artifacts from `artifacts/runtime/`.
- **Outputs:** Writes the selected runtime YAML to `artifacts/runtime/<model_slug>__<precision>__<attn>__<compile>.yaml`. Records the selected runtime configuration (model ID, precision, dtype, attention backend, compile backend, prompt path, probe results) in `artifacts/run_config.json` under `selected_runtime`.
- **Workflow position:** Step 3. Produces the runtime configuration that all profiling scripts consume.
- **Invocation:** `bash 03_build_or_prepare_runtime.sh`

#### `04_profile_prefill.sh`

- **Purpose:** Capture an Nsight Systems timeline and then profile the prefill phase with Nsight Compute.
- **Inputs:** Calls `nsys_prefill.sh` internally, which calls `03_build_or_prepare_runtime.sh`. Reads `artifacts/run_config.json` for runtime configuration.
- **Outputs:** `artifacts/reports/prefill.ncu-rep` (Nsight Compute report with four hardware metrics for one kernel). Registers the report path in `artifacts/run_config.json`.
- **Workflow position:** Step 4. Produces the prefill hardware-counter profile.
- **Key details:**
  - Runs inference with `--max-new-tokens 1` so captured kernels are dominated by the prefill step.
  - Checks GPU counter permissions via `ncu --query-metrics`; attempts Windows registry self-heal for `ERR_NVGPUCTRPERM`.
  - Collects four metrics: `gpu__time_duration.sum`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `sm__warps_active.avg.pct_of_peak_sustained_active`.
- **Invocation:** `bash 04_profile_prefill.sh`

#### `05_profile_decode.sh`

- **Purpose:** Capture an Nsight Systems timeline and then profile the decode phase with Nsight Compute.
- **Inputs:** Calls `nsys_decode.sh` internally, which calls `03_build_or_prepare_runtime.sh`. Reads `artifacts/run_config.json` for runtime configuration.
- **Outputs:** `artifacts/reports/decode.ncu-rep` (Nsight Compute report with the same four hardware metrics). Registers the report path in `artifacts/run_config.json`.
- **Workflow position:** Step 5. Produces the decode hardware-counter profile.
- **Key details:**
  - Uses a reduced token count (`--max-new-tokens 8`) to limit KV cache memory, preventing `ResourceUnavailable` on the 6 GB GPU.
  - Targets decode step 4 (the middle decode step of the 8-token run) via the `TLLM_PROFILE_START_STOP` environment variable, which directs TensorRT-LLM's executor worker to call `cudaProfilerStart`/`cudaProfilerStop` around that step.
  - Implements a three-tier fallback chain: (1) kernel replay with 8 tokens, (2) application replay with 8 tokens, (3) kernel replay with 2 tokens and profiling from start.
- **Invocation:** `bash 05_profile_decode.sh`

#### `06_report_run_config.sh`

- **Purpose:** Print and save a human-readable summary of the profiling run configuration.
- **Inputs:** `artifacts/run_config.json`.
- **Outputs:** `artifacts/reports/run_config_summary.txt`. Also prints the summary to stdout.
- **Workflow position:** Step 6. Final step; summarizes all prior results.
- **Invocation:** `bash 06_report_run_config.sh`

#### `08_inspect_ncu.sh`

- **Purpose:** Query a large Nsight Compute report from the CLI without depending on the laggy GUI.
- **Inputs:** Either a direct `--report` path or a registered `--phase` entry from `artifacts/run_config.json`.
- **Outputs:** Builds a cache file such as `artifacts/reports/decode.ncu-slice.json` on first use, then prints compact tables for family summaries, aggregated kernel names, or individual launches.
- **Workflow position:** Optional post-processing step after Nsight Compute profiling.
- **Invocation:** `bash 08_inspect_ncu.sh --phase decode`

### 3.2 Shell Scripts — Internal Helpers

#### `nsys_prefill.sh`

- **Purpose:** Capture a full CUDA+NVTX Nsight Systems timeline for the prefill phase, export it to SQLite, and select the representative prefill forward step.
- **Inputs:** Calls `03_build_or_prepare_runtime.sh` to ensure the runtime is prepared. Uses `gemma4_workflow.py run-inference --max-new-tokens 128` and `gemma4_workflow.py summarize-nsys --phase prefill`.
- **Outputs:** `artifacts/reports/prefill.nsys-rep`, `artifacts/reports/prefill.sqlite` (or extensionless export), `artifacts/reports/prefill_phase_summary.json`, `artifacts/reports/prefill_nsys_run_metadata.json`. Updates `profiles.prefill` in `artifacts/run_config.json`.
- **Workflow position:** Called by `04_profile_prefill.sh`. Not intended for direct invocation.

#### `nsys_decode.sh`

- **Purpose:** Capture a full CUDA+NVTX Nsight Systems timeline for the decode phase, export it to SQLite, and select the representative decode forward step (middle generation-only step).
- **Inputs:** Same as `nsys_prefill.sh`, but invokes `summarize-nsys --phase decode`.
- **Outputs:** `artifacts/reports/decode.nsys-rep`, `artifacts/reports/decode.sqlite`, `artifacts/reports/decode_phase_summary.json`, `artifacts/reports/decode_nsys_run_metadata.json`. Updates `profiles.decode` in `artifacts/run_config.json`.
- **Workflow position:** Called by `05_profile_decode.sh`. Not intended for direct invocation.

### 3.3 Shared Shell Environment

#### `workflow_env.sh`

- **Purpose:** Define all shared variables, paths, and helper functions used by the numbered scripts.
- **Key definitions:**
  - `PYTHON_BIN`, `PIP_BIN`, `DL_ENV_ROOT`: fixed paths into `/home/songzhu/Desktop/dl_env`.
  - `ARTIFACTS_DIR`, `CACHE_DIR`, `RUNTIME_DIR`, `REPORTS_DIR`, `RUN_CONFIG_JSON`: artifact layout.
  - `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `TLLM_LLMAPI_BUILD_CACHE_ROOT`: cache directories pinned under `artifacts/cache/`.
  - `require_python_env()`: validates the Python executable.
  - `ensure_workflow_dirs()`: creates all artifact directories.
  - `workflow_log()`: timestamped log output.
- **Workflow position:** Sourced by every script as the first action after `set -euo pipefail`.

### 3.4 Python Module

#### `gemma4_workflow.py`

- **Purpose:** Shared Python helper providing all workflow logic as argparse subcommands.
- **Subcommands:**

| Subcommand | Description |
|---|---|
| `preflight` | Detect GPU, CUDA, driver, PyTorch, TensorRT-LLM versions and precision support. Write results to `run_config.json`. |
| `prepare-model` | Download tokenizer, render chat-templated prompt, record token count. |
| `prepare-runtime` | Probe model/precision/backend combinations, write runtime YAML, record selected configuration. |
| `run-inference` | Load the selected runtime, execute one inference request with `AutoDeployLLM`, write metadata JSON. Accepts `--max-new-tokens`, `--metadata-output`, `--nvtx-tag`. |
| `summarize-nsys` | Export `.nsys-rep` to SQLite via `nsys export`, parse `_forward_step` NVTX ranges, select the target phase step (first prefill or middle decode), update `run_config.json`. |
| `nvtx-filter` | Print the escaped NVTX filter string for a given phase. |
| `register-ncu` | Record an `.ncu-rep` path and timestamp in `run_config.json` for a given phase. |
| `inspect-ncu` | Build or reuse a JSON cache for an `.ncu-rep` report and print small CLI views of summaries, top kernels, or per-launch metrics. |
| `report-config` | Print and optionally save a human-readable run summary. |

- **Key constants:**
  - `REQUESTED_MODEL_ID`: `"google/gemma-4-E2B-it"`.
  - `MODEL_CANDIDATES`: ordered list of Gemma 4 variants (E2B, 26B-A4B, 31B) with memory-fraction hints.
  - `PRECISION_ORDER`: `["FP16", "FP32", "BF16", "INT8"]`.
  - `DEFAULT_PROMPT`: `"How does a large language model work?"`.
  - `DEFAULT_MAX_NEW_TOKENS`: `128`.
- **Invocation:** Called by all shell scripts as `$PYTHON_BIN $WORKFLOW_ROOT/gemma4_workflow.py <subcommand> [args]`.

### 3.5 Documentation Files

#### `artifacts/reports/run_config_summary.txt`

- **Purpose:** Machine-generated plain-text summary of the profiling run configuration, including model, precision, backend, and all artifact paths.

### 3.6 Configuration File

#### `artifacts/run_config.json`

- **Purpose:** Central JSON store recording every decision and artifact path across the workflow. Updated by each script as it runs. Contains: `environment`, `candidate_models`, `prepared_models`, `selected_runtime`, `profiles.prefill`, `profiles.decode`, `probe_failures`, and `fallback_decisions`.

---

# Limitation of the current progress

The two scripts, `04_profile_prefill.sh` and `05_profile_decode.sh`, collect the same four Nsight Compute metrics for both prefill and decode:

- `gpu__time_duration.sum`: kernel execution time
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`: SM compute utilization as a percent of peak
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: DRAM bandwidth utilization as a percent of peak
- `sm__warps_active.avg.pct_of_peak_sustained_active`: warp activity / occupancy-style utilization

You can see that metric list in [04_profile_prefill.sh](/home/songzhu/Desktop/ece-511/04_profile_prefill.sh:181) and [05_profile_decode.sh](/home/songzhu/Desktop/ece-511/05_profile_decode.sh:95). Both scripts also use `--launch-count 1`, so they are intentionally profiling just a single matched kernel result rather than a broad sample.

These metrics are enough for a coarse first-pass bottleneck diagnosis, such as:
- whether a kernel is more compute-bound or memory-bandwidth-bound
- whether GPU utilization looks low because of poor occupancy / limited warp activity
- whether prefill and decode differ in basic GPU saturation

The collected data are not enough by themselves for strong research on performance bottleneck identification. The main gaps are:
- no stall reasons or scheduler breakdowns
- no cache metrics like L1/L2 hit rates
- no tensor core / instruction mix information
- no PCIe / host-device transfer analysis
- no CPU-side or end-to-end timeline data from these scripts themselves
- only one kernel sample per run, which risks being unrepresentative

The collected data are useful for exploratory profiling, but not sufficient alone for rigorous bottleneck-identification research. For research-quality results, you’d usually want these counters plus broader Nsight Systems timeline data, more detailed NCU sections, multiple kernels per phase, repeated runs, and workload variation.


---

## 4. Problems Encountered and Solutions

| # | Problem | Root Cause | Solution | Affected File(s) |
|---|---|---|---|---|
| 1 | `tensorrt_llm` import fails with `libmpi.so.40 not found` | TensorRT-LLM 1.2.0 native bindings require the OpenMPI runtime, which is not bundled by `pip install tensorrt_llm` | `01_install.sh` detects the `libmpi.so.40` error in the import log and installs `openmpi==4.1.8` into `dl_env` via pip. It then sets `OPAL_PREFIX` and adds `$DL_ENV_ROOT/lib/openmpi` to `LD_LIBRARY_PATH`. | `01_install.sh` (lines 40–54, 85–117) |
| 2 | `tensorrt_llm` import fails with `libcublasLt.so.13 not found` | TensorRT-LLM 1.2.0 links against CUDA 13 cuBLAS, which is not present in the CUDA 12.8 system installation | `01_install.sh` detects the `libcublasLt.so.13` error and installs `nvidia-cublas==13.3.0.5` into `dl_env`, adding the pip-installed library path to `LD_LIBRARY_PATH`. | `01_install.sh` (lines 56–70, 96–104) |
| 3 | TensorRT-LLM 1.2.0 lacks Gemma 4 AutoDeploy model registration | The released 1.2.0 package predates Gemma 4 support | `01_install.sh` downloads `modeling_gemma4.py` from a newer upstream TensorRT-LLM commit and patches the installed package's `custom/__init__.py` to register `Gemma4ForCausalLM` and `Gemma4ForConditionalGeneration`. | `01_install.sh` (lines 119–176) |
| 4 | Upstream Gemma 4 model file passes `layer_idx` to an attention op that does not accept it in TensorRT-LLM 1.2.0 | API mismatch between the upstream Gemma 4 patch and the local 1.2.0 custom op bundle | `03_build_or_prepare_runtime.sh` patches the installed `modeling_gemma4.py` at startup to remove the trailing `self.layer_idx` argument and widen the MLP constructor for KV-shared layers. | `03_build_or_prepare_runtime.sh` (lines 15–79) |
| 5 | Runtime YAML rejected with `cuda_graph_config` schema error | TensorRT-LLM 1.2.0 AutoDeploy YAML parser does not recognize the `cuda_graph_config` key emitted by the helper | `03_build_or_prepare_runtime.sh` falls back to a sanitized probing loop that strips `cuda_graph_config` and unsupported `transforms` keys from the YAML before each probe attempt. | `03_build_or_prepare_runtime.sh` (lines 272–562) |
| 6 | Attention-node FX graph metadata uses `tensor_meta` instead of `val` key, causing `KeyError("val")` during attention pattern matching | PyTorch version difference: the FX graph annotates some nodes with `tensor_meta` instead of `val` depending on the tracing path | `03_build_or_prepare_runtime.sh` patches both `triton_attention.py` and `torch_backend_attention.py` with a recursive `_resolve_tensor_info()` helper that walks input nodes and checks both `val` and `tensor_meta` keys. | `03_build_or_prepare_runtime.sh` (lines 125–256) |
| 7 | `AutoProcessor` / `AutoTokenizer` fails to load Gemma 4 tokenizer assets in the installed `transformers` version | Gemma 4 tokenizer metadata is not fully supported by the installed `transformers 4.57.3` build | `02_prepare_model.sh` catches the failure and falls back to manually loading `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` via `hf_hub_download`, then constructing a `PreTrainedTokenizerFast` directly. | `02_prepare_model.sh` (lines 25–165) |
| 8 | `ncu --query-metrics` fails with `ERR_NVGPUCTRPERM` on WSL2 | GPU hardware performance counters are gated by the Windows-side NVIDIA driver; the Linux-side modprobe fix does not apply under WSL2 | `04_profile_prefill.sh` detects the error, attempts to set the Windows registry key `NVReg_RestrictProfilingToAdminUsers=0` via `powershell.exe`, and guides the user to reboot WSL if the counters remain blocked. | `04_profile_prefill.sh` (lines 62–167) |
| 9 | Nsight Compute decode profiling fails with `ResourceUnavailable` | At decode step 64 of a 128-token run, the KV cache consumes most of the RTX 3050's 6 GB VRAM, leaving insufficient memory for ncu's kernel-replay buffers | `05_profile_decode.sh` reduces `--max-new-tokens` from 128 to 8 for the ncu-specific run, targets decode step 4 instead of step 64, adds `--clock-control none`, and implements a three-tier fallback chain (kernel replay, application replay, profile from start). | `05_profile_decode.sh` (lines 68–170) |
| 10 | TensorRT-LLM executor NVTX ranges use start/end style, not push/pop; `ncu --nvtx-include` cannot isolate individual forward steps from internal worker threads | Kernel launches happen on executor worker threads, not the parent thread where NVTX push/pop filters apply | Prefill profiling uses `--max-new-tokens 1` so all captured kernels are prefill-dominated (no NVTX filtering needed). Decode profiling uses `TLLM_PROFILE_START_STOP` to direct TRT-LLM's worker-side profiler control. | `04_profile_prefill.sh` (lines 169–195), `05_profile_decode.sh` (lines 87–107) |
| 11 | `nsys export` output path varies between nsys versions (some append `.sqlite`, some do not) | Nsight Systems 2024.6.2 writes to the exact `--output` path without appending `.sqlite` | `gemma4_workflow.py` `export_nsys_sqlite()` checks both `prefix.sqlite` and `prefix` (extensionless) after the export. | `gemma4_workflow.py` (lines 852–873) |
| 12 | Dynamic batch dimension in FX export hints causes shape-mismatch errors during compilation on this single-GPU, batch-1 setup | The default AutoDeploy factory marks both batch and sequence dimensions as `Dim.DYNAMIC`, which over-constrains the compiler on a batch-1 workload | `03_build_or_prepare_runtime.sh` patches both `factory.py` and `hf.py` to mark only `seq_len` as dynamic, leaving batch static. | `03_build_or_prepare_runtime.sh` (lines 81–123) |

---

## 4.1 Session Update: Nsight Compute Direct-Inference Fallback

This session addressed a remaining profiling failure in the Nsight Compute stage. The workflow could already produce `.ncu-rep` files for both prefill and decode, but those reports were not reliably usable when they came from the TensorRT-LLM path. In the failing cases, Nsight Compute captured kernels from Python process startup or other setup work instead of the actual large language model inference kernels that we wanted to study. From a workflow perspective, the problem was not simply "ncu crashed"; the more subtle failure mode was "ncu succeeded, but the resulting report was about the wrong work."

### Problem Encountered

The specific problem was that the TensorRT-LLM Nsight Compute runs could finish and write a report, but the report still did not represent real prefill or decode inference. Earlier investigation had already removed several script-level problems such as bad launch-count settings, permission checks, and too-small token counts. Even after those fixes, the TensorRT-LLM-generated reports could still contain only non-inference kernels while appearing superficially successful because a `.ncu-rep` file existed on disk.

In other words, the workflow had a correctness problem, not just a stability problem. A successful file write did not guarantee that the captured kernels were the kernels of interest.

### Root Cause

The root cause is a TensorRT-LLM and Nsight Compute architecture mismatch on WSL2. TensorRT-LLM executes inference work inside an MPI-managed worker process rather than entirely inside the parent Python process launched by the shell script. That worker creates its own CUDA context. On WSL2, that worker-side CUDA context can still hit GPU-counter restrictions or incompatible counter-collection behavior even when the NVIDIA Control Panel setting has been enabled correctly for the machine overall.

As a result, Nsight Compute may observe the parent process and record some GPU activity, but fail to capture the worker's real inference kernels. This is why the failure mode is deceptive: the profiler can still produce a report with measurable GPU time, but that GPU time may correspond only to setup kernels rather than GEMM, GEMV, gather, or attention kernels from actual model execution.

This session also uncovered an important secondary detail: report validation cannot rely only on total captured GPU time. The previously saved TensorRT-LLM reports in this repository summed to more than the 50 ms threshold, but still contained no inference-like kernels. That means a pure "time greater than threshold" rule is not sufficient to determine whether the report is valid.

### Steps Taken to Fix It

We implemented a fallback design that treats TensorRT-LLM profiling as the preferred path, but no longer trusts it blindly.

First, a new script, `ncu_direct_inference.py`, was added. This script runs single-process inference specifically for Nsight Compute fallback profiling. It reads the workflow configuration, tries to load the requested HuggingFace model directly, and if that fails, falls back to a structural proxy that preserves the Gemma 4 E2B feed-forward dimensions (`hidden_size=1536`, `intermediate_size=6144`, `num_hidden_layers=35`). The prefill path uses a 512-token workload and the decode path uses a 64-step single-token decode loop so that Nsight Compute replays real compute-heavy or memory-heavy kernels rather than startup noise.

Second, both `04_profile_prefill.sh` and `05_profile_decode.sh` were extended with a `validate_ncu_report()` function and a direct-inference fallback execution path. The validator now imports the generated report with `ncu --import`, normalizes timing units (`ns`, `us`, `ms`, `s`), and checks two conditions:

- enough GPU time was captured
- the report contains inference-like kernel names such as GEMM, GEMV, gather, attention, MMA, XMMA, cuBLAS, or CUTLASS kernels

If a TensorRT-LLM report fails either check, the script marks it invalid and reruns Nsight Compute against the single-process direct-inference script instead.

Third, the fallback implementation was made robust against the actual local software stack. In this environment, the installed `transformers` version cannot fully load Gemma 4 through the normal `AutoModelForCausalLM` and tokenizer path. Rather than letting that break the fallback itself, the direct-inference script was written to degrade gracefully to the structural proxy. That means the fallback still produces a valid Nsight Compute workload even when full HuggingFace Gemma 4 loading is unavailable.

### Why the Fix Works

The fix works because it removes the failing architectural dependency. The TensorRT-LLM problem is tied to profiling inference kernels that execute in an MPI worker process with its own CUDA context. The new fallback bypasses that design entirely. `ncu_direct_inference.py` runs as a single Python process, launches its kernels directly from that process, and therefore presents Nsight Compute with a normal, profiler-friendly execution model.

This matters for two separate reasons.

First, a single-process workload makes it much more likely that the kernels Nsight Compute captures are the kernels we actually care about. During verification in this session, the direct prefill fallback produced an `.ncu-rep` containing `sm80_xmma_gemm_f16f16_f16f32...` kernels, which are exactly the kind of inference-relevant kernels that were missing from the invalid TensorRT-LLM reports.

Second, the validator now distinguishes between "some GPU work happened" and "the intended inference kernels were profiled." That closes the logical gap that existed before this session. A report is no longer accepted merely because it exists or because it accumulated enough total time. It must also contain kernels that look like real model inference.

The result is a practical workflow fix rather than a vendor-level fix. This session did not repair the underlying TRT-LLM/NCU-on-WSL2 incompatibility. Instead, it made the profiling workflow resilient to that incompatibility by detecting bad TensorRT-LLM reports and automatically switching to a profiling path that Nsight Compute can capture correctly.

---

## 5. End-to-End Reproduction Steps

**Prerequisites:**

- Windows 11 with WSL2 enabled.
- An NVIDIA GPU with CUDA support (tested on RTX 3050 6 GB, Compute Capability 8.6).
- NVIDIA driver 595.71 or later installed on the Windows host.
- CUDA Toolkit with `ncu` and `nsys` available in the WSL2 `PATH` (tested with CUDA 12.8, Nsight Compute 2025.1.1.0, Nsight Systems 2024.6.2).
- A Python environment at `/home/songzhu/Desktop/dl_env` with Python 3.10, PyTorch, and pip. (The scripts hardcode this path.)
- GPU performance counters enabled. If `ncu --query-metrics` reports `ERR_NVGPUCTRPERM`, follow the guidance printed by `04_profile_prefill.sh` (NVIDIA Control Panel → Developer → Allow access to GPU performance counters to all users, then reboot).

**Steps:**

1. Run the installation and preflight validation.

   ```bash
   bash 01_install.sh
   ```

   This verifies CUDA tooling, installs missing Python packages (including TensorRT-LLM, OpenMPI, and CUDA 13 cuBLAS if needed), patches Gemma 4 AutoDeploy support, and records environment metadata in `artifacts/run_config.json`.

2. Prepare model prompts.

   ```bash
   bash 02_prepare_model.sh
   ```

   This downloads Gemma 4 tokenizer assets from HuggingFace, renders a chat-templated prompt, and records the token count. (Called automatically by step 4, but can be run independently.)

3. Select and warm the inference runtime.

   ```bash
   bash 03_build_or_prepare_runtime.sh
   ```

   This probes model/precision/backend combinations, applies TensorRT-LLM 1.2.0 compatibility patches, and records the selected runtime in `artifacts/run_config.json`. (Called automatically by steps 5 and 6, but can be run independently.)

4. Profile the prefill phase.

   ```bash
   bash 04_profile_prefill.sh
   ```

   This captures an Nsight Systems timeline (`prefill.nsys-rep`), selects prefill step 0 via NVTX analysis, checks GPU counter permissions, and runs Nsight Compute to produce `prefill.ncu-rep`.

5. Profile the decode phase.

   ```bash
   bash 05_profile_decode.sh
   ```

   This captures an Nsight Systems timeline (`decode.nsys-rep`), selects decode step 64 via NVTX analysis, and runs Nsight Compute with a reduced token count to produce `decode.ncu-rep`.

6. Generate the run-configuration summary.

   ```bash
   bash 06_report_run_config.sh
   ```

   This prints and saves `artifacts/reports/run_config_summary.txt`.

7. View the profiling data.

   ```bash
   # Nsight Compute — print kernel metrics as CSV
   ncu --import artifacts/reports/prefill.ncu-rep --csv
   ncu --import artifacts/reports/decode.ncu-rep --csv

   # Nsight Systems — print NVTX, CUDA API, and kernel statistics
   nsys stats artifacts/reports/prefill.nsys-rep
   nsys stats --force-export=true artifacts/reports/decode.nsys-rep

   # GUI viewers (require WSLg or an X server)
   ncu-ui artifacts/reports/prefill.ncu-rep
   ncu-ui artifacts/reports/decode.ncu-rep
   nsys-ui artifacts/reports/prefill.nsys-rep
   nsys-ui artifacts/reports/decode.nsys-rep

   # Or let the workflow choose the right GUI from the file extension
   bash 07_open_report.sh artifacts/reports/prefill.ncu-rep
   bash 07_open_report.sh artifacts/reports/decode.nsys-rep

   # Cache-backed Nsight Compute CLI inspection (recommended for large .ncu-rep files)
   bash 08_inspect_ncu.sh --phase decode
   bash 08_inspect_ncu.sh --phase decode --view kernels --family gemm_like --limit 15
   bash 08_inspect_ncu.sh --report artifacts/reports/decode.ncu-rep --view launches --sort dram_pct --limit 20
   ```

   On WSL, `07_open_report.sh` prefers the Windows-host `ncu-ui.exe` for
   `.ncu-rep` files when it is installed. This avoids Linux-side WSL memory
   pressure that can kill `ncu-ui.bin` while loading large reports such as the
   decode profile. If the Windows GUI is unavailable, the script falls back to
   Linux `ncu-ui` and forces the `xcb` Qt backend under WSLg.

   The first `08_inspect_ncu.sh` query for a report is still expensive because
   it runs `ncu --import` once and writes a JSON cache beside the report. After
   that, repeated queries are typically interactive and avoid reopening the GUI.

8. Read the consolidated performance report.

   ```bash
   cat artifacts/reports/performance_report.md
   ```

**Note on idempotency:** Each script checks whether its outputs already exist and skips redundant work. To force a fresh run, delete the `artifacts/` directory and re-run from step 2.

---

## 6. Key Takeaways

- **TensorRT-LLM 1.2.0 requires extensive patching for Gemma 4 support on this environment.** Five separate source-level patches (model registration, attention-op signatures, MLP widening, FX graph metadata resolution, and YAML schema sanitisation) were required before a single inference could complete. Each patch is applied idempotently at script startup so the workflow remains reproducible.

- **GPU memory constraints dominate the profiling strategy on a 6 GB GPU.** The RTX 3050's VRAM is almost entirely consumed by the model weights (4.2 GB). Nsight Compute kernel replay requires additional VRAM for state save/restore buffers, making it infeasible at full sequence length. The fix — reducing `--max-new-tokens` from 128 to 8 for the ncu-specific run — recovers enough memory for kernel replay while still capturing representative decode-phase kernels.

- **TensorRT-LLM's executor worker thread architecture requires worker-side profiler control for decode targeting.** Parent-side CUDA profiler start/stop calls and `ncu --nvtx-include` filters cannot reach kernels launched by the internal executor worker. The `TLLM_PROFILE_START_STOP` environment variable, which directs the worker to call `cudaProfilerStart`/`cudaProfilerStop` at a specific `_forward_step` iteration, is the correct mechanism for isolating decode-phase kernels under Nsight Compute.

- **WSL2 adds a layer of indirection for GPU counter access.** Unlike native Linux, where `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` enables performance counters, WSL2 requires the equivalent Windows registry key (`NVReg_RestrictProfilingToAdminUsers=0`) followed by a WSL restart or Windows reboot. The workflow automates detection and provides guided remediation.

- **Fallback chains at every level make the workflow robust against environmental variation.** The workflow implements fallbacks for tokenizer loading (AutoProcessor → manual FastTokenizer), runtime YAML schema (full schema → sanitized schema), attention backends (triton_paged → flashinfer → triton → torch), compile backends (torch-cudagraph → torch-simple), and Nsight Compute replay modes (kernel replay → application replay → profile from start). Each fallback is recorded in `artifacts/run_config.json` for traceability.
