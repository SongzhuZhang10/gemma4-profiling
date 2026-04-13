# Gemma 4 WSL2 Profiling Workflow Using `/home/songzhu/Desktop/dl_env`

## Summary
- Build the deliverable around **TensorRT-LLM AutoDeploy on the PyTorch backend**, using the existing Python environment at `/home/songzhu/Desktop/dl_env` for **all Python installs and all Python command execution**.
- Do **not** create a new venv. Install Python packages into `/home/songzhu/Desktop/dl_env` with `/home/songzhu/Desktop/dl_env/bin/pip`.
- Avoid system-wide installation unless strictly required. The plan should assume:
  - Python-side packages go into `/home/songzhu/Desktop/dl_env`
  - existing system tools like `ncu`, `nsys`, `git`, CUDA user-space tools, and shell utilities are reused if already present
  - JSON/SQLite parsing should be done in Python to avoid requiring `jq` or `sqlite3`
- Current verified environment facts:
  - `/home/songzhu/Desktop/dl_env/bin/python` is **Python 3.10.12**
  - `/home/songzhu/Desktop/dl_env/bin/pip` is available
  - `torch 2.10.0+cu128` and `transformers` are already installed there
  - `torch.cuda.is_available()` is true on the **RTX 3050 6GB Laptop GPU**
  - `tensorrt_llm` is **not** installed there yet

## Implementation Changes
- Create the required scripts `01_install.sh` through `08_report_run_config.sh`, plus one shared Python helper module and one shared shell env file.
- Hard-code the Python entrypoints in every script to use:
  - `PYTHON_BIN=/home/songzhu/Desktop/dl_env/bin/python`
  - `PIP_BIN=/home/songzhu/Desktop/dl_env/bin/pip`
- Standardize artifacts under:
  - `artifacts/cache/`
  - `artifacts/runtime/`
  - `artifacts/reports/`
  - `artifacts/run_config.json`

- `01_install.sh`
  - Validate WSL2, GPU visibility, CUDA user-space visibility, and profiler availability.
  - Reuse `/home/songzhu/Desktop/dl_env`; do not recreate or modify interpreter selection.
  - Install Python-side dependencies into that environment with `"$PIP_BIN" install ...`, including `tensorrt_llm` and any missing helper packages required by the workflow.
  - Do not install `python3-pip`, `python3-venv`, `torch`, or `transformers` system-wide.
  - Only perform system-wide installation if a strictly required non-Python tool is missing and cannot reasonably be replaced by Python or existing user-space tools.
  - Prefer failing with a clear message over broad `apt` installation for convenience packages.

- Preflight and selection logic
  - Keep the previously agreed support policy:
    - exact registry hit for `google/gemma-4-E2B-it` is **not required**
    - architecture-level Gemma 4 support is acceptable
    - family fallback is allowed if needed
  - Candidate order:
    - `google/gemma-4-E2B-it`
    - `google/gemma-4-26B-A4B-it`
    - `google/gemma-4-31B-it`
  - Precision order:
    - `FP16`
    - `FP32`
    - `BF16`
    - `INT8`
  - Preflight must run from `/home/songzhu/Desktop/dl_env/bin/python` and verify:
    - installed TensorRT-LLM version
    - Gemma 4 AutoDeploy support in that installed release
    - HF architecture compatibility
    - GPU support and memory feasibility
    - chosen runtime/backend path

- Runtime preparation
  - Use AutoDeploy text-only execution for `Gemma4ForConditionalGeneration`.
  - For `google/gemma-4-E2B-it`, generate a local YAML tuned for batch size 1 and the exact token budget.
  - For sibling fallback models, start from NVIDIA’s registry YAML if available, then override batch/token limits to the fixed workload.
  - `03_build_or_prepare_runtime.sh` should prepare and warm the runtime using `/home/songzhu/Desktop/dl_env/bin/python`, and persist any compile/cache artifacts under `artifacts/cache/`.

- Profiling scripts
  - `04_profile_prefill.sh`, `05_profile_decode.sh`, `06_nsys_prefill.sh`, and `07_nsys_decode.sh` should launch the profiled application via `/home/songzhu/Desktop/dl_env/bin/python`.
  - Use TensorRT-LLM executor NVTX ranges to identify:
    - prefill: `_forward_step` with `ctx reqs > 0`
    - decode: `_forward_step` with `ctx reqs = 0` and `gen reqs > 0`
  - Parse `.nsys` exports in Python from the same venv instead of requiring system `sqlite3`.
  - Save the selected phase ordinals and report paths into `artifacts/run_config.json`.

- Reporting
  - `08_report_run_config.sh` prints and saves:
    - requested model id
    - actual profiled model id
    - TensorRT-LLM version from `/home/songzhu/Desktop/dl_env`
    - workflow/backend used
    - precision used
    - templated input token count
    - output token cap
    - fallback decisions
    - artifact locations

## Test Plan
- Existing-env path: `/home/songzhu/Desktop/dl_env` exists, `pip` works, `torch` sees the GPU, and `tensorrt_llm` is installed into that same environment successfully.
- No-new-venv path: scripts never create or switch to another Python environment.
- Minimal-privilege path: Python dependencies are installed only into `/home/songzhu/Desktop/dl_env`; no unnecessary system-wide package installation occurs.
- Unsupported-model path: preflight fails early if the installed TensorRT-LLM release in `/home/songzhu/Desktop/dl_env` cannot support Gemma 4 AutoDeploy.
- Family-fallback path: if E2B is not runnable, the workflow records the fallback to another Gemma 4 variant.
- Phase-validation path: Nsight Systems trace parsing finds both a prefill step and a decode-only step before Nsight Compute targeting proceeds.
- Deliverable path: separate `.ncu-rep` and `.nsys-rep` files are produced for prefill and decode, and the final config report records the exact environment and runtime details.

## Assumptions And Defaults
- `/home/songzhu/Desktop/dl_env` remains the authoritative Python environment for the whole workflow.
- Python-based replacements are preferred over extra system packages for parsing and reporting.
- Existing `ncu` and `nsys` binaries are reused; if they become unavailable, the scripts should report that explicitly before attempting privileged installation.
- Text-only inference on Gemma 4 is still the intended execution path for profiling.
