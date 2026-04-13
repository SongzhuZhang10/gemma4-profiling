The background of this project is in `implementation-plan.md`.

# Current Progress

The `01_install.sh`, `02_prepare_model.sh`, and `03_build_or_prepare_runtime.sh` have been executed successfully.

The following content is the progress report on running `04_profile_prefill.sh`:

## Changes Made

1. **`gemma4_workflow.py`** — two fixes:
    * **`export_nsys_sqlite()` (line 843):** Fixed SQLite output path detection. Nsys 2024.6.2 writes the exported SQLite file without appending `.sqlite` to the `--output` path. The function now checks both `prefix.sqlite` and `prefix` (extensionless) to accommodate both old and new nsys versions.
    * **`run_inference_command()` (line 807):** Added `--nvtx-tag` argument support. When provided, wraps `llm.generate()` in an NVTX push/pop range via `torch.cuda.nvtx.range_push()` / `range_pop()`. (This was an intermediate fix attempt; the final script doesn't use it because of the executor thread issue described below.)

2. **`04_profile_prefill.sh`** — rewritten to handle three issues:
    * **NVTX start/end vs push/pop:** TRT-LLM's executor emits NVTX start/end ranges and launches kernels from internal threads. `ncu --nvtx-include` only matches push/pop ranges on the same thread. The script now profiles without NVTX filtering and uses `--max-new-tokens 1` so the captured kernels are dominated by the prefill step.
    * **`--set basic` instead of `--set full`:** Reduces metrics from ~5900 to ~190, keeping memory pressure manageable on the 6GB GPU.
    * **`ERR_NVGPUCTRPERM` handling:** The script detects the WSL2 GPU counter permission issue, attempts to self-heal by setting the Windows `NVReg_RestrictProfilingToAdminUsers=0` registry key via an elevated PowerShell process, and gracefully skips `ncu` profiling if counters remain blocked.

---

## Artifacts Produced

| Artifact | Status |
| :--- | :--- |
| `artifacts/reports/prefill.nsys-rep` (17.6 MB) | Created, valid nsys trace |
| `artifacts/reports/prefill` (99 MB) | SQLite export of nsys trace |
| `artifacts/reports/prefill_phase_summary.json` | Created, prefill step identified (step 0, 1 ctx req, 2.77s duration) |
| `artifacts/reports/prefill.ncu-rep` | **NOT created** — blocked by `ERR_NVGPUCTRPERM` |

---

## Remaining Issue: `ERR_NVGPUCTRPERM`

`ncu` cannot access GPU hardware performance counters on this WSL2 system. Even `sudo` doesn't help because the counters are gated by the Windows-side NVIDIA driver, not Linux-side permissions.

The fix has been applied but requires a Windows reboot to take effect, which has been done already. The registry key `HKLM\SOFTWARE\NVIDIA Corporation\Global\NVTweak\NVReg_RestrictProfilingToAdminUsers = 0` has been set (verified). To activate it:

1. Close this WSL session
2. Run `wsl --shutdown` from a Windows command prompt
3. Relaunch WSL
4. Re-run `bash 04_profile_prefill.sh`

Alternatively, a full Windows reboot will also work. After the restart which has been done already, the script's self-check should detect that counters are now accessible and proceed to generate the `prefill.ncu-rep` file.

# Your Task

Proceed to run `04_profile_prefill.sh`. If you encounter any issues during that process, fix them by modifying `04_profile_prefill.sh` so that the issues will not happen again in the future.