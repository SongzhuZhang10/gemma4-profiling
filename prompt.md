The background of this task is in `implementation-plan.md`.

# Current Progress

The `01_install.sh` and `02_prepare_model.sh` have been executed.

`03_build_or_prepare_runtime.sh` now completes successfully. I updated [03_build_or_prepare_runtime.sh](/home/songzhu/Desktop/ece-511/03_build_or_prepare_runtime.sh:1) so it can self-heal the broken TensorRT-LLM 1.2.0 Gemma 4 attention metadata path, reorder the sanitized fallback probes for this machine, and try a compile-backend fallback instead of insisting on `torch-cudagraph` after GPU-specific failures. I also patched the currently installed attention backends in `/home/songzhu/Desktop/dl_env` so the live environment matched the script fixes during reruns.

The successful runtime selection is recorded in [artifacts/run_config.json](/home/songzhu/Desktop/ece-511/artifacts/run_config.json:1): `google/gemma-4-E2B-it`, `FP16`, `torch` attention, `non_paged`, `torch-simple`. The final runtime YAML is [google__gemma-4-E2B-it__fp16__torch_non_paged__torch_simple.yaml](/home/songzhu/Desktop/ece-511/artifacts/runtime/google__gemma-4-E2B-it__fp16__torch_non_paged__torch_simple.yaml:1). I also cleaned up the stale temporary fallback scripts from earlier failed attempts.

One important note for the next steps: on this RTX 3050, the originally preferred paths are not actually feasible in the installed stack. `triton` attention fails on shared-memory limits, and `torch` attention fails under CUDA graph capture, so the workflow had to fall back to `torch-simple` to get a usable warmed runtime.

# Your Task

