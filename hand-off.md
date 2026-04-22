# Hand-off Report: Gemma4 AutoDeploy Runtime Probe Fix

## What the Problem Is

Running `03_build_or_prepare_runtime.sh` fails — every probe in the `prepare-runtime` step
silently catches exceptions and exhausts all model/precision candidates, ending with:

```
No supported Gemma 4 model/precision combination completed a successful probe.
```

The root cause is a cascade of incompatibilities between the upstream Gemma 4 AutoDeploy patch
(downloaded from TRT-LLM commit `26a28ea`) and the pinned environment (TRT-LLM 1.1.0,
PyTorch 2.9.0+cu130, Blackwell DGX).

---

## Fixes Applied (All Persisted)

### 1. `pydantic-settings 2.14.0` x TRT-LLM 1.1.0 — FIXED

`pydantic-settings 2.14.0` added a `deep_merge` kwarg to `_read_files()` that TRT-LLM 1.1.0's
`DynamicYamlWithDeepMergeSettingsSource` override didn't accept → `TypeError` on every probe.

**Fix:** Pin `pydantic-settings<2.14.0` in `01_install.sh` + defensive patch of the
`_read_files` signature in `.venv/.../tensorrt_llm/_torch/auto_deploy/utils/_config.py`.
Verified working (pydantic-settings downgraded to 2.13.1).

### 2. `LlmArgs` validation errors — FIXED in `gemma4_workflow.py`

Three separate validation errors fired before the model even loaded:

- `model_factory: 'Gemma4ForConditionalGeneration'` not in registry — added
  `import tensorrt_llm._torch.auto_deploy.models.custom` at the top of
  `attempt_runtime_probe` to trigger the `@ModelFactoryRegistry.register(...)` decorator.
- `attn_backend: 'triton_paged'` invalid — changed to `'triton'` (valid literal in 1.1.0).
- `cuda_graph_config` extra-forbidden — changed to top-level `cuda_graph_batch_sizes: [1]`.

### 3. MPI spawn hang (`PMPI_Comm_accept` segfault) — FIXED

pip-installed OpenMPI 4.1.8 on aarch64 crashes in `MPI_Comm_accept` when TRT-LLM tries to
spawn a worker process via `MPI_Comm_spawn`. The spawned worker calls `PMPI_Comm_accept` and
segfaults; the parent process hangs in a barrier forever (GPU at 0%, 2.5+ hours observed).

`executor/proxy.py` always creates `MpiPoolSession` regardless of world_size, so the normal
`AutoDeployLLM` path is unusable on this machine.

**Fix:** Switched `attempt_runtime_probe` from `AutoDeployLLM` to `DemoLLM(world_size=0)`.
`DemoLLM` with `world_size=0` runs the model in the main process via `DemoGenerationExecutor`
with no MPI spawning at all, while still applying full cuda-graph compilation.

### 4. `torch.ops.auto_deploy.torch_attention` missing — FIXED

The upstream patch calls `torch.ops.auto_deploy.torch_attention(q, k, v, ..., "bsnd", layer_idx)`
which doesn't exist in TRT-LLM 1.1.0 (introduced in a later version).
→ `AttributeError: '_OpNamespace' 'auto_deploy' object has no attribute 'torch_attention'`

**Fix:** Patched `modeling_gemma4.py` line 472 to use
`torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa(q, k, v, attn_mask=None,
dropout_p=0.0, is_causal=True, scale=1.0, sinks=None, sliding_window=self.sliding_window,
logit_cap=None)` — the TRT-LLM 1.1.0 equivalent that takes the same BSND layout and
sliding-window args, without the `"bsnd"` string sentinel and `layer_idx`.

Applied to both the installed `.venv` file and `01_install.sh` (so it survives a reinstall).

### 5. `use_double_wide_mlp` not honoured — FIXED

All Gemma4 dense variants set `use_double_wide_mlp: true` in their config. The last
`num_kv_shared_layers` layers (layers 15-34 for E2B with `num_hidden_layers=35`,
`num_kv_shared_layers=20`) have `2 x intermediate_size` in the checkpoint weights
(`[12288, 1536]`), but `Gemma4TextMLP` was always built with `config.intermediate_size=6144`
→ `RuntimeError: size mismatch` on 20 layers during weight loading.

**Fix:**
- Modified `Gemma4TextMLP.__init__` to accept an optional `intermediate_size` override.
- Modified `Gemma4TextDecoderLayer.__init__` to pass `2 x config.intermediate_size` for
  `layer_idx >= config.num_hidden_layers - config.num_kv_shared_layers` when
  `config.use_double_wide_mlp` is True.

Applied to both the installed `.venv` file and `01_install.sh`.

---

## Current Blocker

After all the above fixes, the probe reaches Triton kernel compilation and crashes:

```
triton.runtime.errors.PTXASError: PTXAS error: Internal Triton PTX codegen error
ptxas fatal   : Value 'sm_110a' is not defined for option 'gpu-name'

Repro command: ptxas --gpu-name=sm_110a ...
```

**Root cause:** The DGX has Blackwell GPUs (`sm_110a`). The Triton version bundled with
PyTorch 2.9.0+cu130 / TRT-LLM 1.1.0 does not support `sm_110a`. With
`attn_backend: "triton"`, TRT-LLM JIT-compiles a Triton CUDA attention kernel that ptxas
cannot assemble for this architecture.

---

## What to Do Next

### Use Llama 3.1 8B Instruct instead of Gemma LLM


---

