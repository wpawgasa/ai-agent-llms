# TurboQuant Compatibility Patches

**Files:**
- `src/llm_workflow_agents/serving/launch_vllm_turboquant.py` — startup hooks
- `src/llm_workflow_agents/vllm_plugin.py` — vLLM general plugin (runs in every process)

These patches are installed whenever `--kv-cache-dtype` is a `turboquant*` variant. They address upstream blockers in vLLM v1 for Gemma-4 and Qwen3.5/3.6.

---

## Concepts: attention backend vs KV cache dtype

These are separate concerns in vLLM v1:

- **Attention backend** — the kernel that *computes* attention (e.g. `TurboQuantAttentionBackend`, `FlashAttentionBackend`, `TRITON_ATTN`)
- **KV cache dtype** — how KV tensors are *stored* (e.g. `turboquant_3bit_nc`, `fp8`, `auto`)

Upstream vLLM ships `vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend` — a full `AttentionBackend` subclass that integrates KV compression into its `forward()`. When `--kv-cache-dtype turboquant_*` is set, vLLM's per-layer router (`cuda.py:260`) selects `TurboQuantAttentionBackend` for non-skip layers and a standard attention backend (with `kv_cache_dtype=auto`, i.e. uncompressed BF16) for layers in `kv_cache_dtype_skip_layers`.

`TRITON_ATTN` is a *separate* standard attention backend (Triton-based, uncompressed). It has nothing to do with TurboQuant.

---

## Gemma-4 (26B-A4B, 31B)

Gemma-4 has two structural properties that require special handling:

1. **Heterogeneous head dimensions** — `full_attention` layers use `global_head_dim` (e.g. 512) while sliding-window layers use a smaller `head_dim` (e.g. 256). This creates mixed KV cache groups with incompatible page sizes.
2. **`verify_and_update_config` forces TRITON_ATTN globally** — `Gemma4Config.verify_and_update_config` (`vllm/model_executor/models/config.py:100`) sets the attention backend to `TRITON_ATTN` for every layer, overriding the per-layer `TurboQuantAttentionBackend` routing.

### Per-layer backend routing for Gemma-4

With patches applied, each layer's attention backend is determined as follows:

| Layer | Applicable head dim | In skip list? | KV cache dtype | Attention backend |
|---|---|---|---|---|
| Narrow (≤ 256), not boundary | ≤ 256 | No | `turboquant_*` | `TurboQuantAttentionBackend` |
| Wide (`head_dim > 256`), in skip list | > 256 | Yes (added by mixed-backend hook) | `auto` | `TRITON_ATTN` (FA2 rejected — see below) |
| Boundary layers (first/last 2) | any | Yes (standard TurboQuant skip) | `auto` | `FlashAttentionBackend` or `TRITON_ATTN` depending on head dim |

Wide skip-list layers cannot use `FlashAttentionBackend` even though they have `kv_cache_dtype=auto`: when `turboquant_*` is active, `arg_utils.py:1982` forces FlashAttention down to v2, and FA2 does not support `head_size > 256`. The `_patch_flash_attn_head_size_for_turboquant` plugin patch makes `FlashAttentionBackend.supports_head_size` return `False` for `head_size > 256` when TurboQuant is active, which pushes those layers to `TRITON_ATTN` instead.

### Hook 1 — `_install_gemma4_mixed_backend_hook()` (launcher, line 57)

**Problem:** `Gemma4Config.verify_and_update_config` forces `TRITON_ATTN` on every layer, preventing `TurboQuantAttentionBackend` from being selected for any layer.

**Fix:** Replaces `Gemma4Config.verify_and_update_config` with a patched version that:
- Reads `layer_types` from `hf_text_config` to find layers where the applicable head dim exceeds 256
- Adds those layer indices to `cache_config.kv_cache_dtype_skip_layers` so they fall back to `kv_cache_dtype=auto` (uncompressed)
- Does **not** call the original — intentionally avoids the global `TRITON_ATTN` force

Narrow layers (head dim ≤ 256) are not added to the skip list and receive `TurboQuantAttentionBackend` via the normal per-layer router.

### Hook 2 — `_install_turboquant_engine_config_hook()` (launcher, line 133), Gemma-4 path

**Problem:** Mixed KV cache groups (heterogeneous head dims → different page sizes) trigger two failure modes:
- `kv_cache_utils.py:952,958` — `unify_kv_cache_spec_page_size` requires pairwise divisibility across all spec page sizes (e.g. 198, 1024, 2048 bytes), which Gemma-4 + TurboQuant violates.
- `kv_cache_coordinator.py:402` — `HybridKVCacheCoordinator` asserts each group's `block_size` divides `hash_block_size`, which fails after LCM-based block-size unification.

**Fix:** When `_model_is_gemma4()` is true:
- **Does not force `enforce_eager`** — the page-size crash is handled by `_patch_unify_kv_cache_spec_page_size` in `vllm_plugin.py`, and dev vLLM already iterates per-group in `_reshape_kv_cache_tensors`. CUDA graphs are allowed; validate with a live H100 run.
- Pins `enable_prefix_caching=False` — routes to `KVCacheCoordinatorNoPrefixCache`, bypassing the block-size divisibility assertion. Re-enabling prefix caching would require patching `HybridKVCacheCoordinator` separately — out of scope.

**Previous state:** Earlier versions auto-injected `enforce_eager=True` for Gemma-4, costing ~15–25% decode throughput (CUDA graphs off). This was lifted once the LCM unify patch and dev vLLM's per-group `_reshape_kv_cache_tensors` made it unnecessary.

### Plugin patch — `_patch_flash_attn_head_size_for_turboquant()` (vllm_plugin.py, line 43)

**Problem:** When TurboQuant is active, `arg_utils.py:1982` forces FA down to v2. FA2 rejects `head_size > 256` at runtime (`FlashAttention forward only supports head dimension at most 256`). But `FlashAttentionBackend.supports_head_size` still claims support for up to 512 (if FA v4 is installed), so `get_attn_backend` incorrectly selects it for wide skip-list layers.

**Fix:** Patches `FlashAttentionBackend.supports_head_size` to return `False` for `head_size > 256` when TurboQuant is active. Wide layers are then routed to `TRITON_ATTN`, which correctly handles uncompressed BF16 at any head size.

### Plugin patch — `_patch_sliding_window_turboquant()` (vllm_plugin.py, line 124)

**Problem:** In `Attention.get_kv_cache_spec` (`attention.py:585–596`), the sliding-window check runs before the `turboquant_*` check. A Gemma-4 narrow sliding-window layer (not in the skip list) returns `SlidingWindowSpec`, whose `page_size_bytes` uses the uncompressed formula (`2 × block × num_kv × head_size × dtype_size`). But its attention backend is `TurboQuantAttentionBackend`, whose `get_kv_cache_shape` packs K+V into a single `slot_size_aligned` slot. The spec and backend disagree on page size, causing a `RuntimeError` at `gpu_model_runner.py:6598`.

**Fix:** Injects a `TQSlidingWindowSpec` subclass of `SlidingWindowSpec` that overrides `real_page_size_bytes` to use `block × num_kv × tq_slot_size` (matching `TurboQuantAttentionBackend`). Patches `Attention.get_kv_cache_spec` to return this spec whenever `sliding_window is not None` and `kv_cache_dtype.startswith("turboquant_")`.

### Plugin patch — `_patch_unify_kv_cache_spec_page_size()` (vllm_plugin.py, line 206)

**Problem:** Gemma-4 + TurboQuant produces attention specs with pairwise non-divisible page sizes (e.g. 198, 1024, 2048 bytes for a narrow TQ layer, a wide skip layer, and a boundary skip layer). Upstream `unify_kv_cache_spec_page_size` requires `max_page_size % other == 0` for every spec pair; this assertion fails.

**Fix:** Replaces the upstream function with an LCM-based unifier. For attention-only mixed specs, it sets `target = LCM(all attention page sizes)` — always a valid common multiple — and scales each spec's `block_size` by the integer ratio. For mixed attention+Mamba specs (Qwen3.5/3.6), it additionally covers the largest raw Mamba page size.

---

## Qwen3.5/3.6 (35B-A3B, 35B-A3B-FP8) — DeltaNet/GDN Hybrid

Qwen3.5 and Qwen3.6 interleave standard attention layers with DeltaNet (GDN) `linear_attention` layers. Attention layers receive `TurboQuantAttentionBackend` normally; GDN layers are not `Attention()` objects and are unaffected by KV cache dtype routing.

### Hook — `_install_turboquant_engine_config_hook()` (launcher, line 133), hybrid path

**Problem 1 — hybrid guard:** `EngineArgs.create_engine_config` (`arg_utils.py:1649`) raises `NotImplementedError` for any hybrid model + `turboquant_*` dtype, blocking startup.

**Problem 2 — MambaSpec block_size:** After bypassing the guard, `HybridAttentionMambaModelConfig.verify_and_update_config` is never called (dispatched only when `is_hybrid=True`). This leaves `MambaSpec.block_size=None`, which crashes `unify_kv_cache_spec_page_size` at `kv_cache_utils.py:958`.

**Problem 3 — FlashInfer JIT OOM:** FlashInfer JIT-compiles ~67 CUDA kernels serially for GDN layers during warm-up. On hosts with modest RAM, `nvcc` processes can be OOM-killed before the health-check timeout.

**Fixes:**

1. **Hybrid guard bypass:** Temporarily replaces `ModelConfig.is_hybrid` with a property returning `False` for the duration of `orig_create(...)`, then restores it in `finally`. Downstream hybrid routing (DeltaNet dispatch, Mamba state management) is unaffected after the call returns.

2. **MambaSpec repair:** After `orig_create` returns, if `vllm_config.model_config.is_hybrid` is `True` (restored), manually calls `HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)` to populate `mamba_block_size` and disable `calculate_kv_scales`.

3. **GDN prefill backend:** Forces `gdn_prefill_backend="triton"` when `_model_has_gdn_layers()` is true and no explicit backend is set.

**Why masking `is_hybrid` is safe:** DeltaNet/GDN layers never construct `Attention()` objects and are not reached by `kv_cache_dtype_skip_layers` logic. The mask only affects the config-creation guard; it does not change inference-time layer routing.

### Plugin patch — `_patch_unify_kv_cache_spec_page_size()` (vllm_plugin.py, line 206)

Also applies to Qwen3.5/3.6. When the spec dict mixes `MambaSpec` with TurboQuant attention specs, it computes `target = ceil_to_multiple(max(attn_LCM, max_raw_mamba_page), attn_LCM)`, sets `MambaSpec.page_size_padded = target`, and scales attention `block_size` by the integer ratio.

**Note — Nemotron-3-Nano 30B excluded:** Nemotron uses true Mamba SSM layers (not DeltaNet). It triggers separate vLLM compatibility issues (R6) beyond the hybrid guard. TurboQuant is not viable; it falls back to HF `generate()`.

---

## Mixed KV groups: shared behavior (Gemma-4 and Qwen3.5/3.6)

When either model is detected with a `turboquant*` dtype, the engine-config hook always pins:

- `enable_prefix_caching=False` — `HybridKVCacheCoordinator` block-size divisibility assertion applies to both

For GDN hybrids (Qwen3.5/3.6) only, the hook also forces:

- `enforce_eager=True` — Mamba kernel warm-up issues beyond the page-size fix

Gemma-4 no longer receives `enforce_eager=True` (CUDA graphs allowed). Both behaviors are logged at `WARNING` level with `reason` fields.

---

## Launch path

```
serving/launch_vllm.sh
  └── (kv_cache_dtype matches turboquant*)
        └── python -m llm_workflow_agents.serving.launch_vllm_turboquant
              ├── register_turboquant_backend()            # bare "turboquant" only
              ├── _install_gemma4_mixed_backend_hook()     # patch Gemma4Config
              └── _install_turboquant_engine_config_hook() # patch EngineArgs
                    ├── Gemma-4 path: disable prefix cache only (CUDA graphs allowed)
                    └── Qwen3.5/3.6 path: enforce_eager + disable prefix cache
                                        + mask is_hybrid + repair MambaSpec + triton GDN

vllm general plugin (every process, registered via entry point)
  ├── _patch_flash_attn_head_size_for_turboquant()  # cap FA head size at 256 when TQ active
  ├── _patch_sliding_window_turboquant()            # TQSlidingWindowSpec for narrow SW layers
  └── _patch_unify_kv_cache_spec_page_size()        # LCM-based page size unification
```

Upstream `turboquant_*` variants (`turboquant_3bit_nc`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_k8v4`) are first-class vLLM dtypes. They use upstream's `TurboQuantAttentionBackend` directly and bypass the project's Pydantic scaffolding patch, but still require all hooks above for Gemma-4 and Qwen3.5/3.6 compatibility.

---

## Affected vLLM source locations (v1)

| Location | Relevance |
|---|---|
| `vllm/model_executor/models/config.py:100` | `Gemma4Config.verify_and_update_config` forces `TRITON_ATTN` globally |
| `vllm/engine/arg_utils.py:1649` | Hybrid guard raises `NotImplementedError` on hybrid + `turboquant_*` |
| `vllm/engine/arg_utils.py:1982` | Forces FlashAttention down to v2 when `turboquant_*` active |
| `vllm/config/model.py:1563–1572` | `ModelConfig.is_hybrid` property |
| `vllm/v1/core/kv_cache_utils.py:952,958–960` | `unify_kv_cache_spec_page_size` divisibility and `MambaSpec` assertions (handled by plugin LCM patch) |
| `vllm/v1/worker/gpu_model_runner.py:3797` | `_reshape_kv_cache_tensors` — dev vLLM already iterates per-group; no patch needed |
| `vllm/kv_cache_coordinator.py:402` | `HybridKVCacheCoordinator` block-size divisibility assertion |
| `vllm/config/vllm.py:1721` | `try_verify_and_update_config` dispatches to `HybridAttentionMambaModelConfig` |
| `vllm/v1/attention/backends/flash_attn.py:174–181` | `FlashAttentionBackend.supports_head_size` claims FA v4 head sizes |
| `vllm/model_executor/layers/attention/attention.py:585–596` | Sliding-window check before `turboquant_*` check in `get_kv_cache_spec` |
