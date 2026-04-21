"""vLLM general plugin for TurboQuant + hybrid-model KV-cache patches.

Registered under the ``vllm.general_plugins`` entry point so it loads in every
vLLM process (APIServer, engine-core, worker) — our launcher-only hooks only
apply in the APIServer process, but some patches must run in engine-core where
KV cache configuration actually happens.

Patches applied (all idempotent, all self-gated at runtime):

- ``unify_kv_cache_spec_page_size``: upstream scales ``block_size`` to bring
  a smaller-page-size spec up to ``max_page_size``. Two failure modes for
  TurboQuant + hybrid-Mamba models like Qwen3.5/3.6:
  1. ``MambaSpec.page_size_bytes`` is fixed by state shapes — scaling
     ``block_size`` doesn't change it, so upstream's post-scale assertion at
     ``kv_cache_utils.py:960`` trips.
  2. Upstream's ``update_block_size_for_backend`` pads Mamba using regular
     ``FullAttentionSpec`` math (e.g. ``2 * head_size * sizeof(uint8)`` for
     ``turboquant_3bit_nc``), which is not a multiple of TurboQuant's
     compressed ``tq_slot_size`` (102 vs 256 for head_dim=128), so the
     divisibility check at ``kv_cache_utils.py:952`` raises NotImplementedError.

  Our patched version: when the spec dict mixes ``MambaSpec`` with attention
  specs, compute a unified target = ``LCM(attention page sizes)`` rounded up
  to cover every Mamba layer's raw (unpadded) state size. Set
  ``MambaSpec.page_size_padded`` to that target; scale attention ``block_size``
  by the integer ratio. Falls through to the original upstream logic in the
  non-mixed case.
"""

from __future__ import annotations

from functools import reduce
from math import lcm, prod


def register() -> None:
    """Entry point called by ``vllm.plugins.load_general_plugins()``."""
    _patch_unify_kv_cache_spec_page_size()
    _patch_sliding_window_turboquant()
    _patch_flash_attn_head_size_for_turboquant()


def _patch_flash_attn_head_size_for_turboquant() -> None:
    """Cap ``FlashAttention.supports_head_size`` at 256 when TurboQuant is active.

    Upstream's ``FlashAttention.supports_head_size`` (at
    flash_attn.py:174-181) returns True for head_size up to 512 if FA v4 is
    available on this SM. But when ``--kv-cache-dtype=turboquant_*``,
    ``arg_utils.py:1982`` forces the flash_attn version down to 2 (TurboQuant
    incompatible with FA≥3). FA2 does NOT support head_size > 256; the
    runtime error is ``FlashAttention forward only supports head dimension
    at most 256``.

    On Gemma-4 with wide (head_dim=512) full-attention layers in the skip
    list, ``get_attn_backend`` with dtype=auto picks FlashAttention because
    it claims support for head=512. This patch forces those layers onto the
    next valid backend (TRITON_ATTN) by making FA's declared support
    match the forced-FA2 runtime reality.
    """
    try:
        import vllm.v1.attention.backends.flash_attn as fa_mod
    except ImportError:
        return

    raw = fa_mod.FlashAttentionBackend.__dict__.get("supports_head_size")
    if raw is not None and getattr(
        getattr(raw, "__func__", raw), "_tq_patched", False
    ):
        return

    orig = fa_mod.FlashAttentionBackend.supports_head_size

    def _is_turboquant_active() -> bool:
        try:
            from vllm.config.vllm import get_current_vllm_config

            cache_dtype = get_current_vllm_config().cache_config.cache_dtype
            return isinstance(cache_dtype, str) and cache_dtype.startswith(
                "turboquant"
            )
        except Exception:
            return False

    @classmethod
    def patched(cls, head_size: int) -> bool:
        # Gate: only cap at 256 when TurboQuant is active (FA forced to v2 by
        # arg_utils.py:1982). For any other config, defer to upstream.
        if head_size > 256 and _is_turboquant_active():
            return False
        return orig.__func__(cls, head_size)

    patched.__func__._tq_patched = True  # type: ignore[attr-defined]
    patched.__func__._orig = orig.__func__  # type: ignore[attr-defined]
    fa_mod.FlashAttentionBackend.supports_head_size = patched


def _attention_only_lcm_unify(kv_cache_spec, target):
    """Scale each attention spec's block_size so its page_size == target.

    Precondition: ``target`` is a multiple of every spec's ``page_size_bytes``
    (guaranteed by using LCM). Because attention page size is linear in
    ``block_size``, the ratio is integer and the new ``page_size_bytes``
    equals ``target``.
    """
    from dataclasses import replace

    new_spec = {}
    for name, layer in kv_cache_spec.items():
        layer_page = layer.page_size_bytes
        if layer_page == target:
            new_spec[name] = layer
            continue
        assert target % layer_page == 0, (
            f"target {target} not divisible by layer {name} "
            f"page_size {layer_page} — LCM math wrong?"
        )
        ratio = target // layer_page
        new_layer = replace(layer, block_size=layer.block_size * ratio)
        assert new_layer.page_size_bytes == target
        new_spec[name] = new_layer
    return new_spec


def _patch_sliding_window_turboquant() -> None:
    """Fix Gemma-4 sliding attention + TurboQuant.

    In ``vllm.model_executor.layers.attention.attention.Attention.get_kv_cache_spec``
    (at attention.py:585-596), the sliding-window check comes BEFORE the
    turboquant check, so a layer that is both sliding and turboquant_* returns
    ``SlidingWindowSpec`` — whose ``page_size_bytes`` uses the FullAttention
    math (``2 * block * num_kv * head_size * dtype_size``). But the selected
    attention backend is ``TurboQuantAttentionBackend``, whose
    ``get_kv_cache_shape`` packs K+V into a single ``slot_size_aligned`` slot.
    The spec and the backend disagree: raw tensor allocated at 512 bytes/slot
    (head=256, uint8), view shape wants 198 bytes/slot → RuntimeError at
    gpu_model_runner.py:6598.

    Fix: inject a ``TQSlidingWindowSpec`` subclass of ``SlidingWindowSpec``
    that overrides ``real_page_size_bytes`` the same way ``TQFullAttentionSpec``
    does (block * num_kv * tq_slot_size). Patch ``Attention.get_kv_cache_spec``
    so sliding + ``turboquant_*`` returns the new spec with the right slot size.
    """
    import vllm.model_executor.layers.attention.attention as attn_mod
    import vllm.v1.kv_cache_interface as kvc_mod

    if getattr(attn_mod.Attention.get_kv_cache_spec, "_tq_patched", False):
        return

    from dataclasses import dataclass

    @dataclass(frozen=True, kw_only=True)
    class TQSlidingWindowSpec(kvc_mod.SlidingWindowSpec):
        tq_slot_size: int = 0

        @property
        def real_page_size_bytes(self) -> int:
            if self.tq_slot_size > 0:
                return self.block_size * self.num_kv_heads * self.tq_slot_size
            return super().real_page_size_bytes  # type: ignore[misc]

    # Expose so downstream code can do isinstance / import.
    kvc_mod.TQSlidingWindowSpec = TQSlidingWindowSpec

    # Register TQSlidingWindowSpec in the spec→manager map so the scheduler's
    # KVCacheManager can look up a manager for this spec type (upstream already
    # does this for TQFullAttentionSpec at single_type_kv_cache_manager.py:1118).
    try:
        import vllm.v1.core.single_type_kv_cache_manager as mgr_mod

        mgr_mod.spec_manager_map[TQSlidingWindowSpec] = mgr_mod.SlidingWindowManager
    except (ImportError, AttributeError):
        pass  # Upstream module layout changed — fail open.

    orig_get_spec = attn_mod.Attention.get_kv_cache_spec

    def patched_get_kv_cache_spec(self, vllm_config):
        kv_cache_dtype = getattr(self, "kv_cache_dtype", "") or ""
        if (
            getattr(self, "sliding_window", None) is not None
            and kv_cache_dtype.startswith("turboquant_")
        ):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            tq_config = TurboQuantConfig.from_cache_dtype(
                kv_cache_dtype, self.head_size
            )
            block_size = vllm_config.cache_config.block_size
            return TQSlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=kvc_mod.get_kv_quant_mode(kv_cache_dtype),
                sliding_window=self.sliding_window,
                tq_slot_size=tq_config.slot_size_aligned,
            )
        return orig_get_spec(self, vllm_config)

    patched_get_kv_cache_spec._tq_patched = True  # type: ignore[attr-defined]
    patched_get_kv_cache_spec._orig = orig_get_spec  # type: ignore[attr-defined]
    attn_mod.Attention.get_kv_cache_spec = patched_get_kv_cache_spec


def _patch_unify_kv_cache_spec_page_size() -> None:
    import vllm.v1.core.kv_cache_utils as mod

    if getattr(mod.unify_kv_cache_spec_page_size, "_tq_patched", False):
        return

    orig = mod.unify_kv_cache_spec_page_size

    def patched(kv_cache_spec):
        from dataclasses import replace

        from vllm.utils.torch_utils import get_dtype_size
        from vllm.v1.kv_cache_interface import MambaSpec

        if not kv_cache_spec:
            return kv_cache_spec

        page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
        if len(page_sizes) <= 1:
            return kv_cache_spec

        mamba_specs = [
            s for s in kv_cache_spec.values() if isinstance(s, MambaSpec)
        ]
        attn_specs = [
            s for s in kv_cache_spec.values() if not isinstance(s, MambaSpec)
        ]

        if not attn_specs:
            # All-mamba case: we have no clear heuristic for unifying multiple
            # Mamba specs with different page sizes. Delegate to upstream.
            return orig(kv_cache_spec)

        attn_page_sizes = [s.page_size_bytes for s in attn_specs]
        attn_lcm = reduce(lcm, attn_page_sizes)

        # Attention-only case: upstream scales block_size but requires
        # max % other == 0 for every spec. Gemma-4 + TurboQuant violates that
        # (pages 198, 1024, 2048 — pairwise non-divisible). LCM is always a
        # valid target for attention specs because page_size is linear in
        # block_size. This may produce large block_sizes; scheduler handles it.
        if not mamba_specs:
            return _attention_only_lcm_unify(kv_cache_spec, attn_lcm)

        # Raw (unpadded) page size for each Mamba layer, computed from the
        # state shapes. We need this because `page_size_bytes` returns the
        # padded value when set, and upstream's padding is the exact value
        # we're about to replace.
        max_mamba_raw = 0
        for ms in mamba_specs:
            raw = sum(
                prod(shape) * get_dtype_size(dtype)
                for shape, dtype in zip(ms.shapes, ms.dtypes)
            )
            if raw > max_mamba_raw:
                max_mamba_raw = raw

        # Target must be: a multiple of every attention page size AND
        # >= the largest raw Mamba page.
        target = max(attn_lcm, max_mamba_raw)
        remainder = target % attn_lcm
        if remainder:
            target += attn_lcm - remainder

        new_spec: dict = {}
        for name, layer in kv_cache_spec.items():
            if isinstance(layer, MambaSpec):
                new_layer = replace(layer, page_size_padded=target)
                assert new_layer.page_size_bytes == target
                new_spec[name] = new_layer
                continue

            layer_page = layer.page_size_bytes
            if layer_page == target:
                new_spec[name] = layer
                continue

            assert target % layer_page == 0, (
                f"target {target} not divisible by attention layer "
                f"page_size {layer_page} (layer {name}) — LCM math wrong?"
            )
            ratio = target // layer_page
            new_layer = replace(layer, block_size=layer.block_size * ratio)
            assert new_layer.page_size_bytes == target
            new_spec[name] = new_layer

        return new_spec

    patched._tq_patched = True  # type: ignore[attr-defined]
    patched._orig = orig  # type: ignore[attr-defined]  # kept for debugging
    mod.unify_kv_cache_spec_page_size = patched
