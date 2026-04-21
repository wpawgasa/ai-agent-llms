"""Tests for the vLLM general plugin that patches
``vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size``.

The plugin handles two failure modes upstream exhibits on TurboQuant + hybrid
(Mamba/GDN) models:

1. ``MambaSpec`` page size is fixed by state shapes; scaling ``block_size``
   doesn't grow it, so upstream's post-scale assertion at
   ``kv_cache_utils.py:960`` trips.
2. Upstream pads Mamba using ``FullAttentionSpec`` math that is not a
   multiple of ``TurboQuantConfig.slot_size_aligned`` (256 vs 102 for
   ``turboquant_3bit_nc`` head_dim=128), so the divisibility check at
   ``kv_cache_utils.py:952`` raises NotImplementedError.

Patched behavior: for mixed Mamba + attention specs, pick target =
LCM(attention page sizes) rounded up to cover every Mamba layer's raw
state size. Pad Mamba; scale attention by the integer ratio.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, replace

import pytest


def _install_stub_kv_cache_modules(monkeypatch):
    def _original_unify(kv_cache_spec):
        # Sentinel value — tests assert this is NOT reached for the fixed paths.
        return {"_orig_called": True}

    core_utils_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
    core_utils_mod.unify_kv_cache_spec_page_size = _original_unify

    @dataclass(frozen=True)
    class StubMambaSpec:
        name: str
        shapes: tuple
        dtypes: tuple
        page_size_padded: int | None = None

        @property
        def page_size_bytes(self) -> int:
            raw = sum(
                _prod(shape) * _dtype_size(dtype)
                for shape, dtype in zip(self.shapes, self.dtypes)
            )
            return self.page_size_padded if self.page_size_padded else raw

    @dataclass(frozen=True)
    class StubFullAttentionSpec:
        name: str
        block_size: int
        per_token_page: int = 10

        @property
        def page_size_bytes(self) -> int:
            return self.block_size * self.per_token_page

    interface_mod = types.ModuleType("vllm.v1.kv_cache_interface")
    interface_mod.MambaSpec = StubMambaSpec
    interface_mod.AttentionSpec = StubFullAttentionSpec

    # get_dtype_size stub — returns bytes per element for the identifier we
    # use as "dtype" in test specs (int).
    torch_utils_mod = types.ModuleType("vllm.utils.torch_utils")
    torch_utils_mod.get_dtype_size = lambda d: d if isinstance(d, int) else 1

    # Extra stubs so _patch_sliding_window_turboquant (the OTHER patch
    # installed by register()) doesn't crash on missing modules.
    interface_mod.SlidingWindowSpec = StubFullAttentionSpec
    interface_mod.get_kv_quant_mode = lambda _: 0

    class _StubAttention:
        def get_kv_cache_spec(self, vllm_config):  # noqa: D401
            return None

    attn_stub_mod = types.ModuleType(
        "vllm.model_executor.layers.attention.attention"
    )
    attn_stub_mod.Attention = _StubAttention

    tq_cfg_mod = types.ModuleType(
        "vllm.model_executor.layers.quantization.turboquant.config"
    )

    parents = [
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.utils",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.turboquant",
    ]
    for name in parents:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "vllm.v1.core.kv_cache_utils", core_utils_mod
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.kv_cache_interface", interface_mod
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention.attention",
        attn_stub_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.turboquant.config",
        tq_cfg_mod,
    )

    return core_utils_mod, StubMambaSpec, StubFullAttentionSpec


def _prod(shape):
    out = 1
    for s in shape:
        out *= s
    return out


def _dtype_size(d):
    return d if isinstance(d, int) else 1


def test_plugin_mixed_pads_mamba_and_scales_attention(monkeypatch):
    """Qwen3.6-style mismatch: attention page=102 per token × block, Mamba
    padded by upstream to 256 × block. Our patch must find a common
    multiple and scale attention + pad Mamba to it.
    """
    core_utils_mod, StubMambaSpec, StubFullAttentionSpec = (
        _install_stub_kv_cache_modules(monkeypatch)
    )

    from llm_workflow_agents.vllm_plugin import register

    register()

    # Attention spec: 102 bytes/token × block_size=32 = page 3264.
    attn = StubFullAttentionSpec(name="attn.0", block_size=32, per_token_page=102)
    # Mamba raw page: 2000 bytes (under the target). Upstream would have
    # padded it to 256×32 = 8192, which our patch must REPLACE.
    mamba = StubMambaSpec(
        name="mamba.0",
        shapes=((2000,),),
        dtypes=(1,),  # 1 byte/elem → raw page = 2000
    )
    kv_cache_spec = {"mamba.0": mamba, "attn.0": attn}

    result = core_utils_mod.unify_kv_cache_spec_page_size(kv_cache_spec)

    # Target = LCM(3264) = 3264; must be >= raw_mamba=2000. Target=3264.
    # attn.page = 3264 → unchanged.
    # mamba padded to 3264.
    assert result["attn.0"].page_size_bytes == 3264
    assert result["attn.0"].block_size == 32
    assert result["mamba.0"].page_size_bytes == 3264
    assert result["mamba.0"].page_size_padded == 3264


def test_plugin_target_bumped_when_mamba_raw_exceeds_attn_page(monkeypatch):
    core_utils_mod, StubMambaSpec, StubFullAttentionSpec = (
        _install_stub_kv_cache_modules(monkeypatch)
    )
    from llm_workflow_agents.vllm_plugin import register

    register()

    # attn page=100; mamba raw=250. Target = smallest multiple of 100 >= 250 = 300.
    attn = StubFullAttentionSpec(name="attn.0", block_size=10, per_token_page=10)
    mamba = StubMambaSpec(name="mamba.0", shapes=((250,),), dtypes=(1,))

    result = core_utils_mod.unify_kv_cache_spec_page_size(
        {"attn.0": attn, "mamba.0": mamba}
    )

    # attn needs ratio=3, block 10→30.
    assert result["attn.0"].block_size == 30
    assert result["attn.0"].page_size_bytes == 300
    assert result["mamba.0"].page_size_bytes == 300


def test_plugin_lcm_unifies_incompatible_attention_pages(monkeypatch):
    """Gemma-4 + TurboQuant: pages 198, 1024, 2048 are pairwise non-divisible.
    Upstream's unify raises; our LCM-based path scales each block_size so
    every spec ends up at target = LCM(pages).
    """
    core_utils_mod, _, StubFullAttentionSpec = _install_stub_kv_cache_modules(
        monkeypatch
    )
    from llm_workflow_agents.vllm_plugin import register

    register()

    # Mirror Gemma-4 pattern: three layers, three distinct page sizes, no Mamba.
    tq = StubFullAttentionSpec(name="narrow_tq", block_size=32, per_token_page=198)
    wide = StubFullAttentionSpec(name="wide", block_size=32, per_token_page=2048)
    sw = StubFullAttentionSpec(name="sw_skip", block_size=32, per_token_page=1024)
    # LCM(198*32, 2048*32, 1024*32) = 32 * LCM(198, 2048, 1024) = 32 * 202752 = 6488064

    result = core_utils_mod.unify_kv_cache_spec_page_size(
        {"narrow_tq": tq, "wide": wide, "sw_skip": sw}
    )

    expected_target = 32 * 202752
    assert result["narrow_tq"].page_size_bytes == expected_target
    assert result["wide"].page_size_bytes == expected_target
    assert result["sw_skip"].page_size_bytes == expected_target
    # Verify block_sizes were scaled proportionally.
    assert result["narrow_tq"].block_size == 32 * (202752 // 198)
    assert result["wide"].block_size == 32 * (202752 // 2048)
    assert result["sw_skip"].block_size == 32 * (202752 // 1024)


def test_plugin_lcm_passthrough_for_compatible_pages(monkeypatch):
    """When pages already divide cleanly (say 512 and 1024), LCM=1024; 512
    gets scaled to 1024. Preserves the cheap case."""
    core_utils_mod, _, StubFullAttentionSpec = _install_stub_kv_cache_modules(
        monkeypatch
    )
    from llm_workflow_agents.vllm_plugin import register

    register()

    small = StubFullAttentionSpec(name="s", block_size=16, per_token_page=512)
    large = StubFullAttentionSpec(name="l", block_size=16, per_token_page=1024)
    result = core_utils_mod.unify_kv_cache_spec_page_size({"s": small, "l": large})

    # LCM = 1024 * 16 = 16384. small scales by 2 (block 16 → 32).
    assert result["s"].block_size == 32
    assert result["s"].page_size_bytes == 16384
    assert result["l"] is large


def test_plugin_noop_when_all_equal(monkeypatch):
    core_utils_mod, _, StubFullAttentionSpec = _install_stub_kv_cache_modules(
        monkeypatch
    )
    from llm_workflow_agents.vllm_plugin import register

    register()

    a = StubFullAttentionSpec(name="a", block_size=16, per_token_page=10)
    b = StubFullAttentionSpec(name="b", block_size=16, per_token_page=10)
    spec = {"a": a, "b": b}
    result = core_utils_mod.unify_kv_cache_spec_page_size(spec)
    assert result is spec


def test_plugin_idempotent(monkeypatch):
    core_utils_mod, _, _ = _install_stub_kv_cache_modules(monkeypatch)
    from llm_workflow_agents.vllm_plugin import register

    register()
    patched_once = core_utils_mod.unify_kv_cache_spec_page_size
    register()
    patched_twice = core_utils_mod.unify_kv_cache_spec_page_size
    assert patched_once is patched_twice


def test_plugin_delegates_for_all_mamba(monkeypatch):
    """Only Mamba specs with different page sizes — upstream logic applies
    (as we have no clear heuristic to interpret mamba-only divergence)."""
    core_utils_mod, StubMambaSpec, _ = _install_stub_kv_cache_modules(monkeypatch)
    from llm_workflow_agents.vllm_plugin import register

    register()

    a = StubMambaSpec(name="a", shapes=((100,),), dtypes=(1,))
    b = StubMambaSpec(name="b", shapes=((200,),), dtypes=(1,))
    result = core_utils_mod.unify_kv_cache_spec_page_size({"a": a, "b": b})
    assert result == {"_orig_called": True}
