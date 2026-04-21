"""Tests for the plugin's sliding-window + TurboQuant patch.

Gemma-4 narrow layers are sliding (head_dim=256, sliding_window set), so
``Attention.get_kv_cache_spec`` originally returns ``SlidingWindowSpec`` —
whose ``page_size_bytes`` uses FullAttention math (2×head×dtype_size) and
mismatches ``TurboQuantAttentionBackend.get_kv_cache_shape`` which uses
``slot_size_aligned``. The plugin patches Attention.get_kv_cache_spec to
return a new ``TQSlidingWindowSpec`` whose ``real_page_size_bytes`` is
``block × num_kv × tq_slot_size`` instead.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


def _install_stubs(monkeypatch):
    """Stub vllm modules needed by _patch_sliding_window_turboquant."""

    # kv_cache_interface: SlidingWindowSpec, get_kv_quant_mode.
    @dataclass(frozen=True, kw_only=True)
    class StubSlidingWindowSpec:
        block_size: int
        num_kv_heads: int
        head_size: int
        dtype: object
        kv_quant_mode: int = 0
        sliding_window: int = 0

        @property
        def real_page_size_bytes(self) -> int:
            # Mimic FullAttention math: 2 × block × num_kv × head × dtype_size
            dtype_size = self.dtype if isinstance(self.dtype, int) else 1
            return 2 * self.block_size * self.num_kv_heads * self.head_size * dtype_size

        @property
        def page_size_bytes(self) -> int:
            return self.real_page_size_bytes

    def get_kv_quant_mode(kv_cache_dtype):
        return 0  # KVQuantMode.NONE sentinel

    kvc_mod = types.ModuleType("vllm.v1.kv_cache_interface")
    kvc_mod.SlidingWindowSpec = StubSlidingWindowSpec
    kvc_mod.get_kv_quant_mode = get_kv_quant_mode

    # Attention module
    class StubAttention:
        def __init__(
            self,
            *,
            kv_cache_dtype="auto",
            sliding_window=None,
            head_size=256,
            num_kv_heads=8,
            kv_cache_torch_dtype=1,
        ):
            self.kv_cache_dtype = kv_cache_dtype
            self.sliding_window = sliding_window
            self.head_size = head_size
            self.num_kv_heads = num_kv_heads
            self.kv_cache_torch_dtype = kv_cache_torch_dtype

        def get_kv_cache_spec(self, vllm_config):
            # Original implementation — our patch wraps this.
            return ("ORIG_CALLED", self.kv_cache_dtype, self.sliding_window)

    attn_mod = types.ModuleType("vllm.model_executor.layers.attention.attention")
    attn_mod.Attention = StubAttention

    # TurboQuant config module
    class StubTurboQuantConfig:
        def __init__(self, slot_size_aligned):
            self.slot_size_aligned = slot_size_aligned

        @staticmethod
        def from_cache_dtype(cache_dtype, head_dim):
            # Deterministic fake slot size per (dtype, head) for test assertions.
            table = {
                ("turboquant_3bit_nc", 256): 198,
                ("turboquant_3bit_nc", 128): 102,
                ("turboquant_4bit_nc", 256): 260,
            }
            return StubTurboQuantConfig(slot_size_aligned=table[(cache_dtype, head_dim)])

    tq_cfg_mod = types.ModuleType(
        "vllm.model_executor.layers.quantization.turboquant.config"
    )
    tq_cfg_mod.TurboQuantConfig = StubTurboQuantConfig

    # Register all modules + parents.
    parents = [
        "vllm",
        "vllm.v1",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.turboquant",
    ]
    for name in parents:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", kvc_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention.attention",
        attn_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.turboquant.config",
        tq_cfg_mod,
    )
    # Also need kv_cache_utils for the OTHER patch function that register()
    # calls. Provide a minimal stub so register() doesn't crash.
    core_utils_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
    core_utils_mod.unify_kv_cache_spec_page_size = lambda d: d
    monkeypatch.setitem(
        sys.modules, "vllm.v1.core.kv_cache_utils", core_utils_mod
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.core", types.ModuleType("vllm.v1.core")
    )

    # torch_utils for the plugin's other patch.
    torch_utils_mod = types.ModuleType("vllm.utils.torch_utils")
    torch_utils_mod.get_dtype_size = lambda d: d if isinstance(d, int) else 1
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils_mod)
    if "vllm.utils" not in sys.modules:
        monkeypatch.setitem(sys.modules, "vllm.utils", types.ModuleType("vllm.utils"))

    # Stub single_type_kv_cache_manager.spec_manager_map so the plugin's
    # registration of TQSlidingWindowSpec → SlidingWindowManager runs.
    class StubSlidingWindowManager:
        pass

    mgr_mod = types.ModuleType("vllm.v1.core.single_type_kv_cache_manager")
    mgr_mod.spec_manager_map = {}
    mgr_mod.SlidingWindowManager = StubSlidingWindowManager
    monkeypatch.setitem(
        sys.modules, "vllm.v1.core.single_type_kv_cache_manager", mgr_mod
    )
    if "vllm.v1.core" not in sys.modules:
        monkeypatch.setitem(
            sys.modules, "vllm.v1.core", types.ModuleType("vllm.v1.core")
        )

    return StubAttention, kvc_mod, mgr_mod


def _make_vllm_config(block_size=32):
    return SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size))


def test_sliding_turboquant_returns_tq_slot_size(monkeypatch):
    StubAttention, kvc_mod, mgr_mod = _install_stubs(monkeypatch)

    from llm_workflow_agents.vllm_plugin import register

    register()

    attn = StubAttention(
        kv_cache_dtype="turboquant_3bit_nc",
        sliding_window=1024,
        head_size=256,
        num_kv_heads=8,
    )
    spec = attn.get_kv_cache_spec(_make_vllm_config(block_size=32))

    # Should be an instance of the new TQSlidingWindowSpec injected onto the
    # kv_cache_interface module.
    assert isinstance(spec, kvc_mod.TQSlidingWindowSpec)
    # And a subclass of the upstream SlidingWindowSpec, so downstream
    # isinstance(_, SlidingWindowSpec) checks still see it.
    assert isinstance(spec, kvc_mod.SlidingWindowSpec)
    assert spec.tq_slot_size == 198
    assert spec.sliding_window == 1024
    # real_page_size_bytes must use TQ math, not FullAttention math.
    assert spec.real_page_size_bytes == 32 * 8 * 198


def test_sliding_non_turboquant_delegates_to_original(monkeypatch):
    StubAttention, _, _ = _install_stubs(monkeypatch)

    from llm_workflow_agents.vllm_plugin import register

    register()

    attn = StubAttention(kv_cache_dtype="auto", sliding_window=1024)
    spec = attn.get_kv_cache_spec(_make_vllm_config())
    assert spec[0] == "ORIG_CALLED"


def test_non_sliding_turboquant_delegates_to_original(monkeypatch):
    """Full-attention + turboquant is upstream's job (TQFullAttentionSpec)."""
    StubAttention, _, _ = _install_stubs(monkeypatch)

    from llm_workflow_agents.vllm_plugin import register

    register()

    attn = StubAttention(
        kv_cache_dtype="turboquant_3bit_nc", sliding_window=None, head_size=256
    )
    spec = attn.get_kv_cache_spec(_make_vllm_config())
    assert spec[0] == "ORIG_CALLED"


def test_non_sliding_non_turboquant_delegates_to_original(monkeypatch):
    StubAttention, _, _ = _install_stubs(monkeypatch)

    from llm_workflow_agents.vllm_plugin import register

    register()

    attn = StubAttention(kv_cache_dtype="auto", sliding_window=None)
    spec = attn.get_kv_cache_spec(_make_vllm_config())
    assert spec[0] == "ORIG_CALLED"


def test_tq_slot_size_varies_with_head_size(monkeypatch):
    """Smaller head_dim → smaller tq_slot_size (per our test table)."""
    StubAttention, kvc_mod, mgr_mod = _install_stubs(monkeypatch)

    from llm_workflow_agents.vllm_plugin import register

    register()

    attn = StubAttention(
        kv_cache_dtype="turboquant_3bit_nc",
        sliding_window=1024,
        head_size=128,  # Qwen-style
        num_kv_heads=16,
    )
    spec = attn.get_kv_cache_spec(_make_vllm_config(block_size=64))
    assert spec.tq_slot_size == 102
    assert spec.real_page_size_bytes == 64 * 16 * 102


def test_tq_sliding_window_spec_registered_in_manager_map(monkeypatch):
    """Plugin must add TQSlidingWindowSpec → SlidingWindowManager to
    single_type_kv_cache_manager.spec_manager_map, otherwise the scheduler's
    KVCacheManager crashes with KeyError when it looks up the type.
    """
    _, kvc_mod, mgr_mod = _install_stubs(monkeypatch)

    from llm_workflow_agents.vllm_plugin import register

    register()

    assert kvc_mod.TQSlidingWindowSpec in mgr_mod.spec_manager_map
    assert (
        mgr_mod.spec_manager_map[kvc_mod.TQSlidingWindowSpec]
        is mgr_mod.SlidingWindowManager
    )


def test_patch_idempotent(monkeypatch):
    StubAttention, _, _ = _install_stubs(monkeypatch)

    from llm_workflow_agents.vllm_plugin import register

    register()
    patched_once = StubAttention.get_kv_cache_spec
    register()
    patched_twice = StubAttention.get_kv_cache_spec
    assert patched_once is patched_twice
    assert getattr(patched_twice, "_tq_patched", False) is True
