"""Tests for the plugin's FlashAttention supports_head_size cap.

When cache_dtype starts with ``turboquant_``, upstream's ``arg_utils.py:1982``
forces the flash_attn version down to 2 (TurboQuant incompatible with FA≥3).
FA v2 does NOT support head_size > 256 at runtime. But
``FlashAttention.supports_head_size`` at flash_attn.py:174-181 returns True
for head_size ≤ 512 when FA v4 is detected on the SM, so backend selection
picks FA for wide Gemma-4 layers and crashes later at
``FlashAttention forward only supports head dimension at most 256``.

The plugin caps ``supports_head_size`` at 256 whenever the current
cache_dtype is turboquant-family, forcing the selector onto TRITON_ATTN.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace


def _install_stubs(monkeypatch, cache_dtype="turboquant_3bit_nc"):
    """Stub the FlashAttentionBackend and vllm_config machinery."""

    class StubFABackend:
        @classmethod
        def supports_head_size(cls, head_size: int) -> bool:
            # Upstream-ish: accept anything up to 512 (emulates FA v4 available).
            return head_size % 8 == 0 and head_size <= 512

    fa_mod = types.ModuleType("vllm.v1.attention.backends.flash_attn")
    fa_mod.FlashAttentionBackend = StubFABackend

    # vllm.config.vllm.get_current_vllm_config returns something with
    # .cache_config.cache_dtype — respected by our gate.
    vllm_config_mod = types.ModuleType("vllm.config.vllm")

    def get_current_vllm_config():
        return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))

    vllm_config_mod.get_current_vllm_config = get_current_vllm_config

    # Minimal stubs for the OTHER patches register() runs.
    core_utils_mod = types.ModuleType("vllm.v1.core.kv_cache_utils")
    core_utils_mod.unify_kv_cache_spec_page_size = lambda d: d

    interface_mod = types.ModuleType("vllm.v1.kv_cache_interface")

    class _Dummy:
        pass

    interface_mod.SlidingWindowSpec = _Dummy
    interface_mod.get_kv_quant_mode = lambda _: 0

    attn_layer_mod = types.ModuleType(
        "vllm.model_executor.layers.attention.attention"
    )

    class _StubAttention:
        def get_kv_cache_spec(self, vllm_config):
            return None

    attn_layer_mod.Attention = _StubAttention

    tq_cfg_mod = types.ModuleType(
        "vllm.model_executor.layers.quantization.turboquant.config"
    )

    mgr_mod = types.ModuleType("vllm.v1.core.single_type_kv_cache_manager")
    mgr_mod.spec_manager_map = {}
    mgr_mod.SlidingWindowManager = _Dummy

    torch_utils_mod = types.ModuleType("vllm.utils.torch_utils")
    torch_utils_mod.get_dtype_size = lambda d: 1

    parents = [
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.config",
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

    for name, mod in [
        ("vllm.v1.attention.backends.flash_attn", fa_mod),
        ("vllm.config.vllm", vllm_config_mod),
        ("vllm.v1.core.kv_cache_utils", core_utils_mod),
        ("vllm.v1.kv_cache_interface", interface_mod),
        ("vllm.model_executor.layers.attention.attention", attn_layer_mod),
        (
            "vllm.model_executor.layers.quantization.turboquant.config",
            tq_cfg_mod,
        ),
        ("vllm.v1.core.single_type_kv_cache_manager", mgr_mod),
        ("vllm.utils.torch_utils", torch_utils_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    return StubFABackend


def test_turboquant_caps_head_size_at_256(monkeypatch):
    StubFABackend = _install_stubs(monkeypatch, cache_dtype="turboquant_3bit_nc")

    from llm_workflow_agents.vllm_plugin import register

    register()

    # 256 is still fine.
    assert StubFABackend.supports_head_size(256) is True
    assert StubFABackend.supports_head_size(128) is True
    # 512 must be refused under turboquant.
    assert StubFABackend.supports_head_size(512) is False


def test_non_turboquant_preserves_upstream_head_size_support(monkeypatch):
    """auto dtype → upstream's FA can claim 512 as usual."""
    StubFABackend = _install_stubs(monkeypatch, cache_dtype="auto")

    from llm_workflow_agents.vllm_plugin import register

    register()

    assert StubFABackend.supports_head_size(512) is True
    assert StubFABackend.supports_head_size(256) is True


def test_fp8_kv_dtype_unaffected(monkeypatch):
    StubFABackend = _install_stubs(monkeypatch, cache_dtype="fp8")

    from llm_workflow_agents.vllm_plugin import register

    register()

    assert StubFABackend.supports_head_size(512) is True


def test_odd_head_size_still_rejected(monkeypatch):
    """Upstream check (head % 8 == 0) still applies."""
    StubFABackend = _install_stubs(monkeypatch, cache_dtype="turboquant_3bit_nc")

    from llm_workflow_agents.vllm_plugin import register

    register()

    assert StubFABackend.supports_head_size(257) is False  # not divisible by 8
    assert StubFABackend.supports_head_size(248) is True  # 248 % 8 == 0, ≤ 256


def test_patch_idempotent(monkeypatch):
    StubFABackend = _install_stubs(monkeypatch)

    from llm_workflow_agents.vllm_plugin import register

    register()
    # Classmethod access produces a new bound method object per access, so
    # compare the underlying function (__func__) and the sentinel attribute.
    raw_once = StubFABackend.__dict__["supports_head_size"]
    func_once = getattr(raw_once, "__func__", raw_once)
    register()
    raw_twice = StubFABackend.__dict__["supports_head_size"]
    func_twice = getattr(raw_twice, "__func__", raw_twice)
    assert func_once is func_twice
    assert getattr(func_twice, "_tq_patched", False) is True
