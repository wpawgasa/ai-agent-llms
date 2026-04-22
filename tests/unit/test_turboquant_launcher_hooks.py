"""Tests for the project's TurboQuant launcher hooks.

Two hooks are exercised here:

1. ``_install_gemma4_mixed_backend_hook`` — patches
   ``vllm.model_executor.models.config.Gemma4Config.verify_and_update_config``
   so wide Gemma-4 full-attention layers land in
   ``cache_config.kv_cache_dtype_skip_layers`` instead of vLLM forcing
   TRITON_ATTN globally.

2. ``_install_turboquant_engine_config_hook`` — wraps
   ``vllm.engine.arg_utils.EngineArgs.create_engine_config`` to
   (a) bypass the hybrid-model NotImplementedError at arg_utils.py:1649 by
   temporarily masking ``ModelConfig.is_hybrid`` to False, and
   (b) auto-inject ``enforce_eager=True`` for Gemma-4 models so vLLM v1's
   broken mixed-KV profiler (gpu_model_runner.py:6598) is skipped.

All tests stub vLLM imports via ``sys.modules`` so no CUDA paths load.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Gemma-4 backend hook: shared stub + 6 existing tests
# ---------------------------------------------------------------------------


def _install_stub_vllm_config(monkeypatch):
    """Register a fake ``vllm.model_executor.models.config`` module."""

    call_log: list[str] = []

    class StubGemma4Config:
        @staticmethod
        def verify_and_update_config(vllm_config):
            call_log.append("original")

    stub_module = types.ModuleType("vllm.model_executor.models.config")
    stub_module.Gemma4Config = StubGemma4Config

    parents = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
    ]
    for name in parents:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.config", stub_module
    )

    return StubGemma4Config, call_log


def _make_vllm_config(
    *,
    cache_dtype: str,
    layer_types,
    head_dim: int = 128,
    global_head_dim: int = 512,
    num_hidden_layers: int | None = None,
    existing_skip: list[str] | None = None,
):
    if num_hidden_layers is None and layer_types is not None:
        num_hidden_layers = len(layer_types)
    hf_text_config = SimpleNamespace(
        layer_types=layer_types,
        num_hidden_layers=num_hidden_layers,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
    )
    cache_config = SimpleNamespace(
        cache_dtype=cache_dtype,
        kv_cache_dtype_skip_layers=list(existing_skip or []),
    )
    model_config = SimpleNamespace(hf_text_config=hf_text_config)
    return SimpleNamespace(cache_config=cache_config, model_config=model_config)


def test_auto_dtype_delegates_to_original(monkeypatch):
    stub_cls, call_log = _install_stub_vllm_config(monkeypatch)

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_gemma4_mixed_backend_hook,
    )

    _install_gemma4_mixed_backend_hook()

    cfg = _make_vllm_config(
        cache_dtype="auto",
        layer_types=["full_attention", "sliding", "full_attention", "sliding"],
        existing_skip=["0", "3"],
    )
    stub_cls.verify_and_update_config(cfg)

    assert call_log == ["original"]
    assert cfg.cache_config.kv_cache_dtype_skip_layers == ["0", "3"]


def test_plain_turboquant_also_triggers_routing(monkeypatch):
    stub_cls, call_log = _install_stub_vllm_config(monkeypatch)

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_gemma4_mixed_backend_hook,
    )

    _install_gemma4_mixed_backend_hook()

    cfg = _make_vllm_config(
        cache_dtype="turboquant",
        layer_types=["sliding", "full_attention", "sliding"],
        existing_skip=[],
    )
    stub_cls.verify_and_update_config(cfg)

    assert call_log == []
    assert cfg.cache_config.kv_cache_dtype_skip_layers == ["1"]


def test_turboquant_routes_wide_full_attention_layers(monkeypatch):
    stub_cls, call_log = _install_stub_vllm_config(monkeypatch)

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_gemma4_mixed_backend_hook,
    )

    _install_gemma4_mixed_backend_hook()

    layer_types = [
        "sliding",
        "full_attention",
        "sliding",
        "sliding",
        "full_attention",
        "sliding",
    ]
    cfg = _make_vllm_config(
        cache_dtype="turboquant_3bit_nc",
        layer_types=layer_types,
        existing_skip=["0", "5"],
    )
    stub_cls.verify_and_update_config(cfg)

    assert call_log == []
    assert cfg.cache_config.kv_cache_dtype_skip_layers == ["0", "1", "4", "5"]


def test_missing_layer_types_falls_back_to_original(monkeypatch):
    stub_cls, call_log = _install_stub_vllm_config(monkeypatch)

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_gemma4_mixed_backend_hook,
    )

    _install_gemma4_mixed_backend_hook()

    cfg = _make_vllm_config(
        cache_dtype="turboquant_3bit_nc",
        layer_types=None,
        num_hidden_layers=4,
        existing_skip=[],
    )
    stub_cls.verify_and_update_config(cfg)

    assert call_log == ["original"]
    assert cfg.cache_config.kv_cache_dtype_skip_layers == []


def test_layer_types_length_mismatch_raises(monkeypatch):
    stub_cls, _ = _install_stub_vllm_config(monkeypatch)

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_gemma4_mixed_backend_hook,
    )

    _install_gemma4_mixed_backend_hook()

    cfg = _make_vllm_config(
        cache_dtype="turboquant_3bit_nc",
        layer_types=["sliding", "full_attention"],
        num_hidden_layers=4,
    )
    with pytest.raises(AssertionError, match="layer_types"):
        stub_cls.verify_and_update_config(cfg)


def test_narrow_global_head_dim_leaves_skip_list_unchanged(monkeypatch):
    stub_cls, call_log = _install_stub_vllm_config(monkeypatch)

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_gemma4_mixed_backend_hook,
    )

    _install_gemma4_mixed_backend_hook()

    cfg = _make_vllm_config(
        cache_dtype="turboquant_3bit_nc",
        layer_types=["full_attention", "sliding", "full_attention"],
        head_dim=128,
        global_head_dim=256,
        existing_skip=["0", "2"],
    )
    stub_cls.verify_and_update_config(cfg)

    assert call_log == []
    assert cfg.cache_config.kv_cache_dtype_skip_layers == ["0", "2"]


# ---------------------------------------------------------------------------
# Engine-config hook: hybrid bypass + Gemma-4 enforce_eager auto-inject
# ---------------------------------------------------------------------------


def _install_stub_engine_config(monkeypatch):
    """Fake ``vllm.engine.arg_utils.EngineArgs``, ``vllm.config.model.ModelConfig``,
    and ``vllm.model_executor.models.config.HybridAttentionMambaModelConfig``.

    Returns ``(EngineArgsStub, ModelConfigStub, call_log)``. The stub
    ``create_engine_config`` raises NotImplementedError iff the ``EngineArgs``
    instance reports ``model_config.is_hybrid`` True — mirroring
    ``arg_utils.py:1645-1668``. It returns a ``SimpleNamespace`` carrying a
    ``model_config`` attribute so our post-hoc hybrid dispatcher can inspect
    ``model_config.is_hybrid`` after the call returns.
    """

    call_log: list[dict] = []

    class StubModelConfig:
        _force_hybrid = True

        @property
        def is_hybrid(self):  # type: ignore[override]
            return self._force_hybrid

    hybrid_dispatch_calls: list[object] = []

    class StubHybridAttentionMambaModelConfig:
        @classmethod
        def verify_and_update_config(cls, vllm_config):
            hybrid_dispatch_calls.append(vllm_config)

    class StubEngineArgs:
        def create_engine_config(self, *args, **kwargs):
            hybrid_seen = self.model_config.is_hybrid
            call_log.append({
                "model": self.model,
                "kv_cache_dtype": self.kv_cache_dtype,
                "enforce_eager": self.enforce_eager,
                "is_hybrid_during_call": hybrid_seen,
            })
            if (
                self.kv_cache_dtype
                and self.kv_cache_dtype.startswith("turboquant_")
                and hybrid_seen
            ):
                raise NotImplementedError(
                    "TurboQuant KV cache is not supported for hybrid models"
                )
            return SimpleNamespace(
                dummy_vllm_config=True,
                model_config=self.model_config,
            )

    engine_module = types.ModuleType("vllm.engine.arg_utils")
    engine_module.EngineArgs = StubEngineArgs

    model_module = types.ModuleType("vllm.config.model")
    model_module.ModelConfig = StubModelConfig

    mm_config_module = types.ModuleType("vllm.model_executor.models.config")
    # The existing Gemma-4 stub module registers the same import path.
    # Preserve whatever Gemma4Config lives there (if any) by attribute-copy.
    existing = sys.modules.get("vllm.model_executor.models.config")
    if existing is not None and hasattr(existing, "Gemma4Config"):
        mm_config_module.Gemma4Config = existing.Gemma4Config
    mm_config_module.HybridAttentionMambaModelConfig = (
        StubHybridAttentionMambaModelConfig
    )

    parents = [
        "vllm",
        "vllm.engine",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.models",
    ]
    for name in parents:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", engine_module)
    monkeypatch.setitem(sys.modules, "vllm.config.model", model_module)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.config", mm_config_module
    )

    # Stash hybrid_dispatch_calls on the module for test inspection.
    mm_config_module._hybrid_dispatch_calls = hybrid_dispatch_calls

    return StubEngineArgs, StubModelConfig, call_log


def _install_stub_transformers(monkeypatch, architectures, layer_types=None):
    """Fake ``transformers.AutoConfig.from_pretrained`` returning given archs."""

    class StubAutoConfig:
        @staticmethod
        def from_pretrained(model_name, trust_remote_code=False):
            text_config = SimpleNamespace(layer_types=layer_types)
            return SimpleNamespace(
                architectures=list(architectures),
                text_config=text_config,
            )

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = StubAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)


def _make_engine_args(
    *,
    model: str,
    kv_cache_dtype: str,
    enforce_eager: bool,
    model_config,
    gdn_prefill_backend=None,
    enable_prefix_caching=None,  # real EngineArgs default
):
    return SimpleNamespace(
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
        model_config=model_config,
        gdn_prefill_backend=gdn_prefill_backend,
        enable_prefix_caching=enable_prefix_caching,
    )


def test_hybrid_bypass_suppresses_notimplemented(monkeypatch):
    EngineArgsStub, ModelConfigStub, call_log = _install_stub_engine_config(
        monkeypatch
    )
    _install_stub_transformers(monkeypatch, architectures=["Qwen3MoeForCausalLM"])

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    pre_install_property = ModelConfigStub.__dict__["is_hybrid"]
    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
    )
    # MUST NOT raise — patched wrapper masks is_hybrid to False.
    result = EngineArgsStub.create_engine_config(args)
    assert result.dummy_vllm_config is True
    assert len(call_log) == 1
    assert call_log[0]["is_hybrid_during_call"] is False

    # After the call, ModelConfig.is_hybrid must be restored to the
    # original property descriptor.
    assert ModelConfigStub.__dict__["is_hybrid"] is pre_install_property
    # Not a Gemma-4 model, so enforce_eager stays False.
    assert args.enforce_eager is False


def test_gemma4_no_longer_forces_enforce_eager(monkeypatch):
    """Gemma-4 + TurboQuant no longer auto-sets enforce_eager.
    Dev vLLM already iterates _reshape_kv_cache_tensors per-group, and
    _patch_unify_kv_cache_spec_page_size resolves the page-size divisibility
    issue. CUDA graphs are now allowed for Gemma-4.
    """
    EngineArgsStub, ModelConfigStub, call_log = _install_stub_engine_config(
        monkeypatch
    )
    _install_stub_transformers(
        monkeypatch, architectures=["Gemma4ForCausalLM"]
    )
    ModelConfigStub._force_hybrid = False  # Gemma-4 is not hybrid

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="google/gemma-4-31B-it",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
    )
    EngineArgsStub.create_engine_config(args)

    # Hook must NOT flip enforce_eager for Gemma-4 — CUDA graphs are allowed.
    assert args.enforce_eager is False
    assert call_log[0]["enforce_eager"] is False


def test_non_turboquant_dtype_passes_through(monkeypatch):
    EngineArgsStub, ModelConfigStub, call_log = _install_stub_engine_config(
        monkeypatch
    )
    _install_stub_transformers(
        monkeypatch, architectures=["Gemma4ForCausalLM"]
    )
    ModelConfigStub._force_hybrid = True  # even hybrid should pass through

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    pre_install_property = ModelConfigStub.__dict__["is_hybrid"]
    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="google/gemma-4-31B-it",
        kv_cache_dtype="auto",
        enforce_eager=False,
        model_config=ModelConfigStub(),
    )
    # auto dtype → orig called unchanged, BUT the stub raises only for
    # turboquant_* + hybrid, so "auto" + hybrid=True must NOT raise.
    EngineArgsStub.create_engine_config(args)

    # No flip, no mask.
    assert args.enforce_eager is False
    assert call_log[0]["is_hybrid_during_call"] is True
    assert ModelConfigStub.__dict__["is_hybrid"] is pre_install_property


def test_idempotent_install(monkeypatch):
    EngineArgsStub, ModelConfigStub, _ = _install_stub_engine_config(monkeypatch)
    _install_stub_transformers(monkeypatch, architectures=["Qwen3MoeForCausalLM"])

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()
    patched_once = EngineArgsStub.create_engine_config

    _install_turboquant_engine_config_hook()
    patched_twice = EngineArgsStub.create_engine_config

    # Second install must be a no-op; same wrapper object.
    assert patched_once is patched_twice
    assert getattr(patched_twice, "_turboquant_wrapped", False) is True


def test_gdn_layers_auto_switch_to_triton(monkeypatch):
    """Qwen3.5/3.6 has ``linear_attention`` in layer_types → triton GDN."""
    EngineArgsStub, ModelConfigStub, call_log = _install_stub_engine_config(
        monkeypatch
    )
    _install_stub_transformers(
        monkeypatch,
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        layer_types=["full_attention", "linear_attention", "full_attention"],
    )

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
        gdn_prefill_backend=None,
    )
    EngineArgsStub.create_engine_config(args)

    assert args.gdn_prefill_backend == "triton"
    # GDN models also have mixed KV groups → enforce_eager required.
    assert args.enforce_eager is True


def test_gdn_explicit_user_choice_preserved(monkeypatch):
    """If the user already set gdn_prefill_backend, don't override."""
    EngineArgsStub, ModelConfigStub, _ = _install_stub_engine_config(monkeypatch)
    _install_stub_transformers(
        monkeypatch,
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        layer_types=["full_attention", "linear_attention"],
    )

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
        gdn_prefill_backend="flashinfer",  # user explicit choice
    )
    EngineArgsStub.create_engine_config(args)

    assert args.gdn_prefill_backend == "flashinfer"


def test_hybrid_dispatcher_runs_post_create(monkeypatch):
    """After is_hybrid mask is lifted, our hook must run
    HybridAttentionMambaModelConfig.verify_and_update_config so
    cache_config.mamba_block_size (and related hybrid defaults) get set.
    Without this, unify_kv_cache_spec_page_size at kv_cache_utils.py:958
    crashes with ``None * int``.
    """
    EngineArgsStub, ModelConfigStub, _ = _install_stub_engine_config(monkeypatch)
    _install_stub_transformers(
        monkeypatch,
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        layer_types=["full_attention", "linear_attention"],
    )

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
        gdn_prefill_backend=None,
    )
    EngineArgsStub.create_engine_config(args)

    mm_config_module = sys.modules["vllm.model_executor.models.config"]
    hybrid_calls = mm_config_module._hybrid_dispatch_calls
    # Should have been dispatched exactly once after orig returned.
    assert len(hybrid_calls) == 1
    assert hybrid_calls[0].model_config is args.model_config


def test_hybrid_dispatcher_not_called_on_non_hybrid(monkeypatch):
    """Non-hybrid models (Gemma-4 included) don't need the dispatcher."""
    EngineArgsStub, ModelConfigStub, _ = _install_stub_engine_config(monkeypatch)
    _install_stub_transformers(
        monkeypatch,
        architectures=["Gemma4ForCausalLM"],
        layer_types=["full_attention", "sliding"],
    )
    ModelConfigStub._force_hybrid = False

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="google/gemma-4-31B-it",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
    )
    EngineArgsStub.create_engine_config(args)

    mm_config_module = sys.modules["vllm.model_executor.models.config"]
    assert mm_config_module._hybrid_dispatch_calls == []


def test_mixed_kv_disables_prefix_caching(monkeypatch):
    """Gemma-4 or GDN hybrid: our LCM scaling produces heterogeneous group
    block_sizes, which HybridKVCacheCoordinator refuses. Disable prefix
    caching to route to KVCacheCoordinatorNoPrefixCache (no assertion).
    Gemma-4 no longer forces enforce_eager; prefix caching is still disabled
    because the HybridKVCacheCoordinator block-size assertion is a separate
    issue unrelated to the profiler fix."""
    EngineArgsStub, ModelConfigStub, _ = _install_stub_engine_config(monkeypatch)
    _install_stub_transformers(
        monkeypatch, architectures=["Gemma4ForCausalLM"], layer_types=[]
    )
    ModelConfigStub._force_hybrid = False

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="google/gemma-4-31B-it",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
        enable_prefix_caching=True,
    )
    EngineArgsStub.create_engine_config(args)

    assert args.enforce_eager is False  # Gemma-4 no longer forced
    assert args.enable_prefix_caching is False


def test_mixed_kv_disables_prefix_caching_from_none_default(monkeypatch):
    """EngineArgs.enable_prefix_caching defaults to None (auto-resolved to
    True later). Must pin to False explicitly, otherwise the auto-resolver
    re-enables it and the HybridKVCacheCoordinator assertion fires.
    """
    EngineArgsStub, ModelConfigStub, _ = _install_stub_engine_config(monkeypatch)
    _install_stub_transformers(
        monkeypatch, architectures=["Gemma4ForCausalLM"], layer_types=[]
    )
    ModelConfigStub._force_hybrid = False

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="google/gemma-4-31B-it",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
        enable_prefix_caching=None,  # EngineArgs true default
    )
    EngineArgsStub.create_engine_config(args)

    assert args.enable_prefix_caching is False


def test_dense_model_preserves_prefix_caching(monkeypatch):
    """Dense (non-Gemma4, non-GDN) TurboQuant runs: don't touch prefix cache."""
    EngineArgsStub, ModelConfigStub, _ = _install_stub_engine_config(monkeypatch)
    _install_stub_transformers(
        monkeypatch, architectures=["Qwen3ForCausalLM"], layer_types=["full_attention"]
    )
    ModelConfigStub._force_hybrid = False

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="Qwen/Qwen3-32B",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
        enable_prefix_caching=True,
    )
    EngineArgsStub.create_engine_config(args)

    # Dense model → no enforce_eager flip → no prefix-cache flip.
    assert args.enforce_eager is False
    assert args.enable_prefix_caching is True


def test_dense_model_does_not_touch_gdn_backend(monkeypatch):
    """Qwen3-32B (dense) has no linear_attention layers — no GDN override."""
    EngineArgsStub, ModelConfigStub, _ = _install_stub_engine_config(monkeypatch)
    _install_stub_transformers(
        monkeypatch,
        architectures=["Qwen3ForCausalLM"],
        layer_types=["full_attention"] * 8,
    )
    ModelConfigStub._force_hybrid = False

    from llm_workflow_agents.serving.launch_vllm_turboquant import (
        _install_turboquant_engine_config_hook,
    )

    _install_turboquant_engine_config_hook()

    args = _make_engine_args(
        model="Qwen/Qwen3-32B",
        kv_cache_dtype="turboquant_3bit_nc",
        enforce_eager=False,
        model_config=ModelConfigStub(),
        gdn_prefill_backend=None,
    )
    EngineArgsStub.create_engine_config(args)

    # No GDN layers → no override, stays None.
    assert args.gdn_prefill_backend is None
