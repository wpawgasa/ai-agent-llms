"""Launch vLLM's OpenAI API server with the project's custom TurboQuant KV cache.

vLLM's CLI parser builds ``--kv-cache-dtype`` choices from a ``Literal[...]``
at import time, so plain ``"turboquant"`` is rejected before any project hook
can run. This launcher installs the TurboQuant runtime hooks, appends
``"turboquant"`` to the argparse action's choices, then runs the server
in-process.

Usage mirrors ``python -m vllm.entrypoints.openai.api_server`` — argv is
forwarded unchanged.
"""

from __future__ import annotations

import asyncio
import sys

from llm_workflow_agents.quantization.turboquant.vllm_integration import (
    TurboQuantConfig,
    register_turboquant_backend,
)


def _extend_kv_cache_choices(parser) -> None:
    for action in parser._actions:
        if "--kv-cache-dtype" in action.option_strings:
            if action.choices and "turboquant" not in action.choices:
                action.choices = list(action.choices) + ["turboquant"]
            return
    raise RuntimeError("vLLM parser has no --kv-cache-dtype action")


def _turboquant_bit_width_from_argv(argv: list[str]) -> int:
    for i, tok in enumerate(argv):
        if tok == "--turboquant-bit-width" and i + 1 < len(argv):
            argv.pop(i)
            return int(argv.pop(i))
        if tok.startswith("--turboquant-bit-width="):
            argv.pop(i)
            return int(tok.split("=", 1)[1])
    return 3


def _peek_kv_cache_dtype(argv: list[str]) -> str:
    """Non-destructive scan for ``--kv-cache-dtype VALUE`` / ``--kv-cache-dtype=VALUE``."""
    for i, tok in enumerate(argv):
        if tok == "--kv-cache-dtype" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--kv-cache-dtype="):
            return tok.split("=", 1)[1]
    return ""


_WIDE_HEAD_DIM_THRESHOLD = 256


def _install_gemma4_mixed_backend_hook() -> None:
    import structlog
    from vllm.model_executor.models.config import Gemma4Config

    logger = structlog.get_logger(__name__)
    original = Gemma4Config.verify_and_update_config

    def patched(vllm_config):
        cache_dtype = vllm_config.cache_config.cache_dtype
        if not cache_dtype.startswith("turboquant"):
            return original(vllm_config)

        hf = vllm_config.model_config.hf_text_config
        layer_types = getattr(hf, "layer_types", None)
        if layer_types is None:
            logger.warning("gemma4_hook_no_layer_types_fallback")
            return original(vllm_config)

        num_hidden = hf.num_hidden_layers
        assert len(layer_types) == num_hidden, (
            f"Gemma4 layer_types len={len(layer_types)} != "
            f"num_hidden_layers={num_hidden}"
        )

        head_dim = getattr(hf, "head_dim", 0) or 0
        global_head_dim = getattr(hf, "global_head_dim", head_dim) or head_dim

        wide = [
            str(i)
            for i, lt in enumerate(layer_types)
            if (global_head_dim if lt == "full_attention" else head_dim)
            > _WIDE_HEAD_DIM_THRESHOLD
        ]
        cc = vllm_config.cache_config
        merged = sorted(set(cc.kv_cache_dtype_skip_layers) | set(wide), key=int)
        cc.kv_cache_dtype_skip_layers = merged

        logger.info(
            "gemma4_mixed_backend_route",
            wide_layers=wide,
            merged_skip=merged,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
        )
        # Intentionally skip the original: it forces TRITON_ATTN globally,
        # which disables turboquant_* on every layer.

    Gemma4Config.verify_and_update_config = staticmethod(patched)


def _hf_config(model_name: str):
    """Load HF config with a single AutoConfig call (uses HF cache)."""
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


def _model_is_gemma4(hf_cfg) -> bool:
    if hf_cfg is None:
        return False
    archs = getattr(hf_cfg, "architectures", None) or []
    return any(a.startswith("Gemma4") for a in archs)


def _model_has_gdn_layers(hf_cfg) -> bool:
    """True if any layer is ``linear_attention`` (DeltaNet/GDN). Qwen3.5/3.6."""
    if hf_cfg is None:
        return False
    text_cfg = getattr(hf_cfg, "text_config", None) or hf_cfg
    layer_types = getattr(text_cfg, "layer_types", None) or []
    return "linear_attention" in layer_types


def _install_turboquant_engine_config_hook() -> None:
    """Wrap ``EngineArgs.create_engine_config`` to clear two upstream blockers.

    Verified against vLLM files:
      vllm/engine/arg_utils.py:1645-1668     (hybrid guard)
      vllm/config/model.py:1563-1572         (is_hybrid property)
      vllm/v1/worker/gpu_worker.py:380-385   (profile_cudagraph_memory gate)
      vllm/v1/worker/gpu_model_runner.py:6598 (reshape failure root)

    Fix 1 (hybrid): upstream raises NotImplementedError on any hybrid model +
    turboquant_* dtype. Mamba/DeltaNet layers never hit the skip-layer check
    (they don't construct ``Attention()``), so absolute-index boundary skips
    are harmless. We temporarily mask ``ModelConfig.is_hybrid`` to False
    inside create_engine_config only; downstream hybrid routing is
    unaffected once the wrapper returns.

    Fix 2 (Gemma-4 profiler): mixed KV cache groups break the view() in
    _reshape_kv_cache_tensors during profile_cudagraph_memory. enforce_eager
    skips that path (gpu_worker.py:382-383 gates on cudagraph_mode != NONE),
    at a ~15-25% decode throughput cost. Auto-inject for Gemma-4 only.
    """
    import structlog
    from vllm.config.model import ModelConfig
    from vllm.engine.arg_utils import EngineArgs

    logger = structlog.get_logger(__name__)

    if getattr(EngineArgs.create_engine_config, "_turboquant_wrapped", False):
        return

    orig_create = EngineArgs.create_engine_config
    orig_is_hybrid = ModelConfig.is_hybrid  # property object, not a bound method

    def patched(self, *args, **kwargs):
        cache_dtype = getattr(self, "kv_cache_dtype", "") or ""
        if not cache_dtype.startswith("turboquant"):
            return orig_create(self, *args, **kwargs)

        hf_cfg = _hf_config(self.model)
        has_mixed_kv_groups = _model_is_gemma4(hf_cfg) or _model_has_gdn_layers(
            hf_cfg
        )

        # GDN hybrid models (Qwen3.5/3.6): Mamba state + attention cache →
        # mixed KV cache groups. `_dummy_run` and kernel warm-up for Mamba
        # paths have issues beyond the page-size unification fix, so keep
        # enforce_eager=True. Gemma-4 no longer needs this: dev vLLM already
        # iterates per-group in _reshape_kv_cache_tensors, and our LCM unify
        # patch (_patch_unify_kv_cache_spec_page_size in vllm_plugin.py)
        # resolves the page-size divisibility issue. CUDA graphs are allowed
        # for Gemma-4; validate with a live run before claiming gate 2 passed.
        is_gemma4 = _model_is_gemma4(hf_cfg)
        has_gdn = _model_has_gdn_layers(hf_cfg)
        if has_gdn and not self.enforce_eager:
            logger.warning(
                "mixed_kv_forcing_enforce_eager",
                reason="GDN hybrid: Mamba warm-up issues beyond page-size fix",
                model=self.model,
                has_gdn=has_gdn,
            )
            self.enforce_eager = True

        # When our LCM-based unify produces heterogeneous group block_sizes,
        # HybridKVCacheCoordinator.__init__ at kv_cache_coordinator.py:402
        # asserts every group's block_size divides hash_block_size —
        # which doesn't hold after aggressive block-size scaling. Routing
        # to KVCacheCoordinatorNoPrefixCache (enable_caching=False) skips
        # that assertion. Applies to both Gemma-4 and GDN hybrids.
        # Unconditional write: EngineArgs.enable_prefix_caching defaults to
        # None and is auto-resolved to True later; we must pin it to False.
        if has_mixed_kv_groups and self.enable_prefix_caching is not False:
            logger.warning(
                "mixed_kv_disabling_prefix_caching",
                reason="HybridKVCacheCoordinator block_size divisibility assertion",
                model=self.model,
                previous=self.enable_prefix_caching,
            )
            self.enable_prefix_caching = False

        # Auto-pick triton for GDN prefill on hybrid DeltaNet models — FlashInfer
        # JIT-compiles 67 CUDA kernels serially and can be OOM-killed on modest
        # host RAM, blowing past the health-check timeout even when compile
        # succeeds. Respect an explicit user choice.
        if (
            _model_has_gdn_layers(hf_cfg)
            and getattr(self, "gdn_prefill_backend", None) is None
        ):
            logger.warning(
                "hybrid_gdn_forcing_triton_backend",
                reason="flashinfer GDN JIT compile can OOM nvcc; use triton",
                model=self.model,
            )
            self.gdn_prefill_backend = "triton"

        ModelConfig.is_hybrid = property(lambda _self: False)
        try:
            vllm_config = orig_create(self, *args, **kwargs)
        finally:
            ModelConfig.is_hybrid = orig_is_hybrid

        # Our is_hybrid mask suppressed try_verify_and_update_config's dispatch
        # to HybridAttentionMambaModelConfig (see config/vllm.py:1721), which
        # is normally what sets cache_config.mamba_block_size and disables
        # calculate_kv_scales. Run it here by hand on the restored config so
        # MambaSpec doesn't end up with block_size=None (which crashes
        # unify_kv_cache_spec_page_size at kv_cache_utils.py:958).
        if vllm_config.model_config.is_hybrid:
            from vllm.model_executor.models.config import (
                HybridAttentionMambaModelConfig,
            )

            HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)

        return vllm_config

    patched._turboquant_wrapped = True  # type: ignore[attr-defined]
    EngineArgs.create_engine_config = patched


def main() -> None:
    argv = sys.argv[1:]
    bit_width = _turboquant_bit_width_from_argv(argv)
    cache_dtype = _peek_kv_cache_dtype(argv)

    # Scaffolding Pydantic patch is only needed for bare "turboquant"; upstream
    # variants are already first-class vLLM dtypes.
    if cache_dtype == "turboquant":
        register_turboquant_backend(TurboQuantConfig(bit_width=bit_width))

    _install_gemma4_mixed_backend_hook()
    _install_turboquant_engine_config_hook()

    import vllm.entrypoints.openai.api_server as api_server
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="vLLM + project TurboQuant KV cache")
    parser = api_server.make_arg_parser(parser)
    _extend_kv_cache_choices(parser)

    args = parser.parse_args(argv)
    api_server.validate_parsed_serve_args(args)

    asyncio.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
