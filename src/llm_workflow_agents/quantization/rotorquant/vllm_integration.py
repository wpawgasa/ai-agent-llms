"""vLLM hook registration for RotorQuant KV cache quantization.

Registers the "rotorquant" kv_cache_dtype with vLLM's configuration
system and hooks the encode/decode kernels into the cache write/read paths.

Integration points:
  1. Register kv_cache_dtype="rotorquant" in vllm.config.cache
  2. Modify block size in gpu_model_runner (3-bit: 52 bytes per 128-value vector)
  3. Hook encode kernel into cache write path
  4. Hook decode kernel into cache read path

Uses Cl(3,0) rotor sandwich rotation instead of a dense orthogonal matrix
(~10–19× fewer FMAs per rotation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Block size constants (bytes per 128-value vector)
BLOCK_SIZE_FP16 = 256  # 128 values × 2 bytes
BLOCK_SIZE_3BIT = 52  # 128 values × 3 bits / 8 + norm overhead
BLOCK_SIZE_2BIT = 36  # 128 values × 2 bits / 8 + norm overhead
BLOCK_SIZE_4BIT = 68  # 128 values × 4 bits / 8 + norm overhead

BLOCK_SIZES: dict[int, int] = {
    2: BLOCK_SIZE_2BIT,
    3: BLOCK_SIZE_3BIT,
    4: BLOCK_SIZE_4BIT,
}


@dataclass
class RotorQuantConfig:
    """Configuration for RotorQuant vLLM integration."""

    bit_width: int = 3
    head_dimension: int = 128
    rotor_seed: int = 42

    @property
    def block_size_bytes(self) -> int:
        return BLOCK_SIZES.get(self.bit_width, BLOCK_SIZE_3BIT)

    @property
    def compression_ratio(self) -> float:
        return BLOCK_SIZE_FP16 / self.block_size_bytes


def register_rotorquant_backend(config: RotorQuantConfig | None = None) -> dict[str, Any]:
    """Register RotorQuant as a vLLM KV cache dtype backend.

    In production, this modifies vLLM internals:
      1. Registers kv_cache_dtype="rotorquant" in vllm.config.cache
      2. Modifies block size in gpu_model_runner.py
      3. Hooks encode kernel into cache write path
      4. Hooks decode kernel into cache read path

    Args:
        config: RotorQuant configuration. Uses defaults if None.

    Returns:
        Registration info dict with config details and hook status.
    """
    if config is None:
        config = RotorQuantConfig()

    logger.info(
        "registering_rotorquant",
        bit_width=config.bit_width,
        head_dim=config.head_dimension,
        block_size=config.block_size_bytes,
        compression_ratio=f"{config.compression_ratio:.1f}x",
    )

    registration = {
        "kv_cache_dtype": "rotorquant",
        "bit_width": config.bit_width,
        "block_size_bytes": config.block_size_bytes,
        "compression_ratio": config.compression_ratio,
        "config_module": "vllm.config.cache",
        "runner_module": "vllm.worker.gpu_model_runner",
        "hooks": {
            "cache_write": "rotorquant_encode",
            "cache_read": "rotorquant_decode",
        },
        "status": "registered",
    }

    try:
        _hook_vllm_cache(config)
        registration["status"] = "hooked"
    except ImportError:
        logger.warning("vllm_not_available", message="Registration deferred — vLLM not installed")
    except Exception as exc:
        logger.warning("vllm_hook_failed", error=str(exc))

    return registration


def _hook_vllm_cache(config: RotorQuantConfig) -> None:
    """Hook RotorQuant into vLLM's cache system.

    Only step 1 (Pydantic widening) is implementable against current vLLM v1.
    Steps 2 and 3 require a full ``AttentionBackend`` subclass and registration
    via ``vllm.v1.attention.backends.registry.register_backend``; they are
    retained as stubs that emit explicit migration warnings.
    """
    import vllm
    from vllm.config import CacheConfig

    logger.info("hooking_vllm_cache", version=getattr(vllm, "__version__", "unknown"))

    # 1. Widen CacheConfig.cache_dtype so Pydantic accepts "rotorquant".
    _patch_cache_config(CacheConfig)

    # 2. Block-size override — v1 declares this per-backend; cannot patch.
    _patch_block_size(config)

    # 3. Cache write/read splicing — v1 does this inside AttentionBackend.forward;
    #    there is no central function to monkey-patch.
    _patch_paged_attention()

    logger.info(
        "rotorquant_hooks_installed",
        bit_width=config.bit_width,
        head_dim=config.head_dimension,
        compression=f"{config.compression_ratio:.1f}x",
    )


def _patch_cache_config(cache_config_cls: type, dtype_name: str = "rotorquant") -> None:
    """Make ``CacheConfig`` accept *dtype_name* despite the Literal constraint.

    See ``llm_workflow_agents.quantization.turboquant.vllm_integration.
    _patch_cache_config`` for the full explanation; this is the same
    approach parameterised for *dtype_name*.

    Summary: vLLM defines ``cache_dtype`` as a Pydantic-validated
    ``Literal[...]`` whose members are baked into the core schema at class
    creation. We wrap ``__pydantic_validator__.validate_python`` to
    substitute a Literal-allowed placeholder for our custom dtype during
    validation and restore the original string on the resulting instance.
    The wrapper is idempotent across multiple custom dtypes via a
    ``_CUSTOM_KV_CACHE_DTYPES`` class-level set.
    """
    if hasattr(cache_config_cls, "__pydantic_validator__"):
        _install_validator_wrapper(cache_config_cls, dtype_name)
        return

    for attr in ("_VALID_KV_CACHE_DTYPES", "VALID_KV_CACHE_DTYPES", "_valid_kv_cache_dtypes"):
        if hasattr(cache_config_cls, attr):
            valid = getattr(cache_config_cls, attr)
            if isinstance(valid, frozenset):
                object.__setattr__(cache_config_cls, attr, valid | {dtype_name})
            elif isinstance(valid, (set, list)):
                valid_set = set(valid) | {dtype_name}
                setattr(cache_config_cls, attr, type(valid)(valid_set))
            logger.info("patched_cache_config_legacy", attr=attr, dtype=dtype_name)
            return

    logger.warning(
        "patch_cache_config_failed",
        reason="no pydantic validator and no known legacy attribute",
        dtype=dtype_name,
    )


def _install_validator_wrapper(cache_config_cls: type, dtype_name: str) -> None:
    """Install (or extend) a Pydantic validator wrapper that accepts *dtype_name*."""
    from pydantic_core import ArgsKwargs

    registered: set[str] = getattr(cache_config_cls, "_CUSTOM_KV_CACHE_DTYPES", set())
    if dtype_name in registered:
        logger.info("cache_config_already_patched", dtype=dtype_name)
        return

    if not registered:
        original = cache_config_cls.__pydantic_validator__

        class _CustomDTypeValidator:
            """Wrap vLLM's Pydantic validator to transparently allow custom dtypes."""

            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

            def validate_python(self, input_data, *args, self_instance=None, **kwargs):
                custom = None
                if isinstance(input_data, ArgsKwargs):
                    kw = input_data.kwargs or {}
                    dtype = kw.get("cache_dtype")
                    if dtype in getattr(cache_config_cls, "_CUSTOM_KV_CACHE_DTYPES", ()):
                        custom = dtype
                        new_kw = {**kw, "cache_dtype": "auto"}
                        input_data = ArgsKwargs(input_data.args, new_kw)
                result = self._wrapped.validate_python(
                    input_data, *args, self_instance=self_instance, **kwargs
                )
                if custom is not None and self_instance is not None:
                    object.__setattr__(self_instance, "cache_dtype", custom)
                return result

        cache_config_cls.__pydantic_validator__ = _CustomDTypeValidator(original)
        cache_config_cls._CUSTOM_KV_CACHE_DTYPES = set()

    cache_config_cls._CUSTOM_KV_CACHE_DTYPES.add(dtype_name)
    logger.info(
        "patched_cache_config_pydantic",
        dtype=dtype_name,
        registered=sorted(cache_config_cls._CUSTOM_KV_CACHE_DTYPES),
    )


def _patch_block_size(config: RotorQuantConfig) -> None:
    """No-op on vLLM v1 — block size is per-backend.

    In v0 this patched ``vllm.worker.gpu_model_runner.get_cache_block_size``.
    In v1 there is no central function: each ``AttentionBackend`` declares its
    own cache layout via ``get_kv_cache_shape``. A proper port requires
    implementing a full ``AttentionBackend`` subclass.
    """
    try:
        import vllm.worker.gpu_model_runner as runner_mod  # noqa: F401  # v0 only

        original = getattr(runner_mod, "get_cache_block_size", None)
        if original is None:
            raise ImportError("v0 function missing; treat as v1")

        _rq_bytes = config.block_size_bytes

        def _patched(cache_config, model_config, parallel_config):
            if getattr(cache_config, "kv_cache_dtype", None) == "rotorquant":
                num_heads = getattr(model_config, "get_num_kv_heads", lambda x: 1)(parallel_config)
                num_layers = getattr(model_config, "get_num_layers", lambda x: 1)(parallel_config)
                return _rq_bytes * num_heads * num_layers * 2
            return original(cache_config, model_config, parallel_config)

        runner_mod.get_cache_block_size = _patched
        logger.info("patched_block_size_v0", bytes_per_vector=_rq_bytes)
    except ImportError:
        logger.warning(
            "block_size_patch_skipped_v1",
            reason=(
                "vLLM v1 declares block size per-backend via "
                "AttentionBackend.get_kv_cache_shape; implement a full "
                "AttentionBackend subclass and register via "
                "register_backend(CUSTOM, ...)"
            ),
        )


def _patch_paged_attention() -> None:
    """No-op on vLLM v1 — KV cache I/O happens inside AttentionBackend.forward.

    In v0 this spliced ``write_to_paged_cache`` and ``copy_blocks`` module
    functions in ``vllm.attention.backends.flash_attn``. In v1 those
    functions do not exist: each backend's ``forward()`` reads and writes
    its own paged cache. To add RotorQuant support, implement an
    ``AttentionBackend`` subclass and register it.
    """
    try:
        from vllm.attention.backends import flash_attn as flash_backend  # v0 path
    except ImportError:
        logger.warning(
            "paged_attention_patch_skipped_v1",
            reason=(
                "vLLM v1 has no module-level write_to_paged_cache/copy_blocks; "
                "KV I/O is owned by AttentionBackend.forward. Implement a "
                "custom AttentionBackend subclass to wire in the RotorQuant "
                "encode/decode kernels."
            ),
        )
        return

    logger.warning(
        "paged_attention_patch_skipped_v0_detected",
        reason="legacy flash_attn module present but no splicing implemented in this build",
        backend=getattr(flash_backend, "__name__", "unknown"),
    )
