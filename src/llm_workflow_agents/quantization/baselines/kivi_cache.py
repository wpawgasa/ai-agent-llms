"""KIVI asymmetric KV cache quantization wrapper.

Implements KIVI (ICML 2024): asymmetric quantization with per-channel
quantization for Keys and per-token quantization for Values.
No calibration required.

Reference: Liu et al., ICML 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KIVIConfig:
    """Configuration for KIVI quantization."""

    key_bits: int = 2
    value_bits: int = 2
    key_quantization: str = "per_channel"  # Per-channel for keys
    value_quantization: str = "per_token"  # Per-token for values
    group_size: int = 128


@dataclass
class KIVIQuantizedCache:
    """Container for KIVI-quantized KV cache."""

    key_quantized: "torch.Tensor"
    key_scales: "torch.Tensor"
    key_zeros: "torch.Tensor"
    value_quantized: "torch.Tensor"
    value_scales: "torch.Tensor"
    value_zeros: "torch.Tensor"
    config: KIVIConfig


def quantize_per_channel(
    tensor: "torch.Tensor", bits: int
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Quantize a tensor with per-channel (per-head-dim) scale and zero point.

    Args:
        tensor: Input tensor of shape (..., d).
        bits: Number of quantization bits.

    Returns:
        (quantized, scales, zeros) tuple.
    """
    import torch

    n_levels = 2**bits
    # Compute per-channel min/max
    ch_min = tensor.min(dim=-2, keepdim=True).values
    ch_max = tensor.max(dim=-2, keepdim=True).values

    scales = (ch_max - ch_min) / (n_levels - 1)
    scales = torch.clamp(scales, min=1e-8)
    zeros = ch_min

    quantized = torch.round((tensor - zeros) / scales).clamp(0, n_levels - 1).to(torch.int8)
    return quantized, scales, zeros


def quantize_per_token(
    tensor: "torch.Tensor", bits: int
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Quantize a tensor with per-token (per-sequence-position) scale and zero point.

    Args:
        tensor: Input tensor of shape (..., seq, d).
        bits: Number of quantization bits.

    Returns:
        (quantized, scales, zeros) tuple.
    """
    import torch

    n_levels = 2**bits
    # Compute per-token min/max
    tok_min = tensor.min(dim=-1, keepdim=True).values
    tok_max = tensor.max(dim=-1, keepdim=True).values

    scales = (tok_max - tok_min) / (n_levels - 1)
    scales = torch.clamp(scales, min=1e-8)
    zeros = tok_min

    quantized = torch.round((tensor - zeros) / scales).clamp(0, n_levels - 1).to(torch.int8)
    return quantized, scales, zeros


def dequantize(
    quantized: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
) -> "torch.Tensor":
    """Dequantize a tensor using scales and zero points."""
    return quantized.float() * scales + zeros


def kivi_quantize(
    keys: "torch.Tensor",
    values: "torch.Tensor",
    config: KIVIConfig | None = None,
) -> KIVIQuantizedCache:
    """Quantize KV cache using KIVI asymmetric quantization.

    Keys use per-channel quantization, Values use per-token quantization.

    Args:
        keys: Key tensor of shape [batch, heads, seq, d].
        values: Value tensor of shape [batch, heads, seq, d].
        config: KIVI configuration.

    Returns:
        KIVIQuantizedCache with quantized tensors and metadata.
    """
    if config is None:
        config = KIVIConfig()

    logger.debug("kivi_quantize", key_bits=config.key_bits, value_bits=config.value_bits)

    k_q, k_s, k_z = quantize_per_channel(keys, config.key_bits)
    v_q, v_s, v_z = quantize_per_token(values, config.value_bits)

    return KIVIQuantizedCache(
        key_quantized=k_q,
        key_scales=k_s,
        key_zeros=k_z,
        value_quantized=v_q,
        value_scales=v_s,
        value_zeros=v_z,
        config=config,
    )


def kivi_dequantize(cache: KIVIQuantizedCache) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Dequantize a KIVI-quantized KV cache.

    Args:
        cache: KIVI quantized cache.

    Returns:
        (keys, values) tuple of dequantized tensors.
    """
    keys = dequantize(cache.key_quantized, cache.key_scales, cache.key_zeros)
    values = dequantize(cache.value_quantized, cache.value_scales, cache.value_zeros)
    return keys, values
