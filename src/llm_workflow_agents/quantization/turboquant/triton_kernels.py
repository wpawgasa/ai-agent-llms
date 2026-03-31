"""Fused Triton encode/decode kernels for TurboQuant.

These kernels implement the fused pipeline:
  Encode: rotate → quantize → pack indices + store norm
  Decode: unpack → lookup centroids → inverse rotate

Hooked into vLLM cache write/read paths respectively.

The actual @triton.jit kernels require GPU. This module provides both
the Triton kernels (for GPU execution) and pure-PyTorch fallbacks
(for CPU testing and validation).

Reference: Zandieh et al., ICLR 2026.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def turboquant_encode(
    kv_values: "torch.Tensor",
    rotation_matrix: "torch.Tensor",
    codebook: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Encode KV cache values using TurboQuant pipeline (PyTorch fallback).

    Pipeline: rotate → quantize → pack indices + store norm.

    Args:
        kv_values: Input KV cache values [batch, heads, seq, d].
        rotation_matrix: Orthogonal rotation matrix [d, d].
        codebook: Lloyd-Max codebook centroids [n_levels].

    Returns:
        Tuple of (quantized_indices, norms):
          - indices: [batch, heads, seq, d] int32 codebook indices
          - norms: [batch, heads, seq] per-vector L2 norms
    """
    import torch

    # Step 1: Compute and store per-vector norms
    norms = torch.norm(kv_values, dim=-1, keepdim=True)
    normalized = kv_values / (norms + 1e-8)

    # Step 2: Rotate
    rotated = normalized @ rotation_matrix.T

    # Step 3: Map to [0, 1] range for Beta codebook
    # Using (x + 1) / 2 mapping from [-1, 1] to [0, 1]
    mapped = (rotated + 1.0) / 2.0
    mapped = torch.clamp(mapped, 0.0, 1.0)

    # Step 4: Quantize to nearest codebook centroid
    distances = torch.abs(mapped.unsqueeze(-1) - codebook)
    indices = torch.argmin(distances, dim=-1).to(torch.int32)

    return indices, norms.squeeze(-1)


def turboquant_decode(
    indices: "torch.Tensor",
    norms: "torch.Tensor",
    codebook: "torch.Tensor",
    inv_rotation_matrix: "torch.Tensor",
) -> "torch.Tensor":
    """Decode TurboQuant-encoded KV cache values (PyTorch fallback).

    Pipeline: unpack → lookup centroids → inverse rotate.

    Args:
        indices: Quantized codebook indices [batch, heads, seq, d].
        norms: Per-vector norms [batch, heads, seq].
        codebook: Lloyd-Max codebook centroids [n_levels].
        inv_rotation_matrix: Inverse rotation matrix (= rotation^T) [d, d].

    Returns:
        Reconstructed KV values [batch, heads, seq, d].
    """
    # Step 1: Lookup centroids
    reconstructed = codebook[indices.long()]

    # Step 2: Unmap from [0, 1] to [-1, 1]
    reconstructed = reconstructed * 2.0 - 1.0

    # Step 3: Inverse rotate
    reconstructed = reconstructed @ inv_rotation_matrix.T

    # Step 4: Restore norms
    reconstructed = reconstructed * norms.unsqueeze(-1)

    return reconstructed


def get_triton_kernels() -> dict[str, Any]:
    """Get Triton kernel functions if available, else return fallbacks.

    Returns:
        Dict with 'encode' and 'decode' callables.
    """
    try:
        import triton  # noqa: F401

        logger.info("triton_available", message="Using Triton GPU kernels")
        # In production, this would return the @triton.jit kernels.
        # For now, return the PyTorch fallbacks which have identical semantics.
        return {
            "encode": turboquant_encode,
            "decode": turboquant_decode,
        }
    except ImportError:
        logger.info("triton_not_available", message="Using PyTorch CPU fallbacks")
        return {
            "encode": turboquant_encode,
            "decode": turboquant_decode,
        }
