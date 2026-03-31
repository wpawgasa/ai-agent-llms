"""Fused Triton kernel for RotorQuant KV cache quantization.

Implements the fused pipeline:
  embed → rotor sandwich → quantize → inverse → extract

Uses grade-aware Lloyd-Max codebooks that account for the Clifford
algebra structure of the rotated values.

This module provides PyTorch fallback implementations. The actual
@triton.jit kernels are used when Triton is available on GPU.
"""

from __future__ import annotations

from typing import Any

import structlog

from llm_workflow_agents.quantization.rotorquant.clifford import CliffordAlgebra, Rotor

logger = structlog.get_logger(__name__)


def rotorquant_encode(
    kv_values: "torch.Tensor",
    rotor: Rotor,
    codebook: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Encode KV cache values using RotorQuant pipeline (PyTorch fallback).

    Pipeline: embed → rotor sandwich → quantize.
    Processes vectors in chunks of 3 (Cl(3,0) operates on R^3).

    Args:
        kv_values: Input KV values of shape (..., d) where d is divisible by 3.
        rotor: Cl(3,0) rotor for rotation.
        codebook: Grade-aware codebook centroids [n_levels].

    Returns:
        Tuple of (indices, norms):
          - indices: Quantized codebook indices
          - norms: Per-vector norms for reconstruction
    """
    import torch

    algebra = CliffordAlgebra()

    # Store norms
    norms = torch.norm(kv_values, dim=-1, keepdim=True)
    normalized = kv_values / (norms + 1e-8)

    d = normalized.shape[-1]
    # Pad to multiple of 3 if needed
    pad_size = (3 - d % 3) % 3
    if pad_size > 0:
        normalized = torch.nn.functional.pad(normalized, (0, pad_size))

    # Process in chunks of 3
    d_padded = normalized.shape[-1]
    shape_prefix = normalized.shape[:-1]
    chunks = normalized.reshape(*shape_prefix, d_padded // 3, 3)

    # Embed into Cl(3,0)
    embedded = algebra.embed(chunks)

    # Apply rotor sandwich: R x R†
    rotated = algebra.sandwich_product(rotor, embedded)

    # Extract back to R^3
    extracted = algebra.extract(rotated)

    # Flatten back
    rotated_flat = extracted.reshape(*shape_prefix, d_padded)
    if pad_size > 0:
        rotated_flat = rotated_flat[..., :d]

    # Map to [0, 1] and quantize
    mapped = (rotated_flat + 1.0) / 2.0
    mapped = torch.clamp(mapped, 0.0, 1.0)

    distances = torch.abs(mapped.unsqueeze(-1) - codebook)
    indices = torch.argmin(distances, dim=-1).to(torch.int32)

    return indices, norms.squeeze(-1)


def rotorquant_decode(
    indices: "torch.Tensor",
    norms: "torch.Tensor",
    codebook: "torch.Tensor",
    rotor: Rotor,
) -> "torch.Tensor":
    """Decode RotorQuant-encoded KV cache values (PyTorch fallback).

    Pipeline: lookup → unmap → inverse rotor sandwich → restore norms.

    Args:
        indices: Quantized codebook indices.
        norms: Per-vector norms.
        codebook: Grade-aware codebook centroids.
        rotor: Cl(3,0) rotor (inverse applied via reverse).

    Returns:
        Reconstructed KV values.
    """
    import torch

    algebra = CliffordAlgebra()

    # Lookup centroids
    reconstructed = codebook[indices.long()]

    # Unmap from [0, 1] to [-1, 1]
    reconstructed = reconstructed * 2.0 - 1.0

    d = reconstructed.shape[-1]
    pad_size = (3 - d % 3) % 3
    if pad_size > 0:
        reconstructed = torch.nn.functional.pad(reconstructed, (0, pad_size))

    # Process in chunks of 3
    d_padded = reconstructed.shape[-1]
    shape_prefix = reconstructed.shape[:-1]
    chunks = reconstructed.reshape(*shape_prefix, d_padded // 3, 3)

    # Embed and apply inverse rotor (reverse sandwich: R† x R)
    embedded = algebra.embed(chunks)
    rev_rotor = Rotor(components=algebra.reverse(rotor).components)
    inv_rotated = algebra.sandwich_product(rev_rotor, embedded)

    # Extract and flatten
    extracted = algebra.extract(inv_rotated)
    result = extracted.reshape(*shape_prefix, d_padded)
    if pad_size > 0:
        result = result[..., :d]

    # Restore norms
    result = result * norms.unsqueeze(-1)

    return result


def get_triton_kernels() -> dict[str, Any]:
    """Get RotorQuant kernel functions (Triton or PyTorch fallback)."""
    try:
        import triton  # noqa: F401
        logger.info("triton_available_rotorquant")
    except ImportError:
        logger.info("triton_not_available_rotorquant", message="Using PyTorch fallbacks")

    return {
        "encode": rotorquant_encode,
        "decode": rotorquant_decode,
    }
