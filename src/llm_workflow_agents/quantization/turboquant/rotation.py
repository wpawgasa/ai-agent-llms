"""Orthogonal rotation matrix generation for TurboQuant.

Generates a random orthogonal matrix via QR decomposition of a Gaussian
random matrix. The rotation decorrelates KV cache vector components
before quantization, improving codebook efficiency.

Reference: Zandieh et al., ICLR 2026.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def generate_rotation_matrix(d: int, seed: int = 42) -> "torch.Tensor":
    """Generate a random orthogonal rotation matrix via QR decomposition.

    Generates a d×d Gaussian random matrix and computes its QR
    decomposition to obtain an orthogonal matrix Q. The result is
    seed-deterministic for reproducibility.

    Args:
        d: Head dimension (matrix will be d × d).
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (d, d) such that Q^T Q = I.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(d, d, generator=generator)
    q, r = torch.linalg.qr(gaussian)

    # Ensure deterministic sign convention (positive diagonal of R)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs.unsqueeze(0)

    logger.debug("rotation_matrix_generated", dim=d, seed=seed)
    return q


def verify_orthogonality(matrix: "torch.Tensor", tol: float = 1e-5) -> bool:
    """Verify that a matrix is orthogonal (Q^T Q ≈ I).

    Args:
        matrix: Matrix to verify.
        tol: Tolerance for deviation from identity.

    Returns:
        True if the matrix is orthogonal within tolerance.
    """
    import torch

    d = matrix.shape[0]
    product = matrix.T @ matrix
    identity = torch.eye(d, dtype=matrix.dtype, device=matrix.device)
    max_deviation = torch.max(torch.abs(product - identity)).item()
    return max_deviation < tol


def rotate_vectors(vectors: "torch.Tensor", rotation: "torch.Tensor") -> "torch.Tensor":
    """Apply rotation matrix to a batch of vectors.

    Args:
        vectors: Input vectors of shape (..., d).
        rotation: Orthogonal matrix of shape (d, d).

    Returns:
        Rotated vectors of shape (..., d).
    """
    return vectors @ rotation.T


def inverse_rotate_vectors(vectors: "torch.Tensor", rotation: "torch.Tensor") -> "torch.Tensor":
    """Apply inverse rotation (transpose) to a batch of vectors.

    For orthogonal matrices, the inverse equals the transpose.

    Args:
        vectors: Rotated vectors of shape (..., d).
        rotation: Orthogonal matrix of shape (d, d).

    Returns:
        Original-space vectors of shape (..., d).
    """
    return vectors @ rotation
