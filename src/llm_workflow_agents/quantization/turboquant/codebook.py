"""Lloyd-Max codebook pre-computation for TurboQuant.

Pre-computes optimal quantization codebooks for the Beta(alpha, alpha)
distribution that models the angular distribution of KV cache vectors
after orthogonal rotation. alpha = (d - 1) / 2 where d is head dimension.

Reference: Zandieh et al., ICLR 2026.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def _beta_alpha_from_head_dim(d: int) -> float:
    """Compute Beta distribution alpha parameter from head dimension."""
    return (d - 1) / 2.0


def lloyd_max_quantize(
    pdf: Any,
    support: tuple[float, float],
    n_levels: int,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute Lloyd-Max optimal quantization codebook for a given PDF.

    The Lloyd-Max algorithm iteratively refines codebook centroids to
    minimize mean squared quantization error for a given distribution.

    Args:
        pdf: Probability density function (callable).
        support: (low, high) support interval of the distribution.
        n_levels: Number of quantization levels (2^bits).
        max_iter: Maximum iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        Sorted array of codebook centroids of shape (n_levels,).
    """
    from scipy import integrate

    # Initialize centroids uniformly across the support
    low, high = support
    centroids = np.linspace(low, high, n_levels + 2)[1:-1]  # Interior points

    for iteration in range(max_iter):
        # Compute decision boundaries (midpoints between centroids)
        boundaries = np.concatenate([[low], (centroids[:-1] + centroids[1:]) / 2, [high]])

        # Update centroids: centroid_i = E[X | boundary_{i} <= X < boundary_{i+1}]
        new_centroids = np.zeros_like(centroids)
        for i in range(n_levels):
            b_lo, b_hi = boundaries[i], boundaries[i + 1]

            # Numerator: integral of x * pdf(x) over [b_lo, b_hi]
            numerator, _ = integrate.quad(lambda x: x * pdf(x), b_lo, b_hi)
            # Denominator: integral of pdf(x) over [b_lo, b_hi]
            denominator, _ = integrate.quad(pdf, b_lo, b_hi)

            if denominator > 1e-15:
                new_centroids[i] = numerator / denominator
            else:
                new_centroids[i] = (b_lo + b_hi) / 2

        # Check convergence
        max_shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids

        if max_shift < tol:
            logger.debug("lloyd_max_converged", iteration=iteration, max_shift=max_shift)
            break

    return np.sort(centroids)


def precompute_codebooks(
    head_dimensions: list[int] | None = None,
    bit_widths: list[int] | None = None,
    output_dir: Path | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Pre-compute Lloyd-Max codebooks for Beta(alpha, alpha) distributions.

    This is a one-time offline computation. For each (head_dim, bits) pair,
    computes the optimal 2^bits codebook centroids for the Beta distribution
    that models the angular distribution after rotation.

    Args:
        head_dimensions: Head dimensions to compute for (default: [128, 256]).
        bit_widths: Bit widths to compute for (default: [2, 3, 4]).
        output_dir: Optional directory to save codebooks as .npy files.

    Returns:
        Dict mapping (head_dim, bits) to codebook ndarray of shape (2^bits,).
    """
    from scipy.stats import beta as beta_dist

    if head_dimensions is None:
        head_dimensions = [128, 256]
    if bit_widths is None:
        bit_widths = [2, 3, 4]

    codebooks: dict[tuple[int, int], np.ndarray] = {}

    for d in head_dimensions:
        alpha = _beta_alpha_from_head_dim(d)

        for bits in bit_widths:
            n_levels = 2**bits
            logger.info(
                "computing_codebook",
                head_dim=d,
                bits=bits,
                n_levels=n_levels,
                alpha=alpha,
            )

            dist = beta_dist(alpha, alpha)
            codebook = lloyd_max_quantize(
                pdf=dist.pdf,
                support=(0.0, 1.0),
                n_levels=n_levels,
            )

            codebooks[(d, bits)] = codebook

            # Optionally save to disk
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / f"codebook_d{d}_b{bits}.npy"
                np.save(path, codebook)
                logger.info("codebook_saved", path=str(path))

    return codebooks


def load_codebook(path: Path) -> np.ndarray:
    """Load a pre-computed codebook from disk."""
    return np.load(path)


def quantize_to_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Quantize values to nearest codebook centroids.

    Args:
        values: Input values to quantize.
        codebook: Sorted codebook centroids.

    Returns:
        Array of indices into the codebook (same shape as values).
    """
    # Compute distances to all centroids and find nearest
    distances = np.abs(values[..., np.newaxis] - codebook[np.newaxis, :])
    return np.argmin(distances, axis=-1).astype(np.int32)


def dequantize_from_codebook(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Reconstruct values from codebook indices.

    Args:
        indices: Codebook indices.
        codebook: Codebook centroids.

    Returns:
        Reconstructed values.
    """
    return codebook[indices]
