"""KVQuant calibration and non-uniform quantization (NUQ) codebook generation.

Implements KVQuant (NeurIPS 2024) with:
  - Pre-RoPE quantization for better key compression
  - Non-uniform quantization (NUQ) codebooks via k-means
  - Dense-sparse decomposition for outlier handling
  - Per-model calibration pass required

Reference: Hooper et al., NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KVQuantConfig:
    """Configuration for KVQuant calibration."""

    bits: int = 4
    pre_rope: bool = True
    dense_sparse: bool = True
    outlier_threshold: float = 6.0  # Standard deviations for outlier detection
    calibration_samples: int = 128
    n_clusters: int | None = None  # Defaults to 2^bits


@dataclass
class NUQCodebook:
    """Non-uniform quantization codebook from k-means clustering."""

    centroids: np.ndarray  # Shape: (n_clusters,)
    bits: int
    channel_idx: int | None = None  # Per-channel codebook index

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize values to nearest centroid index."""
        distances = np.abs(values[..., np.newaxis] - self.centroids)
        return np.argmin(distances, axis=-1).astype(np.int32)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Reconstruct values from centroid indices."""
        return self.centroids[indices]


@dataclass
class KVQuantCalibrationResult:
    """Result of KVQuant calibration for a model."""

    codebooks: list[NUQCodebook] = field(default_factory=list)
    outlier_channels: list[int] = field(default_factory=list)
    channel_stats: dict[str, Any] = field(default_factory=dict)
    config: KVQuantConfig = field(default_factory=KVQuantConfig)


def compute_nuq_codebook(
    calibration_values: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
) -> NUQCodebook:
    """Compute a non-uniform quantization codebook via k-means.

    Args:
        calibration_values: 1D array of calibration values.
        n_clusters: Number of codebook centroids (2^bits).
        max_iter: Maximum k-means iterations.

    Returns:
        NUQCodebook with optimized centroids.
    """
    from scipy.cluster.vq import kmeans

    # Flatten and remove NaN/Inf
    flat = calibration_values.flatten().astype(np.float64)
    flat = flat[np.isfinite(flat)]

    if len(flat) < n_clusters:
        # Fallback to uniform quantization
        centroids = np.linspace(flat.min(), flat.max(), n_clusters)
    else:
        centroids, _ = kmeans(flat, n_clusters, iter=max_iter)
        centroids = np.sort(centroids)

    bits = int(np.log2(n_clusters))
    return NUQCodebook(centroids=centroids, bits=bits)


def detect_outlier_channels(
    calibration_values: np.ndarray,
    threshold: float = 6.0,
) -> list[int]:
    """Detect outlier channels for dense-sparse decomposition.

    A channel is an outlier if its values exceed the threshold number
    of standard deviations from the mean.

    Args:
        calibration_values: Array of shape (samples, channels).
        threshold: Number of standard deviations for outlier detection.

    Returns:
        List of outlier channel indices.
    """
    if calibration_values.ndim < 2:
        return []

    channel_max = np.max(np.abs(calibration_values), axis=0)
    overall_std = np.std(calibration_values)
    overall_mean = np.mean(np.abs(calibration_values))

    if overall_std < 1e-8:
        return []

    outlier_mask = channel_max > overall_mean + threshold * overall_std
    return list(np.where(outlier_mask)[0])


def calibrate(
    calibration_data: np.ndarray,
    config: KVQuantConfig | None = None,
) -> KVQuantCalibrationResult:
    """Run KVQuant calibration to produce NUQ codebooks.

    Args:
        calibration_data: Calibration tensor of shape (samples, channels).
        config: KVQuant configuration.

    Returns:
        KVQuantCalibrationResult with codebooks and outlier info.
    """
    if config is None:
        config = KVQuantConfig()

    n_clusters = config.n_clusters or (2**config.bits)

    logger.info(
        "kvquant_calibrating",
        bits=config.bits,
        n_clusters=n_clusters,
        samples=calibration_data.shape[0],
        pre_rope=config.pre_rope,
    )

    result = KVQuantCalibrationResult(config=config)

    # Detect outlier channels for dense-sparse decomposition
    if config.dense_sparse:
        result.outlier_channels = detect_outlier_channels(
            calibration_data, config.outlier_threshold
        )
        logger.info("outlier_channels_detected", count=len(result.outlier_channels))

    # Compute per-channel NUQ codebooks
    n_channels = calibration_data.shape[-1] if calibration_data.ndim > 1 else 1

    if calibration_data.ndim == 1:
        codebook = compute_nuq_codebook(calibration_data, n_clusters)
        codebook.channel_idx = 0
        result.codebooks.append(codebook)
    else:
        for ch in range(n_channels):
            if ch in result.outlier_channels:
                # Outlier channels are kept in full precision (dense-sparse)
                continue

            ch_data = calibration_data[:, ch]
            codebook = compute_nuq_codebook(ch_data, n_clusters)
            codebook.channel_idx = ch
            result.codebooks.append(codebook)

    # Compute channel statistics
    result.channel_stats = {
        "n_channels": n_channels,
        "n_codebooks": len(result.codebooks),
        "n_outlier_channels": len(result.outlier_channels),
        "compression_ratio": _estimate_compression(config, n_channels, result.outlier_channels),
    }

    logger.info("calibration_complete", **result.channel_stats)
    return result


def _estimate_compression(
    config: KVQuantConfig,
    n_channels: int,
    outlier_channels: list[int],
) -> float:
    """Estimate compression ratio from quantization config."""
    n_quantized = n_channels - len(outlier_channels)
    # Quantized channels: bits per value, outlier channels: 16 bits (FP16)
    total_bits = n_quantized * config.bits + len(outlier_channels) * 16
    original_bits = n_channels * 16
    return original_bits / total_bits if total_bits > 0 else 1.0


def save_calibration(result: KVQuantCalibrationResult, path: Path) -> None:
    """Save calibration result to disk."""
    path.mkdir(parents=True, exist_ok=True)

    for codebook in result.codebooks:
        ch_idx = codebook.channel_idx or 0
        np.save(path / f"codebook_ch{ch_idx}.npy", codebook.centroids)

    np.save(path / "outlier_channels.npy", np.array(result.outlier_channels))
    logger.info("calibration_saved", path=str(path))
