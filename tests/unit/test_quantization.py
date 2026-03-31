"""Tests for quantization module — codebooks, rotation, encode/decode, baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from llm_workflow_agents.quantization.turboquant.codebook import (
    _beta_alpha_from_head_dim,
    dequantize_from_codebook,
    lloyd_max_quantize,
    precompute_codebooks,
    quantize_to_codebook,
)
from llm_workflow_agents.quantization.turboquant.rotation import (
    generate_rotation_matrix,
    inverse_rotate_vectors,
    rotate_vectors,
    verify_orthogonality,
)
from llm_workflow_agents.quantization.turboquant.triton_kernels import (
    turboquant_decode,
    turboquant_encode,
)
from llm_workflow_agents.quantization.turboquant.vllm_integration import (
    BLOCK_SIZES,
    TurboQuantConfig,
    register_turboquant_backend,
)
from llm_workflow_agents.quantization.rotorquant.clifford import (
    CliffordAlgebra,
    N_COMPONENTS,
    Rotor,
)
from llm_workflow_agents.quantization.rotorquant.rotor_kernels import (
    rotorquant_decode,
    rotorquant_encode,
)
from llm_workflow_agents.quantization.baselines.kivi_cache import (
    KIVIConfig,
    kivi_dequantize,
    kivi_quantize,
)
from llm_workflow_agents.quantization.baselines.kvquant_calibrate import (
    KVQuantConfig,
    NUQCodebook,
    calibrate,
    compute_nuq_codebook,
    detect_outlier_channels,
)


# ============================================================
# TurboQuant Codebook Tests
# ============================================================


class TestBetaAlpha:
    def test_d128(self) -> None:
        assert _beta_alpha_from_head_dim(128) == 63.5

    def test_d256(self) -> None:
        assert _beta_alpha_from_head_dim(256) == 127.5


class TestLloydMaxQuantize:
    def test_uniform_distribution(self) -> None:
        codebook = lloyd_max_quantize(
            pdf=lambda x: 1.0,  # Uniform
            support=(0.0, 1.0),
            n_levels=4,
        )
        assert len(codebook) == 4
        assert codebook[0] > 0.0
        assert codebook[-1] < 1.0
        # Uniform codebook should be roughly evenly spaced
        diffs = np.diff(codebook)
        assert np.std(diffs) < 0.1

    def test_codebook_sorted(self) -> None:
        codebook = lloyd_max_quantize(
            pdf=lambda x: 1.0,
            support=(0.0, 1.0),
            n_levels=8,
        )
        assert np.all(np.diff(codebook) > 0)


class TestPrecomputeCodebooks:
    def test_default_dimensions(self, tmp_path: Path) -> None:
        codebooks = precompute_codebooks(
            head_dimensions=[128],
            bit_widths=[2],
            output_dir=tmp_path,
        )
        assert (128, 2) in codebooks
        assert len(codebooks[(128, 2)]) == 4  # 2^2
        assert (tmp_path / "codebook_d128_b2.npy").exists()

    def test_multiple_configs(self) -> None:
        codebooks = precompute_codebooks(
            head_dimensions=[128],
            bit_widths=[2, 3],
        )
        assert len(codebooks) == 2
        assert len(codebooks[(128, 2)]) == 4
        assert len(codebooks[(128, 3)]) == 8


class TestQuantizeDequantize:
    def test_roundtrip(self) -> None:
        codebook = np.array([0.1, 0.3, 0.6, 0.9])
        values = np.array([0.12, 0.58, 0.85])
        indices = quantize_to_codebook(values, codebook)
        reconstructed = dequantize_from_codebook(indices, codebook)
        # Each value should map to nearest centroid
        assert indices[0] == 0  # 0.12 → 0.1
        assert indices[1] == 2  # 0.58 → 0.6
        assert indices[2] == 3  # 0.85 → 0.9
        assert reconstructed[0] == pytest.approx(0.1)


# ============================================================
# TurboQuant Rotation Tests
# ============================================================


class TestRotationMatrix:
    def test_orthogonality(self) -> None:
        q = generate_rotation_matrix(64)
        assert verify_orthogonality(q)

    def test_deterministic(self) -> None:
        q1 = generate_rotation_matrix(32, seed=42)
        q2 = generate_rotation_matrix(32, seed=42)
        assert torch.allclose(q1, q2)

    def test_different_seeds(self) -> None:
        q1 = generate_rotation_matrix(16, seed=1)
        q2 = generate_rotation_matrix(16, seed=2)
        assert not torch.allclose(q1, q2)

    def test_shape(self) -> None:
        q = generate_rotation_matrix(128)
        assert q.shape == (128, 128)


class TestRotateVectors:
    def test_roundtrip(self) -> None:
        q = generate_rotation_matrix(16)
        v = torch.randn(4, 16)
        rotated = rotate_vectors(v, q)
        recovered = inverse_rotate_vectors(rotated, q)
        assert torch.allclose(v, recovered, atol=1e-5)

    def test_norm_preservation(self) -> None:
        q = generate_rotation_matrix(32)
        v = torch.randn(8, 32)
        rotated = rotate_vectors(v, q)
        assert torch.allclose(
            torch.norm(v, dim=-1),
            torch.norm(rotated, dim=-1),
            atol=1e-5,
        )


# ============================================================
# TurboQuant Encode/Decode Tests
# ============================================================


class TestTurboQuantEncodeDecode:
    def test_encode_output_shapes(self) -> None:
        kv = torch.randn(2, 4, 8, 16)  # [batch, heads, seq, d]
        rotation = generate_rotation_matrix(16)
        codebook = torch.linspace(0, 1, 8)

        indices, norms = turboquant_encode(kv, rotation, codebook)
        assert indices.shape == (2, 4, 8, 16)
        assert norms.shape == (2, 4, 8)
        assert indices.dtype == torch.int32

    def test_encode_decode_roundtrip(self) -> None:
        torch.manual_seed(42)
        kv = torch.randn(1, 2, 4, 16)
        rotation = generate_rotation_matrix(16)
        codebook = torch.linspace(0, 1, 16)  # 4-bit

        indices, norms = turboquant_encode(kv, rotation, codebook)
        reconstructed = turboquant_decode(indices, norms, codebook, rotation.T)

        # Quantization introduces error, but should be bounded
        error = torch.mean((kv - reconstructed) ** 2).item()
        assert error < 0.05  # Tight bound for 4-bit (16 levels) 16-dim quantization


# ============================================================
# TurboQuant vLLM Integration Tests
# ============================================================


class TestTurboQuantConfig:
    def test_default_config(self) -> None:
        cfg = TurboQuantConfig()
        assert cfg.bit_width == 3
        assert cfg.block_size_bytes == 52
        assert cfg.compression_ratio > 1.0

    def test_block_sizes(self) -> None:
        assert BLOCK_SIZES[2] == 36
        assert BLOCK_SIZES[3] == 52
        assert BLOCK_SIZES[4] == 68

    def test_register_backend(self) -> None:
        result = register_turboquant_backend()
        assert result["kv_cache_dtype"] == "turboquant"
        assert result["status"] in ("registered", "hooked")


# ============================================================
# RotorQuant Clifford Algebra Tests
# ============================================================


class TestCliffordAlgebra:
    def test_embed_extract_roundtrip(self) -> None:
        algebra = CliffordAlgebra()
        v = torch.tensor([1.0, 2.0, 3.0])
        mv = algebra.embed(v)
        assert mv.shape == (N_COMPONENTS,)
        recovered = algebra.extract(mv)
        assert torch.allclose(v, recovered)

    def test_embed_batch(self) -> None:
        algebra = CliffordAlgebra()
        v = torch.randn(5, 3)
        mv = algebra.embed(v)
        assert mv.shape == (5, N_COMPONENTS)

    def test_rotor_from_params(self) -> None:
        algebra = CliffordAlgebra()
        params = torch.tensor([1.0, 0.0, 0.0, 0.0])
        rotor = algebra.rotor_from_params(params)
        assert rotor.components.shape == (N_COMPONENTS,)
        # Identity rotor: scalar=1, bivectors=0
        assert rotor.scalar_part.item() == pytest.approx(1.0, abs=1e-5)

    def test_identity_sandwich_preserves_vector(self) -> None:
        algebra = CliffordAlgebra()
        params = torch.tensor([1.0, 0.0, 0.0, 0.0])
        rotor = algebra.rotor_from_params(params)
        v = torch.tensor([1.0, 2.0, 3.0])
        mv = algebra.embed(v)
        rotated = algebra.sandwich_product(rotor, mv)
        result = algebra.extract(rotated)
        assert torch.allclose(v, result, atol=1e-5)

    def test_rotation_preserves_norm(self) -> None:
        algebra = CliffordAlgebra()
        params = torch.tensor([0.5, 0.5, 0.3, 0.1])
        rotor = algebra.rotor_from_params(params)
        v = torch.tensor([3.0, 4.0, 0.0])
        mv = algebra.embed(v)
        rotated_mv = algebra.sandwich_product(rotor, mv)
        result = algebra.extract(rotated_mv)
        assert torch.norm(v).item() == pytest.approx(torch.norm(result).item(), abs=1e-4)


# ============================================================
# RotorQuant Encode/Decode Tests
# ============================================================


class TestRotorQuantEncodeDecode:
    def test_encode_output_shapes(self) -> None:
        kv = torch.randn(1, 2, 4, 6)  # d=6, divisible by 3
        algebra = CliffordAlgebra()
        params = torch.tensor([1.0, 0.0, 0.0, 0.0])
        rotor = algebra.rotor_from_params(params)
        codebook = torch.linspace(0, 1, 16)

        indices, norms = rotorquant_encode(kv, rotor, codebook)
        assert indices.shape == (1, 2, 4, 6)
        assert norms.shape == (1, 2, 4)

    def test_identity_rotor_roundtrip(self) -> None:
        torch.manual_seed(42)
        kv = torch.randn(1, 1, 2, 6)
        algebra = CliffordAlgebra()
        params = torch.tensor([1.0, 0.0, 0.0, 0.0])
        rotor = algebra.rotor_from_params(params)
        codebook = torch.linspace(0, 1, 256)  # 8-bit for low error

        indices, norms = rotorquant_encode(kv, rotor, codebook)
        reconstructed = rotorquant_decode(indices, norms, codebook, rotor)

        error = torch.mean((kv - reconstructed) ** 2).item()
        assert error < 0.5


# ============================================================
# KIVI Baseline Tests
# ============================================================


class TestKIVI:
    def test_quantize_dequantize(self) -> None:
        keys = torch.randn(1, 4, 16, 64)
        values = torch.randn(1, 4, 16, 64)

        cache = kivi_quantize(keys, values)
        k_recon, v_recon = kivi_dequantize(cache)

        # Reconstruction error should be bounded
        k_error = torch.mean((keys - k_recon) ** 2).item()
        v_error = torch.mean((values - v_recon) ** 2).item()
        assert k_error < 1.0
        assert v_error < 1.0

    def test_custom_config(self) -> None:
        config = KIVIConfig(key_bits=4, value_bits=4)
        keys = torch.randn(1, 2, 8, 32)
        values = torch.randn(1, 2, 8, 32)

        cache = kivi_quantize(keys, values, config)
        assert cache.config.key_bits == 4

    def test_2bit_quantization(self) -> None:
        keys = torch.randn(1, 2, 4, 16)
        values = torch.randn(1, 2, 4, 16)

        config = KIVIConfig(key_bits=2, value_bits=2)
        cache = kivi_quantize(keys, values, config)

        # 2-bit should have values in [0, 3]
        assert cache.key_quantized.max() <= 3
        assert cache.value_quantized.max() <= 3


# ============================================================
# KVQuant Baseline Tests
# ============================================================


class TestKVQuant:
    def test_nuq_codebook(self) -> None:
        data = np.random.randn(1000)
        codebook = compute_nuq_codebook(data, n_clusters=8)
        assert len(codebook.centroids) == 8
        assert codebook.bits == 3

    def test_nuq_quantize_dequantize(self) -> None:
        centroids = np.array([0.1, 0.3, 0.5, 0.7])
        codebook = NUQCodebook(centroids=centroids, bits=2)
        values = np.array([0.12, 0.45, 0.68])
        indices = codebook.quantize(values)
        recon = codebook.dequantize(indices)
        assert indices[0] == 0
        assert recon[0] == pytest.approx(0.1)

    def test_detect_outliers(self) -> None:
        data = np.random.randn(100, 10)
        data[:, 5] = np.random.randn(100) * 100  # Make channel 5 an outlier
        outliers = detect_outlier_channels(data, threshold=3.0)
        assert 5 in outliers

    def test_calibrate(self) -> None:
        data = np.random.randn(100, 16)
        result = calibrate(data, KVQuantConfig(bits=2))
        assert len(result.codebooks) > 0
        assert result.channel_stats["n_channels"] == 16

    def test_calibrate_1d(self) -> None:
        data = np.random.randn(100)
        result = calibrate(data, KVQuantConfig(bits=2))
        assert len(result.codebooks) == 1


# ============================================================
# Module Import Tests
# ============================================================


class TestModuleImports:
    def test_import_turboquant(self) -> None:
        from llm_workflow_agents.quantization import turboquant
        assert hasattr(turboquant, "precompute_codebooks")
        assert hasattr(turboquant, "generate_rotation_matrix")
        assert hasattr(turboquant, "turboquant_encode")

    def test_import_rotorquant(self) -> None:
        from llm_workflow_agents.quantization import rotorquant
        assert hasattr(rotorquant, "CliffordAlgebra")
        assert hasattr(rotorquant, "rotorquant_encode")

    def test_import_baselines(self) -> None:
        from llm_workflow_agents.quantization import baselines
        assert hasattr(baselines, "kivi_quantize")
        assert hasattr(baselines, "calibrate")
