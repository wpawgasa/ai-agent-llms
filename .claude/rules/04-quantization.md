# Quantization Module

## Overview
`quantization/` implements Phase 3: KV cache quantization methods benchmarked against all 3 fine-tuned models (plus their pre-trained baselines) across 6 methods.

## Methods
| Method | Config | Notes |
|--------|--------|-------|
| FP8 E4M3 | fp8.yaml | Native vLLM support |
| KIVI 2/4-bit | kivi.yaml | Asymmetric, no calibration |
| KVQuant | kvquant.yaml | NUQ codebooks, calibration required |
| AWQ-INT4 + FP8 KV | awq_fp8.yaml | Weight + KV quantization |
| TurboQuant | turboquant.yaml | Beta codebook + rotation + QJL residual |
| RotorQuant | rotorquant.yaml | Cl(3,0) rotor sandwich rotation |

## TurboQuant Pipeline (`quantization/turboquant/`)

### codebook.py — Lloyd-Max Codebook Pre-computation
```python
def precompute_codebooks(
    head_dimensions: list[int] = [128, 256],
    bit_widths: list[int] = [2, 3, 4],
    output_dir: Path = Path("quantization/turboquant/codebooks"),
) -> dict[tuple[int, int], np.ndarray]
```
- Pre-compute Lloyd-Max codebooks for Beta(α, α), α = (d−1)/2
- One-time offline; outputs codebooks[(d, bits)] → np.ndarray of shape (2^bits,)

### rotation.py — Orthogonal Rotation Matrix
```python
def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor
```
- QR decomposition of Gaussian random matrix → orthogonal Π (d×d)
- Seed-deterministic per model

### triton_kernels.py — Fused Encode/Decode Kernels
- `turboquant_encode_kernel`: rotate → quantize → pack indices + store norm + QJL residual (vLLM cache write hook)
- `turboquant_decode_kernel`: unpack → lookup centroids → inverse rotate (vLLM cache read hook)

### vllm_integration.py — vLLM Hook Registration
1. Register `kv_cache_dtype="turboquant"` in `vllm.config.cache`
2. Hook encode kernel into PagedAttention cache write path
3. Hook decode kernel into flash_attn backend cache read path
4. Modify block size: 3-bit = 52 bytes per 128-value vector

## RotorQuant Pipeline (`quantization/rotorquant/`)

### clifford.py — Cl(3,0) Geometric Algebra Primitives
- `embed(v)`: R^d → Cl(3,0) multivector
- `rotor_sandwich(R, x)`: R x R† (~100 FMAs for d=128 vs 16,384 for dense rotation), 10–19× faster
- `rotor_from_params(params)`: Construct rotor from learnable parameters

### rotor_kernels.py — Fused Triton Kernel
- `rotorquant_fused_kernel`: embed → rotor sandwich → quantize → inverse → extract
- Grade-aware Lloyd-Max codebooks for Clifford algebra structure

## Baselines (`quantization/baselines/`)

### kivi_cache.py
- KIVI (ICML 2024): asymmetric per-channel K, per-token V. No calibration.

### kvquant_calibrate.py
- KVQuant (NeurIPS 2024): pre-RoPE + NUQ codebooks + dense-sparse decomposition
- Requires per-model calibration pass

## Benchmark Config
Quality tasks: wikitext2_ppl, c4_ppl, longbench_15task, needle_in_haystack, tool_call_f1
Performance metrics: peak_vram_gb, kv_cache_size_gb, throughput (prefill/decode tok/s), TTFT/TPOT/ITL p50/p95/p99, max_concurrent_batch_4096ctx
Runs: 5 repetitions, 500 prompts per run

### Expected Concurrency (4096 ctx on H100 80GB)
| Method | Concurrent Sessions |
|--------|-------------------|
| BF16 | ~175 |
| FP8 | ~350 |
| TQ 3-bit | ~925 |

## Checklist
- [x] Implement codebook.py with Lloyd-Max optimization (scipy)
- [x] Implement rotation.py with QR decomposition
- [x] Implement turboquant_encode_kernel (Triton) — include QJL residual
- [x] Implement turboquant_decode_kernel (Triton)
- [x] Implement vllm_integration.py hook registration
- [x] Implement clifford.py Cl(3,0) algebra primitives
- [x] Implement rotor_kernels.py fused Triton kernel
- [x] Implement kivi_cache.py baseline wrapper
- [x] Implement kvquant_calibrate.py with NUQ codebooks
- [x] Write test_triton_kernels.py: encode→decode approx identity
- [x] Verify TurboQuant community fork compatibility (0xSero/turboquant)
