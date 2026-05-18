# Concurrency Benchmark Summary

**Date:** 2026-05-12
**Hardware:** Single NVIDIA H100 SXM 80GB
**Workload:** input 512–2048 tok (uniform), output 128 tok, 2048-context batch
**Degradation policy:** `ttft_multiplier = 32.0`, `max_failure_rate = 0.01`
**Concurrency sweep:** 1, 8, 16, 32, 64, 128, 256 (1024 also tested for the remote frontier baseline)

## Headline numbers (best output-throughput point per run)

| Model | Engine | KV | Spec. decoding | Max sustainable | Best tok/s | Best goodput req/s | TTFT p95 @ peak | TPOT p95 @ peak | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B-FP8** (MoE, 3B active) | vllm | fp8 | — | 256 | **1792.8** | **14.01** | 627 ms | 15.1 ms | 72.1 GB |
| Qwen3.6-35B-A3B-FP8 (MoE, 3B active) | sglang | fp8 | — | 256 | 1635.6 | 12.78 | 906 ms | 16.2 ms | 79.1 GB |
| gemma-4-26B-A4B-FP8-Dynamic (MoE, 4B active) | sglang | fp8 | — | 256 | 1744.3 | 13.63 | 840 ms | 14.7 ms | 70.6 GB |
| gemma-4-26B-A4B-FP8-Dynamic (MoE, 4B active) | vllm | fp8 | — | 256 | 1433.3 | 11.20 | 686 ms | 19.7 ms | 72.4 GB |
| Qwen3.6-27B-FP8 (dense) | vllm | fp8 | — | 256 | 650.3 | 5.08 | 2308 ms | 41.8 ms | 77.3 GB |
| Qwen3.6-27B-FP8 (dense) | sglang | fp8 | — | 256 | 600.4 | 4.69 | 3004 ms | 44.0 ms | 78.9 GB |
| gemma-4-31B-it-FP8-block (dense) | vllm | fp8 | — | 256 | 462.1 | 3.61 | 3174 ms | 61.1 ms | 72.8 GB |
| gemma-4-31B-it-FP8-block (dense) | sglang | fp8 | — | 256 | 454.5 | 3.55 | 4232 ms | 59.9 ms | 71.4 GB |
| Qwen3.6-35B-A3B-FP8 (DFlash spec-decode) | sglang | bf16 | DFlash | 256 | 475.2 | 8.17 | 675 ms | 68.8 ms | 76.9 GB |
| gemini-3.1-flash-lite (frontier baseline) | bifrost (remote) | — | — | 1024 | 237.0 | 10.37 | 1180 ms | 108.7 ms | n/a |

Baseline (concurrency=1) TTFT p95 for the local runs ranges 56–225 ms, so peak-load TTFTs above are well within the 32× degradation budget except for the gemma-4-31B-block dense run, which is the borderline case.

## Takeaways

### 1. MoE dominates dense at this VRAM budget
Qwen3.6-35B-A3B (3B active) on vLLM + FP8 KV hits **~1.8k tok/s and 14 req/s** at 72 GB peak — roughly **3× the throughput** of the similarly-sized dense Qwen3.6-27B and gemma-4-31B on the same hardware. gemma-4-26B-A4B (4B active MoE) is in the same league (~1.7k tok/s on SGLang). For the Category A orchestrator role, MoE-with-FP8 is clearly the deployment shape to favor.

### 2. vLLM ≈ SGLang for the MoEs; both saturate at concurrency 256
- vLLM is slightly faster on Qwen3.6-35B-A3B (1793 vs 1636 tok/s).
- SGLang is slightly faster on gemma-4-26B-A4B (1744 vs 1433 tok/s).
- Both engines plateau by concurrency 256, indicating the local cap is GPU compute / KV memory, not the scheduler.

### 3. FP8 KV cache beats BF16-KV + DFlash speculative decoding at high concurrency
On Qwen3.6-35B-A3B (SGLang), the BF16-KV + DFlash spec-decode config sustains only **475 tok/s** at concurrency 256, vs **1636 tok/s** with FP8 KV (no spec-decode) — a 3.4× gap. DFlash buys lower per-stream latency at low load but loses to FP8-KV under heavy batching, where the KV memory savings let many more requests pack into the same 80 GB. This confirms FP8 as the Phase 3 baseline against which TurboQuant/KIVI/KVQuant are scored; DFlash remains a useful low-concurrency latency lever, not a throughput one.

### 4. gemma-4-31B-block is the laggard
Dense 31B + per-block FP8 hits hard latency walls (TTFT p95 > 3s) by concurrency 256 on both engines. Not competitive on this rig vs Qwen3.6-35B-A3B or gemma-4-26B-A4B.

### 5. Frontier reference is throughput-bound by network, not GPU
gemini-3.1-flash-lite via bifrost sustains 1024 concurrent sessions but only **237 output tok/s** — remote API latency dominates. Use only as an external quality/latency baseline, not a throughput peer.

## File index

| File | Notes |
|---|---|
| `Qwen_Qwen3.6-35B-A3B-FP8_vllm_fp8.json` | Top local throughput |
| `Qwen_Qwen3.6-35B-A3B-FP8_sglang_fp8.json` | Close second |
| `Qwen_Qwen3.6-35B-A3B-FP8_sglang_auto.json` | BF16-KV + DFlash speculative-decoding config |
| `Qwen_Qwen3.6-27B-FP8_{vllm,sglang}_fp8.json` | Dense 27B Qwen |
| `RedHatAI_gemma-4-26B-A4B-it-FP8-Dynamic_{vllm,sglang}_fp8.json` | MoE Gemma-4 |
| `RedHatAI_gemma-4-31B-it-FP8-block_{vllm,sglang}_fp8.json` | Dense Gemma-4 (laggard) |
| `gemini_gemini-3.1-flash-lite_frontier.json` | Remote frontier baseline |

`.log` files alongside each JSON contain the raw client output.
