# LLM Workflow-Orchestrating Agents with KV Cache Quantization

**Version:** 3.0 — March 2026

## Project Overview

**Benchmark-first, fine-tune-selectively** pipeline for workflow-orchestrating LLM agents on a single NVIDIA H100 SXM 80GB.

### Four-Phase Pipeline

| Phase | Scope | Output |
|-------|-------|--------|
| **Phase 1** | Benchmark 15 pre-trained candidates across 3 task categories | Rankings + 3 category winners |
| **Phase 2** | SFT then GRPO RL fine-tune only the 3 winners | 3 fine-tuned specialist models |
| **Phase 3** | KV cache quantization benchmark (models × 6 methods) | Quality/performance matrix |
| **Phase 4** | Multi-agent integration + Pareto analysis | Optimal deployment config |

### Task Categories & Model Inventory

**Category A — Prompt-Encoded Business Logic (15–35B):**
- Gemma 3 27B-IT, Qwen3-32B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B, Qwen3.6-27B, Mistral Small 3.1 24B, Nemotron-3-Nano 30B, GLM-4.7-Flash, Gemma 4 26B-A4B-IT, Gemma 4 31B-IT

**Category B–C — Specialist Subagent & Graph Extraction (2–5B):**
- Qwen2.5-3B-Instruct, Qwen3.5-4B, GLM-4.7-Flash, Gemma-2B, Gemma-3-4B-it, Gemma-4-E4B-it, Gemma-4-E2B-it

## Reference Repository

[wpawgasa/vlm-ocr-gpu-benchmark](https://github.com/wpawgasa/vlm-ocr-gpu-benchmark) serves as a reference design for this project.

## Tech Stack

- Python, PyTorch >=2.4, Triton >=3.0, vLLM >=0.8 (PagedAttention v2)
- **Unsloth >=2025.3** — SFT + GRPO RL (2x faster, 70% less VRAM, MoE kernels)
- HF Transformers >=5.0, PEFT >=0.14, TRL >=0.15 (GRPOTrainer)
- W&B for logging, Outlines/XGrammar for constrained decoding
- Target: Single NVIDIA H100 SXM 80GB

## Repository Structure

```
configs/          - YAML configs (models/cat_a/, models/cat_bc/, training/, quantization/, benchmark/, serving/)
data/             - Data generation and templates (output: task_a/, task_b/, task_c/)
training/         - Phase 2: SFT + GRPO RL (sft.py, grpo.py, rewards/, lora_targets.py, pilot_check.py)
quantization/     - Phase 3: KV cache quantization (turboquant/, rotorquant/, baselines/)
eval/             - Evaluation modules (state accuracy, tool F1, graph, PPL, composite_score.py, quant_benchmark.py)
integration/      - Phase 4: multi-agent orchestrator, E2E benchmark, Pareto (orchestrator.py, pareto.py)
serving/          - vLLM serving utilities (launch_vllm.sh, vllm_utils.py)
analysis/         - Visualization (plot_phase1_rankings.py, plot_sft_vs_rl.py, plot_quant_matrix.py, plot_pareto.py)
scripts/          - Experiment runners (run_phase1.sh, run_phase2_sft.sh, run_phase2_grpo.sh, run_phase3.sh, run_phase4.sh)
tests/            - Unit and integration tests
```

## Module Implementation Details

Detailed specs for each module are in `.claude/rules/`:
- `01-configs.md` - Configuration schema and YAML files
- `02-data-generation.md` - Data generation layer (Tasks A, B, C)
- `03-training.md` - SFT + GRPO RL training (Unsloth, reward functions)
- `04-quantization.md` - KV cache quantization (TurboQuant, RotorQuant, baselines)
- `05-eval.md` - Evaluation metrics and benchmarks
- `06-serving.md` - vLLM serving utilities
- `07-analysis.md` - Result analysis and visualization
- `08-testing.md` - Testing strategy
- `10-integration.md` - Phase 4 multi-agent integration and Pareto

Phase 1 benchmarking runs through `eval/agent_benchmark.py` (invoked by `scripts/run_exp_a*.sh`). The separate `benchmark/` package was removed; see git history.

## Implementation Progress

### Phase 1: Foundation — Data & Configs
- [x] Project scaffolding (pyproject.toml, requirements.txt, directory structure)
- [x] Configuration schema and YAML files (`configs/`)
- [x] Add Gemma 4 model YAML configs (`configs/models/cat_a/gemma4_26b_a4b.yaml`, `gemma4_31b.yaml`; `configs/models/cat_bc/gemma4_e4b.yaml`, `gemma4_e2b.yaml`)
- [x] Add Qwen3.6 model YAML configs (`configs/models/cat_a/qwen36_35b_a3b.yaml`, `qwen36_35b_a3b_fp8.yaml`, `qwen36_27b.yaml`, `qwen36_27b_fp8.yaml`)
- [x] Data generation module (`data/`)
- [x] Data validation and chat template converter

### Phase 2: Training
- [x] LoRA target module registry (`training/lora_targets.py`) — 10 models incl. qwen35_35b_a3b, nemotron_30b
- [x] Add Gemma 4 model entries to `training/lora_targets.py` (gemma4_26b_a4b, gemma4_31b, gemma4_e4b, gemma4_e2b)
- [x] Add Qwen3.6 model entries to `training/lora_targets.py` (qwen36_35b_a3b, qwen36_27b)
- [x] Unified SFTTrainer entry point (`training/train_specialist.py` — v2 backward-compat)
- [x] Graph extraction trainer (`training/train_graph_extractor.py` — v2 backward-compat)
- [x] Adapter merge utility (`training/merge_adapter.py`) — with quantize_merged param
- [x] Unsloth SFT entry point (`training/sft.py`)
- [x] Unsloth GRPO RL entry point (`training/grpo.py`)
- [x] Task-specific reward functions (`training/rewards/`)
- [x] Shared reward helpers (`training/reward_utils.py`)
- [x] Pilot check for fine-tuning response (`training/pilot_check.py`)

### Phase 3: Experiment A Evaluation
- [x] State machine accuracy eval (`eval/state_accuracy.py`)
- [x] Tool-call F1 eval (`eval/tool_call_f1.py`)
- [x] Tool chain propagation eval (`eval/tool_chain_propagation.py`)
- [x] Agent benchmark composite (`eval/agent_benchmark.py`)
- [x] vLLM launch script (`serving/launch_vllm.sh`)
- [x] Composite score module (`eval/composite_score.py`)
- [x] Quantization benchmark harness (`eval/quant_benchmark.py`)

### Phase 4: Quantization (KV Cache)
- [x] TurboQuant codebook + rotation (`quantization/turboquant/`)
- [x] TurboQuant Triton kernels + vLLM integration
- [x] RotorQuant Clifford algebra + Triton kernels (`quantization/rotorquant/`)
- [x] Baseline wrappers: KIVI, KVQuant (`quantization/baselines/`)

### Phase 5: Quantization Benchmarks
- [x] Perplexity eval (`eval/perplexity.py`)
- [x] LongBench eval (`eval/longbench.py`)
- [x] Needle-in-a-Haystack eval (`eval/needle_haystack.py`)
- [x] Benchmark matrix execution (10 models x 6 methods)

### Phase 6: Graph Extraction Eval
- [x] Graph extraction eval (`eval/graph_extraction_eval.py`)
- [x] Constrained decoding integration (Outlines/XGrammar)

### Phase 7 (v2) / Phase 4 (v3): Integration & E2E
- [x] Multi-agent orchestrator (`serving/orchestrator.py` → v3: `integration/orchestrator.py`)
- [x] E2E benchmark (`serving/benchmark_e2e.py` → v3: `integration/benchmark_e2e.py`)
- [x] Pareto frontier analysis (`analysis/pareto.py` → v3: `integration/pareto.py`)
- [x] Visualization (`analysis/plot_results.py` → v3: split into `analysis/plot_*.py`)
- [x] Experiment runner scripts
- [x] v3 integration module (`integration/`)
- [x] v3 analysis plots (plot_phase1_rankings.py, plot_sft_vs_rl.py, plot_quant_matrix.py, plot_pareto.py)

## Key Architecture Decisions

- **ADR-001**: Benchmark-first model selection (evaluate all 15 pre-trained first, fine-tune only 3 winners)
- **ADR-002**: Unsloth over standard PEFT/TRL (2x speed, 70% less VRAM, GRPO + MoE kernels)
- **ADR-003**: SFT then GRPO two-stage training (SFT for format/domain, GRPO for task metric optimization)
- **ADR-004**: Shared SFT base for dual-category winners (diverge at GRPO stage)
- **ADR-005**: Triton for custom quantization kernels (TurboQuant, RotorQuant)
- **ADR-006**: Use upstream vLLM TurboQuant variants (`turboquant_3bit_nc`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_k8v4`) for Phase 3 benchmarks instead of the project's custom `"turboquant"` string. vLLM v1's plugin architecture requires a full `AttentionBackend` subclass for any new KV compression scheme; monkey-patching module-level cache I/O (the v0 approach) no longer works. Upstream ships `vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend`, so we adopt it directly.

## Pending Work: Custom KV Cache Backends

The project's custom TurboQuant/RotorQuant code (`src/llm_workflow_agents/quantization/{turboquant,rotorquant}/`) is **scaffolding-only** against current vLLM. Launchers (`src/llm_workflow_agents/serving/launch_vllm_{turboquant,rotorquant}.py`) wire plain `"turboquant"` / `"rotorquant"` through argparse and a Pydantic validator wrapper, so `CacheConfig` accepts them and the server starts — but no compression happens because stock attention backends ignore the dtype string.

To make the custom implementations actually compress KV cache at inference time, both would need a full v1 `AttentionBackend` subclass (reference: `vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend`, ~800 LOC). Each port must implement `get_kv_cache_shape`, `supported_kv_cache_dtypes`, `AttentionImpl.forward` (prefill + decode), and `AttentionMetadataBuilder`; register via `register_backend(AttentionBackendEnum.CUSTOM, "...")` and monkey-patch `STR_DTYPE_TO_TORCH_DTYPE` in the launcher for the new dtype string.

**Decision (2026-04-21): neither port will be written for Phase 3.**

- **TurboQuant (project variant):** not porting. The only thing the project's version has that upstream's four variants don't is the QJL residual path — a ~1-2 PPL optimization at best. Cost (2-3 days port + 2-3 days validation) is not justified when upstream's `turboquant_3bit_nc` / `_4bit_nc` / `_k3v4_nc` / `_k8v4` already cover the compression/quality curve. Use upstream.
- **RotorQuant:** porting gated on a standalone microbenchmark. Before investing 1-2 weeks on a v1 backend, validate the Cl(3,0)-rotor-vs-Hadamard quality claim with a PyTorch-eager microbenchmark on Qwen2.5-3B held-out KV tensors. Compare reconstruction error + downstream PPL against Hadamard rotation at matched bit budget. If RotorQuant wins materially, port. If not, drop it from Phase 3 and the research track.
- **Scaffolding (`launch_vllm_turboquant.py` project-custom path, `launch_vllm_rotorquant.py`, `register_turboquant_backend` Pydantic patch):** keep in-tree but route only exact `"turboquant"` / `"rotorquant"` strings through the custom launchers (`serving/launch_vllm.sh:148-154`). Upstream `turboquant_*` variants bypass the project path entirely and go straight to stock vLLM.
- **`_patch_block_size` / `_patch_paged_attention` in `vllm_integration.py`:** leave as no-op with migration warnings until a port actually lands (or until RotorQuant is dropped, whichever comes first).

Phase 3 benchmark matrix: upstream `turboquant_*` variants + FP8 + KIVI + KVQuant + AWQ-INT4+FP8 KV. RotorQuant column is **provisionally deferred** pending the microbenchmark result.

### KV Cache CPU Offload (LMCache) — serving knob, not a benchmark target

The docker serving stack (`deployments/models/docker-compose.yml` + `.l4.yml`) exposes an opt-in LMCache CPU KV-cache offload knob via `VLLM_LMCACHE_ENABLED` (+ `VLLM_LMCACHE_CPU_SIZE`/`_CHUNK_SIZE`/`_VERSION`/`_KV_TRANSFER_CONFIG`; see `.env.example`). When enabled, the entrypoint pip-installs `lmcache` at container start and passes `--kv-transfer-config` to vLLM, offloading **full-precision** KV to CPU RAM and re-fetching on prefix reuse. This is **orthogonal to and not stackable with** the on-GPU KV-quant backends: combining `VLLM_LMCACHE_ENABLED` with `VLLM_KV_CACHE_DTYPE=turboquant*`/`rotorquant*` hard-fails in the entrypoint (both hook KV cache I/O). `fp8` KV is *not* blocked but is untested with LMCache — its full-precision CPU buffer alongside an fp8 on-GPU cache has unvalidated round-trip semantics. It is a serving/deployment knob only — not a Phase 3 quantization benchmark column. Entrypoint logic is unit-tested via `tests/test_lmcache_entrypoint.py` using the `VLLM_ENTRYPOINT_DRY_RUN` affordance (no GPU/docker required).

**LMCache ⊥ hybrid models (verified 2026-07-02).** LMCache is **hard-broken with any hybrid (sliding+full attention or Mamba) model on vLLM 0.20.0** — this covers **all Gemma-4 variants (E2B/E4B/26B-A4B/31B)** and **Nemotron**. `LMCacheConnectorV1` does not implement `SupportsHMA`, so its `--kv-transfer-config` forces vLLM's hybrid KV cache manager *off*; but Gemma-4's mixed sliding/full attention groups *require* it. There is no working flag combination — both branches crash `EngineCore` at startup during KV-cache init:
- **Default** (hybrid manager off): `KeyError: '…self_attn.attn'` in `get_attn_backends_for_group` (`gpu_model_runner.py:6265`) — layers of one group aren't in that group's `kv_cache_specs`. Fires in *both* the CUDA-graph profiler path (`profile_cudagraph_memory`) and the real `initialize_from_config` path, so `--enforce-eager` does **not** help (it only gates the former).
- **Force it on** (`--no-disable-hybrid-kv-cache-manager` via `VLLM_EXTRA_ARGS`): `ValueError: Connector LMCacheConnectorV1 does not support HMA but HMA is enabled`.

`enforce_eager` and `fp8` KV are red herrings here — the blocker is architectural. **Only enable LMCache for non-hybrid dense models** (e.g. Qwen3.6 dense variants). Consider a future entrypoint guard that hard-fails `VLLM_LMCACHE_ENABLED` + a known-hybrid model, mirroring the existing turboquant/rotorquant guard, so it fails fast with a clear message instead of the cryptic `KeyError`.

### Known Model × TurboQuant Incompatibilities

| Model | Reason | Workaround |
|-------|--------|------------|
| Gemma4 26B-A4B, Gemma4 31B | vLLM v1's KV cache profiler (`gpu_model_runner.py:6598`) fails to `.view()` padded raw tensors across mixed KV cache groups when the sliding:full layer ratio requires padding. | Auto-resolved. `_install_turboquant_engine_config_hook` in `launch_vllm_turboquant.py` detects Gemma-4 via `AutoConfig.architectures` and auto-injects `enforce_eager=True`, which gates out `profile_cudagraph_memory` (`gpu_worker.py:380-385`). Decode throughput drops ~15-25% (no CUDA graphs); numerical correctness unaffected. Restore graph-mode once upstream lands a per-layer-aware profiler. |
| Nemotron-3-Nano 30B (Mamba hybrid) | `arg_utils.py:1649` raises `NotImplementedError` on any hybrid + `turboquant_*`. Also blocked by separate Mamba+vLLM compat issues (Risk R6). | Use HF `generate()` fallback path; TurboQuant not viable here. |

Compatible targets for the TurboQuant cells: Qwen3-32B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B (+ FP8), Qwen3.6-27B (+ FP8), Mistral-Small-3.1-24B, Gemma-3-27B, GLM-4.7-Flash, Gemma4-26B-A4B, Gemma4-31B. Qwen3.5/3.6 unblocked via `_install_turboquant_engine_config_hook` which masks `ModelConfig.is_hybrid` to False during `create_engine_config` — DeltaNet/Mamba layer indices in the boundary-skip list are harmless no-ops since those layers don't construct `Attention()` modules. Gemma-4 unblocked as described in the table.

## Known Risks

- R1: Qwen3.5-35B-A3B BF16 ~70GB → Unsloth QLoRA 4-bit (~17.5GB) + FP8 RL
- R2: Same model wins Cat B + Cat C → share SFT base, diverge at GRPO
- R3: Phase 1 winner doesn't respond to fine-tuning → 100-step pilot SFT on top-2 (`training/pilot_check.py`)
- R4: TurboQuant PR not merged → use `0xSero/turboquant` fork
- R5: GRPO reward hacking → held-out eval every 50 steps, KL monitoring, auto-stop
- R6: Nemotron Mamba + vLLM incompatibility → HF `generate()` fallback. Qwen3.5/3.6 hybrid (DeltaNet) unblocked for `turboquant_*` via `_install_turboquant_engine_config_hook` in `launch_vllm_turboquant.py`; Nemotron-3-Nano remains out of scope (Mamba layers + vLLM compat issues beyond the hybrid guard).
- R7: GLM LoRA VRAM overflow → auto-reduce rank 64→32 or inference-only
- R8: Gemma4 + TurboQuant → **works with perf caveat**. Two hooks in `launch_vllm_turboquant.py`: (1) `_install_gemma4_mixed_backend_hook` bypasses the TRITON_ATTN global force at `vllm/model_executor/models/config.py:100`; (2) `_install_turboquant_engine_config_hook` auto-injects `enforce_eager=True` for Gemma-4 + `turboquant_*`, skipping vLLM v1's broken mixed-KV profiler (`gpu_model_runner.py:6598`). CUDA graphs off → expect ~15-25% decode throughput penalty vs graph-mode. Numerical correctness unaffected. Restore graph-mode once upstream lands a per-layer-aware profiler.
- R9: Gemma-4 (26B-A4B, 31B) + Unsloth `fast_inference=True` → **architecture allowlist rejection**. Unsloth 2026.5.2's `VLLM_SUPPORTED_VLM` at `unsloth/models/vision.py:242` is hardcoded to `{qwen2_5_vl, gemma3, mistral3, qwen3_vl, qwen3_vl_moe}`. Gemma-4 ships as a SigLIP+Gemma4 multimodal stack, so `FastLanguageModel.from_pretrained(..., fast_inference=True)` raises `RuntimeError: Found architectures: siglip, gemma4!` (`unsloth/models/vision.py:610`). **Auto-resolved.** `training/grpo.py::_detect_model_family` reads the SFT checkpoint's `adapter_config.json → base_model_name_or_path` and resolves `AutoConfig.model_type`. When the family is in `UNSLOTH_VLLM_INCOMPATIBLE_FAMILIES = {"gemma4"}`, `use_vllm` is forced to `False` with a `vllm_rollout_disabled_unsloth_incompat` warning; training proceeds with HF `model.generate()` rollouts (~31 s/step on H100 80GB → ~8.6 h per 1000-step GRPO run). Other Cat A models (Qwen3-32B, Qwen3.5-35B-A3B, Mistral-Small-3.1-24B, Qwen3.6 variants) are unaffected — they continue to get Unsloth colocate vLLM rollouts. To re-enable vLLM rollouts on Gemma-4: (a) wait for Unsloth to add `gemma4` to the allowlist (file an issue at github.com/unslothai/unsloth), (b) switch to external vLLM server mode (`vllm_mode="server"` + a separate `trl vllm-serve` process — tight on 80GB but viable with FP8 weights), or (c) monkey-patch `unsloth.models.vision.VLLM_SUPPORTED_VLM` at import time (quick to try, may fail deeper in vLLM-engine init).
- R10: DiffusionGemma 26B-A4B + vLLM → **no tagged release supports it** (verified 2026-06-11: vLLM 0.22.1 model registry has no `diffusion_gemma` entry; transformers added the model defs in 5.11.0; the HF checkpoints ship no remote code). Serving works ONLY via vLLM's `dgemma` branch (PR #45163, open) — prebuilt images `vllm/vllm-openai:gemma-cu129` (CUDA 12.9) / `:gemma` (CUDA 13.0, driver ≥ 580). Required flags per the RedHatAI model card: `VLLM_USE_V2_MODEL_RUNNER=1`, `--trust-remote-code`, `--max-num-seqs 4`, `--hf-overrides '{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1}'`. Wired into `deployments/models/` (`VLLM_IMAGE` + `VLLM_USE_V2_MODEL_RUNNER` + `VLLM_TRUST_REMOTE_CODE` knobs; preset in `.env.example`). `serving/launch_vllm.sh`'s `.venv-infer` (vLLM 0.20.0) cannot serve it — fails at ModelConfig validation ("model type `diffusion_gemma` ... not recognize this architecture"). Keep `kv-cache-dtype` blank (fp8 KV on the diffusion decode path unvalidated). Re-pin to a tagged image once PR #45163 merges into a release.
