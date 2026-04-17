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
- Gemma 3 27B-IT, Qwen3-32B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B, Mistral Small 3.1 24B, Nemotron-3-Nano 30B, GLM-4.7-Flash, Gemma 4 26B-A4B-IT, Gemma 4 31B-IT

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
- [x] Add Qwen3.6 model YAML configs (`configs/models/cat_a/qwen36_35b_a3b.yaml`, `qwen36_35b_a3b_fp8.yaml`)
- [x] Data generation module (`data/`)
- [x] Data validation and chat template converter

### Phase 2: Training
- [x] LoRA target module registry (`training/lora_targets.py`) — 10 models incl. qwen35_35b_a3b, nemotron_30b
- [x] Add Gemma 4 model entries to `training/lora_targets.py` (gemma4_26b_a4b, gemma4_31b, gemma4_e4b, gemma4_e2b)
- [x] Add Qwen3.6 model entry to `training/lora_targets.py` (qwen36_35b_a3b)
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

## Known Risks

- R1: Qwen3.5-35B-A3B BF16 ~70GB → Unsloth QLoRA 4-bit (~17.5GB) + FP8 RL
- R2: Same model wins Cat B + Cat C → share SFT base, diverge at GRPO
- R3: Phase 1 winner doesn't respond to fine-tuning → 100-step pilot SFT on top-2 (`training/pilot_check.py`)
- R4: TurboQuant PR not merged → use `0xSero/turboquant` fork
- R5: GRPO reward hacking → held-out eval every 50 steps, KL monitoring, auto-stop
- R6: Nemotron Mamba + vLLM incompatibility → HF `generate()` fallback
- R7: GLM LoRA VRAM overflow → auto-reduce rank 64→32 or inference-only
