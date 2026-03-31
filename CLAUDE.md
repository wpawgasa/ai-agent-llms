# LLM Workflow-Orchestrating Agents with KV Cache Quantization

## Project Overview

Four-track experimental program investigating how LLMs of different sizes serve distinct roles in workflow-orchestrating agent systems, benchmarked on a single NVIDIA H100 SXM 80GB GPU.

| Track | Scope | Models |
|-------|-------|--------|
| **Exp A** | Prompt-encoded business logic + tool calling | 5 x 15-30B |
| **Exp B** | Fine-tuned specialist subagents | 5 x 2-5B |
| **Exp C** | Prompt-to-graph extraction | 5 x 2-5B (same) |
| **Exp D** | KV cache quantization benchmark | All 10 models x 6 methods |
| **E2E** | Integration deployment | Best combos from A-D |

## Reference Repository

[wpawgasa/vlm-ocr-gpu-benchmark](https://github.com/wpawgasa/vlm-ocr-gpu-benchmark) serves as a reference design for this project. When implementing modules, look for similar patterns in that repo:

| This Project | Reference Repo Equivalent |
|---|---|
| `configs/` (YAML model/experiment configs) | `configs/` (models, experiments, hardware, phases) |
| `data/` (data generation, validation) | `src/vlm_ocr_bench/data/` |
| `training/` (LoRA, SFTTrainer) | `src/vlm_ocr_bench/training/` (lora.py, trainer.py, callbacks.py) |
| `eval/` (metrics, benchmarks) | `src/vlm_ocr_bench/evaluation/` + `inference/metrics.py` |
| `serving/` (vLLM launch, orchestration) | `src/vlm_ocr_bench/inference/` (engine.py, vllm_config.py, runner.py) |
| `analysis/` (Pareto, plots) | `src/vlm_ocr_bench/analysis/` |
| `scripts/` (experiment runners) | `src/vlm_ocr_bench/cli/` |
| `tests/` | `tests/` (unit, integration, smoke test structure) |
| Model adapters / registry | `src/vlm_ocr_bench/models/` (registry.py, adapters/, loader.py) |
| GPU profiling | `src/vlm_ocr_bench/profiling/` (monitor.py, nvidia_smi.py) |
| `pyproject.toml`, CI | `pyproject.toml`, `.github/workflows/`, `Makefile` |

## Tech Stack

- Python, PyTorch >=2.4, Triton >=3.0, vLLM >=0.8 (PagedAttention v2)
- HF Transformers >=5.0, PEFT >=0.14, TRL >=0.15
- W&B for logging, Outlines/XGrammar for constrained decoding
- Target: Single NVIDIA H100 SXM 80GB

## Repository Structure

```
configs/          - YAML configs (model, quantization, serving)
data/             - Data generation and templates
training/         - Fine-tuning (Exp B, C) with LoRA/SFTTrainer
quantization/     - KV cache quantization (TurboQuant, RotorQuant, baselines)
eval/             - Evaluation modules (state accuracy, tool F1, graph, PPL)
serving/          - vLLM serving and multi-agent orchestration
analysis/         - Pareto frontier and visualization
scripts/          - Top-level experiment runners (run_exp_{a,b,c,d,e2e}.sh)
tests/            - Unit and integration tests
```

## Module Implementation Details

Detailed specs and checklists for each module are in `.claude/rules/`:
- `01-configs.md` - Configuration schema and YAML files
- `02-data-generation.md` - Data generation layer (Exp A, B, C data)
- `03-training.md` - Fine-tuning layer (LoRA, SFTTrainer)
- `04-quantization.md` - KV cache quantization (TurboQuant, RotorQuant, baselines)
- `05-eval.md` - Evaluation metrics and benchmarks
- `06-serving.md` - vLLM serving and multi-agent orchestrator
- `07-analysis.md` - Result analysis and Pareto computation
- `08-testing.md` - Testing strategy

## Implementation Progress

### Phase 1: Foundation (Weeks 1-2)
- [x] Project scaffolding (pyproject.toml, requirements.txt, directory structure)
- [x] Configuration schema and YAML files (`configs/`)
- [x] Data generation module (`data/`)
- [x] Data validation and chat template converter

### Phase 2: Training (Weeks 3-4)
- [x] LoRA target module registry (`training/lora_targets.py`)
- [x] Unified SFTTrainer entry point (`training/train_specialist.py`)
- [x] Graph extraction trainer (`training/train_graph_extractor.py`)
- [x] Adapter merge utility (`training/merge_adapter.py`)

### Phase 3: Experiment A Evaluation (Weeks 5-6)
- [x] State machine accuracy eval (`eval/state_accuracy.py`)
- [x] Tool-call F1 eval (`eval/tool_call_f1.py`)
- [x] Tool chain propagation eval (`eval/tool_chain_propagation.py`)
- [x] Agent benchmark composite (`eval/agent_benchmark.py`)
- [x] vLLM launch script (`serving/launch_vllm.sh`)

### Phase 4: Quantization (Weeks 7-8)
- [ ] TurboQuant codebook + rotation (`quantization/turboquant/`)
- [ ] TurboQuant Triton kernels + vLLM integration
- [ ] RotorQuant Clifford algebra + Triton kernels (`quantization/rotorquant/`)
- [ ] Baseline wrappers: KIVI, KVQuant (`quantization/baselines/`)

### Phase 5: Experiment D Benchmarks (Weeks 9-10)
- [ ] Perplexity eval (`eval/perplexity.py`)
- [ ] LongBench eval (`eval/longbench.py`)
- [ ] Needle-in-a-Haystack eval (`eval/needle_haystack.py`)
- [ ] Benchmark matrix execution (10 models x 6 methods)

### Phase 6: Experiment C (Week 11)
- [ ] Graph extraction eval (`eval/graph_extraction_eval.py`)
- [ ] Constrained decoding integration (Outlines/XGrammar)

### Phase 7: Integration & E2E (Weeks 12-14)
- [ ] Multi-agent orchestrator (`serving/orchestrator.py`)
- [ ] E2E benchmark (`serving/benchmark_e2e.py`)
- [ ] Pareto frontier analysis (`analysis/pareto.py`)
- [ ] Visualization (`analysis/plot_results.py`)
- [ ] Experiment runner scripts (`scripts/run_exp_*.sh`)
- [ ] Final integration testing

## Key Architecture Decisions

- **ADR-001**: vLLM as unified serving backend (tool-call parsing, LoRA multi-adapter, custom KV cache dtypes)
- **ADR-002**: LoRA over full fine-tuning (VRAM constraint on single H100)
- **ADR-003**: Triton for custom quantization kernels (TurboQuant, RotorQuant)
- **ADR-004**: Teacher-model (GPT-4o/Claude) synthetic data generation

## Known Risks

- R1: Qwen3-32B BF16 ~64GB leaves only 16GB for KV cache -> AWQ-INT4 fallback
- R3: Nemotron Mamba layers may not work with vLLM -> HF generate() fallback
- R4: TurboQuant PR not merged -> use community fork
- R5: RotorQuant has no vLLM integration -> standalone benchmark
- R7: GLM LoRA VRAM overflow -> auto-fallback rank 64->32
