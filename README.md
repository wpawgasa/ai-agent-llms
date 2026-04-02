# LLM Workflow-Orchestrating Agents with KV Cache Quantization

**Version:** 3.0 — March 2026 | **Hardware:** Single NVIDIA H100 SXM 80GB | **Training:** Unsloth SFT + GRPO RL

A **benchmark-first, fine-tune-selectively** pipeline for deploying workflow-orchestrating LLM agents. Instead of fine-tuning every candidate, we benchmark all pre-trained models first, fine-tune only the three category winners, then compress them with novel KV cache quantization for concurrent serving on a single H100.

---

## Pipeline Overview

```
Phase 1: Benchmark          Phase 2: Fine-Tune          Phase 3: Quantize           Phase 4: Deploy
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ 11 pre-trained  │         │ 3 category       │         │ 6 KV cache      │         │ Multi-agent     │
│ candidates      │──rank──▶│ winners:         │──SFT──▶ │ methods ×       │──best──▶│ orchestrator    │
│ × 3 task cats   │  &      │ Cat A (15–35B)   │  then   │ 10 models       │  config │ + Pareto        │
│ → composite     │  pick   │ Cat B (2–5B)     │  GRPO   │ → quality /     │         │ analysis        │
│   scores        │         │ Cat C (2–5B)     │         │   perf matrix   │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘         └─────────────────┘
  scripts/run_exp_a.sh        scripts/run_exp_b.sh         scripts/run_exp_d.sh        scripts/run_exp_e2e.sh
  scripts/run_exp_c.sh                                                                 
```

---

## Model Inventory

### Category A — Prompt-Encoded Business Logic (15–35B)

| Model | Params | Active | Architecture | Context | Tool Parser | VRAM (BF16) |
|-------|--------|--------|-------------|---------|-------------|-------------|
| Gemma 3 27B-IT | 27B | 27B | Dense GQA | 128K | `gemma` | ~54 GB |
| Qwen3-32B | 32B | 32B | Dense + `<think>` | 128K | `hermes` | ~64 GB |
| Qwen3.5-35B-A3B | 35B | 3B | DeltaNet MoE | 262K | `qwen3_coder` | ~70 GB |
| Mistral Small 3.1 24B | 24B | 24B | Dense sliding-window | 128K | `mistral` | ~48 GB |
| Nemotron-3-Nano 30B | 30B | 3.6B | MoE + Mamba-2 | 1M | `nemotron` | ~60 GB |
| GLM-4.7-Flash | 30B | 3.6B | MoE + MLA | 200K | `glm4` | ~60 GB |

### Category B–C — Specialist Subagent & Graph Extraction (2–5B)

| Model | Params | Active | Architecture | Context | VRAM |
|-------|--------|--------|-------------|---------|------|
| Qwen2.5-3B-Instruct | 3B | 3B | Dense GQA | 32K | ~6 GB |
| Qwen3.5-4B | 4B | ~3B | DeltaNet MoE | 262K | ~8 GB |
| GLM-4.7-Flash | 30B | 3.6B | MoE + MLA | 200K | ~60 GB |
| Gemma-2B | 2.5B | 2.5B | Dense MQA | 8K | ~5 GB |
| Gemma-3-4B-it | 4B | 4B | Dense GQA | 128K | ~8 GB |

---

## Repository Structure

```
.
├── configs/
│   ├── models/cat_a/          # 6 × Category A model configs
│   ├── models/cat_bc/         # 5 × Category B–C model configs
│   ├── models_exp_a/          # Experiment A serving configs (with benchmark params)
│   ├── models_exp_bc/         # Experiment B/C serving configs
│   ├── training/              # SFT + GRPO hyperparameter configs (6 files)
│   ├── quantization/          # 6 KV cache quantization method configs
│   ├── benchmark/             # Phase 1 matrix + composite score weights
│   └── serving/               # Single-model, multi-agent, E2E benchmark configs
│
├── src/llm_workflow_agents/
│   ├── data/                  # Synthetic data generation (Tasks A, B, C)
│   ├── benchmark/             # Phase 1 orchestration + model selection
│   ├── training/              # Unsloth SFT + GRPO RL + reward functions
│   ├── quantization/          # TurboQuant, RotorQuant, KIVI, KVQuant
│   ├── eval/                  # All evaluation metrics
│   ├── integration/           # Multi-agent orchestrator + Pareto analysis
│   ├── serving/               # vLLM launch + adapter utilities
│   └── analysis/              # Result visualization
│
├── scripts/
│   ├── generate_benchmark_data.sh   # 1 000 benchmark samples (no API key)
│   ├── generate_sft_data.sh         # ~12 504 SFT training samples
│   ├── generate_grpo_data.sh        # 2 250 GRPO training prompts
│   ├── generate_eval_data.sh        # 1 000 val + test samples
│   ├── run_exp_a.sh                 # Experiment A: Cat A benchmark
│   ├── run_exp_b.sh                 # Experiment B: Cat B–C fine-tuning
│   ├── run_exp_c.sh                 # Experiment C: graph extraction
│   ├── run_exp_d.sh                 # Experiment D: KV cache quant benchmark
│   └── run_exp_e2e.sh               # E2E integration + Pareto
│
├── docs/
│   ├── data_generation_recipes.md   # Data generation guide with cost estimates
│   └── Codebase_Spec_LLM_Workflow_Agents_Experiment.md
│
└── tests/                     # Unit + integration tests
```

---

## Setup

**Requirements:** Python 3.11+, CUDA 12.x, NVIDIA H100 (80 GB recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=...
#   GEMINI_API_KEY=...
#   WANDB_API_KEY=...
```

---

## Data Generation

See [`docs/data_generation_recipes.md`](docs/data_generation_recipes.md) for the full guide. Quick start:

```bash
# Benchmark data — no API key needed
./scripts/generate_benchmark_data.sh

# SFT training data (~12 504 samples, ~$42 in API costs)
OPENAI_API_KEY=... GEMINI_API_KEY=... ./scripts/generate_sft_data.sh

# GRPO training prompts (2 250 samples, L3–L5 only)
OPENAI_API_KEY=... GEMINI_API_KEY=... ./scripts/generate_grpo_data.sh

# Validation + test splits
OPENAI_API_KEY=... ./scripts/generate_eval_data.sh
```

Each script accepts `--dry-run` to preview commands without executing, and `--output-dir` to redirect output.

### Data Design

| Split | Samples | Seed | Behavior preset | Languages |
|-------|---------|------|-----------------|-----------|
| Benchmark | 1 000 | 100 | default | mixed (en/th) |
| SFT | ~12 504 | 42 | **adversarial** | en + th + code-switch |
| GRPO | 2 250 | 200 | **balanced** | mixed + code-switch |
| Validation | 500 | 300 | default | mixed |
| Test | 500 | 400 | default | mixed |

**Code-switching** (`language="code_switch"`) generates Thai-English mixed conversations — Thai sentence structure with embedded English terms — reflecting real call-centre interactions and providing harder training examples.

**Behavior presets:**

| Preset | cooperative | adversarial_probing | digressing | invalid_tool_inputs |
|--------|-------------|---------------------|------------|---------------------|
| default | 60% | 15% | 10% | 15% |
| adversarial | 45% | 25% | 15% | 15% |
| balanced | 25% | 25% | 25% | 25% |

---

## Running Experiments

### Experiment A — Category A Benchmark

Launches each 15–35B model sequentially via vLLM, runs workflow quality evaluation, saves results to `results/exp_a/`.

```bash
./scripts/run_exp_a.sh
./scripts/run_exp_a.sh --kv-cache-dtype fp8   # with quantization
./scripts/run_exp_a.sh --dry-run
```

### Experiment B — Specialist Subagent Fine-Tuning

LoRA fine-tunes each 2–5B model (SFT then GRPO), evaluates tool-call F1.

```bash
./scripts/run_exp_b.sh
./scripts/run_exp_b.sh --skip-training        # eval only, using existing checkpoints
```

### Experiment C — Graph Extraction

Trains graph extractor variants with constrained JSON decoding (Outlines/XGrammar), evaluates Node F1 / Edge F1 / GED.

```bash
./scripts/run_exp_c.sh
```

### Experiment D — KV Cache Quantization Benchmark

Runs all (model, quantization method) pairs, measuring PPL, LongBench, Needle-in-Haystack, VRAM, and latency.

```bash
./scripts/run_exp_d.sh
./scripts/run_exp_d.sh --models-only exp_a    # Cat A models only
./scripts/run_exp_d.sh --models-only exp_bc   # Cat B–C models only
```

### E2E Integration

Deploys the three fine-tuned + quantized winners as a multi-agent system, measures concurrency, computes Pareto frontier.

```bash
./scripts/run_exp_e2e.sh
./scripts/run_exp_e2e.sh --kv-cache-dtype turboquant
```

---

## KV Cache Quantization Methods

| Method | Config | Key property |
|--------|--------|-------------|
| FP8 E4M3 | `configs/quantization/fp8.yaml` | Native vLLM, no calibration |
| KIVI 2/4-bit | `kivi.yaml` | Asymmetric per-channel/token, no calibration |
| KVQuant | `kvquant.yaml` | NUQ codebooks, calibration required |
| AWQ-INT4 + FP8 KV | `awq_fp8.yaml` | Weight + KV quantization |
| **TurboQuant** | `turboquant.yaml` | Beta codebook + QR rotation + QJL residual, Triton kernels |
| **RotorQuant** | `rotorquant.yaml` | Cl(3,0) rotor sandwich rotation, 10–19× faster than dense |

Expected concurrency at 4096-token context on H100 80 GB:

| Method | Concurrent sessions |
|--------|-------------------|
| BF16 | ~175 |
| FP8 | ~350 |
| TurboQuant 3-bit | ~925 |

---

## Training

Fine-tuning uses [Unsloth](https://github.com/unslothai/unsloth) for 2× speed and 70% less VRAM vs standard PEFT/TRL.

### Two-stage pipeline

```
SFT (format + domain adaptation)  →  GRPO RL (task metric optimisation)
```

### GRPO reward functions

| Category | Reward components |
|----------|-------------------|
| Cat A (business logic) | state transition acc (0.30) + tool-call F1 (0.30) + chain propagation (0.20) + format (0.10) + task completion (0.10) |
| Cat B (specialist) | tool-call F1 (0.40) + slot extraction (0.30) + state sequence (0.20) + format (0.10) |
| Cat C (graph extraction) | node F1 (0.35) + edge F1 (0.35) + structural validity (0.10) + GED (0.10) + JSON validity (0.10) |

Reward hacking is mitigated by held-out evaluation every 50 steps with auto-stop if the held-out metric drops while training reward increases.

---

## Evaluation Targets

| Metric | Target |
|--------|--------|
| State transition accuracy | ≥ 85% |
| Task completion rate | ≥ 70% |
| Tool-call F1 | ≥ 85% |
| Invalid transition rate | ≤ 5% |
| Graph Node F1 | ≥ 85% |
| Graph Edge F1 | ≥ 75% |
| Normalised GED | ≤ 0.20 |
| Full workflow success rate | ≥ 55% |

---

## Key Architecture Decisions

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-001 | Benchmark-first model selection | Avoids wasting compute fine-tuning sub-optimal models |
| ADR-002 | Unsloth over standard PEFT/TRL | 2× speed, 70% less VRAM, native GRPO + MoE kernels |
| ADR-003 | SFT then GRPO two-stage training | SFT establishes format; GRPO optimises task metrics |
| ADR-004 | Shared SFT base for dual-category winners | Diverge only at GRPO stage (Risk R2) |
| ADR-005 | Triton for custom quantization kernels | Fused encode/decode in PagedAttention cache path |

---

## Known Risks & Mitigations

| Risk | Description | Mitigation |
|------|-------------|-----------|
| R1 | Qwen3.5-35B-A3B BF16 ~70 GB | Unsloth QLoRA 4-bit (~17.5 GB) + FP8 RL |
| R2 | Same model wins Cat B + Cat C | Share SFT base, diverge at GRPO |
| R3 | Phase 1 winner doesn't respond to fine-tuning | 100-step pilot SFT on top-2 (`training/pilot_check.py`) |
| R4 | TurboQuant PR not merged upstream | Use `0xSero/turboquant` fork |
| R5 | GRPO reward hacking | Held-out eval every 50 steps, KL monitoring, auto-stop |
| R6 | Nemotron Mamba + vLLM incompatibility | HF `generate()` fallback in `task_runner.py` |
| R7 | GLM LoRA VRAM overflow | Auto-reduce LoRA rank 64 → 32 |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Training | PyTorch ≥ 2.4, Unsloth ≥ 2025.3, TRL ≥ 0.15, PEFT ≥ 0.14 |
| Inference | vLLM ≥ 0.8 (PagedAttention v2) |
| Quantization kernels | Triton ≥ 3.0 |
| Constrained decoding | Outlines / XGrammar |
| Experiment tracking | Weights & Biases |
| Data & config | HF Datasets, Pydantic, PyYAML, structlog |

---

## License

Apache 2.0
