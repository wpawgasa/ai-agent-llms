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
| Gemma 4 26B-A4B-IT | 26B | 4B | MoE GQA | 128K | `gemma` | ~52 GB |
| Gemma 4 26B-A4B-IT (FP8) | 26B | 4B | MoE GQA | 128K | `gemma` | ~26 GB |
| Gemma 4 31B-IT | 31B | 31B | Dense GQA | 128K | `gemma` | ~62 GB |
| Gemma 4 31B-IT (FP8) | 31B | 31B | Dense GQA | 128K | `gemma` | ~31 GB |
| Qwen3.6-35B-A3B | 35B | 3B | DeltaNet MoE | 262K | `qwen3_coder` | ~70 GB |
| Qwen3.6-35B-A3B (FP8) | 35B | 3B | DeltaNet MoE | 262K | `qwen3_coder` | ~35 GB |
| Qwen3.6-27B | 27B | 27B | Dense GQA | 262K | `qwen3_coder` | ~54 GB |
| Qwen3.6-27B (FP8) | 27B | 27B | Dense GQA | 262K | `qwen3_coder` | ~27 GB |

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
│   ├── models/cat_a/          # 14 × Category A model configs
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
│   ├── filter_grpo_data.py          # GRPO prompts: L3–L5 filter over cleaned SFT splits
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

## Serving Backends

The project supports three local serving backends. Every backend exposes an OpenAI-compatible API on port 8000, so all eval scripts work unchanged across them.

| Backend | Venv | Installer | Launcher |
|---------|------|-----------|----------|
| vLLM 0.20 | `.venv-infer` | `scripts/install_infer.sh` | `serving/launch_vllm.sh` |
| SGLang 0.5.11 | `.venv-sglang` | `scripts/install_sglang.sh` | `serving/launch_sglang.sh` |
| TensorRT-LLM 1.2.1 | `.venv-trtllm` | `scripts/install_trtllm.sh` | `serving/launch_trtllm.sh` |

The venvs are **mutually exclusive** (torch pin conflicts). Never activate two at once.

`serving/launch.sh` is the unified dispatcher — it reads `serving.engine` from the model YAML and execs the right backend launcher automatically.

### Quick start

```bash
# Install a backend (one-time)
./scripts/install_sglang.sh      # or install_trtllm.sh

# Start the server using the dispatcher
./serving/launch.sh configs/models_exp_bc/qwen25_3b_sglang.yaml

# Run the benchmark (engine is read from the YAML)
source .venv-sglang/bin/activate
python -m llm_workflow_agents.eval.agent_benchmark \
    --model Qwen/Qwen2.5-3B-Instruct --engine sglang \
    --output results/smoke_sglang.json

# Or use the experiment runner with --backend
./scripts/run_exp_a.sh --backend sglang --max-samples 50
```

For TensorRT-LLM, the first launch performs JIT compilation from HF weights (~5–15 min). Subsequent launches reuse the cached engine. The health-poll budget is automatically extended to 30 min.

### Per-backend model YAMLs

Each model has per-backend YAML variants following the naming convention `<model>_<backend>.yaml`:

```
configs/models_exp_a/qwen3_32b_sglang.yaml
configs/models_exp_a/qwen3_32b_trtllm.yaml
configs/models_exp_bc/qwen25_3b_sglang.yaml
configs/models_exp_bc/qwen25_3b_trtllm.yaml
```

### Caveats

**TensorRT-LLM devcontainer:** `.devcontainer/Dockerfile.tensorrt` is the recommended way to run TRT-LLM. It is based on the official NVIDIA TRT-LLM NGC image (`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`), which pre-installs CUDA 13.1, `libopenmpi-dev`, and `tensorrt_llm` itself. Swap `"dockerfile": "Dockerfile.tensorrt"` in `devcontainer.json` to use it. An NGC account is required to pull the image — generate a free API key at https://ngc.nvidia.com/setup/api-key, then `docker login nvcr.io`.

**TensorRT-LLM bare-metal apt prerequisite:** If not using the devcontainer, `sudo apt-get install -y libopenmpi-dev` must be installed before running `install_trtllm.sh`. The script checks for it and exits with a hint if it's missing.

**KV-cache quantization vocabulary:** The vLLM-only quant strings (`turboquant_*`, `rotorquant_*`, `kivi_*`, `kvquant*`) are rejected by `launch_sglang.sh` and `launch_trtllm.sh` with a clear error. Phase 3 KV-quant benchmarks (`run_exp_d.sh`) are intentionally vLLM-only.

**SGLang tool-call parsers:** SGLang uses different parser names than vLLM. The per-backend YAMLs already carry the correct value — do not copy `tool_call_parser` between vLLM and SGLang configs.

| vLLM parser | SGLang parser |
|-------------|---------------|
| `hermes`, `qwen3_coder` | `qwen25` |
| `mistral` | `mistral` |
| `gemma`, `gemma4`, `glm4`, `nemotron` | `pythonic` |

**Nemotron-3-Nano (Mamba hybrid):** Not supported by SGLang or TRT-LLM in their current versions. The YAMLs carry `serving.skip_reason` and the runner shells skip them automatically.

---

## Setup

**Requirements:** Python 3.11+, CUDA 13.0, NVIDIA H100 (80 GB recommended)

### Venv layout

All serving backends and training pin mutually-incompatible versions of `torch`, so the project ships with **four separate virtualenvs**:

| Venv | Purpose | Key pins |
|------|---------|----------|
| `.venv-train` | Phase 2 SFT + GRPO (Unsloth) | torch 2.10.0+cu130, vllm 0.19.1+cu130, transformers 4.57.6, trl 0.24.0, unsloth 2026.4.x |
| `.venv-infer` | Phase 1/3/4 vLLM serving + benchmarks | torch 2.11.0+cu130, vllm 0.20.0, transformers 5.6.2 |
| `.venv-sglang` | SGLang serving + benchmarks | torch 2.11.0+cu130, sglang 0.5.11, transformers 5.6.0+ |
| `.venv-trtllm` | TensorRT-LLM serving + benchmarks | torch 2.10.0+cu130, tensorrt_llm 1.2.1 |

Each venv is bootstrapped by its own installer script:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Bootstrap the training environment (.venv-train)
./scripts/install_train.sh

# Bootstrap the inference / serving environment (.venv-infer, vLLM)
./scripts/install_infer.sh

# Bootstrap the SGLang serving environment (.venv-sglang)
./scripts/install_sglang.sh

# Bootstrap the TensorRT-LLM serving environment (.venv-trtllm)
# Requires: sudo apt-get install -y libopenmpi-dev
./scripts/install_trtllm.sh
```

The installers call `uv venv` + `uv pip install -e ".[train,dev]"` (or
`".[infer,dev]"`) using the corresponding extras defined in
`pyproject.toml`. `scripts/install_train.sh` additionally pulls the
`vllm-0.19.1+cu130` wheel from the GitHub release (PyPI only ships the
cu129 build).

#### Unsloth pre-built container (`Dockerfile.unsloth`)

When using `.devcontainer/Dockerfile.unsloth` (based on `unsloth/unsloth`),
the container already ships torch, Unsloth, vLLM, and trl at the system level.
Use `install_train_unsloth.sh` instead — it layers the project on top without
re-resolving or clobbering those pre-installed versions:

```bash
./scripts/install_train_unsloth.sh
```

What it does:

1. Creates `.venv-train` with `--system-site-packages` so the container's
   torch / Unsloth / vLLM are visible without redundant downloads.
2. Freezes the container packages as uv constraints (all versions locked).
3. Installs the project `src/` in editable mode (`--no-deps`) so that
   `from llm_workflow_agents import …` works without re-resolving the training
   stack.
4. Installs the project's base dependencies and dev tools under those
   constraints; packages already in the container are kept at their
   container-pinned versions.
5. Upgrades `transformers` to `>=5.6.0` inside the venv (required for
   Gemma-4 and Qwen3.6 support; overrides the Unsloth-pinned version for any
   process that activates this venv).
6. Installs `dvc[gs]` for GCS-backed data versioning.

### Activating

```bash
# For training, data generation, evaluation runs that import unsloth:
source .venv-train/bin/activate

# For vLLM-backed serving and the Phase 3 quantization benchmarks:
source .venv-infer/bin/activate

# For SGLang-backed serving:
source .venv-sglang/bin/activate

# For TensorRT-LLM-backed serving:
source .venv-trtllm/bin/activate
```

The runner scripts pick the right venv automatically:
- `scripts/run_phase2_sft.sh` sources `.venv-train`.
- `serving/launch_vllm.sh` sources `.venv-infer` if no venv is currently active.
- `serving/launch_sglang.sh` sources `.venv-sglang` if no venv is currently active.
- `serving/launch_trtllm.sh` sources `.venv-trtllm` if no venv is currently active.

### Locked requirements files

If you prefer plain `pip` over `uv`, `requirements-train.txt` and
`requirements-infer.txt` mirror the two extras (the cu130 vLLM wheel
URL still has to be installed manually for training).

### Notes

- **`flash-attn` is not in the `infer` extras.** Its sdist build fails on CUDA 13 hosts under uv's build isolation (cu129 torch gets pulled into the build env). vLLM 0.20.0 ships its own FlashAttention path, so this is optional. To install manually after `install_infer.sh`:
  ```bash
  source .venv-infer/bin/activate
  uv pip install flash-attn --no-build-isolation
  ```
- **vLLM 0.20.0 has no published cu130 wheel.** The cu129 build runs on CUDA 13 via forward compatibility.
- **Cache permission errors** on shared cloud nodes (Shadeform etc.):
  ```bash
  UV_CACHE_DIR=/tmp/uv-cache ./scripts/install_train.sh
  ```
- **Using `.devcontainer/Dockerfile.unsloth` as the base image.** The image
  pre-installs a working Unsloth + torch + vLLM stack. Running
  `uv pip install -e ".[dev]"` would re-resolve those pins and clobber the
  pre-installed versions. Install with `--no-deps` instead, then add the dev
  tools individually (they don't overlap on torch/transformers/vLLM):
  ```bash
  uv pip install -e . --no-deps
  uv pip install --no-deps pytest pytest-cov pytest-asyncio ruff mypy 'dvc[gs]'
  ```
  For a stricter guarantee, freeze the existing env first and pass it as
  constraints — uv will resolve `[dev]` normally but cannot bump anything
  already installed:
  ```bash
  uv pip freeze > /tmp/constraints.txt
  uv pip install -e ".[dev]" -c /tmp/constraints.txt
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

# GRPO training prompts — L3–L5 subset of the cleaned SFT splits (no new generation)
python scripts/filter_grpo_data.py

# Validation + test splits
OPENAI_API_KEY=... ./scripts/generate_eval_data.sh
```

Each script accepts `--dry-run` to preview commands without executing, and `--output-dir` to redirect output.

### Data Design

| Split | Samples | Seed | Behavior preset | Languages |
|-------|---------|------|-----------------|-----------|
| Benchmark | 1 000 | 100 | default | mixed (en/th) |
| SFT | ~12 504 | 42 | **adversarial** | en + th + code-switch |
| GRPO | L3–L5 of cleaned SFT splits | — | inherits SFT | inherits SFT |
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

## DVC Data Pipeline

Data outputs are versioned with [DVC](https://dvc.org) and stored on GCS. Use `dvc pull` to restore any cached dataset without re-running generation scripts.

**Remote:** `gs://looloo-voicebot-llm-weights-and-data/llm-workflow-agents`

### Authentication

The remote uses a GCP service account key. Place the JSON key one level above the project root:

```
/workspaces/looloo-ocr-9e0b69945c03.json   ← expected default path
```

Or override at runtime:

```bash
# Option A: override in DVC config
dvc remote modify gcs credentialpath /path/to/key.json

# Option B: standard GCP env var
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

### Pulling data

```bash
# Pull all tracked outputs
dvc pull

# Pull a specific stage output
dvc pull data/output/benchmark/task_a
```

### Pipeline stages

| Stage | Output | Samples | API key needed |
|-------|--------|---------|----------------|
| `task_a_benchmark` | `data/output/benchmark/task_a` | 1 000 | No |
| `task_a_sft` | `data/output/sft/task_a` | ~12 504 | `OPENAI_API_KEY`, `GEMINI_API_KEY` |
| `task_a_grpo` | `data/output/grpo/task_a` | 2 250 | `OPENAI_API_KEY`, `GEMINI_API_KEY` |
| `task_a_eval` | `data/output/val/task_a`, `data/output/test/task_a` | 500 + 500 | `OPENAI_API_KEY` |

### Reproducing a stage

`dvc repro` re-runs a stage only if its dependencies (scripts, templates, config params) have changed, then caches the result:

```bash
dvc repro task_a_benchmark   # no API key required
dvc repro task_a_sft         # requires API keys in .env
dvc repro                    # run all out-of-date stages
```

Check pipeline status without running anything:

```bash
dvc status     # compares local cache vs remote + dep hashes
dvc dag        # print the dependency graph
```

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

### Prerequisites

```bash
# 1. Training venv with Unsloth installed
./scripts/install_train.sh
source .venv-train/bin/activate

# 2. Data: cleaned SFT corpus + deterministic 85/10/5 splits
#    (skip if you've already run the data-generation pipeline)
OPENAI_API_KEY=... GEMINI_API_KEY=... ./scripts/generate_sft_data.sh
python scripts/clean_task_a_sft.py \
    --input-dir data/output/sft/task_a \
    --output-dir data/output/sft/task_a_cleaned
python scripts/split_task_a_sft.py    # → data/output/sft/task_a_splits/

# 3. HF token for gated models (Gemma, Mistral)
export HF_TOKEN=hf_...
```

The SFT and GRPO runners both auto-load `.env` and auto-activate `.venv-train/` if it exists, so steps 1 + 3 only need to be done once.

### SFT — `run_phase2_sft.sh`

Default invocation trains Gemma4-26B-A4B on Task A with the Cat A SFT config:

```bash
./scripts/run_phase2_sft.sh
```

Common variations:

```bash
# Different base model (any YAML in configs/models_exp_a/)
./scripts/run_phase2_sft.sh --model-config configs/models_exp_a/qwen36_27b.yaml

# Different SFT config (Cat B / Cat C use their own data sources)
./scripts/run_phase2_sft.sh --sft-config configs/training/sft_cat_b.yaml

# Smoke test: prepare splits + patched config, exit before training
./scripts/run_phase2_sft.sh --dry-run

# Disable W&B for a local run
./scripts/run_phase2_sft.sh --no-wandb

# Resume after Ctrl+C (auto-picks latest checkpoint)
./scripts/run_phase2_sft.sh --resume

# Resume from a specific checkpoint
./scripts/run_phase2_sft.sh --resume-from checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1500
```

**Outputs:** `checkpoints/sft_cat_a/<HF-model-basename>/checkpoint-N/` every `save_steps` (500 by default) plus `train.log`. Optimizer state, scheduler, RNG, and epoch counter are all restored on resume.

### GRPO — `run_phase2_grpo.sh`

GRPO consumes an SFT checkpoint plus the L3–L5 filtered prompt set. Rewards are recomputed online from policy generations, so no new ground truth is required.

```bash
# Default: auto-picks latest SFT checkpoint, filters SFT splits to L3-L5
./scripts/run_phase2_grpo.sh
```

Common variations:

```bash
# Pin a specific SFT checkpoint
./scripts/run_phase2_grpo.sh \
    --sft-checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-2000

# Use a different complexity-level mix
./scripts/run_phase2_grpo.sh --levels L4 L5

# Use an already-prepared prompt directory, skip the filter step
./scripts/run_phase2_grpo.sh --data-dir data/output/grpo/task_a --skip-filter

# Different GRPO config (Cat B / Cat C reward functions)
./scripts/run_phase2_grpo.sh --grpo-config configs/training/grpo_cat_b.yaml

# Smoke test
./scripts/run_phase2_grpo.sh --dry-run

# Local run without W&B
./scripts/run_phase2_grpo.sh --no-wandb
```

**Outputs:** `checkpoints/grpo_cat_a/<HF-model-basename>/` plus reward curves and held-out scores in the W&B run (and `train.log`).

> **Note:** `training/grpo.py` does not currently expose a resume hook, so re-launching starts a fresh trainer. Held-out evaluation runs every 50 steps; training auto-stops if the held-out metric drops while training reward increases (reward-hacking detector, Risk R5).

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
