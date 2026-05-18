# Codebase Specification: Benchmark-First LLM Workflow Agents with SFT+RL Fine-Tuning

**Project:** `llm-workflow-agents` (package: `llm_workflow_agents`)
**Version:** 4.1 — May 2026 (as-built)
**Hardware Target:** Single NVIDIA H100 SXM 80GB
**Training Framework:** Unsloth (SFT + GRPO RL), with TRL fallback for Gemma-4 MoE LoRA

---

## 1. System Overview

### 1.1 Core Thesis

This codebase implements a **benchmark-first, fine-tune-selectively** pipeline for workflow-orchestrating LLM agents. Instead of fine-tuning all candidate models, we:

1. **Benchmark** all pre-trained candidates across three task categories
2. **Select** the single best model per category
3. **Fine-tune** each winner with SFT then GRPO reinforcement learning
4. **Deploy** with aggressive KV cache quantization on a single H100

The final deliverable is **three fine-tuned specialist models** — one per task — served concurrently via vLLM (or SGLang / TensorRT-LLM) with FP8 KV by default and TurboQuant variants for the high-concurrency frontier.

### 1.2 Four-Phase Pipeline

```
╔══════════════╗     ╔══════════════╗     ╔══════════════╗     ╔══════════════╗
║   PHASE 1    ║     ║   PHASE 2    ║     ║   PHASE 3    ║     ║   PHASE 4    ║
║  Benchmark   ║────▸║  SFT + RL    ║────▸║  KV Cache    ║────▸║  Integration ║
║  Pre-trained ║     ║  Fine-Tune   ║     ║  Quant       ║     ║  & Pareto    ║
╚══════╤═══════╝     ╚══════╤═══════╝     ╚══════╤═══════╝     ╚══════╤═══════╝
       │                     │                     │                     │
  15 candidates         3 winners            6 quant methods      Pareto-optimal
  → rank & select       → SFT → GRPO        → quality/perf       deployment config
                                              matrix
```

### 1.3 Research Questions → Code Paths

All code lives under the `llm_workflow_agents` Python package (`src/llm_workflow_agents/...`).

| RQ | Description | Primary Code Path |
|----|-------------|-------------------|
| RQ1 | Best 15–35B for prompt-encoded logic; SFT+RL improvement | `eval/agent_benchmark.py` → `training/sft.py` → `training/grpo.py` → `eval/` |
| RQ2 | Best 2–5B specialist; GRPO vs SFT-only on tool F1 | `eval/agent_benchmark.py` → `training/sft.py` → `training/grpo.py` → `eval/tool_call_f1.py` |
| RQ3 | Graph extraction with ≥85% Node F1 / ≥75% Edge F1 | `data/generate_graph_pairs.py` → `training/sft.py` → `training/grpo.py` → `eval/graph_extraction_eval.py` |
| RQ4 | TurboQuant/RotorQuant vs FP8/KIVI/KVQuant for multi-agent concurrency | `quantization/` + upstream `vllm.v1.attention.backends.turboquant_attn` → `eval/quant_benchmark.py` + `eval/concurrency_benchmark.py` → `analysis/pareto.py` |
| RQ5 (new) | Orchestrator routing accuracy (intent → specialist selection) | `data/generate_orchestrator_data.py` → SFT/GRPO → `eval/intent_classification.py` |

### 1.4 Model Inventory

**Category A — Prompt-Encoded Business Logic (15–35B)** — 14 YAMLs at `configs/models/cat_a/`:

| Model | Total | Active | Architecture | Context | Tool Parser | BF16 VRAM |
|-------|-------|--------|-------------|---------|-------------|-----------|
| Gemma 3 27B-IT | 27B | 27B | Dense GQA | 128K | `gemma` | ~54 GB |
| Qwen3-32B | 32B | 32B | Dense + `<think>` | 128K | `hermes` | ~64 GB |
| Qwen3.5-35B-A3B | 35B | 3B | DeltaNet + MoE | 262K | `qwen3_coder` | ~70 GB |
| Mistral Small 3.1 24B | 24B | 24B | Dense, sliding-win | 128K | `mistral` | ~48 GB |
| Nemotron-3-Nano 30B | 30B | 3.6B | MoE + Mamba-2 | 1M | `nemotron` | ~60 GB |
| GLM-4.7-Flash | 30B | 3.6B | MoE + MLA | 200K | `glm4` | ~60 GB |
| Gemma 4 26B-A4B-IT | 26B | 4B | MoE GQA | 128K | `gemma` | ~52 GB |
| Gemma 4 26B-A4B-FP8 | 26B | 4B | MoE GQA (FP8) | 128K | `gemma` | ~26 GB |
| Gemma 4 31B-IT | 31B | 31B | Dense GQA | 128K | `gemma` | ~62 GB |
| Gemma 4 31B-FP8 | 31B | 31B | Dense GQA (FP8) | 128K | `gemma` | ~31 GB |
| Qwen3.6-35B-A3B | 35B | 3B | DeltaNet + MoE | 262K | `qwen3_coder` | ~70 GB |
| Qwen3.6-35B-A3B-FP8 | 35B | 3B | DeltaNet + MoE | 262K | `qwen3_coder` | ~35 GB |
| Qwen3.6-27B | 27B | 27B | Dense GQA | 262K | `qwen3_coder` | ~54 GB |
| Qwen3.6-27B-FP8 | 27B | 27B | Dense GQA | 262K | `qwen3_coder` | ~27 GB |

**Category B–C — Specialist Subagent & Graph Extraction (2–5B)** — 7 YAMLs at `configs/models/cat_bc/`:

| Model | Params | Active | Architecture | Context | VRAM | Unsloth |
|-------|--------|--------|-------------|---------|------|---------|
| Qwen2.5-3B-Instruct | 3B | 3B | Dense GQA | 32K | ~6 GB | Full (SFT+RL) |
| Qwen3.5-4B | 4B | ~3B | DeltaNet + MoE | 262K | ~8 GB | Full (SFT+RL) |
| GLM-4.7-Flash | 30B | 3.6B | MoE + MLA | 200K | ~60 GB | Full (SFT+RL) |
| Gemma-2B | 2.5B | 2.5B | Dense MQA | 8K | ~5 GB | Full (SFT+RL) |
| Gemma-3-4B-it | 4B | 4B | Dense GQA | 128K | ~8 GB | Full (SFT+RL) |
| Gemma-4-E4B-it | 4B | 4B | Dense GQA | 128K | ~8 GB | Full (SFT+RL) |
| Gemma-4-E2B-it | 2B | 2B | Dense GQA | 128K | ~4 GB | Full (SFT+RL) |

**Orchestrator routing dataset** (new in v4.1) feeds the Cat A winner with intent→specialist routing supervision so it can play the orchestrator role in Phase 4 multi-agent deployments. See `data/generate_orchestrator_data.py` (§4.4) and `eval/intent_classification.py` (§8.7).

---

## 2. Repository Structure

All Python code lives under `src/llm_workflow_agents/`. Top-level directories (`data/`, `configs/`, `serving/`, `scripts/`, `tests/`, etc.) hold non-Python artifacts. Earlier drafts of this spec described a flat top-level layout (`data/`, `training/`, `eval/`, …); those names are now Python sub-packages.

```
ai-agent-llms/
│
├── src/llm_workflow_agents/                  # Python package
│   ├── __init__.py
│   ├── vllm_plugin.py                        # vLLM general_plugins entry — see §7.5
│   ├── data/                                 # Data generation (§4)
│   │   ├── generate_workflows.py             # Task A (workflow conversations, L1–L5)
│   │   ├── generate_tool_call_data.py        # Task B (specialist tool-call data)
│   │   ├── generate_graph_pairs.py           # Task C ((prompt, graph) pairs)
│   │   ├── generate_orchestrator_data.py     # NEW: Cat A orchestrator routing data
│   │   ├── chat_template_converter.py
│   │   ├── data_validator.py
│   │   ├── domain_registry.py                # 18 call-center domains
│   │   ├── system_prompt.py                  # build_enriched_system_prompt + FORMAT_RULES
│   │   ├── _teacher_client.py                # OpenAI/Anthropic/Gemini dispatch
│   │   └── _workflow_script.py               # Human-readable workflow script builder
│   ├── training/                             # Phase 2 (§6)
│   │   ├── sft.py                            # Unsloth SFT (with TRL fallback for Gemma-4)
│   │   ├── grpo.py                           # Unsloth GRPO RL
│   │   ├── rewards/
│   │   │   ├── reward_business_logic.py      # Cat A
│   │   │   ├── reward_subagent.py            # Cat B
│   │   │   └── reward_graph_extraction.py    # Cat C
│   │   ├── reward_utils.py
│   │   ├── lora_targets.py                   # 16-model LoRA registry
│   │   ├── merge_adapter.py
│   │   ├── pilot_check.py                    # Risk R3
│   │   ├── _utils.py                         # Shared TrainingArguments builder
│   │   ├── train_specialist.py               # v2 backward-compat entry
│   │   └── train_graph_extractor.py          # v2 backward-compat entry
│   ├── eval/                                 # Evaluation (§8)
│   │   ├── agent_benchmark.py                # Phase 1 (Exp A) entry point
│   │   ├── state_accuracy.py
│   │   ├── tool_call_f1.py
│   │   ├── tool_chain_propagation.py
│   │   ├── graph_extraction_eval.py
│   │   ├── composite_score.py
│   │   ├── perplexity.py
│   │   ├── longbench.py
│   │   ├── needle_haystack.py
│   │   ├── quant_benchmark.py                # Phase 3 quality/perf matrix
│   │   ├── concurrency_benchmark.py          # NEW: throughput/latency at concurrency sweep
│   │   ├── intent_classification.py          # NEW: orchestrator routing eval
│   │   └── constrained_decoding.py           # NEW: Outlines/XGrammar integration
│   ├── quantization/                         # Phase 3 (§7)
│   │   ├── turboquant/                       # codebook.py, rotation.py, triton_kernels.py, vllm_integration.py
│   │   ├── rotorquant/                       # clifford.py, rotor_kernels.py, vllm_integration.py
│   │   └── baselines/                        # kivi_cache.py, kvquant_calibrate.py
│   ├── serving/                              # vLLM/SGLang/TRT-LLM launchers + Phase 4 (§9)
│   │   ├── orchestrator.py                   # MultiAgentOrchestrator
│   │   ├── benchmark_e2e.py                  # E2E concurrency + latency
│   │   ├── launch_vllm_turboquant.py         # Custom dtype hook (scaffolding, see §7)
│   │   └── launch_vllm_rotorquant.py         # Custom dtype hook (scaffolding, see §7)
│   ├── analysis/                             # Phase 4 + reporting (§9.3)
│   │   ├── pareto.py
│   │   ├── plot_phase1_rankings.py
│   │   ├── plot_sft_vs_rl.py
│   │   ├── plot_quant_matrix.py
│   │   ├── plot_pareto.py
│   │   └── plot_results.py                   # General results plotter
│   ├── integration/                          # __init__.py only — re-exports from serving/, analysis/
│   ├── config/                               # loader.py, schema.py (ComplexitySpec, UserBehaviorDistribution, …)
│   ├── benchmark/                            # empty (legacy package removed; agent_benchmark.py is canonical)
│   └── utils/                                # Misc helpers
│
├── configs/                                  # All YAML configs (§3)
│   ├── models/
│   │   ├── cat_a/                            # 14 YAMLs (incl. FP8 variants)
│   │   └── cat_bc/                           # 7 YAMLs
│   ├── models_exp_a/, models_exp_bc/         # Mirror layout used by some DVC stages
│   ├── training/                             # sft_cat_{a,b,c}.yaml, grpo_cat_{a,b,c}.yaml
│   ├── quantization/                         # fp8, kivi, kivi_2bit, kivi_4bit, kvquant, awq_fp8, turboquant, rotorquant
│   ├── benchmark/                            # phase1_matrix.yaml, selection_weights.yaml
│   └── serving/                              # single_model, single_lora, multi_agent, multi_instance,
│                                             # benchmark_e2e, benchmark_matrix
│
├── serving/                                  # Top-level shell launchers (used by DVC + scripts)
│   ├── launch.sh                             # Generic launcher dispatcher
│   ├── launch_vllm.sh                        # vLLM (FP8/BF16, TurboQuant variants)
│   ├── launch_sglang.sh                      # SGLang backend
│   ├── launch_trtllm.sh                      # TensorRT-LLM
│   └── _yaml_helper.sh                       # Shared config parsing
│
├── data/                                     # Templates + generated datasets
│   ├── templates/
│   │   ├── workflow_prompt_template.txt
│   │   ├── tool_schemas_L1_to_L5.json
│   │   └── graph_output_schema.json
│   └── output/                               # Generated JSONL (DVC-tracked)
│       ├── benchmark/task_a/                 # 1000 placeholder convs for Phase 1 ranking
│       ├── sft/task_a/                       # 4450 SFT conversations
│       ├── sft/task_a_cleaned/               # 4445 cleaned via scripts/clean_task_a_sft.py
│       ├── grpo/task_a/                      # 2250 L3–L5 GRPO prompts
│       ├── val/task_a/, test/task_a/         # 500 + 500 held-out (seeds 300/400)
│       ├── exp_b/                            # Task B tool-call data
│       └── exp_c/                            # Task C (prompt, graph) pairs
│
├── scripts/                                  # Experiment + utility scripts
│   ├── run_exp_a.sh                          # Phase 1 (Exp A) — see also _single, _per_level variants
│   ├── run_exp_a_single.sh, run_exp_a_per_level.sh
│   ├── run_exp_b.sh, run_exp_c.sh, run_exp_d.sh, run_exp_e2e.sh
│   ├── run_phase2_sft.sh                     # SFT runner (model_config + sft_config)
│   ├── run_concurrency_benchmark.sh
│   ├── generate_benchmark_data.sh            # Task A placeholders (no API)
│   ├── generate_sft_data.sh                  # Task A SFT data (OpenAI + Gemini)
│   ├── split_task_a_sft.py                   # Task A 85/10/5 train/val/test split (seed=42)
│   ├── filter_grpo_data.py                   # Task A GRPO prompts (L3–L5 filter over SFT splits)
│   ├── generate_eval_data.sh                 # val + test
│   ├── generate_data.sh, generate_benchmark_data_teacher.sh
│   ├── install_train.sh, install_train_unsloth.sh, install_train_cu128.sh
│   ├── install_infer.sh, install_sglang.sh, install_trtllm.sh
│   ├── build_trtllm_engines.sh, build_concurrency_report.py
│   ├── clean_task_a_sft.py, concat_task_a.py, patch_model_config.py
│   └── (Note: run_phase2_grpo.sh / run_phase3.sh / run_phase4.sh not yet authored —
│           training currently driven by direct shell invocation or DVC stages.)
│
├── deployments/local/                        # Docker Compose dev env (BiFrost proxy + data dirs)
├── checkpoints/                              # SFT/GRPO adapter outputs (DVC-tracked per stage)
├── results/                                  # Benchmark JSON outputs (exp_a/, concurrency/, …)
├── tests/                                    # Unit + integration tests
├── docs/                                     # This spec + recipe docs + benchmark reports
├── dvc.yaml, dvc.lock                        # DVC pipeline (§10.3)
├── pyproject.toml                            # llm-workflow-agents v0.1.0, Python ≥3.11
├── requirements-train.txt, requirements-infer.txt
├── CLAUDE.md, README.md
```

---

## 3. Configuration Schema

### 3.1 Model Configuration

```yaml
# configs/models/cat_a/qwen36_35b_a3b.yaml  (representative)
model:
  name: "Qwen/Qwen3.6-35B-A3B"
  family: "qwen36"
  architecture: "deltanet_moe"
  params_total: 35_000_000_000
  params_active: 3_000_000_000
  num_experts_total: 256
  num_experts_active: 9                    # 8 routed + 1 shared
  context_length: 262144
  precision: "bfloat16"
  vram_estimate_gb: 70
  mtp_enabled: true                        # Multi-Token Prediction for spec decode

serving:
  engine: "vllm"                           # also accepts "sglang", "trtllm"
  tool_call_parser: "qwen3_coder"
  chat_template: "qwen3.6"
  gpu_memory_utilization: 0.90
  max_model_len: 8192
  enforce_eager: false                     # Auto-overridden to true for Gemma-4 + turboquant (see §7 / R8)

category: "A"
benchmark_tasks: ["task_a", "task_a_router"]   # task_a_router added in v4.1 for orchestrator routing
```

### 3.2 SFT Training Configuration

```yaml
# configs/training/sft_cat_a.yaml
stage: "sft"
framework: "unsloth"                         # "trl" for Gemma-4 MoE LoRA fallback

model:
  config_ref: null                           # Populated at runtime from Phase 1 winner

lora:
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules: "auto"                     # Resolved via lora_targets.py registry (16 models)
  freeze_router: true                        # MoE router frozen

training:
  precision: "qlora_4bit"                    # QLoRA for MoE / BF16 for dense ≤8GB
  learning_rate: 1.0e-4
  lr_scheduler: "cosine"
  lr_end: 5.0e-5
  warmup_ratio: 0.05
  effective_batch_size: 16
  gradient_accumulation_steps: 4
  num_epochs: 3
  max_seq_length: 4096
  gradient_checkpointing: true
  packing: true                              # Unsloth ConstantLengthDataset / FFD packing

data:
  source: "data/output/sft/task_a_splits"   # Pre-split via scripts/split_task_a_sft.py
  format: "chatml"
  splits:
    train: "train.jsonl"        # ~4 414 conversations
    val:   "validation.jsonl"   # ~519
    test:  "test.jsonl"         # ~261 — final eval only

logging:
  wandb_project: "llm-workflow-agents"
  wandb_run_prefix: "sft-cat-a"
  save_strategy: "steps"
  save_steps: 500
  eval_steps: 500
  metric_for_best_model: "eval_loss"
```

The per-model chat template is injected at SFT time by `chat_template_converter.py`. System prompts are **idempotently re-enriched** at training time via `data/system_prompt.py::build_enriched_system_prompt` — a safety net for legacy datasets; on new data the call is a no-op.

### 3.3 GRPO RL Configuration

```yaml
# configs/training/grpo_cat_a.yaml
stage: "grpo"
framework: "unsloth"

model:
  base_checkpoint: null                      # Populated: path to best SFT checkpoint

grpo:
  algorithm: "GRPO"
  normalization: "DAPO"                      # Removes length bias
  num_generations: 4                         # Completions per prompt for group comparison
  beta: 0.04                                 # KL penalty: constrain near SFT policy
  epsilon: 0.2                               # PPO-style clipping
  learning_rate: 5.0e-6
  lr_scheduler: "constant"
  training_steps: 1000
  per_device_batch_size: 2
  gradient_accumulation_steps: 4
  max_completion_length: 2048

  generation_backend: "vllm"                 # 11× faster RL inference
  fp8_rl: true                               # H100: 1.4× faster, 60% less VRAM

reward:
  function: "reward_business_logic"          # From training/rewards/

data:
  prompts_source: "data/output/grpo/task_a/train.jsonl"        # L3-L5 filter of SFT train split
  ground_truth_source: "data/output/grpo/task_a/train.jsonl"

monitoring:
  wandb_run_prefix: "grpo-cat-a"
  eval_held_out_every: 50                    # Steps between held-out eval
  reward_hacking_detector: true              # TRL callback in grpo.py — see §6.2
  kl_divergence_log: true
```

### 3.4 Benchmark Selection Weights

```yaml
# configs/benchmark/selection_weights.yaml
category_a:
  response_quality: 0.40                     # State transition acc + tool call F1
  latency_p95: 0.25
  throughput_tok_s: 0.20
  memory_peak_vram: 0.15

category_b:
  response_quality: 0.35
  latency_p95: 0.30
  throughput_tok_s: 0.20
  memory_peak_vram: 0.15

category_c:
  response_quality: 0.40
  latency_p95: 0.20
  throughput_tok_s: 0.20
  memory_peak_vram: 0.20
```

### 3.5 Quantization Configuration

`configs/quantization/` contains 8 YAMLs: `fp8.yaml`, `kivi.yaml`, `kivi_2bit.yaml`, `kivi_4bit.yaml`, `kvquant.yaml`, `awq_fp8.yaml`, `turboquant.yaml`, `rotorquant.yaml`.

**TurboQuant integration note (ADR-006, see §11):** the project's `turboquant.yaml` (`kv_cache_dtype: "turboquant"`) routes through `serving/launch_vllm_turboquant.py` for legacy/dev paths only. Production benchmark runs use **upstream vLLM v1 variants** directly: `turboquant_3bit_nc`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_k8v4`. Same for RotorQuant — its custom backend is provisional pending a microbenchmark gate (see §7).

```yaml
# configs/quantization/turboquant.yaml (legacy/dev path)
method:
  name: "turboquant"
  paper: "Zandieh et al., ICLR 2026"
  upstream_variants: ["turboquant_3bit_nc", "turboquant_4bit_nc",
                      "turboquant_k3v4_nc", "turboquant_k8v4"]
  status: "Project custom backend is scaffolding; prefer upstream variants for Phase 3."

# ... codebook, rotation, residual, triton_kernels, vllm_integration sections unchanged ...
```

### 3.6 Serving Configurations

`configs/serving/` contains 6 YAMLs:
- `single_model.yaml` — single base model
- `single_lora.yaml` — single base + LoRA adapter
- `multi_agent.yaml` — orchestrator + specialist + visualizer
- `multi_instance.yaml` — multiple model instances on a single H100
- `benchmark_e2e.yaml` — full Phase 4 E2E sweep
- `benchmark_matrix.yaml` — Phase 3 quantization matrix sweep

---

## 4. Data Generation Layer (`src/llm_workflow_agents/data/`)

### 4.1 `generate_workflows.py` — Task A Data

**Purpose:** Generate multi-turn conversation datasets at 5 complexity levels with state-machine annotations and tool-calling ground truth.

```python
# ComplexitySpec lives in src/llm_workflow_agents/config/schema.py
@dataclass
class ComplexitySpec:
    level: str
    num_states: tuple[int, int]
    branching_factor: tuple[int, int]
    num_tools: int
    chain_depth: int
    nesting_depth: int
    num_samples: int
    domain: str | None

COMPLEXITY_SPECS = {
    "L1": ComplexitySpec("L1", (3,4),   (1,2), 1, 0, 0, 200, None),
    "L2": ComplexitySpec("L2", (5,7),   (2,3), 2, 1, 1, 200, None),
    "L3": ComplexitySpec("L3", (8,12),  (3,5), 4, 2, 2, 200, None),
    "L4": ComplexitySpec("L4", (13,20), (5,8), 6, 3, 3, 200, None),
    "L5": ComplexitySpec("L5", (21,30), (8,99),7, 4, 4, 200, None),
}

# Default behavior mix (`behavior_preset="default"`)
USER_BEHAVIOR_DISTRIBUTION = {
    "cooperative":         0.60,
    "adversarial_probing": 0.15,
    "digressing":          0.10,
    "invalid_tool_inputs": 0.15,
}

TOOL_ERROR_RATE = 0.20  # 20% of tool calls return error payloads

# Other presets in generate_workflows.py: "adversarial" (25/25/15/15 cooperative-light),
# "balanced" (25/25/25/25), "cooperative_only" (100/0/0/0)

def generate_workflow_dataset(
    complexity_level: Literal["L1","L2","L3","L4","L5"],
    num_samples: int = 200,
    teacher_model: str | None = None,           # e.g. "gemini-3-flash", "gemini-3.1-flash-lite",
                                                # "gpt-5.4-nano", "gpt-4o"; None → local placeholder
    output_dir: Path = Path("data/output/benchmark/task_a"),
    seed: int = 42,
    domain: str | None = None,                  # None → uniform draw from domain_registry
    language: str | None = None,                # None → English; "th" → Thai; "code_switch" → mixed
    behavior_preset: str = "default",
    rich_prompt_rate: float = 0.30,             # 30% teacher-authored natural-language sysprompts
    intent_category_preset: str = "default",
) -> DatasetMetadata:
    """
    Each conversation includes:
      - System prompt — assembled by `data/system_prompt.py::build_enriched_system_prompt`,
        which **all data paths (benchmark, SFT, GRPO, eval) run through before training or
        scoring**. The enricher is idempotent: the helper is called both at generation time
        and (as a safety net) at training/inference time.
        Every enriched prompt contains, in order:
          1. Role line — either the bare one-liner (default) or a teacher-authored
             natural-language voicebot description (persona / GOAL / per-state blocks).
             With `rich_prompt_rate=0.30`, ~30% of teacher-generated samples carry the
             rich natural-language description; the other ~70% use the bare role.
          2. **Workflow script** — a structured-format description of the state machine
             (states, transitions, conditions, tools per state, entry actions), produced
             by `_workflow_script.build_workflow_script` / `_graph_to_script`. **Appended
             to every sample regardless of which role-line variant was used** — the rich
             natural-language description never replaces this structured reference.
          3. Tool list + structured reference (initial state, terminal states, tool names).
          4. 7-rule output format guide (`FORMAT_RULES` in `system_prompt.py`).
      - User/assistant turns with [STATE: X → Y] annotations
      - <tool_call>{...}</tool_call> blocks (Hermes/OpenAI tools schema)
      - Tool response turns (80% success, 20% error)
      - Multi-step tool chains (output of tool N → input of tool N+1)

    Output JSONL is written to {output_dir}/{level}_{lang}_{teacher_model}_merged.jsonl
    (or _placeholder if no teacher). The placeholder (no-teacher) path ignores
    `rich_prompt_rate` and always uses the bare role line; the structured workflow
    script is still appended.
    """
```

**Domains decoupled from complexity.** `data/domain_registry.py` defines **18 domains** (`account_management`, `billing_payments`, `order_management`, `technical_support`, `product_info`, `healthcare`, `banking`, `telecom`, `utilities`, `travel`, `ecommerce`, `government`, `insurance`, `complaints`, `scheduling`, `sales`, `surveys`, `emergency`). Each carries tools, state templates, intents, and entity slots. When `domain=None`, each sample draws uniformly.

**Output format (per sample):**

```json
{
  "conversation_id": "L3_042",
  "complexity_level": "L3",
  "domain": "booking_payment",
  "num_states": 10,
  "num_tools": 4,
  "chain_depth": 2,
  "language": "en",
  "workflow_graph": {
    "states": ["S1", "S2", "..."],
    "state_details": [{"name": "S1", "tools": [...], "entry_actions": [...]}],
    "transitions": [{"from": "S1", "to": "S2", "condition": "...", "priority": 1}],
    "initial": "S1",
    "terminal": ["S9", "S10"]
  },
  "workflow_script": "<human-readable script built by _workflow_script.build_workflow_script>",
  "tool_schemas": [...],
  "messages": [
    {"role": "system", "content": "<enriched workflow prompt>"},
    {"role": "user",   "content": "I need to cancel order #4521"},
    {"role": "assistant",
     "content": "[STATE: GREETING → COLLECT_INFO]\n<tool_call>...</tool_call>"},
    {"role": "tool", "content": "{\"order_id\":\"4521\",\"status\":\"active\"}"}
  ],
  "user_behavior": "cooperative",
  "ground_truth": {
    "state_sequence": [{"from": "GREETING", "to": "COLLECT_INFO"}, ...],
    "tool_calls":     [{"name": "lookup_order", "arguments": {"order_id": "4521"}}],
    "tool_chain_dependencies": [{"from_tool": "lookup_order",
                                  "to_tool":   "cancel_order",
                                  "propagated_field": "order_id"}],
    "terminal_state": "CONFIRM"
  }
}
```

### 4.2 `generate_tool_call_data.py` — Task B Data

```python
def generate_tool_call_dataset(
    external_sources: list[str] | None = None,    # Default: xlam-function-calling-60k + ToolBench
    custom_synthetic_size: int = 15000,
    teacher_model: str | None = None,
    negative_ratio: float = 0.15,
    output_dir: Path = Path("data/output/exp_b"),
    seed: int = 42,
) -> DatasetSplits:
    """
    Merge external + synthetic into unified JSONL in Hermes tool-call schema.
    Negatives (15%): wrong tool 5% / hallucinated 4% / invalid state 3% / error recovery 3%.
    Splits: 85% train / 10% val / 5% test.
    """
```

### 4.3 `generate_graph_pairs.py` — Task C Data

```python
def generate_graph_pairs(
    workflow_prompts_dir: Path = Path("data/output/exp_a"),
    gold_annotations: int = 200,
    teacher_generated: int = 800,
    augmentation_target: int = 5000,
    teacher_model: str | None = None,
    output_dir: Path = Path("data/output/exp_c"),
    seed: int = 42,
) -> DatasetSplits:
    """5000 (prompt, graph) pairs. Splits: 4000 train / 500 val / 500 test."""
```

**Graph output JSON schema** (also in `data/templates/graph_output_schema.json`): `nodes[{id,name,tools,entry_actions}]`, `edges[{from,to,condition,priority}]`, `initial_state`, `terminal_states`.

### 4.4 `generate_orchestrator_data.py` — Cat A Orchestrator Routing Data (NEW v4.1)

```python
ROUTING_TARGETS = ("tool_execution", "graph_extraction", "self_handle")

def generate_orchestrator_dataset(
    num_samples: int = 1000,
    output_dir: Path = Path("data/output/task_orchestrator"),
    seed: int = 42,
    routing_distribution: dict[str, float] | None = None,  # Default: 60/20/20
    teacher_model: str = "gpt-4o",
) -> OrchestratorDatasetMetadata:
    """
    Generates Cat-A orchestrator routing data:
      - Cat A classifies user intent → selects routing target
      - tool_execution    → delegate to Cat B specialist (60%)
      - graph_extraction  → delegate to Cat C visualizer    (20%)
      - self_handle       → Cat A handles directly          (20%)
      - Delegates with well-formed request, synthesizes specialist response,
        continues workflow.
    Feeds eval/intent_classification.py and the Phase 4 orchestrator role.
    """
```

### 4.5 Chat Template Converter

```python
SUPPORTED_TEMPLATES = {
    "qwen":     {"chat": "chatml",                "tool": "hermes"},
    "qwen35":   {"chat": "chatml",                "tool": "qwen3_coder"},
    "qwen36":   {"chat": "chatml",                "tool": "qwen3_coder"},
    "gemma":    {"chat": "gemma",                 "tool": "gemma_native"},
    "mistral":  {"chat": "mistral_instruct_v3",   "tool": "mistral_tool_calls"},
    "nemotron": {"chat": "nemotron",              "tool": "nemotron_native"},
    "glm":      {"chat": "glm_chatml",            "tool": "glm4_tool"},
}

def convert_to_model_format(input_jsonl: Path, model_family: str, output_path: Path) -> ConversionStats:
    """Convert unified JSONL to model-specific chat template + tool format."""
```

### 4.6 Helper Modules

- **`_teacher_client.py`** — `call_teacher_model(teacher_model, system_prompt, user_prompt)` routes to OpenAI / Anthropic / Gemini APIs based on the model name prefix; retries and rate-limits centralized here.
- **`_workflow_script.py`** — single source of truth for `build_workflow_script(workflow_graph, tool_schemas, language, messages)`; builds the human-readable state/transition script used both at generation time and at inference enrichment. Optionally self-heals tool calls present in `messages`.
- **`domain_registry.py`** — `DomainSpec` dataclass + `DOMAINS` mapping (18 entries); decoupled from L1–L5 so any complexity level can be paired with any domain.
- **`system_prompt.py`** — `build_enriched_system_prompt(sample, original_content=None, force_rebuild=False)` is idempotent (detects already-enriched content). Exports `FORMAT_RULES` (the 7-rule guide).

### 4.7 Data Generation Recipes (DVC stages)

`dvc.yaml` defines four Task-A data stages (Task B / Task C stages are pending). **GRPO and held-out evaluation reuse splits of the SFT corpus** — no separate generation is performed for them.

| Stage | Output | Notes |
|-------|--------|-------|
| `task_a_benchmark` | `data/output/benchmark/task_a` | **~200 conversations** (40 per level × L1–L5) for Phase 1 model ranking. `language=mixed` (50/50 en/th), `behavior=default`, seed 100. **Teacher models:** `gemini-3-flash` and `gemini-3.1-flash-lite` (fast, cheap, reliable JSON emitters); two teacher runs merged into `l{1..5}_mixed_gemini-3_merged.jsonl` per level. `scripts/generate_benchmark_data.sh` retains a no-teacher placeholder path for smoke tests; `scripts/generate_benchmark_data_teacher.sh` is the canonical path. |
| `task_a_sft` | `data/output/sft/task_a` + `…/task_a_cleaned` | **~5 000 conversations** (4 450 raw → 4 445 cleaned by `scripts/clean_task_a_sft.py`). Three legs per level: en / `gpt-5.4-mini-2026-03-17`, th / `gemini-3-flash-preview`, code_switch / `gpt-5.4-nano-2026-03-17`. Behavior mix: ~87.5% adversarial + ~12.5% cooperative_only. Cleanup drops 5 truncated rows, strips 264 role-confused tool messages, flags 476 empty-terminal convs. |
| `task_a_sft_splits` | `data/output/sft/task_a_splits/{train,validation,test}.jsonl` | Deterministic 85 / 10 / 5 split of the cleaned corpus (seed 42), via `scripts/split_task_a_sft.py`. Current counts: ~4 414 / 519 / 261. **`test.jsonl` is reserved for final evaluation only.** |
| `task_a_grpo` | `data/output/grpo/task_a/{train,validation}.jsonl` | L3–L5 filter over the SFT splits, via `scripts/filter_grpo_data.py`. No new generation, no API calls. `test.jsonl` is intentionally excluded. |

**GRPO** reads prompts from `data/output/grpo/task_a/train.jsonl` (L3–L5 subset of the SFT train split) — rewards are recomputed online from policy generations, so no separate prompt corpus is required.

See `docs/data_generation_recipes.md` for full rationale.

---

## 5. Phase 1: Pre-Trained Benchmarking (Experiment A)

Phase 1 evaluation is implemented in `src/llm_workflow_agents/eval/agent_benchmark.py`, invoked by `scripts/run_exp_a.sh` (and the `_single` / `_per_level` variants).

**Entry CLI (representative):**

```bash
./scripts/run_exp_a.sh \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --engine vllm \
  --data-dir data/output/benchmark/task_a \
  --complexity-level mixed \
  --stochastic-trials 2 \
  --kv-cache-dtype fp8 \
  --output results/exp_a/qwen36_35b_a3b_fp8_vllm_fp8.json
```

**Supported engines:** `vllm`, `sglang`, `tensorrt_llm`, `bifrost` (remote frontier baseline).

**Result JSON shape (`results/exp_a/...json`):**

```json
{
  "model": "...",
  "engine": "vllm|sglang|tensorrt_llm|bifrost",
  "endpoint": "http://...",
  "kv_cache_dtype": "fp8|bf16|turboquant_3bit_nc|...",
  "data_dir": "data/output/benchmark/task_a",
  "complexity_level": "L1|L2|L3|L4|L5|mixed",
  "num_samples": 40,
  "stochastic_trials": 2,
  "metrics": {
    "weighted_workflow_score":    0.553,
    "full_workflow_success":      0.000,
    "state_metrics":              { "...": "..." },
    "tool_metrics":               { "...": "..." },   // per-turn (strict, zip-aligned)
    "tool_metrics_conversation":  { "...": "..." },   // per-conversation (lenient, set-aligned)
    "chain_metrics":              { "...": "..." },
    "latency_per_turn_median_ms": 0.0,
    "latency_per_turn_avg_ms":    993.5,
    "ttft_avg_ms":                808.0
  }
}
```

**Composite formula** (implemented in `eval/agent_benchmark.py::compute_weighted_score`):

```
weighted_workflow_score = 0.4 · max(state_transition_acc, state_sequence_acc)
                        + 0.4 · tool_call_f1
                        + 0.2 · task_completion_rate
```

**Legacy note:** an earlier `benchmark/` package (separate Phase 1 path) was removed; `agent_benchmark.py` is canonical. The `src/llm_workflow_agents/benchmark/` directory remains empty as a placeholder.

See `docs/metrics.md` for per-metric definitions, failure modes, and interpretation guidance.

---

## 6. Phase 2: SFT + GRPO RL Training (`training/`)

### 6.1 `sft.py` — Unsloth SFT Entry Point

```python
@dataclass
class SFTResult:
    checkpoint_path: Path
    best_eval_loss: float
    total_steps: int
    metrics: dict
    param_summary: dict
    error: str | None = None

def train_sft(
    config_path: Path,
    resume_from_checkpoint: bool | str | Path | None = None,
) -> SFTResult:
    """
    Unsloth-based supervised fine-tuning pipeline.
    """
```

**Pipeline:**

1. Load base via `FastLanguageModel.from_pretrained()` — QLoRA 4-bit for MoE, BF16 for dense ≤8GB.
2. Apply LoRA via `FastLanguageModel.get_peft_model()`; targets resolved from `lora_targets.py` registry (pattern-match on model family).
3. Configure SFTTrainer with ConstantLengthDataset packing (or FFD packing flag) + per-model chat template.
4. Train for `num_epochs`, checkpoint every `save_steps`.
5. Select best checkpoint by validation loss.

**Framework dispatch:** `framework: "unsloth"` is the default. For Gemma-4 MoE LoRA, set `framework: "trl"` (or env `SFT_FRAMEWORK=trl`) — this routes through standard TRL `SFTTrainer` because Unsloth's MoE kernels currently mishandle the Gemma-4 expert layout. `sft.py` also applies a RoPE stride patch for Gemma-4 (lines ~75–144) so that long-context attention computes correctly with rotary positional encoding.

**Unsloth pattern:**

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.model.name,
    max_seq_length=config.training.max_seq_length,
    dtype=None,
    load_in_4bit=config.training.precision == "qlora_4bit",
)
model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora.rank,
    lora_alpha=config.lora.alpha,
    lora_dropout=config.lora.dropout,
    target_modules=resolve_lora_targets(config),
    use_gradient_checkpointing="unsloth",
)
```

**System-prompt enrichment alignment.** All data paths (benchmark generation, SFT, GRPO, held-out eval) run `messages[0]` through `data/system_prompt.py::build_enriched_system_prompt` before training or scoring. The helper is **idempotent** — re-applying it to an already-enriched prompt is a no-op — so it acts as both the canonical enricher at generation time and a safety net at training/inference time. The enriched prompt always contains the structured **workflow script** (state machine in `### [STATE]` format), tool list, and 7-rule format guide; the optional teacher-authored natural-language role line (30% of teacher samples) augments but never replaces this structured reference. See `docs/fine_tuning_recipes.md`.

### 6.2 `grpo.py` — Unsloth GRPO RL Entry Point

```python
@dataclass
class GRPOResult:
    checkpoint_path: Path
    reward_curves: dict[str, list[float]]
    held_out_scores: list[float]
    kl_divergence: list[float]
    total_steps: int
    early_stopped: bool
    error: str | None = None

def train_grpo(config_path: Path) -> GRPOResult: ...
```

**Pipeline:**

1. Load SFT checkpoint via `FastLanguageModel.from_pretrained()`.
2. Configure GRPOTrainer with: task-specific reward from `training/rewards/`, vLLM generation backend (11× faster), FP8 RL on H100, DAPO normalization, `num_generations=4`, `beta=0.04` KL penalty.
3. Train for 500–1000 steps.
4. Monitor: reward curve, held-out eval every 50 steps, KL divergence from SFT.
5. **Reward-hacking detector** — optional TRL callback that auto-stops when reward↑ but held-out↓ (Risk R5).

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.model.base_checkpoint,
    max_seq_length=config.grpo.max_completion_length,
    load_in_4bit=True,
)
FastLanguageModel.for_training(model)

grpo_config = GRPOConfig(
    learning_rate=config.grpo.learning_rate,
    num_generations=config.grpo.num_generations,
    max_completion_length=config.grpo.max_completion_length,
    per_device_train_batch_size=config.grpo.per_device_batch_size,
    gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
    max_steps=config.grpo.training_steps,
    beta=config.grpo.beta,
    use_vllm=True,
)

reward_fn = load_reward_function(config.reward.function)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],
    args=grpo_config,
    train_dataset=prompts_dataset,
    callbacks=[RewardHackingDetector(...)] if config.monitoring.reward_hacking_detector else [],
)
trainer.train()
```

### 6.3 Reward Functions (`training/rewards/`)

Component weights match implementation exactly (verified May 2026).

#### 6.3.1 `reward_business_logic.py` — Cat A

```python
def reward_business_logic(prompts, completions, ground_truths) -> list[float]:
    # state_transition_correctness    0.30
    # tool_call_f1 (AST match)         0.30
    # chain_propagation_accuracy       0.20
    # format_compliance                0.10
    # task_completion (terminal state) 0.10
    # When ground truth has no terminal, weights rescale to sum 1 over remaining components.
```

#### 6.3.2 `reward_subagent.py` — Cat B

```python
def reward_subagent(prompts, completions, ground_truths) -> list[float]:
    # tool_call_f1:         0.40
    # slot_extraction_acc:  0.30
    # state_sequence_match: 0.20
    # format_compliance:    0.10
```

#### 6.3.3 `reward_graph_extraction.py` — Cat C

```python
def reward_graph_extraction(prompts, completions, ground_truths) -> list[float]:
    # Early exit: if completion is not valid JSON, return 0.0 immediately.
    # json_validity_bonus: 0.10
    # node_f1:             0.35
    # edge_f1:             0.35
    # structural_validity: 0.10
    # 1 - normalized_GED:  0.10
```

### 6.4 `reward_utils.py` — Shared Helpers

```python
def extract_state_annotations(text)           -> list[tuple[str, str]]
def extract_tool_calls(text)                  -> list[dict]
def state_sequence_match(predicted, gt)       -> float
def tool_call_f1(predicted, gt)               -> float        # BFCL AST sub-tree match
def chain_propagation_score(tools, chains)    -> float
def format_compliance_check(text)             -> float
def reached_terminal(text, expected_terminal) -> bool
def node_f1(predicted_nodes, gold_nodes)      -> float
def edge_f1(predicted_edges, gold_edges)      -> float
def structural_validity(graph)                -> float
def normalized_graph_edit_distance(pred, gold)-> float        # networkx-backed
```

### 6.5 `lora_targets.py` — Per-Architecture LoRA Module Registry (16 entries)

| Group | Models | Notes |
|-------|--------|-------|
| Dense (standard q/k/v/o + gate/up/down_proj) | `qwen25_3b`, `qwen3_32b`, `gemma_2b`, `gemma3_4b`, `gemma3_27b`, `mistral_24b` | No router freeze needed |
| Dense Gemma-4 | `gemma4_31b`, `gemma4_e4b`, `gemma4_e2b` | TRL fallback recommended; RoPE patch applied |
| DeltaNet + MoE | `qwen35_4b`, `qwen35_35b_a3b`, `qwen36_35b_a3b`, `qwen36_27b` | + DeltaNet projs (`in_proj_qkv/z/b/a`, `out_proj`); freeze `mlp.gate`; QLoRA 4-bit for 35B (~17.5 GB) |
| MoE + MLA | `glm47_flash` | MLA q_a/q_b/kv_a_proj_with_mqa/kv_b/o_proj + `shared_experts.*`; ~60 GB; auto-fallback rank 64→32 (R7) |
| MoE + Mamba | `nemotron_30b` | Mamba layers Unsloth auto-detect; vLLM compat issues (R6) — HF `generate()` fallback |
| Gemma-4 MoE | `gemma4_26b_a4b` | TRL framework override; freeze `mlp.gate` |

```python
def resolve_lora_targets(config) -> list[str]:
    """Flatten architecture-specific targets into a flat list for Unsloth / TRL."""
```

### 6.6 `pilot_check.py` — Risk R3 Mitigation

```python
def run_pilot_sft(top_2_models, task_data, pilot_steps=100) -> dict[str, PilotResult]:
    """100-step SFT on top-2 Phase 1 candidates; auto-fallback to rank-2 if rank-1 degrades."""
```

### 6.7 `merge_adapter.py`

```python
def merge_and_export(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    push_to_hub: bool = False,
    quantize_merged: str | None = None,    # Optional: "fp8" for deployment
) -> None:
    """Load base + LoRA adapter, merge via model.merge_and_unload(), save."""
```

### 6.8 `_utils.py` — Shared Training Helpers

`_build_training_arguments(config) -> TrainingArguments` converts `TrainingModelConfig` into HF `TrainingArguments` kwargs (batch size, LR, gradient checkpointing, precision, save/eval cadence). Heavy imports (torch, transformers) deferred to call-site. Used by `train_specialist.py` and `train_graph_extractor.py`.

### 6.9 v2 Backward-Compat Entries

- `train_specialist.py` — older unified entry point for Cat B specialist SFT (pre-Unsloth).
- `train_graph_extractor.py` — older unified entry point for Cat C graph extraction SFT.

Both retained for reproducibility of pre-v3 experiments; new work should use `sft.py` + `grpo.py`. See `docs/fine_tuning_recipes.md` for the canonical recipes.

---

## 7. Quantization Layer (`quantization/`) — Phase 3

### 7.1 Method Inventory

| Method | Config | Status |
|--------|--------|--------|
| FP8 E4M3 | `fp8.yaml` | Native vLLM; **default Phase 3 baseline** |
| KIVI (combined) | `kivi.yaml` | Asymmetric per-channel K, per-token V; no calibration |
| KIVI-2bit / KIVI-4bit | `kivi_2bit.yaml` / `kivi_4bit.yaml` | Bit-width-specific variants |
| KVQuant | `kvquant.yaml` | Pre-RoPE + NUQ + dense-sparse; per-model calibration |
| AWQ-INT4 + FP8 KV | `awq_fp8.yaml` | Weight quantization + FP8 KV |
| TurboQuant (upstream) | n/a (use variant strings) | **Production path**: `turboquant_3bit_nc`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_k8v4` from `vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend` |
| TurboQuant (project) | `turboquant.yaml` | **Scaffolding only** — see §7.4 / ADR-006 |
| RotorQuant | `rotorquant.yaml` | **Provisional / deferred** — gated on microbenchmark (§7.4) |

### 7.2 TurboQuant (`quantization/turboquant/`) — Project Variant

Kept in-tree for the QJL-residual research path and as documentation of the algorithm. Not wired into vLLM v1's `AttentionBackend` interface; see §7.4.

- **`codebook.py`** — `precompute_codebooks(head_dimensions=[128,256], bit_widths=[2,3,4])` → Lloyd-Max codebooks for Beta(α, α), α = (d−1)/2. One-time offline; cached to disk.
- **`rotation.py`** — `generate_rotation_matrix(d, seed=42)` → QR decomposition of Gaussian matrix → orthogonal Π (d×d).
- **`triton_kernels.py`** — Fused `turboquant_encode_kernel` (rotate → quantize → pack indices → store norm → QJL residual) and `turboquant_decode_kernel` (unpack → lookup centroids → inverse rotate).
- **`vllm_integration.py`** — `register_turboquant_backend()` (currently a documented no-op against vLLM v1; see ADR-006).

### 7.3 RotorQuant (`quantization/rotorquant/`)

- **`clifford.py`** — `CliffordAlgebra` Cl(3,0). Rotor sandwich product (~100 FMAs vs 16,384 dense d×d rotation), 10–19× speedup on the quantization path.
- **`rotor_kernels.py`** — fused Triton kernel (embed → rotor sandwich → quantize → inverse → extract).
- **`vllm_integration.py`** — placeholder; full v1 backend port is conditional on the microbenchmark gate (§7.4).

### 7.4 vLLM v1 Backend Integration Decision (ADR-006)

vLLM v1's plugin architecture requires a full `AttentionBackend` subclass (~800 LOC: `get_kv_cache_shape`, `supported_kv_cache_dtypes`, `AttentionImpl.forward` for prefill + decode, `AttentionMetadataBuilder`) plus `register_backend(...)` + `STR_DTYPE_TO_TORCH_DTYPE` monkey-patch. The v0 approach of patching module-level cache I/O no longer works.

**Decision (2026-04-21):**

- **TurboQuant (project variant): not porting.** Upstream's four `turboquant_*` variants cover the compression/quality curve; the only missing piece is the QJL 1-bit residual (~1–2 PPL benefit), which doesn't justify the porting cost. Phase 3 benchmark cells use upstream variants directly.
- **RotorQuant: gated on microbenchmark.** Before investing 1–2 weeks on a v1 backend, validate the Cl(3,0)-rotor-vs-Hadamard quality claim with a PyTorch-eager microbenchmark on Qwen2.5-3B held-out KV tensors. If RotorQuant beats Hadamard at matched bit budget, port. Otherwise drop from Phase 3.
- **Scaffolding kept in-tree.** `launch_vllm_turboquant.py` (project path), `launch_vllm_rotorquant.py`, and `register_turboquant_backend` (Pydantic patch for `CacheConfig` validation) remain so the exact strings `"turboquant"` / `"rotorquant"` flow through the legacy path for dev/comparison. Upstream `turboquant_*` variants bypass the project path entirely. `_patch_block_size` / `_patch_paged_attention` in `vllm_integration.py` are no-ops with migration warnings.

### 7.5 `vllm_plugin.py` — TurboQuant + Mamba KV-Cache Patches

Registered as a vLLM `general_plugins` entry point so it loads in every vLLM worker process. Provides `unify_kv_cache_spec_page_size`, which:

1. Fixes `MambaSpec` padding when models mix Mamba/SSM layers with attention layers.
2. Resolves block-size divisibility failures on TurboQuant's compressed `tq_slot_size` (102 vs 256 bytes), preventing the assertion trip in vLLM v1's KV-cache profiler.

Required for Qwen3.5/3.6 hybrid (DeltaNet) compatibility with `turboquant_*` variants.

### 7.6 Known Model × TurboQuant Incompatibilities

| Model | Status | Resolution |
|-------|--------|------------|
| Gemma-4 26B-A4B / 31B | Works with perf caveat (R8) | `_install_turboquant_engine_config_hook` in `launch_vllm_turboquant.py` auto-injects `enforce_eager=True` (skips broken mixed-KV profiler at `gpu_model_runner.py:6598`); 15–25% decode throughput penalty from CUDA graphs off; numerical correctness unaffected |
| Nemotron-3-Nano (Mamba hybrid) | Blocked | `arg_utils.py:1649` raises `NotImplementedError`; HF `generate()` fallback for quality eval only |
| Qwen3.5 / 3.6 hybrid (DeltaNet) | Unblocked | `_install_turboquant_engine_config_hook` masks `ModelConfig.is_hybrid=False` during engine config; DeltaNet/Mamba layers in boundary-skip list are no-ops (no `Attention()` constructed) |

**Compatible targets for Phase 3 TurboQuant cells:** Qwen3-32B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B (+ FP8), Qwen3.6-27B (+ FP8), Mistral-Small-3.1-24B, Gemma-3-27B, GLM-4.7-Flash, Gemma-4-26B-A4B (eager), Gemma-4-31B (eager).

---

## 8. Evaluation Layer (`eval/`)

Per-metric definitions and interpretation live in `docs/metrics.md`. The summary below covers public interfaces and target thresholds.

### 8.1 State-Machine Adherence (`state_accuracy.py`)

```python
@dataclass
class StateMachineMetrics:
    state_transition_accuracy: float       # Strict per-turn — target ≥85%
    state_sequence_accuracy:   float       # LCS-recall, lenient — reporting only
    task_completion_rate:      float       # ≥70%
    invalid_transition_rate:   float       # ≤5%
    recovery_rate:             float       # ≥60%
    consistency_pass5:         float       # ≥0.40 (pass^N where N = stochastic_trials)

def evaluate_state_machine(predictions, ground_truth, num_stochastic_trials=5) -> StateMachineMetrics
```

### 8.2 Tool-Calling Accuracy (`tool_call_f1.py`)

```python
@dataclass
class ToolCallMetrics:
    tool_name_accuracy:         float      # ≥90%
    argument_exact_match:       float      # ≥75%
    tool_call_f1:               float      # ≥85% (BFCL AST match)
    chain_propagation_accuracy: float      # ≥70%
    hallucinated_tool_rate:     float      # ≤3%
    error_recovery_rate:        float      # ≥60%

def evaluate_tool_calls(predictions, ground_truth, tool_schemas) -> ToolCallMetrics
```

`tool_metrics` (per-turn, zip-aligned, strict) and `tool_metrics_conversation` (per-conversation, set-aligned, lenient) are reported separately in `agent_benchmark.py` outputs.

### 8.3 Tool Chain Propagation (`tool_chain_propagation.py`)

Standalone module verifying that tool N's return values correctly populate tool N+1's arguments (per the `tool_chain_dependencies` ground truth).

### 8.4 Graph Extraction (`graph_extraction_eval.py`)

```python
@dataclass
class GraphExtractionMetrics:
    node_f1:               float           # ≥85%
    edge_f1:               float           # ≥75%
    graph_edit_distance:   float           # ≤0.20 (normalized)
    json_validity:         float           # ≥95%
    structural_validity:   float           # ≥90%
    mermaid_renderability: float           # ≥90%
```

### 8.5 Composite (`composite_score.py`)

```python
def compute_weighted_workflow_score(state, tool) -> float:
    return 0.4 * state.state_transition_accuracy + 0.4 * tool.tool_call_f1 + 0.2 * state.task_completion_rate

def full_workflow_success_rate(predictions, ground_truth) -> float:
    """% of conversations where ALL transitions AND ALL tool calls are correct. Target ≥55%."""
```

`agent_benchmark.py` also reports `weighted_workflow_score` (with `max(state_turn, state_seq)` for the state term) and `full_workflow_success` as independent-probability product.

### 8.6 Concurrency Benchmark (`concurrency_benchmark.py`) — NEW v4.1

Measures token throughput and latency (TTFT / TPOT / ITL at p50/p95/p99) at varying concurrency. Supports all four engines (vLLM, SGLang, TensorRT-LLM, BiFrost).

**Max-sustainable concurrency** is defined as the largest N such that:

```
ttft_p95[N] ≤ ttft_multiplier × ttft_p95[1]   (default ttft_multiplier = 32.0)
AND  failure_rate ≤ max_failure_rate          (default 0.01)
```

Always includes `concurrency=1` baseline regardless of the user-provided sweep. Configurable concurrency sweep + context-length sweep. Output goes to `results/concurrency/{model}_{engine}_{kv}.json` + `.log`.

### 8.7 Intent Classification (`intent_classification.py`) — NEW v4.1

Orchestrator routing evaluation: two-level classification.

- **Routing target** (3-way): `tool_execution` (→ Cat B) / `graph_extraction` (→ Cat C) / `self_handle` (→ Cat A).
- **Domain** (17-way, drawn from `domain_registry.py` minus `emergency`): supervised classification.

Supports zero-shot and few-shot evaluation. Dataclasses: `IntentSample`, `IntentClassificationMetrics`.

### 8.8 Constrained Decoding (`constrained_decoding.py`) — NEW v4.1

Outlines + XGrammar integration for the graph-extraction task. `load_graph_schema()` returns the JSON schema; `build_outlines_generator(model, schema)` constructs an Outlines generator for grammar-constrained output. Used by inference pipelines (not a standalone eval).

### 8.9 Quantization Benchmark Harness (`quant_benchmark.py`)

```python
def run_quant_benchmark(
    models: list[str],
    methods: list[str],   # e.g. ["fp8","kivi","kvquant","awq_fp8",
                          #       "turboquant_3bit_nc","turboquant_4bit_nc","turboquant_k3v4_nc","turboquant_k8v4"]
    quality_tasks: list[str],
    num_runs: int = 5,
    prompts_per_run: int = 500,
) -> QuantBenchmarkMatrix:
    """
    Quality: WikiText-2 PPL, C4 PPL, LongBench, Needle-in-Haystack, Tool-call F1.
    Performance: peak VRAM, KV cache size, throughput (prefill/decode), TTFT/TPOT/ITL p50/p95/p99,
                 max concurrent batch at 4096 ctx.
    Report: mean ± std over 3–5 runs.
    """

@dataclass
class QuantBenchmarkMatrix:
    models: list[str]
    methods: list[str]
    results: dict[str, CellResult]            # key = "{model}::{method}"
    total_runs: int
    elapsed_seconds: float
```

Method names are passed through verbatim — callers supply **upstream variant strings** (`turboquant_3bit_nc` etc.) per ADR-006, not the project's legacy `"turboquant"` string.

### 8.10 Headline Concurrency Results (2026-05-12)

Single H100 SXM 80GB, input 512–2048 tok (uniform), output 128 tok, 2048-context batch, `ttft_multiplier=32.0`, `max_failure_rate=0.01`. Concurrency sweep 1, 8, 16, 32, 64, 128, 256.

| Model | Engine | KV | Spec-decode | Max sustainable | Best tok/s | Best req/s | TTFT p95 | TPOT p95 | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B-FP8** (MoE, 3B active) | vllm | fp8 | — | 256 | **1792.8** | **14.01** | 627 ms | 15.1 ms | 72.1 GB |
| Qwen3.6-35B-A3B-FP8 | sglang | fp8 | — | 256 | 1635.6 | 12.78 | 906 ms | 16.2 ms | 79.1 GB |
| gemma-4-26B-A4B-FP8-Dynamic (MoE, 4B active) | sglang | fp8 | — | 256 | 1744.3 | 13.63 | 840 ms | 14.7 ms | 70.6 GB |
| gemma-4-26B-A4B-FP8-Dynamic | vllm | fp8 | — | 256 | 1433.3 | 11.20 | 686 ms | 19.7 ms | 72.4 GB |
| Qwen3.6-27B-FP8 (dense) | vllm | fp8 | — | 256 | 650.3 | 5.08 | 2308 ms | 41.8 ms | 77.3 GB |
| Qwen3.6-27B-FP8 (dense) | sglang | fp8 | — | 256 | 600.4 | 4.69 | 3004 ms | 44.0 ms | 78.9 GB |
| gemma-4-31B-FP8-block (dense) | vllm | fp8 | — | 256 | 462.1 | 3.61 | 3174 ms | 61.1 ms | 72.8 GB |
| Qwen3.6-35B-A3B-FP8 (DFlash spec-decode) | sglang | bf16 | DFlash | 256 | 475.2 | 8.17 | 675 ms | 68.8 ms | 76.9 GB |
| gemini-3.1-flash-lite (frontier baseline) | bifrost (remote) | — | — | 1024 | 237.0 | 10.37 | 1180 ms | 108.7 ms | n/a |

**Takeaways:**

1. **MoE dominates dense at this VRAM budget.** Qwen3.6-35B-A3B (3B active) + FP8 KV hits ~1.8k tok/s and 14 req/s at 72 GB — roughly 3× the dense Qwen3.6-27B / gemma-4-31B on the same hardware. For the Cat A orchestrator role, MoE-with-FP8 is the deployment shape to favor.
2. **vLLM ≈ SGLang for the MoEs**; both plateau at concurrency 256 (GPU/KV memory bound, not scheduler).
3. **FP8 KV beats BF16 + DFlash spec-decode at high concurrency.** On Qwen3.6-35B-A3B (SGLang), DFlash sustains 475 tok/s @ 256 vs 1636 tok/s with FP8 KV — 3.4× gap. DFlash buys low-load latency, not high-load throughput. Confirms FP8 as the Phase 3 baseline against which TurboQuant/KIVI/KVQuant are scored.
4. **gemma-4-31B-block is the laggard** — TTFT p95 > 3 s at concurrency 256 on both engines.
5. **Frontier reference (gemini-3.1-flash-lite via BiFrost) is throughput-bound by network**, not GPU — use as quality/latency baseline only, not throughput peer.

Full report: `docs/concurrency_benchmark_results.md`.

---

## 9. Phase 4: Integration & Pareto

`src/llm_workflow_agents/integration/` contains only `__init__.py` (re-exports). The actual modules now live at:

- `src/llm_workflow_agents/serving/orchestrator.py` — `MultiAgentOrchestrator`
- `src/llm_workflow_agents/serving/benchmark_e2e.py` — Concurrency + latency measurement
- `src/llm_workflow_agents/analysis/pareto.py` — Pareto frontier computation
- `src/llm_workflow_agents/analysis/plot_pareto.py` — 2D projections

### 9.1 `serving/orchestrator.py` — Multi-Agent Deployment

```python
class MultiAgentOrchestrator:
    """
    Production deployment architecture:
      - Orchestrator: Cat A winner (15–35B) — routes intent, manages workflow state
      - Specialist:   Cat B winner (2–5B)  — executes tool calls, state transitions
      - Visualizer:   Cat C winner (2–5B)  — converts prompts to workflow graphs on demand

    All served via vLLM with best quantization from Phase 3.
    vLLM LoRA multi-adapter serving if Cat B and Cat C share a base model.
    """

    def __init__(
        self,
        orchestrator_config: Path,
        specialist_config: Path,
        visualizer_config: Path,
        kv_cache_dtype: str = "turboquant_3bit_nc",   # Upstream variant per ADR-006
    ): ...

    async def run_workflow(self, user_input: str, workflow_graph: dict) -> WorkflowResult:
        """
        1. Orchestrator receives user input
        2. Classifies intent → selects specialist (uses intent_classification metrics during eval)
        3. Specialist executes tool calls + state transitions
        4. Returns to orchestrator for confirmation / next step
        5. Optional: Cat C model generates workflow visualization
        """

    async def run_scenario_battery(
        self,
        num_scenarios: int = 50,
        trials_per_scenario: int = 5,
    ) -> IntegrationResults:
        """50+ scenarios × 5 trials each = 250+ multi-agent runs."""
```

### 9.2 `serving/benchmark_e2e.py` — E2E Concurrency

Cross-references `eval/concurrency_benchmark.py` (§8.6) for the underlying throughput/latency engine. `benchmark_e2e.py` wraps multi-model deployments (orchestrator + specialist + visualizer) and reports session-level metrics.

Expected results on single H100 80GB @ 4096 ctx (multi-agent):

| Method | Concurrent sessions (multi-agent) |
|--------|------------|
| BF16 | ~175 (96 KB/tok) |
| FP8 | ~350 (48 KB/tok) |
| TurboQuant 3-bit (upstream) | ~925 (~18 KB/tok) |

### 9.3 `analysis/pareto.py` — Pareto Frontier

```python
def compute_pareto_frontier(
    results: list[ConfigResult],
    axes: tuple[str, str, str] = ("task_completion", "peak_vram_gb", "p95_latency_ms"),
) -> list[ConfigResult]:
    """Pareto-optimal (model, quantization) configs across quality × memory × latency."""

def plot_pareto_projections(
    pareto_configs: list[ConfigResult],
    output_dir: Path = Path("analysis/figures"),
) -> list[Path]:
    """3 × 2D scatter plots (delegates rendering to analysis/plot_pareto.py)."""
```

---

## 10. Data Flow: End-to-End Pipeline

### 10.1 Phase 1 → Phase 2 Handoff

```
                    ┌──────────────────────────────────────────────────┐
                    │              PHASE 1: BENCHMARK                   │
                    │                                                   │
                    │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
                    │  │ Task A  │  │ Task B  │  │ Task C  │         │
                    │  │14 cands │  │ 7 cands │  │ 7 cands │         │
                    │  └────┬────┘  └────┬────┘  └────┬────┘         │
                    │       └────────────┼────────────┘               │
                    │                    ▼                             │
                    │         ┌──────────────────┐                    │
                    │         │ agent_benchmark  │                    │
                    │         │ composite scores  │                    │
                    │         └────────┬─────────┘                    │
                    └─────────────────┼───────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────────┐
                    │ pilot_check.py: 100-step SFT on top-2 per cat   │
                    │ Confirm winner responds to fine-tuning (Risk R3) │
                    └─────────────────┬───────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ PHASE 2: CAT A      │ │ PHASE 2: CAT B      │ │ PHASE 2: CAT C      │
│                     │ │                     │ │                     │
│ SFT (3 epochs)      │ │ SFT (3 epochs)      │ │ SFT (3 epochs)      │
│   ↓                 │ │   ↓                 │ │   ↓                 │
│ GRPO (500–1000 st.) │ │ GRPO (500–1000 st.) │ │ GRPO (500–1000 st.) │
│ reward_business_    │ │ reward_subagent()   │ │ reward_graph_       │
│   logic()           │ │                     │ │   extraction()      │
└────────┬────────────┘ └────────┬────────────┘ └────────┬────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    3 fine-tuned models (SFT+RL)
```

### 10.2 Phase 3 → Phase 4

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: QUANTIZATION BENCHMARK                   │
│                                                                      │
│  (3 fine-tuned + pre-trained originals) × upstream quant methods     │
│                                                                      │
│  ┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐ ┌────────────────────┐    │
│  │ FP8  │ │ KIVI │ │KVQuant │ │AWQ+FP8 │ │ turboquant_*_nc    │    │
│  └──┬───┘ └──┬───┘ └───┬────┘ └───┬────┘ └────────┬───────────┘    │
│     └────────┴─────────┴──────────┴───────────────┘                 │
│                                  │                                   │
│            ┌─────────────────────┼──────────────────┐                │
│            ▼                     ▼                   ▼                │
│      Quality matrix      Performance matrix    Concurrency est.      │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │       PHASE 4: PARETO       │
                    │                             │
                    │  serving/orchestrator.py     │
                    │    → multi-agent deployment  │
                    │    → 50 scenarios × 5 trials │
                    │                             │
                    │  analysis/pareto.py          │
                    │    → quality × memory × lat  │
                    │    → 2D projections          │
                    │    → optimal config          │
                    └─────────────────────────────┘
```

### 10.3 DVC Pipeline (`dvc.yaml`)

DVC currently drives data generation and select SFT runs:

| Stage | Type | Output |
|-------|------|--------|
| `task_a_benchmark` | Data gen | `data/output/benchmark/task_a` |
| `task_a_sft` | Data gen + clean | `data/output/sft/task_a`, `…/task_a_cleaned` |
| `task_a_grpo` | Data gen | `data/output/grpo/task_a` |
| `task_a_eval` | Data gen | `data/output/val/task_a`, `data/output/test/task_a` |
| `task_a_sft_gemma4_26b_a4b` | SFT | `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/` |

Phase 2 (other models), Phase 3, and Phase 4 stages are not yet wired into DVC; they currently run via direct shell scripts (`scripts/run_phase2_sft.sh`, `scripts/run_exp_*.sh`, `scripts/run_concurrency_benchmark.sh`).

### 10.4 SFT vs SFT+RL Comparison (Cross-Cutting)

```
For each of the 3 winners, the eval layer produces:

  Pre-trained ────────▸ Phase 1 metrics (baseline)
       │
       ▼
  SFT checkpoint ─────▸ Same eval suite (SFT improvement Δ)
       │
       ▼
  SFT+RL checkpoint ──▸ Same eval suite (RL improvement Δ over SFT)

  Answers RQ1 ("how much does SFT+RL improve?")
  and RQ2 ("can GRPO improve tool F1 beyond SFT alone?").
```

---

## 11. Architecture Decision Records

### ADR-001: Benchmark-First Model Selection — Accepted

The v2 proposal fine-tuned all candidates — expensive given limited H100 time. Evaluate all 15 in pre-trained state first, fine-tune only the 3 winners. Include pilot SFT (100 steps) on top-2 to validate fine-tuning response before committing. ~70% reduction in fine-tuning compute. Risk: missing a model that's mediocre pre-trained but excels post-tune (mitigated by pilot check).

### ADR-002: Unsloth over Standard PEFT/TRL — Accepted

Training MoE on a single H100 is VRAM-constrained. Standard PEFT/TRL cannot fit GRPO for large MoE. Unsloth gives 2× speed, 70% less VRAM, native GRPO with vLLM generation, FP8 RL, MoE-optimized kernels (12× faster). Qwen3.5-35B-A3B fits in ~17.5 GB with QLoRA 4-bit. **Exception:** Gemma-4 MoE LoRA currently routes through TRL fallback (`framework: "trl"`) due to expert-layout incompatibility with Unsloth's kernel.

### ADR-003: SFT Then GRPO (Two-Stage Training) — Accepted

SFT alone maximizes next-token prediction; doesn't directly optimize task success. Two-stage: SFT (format + domain), GRPO (task-metric RL). GRPO over PPO because no separate value/reward model needed — verifiable reward functions instead. Risk R5 (reward hacking) mitigated by held-out eval every 50 steps + reward-hacking detector callback.

### ADR-004: Shared SFT Base for Dual-Category Winners — Accepted

If the same model wins Cat B and Cat C, share the SFT base checkpoint. Diverge at the GRPO stage with different reward functions and training data. Saves one SFT run; doubled GRPO compute; two LoRA adapters at serving time.

### ADR-005: Triton for Custom Quantization Kernels — Accepted

TurboQuant + RotorQuant need custom encode/decode kernels. Triton ≥3.0 fused kernels. TurboQuant adds 1-bit QJL residual; RotorQuant uses Cl(3,0) rotor sandwich (~100 FMAs vs 16,384 dense).

### ADR-006: Upstream vLLM TurboQuant Variants over Project Custom Backend — Accepted (2026-04-21)

vLLM v1's plugin architecture requires a full `AttentionBackend` subclass for any new KV compression scheme; the v0 monkey-patch approach no longer works. Upstream ships `vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend` with four production variants: `turboquant_3bit_nc`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_k8v4`.

- **Use upstream for Phase 3.** Project's `quantization/turboquant/` retains the QJL-residual research path and documentation but is **not** wired into vLLM v1.
- **RotorQuant port deferred** pending standalone microbenchmark on Qwen2.5-3B held-out KV (Cl(3,0) rotor vs Hadamard at matched bit budget). If win is not material, drop RotorQuant from Phase 3.
- **Project scaffolding kept** for exact `"turboquant"` / `"rotorquant"` strings only — routes through `launch_vllm_*.py` for dev/comparison. `_patch_block_size` / `_patch_paged_attention` are no-ops with migration warnings.

---

## 12. Risk Registry (Code-Level)

| # | Risk | Probability | Code Impact | Mitigation |
|---|------|-------------|-------------|------------|
| R1 | Qwen3.5-35B-A3B BF16 exceeds 80GB for GRPO | High | `training/grpo.py` | Unsloth QLoRA 4-bit (~17.5GB) + FP8 RL. Configured in `configs/training/grpo_cat_a.yaml`. |
| R2 | Same model wins Cat B + Cat C → doubled fine-tuning | Medium | `training/sft.py`, `training/grpo.py` | Share SFT base, diverge at GRPO. `scripts/run_phase2_sft.sh` (will) detect shared winner. |
| R3 | Phase 1 winner doesn't respond to fine-tuning | Low | `training/pilot_check.py` | 100-step pilot SFT on top-2. Auto-fallback to #2 in `scripts/run_phase2_sft.sh`. |
| R4 | TurboQuant PR not merged | Resolved | n/a | Upstream variants now available (`turboquant_*_nc` / `_k3v4_nc` / `_k8v4`); ADR-006 supersedes the fork dependency. |
| R5 | GRPO reward hacking | Medium | `training/grpo.py`, `training/rewards/` | Multiple orthogonal reward components. Held-out eval every 50 steps. KL divergence monitoring. Reward-hacking detector callback auto-stops if held-out metric drops while reward increases. |
| R6 | Nemotron Mamba + vLLM incompatibility | Medium | `eval/agent_benchmark.py`, `serving/launch_vllm.sh` | HF `generate()` fallback for quality eval. **Qwen3.5/3.6 hybrid (DeltaNet) unblocked** via `_install_turboquant_engine_config_hook` in `launch_vllm_turboquant.py` (masks `is_hybrid` during engine config). Nemotron-3-Nano remains out of scope (Mamba layers + vLLM compat beyond the hybrid guard). |
| R7 | GLM LoRA exceeds 80GB even with Unsloth | Medium | `training/sft.py` | Auto-reduce rank 64→32. Fallback: inference-only evaluation with strong prompting. |
| R8 | Gemma-4 + TurboQuant breaks vLLM v1 mixed-KV profiler | Medium | `serving/launch_vllm_turboquant.py`, `vllm_plugin.py` | **Works with perf caveat.** Two hooks in `launch_vllm_turboquant.py`: (1) `_install_gemma4_mixed_backend_hook` bypasses `TRITON_ATTN` global force at `vllm/model_executor/models/config.py:100`; (2) `_install_turboquant_engine_config_hook` auto-injects `enforce_eager=True` for Gemma-4 + `turboquant_*` (skips broken profiler at `gpu_model_runner.py:6598`). CUDA graphs off → 15–25% decode throughput penalty vs graph mode. Numerical correctness unaffected. Restore graph mode once upstream lands a per-layer-aware profiler. |

---

## 13. Dependency Stack

Authoritative source: `pyproject.toml` + `requirements-train.txt` + `requirements-infer.txt`. Highlights:

```
# Core
torch>=2.4
transformers>=5.0
unsloth>=2025.3                # SFT + GRPO RL with MoE kernels
trl>=0.15                      # GRPOTrainer, GRPOConfig, SFTTrainer (Gemma-4 fallback)
peft>=0.14
vllm>=0.8                      # PagedAttention v2, tool-call parsers, turboquant_* variants
triton>=3.0
flash-attn>=2.5

# Multi-engine serving
sglang                         # Alternative inference engine — competitive with vLLM on MoE
tensorrt-llm                   # NVIDIA TRT-LLM (engine builds via scripts/build_trtllm_engines.sh)
bifrost-client                 # Remote/frontier baseline proxy (Gemini-3.1, etc.)

# Data
datasets>=2.19
outlines>=0.0.40               # Constrained decoding (Task C)
xgrammar>=0.1                  # Alternative constrained decoding

# Evaluation
networkx>=3.2                  # Graph edit distance
jsonschema>=4.21

# Analysis & Logging
wandb>=0.17
matplotlib>=3.8
seaborn>=0.13

# Quantization
scipy>=1.12                    # Lloyd-Max codebook optimization
numpy>=1.26

# DVC pipeline + utilities
dvc>=3, pyyaml>=6.0
```

Install scripts under `scripts/` cover the various env permutations: `install_train.sh`, `install_train_unsloth.sh`, `install_train_cu128.sh`, `install_infer.sh`, `install_sglang.sh`, `install_trtllm.sh`.

---

## 14. Testing Strategy

### 14.1 Unit Tests

| Module | Test File | Key Assertions |
|--------|-----------|----------------|
| Data generation | `test_data_generation.py` | Schema validity, behavior distribution, tool error rate, split ratios |
| Reward functions | `test_reward_functions.py` | Known-answer scores, edge cases (invalid JSON → 0.0), component weights sum to 1.0 |
| Eval metrics | `test_eval_metrics.py` | F1, GED, pass^5 vs hand-computed |
| Composite score | `test_composite_score.py` | Normalization, weight application, ranking stability |
| Triton kernels | `test_triton_kernels.py` | encode→decode ≈ identity (within quantization error) |
| Chat templates | `test_chat_templates.py` | Round-trip fidelity across 6 formats |

### 14.2 Integration Tests

| Test | Description |
|------|-------------|
| Phase 1 smoke | 2 models × 1 task × 10 samples → composite score computation |
| SFT smoke | 50 steps on 100 samples → checkpoint saves → merge → inference |
| GRPO smoke | 10 steps with mock reward → reward logging + policy update |
| Reward hacking detector | Synthetic reward ↑ + held-out ↓ → alert fires |
| Quant round-trip | BF16 → TurboQuant encode → decode → PPL delta within tolerance |
| E2E pipeline | `run_exp_a.sh` on 1 model × 1 task × 10 samples → full output |

### 14.3 Reproducibility

Seed-deterministic configuration throughout. Phase 1 benchmarks use temperature 0.0. Consistency metrics use temperature 0.7 with N trials (`stochastic_trials`, default 5 in `state_accuracy.py`, often 2 in CLI). Quantization benchmarks run 3–5 repetitions on 500+ prompts, reporting mean ± std.

---

## 15. Execution Status — as of 2026-05-13

| Phase | Status | Notes |
|-------|--------|-------|
| Weeks 1–2 — Data preparation | ✓ Complete | Task A: 2 DVC stages — `task_a_benchmark` (~200 conversations, Gemini teachers) and `task_a_sft` (~5 000 conversations, GPT + Gemini teachers, cleaned → 4 445 with 85/10/5 train/val/test splits). GRPO and held-out eval reuse the SFT splits. Task B + Task C generators landed. Orchestrator routing generator new in v4.1. |
| Weeks 3–4 — Phase 1 benchmark | ✓ Complete | `eval/agent_benchmark.py` + `scripts/run_exp_a*.sh`. Per-level results in `results/exp_a/`. |
| Weeks 5–7 — Phase 2 SFT | Partial | Gemma-4 26B-A4B SFT wired into DVC (`task_a_sft_gemma4_26b_a4b`). Other category winners pending selection finalization and pilot check. |
| Weeks 8–9 — Phase 2 GRPO | Not started | `grpo.py`, all 3 rewards, GRPO data ready. Awaiting SFT checkpoints. |
| Weeks 10–11 — Phase 3 quantization | Partial | Concurrency benchmarks landed (`docs/concurrency_benchmark_results.md`). Full TurboQuant × model matrix in progress. RotorQuant gated on microbenchmark (ADR-006). |
| Weeks 12–13 — Phase 4 integration | Code ready | `serving/orchestrator.py` + `benchmark_e2e.py` + `analysis/pareto.py` all implemented. Scenario battery (50 × 5 trials) pending Phase 2/3 outputs. |
| Week 14 — Report & release | Ongoing | `analysis/plot_*.py` plotting suite complete. Final report consolidation in progress. |

**Open work items:**

- Wire Phase 2 GRPO into DVC (currently shell-driven).
- Author `scripts/run_phase3.sh` / `run_phase4.sh` for end-to-end reproducibility.
- Complete the RotorQuant Cl(3,0)-vs-Hadamard microbenchmark; promote or drop.
- Run the orchestrator routing eval (`eval/intent_classification.py`) against the Cat A winner with the new orchestrator dataset.

Cross-references:
- `docs/data_generation_recipes.md` — full data-gen recipes per stage
- `docs/fine_tuning_recipes.md` — Stage-1 SFT and Stage-2 GRPO recipes
- `docs/metrics.md` — per-metric definitions and interpretation
- `docs/concurrency_benchmark_results.md` — 2026-05-12 headline numbers
- `docs/turboquant_model_patches.md` — vLLM v1 TurboQuant variant patches and known model issues
- `CLAUDE.md` — current source-of-truth for ADRs, risk updates, and pending KV-cache work
