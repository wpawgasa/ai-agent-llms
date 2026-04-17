# Codebase Specification: Benchmark-First LLM Workflow Agents with SFT+RL Fine-Tuning

**Project:** `llm-workflow-agents-v4`
**Version:** 4.0 — April 2026
**Hardware Target:** Single NVIDIA H100 SXM 80GB
**Training Framework:** Unsloth (SFT + GRPO RL)

---

## 1. System Overview

### 1.1 Core Thesis

This codebase implements a **benchmark-first, fine-tune-selectively** pipeline for workflow-orchestrating LLM agents. Instead of fine-tuning all candidate models, we:

1. **Benchmark** all pre-trained candidates across three task categories
2. **Select** the single best model per category
3. **Fine-tune** each winner with SFT then GRPO reinforcement learning
4. **Deploy** with aggressive KV cache quantization on a single H100

The final deliverable is **three fine-tuned specialist models** — one per task — served concurrently via vLLM with TurboQuant/RotorQuant compression.

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

| RQ | Description | Primary Code Path |
|----|-------------|-------------------|
| RQ1 | Best 15–35B for prompt-encoded logic; SFT+RL improvement | `eval/agent_benchmark.py` → `training/sft.py` → `training/grpo.py` → `eval/` |
| RQ2 | Best 2–5B specialist; GRPO vs SFT-only on tool F1 | `eval/agent_benchmark.py` → `training/sft.py` → `training/grpo.py` → `eval/tool_call_f1.py` |
| RQ3 | Graph extraction with ≥85% Node F1 / ≥75% Edge F1 | `data/generate_graph_pairs.py` → `training/sft.py` → `training/grpo.py` → `eval/graph_extraction_eval.py` |
| RQ4 | TurboQuant/RotorQuant vs FP8/KIVI/KVQuant for multi-agent concurrency | `quantization/` → `eval/quant_benchmark.py` → `integration/pareto.py` |

### 1.4 Model Inventory

**Category A — Prompt-Encoded Business Logic (15–35B):**

| Model | Total | Active | Architecture | Context | Tool Parser | BF16 VRAM |
|-------|-------|--------|-------------|---------|-------------|-----------|
| Gemma 3 27B-IT | 27B | 27B | Dense GQA | 128K | `gemma` | ~54 GB |
| Qwen3-32B | 32B | 32B | Dense + `<think>` | 128K | `hermes` | ~64 GB |
| Qwen3.5-35B-A3B | 35B | 3B | DeltaNet + MoE | 262K | `qwen3_coder` | ~70 GB |
| Mistral Small 3.1 24B | 24B | 24B | Dense, sliding-win | 128K | `mistral` | ~48 GB |
| Nemotron-3-Nano 30B | 30B | 3.6B | MoE + Mamba-2 | 1M | `nemotron` | ~60 GB |
| GLM-4.7-Flash | 30B | 3.6B | MoE + MLA | 200K | `glm4` | ~60 GB |
| Gemma 4 26B-A4B-IT | 26B | 4B | MoE GQA | 128K | `gemma` | ~52 GB |
| Gemma 4 31B-IT | 31B | 31B | Dense GQA | 128K | `gemma` | ~62 GB |
| Qwen3.6-35B-A3B | 35B | 3B | DeltaNet + MoE | 262K | `qwen3_coder` | ~70 GB |
| Qwen3.6-35B-A3B-FP8 | 35B | 3B | DeltaNet + MoE | 262K | `qwen3_coder` | ~35 GB |

**Category B–C — Specialist Subagent & Graph Extraction (2–5B):**

| Model | Params | Active | Architecture | Context | VRAM | Unsloth |
|-------|--------|--------|-------------|---------|------|---------|
| Qwen2.5-3B-Instruct | 3B | 3B | Dense GQA | 32K | ~6 GB | Full (SFT+RL) |
| Qwen3.5-4B | 4B | ~3B | DeltaNet + MoE | 262K | ~8 GB | Full (SFT+RL) |
| GLM-4.7-Flash | 30B | 3.6B | MoE + MLA | 200K | ~60 GB | Full (SFT+RL) |
| Gemma-2B | 2.5B | 2.5B | Dense MQA | 8K | ~5 GB | Full (SFT+RL) |
| Gemma-3-4B-it | 4B | 4B | Dense GQA | 128K | ~8 GB | Full (SFT+RL) |
| Gemma-4-E4B-it | 4B | 4B | Dense GQA | 128K | ~8 GB | Full (SFT+RL) |
| Gemma-4-E2B-it | 2B | 2B | Dense GQA | 128K | ~4 GB | Full (SFT+RL) |

---

## 2. Repository Structure

```
llm-workflow-agents-v3/
│
├── configs/                              # All YAML configuration (§3)
│   ├── models/
│   │   ├── cat_a/                        # 12 Category A model configs
│   │   │   ├── gemma3_27b.yaml
│   │   │   ├── qwen3_32b.yaml
│   │   │   ├── qwen35_35b_a3b.yaml
│   │   │   ├── mistral_small_24b.yaml
│   │   │   ├── nemotron_30b.yaml
│   │   │   ├── glm47_flash.yaml
│   │   │   ├── gemma4_26b_a4b.yaml
│   │   │   ├── gemma4_31b.yaml
│   │   │   ├── qwen36_35b_a3b.yaml
│   │   │   └── qwen36_35b_a3b_fp8.yaml
│   │   └── cat_bc/                       # 7 Category B–C model configs
│   │       ├── qwen25_3b.yaml
│   │       ├── qwen35_4b.yaml
│   │       ├── glm47_flash_small.yaml
│   │       ├── gemma_2b.yaml
│   │       ├── gemma3_4b.yaml
│   │       ├── gemma4_e4b.yaml
│   │       └── gemma4_e2b.yaml
│   ├── training/
│   │   ├── sft_cat_a.yaml                # SFT hyperparams for Cat A winner
│   │   ├── sft_cat_b.yaml                # SFT hyperparams for Cat B winner
│   │   ├── sft_cat_c.yaml                # SFT hyperparams for Cat C winner
│   │   ├── grpo_cat_a.yaml               # GRPO RL config for Cat A
│   │   ├── grpo_cat_b.yaml               # GRPO RL config for Cat B
│   │   └── grpo_cat_c.yaml               # GRPO RL config for Cat C
│   ├── quantization/
│   │   ├── fp8.yaml
│   │   ├── kivi.yaml
│   │   ├── kvquant.yaml
│   │   ├── awq_fp8.yaml
│   │   ├── turboquant.yaml
│   │   └── rotorquant.yaml
│   ├── benchmark/
│   │   ├── phase1_matrix.yaml            # Which models × which tasks
│   │   └── selection_weights.yaml        # Composite score weights per category
│   └── serving/
│       ├── single_model.yaml
│       ├── multi_agent.yaml
│       └── benchmark_e2e.yaml
│
├── data/                                 # Data generation (§4)
│   ├── generate_workflows.py             # L1–L5 workflow conversations (Task A)
│   ├── generate_tool_call_data.py        # Specialist training data (Task B)
│   ├── generate_graph_pairs.py           # (prompt, graph) pairs (Task C)
│   ├── chat_template_converter.py        # Per-model chat format conversion
│   ├── data_validator.py                 # Schema & quality validation
│   └── templates/
│       ├── workflow_prompt_template.txt
│       ├── tool_schemas_L1_to_L5.json
│       └── graph_output_schema.json
│
├── training/                             # Phase 2: SFT + GRPO RL (§6)
│   ├── sft.py                            # Unsloth SFT training entry point
│   ├── grpo.py                           # Unsloth GRPO RL training entry point
│   ├── rewards/                          # Task-specific GRPO reward functions
│   │   ├── reward_business_logic.py      # Cat A: state + tool + chain rewards
│   │   ├── reward_subagent.py            # Cat B: tool F1 + slot + state rewards
│   │   └── reward_graph_extraction.py    # Cat C: node/edge F1 + structural rewards
│   ├── reward_utils.py                   # Shared reward computation helpers
│   ├── lora_targets.py                   # Per-architecture LoRA module registry
│   ├── merge_adapter.py                  # LoRA → merged model export
│   └── pilot_check.py                    # 100-step pilot SFT on top-2 (Risk R3)
│
├── quantization/                         # Phase 3: KV cache quantization (§7)
│   ├── turboquant/
│   │   ├── codebook.py                   # Lloyd-Max codebook for Beta distribution
│   │   ├── rotation.py                   # QR-based orthogonal rotation matrix
│   │   ├── triton_kernels.py             # Fused Triton encode/decode
│   │   └── vllm_integration.py           # vLLM cache dtype registration + hooks
│   ├── rotorquant/
│   │   ├── clifford.py                   # Cl(3,0) geometric algebra primitives
│   │   └── rotor_kernels.py              # Fused rotor sandwich Triton kernels
│   └── baselines/
│       ├── kivi_cache.py                 # KIVI wrapper
│       └── kvquant_calibrate.py          # KVQuant calibration + NUQ codebook
│
├── eval/                                 # Evaluation modules (§8)
│   ├── state_accuracy.py                 # State transition acc, completion, recovery
│   ├── tool_call_f1.py                   # BFCL-style tool-call evaluation
│   ├── tool_chain_propagation.py         # Multi-step chain propagation accuracy
│   ├── graph_extraction_eval.py          # Node F1, Edge F1, GED, renderability
│   ├── perplexity.py                     # WikiText-2, C4 PPL
│   ├── longbench.py                      # LongBench 15-task evaluation
│   ├── needle_haystack.py                # Needle-in-a-Haystack (2K–32K)
│   ├── quant_benchmark.py                # Full quantization benchmark harness
│   └── composite_score.py                # Weighted composite score computation
│
├── integration/                          # Phase 4: E2E deployment (§9)
│   ├── orchestrator.py                   # Multi-agent orchestrator
│   ├── benchmark_e2e.py                  # Concurrency + latency measurement
│   └── pareto.py                         # Pareto frontier computation + plotting
│
├── serving/                              # vLLM serving utilities
│   ├── launch_vllm.sh                    # Model-aware vLLM launch script
│   └── vllm_utils.py                     # Health checks, adapter loading helpers
│
├── analysis/                             # Visualization & reporting
│   ├── plot_phase1_rankings.py           # Phase 1 ranking bar charts
│   ├── plot_sft_vs_rl.py                # SFT-only vs SFT+RL improvement curves
│   ├── plot_quant_matrix.py             # Quantization quality/perf heatmaps
│   └── plot_pareto.py                    # 2D Pareto frontier projections
│
├── scripts/                              # Top-level experiment runners
│   ├── run_exp_a.sh                      # Phase 1: benchmark all pre-trained (Experiment A)
│   ├── run_phase2_sft.sh                 # Phase 2a: SFT for 3 winners
│   ├── run_phase2_grpo.sh                # Phase 2b: GRPO RL for 3 winners
│   ├── run_phase3.sh                     # Phase 3: quantization benchmark
│   ├── run_phase4.sh                     # Phase 4: integration + Pareto
│   └── run_pilot_check.sh               # Risk R3: pilot SFT on top-2
│
├── tests/                                # Unit and integration tests
│   ├── test_data_generation.py
│   ├── test_reward_functions.py
│   ├── test_eval_metrics.py
│   ├── test_composite_score.py
│   ├── test_triton_kernels.py
│   └── test_chat_templates.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Configuration Schema

### 3.1 Model Configuration

```yaml
# configs/models/cat_a/qwen35_35b_a3b.yaml
model:
  name: "Qwen/Qwen3.5-35B-A3B"
  family: "qwen35"
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
  engine: "vllm"
  tool_call_parser: "qwen3_coder"
  chat_template: "qwen3.5"
  gpu_memory_utilization: 0.90
  max_model_len: 8192
  enforce_eager: false

category: "A"                              # Which benchmark category
benchmark_tasks: ["task_a"]                # Which Phase 1 tasks to run
```

### 3.2 SFT Training Configuration

```yaml
# configs/training/sft_cat_a.yaml
stage: "sft"
framework: "unsloth"

model:
  config_ref: null                         # Populated at runtime from Phase 1 winner

lora:
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules: "auto"                   # Unsloth auto-detection (override in lora_targets.py)
  freeze_router: true                      # MoE router frozen by Unsloth default

training:
  precision: "qlora_4bit"                  # QLoRA for MoE / BF16 for dense ≤8GB
  learning_rate: 1.0e-4
  lr_scheduler: "cosine"
  lr_end: 5.0e-5                           # Cosine decay target
  warmup_ratio: 0.05
  effective_batch_size: 16                 # Adjustable 16–32
  gradient_accumulation_steps: 4
  num_epochs: 3
  max_seq_length: 4096
  gradient_checkpointing: true
  packing: true                            # Unsloth ConstantLengthDataset

data:
  source: "data/output/task_a"             # L1–L5 workflow conversations (1000 total)
  format: "chatml"
  splits:
    train: 0.85
    val: 0.10
    test: 0.05

logging:
  wandb_project: "llm-workflow-agents-v3"
  wandb_run_prefix: "sft-cat-a"
  save_strategy: "steps"
  save_steps: 500
  eval_steps: 500
  metric_for_best_model: "eval_loss"
```

### 3.3 GRPO RL Configuration

```yaml
# configs/training/grpo_cat_a.yaml
stage: "grpo"
framework: "unsloth"

model:
  base_checkpoint: null                    # Populated: path to best SFT checkpoint

grpo:
  algorithm: "GRPO"
  normalization: "DAPO"                    # Removes length bias
  num_generations: 4                       # Completions per prompt for group comparison
  beta: 0.04                               # KL penalty: constrain near SFT policy
  epsilon: 0.2                             # PPO-style clipping
  learning_rate: 5.0e-6                    # 10x lower than SFT
  lr_scheduler: "constant"
  training_steps: 1000                     # 500–1000
  per_device_batch_size: 2
  gradient_accumulation_steps: 4           # Effective batch: 8–16
  max_completion_length: 2048

  generation_backend: "vllm"               # 11x faster RL inference
  fp8_rl: true                             # H100: 1.4x faster, 60% less VRAM

reward:
  function: "reward_business_logic"        # From training/rewards/
  # Component weights (defined in reward function):
  #   state_transition: 0.30
  #   tool_call_f1:     0.30
  #   chain_propagation: 0.20
  #   format_compliance: 0.10
  #   task_completion:   0.10

data:
  prompts_source: "data/output/task_a"     # Same prompts as SFT
  ground_truth_source: "data/output/task_a/ground_truth.jsonl"

monitoring:
  wandb_run_prefix: "grpo-cat-a"
  eval_held_out_every: 50                  # Steps between held-out eval
  reward_hacking_detector: true            # Alert if reward ↑ but held-out ↓ (Risk R5)
  kl_divergence_log: true
```

### 3.4 Benchmark Selection Weights

```yaml
# configs/benchmark/selection_weights.yaml
category_a:
  response_quality: 0.40                   # State transition acc + tool call F1
  latency_p95: 0.25
  throughput_tok_s: 0.20
  memory_peak_vram: 0.15

category_b:
  response_quality: 0.35                   # Tool call F1 + slot accuracy
  latency_p95: 0.30
  throughput_tok_s: 0.20
  memory_peak_vram: 0.15

category_c:
  response_quality: 0.40                   # Node F1 + Edge F1
  latency_p95: 0.20
  throughput_tok_s: 0.20
  memory_peak_vram: 0.20                   # Higher: must coexist with orchestrator
```

### 3.5 Quantization Configuration

```yaml
# configs/quantization/turboquant.yaml
method:
  name: "turboquant"
  paper: "Zandieh et al., ICLR 2026"
  status: "PR #38280 — use 0xSero/turboquant fork"

codebook:
  distribution: "beta"
  alpha_formula: "(d - 1) / 2"
  head_dimensions: [128, 256]
  bit_widths: [2, 3, 4]
  precompute_offline: true

rotation:
  method: "qr_gaussian"
  seed_deterministic: true
  seed_base: 42

residual:
  method: "qjl"                            # 1-bit QJL residual quantization
  bits: 1

triton_kernels:
  encode:
    operations: ["rotate", "quantize", "pack_indices", "store_norm"]
    hook_point: "vllm_cache_write"
  decode:
    operations: ["unpack", "lookup_centroids", "inverse_rotate"]
    hook_point: "vllm_cache_read"

vllm_integration:
  kv_cache_dtype: "turboquant"
  config_module: "vllm.config.cache"
  flash_attn_module: "vllm.attention.backends.flash_attn"
  block_size_bytes_3bit: 52

benchmark:
  quality_tasks: ["wikitext2_ppl", "c4_ppl", "longbench", "needle_haystack", "tool_call_f1"]
  performance_metrics: ["peak_vram", "kv_cache_size", "throughput_prefill", "throughput_decode",
                        "ttft", "tpot", "itl_p50_p95_p99", "max_concurrent_4096ctx"]
  runs: 5
  prompts_per_run: 500
```

---

## 4. Data Generation Layer (`data/`)

### 4.1 `generate_workflows.py` — Task A Data

**Purpose:** Generate multi-turn conversation datasets at 5 complexity levels with state-machine annotations and tool-calling ground truth.

```python
@dataclass
class ComplexitySpec:
    level: str
    num_states: tuple[int, int]
    branching_factor: tuple[int, int]
    num_tools: int
    chain_depth: int
    nesting_depth: int
    num_samples: int
    domain: str

COMPLEXITY_SPECS = {
    "L1": ComplexitySpec("L1", (3,4),   (1,2), 1, 0, 0, 200, "faq_lookup"),
    "L2": ComplexitySpec("L2", (5,7),   (2,3), 2, 1, 1, 200, "order_status_cancel"),
    "L3": ComplexitySpec("L3", (8,12),  (3,5), 4, 2, 2, 200, "booking_payment"),
    "L4": ComplexitySpec("L4", (13,20), (5,8), 6, 3, 3, 200, "it_troubleshoot"),
    "L5": ComplexitySpec("L5", (21,30), (8,99),7, 4, 4, 200, "multi_dept_workflow"),
}

USER_BEHAVIOR_DISTRIBUTION = {
    "cooperative": 0.60,
    "adversarial_probing": 0.15,
    "digressing": 0.10,
    "invalid_tool_inputs": 0.15,
}

TOOL_ERROR_RATE = 0.20  # 20% of tool calls return error payloads


def generate_workflow_dataset(
    complexity_level: Literal["L1", "L2", "L3", "L4", "L5"],
    teacher_model: str = "gpt-4o",
    output_dir: Path = Path("data/output/task_a"),
    seed: int = 42,
) -> DatasetMetadata:
    """
    Generate multi-turn conversations for a single complexity level.

    Each conversation includes:
      - System prompt with workflow graph, tool schemas, rules
      - User/assistant turns with [STATE: X → Y] annotations
      - <tool_call>{JSON}</tool_call> blocks with valid argument JSON
      - Tool response turns (80% success, 20% error)
      - Multi-step tool chains (output of tool N → input of tool N+1)

    Output: JSONL file with ground-truth annotations per turn.
    """
```

**Output Format (per sample):**

```json
{
  "conversation_id": "L3_042",
  "complexity_level": "L3",
  "domain": "booking_payment",
  "num_states": 10,
  "workflow_graph": {
    "states": ["S1", "S2", "..."],
    "transitions": [{"from": "S1", "to": "S2", "condition": "..."}],
    "initial": "S1",
    "terminal": ["S9", "S10"]
  },
  "tool_schemas": [...],
  "messages": [
    {"role": "system", "content": "<full workflow prompt>"},
    {"role": "user", "content": "I need to cancel order #4521"},
    {
      "role": "assistant",
      "content": "[STATE: GREETING → COLLECT_INFO]\n<tool_call>...</tool_call>",
      "annotations": {
        "state_transition": {"from": "GREETING", "to": "COLLECT_INFO"},
        "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "4521"}}]
      }
    },
    {"role": "tool", "content": "{\"order_id\":\"4521\",\"status\":\"active\"}"}
  ],
  "user_behavior": "cooperative",
  "ground_truth": {
    "state_sequence": ["GREETING", "COLLECT_INFO", "EXECUTE", "CONFIRM"],
    "tool_chain_dependencies": [{"from_tool": "lookup_order", "to_tool": "cancel_order",
                                  "propagated_field": "order_id"}],
    "terminal_state": "CONFIRM"
  }
}
```

### 4.2 `generate_tool_call_data.py` — Task B Data

```python
def generate_tool_call_dataset(
    external_sources: list[str] = [
        "Salesforce/xlam-function-calling-60k",
        "ToolBench",
    ],
    custom_synthetic_size: int = 15000,
    teacher_model: str = "gpt-4o",
    negative_ratio: float = 0.15,
    hermes_format: bool = True,
    output_dir: Path = Path("data/output/task_b"),
    seed: int = 42,
) -> DatasetSplits:
    """
    Merge external datasets with custom synthetic data into unified JSONL.
    Output in Hermes tool-call schema format.
    Negative examples: wrong tool (5%), hallucinated tool (4%),
                       invalid state transition (3%), error recovery (3%).
    Splits: 85% train / 10% val / 5% test.
    """
```

### 4.3 `generate_graph_pairs.py` — Task C Data

```python
def generate_graph_pairs(
    workflow_prompts_dir: Path,
    gold_annotations: int = 200,
    teacher_generated: int = 800,
    augmentation_target: int = 5000,
    output_dir: Path = Path("data/output/task_c"),
) -> DatasetSplits:
    """
    5,000 (prompt, graph) pairs.
    - 200 manually annotated gold-standard
    - 800 GPT-4o generated, validated against gold
    - 4,000 paraphrase augmentation (prompts varied, graphs constant)
    Splits: 4,000 train / 500 val / 500 test.
    """
```

**Graph Output JSON Schema:**

```python
GRAPH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":            {"type": "string"},      # e.g. "S1"
                    "name":          {"type": "string"},      # e.g. "Greeting"
                    "tools":         {"type": "array", "items": {"type": "string"}},
                    "entry_actions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "name"]
            }
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from":      {"type": "string"},
                    "to":        {"type": "string"},
                    "condition": {"type": "string"},
                    "priority":  {"type": "integer"},
                },
                "required": ["from", "to", "condition"]
            }
        },
        "initial_state":   {"type": "string"},
        "terminal_states": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["nodes", "edges", "initial_state", "terminal_states"]
}
```

### 4.4 `chat_template_converter.py`

```python
SUPPORTED_TEMPLATES = {
    "qwen":        {"chat": "chatml", "tool": "hermes"},
    "qwen35":      {"chat": "chatml", "tool": "qwen3_coder"},
    "gemma":       {"chat": "gemma",  "tool": "gemma_native"},
    "mistral":     {"chat": "mistral_instruct_v3", "tool": "mistral_tool_calls"},
    "nemotron":    {"chat": "nemotron", "tool": "nemotron_native"},
    "glm":         {"chat": "glm_chatml", "tool": "glm4_tool"},
}

def convert_to_model_format(
    input_jsonl: Path,
    model_family: str,
    output_path: Path,
) -> ConversionStats:
    """Convert unified JSONL to model-specific chat template + tool format."""
```

---

## 5. Phase 1: Pre-Trained Benchmarking

Phase 1 evaluation is implemented in `eval/agent_benchmark.py` and invoked by `scripts/run_exp_a*.sh`. A separate `benchmark/` package previously existed but duplicated the evaluation path with reduced prompt fidelity and has been removed.

---

## 6. Phase 2: SFT + GRPO RL Training (`training/`)

### 6.1 `sft.py` — Unsloth SFT Entry Point

```python
def train_sft(config_path: Path) -> SFTResult:
    """
    Unsloth-based supervised fine-tuning pipeline:

    1. Load base model via FastLanguageModel.from_pretrained()
       - 4-bit QLoRA for MoE models (Qwen3.5-35B-A3B: 17.5GB)
       - BF16 for dense models ≤8GB VRAM
    2. Apply LoRA via FastLanguageModel.get_peft_model()
       - target_modules from lora_targets.py or Unsloth auto-detect
       - Router weights frozen for MoE models
    3. Configure SFTTrainer with:
       - ConstantLengthDataset packing
       - Per-model chat template via chat_template_converter
       - W&B logging
    4. Train for num_epochs, checkpoint every save_steps
    5. Select best checkpoint by validation loss
    6. Return SFTResult with checkpoint path + training metrics

    Unsloth advantages over standard PEFT/TRL:
      - 2x faster training
      - 70% less VRAM
      - MoE-optimized kernels: 12x faster, 35% less VRAM
    """
```

**Unsloth-Specific Code Pattern:**

```python
from unsloth import FastLanguageModel

# Load with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.model.name,
    max_seq_length=config.training.max_seq_length,
    dtype=None,                            # Auto-detect
    load_in_4bit=config.training.precision == "qlora_4bit",
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora.rank,
    lora_alpha=config.lora.alpha,
    lora_dropout=config.lora.dropout,
    target_modules=resolve_lora_targets(config),  # From lora_targets.py
    use_gradient_checkpointing="unsloth",          # Unsloth optimized
)
```

### 6.2 `grpo.py` — Unsloth GRPO RL Entry Point

```python
def train_grpo(config_path: Path) -> GRPOResult:
    """
    Unsloth GRPO reinforcement learning pipeline:

    1. Load SFT checkpoint via FastLanguageModel.from_pretrained()
    2. Configure GRPOTrainer with:
       - reward_funcs: task-specific from training/rewards/
       - vLLM generation backend (11x faster RL inference)
       - FP8 RL on H100 (1.4x faster, 60% less VRAM)
       - DAPO normalization (removes length bias)
       - num_generations=4 per prompt
       - beta=0.04 KL penalty (stay near SFT policy)
    3. Train for training_steps (500–1000)
    4. Monitor:
       - Reward curve (should increase)
       - Held-out eval every 50 steps (Risk R5: reward hacking detection)
       - KL divergence from SFT checkpoint
    5. Save best checkpoint by held-out composite metric
    6. Return GRPOResult with checkpoint path + reward curves

    Key difference from SFT:
      - SFT optimizes cross-entropy loss (next-word prediction)
      - GRPO directly optimizes task success metrics via reward functions
      - No separate value model or reward model needed (memory efficient)
    """
```

**Unsloth GRPO Code Pattern:**

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.model.base_checkpoint,  # SFT checkpoint
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
    use_vllm=True,                             # 11x faster generation
    # vllm_gpu_memory_utilization auto-managed by Unsloth
)

reward_fn = load_reward_function(config.reward.function)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],
    args=grpo_config,
    train_dataset=prompts_dataset,
)
trainer.train()
```

### 6.3 Reward Functions (`training/rewards/`)

#### 6.3.1 `reward_business_logic.py` — Cat A

```python
def reward_business_logic(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[dict],
) -> list[float]:
    """
    GRPO reward function for prompt-encoded business logic.

    Component weights:
      R1: State transition correctness    — 0.30
      R2: Tool call F1 (AST match)        — 0.30
      R3: Chain propagation accuracy       — 0.20
      R4: Format compliance                — 0.10
      R5: Task completion (terminal state) — 0.10

    Returns: list of scalar rewards ∈ [0.0, 1.0]
    """
    scores = []
    for prompt, completion, gt in zip(prompts, completions, ground_truths):
        s = 0.0
        predicted_states = extract_state_annotations(completion)
        predicted_tools = extract_tool_calls(completion)

        s += 0.30 * state_sequence_match(predicted_states, gt["state_sequence"])
        s += 0.30 * tool_call_f1(predicted_tools, gt["tool_calls"])
        s += 0.20 * chain_propagation_score(predicted_tools, gt["chain_dependencies"])
        s += 0.10 * format_compliance_check(completion)
        s += 0.10 * (1.0 if reached_terminal(completion, gt["terminal_state"]) else 0.0)
        scores.append(s)
    return scores
```

#### 6.3.2 `reward_subagent.py` — Cat B

```python
def reward_subagent(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[dict],
) -> list[float]:
    """
    Component weights:
      Tool call F1:         0.40
      Slot extraction acc:  0.30
      State sequence match: 0.20
      Format compliance:    0.10
    """
```

#### 6.3.3 `reward_graph_extraction.py` — Cat C

```python
def reward_graph_extraction(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[dict],
) -> list[float]:
    """
    Component weights:
      JSON validity bonus:         0.10 (0.0 if invalid JSON → total reward 0.0)
      Node F1:                     0.35
      Edge F1:                     0.35
      Structural validity:         0.10
      1 − normalized GED:          0.10

    Early exit: if completion is not valid JSON, return 0.0 immediately.
    """
```

### 6.4 `reward_utils.py` — Shared Helpers

```python
def extract_state_annotations(text: str) -> list[tuple[str, str]]:
    """Parse [STATE: X → Y] from model output. Returns [(from, to), ...]."""

def extract_tool_calls(text: str) -> list[dict]:
    """Parse <tool_call>{JSON}</tool_call> blocks. Returns [{"name": ..., "arguments": ...}]."""

def state_sequence_match(predicted: list, ground_truth: list) -> float:
    """Sequence-level accuracy: fraction of turns with correct transition."""

def tool_call_f1(predicted: list[dict], ground_truth: list[dict]) -> float:
    """BFCL-style AST sub-tree match on (name, arguments) pairs."""

def chain_propagation_score(tools: list[dict], chains: list[dict]) -> float:
    """Check if tool N's return values correctly populate tool N+1's arguments."""

def format_compliance_check(text: str) -> float:
    """Valid JSON in tool calls (0/1) × proper [STATE:] format (0/1), averaged."""

def reached_terminal(text: str, expected_terminal: str) -> bool:
    """Check if the last [STATE: X → Y] reaches the expected terminal state."""

def node_f1(predicted_nodes: list, gold_nodes: list) -> float:
    """Precision/recall on extracted state nodes by name match."""

def edge_f1(predicted_edges: list, gold_edges: list) -> float:
    """Precision/recall on (from, to, condition) triples."""

def structural_validity(graph: dict) -> float:
    """Check: valid initial, reachable terminals, no orphan nodes. Score 0–1."""

def normalized_graph_edit_distance(predicted: dict, gold: dict) -> float:
    """Normalized GED via networkx. Range [0, 1], lower is better."""
```

### 6.5 `lora_targets.py` — Per-Architecture LoRA Module Registry

```python
LORA_TARGET_MODULES = {
    # Dense models — standard projection layers
    "qwen25_3b":    ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    "qwen3_32b":    ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    "gemma_2b":     ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    "gemma3_4b":    ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    "mistral_24b":  ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],

    # DeltaNet + MoE hybrids — include DeltaNet-specific projections
    "qwen35_4b": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "deltanet":  ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
        "mlp":       ["gate_proj", "up_proj", "down_proj"],
        "freeze":    ["mlp.gate"],     # Router frozen by Unsloth default
    },
    "qwen35_35b_a3b": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "deltanet":  ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
        "mlp":       ["gate_proj", "up_proj", "down_proj"],
        "freeze":    ["mlp.gate"],
        "notes":     "256 total experts, 8 routed + 1 shared active. QLoRA 4-bit → ~17.5GB.",
    },
    "qwen36_35b_a3b": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "deltanet":  ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
        "mlp":       ["gate_proj", "up_proj", "down_proj"],
        "freeze":    ["mlp.gate"],
        "notes":     "Same arch as Qwen3.5-35B-A3B. QLoRA 4-bit → ~17.5GB.",
    },

    # MoE + MLA — GLM-specific projections
    "glm47_flash": {
        "attention":      ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
        "shared_experts": ["shared_experts.gate_proj", "shared_experts.up_proj",
                           "shared_experts.down_proj"],
        "freeze":         ["mlp.gate"],
        "notes":          "~60GB BF16. Use Unsloth MoE kernels. Fallback: rank 32 or inference-only.",
    },

    # MoE + Mamba hybrid
    "nemotron_30b": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mamba":     [],               # Mamba layers: Unsloth auto-detect
        "moe_mlp":   ["gate_proj", "up_proj", "down_proj"],
        "freeze":    ["mlp.gate"],
        "notes":     "52 layers: 23 Mamba-2 + 23 MoE + 6 Attention. vLLM compatibility uncertain (R6).",
    },
}

def resolve_lora_targets(config) -> list[str]:
    """Flatten architecture-specific targets into a flat list for Unsloth."""
```

### 6.6 `pilot_check.py` — Risk R3 Mitigation

```python
def run_pilot_sft(
    top_2_models: list[str],
    task_data: Path,
    pilot_steps: int = 100,
) -> dict[str, PilotResult]:
    """
    Run 100-step SFT on the top-2 candidates from Phase 1 before committing.

    Purpose: Detect if the Phase 1 winner doesn't respond well to fine-tuning
    (diminishing returns). If #1 degrades, fall back to #2.

    Returns: {model_name: PilotResult(loss_curve, held_out_quality_delta)}
    """
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

---

## 7. Quantization Layer (`quantization/`) — Phase 3

### 7.1 TurboQuant (`quantization/turboquant/`)

**`codebook.py`:**
```python
def precompute_codebooks(
    head_dimensions: list[int] = [128, 256],
    bit_widths: list[int] = [2, 3, 4],
    output_dir: Path = Path("quantization/turboquant/codebooks"),
) -> dict[tuple[int, int], np.ndarray]:
    """
    Lloyd-Max codebooks for Beta(α, α), α = (d−1)/2.
    One-time offline. Results cached to disk.
    """
```

**`rotation.py`:**
```python
def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """QR decomposition of Gaussian matrix → orthogonal Π (d×d)."""
```

**`triton_kernels.py`:**
```python
@triton.jit
def turboquant_encode_kernel(kv_ptr, rotation_ptr, codebook_ptr,
                              output_indices_ptr, output_norms_ptr, ...):
    """Fused: rotate → quantize → pack indices + store norm + QJL residual."""

@triton.jit
def turboquant_decode_kernel(indices_ptr, norms_ptr, codebook_ptr,
                              inv_rotation_ptr, output_ptr, ...):
    """Fused: unpack → lookup centroids → inverse rotate."""
```

**`vllm_integration.py`:**
```python
def register_turboquant_backend():
    """
    1. Register kv_cache_dtype="turboquant" in vllm.config.cache
    2. Hook encode into PagedAttention cache write
    3. Hook decode into flash_attn backend cache read
    4. Modify block_size: 3-bit = 52 bytes per 128-value vector
    """
```

### 7.2 RotorQuant (`quantization/rotorquant/`)

**`clifford.py`:**
```python
class CliffordAlgebra:
    """
    Cl(3,0) algebra for rotor-based rotation.
    Rotor sandwich product: ~100 FMAs (vs 16,384 for dense d×d rotation).
    10–19× faster quantization path.
    """
    def rotor_from_params(self, params: torch.Tensor) -> Rotor: ...
    def sandwich_product(self, rotor: Rotor, x: torch.Tensor) -> torch.Tensor: ...
```

**`rotor_kernels.py`:**
```python
@triton.jit
def rotorquant_fused_kernel(kv_ptr, rotor_params_ptr, codebook_ptr, output_ptr, ...):
    """Fused: embed → rotor sandwich → quantize → inverse → extract."""
```

### 7.3 Baselines

**`kivi_cache.py`:** KIVI (ICML 2024) — asymmetric per-channel K, per-token V. No calibration.

**`kvquant_calibrate.py`:** KVQuant (NeurIPS 2024) — pre-RoPE + NUQ + dense-sparse. Per-model calibration.

---

## 8. Evaluation Layer (`eval/`)

### 8.1 State-Machine Adherence (`state_accuracy.py`)

```python
@dataclass
class StateMachineMetrics:
    state_transition_accuracy: float       # Target: ≥85%
    task_completion_rate: float            # Target: ≥70%
    invalid_transition_rate: float         # Target: ≤5%
    recovery_rate: float                   # Target: ≥60%
    consistency_pass5: float               # Target: ≥0.40

def evaluate_state_machine(
    predictions: list[ConversationPrediction],
    ground_truth: list[ConversationGroundTruth],
    num_stochastic_trials: int = 5,
) -> StateMachineMetrics: ...
```

### 8.2 Tool-Calling Accuracy (`tool_call_f1.py`)

```python
@dataclass
class ToolCallMetrics:
    tool_name_accuracy: float              # Target: ≥90%
    argument_exact_match: float            # Target: ≥75%
    tool_call_f1: float                    # Target: ≥85% (BFCL AST match)
    chain_propagation_accuracy: float      # Target: ≥70%
    hallucinated_tool_rate: float          # Target: ≤3%
    error_recovery_rate: float             # Target: ≥60%

def evaluate_tool_calls(
    predictions: list[TurnPrediction],
    ground_truth: list[TurnGroundTruth],
    tool_schemas: list[ToolSchema],
) -> ToolCallMetrics: ...
```

### 8.3 Graph Extraction (`graph_extraction_eval.py`)

```python
@dataclass
class GraphExtractionMetrics:
    node_f1: float                         # Target: ≥85%
    edge_f1: float                         # Target: ≥75%
    graph_edit_distance: float             # Target: ≤0.20
    json_validity: float                   # Target: ≥95%
    structural_validity: float             # Target: ≥90%
    mermaid_renderability: float           # Target: ≥90%

def evaluate_graph_extraction(
    predicted_graphs: list[dict],
    gold_graphs: list[dict],
) -> GraphExtractionMetrics: ...
```

### 8.4 Combined Workflow Quality (`composite_score.py`)

```python
def compute_weighted_workflow_score(
    state: StateMachineMetrics,
    tool: ToolCallMetrics,
) -> float:
    """0.4 × StateTransAcc + 0.4 × ToolCallF1 + 0.2 × TaskCompletion. Target: ≥0.75."""
    return 0.4 * state.state_transition_accuracy + \
           0.4 * tool.tool_call_f1 + \
           0.2 * state.task_completion_rate

def full_workflow_success_rate(
    predictions: list[ConversationPrediction],
    ground_truth: list[ConversationGroundTruth],
) -> float:
    """% of conversations where ALL state transitions AND ALL tool calls correct. Target: ≥55%."""
```

### 8.5 Quantization Benchmark Harness (`quant_benchmark.py`)

```python
def run_quant_benchmark(
    models: list[str],                     # Both pre-trained and fine-tuned
    methods: list[str],                    # ["fp8", "kivi", "kvquant", "awq_fp8",
                                           #  "turboquant", "rotorquant"]
    quality_tasks: list[str],
    num_runs: int = 5,
    prompts_per_run: int = 500,
) -> QuantBenchmarkMatrix:
    """
    Full matrix: (pre-trained + fine-tuned models) × 6 quantization methods.

    Quality: WikiText-2 PPL, C4 PPL, LongBench, Needle-in-Haystack, Tool-call F1.
    Performance: Peak VRAM, KV cache size, throughput, latency (TTFT/TPOT/ITL at p50/p95/p99),
                 max concurrent batch at 4096 context.

    Report: mean ± std over 3–5 runs.
    """

@dataclass
class QuantBenchmarkMatrix:
    results: dict[tuple[str, str], QuantResult]  # (model, method) → metrics
    expected_concurrency: dict[str, ConcurrencyEstimate]

@dataclass
class ConcurrencyEstimate:
    kv_dtype: str
    bytes_per_token: int
    available_kv_gb: float
    max_tokens: int
    max_concurrent: int                    # BF16: ~175, FP8: ~350, TQ-3bit: ~925
```

---

## 9. Phase 4: Integration & Pareto (`integration/`)

### 9.1 `orchestrator.py` — Multi-Agent Deployment

```python
class MultiAgentOrchestrator:
    """
    Production deployment architecture:
      - Orchestrator: Cat A winner (15–35B) — routes intent, manages workflow state
      - Specialist:   Cat B winner (2–5B) — executes tool calls, state transitions
      - Visualizer:   Cat C winner (2–5B) — converts prompts to workflow graphs on demand

    All served via vLLM with best quantization method from Phase 3.
    vLLM LoRA multi-adapter serving if Cat B and Cat C are same base model.
    """

    def __init__(
        self,
        orchestrator_config: Path,
        specialist_config: Path,
        visualizer_config: Path,
        kv_cache_dtype: str = "turboquant",
    ): ...

    async def run_workflow(
        self,
        user_input: str,
        workflow_graph: dict,
    ) -> WorkflowResult:
        """
        1. Orchestrator receives user input
        2. Classifies intent → selects specialist
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

### 9.2 `benchmark_e2e.py` — Concurrency Measurement

```python
def benchmark_concurrency(
    deployment_config: Path,
    context_length: int = 4096,
) -> ConcurrencyResult:
    """
    Expected results for multi-agent on single H100 80GB:
      BF16:         ~175 concurrent sessions (96 KB/tok)
      FP8:          ~350 concurrent sessions (48 KB/tok)
      TQ 3-bit:     ~925 concurrent sessions (~18 KB/tok)
    """
```

### 9.3 `pareto.py` — Pareto Frontier Computation

```python
def compute_pareto_frontier(
    results: list[ConfigResult],
    axes: tuple[str, str, str] = ("task_completion", "peak_vram_gb", "p95_latency_ms"),
) -> list[ConfigResult]:
    """
    Identify Pareto-optimal (model, quantization) configurations across
    quality × memory × latency. Produce 2D projections for each pair.
    """

def plot_pareto_projections(
    pareto_configs: list[ConfigResult],
    output_dir: Path = Path("analysis/figures"),
) -> list[Path]:
    """Generate 3 × 2D scatter plots: quality×memory, quality×latency, memory×latency."""
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
                    │  │ 6 models│  │ 5 models│  │ 5 models│         │
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
│    (3 fine-tuned + pre-trained originals) × 6 quant methods          │
│                                                                      │
│    ┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐  │
│    │ FP8  │ │ KIVI │ │KVQuant │ │AWQ+FP8 │ │ Turbo  │ │ Rotor   │  │
│    │ E4M3 │ │ 2-4b │ │ 2-4b   │ │ W4+KV8 │ │ Quant  │ │ Quant   │  │
│    └──┬───┘ └──┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └────┬────┘  │
│       └────────┴─────────┴──────────┴──────────┴───────────┘        │
│                                    │                                 │
│              ┌─────────────────────┼──────────────────┐              │
│              ▼                     ▼                   ▼              │
│        Quality matrix      Performance matrix    Concurrency est.    │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │       PHASE 4: PARETO       │
                    │                             │
                    │  orchestrator.py             │
                    │    → multi-agent deployment  │
                    │    → 50 scenarios × 5 trials │
                    │                             │
                    │  pareto.py                   │
                    │    → quality × memory × lat  │
                    │    → 2D projections          │
                    │    → optimal config          │
                    └─────────────────────────────┘
```

### 10.3 SFT vs SFT+RL Comparison (Cross-Cutting)

```
For each of the 3 winners, the eval layer produces:

  Pre-trained ────────▸ Phase 1 metrics (baseline)
       │
       ▼
  SFT checkpoint ─────▸ Same eval suite (SFT improvement Δ)
       │
       ▼
  SFT+RL checkpoint ──▸ Same eval suite (RL improvement Δ over SFT)

  This directly answers RQ1 ("how much does SFT+RL improve?")
  and RQ2 ("can GRPO improve tool F1 beyond SFT alone?")
```

---

## 11. Architecture Decision Records

### ADR-001: Benchmark-First Model Selection

**Status:** Accepted

**Context:** The v2 proposal fine-tuned all 10 candidate models — an expensive approach given limited H100 time. Many models would be discarded post-evaluation anyway.

**Decision:** Evaluate all 11 candidates in their pre-trained state first (Phase 1). Select only the best model per task category (3 total) for fine-tuning. Include pilot check (100-step SFT) on top-2 candidates to validate fine-tuning response before committing.

**Consequences:**
- Positive: ~70% reduction in fine-tuning compute. Systematic comparison of pre-trained baselines. Early detection of poor fine-tuning candidates.
- Negative: Risk of missing a model that's mediocre pre-trained but excels post fine-tuning (mitigated by pilot check).

### ADR-002: Unsloth over Standard PEFT/TRL

**Status:** Accepted

**Context:** Training MoE models (Qwen3.5-35B-A3B, GLM-4.7-Flash) on a single H100 is VRAM-constrained. Standard PEFT/TRL cannot fit GRPO training for large MoE models.

**Decision:** Use Unsloth for all training: 2x speed, 70% less VRAM, native GRPO with vLLM generation, FP8 RL, MoE-optimized kernels (12x faster MoE training).

**Consequences:**
- Positive: Qwen3.5-35B-A3B fits in ~17.5GB with QLoRA 4-bit. GRPO feasible for all models. vLLM-accelerated generation makes RL 11x faster.
- Negative: Unsloth is a third-party dependency with its own release cycle. API differences from standard TRL may require adaptation.

### ADR-003: SFT Then GRPO (Two-Stage Training)

**Status:** Accepted

**Context:** SFT alone maximizes next-token prediction but doesn't directly optimize task success metrics (state correctness, tool-call F1, workflow completion).

**Decision:** Two-stage pipeline: SFT first (establish format knowledge and domain capability), then GRPO RL (optimize directly for task-specific reward functions). GRPO chosen over PPO because it eliminates the need for a separate value model and reward model — using custom verifiable reward functions instead.

**Consequences:**
- Positive: RL directly optimizes production-relevant metrics. GRPO is memory-efficient (no value/reward model). Verifiable reward functions enable precise control.
- Negative: Risk of reward hacking (R5). Requires careful reward function design. GRPO adds training complexity.
- Mitigation for R5: Monitor held-out eval every 50 steps. Use multiple orthogonal reward components. Add format compliance checks.

### ADR-004: Shared SFT Base for Dual-Category Winners

**Status:** Accepted

**Context:** The same model may win both Cat B and Cat C (e.g., Qwen3.5-4B). This requires two separate fine-tunes.

**Decision:** If the same model wins both categories, share the SFT base checkpoint. Diverge at the GRPO RL stage with different reward functions and training data.

**Consequences:**
- Positive: Saves one SFT training run. Shared base ensures consistent tokenization and format handling.
- Negative: Doubled GRPO compute for that model. Two LoRA adapters to manage at serving time.

### ADR-005: Triton for Custom Quantization Kernels

**Status:** Accepted (carried from v2)

**Context:** TurboQuant and RotorQuant need custom encode/decode kernels not available in any framework.

**Decision:** Triton ≥3.0 fused kernels. TurboQuant adds 1-bit QJL residual. RotorQuant uses Cl(3,0) rotor sandwich (~100 FMAs vs 16,384 for dense rotation).

---

## 12. Risk Registry (Code-Level)

| # | Risk | Probability | Code Impact | Mitigation |
|---|------|-------------|-------------|------------|
| R1 | Qwen3.5-35B-A3B BF16 exceeds 80GB for GRPO | High | `training/grpo.py` | Unsloth QLoRA 4-bit (~17.5GB) + FP8 RL. Configured in `configs/training/grpo_cat_a.yaml`. |
| R2 | Same model wins Cat B + Cat C → doubled fine-tuning | Medium | `training/sft.py`, `training/grpo.py` | Share SFT base, diverge at GRPO. `scripts/run_phase2_sft.sh` detects shared winner. |
| R3 | Phase 1 winner doesn't respond to fine-tuning | Low | `training/pilot_check.py` | 100-step pilot SFT on top-2. Auto-fallback to #2 in `scripts/run_phase2_sft.sh`. |
| R4 | TurboQuant PR not merged | High | `quantization/turboquant/vllm_integration.py` | `0xSero/turboquant` fork. Standalone benchmark path outside vLLM. |
| R5 | GRPO reward hacking | Medium | `training/grpo.py`, `training/rewards/` | Multiple orthogonal reward components. Held-out eval every 50 steps. KL divergence monitoring. Auto-stop if held-out metric drops while reward increases. |
| R6 | Nemotron Mamba + vLLM incompatibility | Medium | `eval/agent_benchmark.py`, `serving/launch_vllm.sh` | HF `generate()` fallback for quality eval. Report vLLM compatibility separately. |
| R7 | GLM LoRA exceeds 80GB even with Unsloth | Medium | `training/sft.py` | Auto-reduce rank to 32. Fallback: inference-only evaluation with strong prompting. |

---

## 13. Dependency Stack

```
# requirements.txt
# Core
torch>=2.4.0
transformers>=5.0.0
unsloth>=2025.3                # SFT + GRPO RL with MoE kernels
trl>=0.15.0                    # GRPOTrainer, GRPOConfig
peft>=0.14.0                   # LoRA (used via Unsloth)
vllm>=0.8.0                    # PagedAttention v2, tool-call parsers
triton>=3.0.0                  # Custom quantization kernels
flash-attn>=2.5.0

# Data
datasets>=2.19.0
outlines>=0.0.40               # Constrained decoding (Task C)
xgrammar>=0.1.0                # Alternative constrained decoding

# Evaluation
networkx>=3.2                  # Graph edit distance
jsonschema>=4.21.0             # Tool/graph schema validation

# Analysis & Logging
wandb>=0.17.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Quantization
scipy>=1.12.0                  # Lloyd-Max codebook optimization
numpy>=1.26.0

# Utilities
pyyaml>=6.0
```

---

## 14. Testing Strategy

### 14.1 Unit Tests

| Module | Test File | Key Assertions |
|--------|-----------|----------------|
| Data generation | `test_data_generation.py` | Schema validity, behavior distribution, tool error rate, split ratios |
| Reward functions | `test_reward_functions.py` | Known-answer reward scores, edge cases (invalid JSON → 0.0), component weight sums to 1.0 |
| Eval metrics | `test_eval_metrics.py` | F1, GED, pass^5 computation against hand-computed examples |
| Composite score | `test_composite_score.py` | Normalization correctness, weight application, ranking stability |
| Triton kernels | `test_triton_kernels.py` | Numerical: encode→decode ≈ identity (within quantization error) |
| Chat templates | `test_chat_templates.py` | Round-trip fidelity across all 6 model formats |

### 14.2 Integration Tests

| Test | Description |
|------|-------------|
| Phase 1 smoke | 2 models × 1 task × 10 samples → verify composite score computation |
| SFT smoke | 50 steps on 100 samples → checkpoint saves → merge → inference |
| GRPO smoke | 10 steps with mock reward → verify reward logging + policy update |
| Reward hacking detector | Synthetic scenario: reward ↑ + held-out ↓ → verify alert fires |
| Quant round-trip | BF16 → TurboQuant encode → decode → PPL delta within tolerance |
| E2E pipeline | `run_exp_a.sh` on 1 model × 1 task × 10 samples → verify full output |

### 14.3 Reproducibility

All experiments use seed-deterministic configuration. Phase 1 benchmarks use temperature=0.0. Consistency metrics use temperature=0.7 with 5 trials (pass^5). Quantization benchmarks run 3–5 repetitions with 500+ prompts, reporting mean ± std.

---

## 15. Execution Timeline → Code Milestones

| Week | Phase | Code Deliverables |
|------|-------|-------------------|
| 1–2 | Data Preparation | `data/` module complete. L1–L5 JSONL, (prompt, graph) pairs, tool-call training data. |
| 3–4 | Phase 1: Benchmark | `benchmark/` module complete. All candidates evaluated. Ranking tables + 3 winners. |
| 5–7 | Phase 2a: SFT | `training/sft.py` complete. 3 SFT checkpoints (one per category). |
| 8–9 | Phase 2b: GRPO RL | `training/grpo.py` + `training/rewards/` complete. 3 SFT+RL checkpoints. SFT-vs-RL comparison data. |
| 10–11 | Phase 3: Quant | `quantization/` complete. TurboQuant + RotorQuant Triton kernels. Full benchmark matrix. |
| 12–13 | Phase 4: Integration | `integration/` complete. Multi-agent deployment. Concurrency test. Pareto frontier. |
| 14 | Report & Release | `analysis/` plots. Final report. Open-source codebase + model weights. |