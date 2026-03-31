# Codebase Specification: LLM Workflow-Orchestrating Agents with KV Cache Quantization

**Project:** `llm-workflow-agents-experiment`
**Version:** 2.0 — March 2026
**Hardware Target:** Single NVIDIA H100 SXM 80GB

---

## 1. System Overview

### 1.1 Purpose

This codebase implements a four-track experimental program investigating how LLMs of different sizes serve distinct roles in workflow-orchestrating agent systems, benchmarked on a single H100 SXM 80GB GPU. The four tracks are:

| Track | Scope | Models | Core Question |
|-------|-------|--------|---------------|
| **Exp A** | Prompt-encoded business logic + tool calling | 5 × 15–30B | Reliability ceiling of prompt-encoded state machines |
| **Exp B** | Fine-tuned specialist subagents | 5 × 2–5B | Small-model parity with frontier models on subtasks |
| **Exp C** | Prompt-to-graph extraction | 5 × 2–5B (same) | Structured graph output from natural-language workflows |
| **Exp D** | KV cache quantization benchmark | All 10 models × 6 methods | Memory/throughput/quality tradeoffs for multi-model serving |
| **E2E** | Integration deployment | Best combos from A–D | Pareto-optimal multi-agent configuration on single GPU |

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT ORCHESTRATION                        │
│  scripts/run_exp_{a,b,c,d,e2e}.sh → configs/ → W&B logging            │
├────────────┬────────────┬────────────┬────────────┬─────────────────────┤
│  DATA GEN  │  TRAINING  │   QUANT    │    EVAL    │     SERVING         │
│  data/     │  training/ │  quant/    │  eval/     │     serving/        │
├────────────┴────────────┴────────────┴────────────┴─────────────────────┤
│                         INFRASTRUCTURE                                  │
│  vLLM (PagedAttention v2) · PyTorch ≥2.4 · Triton ≥3.0 · HF ≥5.0     │
│  PEFT ≥0.14 · TRL ≥0.15 · W&B · NVIDIA H100 SXM 80GB                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Research Questions Mapped to Code Paths

| RQ | Description | Primary Modules |
|----|-------------|-----------------|
| RQ1 | Prompt-encoded workflow reliability ceiling | `data/generate_workflows.py` → `serving/launch_vllm.sh` → `eval/state_accuracy.py`, `eval/tool_call_f1.py` |
| RQ2 | Fine-tuned 2–5B vs frontier on subtasks | `training/train_specialist.py` → `eval/tool_call_f1.py`, `eval/state_accuracy.py` |
| RQ3 | Prompt → structured graph extraction | `data/generate_graph_pairs.py` → `training/train_graph_extractor.py` → `eval/graph_extraction_eval.py` |
| RQ4 | KV cache quantization comparison | `quantization/` → `eval/perplexity.py`, `eval/longbench.py`, `eval/needle_haystack.py` |

---

## 2. Repository Structure

```
llm-workflow-agents-experiment/
│
├── configs/                          # All YAML configuration files
│   ├── models_exp_a/                 # Experiment A: 15–30B model configs
│   │   ├── gemma3_27b.yaml
│   │   ├── qwen3_32b.yaml
│   │   ├── mistral_24b.yaml
│   │   ├── nemotron_30b.yaml
│   │   └── glm47_flash.yaml
│   ├── models_exp_bc/                # Experiments B–C: 2–5B model configs
│   │   ├── qwen25_3b.yaml
│   │   ├── qwen35_4b.yaml
│   │   ├── glm47_flash.yaml
│   │   ├── gemma_2b.yaml
│   │   └── gemma3_4b.yaml
│   ├── quantization/                 # Experiment D: quantization method configs
│   │   ├── fp8.yaml
│   │   ├── kivi_2bit.yaml
│   │   ├── kivi_4bit.yaml
│   │   ├── kvquant.yaml
│   │   ├── turboquant.yaml
│   │   └── rotorquant.yaml
│   └── serving/                      # Serving / deployment configs
│       ├── single_lora.yaml
│       ├── multi_instance.yaml
│       └── benchmark_matrix.yaml
│
├── data/                             # Data generation and templates
│   ├── generate_workflows.py         # Exp A: workflow + tool schema generation
│   ├── generate_tool_call_data.py    # Exp B: tool-call fine-tuning data
│   ├── generate_graph_pairs.py       # Exp C: (prompt, graph) pair generation
│   ├── chat_template_converter.py    # Convert between ChatML/Gemma/Mistral/Nemotron formats
│   ├── data_validator.py             # Schema validation and quality checks
│   └── templates/
│       ├── workflow_prompt_template.txt
│       ├── tool_schemas_L1_to_L5.json
│       └── graph_output_schema.json
│
├── training/                         # Fine-tuning (Exp B, C)
│   ├── train_specialist.py           # Unified SFTTrainer entry point
│   ├── train_graph_extractor.py      # Graph extraction fine-tuning
│   ├── merge_adapter.py              # LoRA → merged model export
│   └── lora_targets.py               # Per-model LoRA target module registry
│
├── quantization/                     # KV cache quantization (Exp D)
│   ├── turboquant/
│   │   ├── codebook.py               # Lloyd-Max codebook pre-computation
│   │   ├── rotation.py               # Orthogonal rotation matrix generation
│   │   ├── triton_kernels.py         # Triton encode/decode kernels
│   │   └── vllm_integration.py       # vLLM cache dtype hook
│   ├── rotorquant/
│   │   ├── clifford.py               # Cl(3,0) geometric algebra primitives
│   │   └── rotor_kernels.py          # Fused Triton rotor sandwich kernels
│   └── baselines/
│       ├── kivi_cache.py             # KIVI asymmetric quantization wrapper
│       └── kvquant_calibrate.py      # KVQuant calibration + NUQ codebook
│
├── eval/                             # Evaluation modules
│   ├── perplexity.py                 # WikiText-2, C4 PPL measurement
│   ├── longbench.py                  # LongBench 15-task evaluation
│   ├── needle_haystack.py            # Needle-in-a-Haystack (2K–32K)
│   ├── tool_call_f1.py               # BFCL-style tool-call evaluation
│   ├── tool_chain_propagation.py     # Multi-step tool chain accuracy
│   ├── state_accuracy.py             # State transition accuracy + recovery
│   ├── graph_extraction_eval.py      # Node F1, Edge F1, GED, renderability
│   └── agent_benchmark.py            # End-to-end workflow benchmark
│
├── serving/                          # vLLM serving and orchestration
│   ├── launch_vllm.sh                # vLLM server launch with model-specific args
│   ├── orchestrator.py               # Multi-agent orchestrator (E2E)
│   └── benchmark_e2e.py              # End-to-end latency/throughput measurement
│
├── analysis/                         # Result analysis and visualization
│   ├── pareto.py                     # Pareto frontier computation
│   └── plot_results.py               # Matplotlib/Seaborn chart generation
│
├── scripts/                          # Top-level experiment runners
│   ├── run_exp_a.sh
│   ├── run_exp_b.sh
│   ├── run_exp_c.sh
│   ├── run_exp_d.sh
│   └── run_exp_e2e.sh
│
├── tests/                            # Unit and integration tests
│   ├── test_data_generation.py
│   ├── test_eval_metrics.py
│   ├── test_triton_kernels.py
│   └── test_chat_templates.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Configuration Schema

### 3.1 Model Configuration (`configs/models_exp_a/*.yaml`)

```yaml
# configs/models_exp_a/gemma3_27b.yaml
model:
  name: "google/gemma-3-27b-it"
  family: "gemma"                          # Controls chat template + tool parser
  architecture: "dense_gqa_transformer"
  params_total: 27_000_000_000
  params_active: 27_000_000_000
  context_length: 131072
  precision: "bfloat16"
  vram_estimate_gb: 54

serving:
  engine: "vllm"
  tool_call_parser: "pythonic"             # vLLM tool-call parser name
  chat_template: "gemma"                   # HF chat template identifier
  gpu_memory_utilization: 0.90
  max_model_len: 8192                      # Per-experiment cap (not full context)
  enforce_eager: false

inference:
  temperature_deterministic: 0.0
  temperature_stochastic: 0.7
  stochastic_trials: 5                     # For pass^5 consistency metric
  max_tokens: 2048
```

### 3.2 Training Configuration (`configs/models_exp_bc/*.yaml`)

```yaml
# configs/models_exp_bc/qwen25_3b.yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  family: "qwen"
  params_total: 3_000_000_000
  precision: "bfloat16"

lora:
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  modules_to_save: null                    # No full fine-tuned modules

training:
  learning_rate: 2.0e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.03
  effective_batch_size: 32
  gradient_accumulation_steps: 8           # micro_batch=4 × grad_accum=8 = 32
  num_epochs: 3
  max_seq_length: 4096
  gradient_checkpointing: true
  packing: true                            # TRL ConstantLengthDataset
  save_strategy: "steps"
  save_steps: 500
  eval_steps: 500
  metric_for_best_model: "eval_loss"

training_data:
  sources:
    - "Salesforce/xlam-function-calling-60k"
    - "ToolBench"
    - "custom_synthetic"
  custom_synthetic_size: 15000
  negative_example_ratio: 0.15
  splits:
    train: 0.85
    val: 0.10
    test: 0.05

hardware:
  vram_estimate_gb: 28
  mixed_precision: "bf16"
```

### 3.3 Quantization Configuration (`configs/quantization/*.yaml`)

```yaml
# configs/quantization/turboquant.yaml
method:
  name: "turboquant"
  paper: "Zandieh et al., ICLR 2026"
  status: "PR #38280 (community fork: 0xSero/turboquant)"

codebook:
  distribution: "beta"                     # Beta(α, α)
  alpha_formula: "(d - 1) / 2"             # α depends on head dimension d
  head_dimensions: [128, 256]
  bit_widths: [2, 3, 4]
  precompute_offline: true

rotation:
  method: "qr_gaussian"                    # QR decomposition of Gaussian matrix
  seed_deterministic: true
  seed_base: 42

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
  runner_module: "vllm.worker.gpu_model_runner"
  block_size_bytes_3bit: 52                # Per 128-value vector (vs 256 bytes FP16)

benchmark:
  quality_tasks:
    - "wikitext2_ppl"
    - "c4_ppl"
    - "longbench_15task"
    - "needle_in_haystack"
    - "tool_call_f1"
  performance_metrics:
    - "peak_vram_gb"
    - "kv_cache_size_gb"
    - "throughput_prefill_tok_s"
    - "throughput_decode_tok_s"
    - "latency_ttft_ms"
    - "latency_tpot_ms"
    - "latency_itl_p50_p95_p99"
    - "max_concurrent_batch_4096ctx"
  runs: 5
  prompts_per_run: 500
```

---

## 4. Module Specifications

### 4.1 Data Generation Layer (`data/`)

#### 4.1.1 `generate_workflows.py` — Experiment A Data

**Purpose:** Generate multi-turn conversation datasets at 5 complexity levels (L1–L5) with tool-calling annotations and state-machine ground truth.

**Interface:**

```python
def generate_workflow_dataset(
    complexity_level: Literal["L1", "L2", "L3", "L4", "L5"],
    num_samples: int = 200,
    teacher_model: str = "gpt-4o",           # or "claude-sonnet-4"
    output_dir: Path = Path("data/output/exp_a"),
    seed: int = 42,
) -> DatasetMetadata:
    """
    Generate multi-turn conversation dataset for a single complexity level.

    Returns:
        DatasetMetadata with paths to generated JSONL files and statistics.
    """
```

**Complexity Level Parameters:**

```python
@dataclass
class ComplexitySpec:
    level: str
    num_states: tuple[int, int]        # (min, max)
    branching_factor: tuple[int, int]
    num_tools: int
    chain_depth: int                    # Sequential tool dependency depth
    nesting_depth: int                  # Conditional nesting depth
    domain: str                         # e.g., "faq_lookup", "booking_payment"

COMPLEXITY_SPECS = {
    "L1": ComplexitySpec("L1", (3,4),   (1,2), 1, 0, 0, "faq_lookup"),
    "L2": ComplexitySpec("L2", (5,7),   (2,3), 2, 1, 1, "order_status_cancel"),
    "L3": ComplexitySpec("L3", (8,12),  (3,5), 4, 2, 2, "booking_payment"),
    "L4": ComplexitySpec("L4", (13,20), (5,8), 6, 3, 3, "it_troubleshoot_escalation"),
    "L5": ComplexitySpec("L5", (21,30), (8,99),7, 4, 4, "multi_dept_workflow"),
}
```

**User Behavior Distribution:**

```python
USER_BEHAVIOR_DISTRIBUTION = {
    "cooperative": 0.60,
    "adversarial_probing": 0.15,
    "digressing": 0.10,
    "invalid_tool_inputs": 0.15,
}
```

**Tool Error Rate:** 20% of tool calls return error payloads for testing recovery paths.

**Output Format (per sample):**

```json
{
  "conversation_id": "L3_042",
  "complexity_level": "L3",
  "domain": "booking_payment",
  "num_states": 10,
  "num_tools": 4,
  "chain_depth": 2,
  "workflow_graph": {
    "states": ["S1", "S2", "..."],
    "transitions": [{"from": "S1", "to": "S2", "condition": "..."}],
    "initial": "S1",
    "terminal": ["S9", "S10"]
  },
  "tool_schemas": [...],
  "messages": [
    {"role": "system", "content": "<workflow prompt>"},
    {"role": "user", "content": "I need to cancel order #4521"},
    {
      "role": "assistant",
      "content": "[STATE: GREETING → COLLECT_INFO]\n<tool_call>{...}</tool_call>",
      "annotations": {
        "state_transition": {"from": "GREETING", "to": "COLLECT_INFO"},
        "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "4521"}}]
      }
    },
    {"role": "tool", "content": "{\"order_id\":\"4521\",\"status\":\"active\"}"},
    ...
  ],
  "user_behavior": "cooperative"
}
```

#### 4.1.2 `generate_tool_call_data.py` — Experiment B Data

**Purpose:** Merge external datasets (xlam-60k, ToolBench) with custom synthetic data into a unified fine-tuning JSONL.

**Interface:**

```python
def generate_tool_call_dataset(
    external_sources: list[str],             # HF dataset IDs
    custom_synthetic_size: int = 15000,
    teacher_model: str = "gpt-4o",
    negative_ratio: float = 0.15,
    output_dir: Path = Path("data/output/exp_b"),
    seed: int = 42,
) -> DatasetSplits:
    """
    Returns:
        DatasetSplits with train/val/test JSONL paths and token statistics.
    """
```

**Negative Example Categories:**
- Wrong tool selected for intent (5%)
- Hallucinated tool not in schema (4%)
- Invalid state transition (3%)
- Error recovery trajectories (3%)

#### 4.1.3 `generate_graph_pairs.py` — Experiment C Data

**Purpose:** Construct 5,000 (prompt, graph) pairs for prompt-to-graph extraction training.

**Interface:**

```python
def generate_graph_pairs(
    workflow_prompts_dir: Path,              # From Exp A (1000 prompts, 200/level)
    gold_annotations: int = 200,             # Manually annotated gold pairs
    teacher_generated: int = 800,            # GPT-4o generated, validated against gold
    augmentation_target: int = 5000,         # Paraphrase augmentation
    output_dir: Path = Path("data/output/exp_c"),
) -> DatasetSplits:
    """
    Splits: 4000 train / 500 val / 500 test
    """
```

**Graph Output Schema:**

```python
@dataclass
class WorkflowGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    initial_state: str
    terminal_states: list[str]

@dataclass
class GraphNode:
    id: str                                  # e.g. "S1"
    name: str                                # e.g. "Greeting"
    tools: list[str]                         # Tool names available in this state
    entry_actions: list[str]                 # Actions on state entry

@dataclass
class GraphEdge:
    from_state: str
    to_state: str
    condition: str                           # Natural-language condition
    priority: int                            # Evaluation order for ambiguous inputs
```

#### 4.1.4 `chat_template_converter.py`

**Purpose:** Convert unified JSONL to model-specific chat template formats.

**Supported Formats:**

| Model Family | Template Format | Tool Format |
|-------------|----------------|-------------|
| Qwen | ChatML with `<think>` | Hermes `<tool_call>` |
| Gemma | Gemma chat template | Gemma native tool format |
| Mistral | Mistral Instruct v3 | Mistral `tool_calls` JSON |
| Nemotron | Nemotron chat format | Nemotron tool format |
| GLM | GLM ChatML | GLM tool format |

```python
def convert_to_model_format(
    input_jsonl: Path,
    model_family: Literal["qwen", "gemma", "mistral", "nemotron", "glm"],
    output_path: Path,
) -> ConversionStats:
    """Convert unified JSONL to model-specific chat template."""
```

---

### 4.2 Training Layer (`training/`)

#### 4.2.1 `train_specialist.py` — Unified SFTTrainer Entry Point

**Purpose:** Fine-tune any 2–5B model with LoRA using TRL's SFTTrainer with packing.

**Interface:**

```python
def train(config_path: Path) -> TrainingResult:
    """
    Full training pipeline:
      1. Load base model in BF16
      2. Apply LoRA via PEFT to target modules (from config)
      3. Enable gradient checkpointing
      4. Configure SFTTrainer with packing + chat template
      5. Train with W&B logging
      6. Save best checkpoint by val loss (every 500 steps)
      7. Return TrainingResult with metrics and checkpoint path
    """
```

**LoRA Target Module Registry (`lora_targets.py`):**

```python
LORA_TARGET_MODULES = {
    "qwen25_3b": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "qwen35_4b": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "deltanet": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "warnings": ["QLoRA degrades hybrid DeltaNet architecture"],
    },
    "glm47_flash": {
        "attention": ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
        "shared_experts": ["shared_experts.gate_proj", "shared_experts.up_proj",
                           "shared_experts.down_proj"],
        "freeze": ["mlp.gate"],               # Router weights frozen
        "warnings": ["~60GB VRAM for BF16 LoRA — may need Unsloth MoE kernels or rank=32"],
    },
    "gemma_2b": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "gemma3_4b": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}
```

**Hyperparameter Matrix:**

| Parameter | Qwen2.5-3B | Qwen3.5-4B | GLM-4.7-F | Gemma-2B | Gemma3-4B |
|-----------|-----------|-----------|----------|---------|----------|
| Precision | BF16 | BF16 | BF16 | BF16/QLoRA | BF16 |
| LoRA rank | 64 | 64 | 64 | 16 | 64 |
| LoRA alpha | 128 | 128 | 64 | 32 | 32 |
| LR | 2e-4 | 1e-4 | 5e-5 | 2e-4 | 1e-4 |
| Eff. batch | 32 | 16–32 | 8–16 | 32 | 16 |
| Max seq | 4096 | 4096 | 2048 | 2048 | 2048 |
| Est. VRAM | ~28 GB | ~22 GB | ~60 GB | ~12 GB | ~18 GB |

#### 4.2.2 `train_graph_extractor.py` — Experiment C

**Purpose:** Fine-tune models for graph extraction using same LoRA infrastructure but with structured JSON output training.

```python
def train_graph_extractor(config_path: Path) -> TrainingResult:
    """
    Same LoRA pipeline as train_specialist.py, but:
      - System prompt instructs JSON graph extraction
      - User = workflow prompt text
      - Assistant = JSON graph (WorkflowGraph schema)
      - At inference: apply Outlines/XGrammar constrained decoding
    """
```

#### 4.2.3 `merge_adapter.py`

```python
def merge_and_export(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    push_to_hub: bool = False,
) -> None:
    """Load base + LoRA adapter, merge via model.merge_and_unload(), save."""
```

---

### 4.3 Quantization Layer (`quantization/`)

#### 4.3.1 TurboQuant Pipeline (`quantization/turboquant/`)

**`codebook.py` — Lloyd-Max Codebook Pre-computation:**

```python
def precompute_codebooks(
    head_dimensions: list[int] = [128, 256],
    bit_widths: list[int] = [2, 3, 4],
    output_dir: Path = Path("quantization/turboquant/codebooks"),
) -> dict[tuple[int, int], np.ndarray]:
    """
    Pre-compute Lloyd-Max codebooks for Beta(α, α) distribution.
    α = (d - 1) / 2 where d is head dimension.

    One-time offline computation. Outputs:
        codebooks[(d, bits)] → np.ndarray of shape (2^bits,)
    """
```

**`rotation.py` — Orthogonal Rotation Matrix:**

```python
def generate_rotation_matrix(
    d: int,                                  # Head dimension
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate random orthogonal matrix Π via QR decomposition
    of a Gaussian random matrix. Seed-deterministic per model.

    Returns: Π of shape (d, d), orthogonal (Π^T Π = I)
    """
```

**`triton_kernels.py` — Fused Encode/Decode:**

```python
@triton.jit
def turboquant_encode_kernel(
    kv_ptr,                    # Input KV cache values [batch, heads, seq, d]
    rotation_ptr,              # Rotation matrix Π [d, d]
    codebook_ptr,              # Lloyd-Max codebook [2^bits]
    output_indices_ptr,        # Quantized indices [batch, heads, seq, d]
    output_norms_ptr,          # Per-vector norms [batch, heads, seq]
    # ... block sizes, strides
):
    """
    Fused pipeline: rotate → quantize → pack indices + store norm.
    Hooked into vLLM cache write path.
    """

@triton.jit
def turboquant_decode_kernel(
    indices_ptr,               # Packed quantized indices
    norms_ptr,                 # Stored norms
    codebook_ptr,              # Lloyd-Max codebook
    inv_rotation_ptr,          # Π^T (inverse rotation)
    output_ptr,                # Reconstructed KV values
    # ... block sizes, strides
):
    """
    Fused pipeline: unpack → lookup centroids → inverse rotate.
    Hooked into vLLM cache read path.
    """
```

**`vllm_integration.py` — vLLM Hook Registration:**

```python
def register_turboquant_backend():
    """
    1. Register kv_cache_dtype="turboquant" in vllm.config.cache
    2. Modify block size in gpu_model_runner.py
       - 3-bit: 52 bytes per 128-value vector (vs 256 bytes FP16)
    3. Hook encode kernel into cache write path
    4. Hook decode kernel into cache read path
    """
```

#### 4.3.2 RotorQuant Pipeline (`quantization/rotorquant/`)

**`clifford.py` — Cl(3,0) Geometric Algebra Primitives:**

```python
class CliffordAlgebra:
    """
    Cl(3,0) algebra implementation for rotor-based rotation.

    Key operations:
      - embed(v: R^d) → Cl(3,0) multivector
      - rotor_sandwich(R, x) → R x R† (~100 FMAs for d=128)
        vs dense rotation: 16,384 FMAs for d=128
      - extract(mv) → R^d
    """

    def rotor_from_params(self, params: torch.Tensor) -> Rotor:
        """Construct Cl(3,0) rotor from learnable parameters."""

    def sandwich_product(self, rotor: Rotor, x: torch.Tensor) -> torch.Tensor:
        """Apply R x R† rotation. ~100 FMAs vs 16,384 for dense."""
```

**`rotor_kernels.py` — Fused Triton Kernel:**

```python
@triton.jit
def rotorquant_fused_kernel(
    kv_ptr,                    # Input KV cache
    rotor_params_ptr,          # Cl(3,0) rotor parameters
    codebook_ptr,              # Grade-aware Lloyd-Max codebook
    output_ptr,                # Quantized output
    # ...
):
    """
    Fused pipeline: embed → rotor sandwich → quantize → inverse → extract.
    Grade-aware codebooks account for Clifford algebra structure.
    """
```

#### 4.3.3 Baselines (`quantization/baselines/`)

**`kivi_cache.py`:** Wrapper around KIVI (ICML 2024) asymmetric quantization — per-channel for Keys, per-token for Values. No calibration required.

**`kvquant_calibrate.py`:** KVQuant (NeurIPS 2024) with pre-RoPE quantization, non-uniform quantization (NUQ) codebooks, and dense-sparse decomposition. Requires per-model calibration pass.

---

### 4.4 Evaluation Layer (`eval/`)

#### 4.4.1 State-Machine Adherence (`state_accuracy.py`)

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
) -> StateMachineMetrics:
    """
    Parse [STATE: X → Y] annotations from model output.
    Compare against ground-truth transition sequences.
    For pass^5: all 5 temperature=0.7 trials must reach correct terminal.
    """
```

#### 4.4.2 Tool-Calling Accuracy (`tool_call_f1.py`)

```python
@dataclass
class ToolCallMetrics:
    tool_name_accuracy: float              # Target: ≥90%
    argument_exact_match: float            # Target: ≥75%
    tool_call_f1: float                    # Target: ≥85% (BFCL-style AST match)
    hallucinated_tool_rate: float          # Target: ≤3%
    error_recovery_rate: float             # Target: ≥60%

def evaluate_tool_calls(
    predictions: list[TurnPrediction],
    ground_truth: list[TurnGroundTruth],
    tool_schemas: list[ToolSchema],
) -> ToolCallMetrics:
    """
    Parse <tool_call>{JSON}</tool_call> from model output.
    AST sub-tree matching for argument comparison (BFCL style).
    """
```

#### 4.4.3 Tool Chain Propagation (`tool_chain_propagation.py`)

```python
@dataclass
class ChainPropagationMetrics:
    chain_propagation_accuracy: float      # Target: ≥70%
    per_depth_accuracy: dict[int, float]   # Accuracy by chain depth (1, 2, 3, 4+)

def evaluate_chain_propagation(
    predictions: list[ConversationPrediction],
    ground_truth: list[ConversationGroundTruth],
) -> ChainPropagationMetrics:
    """
    Evaluate whether return values from tool N correctly populate
    arguments of tool N+1 in multi-step chains.

    Tracks per-depth accuracy to identify where propagation breaks down.
    """
```

#### 4.4.4 Combined Workflow Quality (`agent_benchmark.py`)

```python
@dataclass
class WorkflowQualityMetrics:
    full_workflow_success: float            # Target: ≥55%
    weighted_workflow_score: float          # Target: ≥0.75
    latency_per_turn_median_ms: float      # Target: ≤2000 (L1-L3), ≤5000 (L4-L5)

    # Composition
    state_metrics: StateMachineMetrics
    tool_metrics: ToolCallMetrics
    chain_metrics: ChainPropagationMetrics

def compute_weighted_score(
    state: StateMachineMetrics,
    tool: ToolCallMetrics,
    completion: float,
) -> float:
    """0.4 × StateTransAcc + 0.4 × ToolCallF1 + 0.2 × TaskCompletion"""
    return 0.4 * state.state_transition_accuracy + \
           0.4 * tool.tool_call_f1 + \
           0.2 * completion
```

#### 4.4.5 Graph Extraction (`graph_extraction_eval.py`)

```python
@dataclass
class GraphExtractionMetrics:
    node_f1: float                         # Target: ≥85%
    edge_f1: float                         # Target: ≥75%
    graph_edit_distance: float             # Target: ≤0.20 (normalized)
    json_validity: float                   # Target: ≥95%
    structural_validity: float             # Target: ≥90%
    mermaid_renderability: float           # Target: ≥90%

def evaluate_graph_extraction(
    predicted_graphs: list[dict],
    gold_graphs: list[WorkflowGraph],
) -> GraphExtractionMetrics:
    """
    Structural validity checks:
      - Valid initial state exists
      - Terminal states are reachable from initial
      - No orphan nodes (unreachable from any path)

    Mermaid renderability: attempt Mermaid.js render, check for errors.
    """
```

#### 4.4.6 Quantization Quality (`perplexity.py`, `longbench.py`, `needle_haystack.py`)

```python
def evaluate_perplexity(
    model_path: str,
    datasets: list[Literal["wikitext2", "c4"]],
    kv_cache_dtype: str = "auto",
) -> dict[str, float]:
    """Returns {dataset_name: perplexity_value}"""

def evaluate_longbench(
    model_path: str,
    tasks: int = 15,                        # All 15 LongBench tasks
    kv_cache_dtype: str = "auto",
) -> dict[str, float]:
    """Returns {task_name: score}"""

def evaluate_needle_in_haystack(
    model_path: str,
    context_lengths: list[int] = [2048, 4096, 8192, 16384, 32768],
    kv_cache_dtype: str = "auto",
) -> dict[int, float]:
    """Returns {context_length: retrieval_accuracy}"""
```

---

### 4.5 Serving Layer (`serving/`)

#### 4.5.1 `launch_vllm.sh` — Model Server Launch

```bash
#!/bin/bash
# Usage: ./launch_vllm.sh <model_config.yaml> [--kv-cache-dtype <dtype>]

# Reads from YAML config:
#   - model.name → --model
#   - serving.tool_call_parser → --tool-call-parser
#   - serving.gpu_memory_utilization → --gpu-memory-utilization
#   - serving.max_model_len → --max-model-len
#   - Optional: quantization.kv_cache_dtype → --kv-cache-dtype

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --dtype bfloat16 \
    --tool-call-parser "${TOOL_PARSER}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --enable-auto-tool-choice \
    ${KV_CACHE_ARGS} \
    --port 8000
```

#### 4.5.2 `orchestrator.py` — Multi-Agent E2E (Integration Experiment)

```python
class MultiAgentOrchestrator:
    """
    Deploys best (model, quantization) combinations on a single H100:
      - Orchestrator: 15–30B model from Exp A (e.g., Qwen3-32B + AWQ-INT4)
      - Specialists: 2–5B models from Exp B (via vLLM LoRA multi-adapter)

    Uses vLLM's LoRA multi-adapter serving for same-architecture specialists.
    """

    def __init__(
        self,
        orchestrator_config: Path,
        specialist_configs: list[Path],
        kv_cache_dtype: str = "turboquant",
    ):
        ...

    async def run_workflow(
        self,
        conversation: list[dict],
        workflow_graph: WorkflowGraph,
    ) -> WorkflowResult:
        """
        Route turns to orchestrator or specialist based on current state.
        Record per-turn latency, tool calls, state transitions.
        """
```

#### 4.5.3 `benchmark_e2e.py` — Concurrency and Pareto Measurement

```python
def benchmark_concurrency(
    kv_cache_dtype: str,
    context_length: int = 4096,
) -> ConcurrencyResult:
    """
    Measure max concurrent requests at given context length.

    Expected results:
      BF16:         ~175 concurrent (96 KB/tok, ~64GB avail)
      FP8:          ~350 concurrent (48 KB/tok)
      TQ 3-bit:     ~925 concurrent (~18 KB/tok)
    """

def compute_pareto_frontier(
    results: list[ExperimentResult],
    axes: tuple[str, str, str] = ("task_completion", "peak_vram_gb", "p95_latency_ms"),
) -> list[ExperimentResult]:
    """
    Identify Pareto-optimal configs across quality × memory × latency.

    Expected optima:
      - Lowest latency:    Gemma-2B + FP8
      - Best quality:      Qwen2.5-3B + TurboQuant-3.5bit
      - Max concurrency:   Gemma-2B + TurboQuant-3bit
    """
```

---

## 5. Data Flow Diagrams

### 5.1 Experiment A Pipeline

```
┌──────────────┐   L1–L5 JSONL    ┌───────────────┐   vLLM API    ┌──────────────┐
│ generate_     │ ──────────────→  │  launch_vllm   │ ───────────→ │  5 × 15–30B  │
│ workflows.py  │                  │  (per model)   │              │  models       │
└──────────────┘                   └───────────────┘              └──────┬───────┘
                                                                         │
    ┌────────────────────────────────────────────────────────────────────┘
    │  Per-turn predictions
    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ state_        │   │ tool_call_   │   │ tool_chain_  │   │ agent_       │
│ accuracy.py   │   │ f1.py        │   │ propagation  │   │ benchmark.py │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
    │                   │                   │                   │
    └───────────────────┴───────────────────┴───────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  plot_results.py     │
                    │  accuracy vs. level  │
                    └──────────────────────┘
```

### 5.2 Experiment B–C Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ xlam-60k     │   │ ToolBench    │   │ Custom       │
│ (HF)         │   │ (HF)        │   │ synthetic    │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       └──────────────────┴──────────────────┘
                          │
                ┌─────────▼──────────┐
                │ generate_tool_call │      ┌──────────────────┐
                │ _data.py           │      │ generate_graph_  │
                └─────────┬──────────┘      │ pairs.py         │
                          │                 └────────┬─────────┘
              ┌───────────▼──────────┐    ┌──────────▼─────────┐
              │ train_specialist.py  │    │ train_graph_        │
              │ (5 models × LoRA)   │    │ extractor.py        │
              └───────────┬──────────┘    └──────────┬─────────┘
                          │                          │
              ┌───────────▼──────────┐    ┌──────────▼─────────┐
              │ merge_adapter.py     │    │ merge_adapter.py    │
              └───────────┬──────────┘    └──────────┬─────────┘
                          │                          │
              ┌───────────▼──────────┐    ┌──────────▼─────────┐
              │ Eval: tool_call_f1,  │    │ Eval: graph_       │
              │ state_accuracy       │    │ extraction_eval    │
              └──────────────────────┘    └────────────────────┘
```

### 5.3 Experiment D Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                 10 Models (5 Exp A + 5 Exp B)            │
└─────────────────────────┬────────────────────────────────┘
                          │
    ┌─────────┬───────────┼───────────┬──────────┬─────────┐
    ▼         ▼           ▼           ▼          ▼         ▼
┌───────┐ ┌───────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ FP8   │ │ KIVI  │ │ KVQuant │ │ AWQ+   │ │ Turbo  │ │ Rotor  │
│ E4M3  │ │ 2/4b  │ │ 2-4b   │ │ FP8    │ │ Quant  │ │ Quant  │
└───┬───┘ └───┬───┘ └────┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
    └─────────┴──────────┴──────────┴──────────┴──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
        ┌──────────┐   ┌──────────────┐   ┌──────────────┐
        │ Quality  │   │ Performance  │   │ Per-model ×  │
        │ PPL,     │   │ VRAM, tok/s, │   │ method       │
        │ LongBench│   │ latency      │   │ matrix       │
        └──────────┘   └──────────────┘   └──────┬───────┘
                                                  │
                                          ┌───────▼───────┐
                                          │  pareto.py    │
                                          └───────────────┘
```

---

## 6. Key Architecture Decisions

### ADR-001: vLLM as Unified Serving Backend

**Status:** Accepted

**Context:** Need a single inference engine supporting BF16, FP8, custom KV cache dtypes, LoRA multi-adapter serving, and tool-call parsing across 5 model families.

**Decision:** Use vLLM ≥0.8.x with PagedAttention v2 as the sole serving backend.

**Consequences:**
- Positive: Unified API, built-in tool-call parsers (hermes, mistral), LoRA hot-swap, community quantization PR support.
- Negative: Nemotron-3-Nano Mamba layers may not be fully supported (Risk R3). TurboQuant requires community fork (Risk R4). RotorQuant has no integration (Risk R5).

### ADR-002: LoRA over Full Fine-Tuning

**Status:** Accepted

**Context:** Five 2–5B models must be fine-tuned for two tasks (specialist + graph extraction) on a single H100 80GB.

**Decision:** Use PEFT LoRA with per-model target module selection. Full fine-tuning is infeasible for GLM-4.7-Flash (~60GB BF16 base alone).

**Consequences:**
- Positive: All models trainable on single GPU. LoRA adapters enable vLLM multi-adapter serving.
- Negative: QLoRA degrades Qwen3.5-4B hybrid DeltaNet. GLM-4.7-Flash at rank 64 may still exceed VRAM — fallback: rank 32 or Unsloth MoE kernels.

### ADR-003: Triton for Custom Quantization Kernels

**Status:** Accepted

**Context:** TurboQuant and RotorQuant require custom encode/decode kernels not available in any framework.

**Decision:** Write Triton ≥3.0 kernels for both methods, with fused pipelines (rotate → quantize → pack in single kernel launch).

**Consequences:**
- Positive: Full control over memory layout and computation. Triton compiles to H100-optimized PTX.
- Negative: Significant development effort. Triton debugging is harder than PyTorch. Kernel correctness requires careful numerical testing.

### ADR-004: Teacher-Model Synthetic Data Generation

**Status:** Accepted

**Context:** No public dataset exists for multi-turn workflow conversations with state-machine annotations and tool-chain dependencies.

**Decision:** Use GPT-4o / Claude Sonnet 4 as teacher model to synthesize training data, with human gold-standard validation for Experiment C.

**Consequences:**
- Positive: High-quality, domain-specific data with ground-truth annotations. Scalable to arbitrary complexity levels.
- Negative: Teacher model cost. Potential teacher bias in synthetic data. 200 gold-standard annotations for Exp C may be insufficient for validation.

---

## 7. Risk Registry (Code-Level)

| # | Risk | Code Impact | Mitigation |
|---|------|-------------|------------|
| R1 | Qwen3-32B BF16 ~64GB → 16GB for KV | `configs/models_exp_a/qwen3_32b.yaml` needs AWQ-INT4 fallback | Add `weight_quantization: "awq_int4"` option in config |
| R2 | HF Transformers ≥5.0 dependency | `requirements.txt` version pin | Pin commit hashes for GLM/Nemotron model code |
| R3 | Nemotron Mamba + vLLM incompatibility | `serving/launch_vllm.sh` may fail | Add HF `generate()` fallback path in eval scripts |
| R4 | TurboQuant PR not merged | `quantization/turboquant/vllm_integration.py` | Use `0xSero/turboquant` fork; add standalone benchmark path |
| R5 | RotorQuant no vLLM integration | `quantization/rotorquant/` | Benchmark as standalone HF modification; report rotation speedup only |
| R6 | Tool-call parser differences | `eval/tool_call_f1.py` | Normalize all formats to canonical JSON before scoring |
| R7 | GLM LoRA VRAM overflow | `training/train_specialist.py` | Auto-fallback: rank 64 → 32 → inference-only |
| R8 | Mistral sliding-window context loss | `eval/state_accuracy.py` | Separate long-conversation (>20 turns) results in reporting |

---

## 8. Dependency Stack

```
# requirements.txt (pinned for reproducibility)
torch>=2.4.0
transformers>=5.0.0
vllm>=0.8.0
peft>=0.14.0
trl>=0.15.0
triton>=3.0.0
flash-attn>=2.5.0
datasets>=2.19.0
wandb>=0.17.0
outlines>=0.0.40            # Constrained decoding for Exp C
xgrammar>=0.1.0             # Alternative constrained decoding
numpy>=1.26.0
scipy>=1.12.0               # Lloyd-Max codebook optimization
networkx>=3.2               # Graph edit distance computation
matplotlib>=3.8.0
seaborn>=0.13.0
pyyaml>=6.0
jsonschema>=4.21.0          # Tool schema validation
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

| Module | Test File | Key Assertions |
|--------|-----------|----------------|
| Data generation | `test_data_generation.py` | Schema validity, behavior distribution, tool error rate |
| Chat templates | `test_chat_templates.py` | Round-trip conversion fidelity across all 5 formats |
| Eval metrics | `test_eval_metrics.py` | Known-answer tests for F1, GED, pass^5 computation |
| Triton kernels | `test_triton_kernels.py` | Numerical correctness: encode→decode ≈ identity (within quantization error) |

### 9.2 Integration Tests

| Test | Description |
|------|-------------|
| Single-model smoke | Generate 10 L1 samples → serve model → evaluate → check metric format |
| LoRA training smoke | Train 50 steps on 100 samples → verify checkpoint saves → merge → inference |
| Quantization round-trip | BF16 → TurboQuant encode → decode → compare PPL delta within tolerance |
| E2E pipeline | Run `run_exp_a.sh` on 1 model × 1 level × 10 samples → verify all outputs |

### 9.3 Benchmark Reproducibility

All experiments use seed-deterministic configuration. Each quantization benchmark runs 3–5 repetitions with 500+ prompts, reporting mean ± std. Temperature=0.0 for deterministic evaluation; temperature=0.7 with 5 trials for consistency (pass^5).

---

## 10. Execution Timeline → Code Milestones

| Week | Phase | Code Deliverables |
|------|-------|-------------------|
| 1–2 | Data Preparation | `data/` module complete. 5× JSONL datasets with tool chains generated. |
| 3–4 | Fine-Tuning | `training/` module complete. 10 LoRA adapters (5 specialist + 5 graph). |
| 5–6 | Experiment A | `eval/state_accuracy.py`, `eval/tool_call_f1.py`, `eval/tool_chain_propagation.py` producing accuracy-vs-complexity curves. |
| 7–8 | Quant Implementation | `quantization/turboquant/` and `quantization/rotorquant/` with working Triton kernels. |
| 9–10 | Experiment D | Full `eval/perplexity.py`, `eval/longbench.py`, `eval/needle_haystack.py` benchmark matrix. |
| 11 | Experiment C | `eval/graph_extraction_eval.py` producing Node F1, Edge F1, GED on 500 test pairs. |
| 12–14 | Integration | `serving/orchestrator.py`, `analysis/pareto.py` → final report and codebase release. |
