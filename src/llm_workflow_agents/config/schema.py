"""Pydantic schema definitions for all configuration types."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


# --- Enums ---


class ModelFamily(StrEnum):
    QWEN = "qwen"
    QWEN35 = "qwen35"
    GEMMA = "gemma"
    MISTRAL = "mistral"
    NEMOTRON = "nemotron"
    GLM = "glm"


class KVCacheDtype(StrEnum):
    AUTO = "auto"
    FP8 = "fp8"
    KIVI = "kivi"
    KVQUANT = "kvquant"
    AWQ_FP8 = "awq_fp8"
    TURBOQUANT = "turboquant"
    ROTORQUANT = "rotorquant"


class ComplexityLevel(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


# --- Config Models ---


class SpeculativeConfig(BaseModel):
    """Optional speculative-decoding sub-block. Fields are engine-native and informational;
    bash launchers bypass Pydantic and read YAML directly."""

    enabled: bool = False
    # Unified / vLLM
    method: str | None = None
    draft_model: str | None = None
    num_speculative_tokens: int | None = None
    extra: dict | None = None
    # vLLM Dflash extras (merged into --speculative-config JSON)
    dflash_block_size: int | None = None
    dflash_draft_window_size: int | None = None
    # SGLang-native
    algorithm: str | None = None
    draft_model_path: str | None = None
    num_draft_tokens: int | None = None
    # TRT-LLM-native
    mode: str | None = None
    build_recipe: str | None = None
    build_params: dict | None = None

    model_config = {"extra": "allow"}


class ServingConfig(BaseModel):
    """Serving configuration (engine-agnostic; engine-specific fields are read by bash launchers)."""

    engine: str = "vllm"
    tool_call_parser: str = "hermes"
    chat_template: str = "chatml"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    enforce_eager: bool = False
    speculative: SpeculativeConfig | None = None

    model_config = {"extra": "allow"}


class InferenceConfig(BaseModel):
    """Inference parameters."""

    temperature_deterministic: float = 0.0
    temperature_stochastic: float = 0.7
    stochastic_trials: int = 5
    max_tokens: int = 2048


class ModelConfig(BaseModel):
    """Model configuration for Phase 1 benchmarking."""

    name: str
    family: ModelFamily
    architecture: str = "dense_gqa_transformer"
    params_total: int
    params_active: int | None = None
    context_length: int = 131072
    precision: str = "bfloat16"
    vram_estimate_gb: float | None = None

    serving: ServingConfig = Field(default_factory=ServingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    category: str | None = None
    benchmark_tasks: list[str] = Field(default_factory=list)


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=list)
    modules_to_save: list[str] | None = None


class TrainingDataConfig(BaseModel):
    """Training data sources and splits."""

    sources: list[str] = Field(default_factory=list)
    custom_synthetic_size: int = 15000
    negative_example_ratio: float = 0.15
    splits: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.85, "val": 0.10, "test": 0.05}
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    effective_batch_size: int = 32
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_seq_length: int = 4096
    gradient_checkpointing: bool = True
    packing: bool = True
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_steps: int = 500
    metric_for_best_model: str = "eval_loss"

    training_data: TrainingDataConfig = Field(default_factory=TrainingDataConfig)
    hardware_vram_estimate_gb: float | None = None
    mixed_precision: str = "bf16"


class TrainingModelConfig(BaseModel):
    """Model config with LoRA and training settings for Exp B/C."""

    model: ModelConfig
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class QuantizationMethodConfig(BaseModel):
    """Quantization method configuration for Experiment D."""

    name: str
    paper: str | None = None
    status: str | None = None
    bit_widths: list[int] = Field(default_factory=lambda: [2, 3, 4])
    head_dimensions: list[int] = Field(default_factory=lambda: [128, 256])
    kv_cache_dtype: str = "auto"
    requires_calibration: bool = False
    benchmark_runs: int = 5
    benchmark_prompts_per_run: int = 500

    quality_tasks: list[str] = Field(
        default_factory=lambda: [
            "wikitext2_ppl",
            "c4_ppl",
            "longbench_15task",
            "needle_in_haystack",
            "tool_call_f1",
        ]
    )
    performance_metrics: list[str] = Field(
        default_factory=lambda: [
            "peak_vram_gb",
            "kv_cache_size_gb",
            "throughput_prefill_tok_s",
            "throughput_decode_tok_s",
            "latency_ttft_ms",
            "latency_tpot_ms",
            "latency_itl_p50_p95_p99",
            "max_concurrent_batch_4096ctx",
        ]
    )


class ComplexitySpec(BaseModel):
    """Workflow complexity level specification.

    New semantic-graph fields drive ``select_subgraph`` (Task 5+):
    - ``target_path_len``: number of spine states to include from the canonical domain graph.
    - ``num_branches``: optional branch edges (off-spine arcs) to add.
    - ``num_loops``: retry / escalation back-edges to add.
    - ``include_recovery``: whether to include ``tool_error`` recovery arcs.

    Legacy fields are kept for backward compatibility with the old random-walk
    generator and will be removed in Task 10 once the generator is rewritten.
    """

    level: ComplexityLevel
    # New semantic subgraph selection fields
    target_path_len: tuple[int, int] = (0, 0)
    num_branches: tuple[int, int] = (0, 0)
    num_loops: tuple[int, int] = (0, 0)
    include_recovery: bool = False
    num_tools: int = 1
    # Legacy fields (backward compat — superseded by the fields above)
    num_states: tuple[int, int] = (0, 0)
    branching_factor: tuple[int, int] = (0, 0)
    chain_depth: int = 0
    nesting_depth: int = 0
    domain: str = ""


class ServingDeploymentConfig(BaseModel):
    """Serving deployment configuration."""

    mode: str = "single_model"
    kv_cache_dtype: str = "auto"
    port: int = 8000
    models: list[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str
    description: str = ""
    models: list[str] = Field(default_factory=list)
    seed: int = 42
    output_dir: Path = Path("results")


# --- Complexity Level Registry ---


COMPLEXITY_SPECS: dict[ComplexityLevel, ComplexitySpec] = {
    ComplexityLevel.L1: ComplexitySpec(
        level=ComplexityLevel.L1,
        target_path_len=(3, 4),
        num_branches=(0, 0),
        num_loops=(0, 0),
        include_recovery=False,
        num_tools=1,
        # Legacy fields
        num_states=(3, 4),
        branching_factor=(1, 2),
        chain_depth=0,
        nesting_depth=0,
        domain="faq_lookup",
    ),
    ComplexityLevel.L2: ComplexitySpec(
        level=ComplexityLevel.L2,
        target_path_len=(5, 7),
        num_branches=(1, 1),
        num_loops=(0, 0),
        include_recovery=False,
        num_tools=2,
        # Legacy fields
        num_states=(5, 7),
        branching_factor=(1, 2),
        chain_depth=1,
        nesting_depth=1,
        domain="order_status_cancel",
    ),
    ComplexityLevel.L3: ComplexitySpec(
        level=ComplexityLevel.L3,
        target_path_len=(8, 12),
        num_branches=(2, 3),
        num_loops=(0, 1),
        include_recovery=True,
        num_tools=4,
        # Legacy fields
        num_states=(8, 12),
        branching_factor=(2, 4),
        chain_depth=2,
        nesting_depth=2,
        domain="booking_payment",
    ),
    ComplexityLevel.L4: ComplexitySpec(
        level=ComplexityLevel.L4,
        target_path_len=(12, 16),
        num_branches=(3, 5),
        num_loops=(1, 1),
        include_recovery=True,
        num_tools=6,
        # Legacy fields
        num_states=(12, 16),
        branching_factor=(3, 5),
        chain_depth=3,
        nesting_depth=3,
        domain="it_troubleshoot_escalation",
    ),
    ComplexityLevel.L5: ComplexitySpec(
        level=ComplexityLevel.L5,
        target_path_len=(16, 20),
        num_branches=(5, 99),
        num_loops=(1, 2),
        include_recovery=True,
        num_tools=7,
        # Legacy fields
        num_states=(16, 20),
        branching_factor=(5, 99),
        chain_depth=4,
        nesting_depth=4,
        domain="multi_dept_workflow",
    ),
}

# --- User Behavior Distribution ---

USER_BEHAVIOR_DISTRIBUTION: dict[str, float] = {
    "cooperative": 0.60,
    "adversarial_probing": 0.15,
    "digressing": 0.10,
    "invalid_tool_inputs": 0.15,
}

TOOL_ERROR_RATE: float = 0.20
