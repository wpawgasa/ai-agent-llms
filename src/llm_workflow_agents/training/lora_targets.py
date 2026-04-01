"""Per-model LoRA target module registry.

Maps model identifiers to their LoRA-compatible module names, with
per-model warnings and constraints (e.g., frozen router weights for MoE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from llm_workflow_agents.config.schema import ModelFamily

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LoRATargetSpec:
    """Specification for LoRA target modules of a model.

    All sequence fields are tuples to preserve true immutability with frozen=True.
    (Lists inside a frozen dataclass can still be mutated; tuples cannot.)
    """

    target_modules: tuple[str, ...]
    modules_to_freeze: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


# --- Registry ---

LORA_TARGET_MODULES: dict[str, LoRATargetSpec] = {
    "qwen25_3b": LoRATargetSpec(
        target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    ),
    "qwen35_4b": LoRATargetSpec(
        target_modules=(
            # Standard attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # DeltaNet layers
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
            # MLP
            "gate_proj", "up_proj", "down_proj",
        ),
        warnings=("QLoRA degrades hybrid DeltaNet architecture",),
    ),
    "glm47_flash": LoRATargetSpec(
        target_modules=(
            # MLA attention
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
            # Shared experts
            "shared_experts.gate_proj", "shared_experts.up_proj",
            "shared_experts.down_proj",
        ),
        modules_to_freeze=("mlp.gate",),
        warnings=("~60GB VRAM for BF16 LoRA — may need Unsloth MoE kernels or rank=32",),
    ),
    "gemma_2b": LoRATargetSpec(
        target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    ),
    "gemma3_4b": LoRATargetSpec(
        target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    ),
    # --- Cat A models ---
    "qwen3_32b": LoRATargetSpec(
        target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    ),
    "qwen35_35b_a3b": LoRATargetSpec(
        target_modules=(
            # Standard attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # DeltaNet layers
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
            # MLP
            "gate_proj", "up_proj", "down_proj",
        ),
        modules_to_freeze=("mlp.gate",),
        warnings=("QLoRA 4-bit required (~17.5GB). DeltaNet hybrid architecture.",),
    ),
    "nemotron_30b": LoRATargetSpec(
        target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
        modules_to_freeze=("mlp.gate",),
        warnings=("Mamba layers: Unsloth auto-detect. vLLM compat uncertain (R6).",),
    ),
    "mistral_24b": LoRATargetSpec(
        target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    ),
    "gemma3_27b": LoRATargetSpec(
        target_modules=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    ),
}

# Model family to default target modules (fallback when specific model not in registry)
_FAMILY_DEFAULTS: dict[ModelFamily, tuple[str, ...]] = {
    ModelFamily.QWEN: ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ModelFamily.QWEN35: ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"),
    ModelFamily.GEMMA: ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ModelFamily.MISTRAL: ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ModelFamily.NEMOTRON: ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ModelFamily.GLM: ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"),
}

# Model name pattern matching (HF model ID substring -> registry key)
_MODEL_NAME_PATTERNS: dict[str, str] = {
    # More specific patterns first (order matters for substring matching)
    "qwen3.5-35b": "qwen35_35b_a3b",
    "qwen3.5-4b": "qwen35_4b",
    "qwen3-32b": "qwen3_32b",
    "qwen2.5-3b": "qwen25_3b",
    "gemma-3-27b": "gemma3_27b",
    "gemma-3-4b": "gemma3_4b",
    "gemma-2b": "gemma_2b",
    "mistral-small": "mistral_24b",
    "nemotron": "nemotron_30b",
    "glm-4": "glm47_flash",
}


def detect_model_key(model_name: str) -> str | None:
    """Detect registry key from HF model ID via pattern matching."""
    name_lower = model_name.lower()
    for pattern, key in _MODEL_NAME_PATTERNS.items():
        if pattern in name_lower:
            return key
    return None


def get_lora_target_spec(
    model_name: str,
    model_family: ModelFamily | None = None,
    explicit_targets: list[str] | None = None,
) -> LoRATargetSpec:
    """Get LoRA target modules for a model.

    Priority order:
      1. Explicit targets from config (if provided)
      2. Model-specific registry entry (by model name pattern)
      3. Family-level defaults

    Args:
        model_name: HF model ID (e.g., "Qwen/Qwen2.5-3B-Instruct").
        model_family: Model family enum for fallback.
        explicit_targets: Explicitly configured target modules (highest priority).

    Returns:
        LoRATargetSpec with resolved target modules and any warnings.
    """
    # Priority 1: explicit config
    if explicit_targets:
        logger.debug("using_explicit_lora_targets", count=len(explicit_targets))
        return LoRATargetSpec(target_modules=tuple(explicit_targets))

    # Priority 2: model-specific registry
    model_key = detect_model_key(model_name)
    if model_key and model_key in LORA_TARGET_MODULES:
        spec = LORA_TARGET_MODULES[model_key]
        for warning in spec.warnings:
            logger.warning("lora_target_warning", model=model_name, warning=warning)
        return spec

    # Priority 3: family defaults
    if model_family and model_family in _FAMILY_DEFAULTS:
        logger.debug("using_family_default_targets", family=model_family)
        return LoRATargetSpec(target_modules=_FAMILY_DEFAULTS[model_family])

    # No match — return empty (PEFT auto-detection)
    logger.warning("no_lora_targets_found", model=model_name)
    return LoRATargetSpec(target_modules=())


def get_trainable_param_summary(model: Any) -> dict[str, Any]:
    """Summarize trainable vs total parameters for a PEFT model.

    Defers torch import to avoid GPU requirement at module level.
    """
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    pct = (trainable / total * 100) if total > 0 else 0.0
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(pct, 4),
    }
