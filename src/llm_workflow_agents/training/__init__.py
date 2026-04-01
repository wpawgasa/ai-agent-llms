"""Training module for SFT + GRPO RL fine-tuning.

Heavy dependencies (torch, transformers, peft, trl, unsloth) are deferred to
function bodies, so this module can be safely imported without GPU access.
"""

from llm_workflow_agents.training.grpo import GRPOResult, train_grpo
from llm_workflow_agents.training.lora_targets import (
    LORA_TARGET_MODULES,
    LoRATargetSpec,
    get_lora_target_spec,
    get_trainable_param_summary,
)
from llm_workflow_agents.training.merge_adapter import merge_and_export
from llm_workflow_agents.training.pilot_check import PilotResult, run_pilot_sft
from llm_workflow_agents.training.rewards import (
    reward_business_logic,
    reward_graph_extraction,
    reward_subagent,
)
from llm_workflow_agents.training.sft import SFTResult, train_sft

# Backward-compatible re-exports from v2 modules
from llm_workflow_agents.training.train_graph_extractor import (
    GRAPH_EXTRACTION_SYSTEM_PROMPT,
    train_graph_extractor,
    train_graph_extractor_from_config,
)
from llm_workflow_agents.training.train_specialist import (
    TrainingResult,
    train,
    train_from_config,
)

__all__ = [
    # v3 API
    "GRPOResult",
    "PilotResult",
    "SFTResult",
    "reward_business_logic",
    "reward_graph_extraction",
    "reward_subagent",
    "train_grpo",
    "train_sft",
    "run_pilot_sft",
    # Shared
    "LORA_TARGET_MODULES",
    "LoRATargetSpec",
    "get_lora_target_spec",
    "get_trainable_param_summary",
    "merge_and_export",
    # v2 backward-compatible
    "GRAPH_EXTRACTION_SYSTEM_PROMPT",
    "TrainingResult",
    "train",
    "train_from_config",
    "train_graph_extractor",
    "train_graph_extractor_from_config",
]
