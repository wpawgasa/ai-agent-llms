"""Training module for LoRA fine-tuning (Experiments B and C).

Heavy dependencies (torch, transformers, peft, trl) are deferred to
function bodies, so this module can be safely imported without GPU access.
"""

from llm_workflow_agents.training.lora_targets import (
    LORA_TARGET_MODULES,
    LoRATargetSpec,
    get_lora_target_spec,
    get_trainable_param_summary,
)
from llm_workflow_agents.training.merge_adapter import merge_and_export
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
    "GRAPH_EXTRACTION_SYSTEM_PROMPT",
    "LORA_TARGET_MODULES",
    "LoRATargetSpec",
    "TrainingResult",
    "get_lora_target_spec",
    "get_trainable_param_summary",
    "merge_and_export",
    "train",
    "train_from_config",
    "train_graph_extractor",
    "train_graph_extractor_from_config",
]
