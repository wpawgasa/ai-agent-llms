"""Composite scoring for Phase 1 model selection.

Provides weighted workflow quality scores and full workflow success rate
for ranking pre-trained candidates across all 3 task categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from llm_workflow_agents.eval.state_accuracy import (
    ConversationGroundTruth,
    ConversationPrediction,
    StateMachineMetrics,
    check_task_completion,
    compute_transition_accuracy,
    parse_state_transitions,
)
from llm_workflow_agents.eval.tool_call_f1 import (
    ToolCallMetrics,
    compute_ast_f1,
    parse_tool_calls,
)

logger = structlog.get_logger(__name__)


@dataclass
class CompositeResult:
    """Composite scoring result for a single model."""

    model_name: str = ""
    category: str = ""
    weighted_workflow_score: float = 0.0
    full_workflow_success_rate: float = 0.0
    num_conversations: int = 0
    per_conversation_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "category": self.category,
            "weighted_workflow_score": self.weighted_workflow_score,
            "full_workflow_success_rate": self.full_workflow_success_rate,
            "num_conversations": self.num_conversations,
        }


def compute_weighted_workflow_score(
    state: StateMachineMetrics,
    tool: ToolCallMetrics,
) -> float:
    """Compute weighted workflow quality score.

    Formula: 0.4 * StateTransAcc + 0.4 * ToolCallF1 + 0.2 * TaskCompletion

    Target: >= 0.75

    Args:
        state: State machine adherence metrics.
        tool: Tool-calling accuracy metrics.

    Returns:
        Weighted score between 0.0 and 1.0.
    """
    return (
        0.4 * state.state_transition_accuracy
        + 0.4 * tool.tool_call_f1
        + 0.2 * state.task_completion_rate
    )


#: Share of Phase 1 quality carried by the voice stratum.
#:
#: NOT in configs/benchmark/selection_weights.yaml on purpose: no code reads
#: that file, so a weight placed there would be a knob that does nothing —
#: the defect shape risks R16 and R18c both record.
DEFAULT_VOICE_WEIGHT = 0.30


def blend_modality_scores(
    score_text: float | None,
    score_voice: float | None,
    voice_weight: float = DEFAULT_VOICE_WEIGHT,
) -> float:
    """Blend the two modality strata into one quality number.

    A weighted mean of two per-stratum scores, NOT a mean over pooled rows.
    A pooled mean takes its effective weight from however many rows each
    stratum happens to hold, so it drifts every time the data is regenerated
    and nobody notices, because the result still looks like a number.

    With one stratum absent the other is returned unchanged, by identity and
    not by arithmetic. That keeps this change inert until a voice corpus
    exists: the ranking moves when a person adds voice data, not when someone
    merges a branch.
    """
    if not 0.0 <= voice_weight <= 1.0:
        raise ValueError(f"voice_weight must be within 0.0 and 1.0, got {voice_weight}")
    if score_voice is None:
        return score_text if score_text is not None else 0.0
    if score_text is None:
        return score_voice
    return voice_weight * score_voice + (1.0 - voice_weight) * score_text


def full_workflow_success_rate(
    predictions: list[ConversationPrediction],
    ground_truth: list[ConversationGroundTruth],
) -> float:
    """Compute percentage of conversations with ALL correct transitions AND tool calls.

    A conversation is fully successful if:
      1. All state transitions match the ground truth exactly
      2. All tool calls match (AST F1 >= 1.0 per turn)

    Target: >= 55%

    Args:
        predictions: List of model predictions per conversation.
        ground_truth: List of ground truth per conversation.

    Returns:
        Success rate between 0.0 and 1.0.
    """
    if not predictions or not ground_truth:
        return 0.0

    gt_by_id = {gt.conversation_id: gt for gt in ground_truth}
    successful = 0
    total = 0

    for pred in predictions:
        gt = gt_by_id.get(pred.conversation_id)
        if gt is None:
            continue
        total += 1

        # Check state transitions
        pred_transitions = parse_state_transitions(pred.messages)
        gt_transitions = parse_state_transitions(gt.messages)
        accuracy, invalid = compute_transition_accuracy(pred_transitions, gt_transitions)
        if accuracy < 1.0 or invalid > 0:
            continue

        # Check terminal state reached
        if not check_task_completion(pred_transitions, gt.terminal_states):
            continue

        # Check tool calls per turn
        all_tools_correct = True
        for pred_msg, gt_msg in zip(pred.messages, gt.messages):
            if pred_msg.get("role") != "assistant":
                continue
            pred_tools = parse_tool_calls(pred_msg.get("content", ""))
            gt_tools = parse_tool_calls(gt_msg.get("content", ""))
            if gt_tools:
                f1 = compute_ast_f1(pred_tools, gt_tools)
                if f1 < 1.0:
                    all_tools_correct = False
                    break

        if all_tools_correct:
            successful += 1

    return successful / total if total > 0 else 0.0
