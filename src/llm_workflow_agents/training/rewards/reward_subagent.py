"""Cat B reward function — Specialist Subagent.

Four weighted components:
  tool_call_f1          0.40
  slot_extraction_acc   0.30
  state_sequence_match  0.20
  format_compliance     0.10
"""

from __future__ import annotations

from typing import Any

import structlog

from llm_workflow_agents.training.reward_utils import (
    extract_state_annotations,
    extract_tool_calls,
    format_compliance_check,
    state_sequence_match,
    tool_call_f1,
)

logger = structlog.get_logger(__name__)

W_TOOL_CALL_F1 = 0.40
W_SLOT_EXTRACTION = 0.30
W_STATE_SEQUENCE = 0.20
W_FORMAT_COMPLIANCE = 0.10


def _compute_slot_accuracy(
    predicted_tools: list[dict[str, Any]],
    gt_tools: list[dict[str, Any]],
) -> float:
    """Compute slot extraction accuracy.

    Compares argument key-value pairs across matched tool calls.
    Returns fraction of correctly extracted slots.
    """
    if not gt_tools:
        return 1.0 if not predicted_tools else 0.0

    total_slots = 0
    correct_slots = 0
    for gt_tool in gt_tools:
        gt_args = gt_tool.get("arguments", {})
        gt_name = gt_tool.get("name", "")
        # Find matching predicted tool by name
        pred_match = next(
            (p for p in predicted_tools if p.get("name") == gt_name), None
        )
        pred_args = pred_match.get("arguments", {}) if pred_match else {}

        for key, value in gt_args.items():
            total_slots += 1
            if pred_args.get(key) == value:
                correct_slots += 1

    return correct_slots / total_slots if total_slots > 0 else 1.0


def reward_subagent(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[dict[str, Any]],
) -> list[float]:
    """Compute Cat B reward for a batch of completions.

    Args:
        prompts: Input prompts (unused but required by GRPOTrainer interface).
        completions: Model completions to score.
        ground_truths: Expected outputs with keys: ``tool_calls``,
            ``state_annotations``.

    Returns:
        List of scalar rewards in [0.0, 1.0].
    """
    rewards: list[float] = []
    for completion, gt in zip(completions, ground_truths):
        pred_tools = extract_tool_calls(completion)
        gt_tools = gt.get("tool_calls", [])
        r_tool = tool_call_f1(pred_tools, gt_tools)

        r_slot = _compute_slot_accuracy(pred_tools, gt_tools)

        pred_states = extract_state_annotations(completion)
        gt_states = gt.get("state_annotations", [])
        r_state = state_sequence_match(pred_states, gt_states)

        r_format = format_compliance_check(completion)

        score = (
            W_TOOL_CALL_F1 * r_tool
            + W_SLOT_EXTRACTION * r_slot
            + W_STATE_SEQUENCE * r_state
            + W_FORMAT_COMPLIANCE * r_format
        )
        rewards.append(max(0.0, min(1.0, score)))

    return rewards
