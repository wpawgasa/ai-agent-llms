"""Cat A reward function — Prompt-Encoded Business Logic.

Five weighted components:
  state_transition_correctness  0.30
  tool_call_f1 (AST match)     0.30
  chain_propagation_accuracy    0.20
  format_compliance             0.10
  task_completion               0.10
"""

from __future__ import annotations

from typing import Any

import structlog

from llm_workflow_agents.training.reward_utils import (
    chain_propagation_score,
    extract_state_annotations,
    extract_tool_calls,
    format_compliance_check,
    reached_terminal,
    state_sequence_match,
    tool_call_f1,
)

logger = structlog.get_logger(__name__)

W_STATE_TRANSITION = 0.30
W_TOOL_CALL_F1 = 0.30
W_CHAIN_PROPAGATION = 0.20
W_FORMAT_COMPLIANCE = 0.10
W_TASK_COMPLETION = 0.10


def reward_business_logic(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[dict[str, Any]],
) -> list[float]:
    """Compute Cat A reward for a batch of completions.

    Args:
        prompts: Input prompts (unused but required by GRPOTrainer interface).
        completions: Model completions to score.
        ground_truths: Expected outputs with keys: ``state_annotations``,
            ``tool_calls``, ``messages``, ``terminal_state``.

    Returns:
        List of scalar rewards in [0.0, 1.0].
    """
    rewards: list[float] = []
    for completion, gt in zip(completions, ground_truths):
        pred_states = extract_state_annotations(completion)
        gt_states = gt.get("state_annotations", [])
        r_state = state_sequence_match(pred_states, gt_states)

        pred_tools = extract_tool_calls(completion)
        gt_tools = gt.get("tool_calls", [])
        r_tool = tool_call_f1(pred_tools, gt_tools)

        pred_msgs = [{"role": "assistant", "content": completion}]
        gt_msgs = gt.get("messages", [])
        r_chain = chain_propagation_score(pred_msgs, gt_msgs)

        r_format = format_compliance_check(completion)

        terminal = gt.get("terminal_state", "")
        r_completion = 1.0 if reached_terminal(completion, terminal) else 0.0

        score = (
            W_STATE_TRANSITION * r_state
            + W_TOOL_CALL_F1 * r_tool
            + W_CHAIN_PROPAGATION * r_chain
            + W_FORMAT_COMPLIANCE * r_format
            + W_TASK_COMPLETION * r_completion
        )
        rewards.append(max(0.0, min(1.0, score)))

    return rewards
