"""Cat A reward function — Prompt-Encoded Business Logic.

Six weighted components (sum = 1.0):
  state_transition_correctness  0.30   (partial credit: 0.5 for `from`-only match)
  tool_call_f1 (AST match)      0.30
  chain_propagation_accuracy    0.10   (was 0.20; dead for per-turn GRPO rows)
  format_compliance             0.10
  task_completion               0.10
  length_band (continuous)      0.10   (tie-breaker — see ADR below)

ADR — Why the length-band term: With per-turn GRPO rows (one assistant turn
per training row) the other five components are coarse:
  - state_transition: ground truth has length 1 → result ∈ {0, 0.5, 1}.
  - tool_call_f1: for 0-N tool calls per turn, F1 is small-N-discrete.
  - chain_propagation: returns 1.0 for ≤1-link chains (always for per-turn).
  - format_compliance: 5 discrete values from `format_compliance_check`.
  - task_completion: binary, only fires on terminal rows.
4 generations per group routinely produced identical totals
(`frac_reward_zero_std ≈ 1.0` in the W&B traces from step 0-260), zeroing
the GRPO advantage. The length-band term is a smooth Gaussian-shaped
score on completion character count — guarantees that two completions
with even small length differences produce different rewards, so the
group has non-zero variance and a real learning signal.
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
    tool_call_f1,
)

logger = structlog.get_logger(__name__)

W_STATE_TRANSITION = 0.30
W_TOOL_CALL_F1 = 0.30
W_CHAIN_PROPAGATION = 0.10
W_FORMAT_COMPLIANCE = 0.10
W_TASK_COMPLETION = 0.10
W_LENGTH_BAND = 0.10

# Length-band target/sigma in characters, tuned to Task A completions
# (observed mean ~600 chars / ~180 tokens in the W&B `completion_length`
# trace). Half-credit at 1σ, zero at 2σ — see `_length_band_score`.
LENGTH_BAND_TARGET_CHARS = 600
LENGTH_BAND_SIGMA_CHARS = 300


def _partial_state_match(
    predicted: list[tuple[str, str]],
    ground_truth: list[tuple[str, str]],
) -> float:
    """State-transition score with partial credit for `from`-only matches.

    Returns 1.0 for an exact (from, to) match and 0.5 when only `from`
    matches — softens the binary {0, 1} signal that exact match gives on
    per-turn rows (gt_len == 1). Empty ground truth returns 1.0 when the
    prediction is also empty, else 0.0 (matches the legacy behavior).
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    total = 0.0
    for i, gt_pair in enumerate(ground_truth):
        if i >= len(predicted):
            continue
        pred = predicted[i]
        if pred == gt_pair:
            total += 1.0
        elif pred[0] == gt_pair[0]:
            total += 0.5
    return total / len(ground_truth)


def _length_band_score(
    completion: str,
    target: int = LENGTH_BAND_TARGET_CHARS,
    sigma: int = LENGTH_BAND_SIGMA_CHARS,
) -> float:
    """Smooth length-similarity score in [0, 1].

    Continuous in completion length → guaranteed to differ across rollouts
    with even small length variation, breaking 4-way ties in the GRPO group.
    Linear decay: 1.0 at target, 0.5 at target±sigma, 0.0 at target±2σ.
    """
    actual = len(completion)
    deviation = abs(actual - target) / max(1, sigma)
    return max(0.0, 1.0 - 0.5 * deviation)


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
        r_state = _partial_state_match(pred_states, gt_states)

        pred_tools = extract_tool_calls(completion)
        gt_tools = gt.get("tool_calls", [])
        r_tool = tool_call_f1(pred_tools, gt_tools)

        pred_msgs = [{"role": "assistant", "content": completion}]
        gt_msgs = gt.get("messages", [])
        r_chain = chain_propagation_score(pred_msgs, gt_msgs)

        r_format = format_compliance_check(completion)
        r_length = _length_band_score(completion)

        terminal = gt.get("terminal_state", "")
        # terminal_reached=False means the data never reached a terminal state
        # (e.g. adversarial L4/L5 timeouts). Skip the completion sub-reward and
        # rescale the remaining weights to sum to 1 to keep scores in [0, 1].
        if not gt.get("terminal_reached", True):
            score = (
                W_STATE_TRANSITION * r_state
                + W_TOOL_CALL_F1 * r_tool
                + W_CHAIN_PROPAGATION * r_chain
                + W_FORMAT_COMPLIANCE * r_format
                + W_LENGTH_BAND * r_length
            ) / (1.0 - W_TASK_COMPLETION)
        else:
            r_completion = 1.0 if reached_terminal(completion, terminal) else 0.0
            score = (
                W_STATE_TRANSITION * r_state
                + W_TOOL_CALL_F1 * r_tool
                + W_CHAIN_PROPAGATION * r_chain
                + W_FORMAT_COMPLIANCE * r_format
                + W_TASK_COMPLETION * r_completion
                + W_LENGTH_BAND * r_length
            )
        rewards.append(max(0.0, min(1.0, score)))

    return rewards
