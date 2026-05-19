"""Cat A reward function — Prompt-Encoded Business Logic.

Four weighted components (sum = 1.0):
  state_transition (graded)     0.40
  tool_call_f1 (argument-graded) 0.40
  task_completion               0.10
  length_band (continuous)      0.10

Why this shape (2026-05-19): the per-flight diagnostic in
``docs/grpo_diagnosis_gemma4_26b.md`` showed the previous 6-component
reward was structurally incapable of producing within-group reward
variance > ~0.05 from anything but the length_band tie-breaker:

  - chain_propagation: returned 1.0 on every single-assistant-turn row
    (n_pairs <= 1 shortcut in ``chain_propagation_score``) → fixed +0.10
    contribution to every reward; carried zero signal. Dropped.
  - format_compliance: 1.0 for any SFT'd model output (no tracebacks,
    matched <tool_call> tags) → fixed +0.10 with zero signal. Dropped.
  - state_transition + tool_call_f1: binary cliffs (0 or 1 per row),
    flipped on only 1-2 of 5 prompts. Replaced with graded variants
    (``_graded_state_match``, ``graded_tool_call_f1``) that give partial
    credit for argument-level overlap and same-target/reverse-direction
    state transitions.

The length_band tie-breaker stays — it's the only smooth component and
guarantees non-zero within-group variance even when the others all agree.
"""

from __future__ import annotations

from typing import Any

import structlog

from llm_workflow_agents.training.reward_utils import (
    extract_state_annotations,
    extract_tool_calls,
    graded_tool_call_f1,
    reached_terminal,
)

logger = structlog.get_logger(__name__)

W_STATE_TRANSITION = 0.40
W_TOOL_CALL_F1 = 0.40
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

    Retained for benchmark eval and back-compat. New training reward uses
    :func:`_graded_state_match`, which adds credit for `to`-only matches
    and reverse-direction transitions.

    Returns 1.0 for an exact (from, to) match and 0.5 when only `from`
    matches. Empty ground truth returns 1.0 when the prediction is also
    empty, else 0.0.
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


def _graded_state_match(
    predicted: list[tuple[str, str]],
    ground_truth: list[tuple[str, str]],
) -> float:
    """State-transition score with five graded tiers (continuous variant for GRPO).

    Per (pred, gt) pair:
      exact match            → 1.0
      `from` matches only    → 0.5    (correct origin, wrong destination)
      `to` matches only      → 0.5    (correct destination, wrong origin)
      reverse direction      → 0.3    (right state-pair, wrong direction)
      no overlap             → 0.0

    The added `to`-only and reverse-direction credit breaks the 0/0.5/1
    cliff of :func:`_partial_state_match` into a {0, 0.3, 0.5, 1.0} ladder
    — finer-grained signal for GRPO advantage when the model gets *part*
    of a transition right.
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
        elif pred[0] == gt_pair[0] or pred[1] == gt_pair[1]:
            total += 0.5
        elif pred[0] == gt_pair[1] and pred[1] == gt_pair[0]:
            total += 0.3
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
            ``tool_calls``, ``terminal_state``, ``terminal_reached``.

    Returns:
        List of scalar rewards in [0.0, 1.0].
    """
    rewards: list[float] = []
    for completion, gt in zip(completions, ground_truths):
        pred_states = extract_state_annotations(completion)
        gt_states = gt.get("state_annotations", [])
        r_state = _graded_state_match(pred_states, gt_states)

        pred_tools = extract_tool_calls(completion)
        gt_tools = gt.get("tool_calls", [])
        r_tool = graded_tool_call_f1(pred_tools, gt_tools)

        r_length = _length_band_score(completion)

        terminal = gt.get("terminal_state", "")
        # terminal_reached=False means the conversation never reached a terminal
        # state in the ground truth (e.g. adversarial L4/L5 timeouts, or this
        # is a non-final per-turn row). Skip the completion sub-reward and
        # rescale the remaining weights to sum to 1 to keep scores in [0, 1].
        if not gt.get("terminal_reached", True):
            score = (
                W_STATE_TRANSITION * r_state
                + W_TOOL_CALL_F1 * r_tool
                + W_LENGTH_BAND * r_length
            ) / (1.0 - W_TASK_COMPLETION)
        else:
            r_completion = 1.0 if reached_terminal(completion, terminal) else 0.0
            score = (
                W_STATE_TRANSITION * r_state
                + W_TOOL_CALL_F1 * r_tool
                + W_TASK_COMPLETION * r_completion
                + W_LENGTH_BAND * r_length
            )
        rewards.append(max(0.0, min(1.0, score)))

    return rewards
