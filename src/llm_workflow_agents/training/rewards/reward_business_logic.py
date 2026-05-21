"""Cat A reward function — Prompt-Encoded Business Logic.

Four weighted components (sum = 1.0):
  state_transition (graded)        0.40
  tool_call_f1 (argument-graded)   0.40
  task_completion                  0.10
  transition_legality              0.10

Why this shape: the previous reward used a ``length_band`` component as the
fourth weight. length_band scores completion length against a target — it is
task-irrelevant, and because the state/tool components collapse to a constant
across a GRPO group (8 generations of one prompt all land on the same discrete
rung), length_band became the *only* source of within-group reward variance.
GRPO therefore optimized completion length, not workflow correctness — the
killed Gemma-4 run showed the entire 0.0076 within-group reward spread was
length_band reacting to completion length.

``transition_legality`` replaces it with a prompt-grounded signal: the fraction
of emitted ``[STATE: X → Y]`` transitions that are legal edges in the workflow
graph. It varies with what the model actually generates (a hallucinated state
scores 0, a real edge scores 1) and is complementary to ``state_transition``,
which scores whether the transition is the *expected* one for the turn.
"""

from __future__ import annotations

from typing import Any

import structlog

from llm_workflow_agents.training.reward_utils import (
    extract_state_annotations,
    extract_tool_calls,
    graded_tool_call_f1,
    reached_terminal,
    transition_legality_score,
)

logger = structlog.get_logger(__name__)

W_STATE_TRANSITION = 0.40
W_TOOL_CALL_F1 = 0.40
W_TASK_COMPLETION = 0.10
W_TRANSITION_LEGALITY = 0.10

# Some dataset tool calls carry a frozen `{"placeholder": "value"}` argument
# stub instead of real arguments (~17% of Task A tool calls). Scoring against
# it would reward the model for emitting the literal placeholder. When a
# ground-truth tool call has exactly these arguments, drop them so scoring
# falls back to name-only matching.
_PLACEHOLDER_ARGS = {"placeholder": "value"}


def _partial_state_match(
    predicted: list[tuple[str, str]],
    ground_truth: list[tuple[str, str]],
) -> float:
    """State-transition score with partial credit for `from`-only matches.

    Retained for benchmark eval and back-compat. The training reward uses
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
    """State-transition score with graded tiers (continuous variant for GRPO).

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


def _strip_placeholder_args(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Blank out frozen `{"placeholder": "value"}` argument stubs.

    Returns a new list; a tool call whose arguments are exactly the
    placeholder stub gets ``arguments`` reset to ``{}`` so downstream
    argument-graded scoring degrades to name-only matching.
    """
    cleaned: list[dict[str, Any]] = []
    for tc in tool_calls:
        if isinstance(tc, dict) and tc.get("arguments") == _PLACEHOLDER_ARGS:
            cleaned.append({**tc, "arguments": {}})
        else:
            cleaned.append(tc)
    return cleaned


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
            ``tool_calls``, ``terminal_state``, ``terminal_reached``,
            ``valid_transitions``.

    Returns:
        List of scalar rewards in [0.0, 1.0]. The score is a weighted mean
        over the *active* components — ``task_completion`` is excluded when
        ``terminal_reached`` is False, and ``transition_legality`` is
        excluded when no ``valid_transitions`` are available. The remaining
        weights are renormalized so the score stays in [0, 1].
    """
    rewards: list[float] = []
    for completion, gt in zip(completions, ground_truths):
        pred_states = extract_state_annotations(completion)

        gt_states = gt.get("state_annotations", [])
        r_state = _graded_state_match(pred_states, gt_states)

        pred_tools = extract_tool_calls(completion)
        gt_tools = _strip_placeholder_args(gt.get("tool_calls", []))
        r_tool = graded_tool_call_f1(pred_tools, gt_tools)

        # Active components: (weight, score) pairs. task_completion and
        # transition_legality are conditionally active.
        components: list[tuple[float, float]] = [
            (W_STATE_TRANSITION, r_state),
            (W_TOOL_CALL_F1, r_tool),
        ]

        valid_transitions = gt.get("valid_transitions", [])
        if valid_transitions:
            r_legality = transition_legality_score(pred_states, valid_transitions)
            components.append((W_TRANSITION_LEGALITY, r_legality))

        # terminal_reached=False means the conversation never reached a terminal
        # state in the ground truth (e.g. adversarial L4/L5 timeouts, or this
        # is a non-final per-turn row). Skip the completion sub-reward.
        if gt.get("terminal_reached", True):
            terminal = gt.get("terminal_state", "")
            r_completion = 1.0 if reached_terminal(completion, terminal) else 0.0
            components.append((W_TASK_COMPLETION, r_completion))

        total_weight = sum(w for w, _ in components)
        score = (
            sum(w * s for w, s in components) / total_weight
            if total_weight > 0
            else 0.0
        )
        rewards.append(max(0.0, min(1.0, score)))

    return rewards
