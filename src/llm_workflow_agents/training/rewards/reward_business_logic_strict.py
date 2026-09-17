"""Cat A reward that can tell a right argument from a plausible wrong one.

``reward_business_logic`` saturates. On the fine-tuned E4B, 86.8% of probe
prompts score exactly 1.0 greedily and 74.6% of 8-sample groups tie completely
(CLAUDE.md R23), so GRPO gets no gradient and best-of-N has nothing to distil.
Two of its design choices cause most of that:

- ``graded_tool_call_f1`` gives 0.4 for the tool NAME before any argument is
  checked, and prorates the rest. Tool selection is already solved (0.16% of
  mined C2 errors), so every sample collects that 0.4; the argument errors that
  make up 77.5% of real mistakes move the score only fractionally.
- ``_graded_state_match`` pays 0.5 for a transition with the right origin or
  destination, and 0.3 for the reverse direction.

This reward removes the free credit. A tool call earns only for arguments that
match the ground truth exactly, a wrong name earns nothing, and a state
transition is right or wrong. What stays graded is the fraction of arguments
that are correct, which is exactly where samples of a well-trained model
differ: one sample copies the order id from the tool result, another copies the
stale one from two calls earlier.

It is a separate function, not a change to ``reward_business_logic``: every
stored probe and training result was scored with the graded reward, and the
pre-registered gates in docs/grpo_viability_investigation.md were measured on
it. Results under this reward are a new scale.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_workflow_agents.training.reward_utils import (
    extract_state_annotations,
    extract_tool_calls,
)
from llm_workflow_agents.training.rewards.reward_business_logic import (
    _strip_placeholder_args,
)

W_STATE = 0.2
W_TOOL = 0.8

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _canonical(value: Any) -> tuple[str, Any]:
    """A comparison key under which "42" and 42 are equal but "ORD-042" and "ORD-42" are not."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    if isinstance(value, str):
        text = value.strip()
        if _NUMBER.fullmatch(text):
            return ("num", float(text))
        return ("str", text)
    if value is None:
        return ("none", None)
    return ("json", json.dumps(value, sort_keys=True, ensure_ascii=False))


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {}
    return args if isinstance(args, dict) else {}


def _flatten(arguments: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in arguments.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _call_score(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    """Exact-argument score for two calls already known to share a name.

    Correct arguments over (ground-truth arguments + arguments the model
    invented), so an extra argument costs as much as a missing one.
    """
    gt_args = _flatten(_arguments(gt))
    pred_args = _flatten(_arguments(pred))
    if not gt_args:
        # The ground truth carries no usable arguments (placeholder stub), so
        # only the name can be checked.
        return 1.0
    correct = sum(
        1
        for key, value in gt_args.items()
        if key in pred_args and _canonical(pred_args[key]) == _canonical(value)
    )
    invented = len(set(pred_args) - set(gt_args))
    return correct / (len(gt_args) + invented)


def strict_tool_call_score(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> float:
    """Exact-argument tool-call score in [0, 1], with no credit for the name alone.

    Each ground-truth call is matched to the unused predicted call of the same
    name that scores best. Unmatched ground-truth calls score 0, and every
    predicted call beyond the ground-truth count enlarges the denominator.
    """
    ground_truth = _strip_placeholder_args(list(ground_truth or []))
    predicted = list(predicted or [])
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    unused = list(range(len(predicted)))
    total = 0.0
    for gt in ground_truth:
        candidates = [i for i in unused if predicted[i].get("name") == gt.get("name")]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: _call_score(predicted[i], gt))
        total += _call_score(predicted[best], gt)
        unused.remove(best)
    return total / max(len(ground_truth), len(predicted))


def exact_state_score(
    predicted: list[tuple[str, str]],
    ground_truth: list[tuple[str, str]],
) -> float:
    """Fraction of transitions that match exactly, over the longer of the two lists."""
    predicted = [tuple(p) for p in predicted]
    ground_truth = [tuple(g) for g in ground_truth]
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return correct / max(len(ground_truth), len(predicted))


def reward_business_logic_strict(
    prompts: list[Any],
    completions: list[str],
    ground_truths: list[dict[str, Any]],
) -> list[float]:
    """Cat A reward with no partial credit except the share of correct arguments.

    Args:
        prompts: Unused; required by the GRPOTrainer reward interface.
        completions: Model completions to score.
        ground_truths: Dicts with ``state_annotations`` and ``tool_calls``, the
            same shape ``reward_business_logic`` reads.

    Returns:
        One reward in [0, 1] per completion:
        ``W_STATE * exact_state_score + W_TOOL * strict_tool_call_score``.
    """
    rewards: list[float] = []
    for completion, gt in zip(completions, ground_truths):
        state = exact_state_score(
            extract_state_annotations(completion), gt.get("state_annotations", [])
        )
        tool = strict_tool_call_score(extract_tool_calls(completion), gt.get("tool_calls", []))
        rewards.append(max(0.0, min(1.0, W_STATE * state + W_TOOL * tool)))
    return rewards
