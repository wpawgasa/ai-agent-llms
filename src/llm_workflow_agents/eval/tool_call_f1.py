"""Tool-calling accuracy evaluation for Experiment A.

Parses <tool_call>{JSON}</tool_call> from model output and compares
against ground-truth tool calls using BFCL-style AST sub-tree matching.
Normalizes all tool-call formats to canonical JSON before scoring (Risk R6).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Patterns for parsing tool calls from model output
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_HERMES_PATTERN = re.compile(
    r'<\|tool_call\|>\s*(\{.*?\})\s*(?:<\|/tool_call\|>|$)', re.DOTALL
)


@dataclass
class TurnPrediction:
    """A single turn's predicted tool calls."""

    turn_id: int
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnGroundTruth:
    """A single turn's expected tool calls."""

    turn_id: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolCallMetrics:
    """Metrics for tool-calling evaluation."""

    tool_name_accuracy: float = 0.0  # Target: >=90%
    argument_exact_match: float = 0.0  # Target: >=75%
    tool_call_f1: float = 0.0  # Target: >=85%
    hallucinated_tool_rate: float = 0.0  # Target: <=3%
    # error_recovery_rate is tracked in StateMachineMetrics (recovery_rate), not here.

    def to_dict(self) -> dict[str, float]:
        return {
            "tool_name_accuracy": self.tool_name_accuracy,
            "argument_exact_match": self.argument_exact_match,
            "tool_call_f1": self.tool_call_f1,
            "hallucinated_tool_rate": self.hallucinated_tool_rate,
        }


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse tool calls from model output in various formats.

    Supports:
      - <tool_call>{JSON}</tool_call>
      - <|tool_call|>{JSON}<|/tool_call|>
      - Raw JSON with 'name' and 'arguments' fields

    Returns:
        List of normalized tool call dicts with 'name' and 'arguments'.
    """
    calls: list[dict[str, Any]] = []

    # Try standard format
    for match in _TOOL_CALL_PATTERN.finditer(content):
        try:
            parsed = json.loads(match.group(1))
            calls.append(_normalize_tool_call(parsed))
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Try Hermes format
    for match in _HERMES_PATTERN.finditer(content):
        try:
            parsed = json.loads(match.group(1))
            calls.append(_normalize_tool_call(parsed))
        except json.JSONDecodeError:
            continue

    return calls


def _normalize_tool_call(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool call dict to canonical format.

    Handles variations like:
      - {"name": "fn", "arguments": {...}}
      - {"function": {"name": "fn", "arguments": {...}}}
      - {"name": "fn", "parameters": {...}}
    """
    if "function" in raw and isinstance(raw["function"], dict):
        fn = raw["function"]
        return {
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", fn.get("parameters", {})),
        }

    name = raw.get("name", "")
    arguments = raw.get("arguments", raw.get("parameters", {}))

    # If arguments is a JSON string, parse it
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass

    return {"name": name, "arguments": arguments}


def compute_name_accuracy(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> float:
    """Compute tool name accuracy (exact match of function name)."""
    if not ground_truth:
        return 1.0 if not predicted else 0.0

    correct = 0
    for i, gt in enumerate(ground_truth):
        if i < len(predicted) and predicted[i]["name"] == gt["name"]:
            correct += 1

    return correct / len(ground_truth)


def compute_argument_match(
    predicted: dict[str, Any],
    ground_truth: dict[str, Any],
) -> bool:
    """Check if predicted arguments exactly match ground truth.

    Uses recursive comparison for nested structures.
    """
    pred_args = predicted.get("arguments", {})
    gt_args = ground_truth.get("arguments", {})
    return _deep_equals(pred_args, gt_args)


def _deep_equals(a: Any, b: Any) -> bool:
    """Deep equality comparison for JSON-compatible values."""
    if type(a) is not type(b):
        # Allow int/float comparison
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equals(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_deep_equals(x, y) for x, y in zip(a, b))
    return a == b


def compute_ast_f1(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> float:
    """Compute BFCL-style AST sub-tree matching F1 score.

    A predicted tool call matches a ground-truth call if:
      - The function name matches exactly
      - All ground-truth argument key-value pairs are present in prediction
        (prediction may have extra keys — sub-tree match)
    """
    if not ground_truth and not predicted:
        return 1.0
    if not ground_truth or not predicted:
        return 0.0

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for pi, pred in enumerate(predicted):
        for gi, gt in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            if pred["name"] == gt["name"] and _is_subtree_match(pred, gt):
                matched_pred.add(pi)
                matched_gt.add(gi)
                break

    tp = len(matched_gt)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _is_subtree_match(predicted: dict[str, Any], ground_truth: dict[str, Any]) -> bool:
    """Check if all ground-truth argument fields are present and correct in prediction."""
    gt_args = ground_truth.get("arguments", {})
    pred_args = predicted.get("arguments", {})

    if not isinstance(gt_args, dict) or not isinstance(pred_args, dict):
        return _deep_equals(gt_args, pred_args)

    for key, gt_val in gt_args.items():
        if key not in pred_args:
            return False
        if not _deep_equals(pred_args[key], gt_val):
            return False
    return True


def detect_hallucinated_tools(
    predicted: list[dict[str, Any]],
    valid_tool_names: list[str],
) -> list[str]:
    """Detect tool calls to tools not in the schema."""
    valid_set = set(valid_tool_names)
    return [tc["name"] for tc in predicted if tc["name"] not in valid_set]


def evaluate_tool_calls(
    predictions: list[TurnPrediction],
    ground_truth: list[TurnGroundTruth],
    tool_schemas: list[dict[str, Any]] | None = None,
) -> ToolCallMetrics:
    """Evaluate tool-calling accuracy across turns.

    Args:
        predictions: Per-turn model predictions.
        ground_truth: Per-turn expected tool calls.
        tool_schemas: Available tool schemas for hallucination detection.

    Returns:
        ToolCallMetrics with all computed metrics.
    """
    gt_map = {gt.turn_id: gt for gt in ground_truth}

    total_name_acc = 0.0
    total_arg_match = 0
    total_arg_comparisons = 0
    total_f1 = 0.0
    total_hallucinated = 0
    total_predicted = 0
    n_turns_with_tools = 0

    # Extract valid tool names from schemas
    valid_names: list[str] = []
    if tool_schemas:
        for schema in tool_schemas:
            fn = schema.get("function", schema)
            name = fn.get("name", "")
            if name:
                valid_names.append(name)

    for pred in predictions:
        gt = gt_map.get(pred.turn_id)
        if gt is None:
            continue

        pred_calls = pred.tool_calls or parse_tool_calls(pred.content)
        gt_calls = gt.tool_calls

        if not gt_calls and not pred_calls:
            continue

        n_turns_with_tools += 1
        total_predicted += len(pred_calls)

        # Name accuracy
        total_name_acc += compute_name_accuracy(pred_calls, gt_calls)

        # Argument exact match
        for i, gt_call in enumerate(gt_calls):
            total_arg_comparisons += 1
            if i < len(pred_calls) and compute_argument_match(pred_calls[i], gt_call):
                total_arg_match += 1

        # AST F1
        total_f1 += compute_ast_f1(pred_calls, gt_calls)

        # Hallucination detection
        if valid_names:
            hallucinated = detect_hallucinated_tools(pred_calls, valid_names)
            total_hallucinated += len(hallucinated)

    if n_turns_with_tools == 0:
        return ToolCallMetrics()

    metrics = ToolCallMetrics(
        tool_name_accuracy=total_name_acc / n_turns_with_tools,
        argument_exact_match=(
            total_arg_match / total_arg_comparisons if total_arg_comparisons > 0 else 0.0
        ),
        tool_call_f1=total_f1 / n_turns_with_tools,
        hallucinated_tool_rate=(
            total_hallucinated / total_predicted if total_predicted > 0 else 0.0
        ),
    )

    logger.info("tool_call_eval_complete", turns=n_turns_with_tools, **metrics.to_dict())
    return metrics


def evaluate_tool_calls_conversation(
    conversations_pred: list[list[TurnPrediction]],
    conversations_gt: list[list[TurnGroundTruth]],
    tool_schemas: list[dict[str, Any]] | None = None,
) -> ToolCallMetrics:
    """Evaluate tool-calling accuracy at the conversation level.

    Unlike :func:`evaluate_tool_calls` which aligns predictions to
    ground truth per-turn, this function pools all tool calls within
    each conversation and computes metrics on the pooled sets, then
    averages across conversations.  A model that calls the right tool
    at a different turn still receives credit.

    Args:
        conversations_pred: Per-conversation lists of turn predictions.
        conversations_gt: Per-conversation lists of turn ground truths.
        tool_schemas: Optional tool schemas for hallucination detection.
    """
    valid_names: list[str] = []
    if tool_schemas:
        for schema in tool_schemas:
            fn = schema.get("function", schema)
            name = fn.get("name", "")
            if name:
                valid_names.append(name)

    total_name_acc = 0.0
    total_f1 = 0.0
    total_arg_match = 0
    total_arg_comparisons = 0
    total_hallucinated = 0
    total_predicted = 0
    n_convs = 0

    for preds, gts in zip(conversations_pred, conversations_gt):
        conv_pred_calls: list[dict[str, Any]] = []
        conv_gt_calls: list[dict[str, Any]] = []

        for pred in preds:
            conv_pred_calls.extend(pred.tool_calls or parse_tool_calls(pred.content))
        for gt in gts:
            conv_gt_calls.extend(gt.tool_calls)

        if not conv_pred_calls and not conv_gt_calls:
            continue

        n_convs += 1
        total_predicted += len(conv_pred_calls)

        total_name_acc += compute_name_accuracy(conv_pred_calls, conv_gt_calls)
        total_f1 += compute_ast_f1(conv_pred_calls, conv_gt_calls)

        for i, gt_call in enumerate(conv_gt_calls):
            total_arg_comparisons += 1
            if i < len(conv_pred_calls) and compute_argument_match(conv_pred_calls[i], gt_call):
                total_arg_match += 1

        if valid_names:
            total_hallucinated += len(detect_hallucinated_tools(conv_pred_calls, valid_names))

    if n_convs == 0:
        return ToolCallMetrics()

    metrics = ToolCallMetrics(
        tool_name_accuracy=total_name_acc / n_convs,
        argument_exact_match=(
            total_arg_match / total_arg_comparisons if total_arg_comparisons > 0 else 0.0
        ),
        tool_call_f1=total_f1 / n_convs,
        hallucinated_tool_rate=(
            total_hallucinated / total_predicted if total_predicted > 0 else 0.0
        ),
    )

    logger.info("tool_call_conv_eval_complete", conversations=n_convs, **metrics.to_dict())
    return metrics
