"""Tool chain propagation evaluation for Experiment A.

Evaluates whether return values from tool N correctly populate
arguments of tool N+1 in multi-step tool chains. Tracks per-depth
accuracy to identify where propagation breaks down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from llm_workflow_agents.eval.tool_call_f1 import parse_tool_calls

logger = structlog.get_logger(__name__)


@dataclass
class ChainPropagationMetrics:
    """Metrics for tool chain propagation evaluation."""

    chain_propagation_accuracy: float = 0.0  # Target: >=70%
    per_depth_accuracy: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_propagation_accuracy": self.chain_propagation_accuracy,
            "per_depth_accuracy": self.per_depth_accuracy,
        }


@dataclass
class ToolChainLink:
    """A single link in a tool chain: tool call followed by tool response."""

    tool_name: str
    arguments: dict[str, Any]
    response: dict[str, Any]
    depth: int


def extract_tool_chains(messages: list[dict[str, Any]]) -> list[ToolChainLink]:
    """Extract sequential tool chain links from a conversation.

    A chain link is: assistant tool_call → tool response → next assistant tool_call.
    """
    links: list[ToolChainLink] = []
    depth = 0

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.get("role") == "assistant":
            calls = parse_tool_calls(msg.get("content", ""))
            if not calls:
                # Also check annotations
                annotations = msg.get("annotations", {})
                ann_calls = annotations.get("tool_calls", [])
                if ann_calls:
                    calls = ann_calls

            for call in calls:
                # Look for the corresponding tool response
                tool_response = _find_next_tool_response(messages, i + 1)
                if tool_response is not None:
                    links.append(
                        ToolChainLink(
                            tool_name=call.get("name", ""),
                            arguments=call.get("arguments", {}),
                            response=tool_response,
                            depth=depth,
                        )
                    )
                    depth += 1

        i += 1

    return links


def _find_next_tool_response(messages: list[dict[str, Any]], start: int) -> dict[str, Any] | None:
    """Find the next tool response message starting from index."""
    for i in range(start, len(messages)):
        if messages[i].get("role") == "tool":
            content = messages[i].get("content", "")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}
    return None


def check_value_propagation(
    prev_response: dict[str, Any],
    next_arguments: dict[str, Any],
) -> bool:
    """Check if any value from the previous tool response appears in the next tool's arguments.

    This is a heuristic check: at least one value from the response should
    appear as a value in the next call's arguments for the chain to be
    considered properly propagated.
    """
    if not prev_response or not next_arguments:
        return False

    response_values = _extract_leaf_values(prev_response)
    argument_values = _extract_leaf_values(next_arguments)

    # Check if any response value appears in the arguments
    return bool(response_values & argument_values)


def _extract_leaf_values(obj: Any, _values: set[str] | None = None) -> set[str]:
    """Extract all leaf string/number values from a nested structure."""
    if _values is None:
        _values = set()

    if isinstance(obj, dict):
        for v in obj.values():
            _extract_leaf_values(v, _values)
    elif isinstance(obj, list):
        for item in obj:
            _extract_leaf_values(item, _values)
    elif isinstance(obj, (str, int, float)) and obj != "":
        _values.add(str(obj))

    return _values


def evaluate_chain_propagation(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> ChainPropagationMetrics:
    """Evaluate tool chain propagation accuracy.

    Args:
        predictions: List of conversation prediction dicts with 'messages'.
        ground_truth: List of conversation ground-truth dicts with 'messages'.

    Returns:
        ChainPropagationMetrics with overall and per-depth accuracy.
    """
    depth_correct: dict[int, int] = {}
    depth_total: dict[int, int] = {}
    total_correct = 0
    total_chains = 0

    for pred, gt in zip(predictions, ground_truth):
        pred_links = extract_tool_chains(pred.get("messages", []))
        gt_links = extract_tool_chains(gt.get("messages", []))

        # Evaluate consecutive links for value propagation
        for i in range(1, len(pred_links)):
            prev = pred_links[i - 1]
            curr = pred_links[i]
            depth = min(curr.depth, 4)  # Cap at 4+ for grouping

            depth_total[depth] = depth_total.get(depth, 0) + 1
            total_chains += 1

            if check_value_propagation(prev.response, curr.arguments):
                depth_correct[depth] = depth_correct.get(depth, 0) + 1
                total_correct += 1

    per_depth: dict[int, float] = {}
    for d in sorted(depth_total.keys()):
        per_depth[d] = depth_correct.get(d, 0) / depth_total[d]

    metrics = ChainPropagationMetrics(
        chain_propagation_accuracy=total_correct / max(total_chains, 1),
        per_depth_accuracy=per_depth,
    )

    logger.info("chain_propagation_eval_complete", total_chains=total_chains, **metrics.to_dict())
    return metrics
