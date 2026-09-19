"""Tool chain propagation evaluation for Experiment A.

Evaluates whether return values from tool N correctly populate arguments of
tool N+1 in multi-step tool chains. Tracks per-depth accuracy to identify where
propagation breaks down.

Scoring is restricted to real *propagation opportunities*: an argument of call
N+1 whose value appears in call N's response and nowhere earlier in the
conversation, so the only way to produce it is to have read the tool result.

The earlier definition scored every consecutive pair of calls, which counted
independent calls -- most pairs -- as propagation failures no matter what the
model did. Scored against itself, the benchmark ground truth reached only
0.2843, below every model measured, so the metric had no headroom and could not
rank anything. The current definition scores ground truth at exactly 1.0 by
construction; :mod:`tests.unit.test_chain_propagation_opportunities` asserts it
against the real corpus.
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
    total_chains: int = 0  # Propagation opportunities scored; 0 means none present
    missed_calls: int = 0  # Opportunities whose ground-truth call was never made
    conversations_with_chains: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_propagation_accuracy": self.chain_propagation_accuracy,
            "per_depth_accuracy": self.per_depth_accuracy,
            "total_chains": self.total_chains,
            "missed_calls": self.missed_calls,
            "conversations_with_chains": self.conversations_with_chains,
        }


@dataclass
class ToolChainLink:
    """A single link in a tool chain: tool call followed by tool response."""

    tool_name: str
    arguments: dict[str, Any]
    response: dict[str, Any]
    depth: int
    call_index: int = -1       # index of the assistant message carrying the call
    response_index: int = -1   # index of the tool message carrying the response


@dataclass(frozen=True)
class PropagationOpportunity:
    """One argument value that can only come from the previous tool response.

    ``link_index`` indexes the ground-truth link list, so the predicted call it
    is scored against is whichever call the alignment matched to that link.
    """

    link_index: int
    tool_name: str
    argument_path: str
    expected_value: str
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
                tool_response, response_index = _find_next_tool_response(messages, i + 1)
                if tool_response is not None:
                    links.append(
                        ToolChainLink(
                            tool_name=call.get("name", ""),
                            arguments=call.get("arguments", {}),
                            response=tool_response,
                            depth=depth + 1,  # 1-indexed: first call is depth 1
                            call_index=i,
                            response_index=response_index,
                        )
                    )
                    depth += 1

        i += 1

    return links


def _find_next_tool_response(
    messages: list[dict[str, Any]], start: int
) -> tuple[dict[str, Any] | None, int]:
    """Find the next tool response message starting from index.

    Stops at the next assistant message boundary to avoid associating a tool
    response from a later exchange with the current tool call. Returns the
    parsed payload and the index of the message it came from, or ``(None, -1)``.
    """
    for i in range(start, len(messages)):
        role = messages[i].get("role")
        if role == "tool":
            content = messages[i].get("content", "")
            try:
                return json.loads(content), i
            except json.JSONDecodeError:
                return {"raw": content}, i
        elif role == "assistant":
            # Stop searching once a new assistant turn begins
            break
    return None, -1


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


def _flatten_arguments(obj: Any, path: str = "", out: dict[str, str] | None = None) -> dict[str, str]:
    """Flatten call arguments to ``{dotted.path: value}`` over leaf scalars."""
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten_arguments(value, f"{path}.{key}" if path else str(key), out)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            _flatten_arguments(item, f"{path}[{index}]", out)
    elif isinstance(obj, bool):
        pass  # booleans carry no identity; "true" matches far too much
    elif isinstance(obj, (str, int, float)) and str(obj) != "":
        out[path] = str(obj)

    return out


def _context_text(messages: list[dict[str, Any]], before: int) -> str:
    """Everything said before message ``before``, as one lowercase string."""
    parts: list[str] = []
    for message in messages[:before]:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            parts.append(json.dumps(tool_calls))
    return " ".join(parts).lower()


def find_propagation_opportunities(
    messages: list[dict[str, Any]],
) -> list[PropagationOpportunity]:
    """Find arguments that can only be filled by reading the previous response.

    An opportunity is an argument of link ``i`` whose value appears in link
    ``i-1``'s response and does NOT appear anywhere earlier in the conversation.
    The second condition is what makes this a test of the model rather than of
    the corpus: a value the user already said can be echoed without ever reading
    the tool result.
    """
    links = extract_tool_chains(messages)
    opportunities: list[PropagationOpportunity] = []

    for i in range(1, len(links)):
        previous, current = links[i - 1], links[i]
        response_values = _extract_leaf_values(previous.response)
        if not response_values:
            continue

        # Everything said before the response that introduced the value.
        boundary = previous.response_index if previous.response_index >= 0 else current.call_index
        earlier = _context_text(messages, boundary)

        for path, value in _flatten_arguments(current.arguments).items():
            if value not in response_values:
                continue
            if value.lower() in earlier:
                continue
            opportunities.append(
                PropagationOpportunity(
                    link_index=i,
                    tool_name=current.tool_name,
                    argument_path=path,
                    expected_value=value,
                    depth=min(current.depth, 4),
                )
            )

    return opportunities


def extract_tool_calls_in_order(messages: list[dict[str, Any]]) -> list[ToolChainLink]:
    """Every tool call in order, whether or not a response followed it.

    :func:`extract_tool_chains` requires a response after the call, which is
    right for the ground-truth side (an opportunity needs the response that
    produced the value) and wrong for the prediction side: the model's last call
    of a conversation usually has no response after it, and dropping it would
    score a call the model did make as a call it missed.
    """
    links: list[ToolChainLink] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = parse_tool_calls(message.get("content", ""))
        if not calls:
            calls = (message.get("annotations", {}) or {}).get("tool_calls", []) or []
            if not calls:
                structured = message.get("tool_calls") or []
                calls = [
                    {
                        "name": (c.get("function", {}) or {}).get("name", c.get("name", "")),
                        "arguments": _coerce_arguments(
                            (c.get("function", {}) or {}).get("arguments", c.get("arguments", {}))
                        ),
                    }
                    for c in structured
                ]
        for call in calls:
            links.append(
                ToolChainLink(
                    tool_name=call.get("name", ""),
                    arguments=call.get("arguments", {}) or {},
                    response={},
                    depth=min(len(links) + 1, 4),
                    call_index=index,
                )
            )
    return links


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """Tool-call arguments arrive as a dict or as a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _align_links(
    pred_links: list[ToolChainLink], gt_links: list[ToolChainLink]
) -> dict[int, ToolChainLink]:
    """Match predicted calls to ground-truth calls by tool name, in order.

    Index alignment breaks as soon as the model makes one extra or one fewer
    call, which would then score every later link against the wrong target.
    Matching by name in order tolerates that.
    """
    matched: dict[int, ToolChainLink] = {}
    cursor = 0
    for gt_index, gt_link in enumerate(gt_links):
        for pred_index in range(cursor, len(pred_links)):
            if pred_links[pred_index].tool_name == gt_link.tool_name:
                matched[gt_index] = pred_links[pred_index]
                cursor = pred_index + 1
                break
    return matched


def evaluate_chain_propagation(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> ChainPropagationMetrics:
    """Evaluate tool chain propagation accuracy.

    Only real propagation opportunities are scored: an argument of call N+1
    whose ground-truth value comes from call N's response and appears nowhere
    earlier in the conversation (see :func:`find_propagation_opportunities`).
    A conversation with no such argument contributes nothing, rather than
    contributing guaranteed failures as it did before.

    Args:
        predictions: List of conversation prediction dicts with 'messages'.
        ground_truth: List of conversation ground-truth dicts with 'messages'.

    Returns:
        ChainPropagationMetrics with overall and per-depth accuracy.
    """
    depth_correct: dict[int, int] = {}
    depth_total: dict[int, int] = {}
    total_correct = 0
    total_opportunities = 0
    missed_calls = 0
    conversations_with_chains = 0

    for pred, gt in zip(predictions, ground_truth):
        gt_messages = gt.get("messages", [])
        opportunities = find_propagation_opportunities(gt_messages)
        if not opportunities:
            continue

        conversations_with_chains += 1
        gt_links = extract_tool_chains(gt_messages)
        pred_links = extract_tool_calls_in_order(pred.get("messages", []))
        matched = _align_links(pred_links, gt_links)

        for opportunity in opportunities:
            depth = opportunity.depth
            depth_total[depth] = depth_total.get(depth, 0) + 1
            total_opportunities += 1

            pred_link = matched.get(opportunity.link_index)
            if pred_link is None:
                missed_calls += 1
                continue

            predicted = _flatten_arguments(pred_link.arguments)
            if predicted.get(opportunity.argument_path) == opportunity.expected_value:
                depth_correct[depth] = depth_correct.get(depth, 0) + 1
                total_correct += 1

    per_depth: dict[int, float] = {}
    for d in sorted(depth_total.keys()):
        per_depth[d] = depth_correct.get(d, 0) / depth_total[d]

    metrics = ChainPropagationMetrics(
        chain_propagation_accuracy=(
            total_correct / total_opportunities if total_opportunities else 0.0
        ),
        per_depth_accuracy=per_depth,
        total_chains=total_opportunities,
        missed_calls=missed_calls,
        conversations_with_chains=conversations_with_chains,
    )

    logger.info("chain_propagation_eval_complete", **metrics.to_dict())
    return metrics
