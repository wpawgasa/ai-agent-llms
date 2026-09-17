"""Select GRPO prompt rows where a policy can plausibly be wrong in more than one way.

The Cat A single-turn RL probes found no headroom (CLAUDE.md R23): the
fine-tuned E4B scores exactly 1.0 greedily on 86.8% of sampled prompts. Most of
those prompts need no tool call at all, and the model already gets them right.
The failures that remain are argument errors — 77.5% of mined C2 mistakes — and
they come from carrying a value from an earlier tool result into a later call.

These filters keep the rows where that can go wrong, so a probe or a training
run spends its samples there instead of on turns every sample already solves.
"""

from __future__ import annotations

import json
from typing import Any

#: Shortest string value treated as a real carried value. Shorter strings
#: ("yes", "en", "1") appear in tool results by coincidence.
MIN_STRING_LEN = 4
#: Shortest number (as written) treated as a carried value, e.g. an order id.
MIN_NUMBER_DIGITS = 3


def _as_text(value: Any) -> str | None:
    """The literal a carried value would appear as inside a tool result, or None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        text = str(value)
        return text if sum(ch.isdigit() for ch in text) >= MIN_NUMBER_DIGITS else None
    if isinstance(value, str):
        text = value.strip()
        return text if len(text) >= MIN_STRING_LEN else None
    return None


def _flatten_values(arguments: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(arguments, dict):
        out: list[tuple[str, Any]] = []
        for key, value in arguments.items():
            out.extend(_flatten_values(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(arguments, list):
        out = []
        for i, value in enumerate(arguments):
            out.extend(_flatten_values(value, f"{prefix}[{i}]"))
        return out
    return [(prefix, arguments)]


def _arguments(call: dict[str, Any]) -> Any:
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {}
    return args


def propagated_arguments(
    prompt_messages: list[dict[str, Any]],
    gt_tool_calls: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Ground-truth arguments whose value first appears in an earlier tool result.

    Returns one ``{"tool", "argument", "value"}`` record per carried value. A
    value that only the user typed is not carried: copying from the user's own
    message is not the chain-propagation failure this filter targets.
    """
    tool_results = "\n".join(
        str(m.get("content") or "") for m in prompt_messages if m.get("role") == "tool"
    )
    if not tool_results:
        return []
    carried: list[dict[str, str]] = []
    for call in gt_tool_calls or []:
        for path, value in _flatten_values(_arguments(call)):
            text = _as_text(value)
            if text is not None and text in tool_results:
                carried.append({"tool": str(call.get("name", "")), "argument": path, "value": text})
    return carried


def is_tool_bearing(ground_truth: dict[str, Any]) -> bool:
    """The turn's ground truth calls at least one tool."""
    return bool(ground_truth.get("tool_calls"))


def is_chain_dependent(
    prompt_messages: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> bool:
    """The turn calls a tool with at least one argument carried from a tool result."""
    return is_tool_bearing(ground_truth) and bool(
        propagated_arguments(prompt_messages, ground_truth.get("tool_calls") or [])
    )


#: Filter names accepted by the probe's ``--prompt-filter``.
PROMPT_FILTERS = ("none", "tool_bearing", "chain_dependent")


def keep_row(name: str, prompt_messages: list[dict[str, Any]], ground_truth: dict[str, Any]) -> bool:
    """Apply the named filter to one decoded row."""
    if name == "none":
        return True
    if name == "tool_bearing":
        return is_tool_bearing(ground_truth)
    if name == "chain_dependent":
        return is_chain_dependent(prompt_messages, ground_truth)
    raise ValueError(f"unknown prompt filter {name!r}; expected one of {PROMPT_FILTERS}")
