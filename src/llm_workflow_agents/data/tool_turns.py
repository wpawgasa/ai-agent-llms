"""Put tool calls and tool results into a shape every chat template renders.

The Task A corpus writes a tool call as text inside an assistant message
(``<tool_call>{json}</tool_call>``) and its result as a following
``{"role": "tool"}`` message with no ``tool_call_id``. The Gemma-4 chat template
renders a ``tool`` message only from inside an assistant message carrying
structured ``tool_calls``, so it silently dropped every tool result: 796 tool
messages rendered as 0 ``<|tool_response>`` blocks. Every Gemma-4 SFT run on
this corpus trained without seeing a tool result, and learned to narrate
results it never received. See
docs/superpowers/specs/2026-09-17-gemma4-tool-results-design.md.

:func:`to_text_tool_turns` is the one transform every render site applies —
SFT, the GRPO loader (and through it the held-out audit, probes and DPO), the
trajectory rollout, and the benchmark's ``--tool-turn-format text`` — so the
model sees the same format when it trains and when it is scored.
"""

from __future__ import annotations

import json
from typing import Any

#: Prefix marking a user turn that is really a tool result. Fixed text, so the
#: format is identical at training and at inference.
TOOL_RESULT_PREFIX = "[Tool result]: "

_TOOL_ONLY_KEYS = {"tool_call_id", "name", "tool_calls"}


def _call_as_text(call: dict[str, Any]) -> str:
    """One structured tool call as corpus text.

    ``json.dumps(..., ensure_ascii=False)`` reproduces the corpus's own
    ``<tool_call>`` text for 99.8% of calls (11,950 of 11,974 checked).
    """
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            pass
    payload = {"name": function.get("name", ""), "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def is_tool_result_turn(message: dict[str, Any]) -> bool:
    """A ``tool`` message, or a user turn this module converted from one."""
    if message.get("role") == "tool":
        return True
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith(TOOL_RESULT_PREFIX)
    )


def tool_result_text(message: dict[str, Any]) -> str:
    """The tool result's own text, without the prefix."""
    content = str(message.get("content") or "")
    if message.get("role") == "user" and content.startswith(TOOL_RESULT_PREFIX):
        return content[len(TOOL_RESULT_PREFIX):]
    return content


def to_text_tool_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages with tool calls as text and tool results as user turns.

    - An assistant message with structured ``tool_calls`` gets each call
      appended to ``content`` as ``<tool_call>{json}</tool_call>`` text and
      loses the field. Text tool calls already in ``content`` are untouched.
    - A ``tool`` message becomes ``{"role": "user", "content":
      TOOL_RESULT_PREFIX + content}``.
    - Every other key (``annotations``, ``loss``) is kept, and messages are
      copied, never mutated. Idempotent.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            converted = {k: v for k, v in message.items() if k not in _TOOL_ONLY_KEYS}
            converted["role"] = "user"
            converted["content"] = TOOL_RESULT_PREFIX + str(message.get("content") or "")
            out.append(converted)
        elif role == "assistant" and message.get("tool_calls"):
            converted = {k: v for k, v in message.items() if k != "tool_calls"}
            text = str(message.get("content") or "")
            calls = "\n".join(_call_as_text(c) for c in message["tool_calls"])
            converted["content"] = f"{text}\n{calls}" if text else calls
            out.append(converted)
        else:
            out.append(dict(message))
    return out
