"""Shared system-prompt enrichment for Task A workflow datasets.

Single source of truth for FORMAT_RULES and the enrichment helper used by
both the data-generation pipeline and the benchmark harness.
"""

from __future__ import annotations

import json
from typing import Any

FORMAT_RULES = """\
Rules:
1. Always annotate every state transition using [STATE: CURRENT → NEXT] at the start of your response.
2. When calling a tool, emit it as <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>.
3. Only use tools available in the current state.
4. Follow transition conditions to move between states.
5. If a tool returns an error, attempt recovery before escalating.
6. Reach a terminal state to complete the workflow.
7. Never skip states or make invalid transitions."""


def build_enriched_system_prompt(sample: dict[str, Any], original_content: str) -> str:
    """Prepend workflow script, structured reference, and format rules to a system prompt.

    Idempotent: if *original_content* already contains the enrichment marker
    ``"Workflow script"`` the function returns it unchanged. This makes the
    benchmark harness safe to call on both legacy (bare) and new (pre-enriched)
    JSONL files without double-enriching.

    Args:
        sample: A dataset sample dict with optional keys ``workflow_script``,
            ``workflow_graph``, and ``tool_schemas``.
        original_content: The existing system-message content (may be bare or
            already enriched).

    Returns:
        Enriched system-message string.
    """
    if "Workflow script" in original_content:
        return original_content

    parts: list[str] = [original_content]

    script = sample.get("workflow_script")
    if script:
        parts.append(f"\nWorkflow script (follow this for conversation flow):\n{script}")

    graph = sample.get("workflow_graph", {})
    initial = graph.get("initial", "")
    terminal = graph.get("terminal", [])
    tool_schemas = sample.get("tool_schemas") or []
    tool_names = [t.get("function", {}).get("name", "") for t in tool_schemas]

    if initial or terminal or tool_names:
        parts.append(
            f"\nStructured reference:\n"
            f"  Initial state: {initial}\n"
            f"  Terminal states: {', '.join(terminal)}\n"
            f"  Available tools: {json.dumps(tool_names)}"
        )

    parts.append(f"\n{FORMAT_RULES}")
    return "\n".join(parts)
