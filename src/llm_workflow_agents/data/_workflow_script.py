"""Workflow-script generation — pure dict-based, no heavy imports.

Single source of truth for converting a workflow graph into the human-readable
script that goes into the system prompt. Used by both the data-generation
pipeline (``generate_workflows.py``) and the runtime prompt builder
(``system_prompt.py``).

The script is built from three inputs:

1. ``state_details`` — list of {name, tools} entries.
2. ``transitions`` — list of {from, to, condition, priority} entries.
3. ``messages`` (optional) — when supplied, the actual tool calls in the
   conversation are used to **override** the per-state ``tools`` field. This
   self-heals samples whose upstream ``state_details[i].tools`` is wrong (a
   known data-generation bug where ``state.tools`` was filled randomly,
   independent of what the conversation actually calls).
"""

from __future__ import annotations

import re
from typing import Any

_SCRIPT_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "header": "### [{section}]",
        "initial_marker": "(initial state)",
        "terminal_marker": "This is the terminal state — end the conversation here.",
        "tools_intro": "Available tools: {tools}",
        "no_tools": "No tools available in this state.",
        "primary_branch": "- On success: proceed to [{to}]",
        "alt_branch": "- If {condition}: go to [{to}]",
        "condition_fallback": "alternative condition met",
    },
    "th": {
        "header": "### [{section}]",
        "initial_marker": "(สถานะเริ่มต้น)",
        "terminal_marker": "นี่คือสถานะสิ้นสุด — จบการสนทนาที่นี่",
        "tools_intro": "เครื่องมือที่ใช้ได้: {tools}",
        "no_tools": "ไม่มีเครื่องมือในสถานะนี้",
        "primary_branch": "- เมื่อสำเร็จ: ดำเนินการต่อที่ [{to}]",
        "alt_branch": "- หาก{condition}: ไปที่ [{to}]",
        "condition_fallback": "เงื่อนไขอื่น",
    },
}

_STATE_RE = re.compile(
    r"\[STATE:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:→|->)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]"
)


def humanise_condition(condition: str) -> str:
    """Convert a snake_case condition name to a readable phrase."""
    cleaned = (condition or "").replace("proceed_from_", "").replace("branch_", "")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\b[Ss]\d+\b", "", cleaned)
    cleaned = re.sub(r"\bto\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def infer_state_tools_from_messages(
    messages: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Walk a conversation and build {state_name: [tool_name, ...]} from GT.

    For each assistant turn, parse the leading ``[STATE: X → Y]`` marker and
    attribute any tool calls in that turn to state ``X``. Tool calls live in
    either ``annotations.tool_calls``, ``tool_calls`` (OpenAI format), or
    inline ``<tool_call>{...}</tool_call>`` tags in ``content``.

    Returns a mapping with deduplicated tool-name lists in first-seen order.
    """
    inferred: dict[str, list[str]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        st = _STATE_RE.search(msg.get("content") or "")
        if not st:
            continue
        from_state = st.group(1)

        tool_names: list[str] = []
        ann_tcs = (msg.get("annotations") or {}).get("tool_calls") or []
        for tc in ann_tcs:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            if name:
                tool_names.append(name)
        for tc in msg.get("tool_calls") or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            if name:
                tool_names.append(name)
        if not tool_names:
            for inline in re.findall(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                msg.get("content") or "",
                re.DOTALL,
            ):
                m = re.search(r'"name"\s*:\s*"([^"]+)"', inline)
                if m:
                    tool_names.append(m.group(1))

        if tool_names:
            bucket = inferred.setdefault(from_state, [])
            for name in tool_names:
                if name not in bucket:
                    bucket.append(name)
    return inferred


def build_workflow_script(
    workflow_graph: dict[str, Any],
    tool_schemas: list[dict[str, Any]] | None = None,
    language: str = "en",
    messages: list[dict[str, Any]] | None = None,
) -> str:
    """Build the per-state workflow script.

    Args:
        workflow_graph: Dict with ``state_details``, ``transitions``,
            ``initial``, ``terminal`` (the shape produced by
            ``WorkflowGraph.to_dict``).
        tool_schemas: OpenAI tool schemas (used for tool descriptions).
        language: Template language (``"en"`` or ``"th"``).
        messages: Optional ground-truth conversation. When provided, tools
            inferred from actual GT tool calls override the (possibly wrong)
            ``state_details[i].tools`` field.
    """
    t = _SCRIPT_TEMPLATES.get(language, _SCRIPT_TEMPLATES["en"])
    state_details = workflow_graph.get("state_details") or []
    transitions = workflow_graph.get("transitions") or []
    initial = workflow_graph.get("initial", "")
    terminals = set(workflow_graph.get("terminal") or [])

    tool_desc: dict[str, str] = {}
    for schema in tool_schemas or []:
        fn = schema.get("function") or schema
        tool_desc[fn.get("name", "")] = fn.get("description", "")

    inferred_tools = infer_state_tools_from_messages(messages) if messages else {}

    outgoing: dict[str, list[dict[str, Any]]] = {}
    for tr in transitions:
        outgoing.setdefault(tr.get("from", ""), []).append(tr)

    lines: list[str] = []
    for sd in state_details:
        name = sd.get("name", "")
        is_initial = name == initial
        is_terminal = name in terminals

        header = t["header"].format(section=name)
        if is_initial:
            header += f"  {t['initial_marker']}"
        lines.append(header)

        if is_terminal:
            lines.append(t["terminal_marker"])
            lines.append("")
            continue

        tools = inferred_tools.get(name) or sd.get("tools") or []
        if tools:
            tool_list = ", ".join(
                f"{n} ({tool_desc.get(n, '')})" if tool_desc.get(n) else n
                for n in tools
            )
            lines.append(t["tools_intro"].format(tools=tool_list))
        else:
            lines.append(t["no_tools"])

        for tr in sorted(outgoing.get(name, []), key=lambda x: x.get("priority", 0)):
            to_name = tr.get("to", "")
            if tr.get("priority", 0) == 0:
                lines.append(t["primary_branch"].format(to=to_name))
            else:
                cond = humanise_condition(tr.get("condition", "")) or t["condition_fallback"]
                lines.append(t["alt_branch"].format(condition=cond, to=to_name))

        lines.append("")

    return "\n".join(lines).strip()
