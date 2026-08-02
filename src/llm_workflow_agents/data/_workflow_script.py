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
        "instruction": "Instruction: {text}",
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
        "instruction": "คำแนะนำ: {text}",
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

# tool_turn_semantics=True templates (opt-in; see build_workflow_script). These
# rewrite the plain "On success: proceed to [Y]" line for tool-bearing states so
# it cannot be read as "advance on the turn that calls the tool" — the exact
# misreading a design review traced a real defect to. The turn that issues the
# tool call stays in the current state; advancing to the next state happens on
# a LATER turn, after the tool result has come back.
_TOOL_TURN_NOTE: dict[str, str] = {
    "en": "- The turn that issues the <tool_call> stays in [{name}]: [STATE: {name} → {name}]",
    "th": "- เทิร์นที่เรียกใช้เครื่องมือ <tool_call> จะคงอยู่ที่ [{name}]: [STATE: {name} → {name}]",
}
_PRIMARY_BRANCH_TOOL: dict[str, str] = {
    "en": "- On success: proceed to [{to}] — on a LATER turn, after the tool result has come back",
    "th": "- เมื่อสำเร็จ: ดำเนินการต่อที่ [{to}] — ในเทิร์นถัดไป หลังได้รับผลลัพธ์จากเครื่องมือแล้ว",
}
_FALLBACK_TEXT: dict[str, str] = {
    "en": "follow the workflow's error path, or state plainly the step cannot be completed and hand off",
    "th": "ดำเนินการตามเส้นทางข้อผิดพลาดของขั้นตอนงาน หรือแจ้งตรงๆ ว่าไม่สามารถทำขั้นตอนนี้ให้เสร็จได้และส่งต่อ",
}
# Branches on retry_budget == 1 vs >= 2 for the same reason system_prompt.py's
# _retry_rule() does (see _RETRY_RULE_NO_RETRY / _RETRY_RULE_WITH_RETRIES
# there): a budget of 1 means "no retry at all", and phrasing that as "retry at
# most 1 time(s)" is both ungrammatical and semantically backwards. This is a
# small, deliberate duplication rather than an import from system_prompt.py —
# that module's rule is English-only and renders a multi-line system-prompt
# paragraph (with its own "[STATE: X → X]" framing); this one needs a single
# script line in both "en" and "th". The retry_budget<=1 branch point mirrors
# _retry_rule exactly; keep the two in sync if either wording changes.
_RETRY_NOTE_NO_RETRY: dict[str, str] = {
    "en": "- On tool error: stay in [{name}] and do NOT retry — this workflow allows only one attempt per call; after that, {fallback}",
    "th": "- หากเครื่องมือผิดพลาด: คงอยู่ที่ [{name}] และห้ามลองใหม่ — ขั้นตอนนี้อนุญาตให้ลองได้เพียงครั้งเดียว; หลังจากนั้น {fallback}",
}
_RETRY_NOTE_WITH_RETRIES: dict[str, str] = {
    "en": "- On tool error: stay in [{name}] and retry at most {n} times; after that, {fallback}",
    "th": "- หากเครื่องมือผิดพลาด: คงอยู่ที่ [{name}] และลองใหม่ได้ไม่เกิน {n} ครั้ง; หลังจากนั้น {fallback}",
}


def _retry_note(language: str, name: str, retry_budget: int) -> str:
    """Render the tool-error retry note for one tool-bearing state.

    Mirrors ``system_prompt.py::_retry_rule``'s branch on ``retry_budget <= 1``
    (see the comment above ``_RETRY_NOTE_NO_RETRY``/``_RETRY_NOTE_WITH_RETRIES``).
    """
    fallback = _FALLBACK_TEXT.get(language, _FALLBACK_TEXT["en"])
    if retry_budget <= 1:
        template = _RETRY_NOTE_NO_RETRY.get(language, _RETRY_NOTE_NO_RETRY["en"])
        return template.format(name=name, fallback=fallback)
    template = _RETRY_NOTE_WITH_RETRIES.get(language, _RETRY_NOTE_WITH_RETRIES["en"])
    return template.format(name=name, n=retry_budget, fallback=fallback)


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


def find_tool_placement_violations(
    allowed_tools_by_state: dict[str, Any],
    messages: list[dict[str, Any]],
    schema_names: set[str] | None = None,
) -> list[str]:
    """Find ground-truth tool calls that violate the curated tool placement.

    A conversation is *coherent* when every tool it calls is allowed in the state
    it is called from. This walks the GT conversation (via
    :func:`infer_state_tools_from_messages`) and returns one human-readable
    description per violation:

    * a tool called in a state whose ``allowed_tools_by_state`` set does not list
      it, or
    * (when ``schema_names`` is provided) a tool not present in the sample's tool
      schemas at all (e.g. a hallucinated tool).

    States not present in ``allowed_tools_by_state`` are skipped — there is no
    curated expectation to judge them against. An empty list means coherent.

    Single source of truth shared by the data validator and the generator's
    inline repair loop.
    """
    violations: list[str] = []
    for state, called in infer_state_tools_from_messages(messages).items():
        allowed = allowed_tools_by_state.get(state)
        for tool in called:
            if schema_names is not None and tool not in schema_names:
                violations.append(
                    f"tool '{tool}' called in state '{state}' is not in tool_schemas"
                )
            elif allowed is not None and tool not in allowed:
                violations.append(
                    f"tool '{tool}' called in state '{state}' is not listed in "
                    f"that state's tools"
                )
    return violations


def find_continuity_violations(
    messages: list[dict[str, Any]],
    initial_state: str,
    terminal_states: set[str],
) -> list[str]:
    """Find state-annotation continuity violations in a conversation.

    The per-edge legality check cannot see *sequence*-level incoherence: a
    conversation whose every edge is individually legal can still skip states,
    start mid-graph, or stop short of a terminal. This walks the assistant
    turns in order and returns one human-readable description per violation:

    * an assistant turn with extra ``[STATE: X → Y]`` markers mid-content, or
      whose first marker is not at the start of the content,
    * an assistant turn with no state annotation at all (neither a structured
      ``annotations.state_transition`` nor an inline marker),
    * consecutive annotations that do not chain (turn N's ``to`` differs from
      turn N+1's ``from``),
    * a first annotation that does not start at ``initial_state``,
    * a last annotation that does not end in ``terminal_states``.

    ``initial_state`` and ``terminal_states`` are state *names* (callers map
    IDs to names). An empty list means coherent.

    Single source of truth shared by the data validator and the generator's
    inline repair loop.
    """
    violations: list[str] = []
    sequence: list[tuple[int, str, str]] = []  # (turn index, from, to)
    turn = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        turn += 1
        content = msg.get("content") or ""
        markers = list(_STATE_RE.finditer(content))
        if markers:
            if len(markers) > 1:
                violations.append(
                    f"assistant turn {turn} has {len(markers) - 1} extra "
                    f"[STATE:] marker(s) mid-content"
                )
            lead = len(content) - len(content.lstrip())
            if markers[0].start() != lead:
                violations.append(
                    f"assistant turn {turn} [STATE:] marker is not at the "
                    f"start of the message"
                )
        transition = (msg.get("annotations") or {}).get("state_transition") or {}
        src, dst = transition.get("from"), transition.get("to")
        if not (src and dst) and markers:
            src, dst = markers[0].group(1), markers[0].group(2)
        if src and dst:
            sequence.append((turn, src, dst))
        else:
            violations.append(f"assistant turn {turn} missing state annotation")

    if not sequence:
        return violations

    first_turn, first_src, _ = sequence[0]
    if initial_state and first_src != initial_state:
        violations.append(
            f"assistant turn {first_turn} starts at '{first_src}', "
            f"not the initial state '{initial_state}'"
        )
    for (_, _, prev_dst), (cur_turn, cur_src, _) in zip(sequence, sequence[1:]):
        if cur_src != prev_dst:
            violations.append(
                f"assistant turn {cur_turn} transitions from '{cur_src}' but "
                f"the previous turn ended at '{prev_dst}'"
            )
    last_turn, _, last_dst = sequence[-1]
    if terminal_states and last_dst not in terminal_states:
        violations.append(
            f"assistant turn {last_turn} ends at '{last_dst}', "
            f"which is not a terminal state"
        )
    return violations


def find_shape_violations(
    messages: list[dict[str, Any]],
    initiator: str = "user",
) -> list[str]:
    """Find conversation-shape violations.

    * The opening turn must match ``initiator``: inbound (``"user"``)
      conversations open with a user message, outbound (``"agent"``) with the
      assistant's opener. A leading ``system`` message is skipped.
    * Consecutive assistant messages are allowed only when the later one is a
      pure tool-call turn (its content, after the leading ``[STATE:]`` marker,
      starts with ``<tool_call>``) — two prose turns in a row break
      strict-alternation chat templates (e.g. Mistral).

    An empty list means the shape is valid.

    Single source of truth shared by the data validator and the generator's
    inline repair loop.
    """
    violations: list[str] = []
    body = [m for m in messages if m.get("role") != "system"]
    if body:
        expected = "assistant" if initiator == "agent" else "user"
        first_role = body[0].get("role")
        if first_role != expected:
            violations.append(
                f"conversation initiator is '{initiator}' but the first "
                f"message role is '{first_role}' (expected '{expected}')"
            )
    for prev, cur in zip(body, body[1:]):
        if prev.get("role") == "assistant" and cur.get("role") == "assistant":
            content = (cur.get("content") or "").lstrip()
            stripped = _STATE_RE.sub("", content, count=1).lstrip()
            if not stripped.startswith("<tool_call>"):
                violations.append(
                    "consecutive assistant prose turns (no user or tool "
                    "message in between)"
                )
    return violations


def build_workflow_script(
    workflow_graph: dict[str, Any],
    tool_schemas: list[dict[str, Any]] | None = None,
    language: str = "en",
    messages: list[dict[str, Any]] | None = None,
    *,
    tool_turn_semantics: bool = False,
    retry_budget: int = 1,
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
        tool_turn_semantics: When ``True``, tool-bearing states get an extra
            "the turn that issues the tool call stays here" line, the primary
            success branch is reworded to say the advance happens on a LATER
            turn (after the tool result comes back, not the turn that calls
            it), and a retry-budget note is appended. Defaults to ``False`` so
            the byte-identical output Task C's teacher-API pipeline depends on
            (via ``_playbook_render.py``) is untouched. Task A call sites pass
            ``True``.
        retry_budget: Max total attempts (including the first) for a failing
            tool call in a tool-bearing state, used only when
            ``tool_turn_semantics`` is ``True``. A budget of 1 means no retry;
            see :func:`_retry_note`.
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

        instruction = (sd.get("instruction") or "").strip()
        if instruction:
            lines.append(t["instruction"].format(text=instruction))

        tools = inferred_tools.get(name) or sd.get("tools") or []
        if tools:
            tool_list = ", ".join(
                f"{n} ({tool_desc.get(n, '')})" if tool_desc.get(n) else n
                for n in tools
            )
            lines.append(t["tools_intro"].format(tools=tool_list))
        else:
            lines.append(t["no_tools"])

        has_tools = bool(tools)
        stay_convention = has_tools and tool_turn_semantics
        if stay_convention:
            lines.append(
                _TOOL_TURN_NOTE.get(language, _TOOL_TURN_NOTE["en"]).format(name=name)
            )

        for tr in sorted(outgoing.get(name, []), key=lambda x: x.get("priority", 0)):
            to_name = tr.get("to", "")
            if tr.get("priority", 0) == 0:
                if stay_convention:
                    lines.append(
                        _PRIMARY_BRANCH_TOOL.get(language, _PRIMARY_BRANCH_TOOL["en"]).format(to=to_name)
                    )
                    lines.append(_retry_note(language, name, retry_budget))
                else:
                    lines.append(t["primary_branch"].format(to=to_name))
            else:
                cond = humanise_condition(tr.get("condition", "")) or t["condition_fallback"]
                lines.append(t["alt_branch"].format(condition=cond, to=to_name))

        lines.append("")

    return "\n".join(lines).strip()
