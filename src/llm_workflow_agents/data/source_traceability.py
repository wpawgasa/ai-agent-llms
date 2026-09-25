"""Single source of truth for value-traceability checks on Task A conversations.

A conversation is only fair to train on or score against if the model could
have produced every value the gold turns contain. Two failure shapes break
that, and both were measured in the Phase 1 benchmark and the SFT corpus
(CLAUDE.md R28):

1. **Unsourced tool arguments.** A gold call's required argument holds a value
   that appears nowhere before it -- not in the user's words, an earlier tool
   result, the rendered system prompt, or the session context. The model is
   scored against a value it cannot know.
2. **Unsourced facts in prose.** A gold assistant turn states a specific code,
   amount or percentage that no tool returned and nobody said. A model trained
   on it learns to invent specifics.

Plus three structural checks the same repair work needs: states that offer or
use more than one tool, identifier values reused across conversations (which
teach constants instead of copying), and consecutive self-loop turns in one
state that can be merged into a single turn.

**Advance-then-stay pairs are deliberately NOT flagged.** An advancing prose
turn ``[W -> X]`` followed by a tool turn ``[X -> X]`` is how the tool-call
stay convention enters a tool state; the two turns are kept as two turns.

Every check returns data, not prose, so a triage report can split *confident*
findings (identifier-shaped values) from *needs_review* ones (amounts, names,
dates -- which may be correct normalizations of the user's words, or arithmetic
on sourced values). Nothing here repairs anything.

Pure stdlib, like :mod:`state_convention`, so the validator, the generator's
repair loop and the triage script can all import it without cycles.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from llm_workflow_agents.data.state_convention import parse_assistant_turns

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_STATE_MARKER_RE = re.compile(r"\[STATE:[^\]]*\]")
# Voice and control markers that sit in assistant content but are not prose.
_NON_PROSE_RE = re.compile(r"</?S>|<unspoken>|\[END_CONVERSATION\]|\[TRANSFER\]")

# An uppercase prefix of two or more letters followed, somewhere, by a digit:
# INT-5541, CUST-882, ACC987654, PLAN_50GB, TG910, PREM20.
_IDENTIFIER_BODY = r"[A-Z]{2,}[-_]?[A-Z0-9_-]*\d[A-Z0-9_-]*"
_IDENTIFIER_RE = re.compile(rf"\b{_IDENTIFIER_BODY}\b")
_IDENTIFIER_FULL_RE = re.compile(_IDENTIFIER_BODY)
# Amounts written with thousands separators, and percentages. Plain digit runs
# are left alone: years, times and scores would drown the report.
_AMOUNT_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])")
_PERCENT_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?\s?%")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# An argument named like an identifier holds one whatever its value looks like:
# customer_id="C-8821" and new_plan_id="unlimited_50gb" are ids too.
_IDENTIFIER_ARGUMENT_RE = re.compile(r"(?:^|_)(?:id|code|number|no|ref|reference)$", re.IGNORECASE)
_PERCENT_WORDS = r"(?:%|percent|per cent|เปอร์เซ็นต์)"

#: Tokens that look like invented identifiers but are general knowledge.
DEFAULT_FACT_ALLOWLIST: frozenset[str] = frozenset({
    "COVID-19", "COVID19",
    "AES-128", "AES-256", "SHA-256", "ISO-27001", "ISO27001",
})


@dataclass(frozen=True)
class UnsourcedValue:
    """One value the model could not have produced from what it was shown."""

    msg_index: int
    kind: str                   # "argument" | "fact"
    value: str
    confidence: str             # "confident" | "needs_review"
    tool_name: str | None = None
    argument: str | None = None

    def describe(self) -> str:
        where = f"message {self.msg_index}"
        if self.kind == "argument":
            return (
                f"{where}: {self.tool_name}.{self.argument}={self.value!r} appears nowhere "
                f"before the call ({self.confidence})"
            )
        return f"{where}: prose states {self.value!r} with no source ({self.confidence})"


@dataclass(frozen=True)
class MultiToolState:
    """A state that offers, or a conversation that uses, more than one tool in it."""

    state: str
    kind: str                   # "offers" | "uses"
    tools: tuple[str, ...]
    msg_indices: tuple[int, ...] = ()


# --------------------------------------------------------------------------- matching


def is_identifier_shaped(value: str) -> bool:
    """True for code-like values (``INT-5541``), false for amounts, names and dates."""
    return bool(_IDENTIFIER_FULL_RE.fullmatch(str(value).strip()))


def _compact(text: str) -> str:
    """Casefold and drop every separator, so ``INT-5541`` matches ``int 5541``."""
    return re.sub(r"[\W_]+", "", text.casefold())


def _as_number(value: Any) -> str | None:
    """Canonical digit string for a numeric value, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    text = str(value).strip().replace(",", "")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    return None


def _numbers_in(text: str) -> set[str]:
    """Every whole number in ``text``, thousands separators removed."""
    found: set[str] = set()
    for raw in _NUMBER_RE.findall(re.sub(r"(?<=\d),(?=\d{3})", "", text)):
        number = float(raw)
        found.add(str(int(number)) if number.is_integer() else raw)
    return found


def value_is_sourced(value: Any, text: str) -> bool:
    """True if ``value`` appears in ``text``, ignoring case and separators.

    Numbers must match a whole number in the text, so ``5`` is not sourced by
    ``15,000`` but ``1000000`` is sourced by ``1,000,000``.
    """
    if isinstance(value, bool) or value is None:
        return True
    raw = str(value).strip()
    if not raw:
        return True

    number = _as_number(value)
    if number is not None:
        return number in _numbers_in(text)

    percent = re.fullmatch(r"(\d+(?:\.\d+)?)\s?%", raw)
    if percent:
        digits = re.escape(percent.group(1))
        return re.search(rf"(?<![\d.]){digits}\s?{_PERCENT_WORDS}", text.casefold()) is not None

    if raw.casefold() in text.casefold():
        return True
    compact = _compact(raw)
    return bool(compact) and compact in _compact(text)


# --------------------------------------------------------------------------- parsing


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    parts = [content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)]
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return " ".join(p for p in parts if p)


def _calls_in(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Tool calls from inline ``<tool_call>`` blocks, else structured ``tool_calls``."""
    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(message.get("content") or ""):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            calls.append(parsed)
    if calls:
        return calls
    for structured in message.get("tool_calls") or []:
        function = structured.get("function", structured)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append({"name": function.get("name", ""), "arguments": arguments})
    return calls


def _required_arguments(tool_schemas: Iterable[Mapping[str, Any]] | None) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {}
    for schema in tool_schemas or []:
        if not schema:
            continue
        function = schema.get("function", schema)
        name = function.get("name")
        if name:
            required[name] = list((function.get("parameters") or {}).get("required") or [])
    return required


def _base_sources(prompt_text: str, session_context: Mapping[str, Any] | None) -> list[str]:
    sources = [prompt_text or ""]
    if session_context:
        sources.append(json.dumps(session_context, ensure_ascii=False))
    return sources


def _prose(content: str) -> str:
    text = _STATE_MARKER_RE.sub(" ", content or "")
    text = _TOOL_CALL_RE.sub(" ", text)
    return _NON_PROSE_RE.sub(" ", text)


# --------------------------------------------------------------------------- checks


def find_unsourced_argument_values(
    messages: list[dict[str, Any]],
    tool_schemas: Iterable[Mapping[str, Any]] | None,
    prompt_text: str = "",
    session_context: Mapping[str, Any] | None = None,
) -> list[UnsourcedValue]:
    """Required tool-argument values that appear nowhere before their call.

    A value counts as sourced if it appears in the rendered system prompt, the
    session context, or any message *before* the calling turn. The calling
    turn itself does not count: prose that names the value in the same breath
    as the call was written by the same model.

    Only required arguments are checked (every argument, for a tool with no
    schema). A value is ``confident`` when it is identifier-shaped or its
    argument is named like one (``*_id``, ``*_code``, ``*_number``); anything
    else is ``needs_review``, because a date, a city or a description may be a
    correct normalization of what the user said in another form or language.
    """
    required = _required_arguments(tool_schemas)
    context = _base_sources(prompt_text, session_context)
    found: list[UnsourcedValue] = []

    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            seen = " ".join(context)
            for call in _calls_in(message):
                name = call.get("name", "")
                arguments = call.get("arguments") or {}
                if not isinstance(arguments, dict):
                    continue
                keys = required.get(name, list(arguments))
                for key in keys:
                    if key not in arguments:
                        continue
                    value = arguments[key]
                    if isinstance(value, (dict, list)) or isinstance(value, bool):
                        continue
                    if value_is_sourced(value, seen):
                        continue
                    found.append(
                        UnsourcedValue(
                            msg_index=index,
                            kind="argument",
                            value=str(value),
                            confidence=(
                                "confident"
                                if is_identifier_shaped(str(value)) or _IDENTIFIER_ARGUMENT_RE.search(key)
                                else "needs_review"
                            ),
                            tool_name=name,
                            argument=key,
                        )
                    )
        context.append(_message_text(message))

    return found


def find_unsourced_facts(
    messages: list[dict[str, Any]],
    prompt_text: str = "",
    session_context: Mapping[str, Any] | None = None,
    allowlist: frozenset[str] = DEFAULT_FACT_ALLOWLIST,
) -> list[UnsourcedValue]:
    """Specific values an assistant turn states before anything supplied them.

    Checks identifiers, thousands-separated amounts and percentages in
    assistant prose (state markers, tool calls and voice markers removed).
    Acronyms with no digit (``PDPA``, ``BKK``) are not facts. A value is
    reported once, at its first appearance: later mentions are sourced by it.

    Identifiers are ``confident``. Amounts and percentages are
    ``needs_review``, because one may be arithmetic on sourced values -- a new
    balance after a fee waiver is derived, not invented.
    """
    context = _base_sources(prompt_text, session_context)
    found: list[UnsourcedValue] = []

    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            seen = " ".join(context)
            prose = _prose(message.get("content") or "")
            tokens: list[tuple[str, str]] = []
            tokens += [(t, "confident") for t in _IDENTIFIER_RE.findall(prose)]
            tokens += [(t, "needs_review") for t in _AMOUNT_RE.findall(prose)]
            tokens += [(re.sub(r"\s", "", t), "needs_review") for t in _PERCENT_RE.findall(prose)]
            reported: set[str] = set()
            for token, confidence in tokens:
                if token in reported or token in allowlist:
                    continue
                reported.add(token)
                if value_is_sourced(token, seen):
                    continue
                found.append(UnsourcedValue(msg_index=index, kind="fact", value=token, confidence=confidence))
        context.append(_message_text(message))

    return found


def find_multi_tool_states(
    workflow_graph: Mapping[str, Any] | None,
    messages: list[dict[str, Any]],
) -> list[MultiToolState]:
    """States offering more than one tool, and states where a conversation uses more than one.

    Repeating one tool in a state (a retry after an error) is not multi-tool.
    """
    found: list[MultiToolState] = []

    details = (workflow_graph or {}).get("state_details") or []
    items = details.items() if isinstance(details, dict) else (
        ((d or {}).get("name"), d) for d in details
    )
    for name, info in items:
        tools = tuple((info or {}).get("tools") or [])
        if name and len(tools) > 1:
            found.append(MultiToolState(state=name, kind="offers", tools=tools))

    used: dict[str, list[str]] = defaultdict(list)
    where: dict[str, list[int]] = defaultdict(list)
    for label in parse_assistant_turns(messages):
        if label is None or not label.tool_names:
            continue
        for tool in label.tool_names:
            if tool not in used[label.from_state]:
                used[label.from_state].append(tool)
        where[label.from_state].append(label.msg_index)
    for state, tools in used.items():
        if len(tools) > 1:
            found.append(
                MultiToolState(state=state, kind="uses", tools=tuple(tools), msg_indices=tuple(where[state]))
            )

    return found


def collect_identifiers(messages: list[dict[str, Any]], vocabulary_text: str = "") -> set[str]:
    """Every identifier-shaped entity value in the conversation, any role.

    Identifiers that also appear in ``vocabulary_text`` -- normally the rendered
    system prompt -- are dropped: state names (``AUTHENTICATE_2FA``), tool names
    and schema enum values are instruction vocabulary every conversation in a
    domain shares, not entity values, and counting them as reuse would report
    the workflow graph instead of the data.
    """
    found: set[str] = set()
    for message in messages:
        found.update(_IDENTIFIER_RE.findall(_message_text(message)))
    if vocabulary_text:
        found -= set(_IDENTIFIER_RE.findall(vocabulary_text))
    return found


def find_identifier_reuse(rows: Mapping[str, set[str]]) -> dict[str, list[str]]:
    """Identifier values that occur in more than one row, mapped to those rows.

    Keys of ``rows`` must be unique per row -- a file path and line number, never
    ``conversation_id``, which repeats across the text and voice strata.
    """
    holders: dict[str, list[str]] = defaultdict(list)
    for key, identifiers in rows.items():
        for identifier in identifiers:
            holders[identifier].append(key)
    return {
        identifier: sorted(keys)
        for identifier, keys in sorted(holders.items())
        if len(keys) > 1
    }


def find_mergeable_stay_pairs(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Adjacent assistant turns that both stay in the same state.

    ``[X -> X]`` followed directly by ``[X -> X]`` says one thing in two turns
    and can be merged. An advancing turn followed by a stay (``[W -> X]`` then
    ``[X -> X]``) is how the stay convention enters a tool state, and is never
    returned.
    """
    labels = parse_assistant_turns(messages)
    pairs: list[tuple[int, int]] = []
    for first, second in zip(labels, labels[1:]):
        if first is None or second is None:
            continue
        if second.msg_index != first.msg_index + 1:
            continue
        states = {first.from_state, first.to_state, second.from_state, second.to_state}
        if len(states) == 1:
            pairs.append((first.msg_index, second.msg_index))
    return pairs


def find_orphan_tool_results(messages: list[dict[str, Any]]) -> list[str]:
    """Tool results that answer no tool call.

    A tool message must follow an assistant turn that makes a call (possibly
    with other tool messages in between, for a turn that makes several). A
    result after a turn that only *announces* the call ("let me get you
    qualified") teaches the model to narrate a call instead of making it --
    the announce-but-don't-call failure. Returns one line per orphan, worded as
    repair feedback for the teacher.
    """
    found: list[str] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        j = index - 1
        while j >= 0 and messages[j].get("role") == "tool":
            j -= 1
        previous = messages[j] if j >= 0 else {}
        if previous.get("role") == "assistant" and _calls_in(previous):
            continue
        found.append(
            f"message {index} is a tool result, but the assistant turn before it makes no "
            f"<tool_call>; put the <tool_call> in that turn, or remove the tool result"
        )
    return found


@dataclass(frozen=True)
class EnumViolation:
    """A gold tool-call value its own schema's ``enum`` does not allow."""

    msg_index: int
    tool: str
    argument: str
    value: Any
    allowed: list[Any]

    def describe(self) -> str:
        return (
            f"message {self.msg_index}: {self.tool}.{self.argument} is {self.value!r}, "
            f"but its schema allows only {self.allowed}"
        )


def _enum_of(spec: Mapping[str, Any]) -> list[Any] | None:
    """The allowed values for a parameter, looking inside an array's items."""
    if not isinstance(spec, Mapping):
        return None
    if isinstance(spec.get("enum"), list):
        return list(spec["enum"])
    items = spec.get("items")
    if isinstance(items, Mapping) and isinstance(items.get("enum"), list):
        return list(items["enum"])
    return None


def find_enum_violations(
    messages: list[dict[str, Any]], tool_schemas: Iterable[Mapping[str, Any]] | None
) -> list[EnumViolation]:
    """Gold tool-call values that their own schema's ``enum`` forbids.

    A conversation whose gold call asks for ``sort_by='cheapest_ever'`` while the
    schema lists ``['relevance','price_low','price_high','rating']`` scores every
    model wrong for answering correctly — 7 such cases on the v4 benchmark, each
    one a defect nothing else catches. Cheap, needs no model, and belongs in the
    generator's repair loop next to :func:`find_orphan_tool_results`.

    Case and spacing differences are NOT reported. The scorer is strict about
    them, but they are a formatting defect of a different kind, and mixing them
    in would bury the values that are genuinely off the list.

    A call the very next tool message REJECTS is not reported either. The corpus
    generates invalid tool inputs deliberately — 15% of conversations by spec —
    so the user asks for ``sort_by='cheapest_ever'``, the agent passes it
    through, the tool errors and the conversation tests recovery. All 8 rows
    this check first flagged on the v4 benchmark were that, and "fixing" them
    would have deleted the error-recovery arcs they exist to exercise.
    """
    properties: dict[str, dict[str, Any]] = {}
    for schema in tool_schemas or []:
        function = schema.get("function", schema)
        name = function.get("name")
        if not name:
            continue
        properties[name] = ((function.get("parameters") or {}).get("properties") or {})

    def rejected_next(index: int) -> bool:
        """True when the tool message after ``index`` reports an error."""
        for message in messages[index + 1:]:
            role = message.get("role")
            if role == "tool":
                content = message.get("content") or ""
                return '"error"' in content or '"Error"' in content
            if role in ("assistant", "user"):
                return False
        return False

    found: list[EnumViolation] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if rejected_next(index):
            continue
        calls = _calls_in(message) or (message.get("annotations") or {}).get("tool_calls") or []
        for call in calls:
            spec_by_argument = properties.get(call.get("name") or "")
            if not spec_by_argument:
                continue
            arguments = call.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            for argument, value in arguments.items():
                allowed = _enum_of(spec_by_argument.get(argument) or {})
                if allowed is None:
                    continue
                for item in value if isinstance(value, list) else [value]:
                    if any(_compact(str(item)) == _compact(str(option)) for option in allowed):
                        continue
                    found.append(
                        EnumViolation(index, call.get("name") or "", argument, item, allowed)
                    )
    return found
