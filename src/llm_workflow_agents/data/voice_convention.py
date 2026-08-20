"""Single source of truth for the voice conversation format.

Convention: an assistant turn in a voice conversation splits its spoken text
into <S>...</S> chunks. The [STATE: X -> Y] marker and every <tool_call> block
sit OUTSIDE the chunks, because the agent never speaks them. A terminal turn
ends with [END_CONVERSATION], also outside the chunks.

See docs/superpowers/specs/2026-08-20-voice-conversation-generation-design.md.

Pure stdlib, no heavy imports, so generate_workflows, data_validator, and the
eval harness can all import this without import cycles. Mirrors the posture of
state_convention.py.
"""

from __future__ import annotations

import re
from typing import Any

from llm_workflow_agents.data._workflow_script import _STATE_RE

#: Target chunk length. A longer chunk delays audio playback.
CHUNK_TARGET_CHARS = 100
#: A chunk longer than this is a violation. The two reference prompts peak at 117.
CHUNK_MAX_CHARS = 160
#: Target chunk count for one assistant turn.
TURN_TARGET_CHUNKS = 3
#: A turn with more chunks than this is a violation. The reference prompts peak at 5.
TURN_MAX_CHUNKS = 5

_CHUNK_RE = re.compile(r"<S>(.*?)</S>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_END_MARKER = "[END_CONVERSATION]"
_UNSPOKEN_MARKER = "<unspoken>"

#: Openers a recovery turn may use after an interruption. The Thai entries come
#: from 4_barge_in_block in data/templates/monomax_prompt_parts.json and
#: data/templates/oceanlife_prompt_parts.json. The teacher prompt states the
#: same list, so the checker and the teacher model never disagree.
ACKNOWLEDGEMENTS: dict[str, tuple[str, ...]] = {
    "th": (
        "ขอโทษที่พูดแทรก",
        "รับทราบ",
        "โทษทีนะ",
        "อ๋อ เข้าใจแล้ว",
        "ได้เลย",
    ),
    "en": (
        "Sorry to interrupt",
        "Understood",
        "Got it",
        "Of course",
        "Right",
    ),
}


def iter_chunks(text: str) -> list[str]:
    """Return the spoken chunks of one turn, in order."""
    return _CHUNK_RE.findall(text)


def strip_voice_markup(text: str) -> str:
    """Delete every voice marker, leaving the spoken words.

    Used by the held-out audit and the reward functions. Both compare a voice
    completion against a text-convention ground truth. The markup must not
    count as a difference.
    """
    out = text.replace("<S>", "").replace("</S>", "")
    out = out.replace(_END_MARKER, "").replace(_UNSPOKEN_MARKER, "")
    return out


def _check_voice_turn(content: str, turn_index: int) -> list[str]:
    """Return every format violation in one voice assistant turn."""
    violations: list[str] = []
    where = f"assistant turn {turn_index}"

    # Rule 4 first: the other rules read chunk boundaries, so unbalanced tags
    # would make their messages meaningless.
    if content.count("<S>") != content.count("</S>"):
        violations.append(
            f"{where}: chunk tags are not balanced "
            f"({content.count('<S>')} <S> vs {content.count('</S>')} </S>); "
            f"every <S> needs one </S>"
        )
        return violations

    chunks = iter_chunks(content)
    if not chunks:
        violations.append(
            f"{where}: holds no <S>...</S> chunk; every spoken word in a voice "
            f"conversation must sit inside a chunk"
        )
        return violations

    joined = "".join(chunks)

    if len(chunks) > TURN_MAX_CHUNKS:
        violations.append(
            f"{where}: holds {len(chunks)} chunks, too many chunks for one "
            f"spoken reply; keep a turn to {TURN_TARGET_CHUNKS} chunks and "
            f"never more than {TURN_MAX_CHUNKS}"
        )
    for position, chunk in enumerate(chunks, start=1):
        if len(chunk) > CHUNK_MAX_CHARS:
            violations.append(
                f"{where}, chunk {position}: {len(chunk)} characters is too "
                f"long for one spoken chunk; keep a chunk to "
                f"{CHUNK_TARGET_CHARS} characters and never more than "
                f"{CHUNK_MAX_CHARS}"
            )

    # Rule 1.
    if _STATE_RE.search(joined):
        violations.append(
            f"{where}: the state marker sits inside a chunk; it is never "
            f"spoken, so put it on the first line outside every <S>"
        )
    # Rule 2.
    if _TOOL_CALL_RE.search(joined):
        violations.append(
            f"{where}: a tool call sits inside a chunk; it is never spoken, "
            f"so put it outside every <S>"
        )
    # Rule 5.
    if _END_MARKER in content and _TOOL_CALL_RE.search(content):
        violations.append(
            f"{where}: holds both [END_CONVERSATION] and a tool call; a turn "
            f"that calls a tool must not end the conversation"
        )
    if _END_MARKER in joined:
        violations.append(
            f"{where}: [END_CONVERSATION] sits inside a chunk; it is never "
            f"spoken, so put it after the last </S>"
        )
    if _END_MARKER in content:
        last_close_tag_pos = content.rfind("</S>")
        end_marker_pos = content.find(_END_MARKER)
        if end_marker_pos < last_close_tag_pos:
            violations.append(
                f"{where}: [END_CONVERSATION] appears before the final </S>; "
                f"it must follow the last chunk"
            )

    # Rule 3: delete the markers that are allowed outside a chunk, then delete
    # the chunks themselves. Anything left is unspoken text the agent would
    # never say aloud.
    remainder = _STATE_RE.sub("", content)
    remainder = _TOOL_CALL_RE.sub("", remainder)
    remainder = remainder.replace(_END_MARKER, "").replace(_UNSPOKEN_MARKER, "")
    remainder = _CHUNK_RE.sub("", remainder)
    if remainder.strip():
        violations.append(
            f"{where}: text sits outside every chunk: "
            f"{remainder.strip()[:60]!r}; all spoken text goes inside <S>...</S>"
        )

    return violations


def _all_acknowledgements() -> tuple[str, ...]:
    out: list[str] = []
    for openers in ACKNOWLEDGEMENTS.values():
        out.extend(openers)
    return tuple(out)


def find_barge_in_violations(messages: list[dict[str, Any]]) -> list[str]:
    """Check the four facts that make a barge-in recovery well formed.

    1. The marker appears exactly once in the conversation.
    2. The marker sits in an assistant turn that is not the last turn.
    3. The next assistant turn starts with a known acknowledgement.
    4. The next assistant turn annotates the same state as the interrupted turn.

    Fact 4 has a reason. A barge-in completes nothing, so the workflow does not
    advance. A conversation with no marker is valid and returns an empty list.
    """
    # Count marker across ALL messages to detect it in wrong roles
    total = sum(
        (msg.get("content") or "").count(_UNSPOKEN_MARKER)
        if isinstance(msg.get("content"), str)
        else 0
        for msg in messages
    )

    if total == 0:
        return []

    # Check if marker is in a non-assistant message
    for index, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        role = msg.get("role")
        if role != "assistant" and _UNSPOKEN_MARKER in content:
            return [
                f"the {_UNSPOKEN_MARKER} marker sits in a {role} message at index {index}; "
                f"it may only sit in an assistant turn"
            ]

    if total > 1:
        return [
            f"the {_UNSPOKEN_MARKER} marker appears {total} times; it must "
            f"appear exactly once in a conversation"
        ]

    marked = [
        index
        for index, msg in enumerate(messages)
        if msg.get("role") == "assistant"
        and isinstance(msg.get("content"), str)
        and _UNSPOKEN_MARKER in msg.get("content")
    ]

    marker_index = marked[0]
    later = [
        index
        for index, msg in enumerate(messages)
        if index > marker_index and msg.get("role") == "assistant"
    ]
    if not later:
        return [
            f"the {_UNSPOKEN_MARKER} marker sits in the last assistant turn; "
            f"the caller interrupted, so a recovery turn must follow"
        ]

    violations: list[str] = []
    interrupted = _STATE_RE.search(messages[marker_index].get("content") or "")
    recovery_content = messages[later[0]].get("content") or ""

    # Guard recovery_content read - if not a string, treat as empty
    if not isinstance(recovery_content, str):
        recovery_content = ""

    recovery = _STATE_RE.search(recovery_content)

    spoken = "".join(iter_chunks(recovery_content)).lstrip()
    if not spoken.startswith(_all_acknowledgements()):
        violations.append(
            f"the recovery turn opens with {spoken[:40]!r} and not with an "
            f"acknowledgement; open it with one of {_all_acknowledgements()}"
        )
    if interrupted and recovery and recovery.group(2) != interrupted.group(1):
        violations.append(
            f"the recovery turn advances the state from "
            f"{interrupted.group(1)} to {recovery.group(2)}; a barge-in "
            f"completes nothing, so annotate "
            f"[{interrupted.group(1)} -> {interrupted.group(1)}]"
        )
    return violations


def find_voice_violations(
    messages: list[dict[str, Any]], modality: str
) -> list[str]:
    """Return every format violation in one conversation.

    For a voice conversation this checks the five format rules and both length
    limits. For a text conversation it checks that no voice marker appears at
    all. Both directions matter. Without the second the modality field is
    advisory.
    """
    if modality == "text":
        return _find_text_violations(messages)
    violations: list[str] = []
    turn_index = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        turn_index += 1
        violations.extend(_check_voice_turn(content, turn_index))
    violations.extend(find_barge_in_violations(messages))
    return violations


def _find_text_violations(messages: list[dict[str, Any]]) -> list[str]:
    """Return a violation for every voice marker found in a text conversation."""
    violations: list[str] = []
    for index, msg in enumerate(messages):
        content = msg.get("content") or ""
        if not isinstance(content, str):
            continue
        for marker in ("<S>", _END_MARKER, _UNSPOKEN_MARKER):
            if marker in content:
                violations.append(
                    f"message {index} of a text conversation holds the voice "
                    f"marker {marker}; a text conversation carries no voice markers"
                )
    return violations
