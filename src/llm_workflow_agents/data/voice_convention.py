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


def find_voice_violations(
    messages: list[dict[str, Any]], modality: str
) -> list[str]:
    """Return every format violation in one conversation.

    For a voice conversation this checks the five format rules, both length
    limits, and the barge-in recovery. For a text conversation it checks that
    no voice marker appears at all. Both directions matter. Without the second
    the modality field is advisory.
    """
    if modality == "text":
        return _find_text_violations(messages)
    violations: list[str] = []
    turn_index = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        turn_index += 1
        violations.extend(_check_voice_turn(msg.get("content") or "", turn_index))
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
