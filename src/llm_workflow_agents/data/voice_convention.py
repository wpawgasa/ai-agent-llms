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


#: A turn can hold at most this many spoken characters and still conform.
SPOKEN_CHARS_MAX = TURN_MAX_CHUNKS * CHUNK_MAX_CHARS


def chunk_spoken_text(text: str) -> str:
    """Split plain spoken text into <S>...</S> chunks. Never delete content.

    Splits on sentence-ending punctuation first. A piece still above
    CHUNK_MAX_CHARS is split again on the last space before the limit, and
    failing that at the limit itself. Every spoken character of the input
    survives into some chunk; only the whitespace *between* chunks is dropped,
    because a chunk boundary is itself the pause.

    Raises ``ValueError`` when the input cannot be chunked within both limits
    — more than ``SPOKEN_CHARS_MAX`` characters of speech simply does not fit
    in one spoken turn. This used to truncate instead: a 2,289-character input
    came back as 776 characters of chunks and ``find_voice_violations``
    reported no violation, so a corpus builder deleted 66% of a turn in
    silence. Silent data loss in a corpus builder is the R12/R13 shape, so it
    fails loudly now. The caller authors the prose it passes here and is
    expected to keep a spoken turn short.

    Deterministic. The placeholder generator needs it, and the placeholder
    generator must stay reproducible.
    """
    pieces = [p.strip() for p in re.split(r"(?<=[.!?。ฯ])\s+", text.strip()) if p.strip()]
    if not pieces:
        pieces = [text.strip() or "..."]

    out: list[str] = []
    for piece in pieces:
        while len(piece) > CHUNK_MAX_CHARS:
            cut = piece.rfind(" ", 0, CHUNK_MAX_CHARS)
            if cut <= 0:
                cut = CHUNK_MAX_CHARS
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)

    # Too many chunks: merge the tail back into one chunk where it still fits
    # inside CHUNK_MAX_CHARS. That re-joins content, it does not drop any.
    if len(out) > TURN_MAX_CHUNKS:
        head = out[: TURN_MAX_CHUNKS - 1]
        tail = " ".join(out[TURN_MAX_CHUNKS - 1 :])
        if len(tail) > CHUNK_MAX_CHARS:
            raise ValueError(
                f"cannot chunk {len(text)} characters of speech within "
                f"{TURN_MAX_CHUNKS} chunks of {CHUNK_MAX_CHARS} characters "
                f"(limit {SPOKEN_CHARS_MAX}); shorten the turn instead of "
                f"letting the chunker delete the overflow"
            )
        out = head + [tail]

    return "".join(f"<S>{c}</S>" for c in out)


def strip_voice_markup(text: str) -> str:
    """Delete every voice marker, leaving the spoken words.

    Used by the held-out audit and the reward functions. Both compare a voice
    completion against a text-convention ground truth. The markup must not
    count as a difference.
    """
    out = text.replace("<S>", "").replace("</S>", "")
    out = out.replace(_END_MARKER, "").replace(_UNSPOKEN_MARKER, "")
    return out


def apply_barge_in_loss_flag(messages: list[dict[str, Any]]) -> bool:
    """Mark the interrupted turn as no-loss. Return whether one was found.

    The orchestrator writes ``<unspoken>`` into the model's own past turn when
    the caller barges in. The model never emits that marker, so training on the
    turn that carries it teaches the model to emit it — spec risk 3, and the
    same unconditional-habit shape risk R15 records.

    This is the single producer of the ``loss`` key. Every consumer
    (``render_response_only_sample``, ``_load_grpo_jsonl``,
    ``build_preference_pairs``, ``mine_model_negatives``) reads
    ``msg.get("loss", True)``, so:

    - a marker-bearing assistant turn gets ``loss: False``;
    - every other message gets any stale ``loss`` key deleted, so a
      conversation with no marker carries no ``loss`` key at all and the
      absent-key default (``True``) governs it.

    Deleting rather than trusting an inbound key matters: the teacher model is
    told to emit no extra keys, so a ``loss`` key on a turn with no marker can
    only be a hallucination, and honouring it would silently drop a real
    training target.

    Idempotent — safe to call more than once on the same message list.
    """
    found = False
    for msg in messages:
        content = msg.get("content")
        carries_marker = (
            msg.get("role") == "assistant"
            and isinstance(content, str)
            and _UNSPOKEN_MARKER in content
        )
        if carries_marker:
            msg["loss"] = False
            found = True
        else:
            msg.pop("loss", None)
    return found


def _check_voice_turn(content: str, turn_index: int) -> list[str]:
    """Return every format violation in one voice assistant turn.

    Chunk presence is orthogonal to every other rule, so no rule below may
    live only inside the "chunks present" branch unless it is genuinely
    meaningless without a chunk — see the per-rule notes. Rule 5's
    tool-call/[END_CONVERSATION] check is the one rule that is meaningful
    either way, so it runs once, before the branch, instead of being
    duplicated in both.
    """
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

    # Rule 5 (tool-call half): a turn that calls a tool must never also end
    # the conversation. This holds regardless of whether the turn carries any
    # spoken chunks — a silent tool-call turn can still illegally claim to
    # end the call — so it runs once here, before the chunk-presence branch,
    # rather than once per branch.
    if _END_MARKER in content and _TOOL_CALL_RE.search(content):
        violations.append(
            f"{where}: holds both [END_CONVERSATION] and a tool call; a turn "
            f"that calls a tool must not end the conversation"
        )

    chunks = iter_chunks(content)
    if not chunks:
        # Rule 3: a turn with no spoken text at all is legal and carries no
        # chunk — a turn that only calls a tool is silent on the line. The
        # production reference states this: "Format spoken text with `</S>`;
        # emit no delimiter when there is no speech." Determine "no spoken
        # text" the same way the chunked case's rule 3 check below does:
        # strip the state marker, every <tool_call> block, the control
        # markers, and any chunks (there are none here). Only flag this turn
        # if spoken text is left sitting outside a chunk.
        #
        # Rules 1, 2 and 5's other two halves (marker/tool-call/end-marker
        # "inside a chunk", and "end-marker before the final </S>") are all
        # relative to a chunk boundary and are vacuously satisfied here: with
        # zero chunks nothing can sit inside one, and "before the final </S>"
        # has no final </S> to be before. They do not need a chunkless
        # counterpart.
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
    # Rule 5 (remaining two halves — both meaningful only relative to a
    # chunk boundary, so they stay here rather than in the chunkless branch).
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


#: The voice format contract, stated in prose for a language model.
#:
#: `find_voice_violations` is the enforced contract; this is the same contract
#: written for a reader that cannot run code. Both the teacher prompt and the
#: serving system prompt render from THIS string, so the two can never drift.
#: The braces in the worked example are doubled because the caller may pass the
#: result through str.format; render_voice_format_rules does the substitution.
_VOICE_FORMAT_RULES_TEMPLATE = """\

VOICE MODE — this conversation is spoken aloud through a text-to-speech engine.
The orchestrator reads your output as a stream, finds each <S>...</S> chunk, and
sends it to the engine in order. Six extra rules apply:

- V1. Put the [STATE: X → Y] marker on the first line, OUTSIDE every <S>. The
  agent never speaks it.
- V2. Put every <tool_call> block OUTSIDE every <S>. The agent never speaks it.
- V3. Put every spoken word INSIDE a chunk. No spoken text may sit outside
  <S>...</S>. A turn with no spoken text at all is legal and carries no chunk;
  a turn that only calls a tool is silent on the line. Never invent filler
  speech to give such a turn a chunk.
- V4. Split at natural pause points. Keep a chunk to {chunk_target} characters
  and never above {chunk_max}. Keep a turn to {turn_target} chunks and never
  above {turn_max}.
- V5. Keep replies short. A spoken reply is one or two sentences. Use no
  markdown, no bullet points, no numbered lists, no headers.
- V6. End a terminal turn with [END_CONVERSATION] after the last </S>, outside
  the chunks. Never put it on a turn that also calls a tool.

Worked example of one voice assistant turn:
    [STATE: VERIFY_PATIENT → VERIFY_PATIENT]
    <S>ได้เลยค่ะ</S><S>ขออนุญาตตรวจสอบข้อมูลสักครู่นะคะ</S>
    <tool_call>{{"name": "request_referral", "arguments": {{"patient_id": "P12345"}}}}</tool_call>
"""


def render_voice_format_rules() -> str:
    """Return the voice format contract with the four limits substituted."""
    return _VOICE_FORMAT_RULES_TEMPLATE.format(
        chunk_target=CHUNK_TARGET_CHARS,
        chunk_max=CHUNK_MAX_CHARS,
        turn_target=TURN_TARGET_CHUNKS,
        turn_max=TURN_MAX_CHUNKS,
    )


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
