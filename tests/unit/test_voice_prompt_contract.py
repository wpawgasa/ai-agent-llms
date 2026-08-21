"""The voice format contract is checked here, not generated.

`voice_convention.py` defines the enforced contract — `find_voice_violations`
is what decides whether a row enters the corpus. The same contract is stated
as prose for a language model via `render_voice_format_rules()`. Both the
teacher prompt and the serving system prompt render from that one function, so
they can never drift.

The old test asserted only that certain strings appeared, so a reviewer inverted
a rule's meaning (changing "outside" to "inside") and every assertion still
passed. The new tests check identity instead: the exact bytes that
`render_voice_format_rules()` produces must appear in the teacher prompt.

`_RICH_VOICE_OVERRIDE` is a different instruction about authoring dialogue in
a system prompt, not a copy of the rules, so it is tested separately.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.data import voice_convention as vc
from llm_workflow_agents.data.generate_workflows import (
    _RICH_VOICE_OVERRIDE,
    _teacher_system_prompt,
)


def test_render_voice_format_rules_substitutes_every_limit():
    from llm_workflow_agents.data.voice_convention import render_voice_format_rules

    text = render_voice_format_rules()
    # No placeholder may survive. Do not assert "no braces at all": the worked
    # example contains real JSON, and .format() un-doubles those braces.
    for name in ("{chunk_target}", "{chunk_max}", "{turn_target}", "{turn_max}"):
        assert name not in text
    for value in (vc.CHUNK_TARGET_CHARS, vc.CHUNK_MAX_CHARS,
                  vc.TURN_TARGET_CHUNKS, vc.TURN_MAX_CHUNKS):
        assert str(value) in text


def test_prose_claims_match_enforced_contract():
    """Assert specific directional claims in the prose against what the code enforces.

    This test guards prose-against-code, not code-against-prose. The old test
    asserted only that "<S>" appeared, so inverting "OUTSIDE every <S>" to
    "INSIDE every <S>" passed all checks. The new checks pin the directional
    phrases themselves, so an inversion fails immediately.

    Before the fix: invert V1 from "OUTSIDE" to "INSIDE" → test passed.
    After the fix: invert V1 from "OUTSIDE" to "INSIDE" → test fails.
    """
    from llm_workflow_agents.data.voice_convention import render_voice_format_rules

    rules = render_voice_format_rules()

    # V1: The state marker must be OUTSIDE every <S>. The code enforces this at
    # voice_convention.py:252-257 by checking if the state marker appears
    # inside chunks (joined text) and flagging it as a violation. Pin this by
    # asserting the V1 text itself contains the directional phrase.
    assert "- V1. Put the [STATE: X → Y] marker on the first line, OUTSIDE every <S>" in rules, (
        "V1 must state state marker goes OUTSIDE chunks, not INSIDE"
    )

    # V2: Tool calls must be OUTSIDE every <S>. The code enforces this at
    # voice_convention.py:258-263 by checking if a tool call appears inside
    # chunks and flagging it. Pin by asserting V2's specific text.
    assert "- V2. Put every <tool_call> block OUTSIDE every <S>" in rules, (
        "V2 must state tool calls go OUTSIDE chunks"
    )

    # V3a: Spoken text must be INSIDE a chunk. The code enforces this at
    # voice_convention.py:283-291 by extracting remainder text after removing
    # all markers and chunks, then flagging if anything is left.
    assert "Put every spoken word INSIDE a chunk" in rules, (
        "V3 must state spoken text goes INSIDE chunks"
    )

    # V3b: A turn with no spoken text is legal and carries no chunk. The code
    # enforces this at voice_convention.py:208-233 by returning early with no
    # violations for chunkless turns.
    assert "A turn with no spoken text at all is legal and carries no chunk" in rules, (
        "V3 must state that turns with no spoken text are legal and carry no chunk"
    )

    # V3c: Never invent filler speech. The code enforces this at
    # voice_convention.py:209-223 by documenting the rule; it is not actively
    # checked but is a guideline for authoring. The prose must state it.
    assert "Never invent filler" in rules, (
        "V3 must forbid inventing filler speech"
    )

    # V6: [END_CONVERSATION] goes after the last </S> and never shares a turn
    # with a tool call. The code enforces this at voice_convention.py:196-205
    # (no tool calls on end-marker turns) and 266-278 (marker after last </S>).
    assert "End a terminal turn with [END_CONVERSATION] after the last </S>" in rules, (
        "V6 must state end marker goes after the last chunk"
    )
    assert "Never put it on a turn that also calls a tool" in rules, (
        "V6 must forbid end marker on turns with tool calls"
    )


def test_teacher_voice_prompt_embeds_the_shared_rules_verbatim():
    """Identity, not keyword presence. The old test asserted only that certain
    strings appeared, so inverting a rule's meaning left it passing."""
    from llm_workflow_agents.data.generate_workflows import _teacher_system_prompt
    from llm_workflow_agents.data.voice_convention import render_voice_format_rules

    assert render_voice_format_rules() in _teacher_system_prompt("voice")


def test_text_teacher_prompt_holds_no_voice_rules():
    from llm_workflow_agents.data.generate_workflows import _teacher_system_prompt
    from llm_workflow_agents.data.voice_convention import render_voice_format_rules

    assert render_voice_format_rules() not in _teacher_system_prompt("text")


@pytest.fixture(scope="module")
def voice_prompt() -> str:
    return _teacher_system_prompt("voice")


def test_voice_prompt_names_every_marker_the_checker_enforces(voice_prompt):
    """Each marker the checker positions must be positioned by the prompt too."""
    for marker in ("<S>", "</S>", "<tool_call>", "[STATE:", "[END_CONVERSATION]"):
        assert marker in voice_prompt, f"the voice teacher prompt never mentions {marker}"


def test_barge_in_rules_interpolate_the_module_openers():
    """Fact 3 of the recovery check reads ACKNOWLEDGEMENTS; so must the prompt.

    The openers reach the teacher through the user prompt, not the system
    prompt, and they must be interpolated rather than restated — a hardcoded
    list would let the checker and the teacher model disagree, and every
    barge-in the teacher wrote would then be rejected at API cost.
    """
    from llm_workflow_agents.data.generate_workflows import _BARGE_IN_RULES

    assert "{openers}" in _BARGE_IN_RULES
    openers = vc.ACKNOWLEDGEMENTS["en"]
    assert openers, "the English acknowledgement list is empty"
    rendered = _BARGE_IN_RULES.format(openers=", ".join(f'"{o}"' for o in openers))
    for opener in openers:
        assert opener in rendered
    # The marker itself is stated, not implied.
    assert "<unspoken>" in _BARGE_IN_RULES


def test_text_prompt_carries_no_voice_rules():
    """The text branch must stay byte-frozen; a voice block would contradict it."""
    text_prompt = _teacher_system_prompt("text")
    for marker in ("<S>", "VOICE MODE", "[END_CONVERSATION]", "<unspoken>"):
        assert marker not in text_prompt


def test_rich_voice_override_asks_for_chunks_and_forbids_control_markers():
    """The rich prompt inverts rule 4 for chunks ONLY.

    `[END_CONVERSATION]` in an authored system prompt is the L4_061_6 hazard;
    the override must keep telling the teacher not to write one, and
    `_RICH_PROMPT_FORBIDDEN["voice"]` must keep listing it for the scrubber.
    """
    assert "<S>" in _RICH_VOICE_OVERRIDE
    for marker in ("[END_CONVERSATION]", "<unspoken>", "[TRANSFER]"):
        assert marker in _RICH_VOICE_OVERRIDE, f"the override stops forbidding {marker}"


def test_rich_prompt_forbidden_covers_the_runtime_markers():
    from llm_workflow_agents.data.generate_workflows import _RICH_PROMPT_FORBIDDEN

    for modality in ("text", "voice"):
        forbidden = _RICH_PROMPT_FORBIDDEN[modality]
        for marker in ("[END_CONVERSATION]", "<unspoken>", "[TRANSFER]"):
            assert marker in forbidden
    # `<S>` is legal in a voice system prompt (the override asks for it) and
    # illegal in a text one.
    assert "<S>" in _RICH_PROMPT_FORBIDDEN["text"]
    assert "<S>" not in _RICH_PROMPT_FORBIDDEN["voice"]
