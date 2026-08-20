"""The teacher prompts state the voice contract in prose; keep them in step.

`voice_convention.py` is the enforced contract — `find_voice_violations` is
what decides whether a row enters the corpus. The same contract is stated a
second and third time, as prose, in `_VOICE_RULES` and `_RICH_VOICE_OVERRIDE`.
Those copies are deliberately not generated from the module (see CLAUDE.md
R20), so nothing but a test stops them drifting.

Drift is only self-correcting in one direction. A prompt that under-states a
rule produces rows the checker rejects and the repair loop regenerates, at API
cost. A prompt that names a marker the checker no longer knows about is dead
prose. Either is worth catching here rather than during a paid 2,400-row run.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.data import voice_convention as vc
from llm_workflow_agents.data.generate_workflows import (
    _RICH_VOICE_OVERRIDE,
    _teacher_system_prompt,
)


@pytest.fixture(scope="module")
def voice_prompt() -> str:
    return _teacher_system_prompt("voice")


def test_voice_prompt_renders_the_module_limits(voice_prompt):
    """The four numeric limits come from the module, not from a literal."""
    for value in (
        vc.CHUNK_TARGET_CHARS,
        vc.CHUNK_MAX_CHARS,
        vc.TURN_TARGET_CHUNKS,
        vc.TURN_MAX_CHUNKS,
    ):
        assert str(value) in voice_prompt


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
