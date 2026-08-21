"""Tests for the voice conversation format convention."""

from __future__ import annotations

import json

import pytest

from llm_workflow_agents.data.voice_convention import (
    ACKNOWLEDGEMENTS,
    CHUNK_MAX_CHARS,
    TURN_MAX_CHUNKS,
    find_voice_violations,
    iter_chunks,
    strip_voice_markup,
)


def test_iter_chunks_returns_each_chunk_in_order():
    assert iter_chunks("<S>one</S><S>two</S>") == ["one", "two"]


def test_iter_chunks_returns_empty_list_when_no_chunks():
    assert iter_chunks("plain text") == []


def test_strip_voice_markup_deletes_chunk_tags():
    assert strip_voice_markup("<S>one</S><S>two</S>") == "onetwo"


def test_strip_voice_markup_deletes_end_conversation():
    assert strip_voice_markup("<S>bye</S>[END_CONVERSATION]") == "bye"


def test_strip_voice_markup_deletes_unspoken_marker():
    assert strip_voice_markup("<S>hel<unspoken>lo</S>") == "hello"


def test_strip_voice_markup_leaves_plain_text_alone():
    assert strip_voice_markup("plain text") == "plain text"


def _voice_turn(content: str) -> list[dict]:
    return [
        {"role": "user", "content": "สวัสดีค่ะ"},
        {"role": "assistant", "content": content},
    ]


def test_conforming_voice_turn_has_no_violations():
    msgs = _voice_turn("[STATE: GREET → GREET]\n<S>สวัสดีค่ะ</S><S>ยินดีให้บริการค่ะ</S>")
    assert find_voice_violations(msgs, "voice") == []


def test_rule_1_state_marker_inside_a_chunk_is_a_violation():
    msgs = _voice_turn("<S>[STATE: GREET → GREET]</S><S>สวัสดีค่ะ</S>")
    assert any("state marker" in v for v in find_voice_violations(msgs, "voice"))


def test_rule_2_tool_call_inside_a_chunk_is_a_violation():
    msgs = _voice_turn(
        '[STATE: GREET → GREET]\n'
        '<S>สักครู่นะคะ<tool_call>{"name": "f", "arguments": {}}</tool_call></S>'
    )
    assert any("tool call" in v for v in find_voice_violations(msgs, "voice"))


def test_rule_3_spoken_text_outside_a_chunk_is_a_violation():
    msgs = _voice_turn("[STATE: GREET → GREET]\nสวัสดีค่ะ<S>ยินดีค่ะ</S>")
    assert any("outside" in v for v in find_voice_violations(msgs, "voice"))


def test_rule_4_unbalanced_chunk_tags_are_a_violation():
    msgs = _voice_turn("[STATE: GREET → GREET]\n<S>สวัสดีค่ะ<S>ยินดีค่ะ</S>")
    assert any("balanced" in v for v in find_voice_violations(msgs, "voice"))


def test_rule_5_end_conversation_with_a_tool_call_is_a_violation():
    msgs = _voice_turn(
        '[STATE: GREET → GREET]\n<S>บายค่ะ</S>\n'
        '<tool_call>{"name": "f", "arguments": {}}</tool_call>[END_CONVERSATION]'
    )
    assert any("END_CONVERSATION" in v for v in find_voice_violations(msgs, "voice"))


def test_assistant_turn_with_no_chunks_at_all_is_a_violation():
    msgs = _voice_turn("[STATE: GREET → GREET]\nสวัสดีค่ะ")
    assert find_voice_violations(msgs, "voice") != []


def test_silent_tool_call_turn_with_no_chunk_is_legal():
    """A turn that only calls a tool is silent on the line (spec rule 3)."""
    msgs = _voice_turn(
        '[STATE: A → A]\n<tool_call>{"name": "check_eligibility", '
        '"arguments": {"customer_id": "C1"}}</tool_call>'
    )
    assert find_voice_violations(msgs, "voice") == []


def test_chunkless_turn_with_tool_call_and_end_conversation_is_a_violation():
    """Rule 5 (tool call + [END_CONVERSATION] together) applies even with no
    chunk — a silent tool-call turn must not also claim to end the call, or
    the orchestrator hangs up before the tool result is ever seen."""
    msgs = _voice_turn(
        '[STATE: A → A]\n<tool_call>{"name": "f", "arguments": {}}'
        '</tool_call>[END_CONVERSATION]'
    )
    violations = find_voice_violations(msgs, "voice")
    assert violations != []
    assert any("END_CONVERSATION" in v and "tool call" in v for v in violations)


def test_chunkless_turn_with_bare_prose_is_still_a_violation():
    """Rule 3 keeps flagging real spoken text left outside a chunk."""
    msgs = _voice_turn("[STATE: A → A]\nสวัสดีค่ะ")
    violations = find_voice_violations(msgs, "voice")
    assert violations != []
    assert any("outside every chunk" in v for v in violations)


def test_chunk_at_the_limit_is_accepted():
    body = "ก" * CHUNK_MAX_CHARS
    msgs = _voice_turn(f"[STATE: A → A]\n<S>{body}</S>")
    assert find_voice_violations(msgs, "voice") == []


def test_chunk_one_over_the_limit_is_a_violation():
    body = "ก" * (CHUNK_MAX_CHARS + 1)
    msgs = _voice_turn(f"[STATE: A → A]\n<S>{body}</S>")
    assert any("too long" in v for v in find_voice_violations(msgs, "voice"))


def test_turn_at_the_chunk_limit_is_accepted():
    body = "<S>ok</S>" * TURN_MAX_CHUNKS
    msgs = _voice_turn(f"[STATE: A → A]\n{body}")
    assert find_voice_violations(msgs, "voice") == []


def test_turn_one_over_the_chunk_limit_is_a_violation():
    body = "<S>ok</S>" * (TURN_MAX_CHUNKS + 1)
    msgs = _voice_turn(f"[STATE: A → A]\n{body}")
    assert any("too many chunks" in v for v in find_voice_violations(msgs, "voice"))


def test_text_conversation_with_a_chunk_tag_is_a_violation():
    msgs = _voice_turn("[STATE: A → A]\n<S>hello</S>")
    assert find_voice_violations(msgs, "text") != []


def test_text_conversation_with_end_conversation_is_a_violation():
    msgs = _voice_turn("[STATE: A → A]\nhello[END_CONVERSATION]")
    assert find_voice_violations(msgs, "text") != []


def test_plain_text_conversation_has_no_violations():
    msgs = _voice_turn("[STATE: A → A]\nHello, how can I help you today?")
    assert find_voice_violations(msgs, "text") == []


def test_acknowledgements_cover_both_languages():
    assert set(ACKNOWLEDGEMENTS) == {"th", "en"}
    assert all(ACKNOWLEDGEMENTS[lang] for lang in ACKNOWLEDGEMENTS)


def test_end_conversation_before_first_chunk_is_a_violation():
    msgs = _voice_turn("[STATE: A → A]\n[END_CONVERSATION]<S>hello</S>")
    assert any("before the final" in v for v in find_voice_violations(msgs, "voice"))


def test_end_conversation_between_chunks_is_a_violation():
    msgs = _voice_turn(
        "[STATE: A → A]\n<S>hello</S>[END_CONVERSATION]<S>world</S>"
    )
    assert any("before the final" in v for v in find_voice_violations(msgs, "voice"))


def test_end_conversation_after_final_chunk_is_accepted():
    msgs = _voice_turn("[STATE: A → A]\n<S>hello</S><S>world</S>[END_CONVERSATION]")
    assert find_voice_violations(msgs, "voice") == []


def test_non_string_content_does_not_crash():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    result = find_voice_violations(msgs, "voice")
    assert isinstance(result, list)


def _barge_in_conversation(recovery: str) -> list[dict]:
    return [
        {"role": "user", "content": "สวัสดีค่ะ"},
        {
            "role": "assistant",
            "content": "[STATE: A → A]\n<S>ราคาแพ็คเกจ<unspoken>อยู่ที่ 5999 บาทค่ะ</S>",
        },
        {"role": "user", "content": "เดี๋ยวก่อนค่ะ"},
        {"role": "assistant", "content": recovery},
    ]


def test_valid_barge_in_recovery_has_no_violations():
    msgs = _barge_in_conversation(
        "[STATE: A → A]\n<S>ขอโทษที่พูดแทรกนะคะ</S><S>ราคาอยู่ที่ 5999 บาทค่ะ</S>"
    )
    assert find_voice_violations(msgs, "voice") == []


def test_recovery_without_an_acknowledgement_is_a_violation():
    msgs = _barge_in_conversation("[STATE: A → A]\n<S>ราคาอยู่ที่ 5999 บาทค่ะ</S>")
    assert any("acknowledgement" in v for v in find_voice_violations(msgs, "voice"))


def test_recovery_that_advances_the_state_is_a_violation():
    msgs = _barge_in_conversation(
        "[STATE: A → B]\n<S>ขอโทษที่พูดแทรกนะคะ</S><S>ราคาอยู่ที่ 5999 บาทค่ะ</S>"
    )
    assert any("advances" in v for v in find_voice_violations(msgs, "voice"))


def test_two_unspoken_markers_are_a_violation():
    msgs = _barge_in_conversation(
        "[STATE: A → A]\n<S>ขอโทษที่พูดแทรกนะคะ<unspoken>ค่ะ</S>"
    )
    assert any("exactly once" in v for v in find_voice_violations(msgs, "voice"))


def test_unspoken_marker_in_the_last_turn_is_a_violation():
    msgs = [
        {"role": "user", "content": "สวัสดีค่ะ"},
        {"role": "assistant", "content": "[STATE: A → A]\n<S>ราคา<unspoken>ค่ะ</S>"},
    ]
    assert any("last" in v for v in find_voice_violations(msgs, "voice"))


def test_non_string_recovery_content_does_not_crash():
    msgs = [
        {"role": "user", "content": "สวัสดีค่ะ"},
        {
            "role": "assistant",
            "content": "[STATE: A → A]\n<S>ราคาแพ็คเกจ<unspoken>อยู่ที่ 5999 บาทค่ะ</S>",
        },
        {"role": "user", "content": "เดี๋ยวก่อนค่ะ"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    result = find_voice_violations(msgs, "voice")
    assert isinstance(result, list)


def test_marker_in_user_message_is_a_violation():
    msgs = [
        {"role": "user", "content": "สวัสดีค่ะ<unspoken>test"},
        {
            "role": "assistant",
            "content": "[STATE: A → A]\n<S>ราคาอยู่ที่ 5999 บาทค่ะ</S>",
        },
    ]
    violations = find_voice_violations(msgs, "voice")
    assert any("user" in v for v in violations)


def test_marker_in_tool_message_is_a_violation():
    msgs = [
        {"role": "user", "content": "สวัสดีค่ะ"},
        {
            "role": "assistant",
            "content": "[STATE: A → A]\n<S>ราคา</S>",
        },
        {
            "role": "tool",
            "content": "result<unspoken>",
        },
    ]
    violations = find_voice_violations(msgs, "voice")
    assert any("tool" in v for v in violations)


def test_chunk_spoken_text_splits_on_sentences():
    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    assert chunk_spoken_text("One. Two.") == "<S>One.</S><S>Two.</S>"


def test_chunk_spoken_text_never_exceeds_the_limits():
    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    # Long enough to need re-splitting, short enough to still fit one turn.
    long_text = ". ".join("word " * 30 for _ in range(4))
    result = chunk_spoken_text(long_text)
    chunks = iter_chunks(result)
    assert len(chunks) <= TURN_MAX_CHUNKS
    assert all(len(c) <= CHUNK_MAX_CHARS for c in chunks)


def test_chunk_spoken_text_preserves_every_spoken_character():
    """The chunker must not delete content to satisfy the limits.

    It used to: a 2,289-character input came back as 776 characters of chunks
    and find_voice_violations reported no violation. Silent data loss in a
    corpus builder is the R12/R13 shape.
    """
    import re as _re

    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    text = (
        "Thank you for holding. I have pulled up your account now. "
        "The outstanding balance is two hundred baht. Would you like to pay it today?"
    )
    chunks = iter_chunks(chunk_spoken_text(text))
    assert _re.sub(r"\s+", "", "".join(chunks)) == _re.sub(r"\s+", "", text)


def test_chunk_spoken_text_preserves_content_when_the_tail_is_merged():
    """The over-many-chunks path re-joins the tail; it must not truncate it."""
    import re as _re

    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    text = ". ".join(f"sentence number {i} here" for i in range(8))
    chunks = iter_chunks(chunk_spoken_text(text))
    assert len(chunks) <= TURN_MAX_CHUNKS
    assert _re.sub(r"\s+", "", "".join(chunks)) == _re.sub(r"\s+", "", text)


def test_chunk_spoken_text_fails_loudly_when_the_turn_cannot_fit():
    from llm_workflow_agents.data.voice_convention import (
        SPOKEN_CHARS_MAX,
        chunk_spoken_text,
    )

    too_long = ". ".join("word " * 40 for _ in range(12))
    assert len(too_long) > SPOKEN_CHARS_MAX
    with pytest.raises(ValueError, match="cannot chunk"):
        chunk_spoken_text(too_long)


def test_chunk_spoken_text_handles_empty_input():
    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    assert iter_chunks(chunk_spoken_text("")) == ["..."]


def test_apply_barge_in_loss_flag_marks_only_the_marker_turn():
    from llm_workflow_agents.data.voice_convention import apply_barge_in_loss_flag

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "<S>a <unspoken>b</S>"},
        {"role": "user", "content": "sorry"},
        {"role": "assistant", "content": "<S>Got it</S>"},
    ]
    assert apply_barge_in_loss_flag(messages) is True
    assert [m.get("loss") for m in messages] == [None, None, False, None, None]


def test_apply_barge_in_loss_flag_reports_absence_and_leaves_no_key():
    from llm_workflow_agents.data.voice_convention import apply_barge_in_loss_flag

    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "<S>hello</S>"},
    ]
    assert apply_barge_in_loss_flag(messages) is False
    assert not any("loss" in m for m in messages)


def test_apply_barge_in_loss_flag_deletes_a_stale_key():
    """A `loss` key on a turn with no marker can only be a hallucination; the
    teacher prompt forbids extra keys. Honouring it would silently drop a real
    training target."""
    from llm_workflow_agents.data.voice_convention import apply_barge_in_loss_flag

    messages = [{"role": "assistant", "content": "<S>hello</S>", "loss": False}]
    assert apply_barge_in_loss_flag(messages) is False
    assert "loss" not in messages[0]


def test_apply_barge_in_loss_flag_is_idempotent():
    from llm_workflow_agents.data.voice_convention import apply_barge_in_loss_flag

    messages = [{"role": "assistant", "content": "<S>a <unspoken>b</S>"}]
    apply_barge_in_loss_flag(messages)
    first = json.dumps(messages, ensure_ascii=False)
    apply_barge_in_loss_flag(messages)
    assert json.dumps(messages, ensure_ascii=False) == first


def test_apply_barge_in_loss_flag_ignores_a_marker_in_a_user_turn():
    """Only an assistant turn is ever a training target, so only an assistant
    turn can need masking. find_barge_in_violations rejects the misplaced
    marker separately."""
    from llm_workflow_agents.data.voice_convention import apply_barge_in_loss_flag

    messages = [{"role": "user", "content": "<unspoken>"}]
    assert apply_barge_in_loss_flag(messages) is False
    assert not any("loss" in m for m in messages)


def test_placeholder_generator_never_trips_the_chunker_limit():
    """The raise must not be reachable from the only production caller.

    ``chunk_spoken_text`` raises rather than truncating (silent data loss in a
    corpus builder is the R12/R13 shape). Its sole production caller is
    ``generate_workflows._render_turn``, which passes placeholder prose. If any
    placeholder line ever exceeded SPOKEN_CHARS_MAX the generator would crash
    mid-batch, so pin it: every spoken turn the offline generator writes must
    fit, at every level.
    """
    import tempfile
    from pathlib import Path

    from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
    from llm_workflow_agents.data.voice_convention import (
        SPOKEN_CHARS_MAX,
        iter_chunks,
    )

    with tempfile.TemporaryDirectory() as tmp:
        for level in ("L1", "L3", "L5"):
            meta = generate_workflow_dataset(
                level,
                num_samples=2,
                output_dir=Path(tmp) / level,
                seed=4242,
                modality_preset="voice_only",
            )
            with open(meta.output_files[0]) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    for msg in json.loads(line)["messages"]:
                        if msg.get("role") != "assistant":
                            continue
                        spoken = "".join(iter_chunks(msg.get("content") or ""))
                        assert len(spoken) <= SPOKEN_CHARS_MAX


def test_acknowledgement_for_code_switch_returns_thai():
    """Code-switched conversations are Thai-primary; English openers read wrong."""
    from llm_workflow_agents.data.voice_convention import ACKNOWLEDGEMENTS, acknowledgement_for

    assert acknowledgement_for("code_switch") == ACKNOWLEDGEMENTS["th"]


def test_acknowledgement_for_known_languages():
    from llm_workflow_agents.data.voice_convention import ACKNOWLEDGEMENTS, acknowledgement_for

    assert acknowledgement_for("th") == ACKNOWLEDGEMENTS["th"]
    assert acknowledgement_for("en") == ACKNOWLEDGEMENTS["en"]


def test_acknowledgement_for_unknown_language_falls_back_to_english():
    from llm_workflow_agents.data.voice_convention import ACKNOWLEDGEMENTS, acknowledgement_for

    assert acknowledgement_for("de") == ACKNOWLEDGEMENTS["en"]
