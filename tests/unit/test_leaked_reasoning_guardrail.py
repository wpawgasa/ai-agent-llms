"""Harness-side guardrail against leaked reasoning in a model completion.

FORMAT_RULES rule 1 requires every assistant turn's visible text to begin with
a ``[STATE: X → Y]`` annotation; ``NO_REASONING_LEAK_RULE`` (system_prompt.py,
added 2026-09-07) additionally forbids anything else appearing before it. A
frontier-model benchmark run (gemini-3.8-flash) observed one turn whose
visible content opened with internal self-talk instead of the annotation — a
decoding-side artifact a system-prompt instruction alone cannot be relied on
to prevent. ``_strip_leaked_reasoning_prefix`` is the harness-side guardrail:
it strips any text preceding the first ``[STATE: ...]`` match (recovering a
clean turn for the conversation context forwarded to later turns) and logs
the event either way, so a recurrence is visible in the run log without a
manual replay.
"""

from __future__ import annotations

from llm_workflow_agents.eval.agent_benchmark import _strip_leaked_reasoning_prefix


def test_well_formed_turn_is_unchanged():
    content = "[STATE: GREETING → GREETING]\nHello! How can I help?"
    assert _strip_leaked_reasoning_prefix(content) == content


def test_leaked_prefix_before_a_valid_annotation_is_stripped():
    leaked = (
        "So the tool was executed even with speech!\n"
        "Then why didn't `change_plan` execute?\n"
        "Wait! Look closely at the turn before:\n"
        "[STATE: PROCESS_CHANGE → CONFIRM_CHANGES]\n"
        "Your plan has been updated."
    )
    result = _strip_leaked_reasoning_prefix(leaked)
    assert result == "[STATE: PROCESS_CHANGE → CONFIRM_CHANGES]\nYour plan has been updated."


def test_content_with_no_annotation_anywhere_is_left_untouched():
    """Nothing to recover — existing state/tool scoring already treats a turn
    with no parseable annotation as a missing transition."""
    content = "So the tool was executed even with speech! Wait, let me reconsider."
    assert _strip_leaked_reasoning_prefix(content) == content


def test_empty_content_is_left_untouched():
    assert _strip_leaked_reasoning_prefix("") == ""


def test_whitespace_only_content_is_left_untouched():
    assert _strip_leaked_reasoning_prefix("   \n  ") == "   \n  "


def test_tool_call_only_turn_is_unaffected():
    """A tool-only turn still starts with the STATE annotation per rule 1; the
    <tool_call> block after it must not be mistaken for a leaked prefix."""
    content = (
        '[STATE: SEARCH_OPTIONS → SEARCH_OPTIONS]\n'
        '<tool_call>{"name": "search_flights", "arguments": {}}</tool_call>'
    )
    assert _strip_leaked_reasoning_prefix(content) == content


def test_arrow_ascii_variant_is_also_recognised():
    """The regex accepts both the unicode arrow and the ASCII "->" form."""
    leaked = "hmm, let me think.\n[STATE: A -> B]\nDone."
    assert _strip_leaked_reasoning_prefix(leaked) == "[STATE: A -> B]\nDone."


def test_logs_a_warning_when_stripping():
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        _strip_leaked_reasoning_prefix("garbage prefix\n[STATE: A → B]\nreply")
    assert any(log["event"] == "leaked_reasoning_prefix_stripped" for log in logs)


def test_logs_a_warning_when_annotation_missing_entirely():
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        _strip_leaked_reasoning_prefix("no annotation anywhere in this text")
    assert any(log["event"] == "turn_missing_state_annotation" for log in logs)
