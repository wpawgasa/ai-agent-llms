"""Guard the TASK_A_STAY_RULE experimental prompt flag.

The flag adds a rule stating WHEN to emit a state self-loop (FORMAT_RULES rule 1
gives only the syntax). It is opt-in because every existing Cat A checkpoint was
trained against the default prompt — if the default ever drifts, the train/eval
prompt contract silently breaks and prior measurements stop being comparable.
See docs/cat_a_state_annotation_convention_review.md §5-6.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.data.system_prompt import (
    STAY_RULE,
    build_enriched_system_prompt,
)

_MARKER = "Tool-execution turns do NOT advance"

SAMPLE = {
    "workflow_graph": {
        "initial_state": "GREETING",
        "nodes": [{"id": "GREETING", "name": "GREETING", "tools": []}],
        "edges": [],
    },
    "messages": [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "[STATE: GREETING → GREETING]\nhi"},
    ],
    "tool_schemas": [],
    "language": "en",
}


def _build(monkeypatch, value):
    monkeypatch.delenv("TASK_A_STAY_RULE", raising=False)
    if value is not None:
        monkeypatch.setenv("TASK_A_STAY_RULE", value)
    return build_enriched_system_prompt(SAMPLE, "You are an agent.", force_rebuild=True)


def test_default_prompt_omits_stay_rule(monkeypatch):
    """Unset flag must leave the prompt every checkpoint was trained on untouched."""
    assert _MARKER not in _build(monkeypatch, None)


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_only_exact_1_enables_rule(monkeypatch, value):
    """Opt-in is strict: anything other than "1" leaves the default prompt intact."""
    assert _MARKER not in _build(monkeypatch, value)


def test_flag_appends_stay_rule(monkeypatch):
    assert _MARKER in _build(monkeypatch, "1")


def test_enabled_prompt_is_default_plus_rule(monkeypatch):
    """The flag must ADD only — never reorder or drop existing prompt content.

    A strict-prefix check makes the experiment single-variable: any diff between
    the two audit runs is attributable to the appended rule alone.
    """
    off = _build(monkeypatch, None)
    on = _build(monkeypatch, "1")
    assert on.startswith(off)
    assert on[len(off) :].strip() == STAY_RULE.strip()


def test_stay_rule_states_the_policy_not_just_syntax(monkeypatch):
    """The rule's whole purpose is reinterpreting the workflow script's advance line."""
    on = _build(monkeypatch, "1")
    assert "on success: proceed to" in on
    assert "<tool_call>" in STAY_RULE
