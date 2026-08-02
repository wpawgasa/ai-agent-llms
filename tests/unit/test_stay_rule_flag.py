"""Guard the TASK_A_STAY_RULE prompt-revision flag.

Since 2026-07-31 the stay rule ships by DEFAULT as rule 2 of FORMAT_RULES
(task-a-sft-v2). ``TASK_A_STAY_RULE=0`` opts back out to the frozen v1 prompt.

The flag selects the whole prompt revision, not just the stay rule: v1 also
carries the wrong rule-2 worked example (an advancing transition on a
tool-calling turn — the defect this revision fixes) and the vague tool-error
one-liner. v1's bytes are what every existing Cat A checkpoint (ckpt-500,
ckpt-1770) was trained against, so the opt-out is asserted byte-for-byte
against a golden captured before the change.

See docs/cat_a_state_annotation_convention_review.md §5-6.
"""

from __future__ import annotations

import importlib
import re

import pytest

from llm_workflow_agents.data import system_prompt

from tests.unit._v1_format_rules_golden import V1_FORMAT_RULES

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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from an unset flag and restores module state after."""
    monkeypatch.delenv("TASK_A_STAY_RULE", raising=False)
    importlib.reload(system_prompt)
    yield
    monkeypatch.delenv("TASK_A_STAY_RULE", raising=False)
    importlib.reload(system_prompt)


def _reload_with(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TASK_A_STAY_RULE", raising=False)
    else:
        monkeypatch.setenv("TASK_A_STAY_RULE", value)
    return importlib.reload(system_prompt)


# --------------------------------------------------------------------------
# Default (v2) content
# --------------------------------------------------------------------------


def test_default_prompt_contains_stay_rule():
    assert _MARKER in system_prompt.FORMAT_RULES
    assert "do NOT advance" in system_prompt.FORMAT_RULES


def test_rule_2_worked_example_is_a_self_loop():
    """The historical bug: rule 2's example showed an ADVANCING transition on a
    tool-call turn, actively teaching the defect. It must now show a self-loop,
    matching the rule 1 syntax example's convention."""
    assert "[STATE: VERIFY_PATIENT → TERMINAL]" not in system_prompt.FORMAT_RULES
    assert "[STATE: VERIFY_PATIENT → VERIFY_PATIENT]" in system_prompt.FORMAT_RULES


def test_retry_rule_states_a_budget_and_fallback():
    rules = system_prompt.FORMAT_RULES
    assert "retry" in rules.lower()
    assert "at most 1 time(s) total" in rules
    assert "hand off" in rules
    # The vague v1 wording must be gone.
    assert "attempt recovery before escalating" not in rules
    # Template placeholders must have been substituted, not shipped literally.
    assert "{n}" not in rules
    assert "{retry_budget}" not in rules


def test_retry_budget_is_parameterised():
    """Task 8 wires a caller-supplied budget through; the renderer must honour it."""
    assert "at most 3 time(s) total" in system_prompt._format_rules(retry_budget=3)


def test_stay_rule_states_the_policy_not_just_syntax():
    """The rule's whole purpose is reinterpreting the workflow script's advance line."""
    assert "on success: proceed to" in system_prompt.STAY_RULE
    assert "<tool_call>" in system_prompt.STAY_RULE
    assert system_prompt.STAY_RULE in system_prompt.FORMAT_RULES


# --------------------------------------------------------------------------
# Rule numbering — a gap or duplicate here ships into every corpus row
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stay_rule", "expected_count"), [(True, 10), (False, 9)]
)
def test_rules_are_numbered_consecutively(stay_rule, expected_count):
    text = system_prompt._format_rules(stay_rule=stay_rule)
    numbers = [int(m.group(1)) for m in re.finditer(r"^(\d+)\. ", text, re.M)]
    assert numbers == list(range(1, expected_count + 1))


# --------------------------------------------------------------------------
# Opt-out semantics
# --------------------------------------------------------------------------


def test_only_exact_0_disables_the_rule(monkeypatch):
    reloaded = _reload_with(monkeypatch, "0")
    assert _MARKER not in reloaded.FORMAT_RULES
    assert "do NOT advance" not in reloaded.FORMAT_RULES


@pytest.mark.parametrize("value", [None, "1", "true", "yes", "", "00", "0 "])
def test_anything_other_than_0_leaves_the_rule_enabled(monkeypatch, value):
    """Opt-out is strict: only the exact string "0" reverts to the v1 prompt."""
    assert _MARKER in _reload_with(monkeypatch, value).FORMAT_RULES


def test_disabled_prompt_is_byte_identical_to_v1_baseline(monkeypatch):
    """The load-bearing guarantee.

    V1_FORMAT_RULES is the exact text every existing checkpoint (ckpt-500,
    ckpt-1770) was trained against, captured from the pre-change module. Do NOT
    edit that golden when changing the default prompt — it exists specifically
    to prove the opt-out path is unchanged and those checkpoints stay
    comparable.
    """
    reloaded = _reload_with(monkeypatch, "0")
    assert reloaded.FORMAT_RULES == V1_FORMAT_RULES


def test_v1_path_ignores_retry_budget(monkeypatch):
    """v1 bytes are frozen; a caller-supplied budget must not leak into them."""
    assert system_prompt._format_rules(retry_budget=5, stay_rule=False) == V1_FORMAT_RULES


# --------------------------------------------------------------------------
# Rendered end-to-end prompt
# --------------------------------------------------------------------------


def _build():
    return system_prompt.build_enriched_system_prompt(
        SAMPLE, "You are an agent.", force_rebuild=True
    )


def test_enriched_prompt_carries_stay_rule_by_default():
    assert _MARKER in _build()


def test_stay_rule_is_never_emitted_twice(monkeypatch):
    """Regression: STAY_RULE used to be appended separately when the flag was
    "1". Now that it is inline as rule 2, appending it too would duplicate it
    with inconsistent numbering."""
    for value in (None, "1"):
        _reload_with(monkeypatch, value)
        assert _build().count(_MARKER) == 1


def test_enriched_prompt_omits_stay_rule_when_disabled(monkeypatch):
    _reload_with(monkeypatch, "0")
    assert _MARKER not in _build()
