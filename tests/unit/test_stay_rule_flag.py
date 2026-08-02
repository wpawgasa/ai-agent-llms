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
import warnings

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
    assert "hand off" in rules
    # The vague v1 wording must be gone.
    assert "attempt recovery before escalating" not in rules
    # Template placeholders must have been substituted, not shipped literally.
    assert "{n}" not in rules
    assert "{retry_budget}" not in rules


# --------------------------------------------------------------------------
# Retry-budget wording — two branches, both must be coherent and grammatical
#
# The budget counts TOTAL attempts including the first, so budget 1 means "no
# retry at all". The original wording said "you may retry the SAME call ...
# retry at most 1 time(s) total (including the first attempt)" — which both
# contradicts itself at budget 1 and renders "1 time(s)". L1/L2 hold budget 1
# permanently, so that text would be frozen into every L1/L2 corpus row.
# --------------------------------------------------------------------------


def _retry_rule_text(text):
    """Extract the tool-error rule block (rule 8 in v2) from rendered rules."""
    match = re.search(r"^8\. .*?(?=\n\n9\. )", text, re.S | re.M)
    assert match, "could not locate the tool-error rule (rule 8) in the rendered text"
    return match.group(0)


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_retry_rule_never_renders_the_plural_placeholder(budget):
    """No "1 time(s)" — the wording must be grammatical at every spec budget
    (L1-L2: 1, L3-L4: 2, L5: 3)."""
    text = system_prompt._format_rules(retry_budget=budget)
    assert "time(s)" not in text
    assert "(s)" not in _retry_rule_text(text)
    assert "{n}" not in text and "{retry_budget}" not in text


def test_retry_rule_at_budget_1_forbids_retrying():
    """Budget 1 means one attempt: it must NOT invite a retry it then forbids."""
    rule = _retry_rule_text(system_prompt._format_rules(retry_budget=1))
    assert "do NOT retry it" in rule
    assert "one attempt per" in rule
    # The permissive wording belongs only to the budget>1 branch.
    assert "you may retry" not in rule
    # A bare "1" budget must not be interpolated as a count anywhere.
    assert "1 attempt" not in rule


@pytest.mark.parametrize("budget", [2, 3])
def test_retry_rule_above_budget_1_permits_retries_up_to_the_cap(budget):
    rule = _retry_rule_text(system_prompt._format_rules(retry_budget=budget))
    assert "you may retry the SAME call" in rule
    assert f"{budget} attempts at that call in total" in rule
    assert "counting the first" in rule
    assert "do NOT retry it" not in rule


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_retry_rule_states_the_same_policy_at_every_budget(budget):
    """Both branches must carry the identical underlying policy."""
    rule = _retry_rule_text(system_prompt._format_rules(retry_budget=budget))
    assert "[STATE: X → X]" in rule            # annotate the self-loop
    assert "never advance" in rule.lower()      # never advance while retrying
    assert "error path if the script names one" in rule   # scripted fallback
    assert "cannot be completed right now" in rule        # unscripted fallback
    assert "hand off" in rule
    assert "Do not invent a transition" in rule


def test_retry_budget_below_1_falls_back_to_the_no_retry_wording():
    """Degenerate budgets must not emit "0 attempts"/"-1 attempts" prose."""
    for budget in (0, -1):
        rule = _retry_rule_text(system_prompt._format_rules(retry_budget=budget))
        assert "do NOT retry it" in rule
        assert "attempts at that call" not in rule


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert _MARKER in _reload_with(monkeypatch, value).FORMAT_RULES


@pytest.mark.parametrize("value", ["true", "yes", "", "00", "0 ", "false", "off"])
def test_unrecognised_flag_value_warns(monkeypatch, value):
    """`!= "0"` fails OPEN toward v2: "false"/"off"/"0 " silently render the v2
    prompt when v1 was probably meant, and that is the direction that breaks
    checkpoint comparability. Parsing stays exact (v1 bytes must not depend on
    fuzzy matching) but a typo must surface rather than quietly produce the
    wrong corpus."""
    with pytest.warns(RuntimeWarning, match="TASK_A_STAY_RULE"):
        reloaded = _reload_with(monkeypatch, value)
    # Warning only — the resolved behaviour is unchanged (still enabled/v2).
    assert _MARKER in reloaded.FORMAT_RULES


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_recognised_flag_values_do_not_warn(monkeypatch, value):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _reload_with(monkeypatch, value)


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
