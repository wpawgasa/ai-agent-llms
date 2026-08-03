"""Per-sample retry budget in the rebuilt Task A system prompt.

Task 9 made ``retry_budget`` a per-level property (L1-L4: 2, L5: 3 — L1/L2
raised from 1 in the final review wave, superseding decision D4) and taught the
generator to emit conversations that really do retry that many times. ``system_prompt.py`` kept rendering the prompt at a hardcoded budget of 1
in BOTH of its retry-mentioning passages, and that prompt is rebuilt at training
and eval load time (``training/sft.py``, ``training/grpo.py``,
``eval/agent_benchmark.py``, ``webui/samples.py``). The result was a direct
train/serve contradiction: an L5 row whose transcript shows three attempts,
shipped under a rule reading "If a tool returns an error, do NOT retry it".

These tests pin the fix, and — just as importantly — pin the two invariants it
could easily have broken:

  1. both halves of the prompt state the SAME budget for a given sample;
  2. the frozen v1 prompt (``TASK_A_STAY_RULE=0``) is still byte-identical,
     because the v1 tool-error rule states no budget at all.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import re

import pytest

from llm_workflow_agents.data import system_prompt as sp
from tests.unit._v1_format_rules_golden import V1_FORMAT_RULES

_V1_SHA256 = "dac5871dab6d9a350255ede51078d1ae8008098ad468d1f635c781d6ce852e10"


def _sample(level: str | None) -> dict:
    """A minimal Task A sample with one tool-bearing state.

    ``state_details`` is what makes ``build_enriched_system_prompt`` regenerate
    the workflow script (rather than reusing the frozen ``workflow_script``
    field), which is the half of the prompt that carries the per-state retry
    note.
    """
    s = {
        "workflow_graph": {
            "states": ["A", "TERMINAL"],
            "state_details": [
                {"name": "A", "tools": ["t"], "entry_actions": [], "instruction": "do it"},
                {"name": "TERMINAL", "tools": [], "entry_actions": [], "instruction": ""},
            ],
            "transitions": [{"from": "A", "to": "TERMINAL", "condition": "", "priority": 0}],
            "initial": "A",
            "terminal": ["TERMINAL"],
        },
        "tool_schemas": [{"type": "function", "function": {"name": "t"}}],
        "messages": [],
        "language": "en",
    }
    if level is not None:
        s["complexity_level"] = level
    return s


def _format_rules_retry_rule(prompt: str) -> str:
    """Extract the numbered tool-error rule out of a rendered prompt."""
    m = re.search(r"\n(\d+)\. If a tool returns an error.*?(?=\n\d+\. )", prompt, re.S)
    assert m, "prompt has no FORMAT_RULES tool-error rule"
    return " ".join(m.group(0).split())


def _script_retry_notes(prompt: str) -> list[str]:
    notes = [ln.strip() for ln in prompt.splitlines() if ln.strip().startswith("- On tool error:")]
    assert notes, "prompt has no workflow-script retry note"
    return notes


# --- budget resolution -------------------------------------------------------


@pytest.mark.parametrize(
    "level,expected",
    [("L1", 2), ("L2", 2), ("L3", 2), ("L4", 2), ("L5", 3)],
)
def test_retry_budget_follows_complexity_level(level, expected):
    """The resolved budget is the level's COMPLEXITY_SPECS value, not a constant."""
    assert sp.retry_budget_for_sample({"complexity_level": level}) == expected


@pytest.mark.parametrize(
    "sample",
    [
        {},                                  # no complexity_level at all
        {"complexity_level": None},          # present but null
        {"complexity_level": "L9"},          # unknown level
        {"complexity_level": "nonsense"},
        {"complexity_level": 3},             # wrong type (would be unhashable-adjacent)
        {"complexity_level": ["L5"]},        # unhashable — must not raise
        None,                                # not a dict at all
    ],
)
def test_unknown_complexity_level_degrades_to_one_attempt(sample):
    """A metadata gap must not crash a prompt rebuild mid-training run."""
    assert sp.retry_budget_for_sample(sample) == sp.DEFAULT_RETRY_BUDGET == 1


def test_level_is_normalised():
    assert sp.retry_budget_for_sample({"complexity_level": " l5 "}) == 3


# --- the actual train/serve contradiction ------------------------------------


def test_l5_prompt_states_three_attempts_and_l1_states_two():
    """Every shipped level now permits a retry; only the count differs.

    L1/L2 moved off budget 1 in the final review wave because the corpus at
    those levels already demonstrates same-tool retries (200/1,251 L1,
    783/1,305 L2) — a "do NOT retry it" rule there would contradict its own
    training data.
    """
    l1 = sp.build_enriched_system_prompt(_sample("L1"), "You are an agent.", force_rebuild=True)
    l5 = sp.build_enriched_system_prompt(_sample("L5"), "You are an agent.", force_rebuild=True)

    l1_rule = _format_rules_retry_rule(l1)
    assert "You get 2 attempts at that call in total, counting the first" in l1_rule
    assert "do NOT retry it" not in l1_rule
    assert "3 attempts" not in l1_rule

    l5_rule = _format_rules_retry_rule(l5)
    assert "You get 3 attempts at that call in total, counting the first" in l5_rule
    assert "do NOT retry it" not in l5_rule


def test_unlabelled_sample_still_renders_the_no_retry_wording():
    """The budget-1 wording is now reachable ONLY via the degradation default.

    No complexity level renders it any more, so this is the last guard that the
    no-retry branch stays wired up and grammatical.
    """
    rule = _format_rules_retry_rule(
        sp.build_enriched_system_prompt(_sample(None), "You are an agent.", force_rebuild=True)
    )
    assert "do NOT retry it" in rule
    assert "one attempt per call" in rule


@pytest.mark.parametrize("level,budget", [("L1", 2), ("L2", 2), ("L3", 2), ("L5", 3)])
def test_both_prompt_halves_state_the_same_budget(level, budget):
    """The workflow-script note and the FORMAT_RULES rule must not disagree.

    They land in the same system prompt a few hundred characters apart, so a
    mismatch would replace one contradiction with another.
    """
    prompt = sp.build_enriched_system_prompt(
        _sample(level), "You are an agent.", force_rebuild=True
    )
    rule = _format_rules_retry_rule(prompt)
    notes = _script_retry_notes(prompt)

    if budget == 1:
        assert "do NOT retry it" in rule
        assert all("do NOT retry" in n for n in notes)
        # Neither half may state a numeral attempt count at budget 1.
        assert not re.search(r"\b\d+ attempts\b", rule)
        assert all(not re.search(r"\b\d+ attempts\b", n) for n in notes)
    else:
        assert f"You get {budget} attempts at that call in total, counting the first" in rule
        assert all(f"up to {budget} attempts total, counting the first" in n for n in notes)


def test_sample_without_complexity_level_renders_the_legacy_prompt():
    """Degradation is to the exact pre-fix text, not merely to "something".

    The pre-fix text is the budget-1 rendering. Since L1 was raised to budget 2
    it is no longer the reference for that; ``_format_rules(retry_budget=1)`` —
    i.e. the frozen ``FORMAT_RULES`` constant — is.
    """
    unlabelled = sp.build_enriched_system_prompt(
        _sample(None), "You are an agent.", force_rebuild=True
    )
    assert sp.FORMAT_RULES in unlabelled
    assert "do NOT retry it" in unlabelled


# --- invariants the fix must not break ---------------------------------------


def test_module_level_format_rules_constant_still_means_budget_one():
    """Importers of FORMAT_RULES keep the constant they had."""
    assert sp._format_rules(retry_budget=1) == sp.FORMAT_RULES
    assert "do NOT retry it" in sp.FORMAT_RULES


def test_v1_bytes_identical_at_every_budget(monkeypatch):
    """TASK_A_STAY_RULE=0 must stay byte-identical regardless of the sample.

    The v1 tool-error rule states no budget, so per-sample budgets are ignored
    on that path. Anything that changes these bytes breaks comparability with
    every checkpoint trained against v1 (ckpt-500, ckpt-1770).
    """
    monkeypatch.setenv("TASK_A_STAY_RULE", "0")
    reloaded = importlib.reload(sp)
    try:
        assert hashlib.sha256(V1_FORMAT_RULES.encode()).hexdigest() == _V1_SHA256
        assert reloaded.FORMAT_RULES == V1_FORMAT_RULES
        for level in ("L1", "L2", "L3", "L4", "L5", None):
            rendered = reloaded.format_rules_for_sample(_sample(level))
            assert rendered == V1_FORMAT_RULES, f"v1 bytes drifted for {level}"
    finally:
        monkeypatch.delenv("TASK_A_STAY_RULE", raising=False)
        importlib.reload(sp)


def test_prompt_rebuild_does_not_mutate_the_sample():
    s = _sample("L5")
    before = copy.deepcopy(s)
    sp.build_enriched_system_prompt(s, "You are an agent.", force_rebuild=True)
    assert s == before
