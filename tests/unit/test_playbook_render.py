"""Tests for Task C playbook rendering (_playbook_render.py)."""

from __future__ import annotations

import json
import random
import re

import llm_workflow_agents.data._playbook_render as pr

TINY_GRAPH = {
    "states": ["START", "WORK", "TERMINAL"],
    "state_details": [
        {"name": "START", "tools": [], "entry_actions": [], "instruction": "Greet the caller."},
        {"name": "WORK", "tools": ["do_thing"], "entry_actions": [], "instruction": "Do the work."},
        {"name": "TERMINAL", "tools": [], "entry_actions": [], "instruction": "Close."},
    ],
    "transitions": [
        {"from": "START", "to": "WORK", "condition": "begin", "priority": 0},
        {"from": "WORK", "to": "TERMINAL", "condition": "done", "priority": 0},
    ],
    "initial": "START",
    "terminal": ["TERMINAL"],
}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "do_thing", "description": "do a thing",
                                      "parameters": {"type": "object", "properties": {}}}}
]
KNOBS = {"distractor_count": 0, "paraphrase_density": "low", "condition_explicitness": "explicit"}


def test_state_script_no_teacher_call(monkeypatch):
    monkeypatch.setattr(
        pr, "call_teacher_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("teacher should not be called")),
    )
    text = pr.render_playbook(TINY_GRAPH, TOOL_SCHEMAS, pr.Register.STATE_SCRIPT, "en", KNOBS,
                              teacher_model="gpt-x", rng=random.Random(1))
    for name in TINY_GRAPH["states"]:
        assert name in text


def test_render_prompt_contains_contract():
    for register in pr.TEACHER_REGISTERS:
        system, user = pr.build_render_prompts(TINY_GRAPH, TOOL_SCHEMAS, register, "en", KNOBS)
        assert system
        for name in TINY_GRAPH["states"]:
            assert name in user
        assert "do_thing" in user
        assert "```json" in user
        assert "verbatim at least once" in user
        assert "before any other state" in user
        assert "explicitly signal" in user


def test_render_prompt_language_and_knobs():
    _, user_th = pr.build_render_prompts(TINY_GRAPH, [], pr.Register.SOP_DOCUMENT, "th", KNOBS)
    assert "English/ASCII" in user_th

    _, user_lo = pr.build_render_prompts(
        TINY_GRAPH, [], pr.Register.BULLET_QUICK_REFERENCE, "en",
        dict(KNOBS, condition_explicitness="listing_order"),
    )
    assert "order they are listed" in user_lo

    knobs2 = dict(KNOBS, distractor_count=2, _distractors=["ALPHA_BOILERPLATE text", "BETA_BOILERPLATE text"])
    _, user_d = pr.build_render_prompts(TINY_GRAPH, [], pr.Register.SOP_DOCUMENT, "en", knobs2)
    assert "ALPHA_BOILERPLATE text" in user_d and "BETA_BOILERPLATE text" in user_d


def test_render_teacher_passthrough_and_corrections(monkeypatch):
    captured: dict[str, str] = {}

    def fake(model, system_prompt, user_prompt):
        captured["user"] = user_prompt
        return json.dumps({"playbook": "rendered text here"})

    monkeypatch.setattr(pr, "call_teacher_model", fake)
    out = pr.render_playbook(TINY_GRAPH, [], pr.Register.SOP_DOCUMENT, "en", KNOBS,
                             "gpt-x", random.Random(1), corrections=["missing state anchor: WORK"])
    assert out == "rendered text here"
    assert "CORRECTIONS REQUIRED" in captured["user"]
    assert "missing state anchor: WORK" in captured["user"]


def test_render_teacher_empty_raises(monkeypatch):
    monkeypatch.setattr(pr, "call_teacher_model", lambda m, s, u: json.dumps({"playbook": ""}))
    try:
        pr.render_playbook(TINY_GRAPH, [], pr.Register.PROSE_NARRATIVE, "en", KNOBS,
                           "gpt-x", random.Random(1))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_distractor_library_globally_pure():
    from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY

    all_names = {s.name for d in DOMAIN_REGISTRY.values() for s in d.states}
    all_tools = {t["function"]["name"] for d in DOMAIN_REGISTRY.values() for t in d.tools}
    forbidden = all_names | all_tools
    for _lang, paras in pr.DISTRACTOR_LIBRARY.items():
        for para in paras:
            for term in forbidden:
                assert not re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", para), (
                    f"distractor leaks {term!r}: {para!r}"
                )


def test_draw_distractors_deterministic_and_filtered():
    a = pr.draw_distractors(2, "en", random.Random(7), forbidden_terms=[])
    b = pr.draw_distractors(2, "en", random.Random(7), forbidden_terms=[])
    assert a == b and len(a) == 2
    # A whole word from the first library entry, used as a forbidden term, excludes that entry.
    entry0 = pr.DISTRACTOR_LIBRARY["en"][0]
    poison = entry0.split()[0]
    filtered = pr.draw_distractors(len(pr.DISTRACTOR_LIBRARY["en"]), "en", random.Random(7),
                                   forbidden_terms=[poison])
    assert entry0 not in filtered


# --- Task 8: build_workflow_script's tool_turn_semantics must not perturb
# Task C's STATE_SCRIPT register, which calls build_workflow_script() with the
# (default-only) `language` kwarg and nothing else. This is what makes
# render_playbook's teacher-API-free STATE_SCRIPT path byte-stable. ---

from llm_workflow_agents.data._workflow_script import build_workflow_script  # noqa: E402

_TOOL_SCHEMAS_SINGLE = [{"type": "function", "function": {"name": "t"}}]

# Captured by rendering `_tool_state_graph()` through the pre-Task-8 version of
# _workflow_script.py (git rev b12e243, the commit this plan's Task 8 brief
# names as the pre-edit baseline) with no tool_turn_semantics/retry_budget
# kwargs at all (they didn't exist yet). This is the actual historical output,
# not just "default == explicit False" (which would still pass if both were
# accidentally changed together).
_GOLDEN_PRE_EDIT_OUTPUT = (
    "### [A]  (initial state)\n"
    "Instruction: do it\n"
    "Available tools: t\n"
    "- On success: proceed to [TERMINAL]\n"
    "\n"
    "### [TERMINAL]\n"
    "This is the terminal state — end the conversation here."
)


def _tool_state_graph():
    return {
        "states": ["A", "TERMINAL"],
        "state_details": [
            {"name": "A", "tools": ["t"], "entry_actions": [], "instruction": "do it"},
            {"name": "TERMINAL", "tools": [], "entry_actions": [], "instruction": ""},
        ],
        "transitions": [{"from": "A", "to": "TERMINAL", "condition": "", "priority": 0}],
        "initial": "A", "terminal": ["TERMINAL"],
    }


def test_state_script_unchanged_by_tool_turn_semantics_default():
    graph = _tool_state_graph()
    default_output = build_workflow_script(graph, tool_schemas=_TOOL_SCHEMAS_SINGLE)
    explicit_off = build_workflow_script(graph, tool_schemas=_TOOL_SCHEMAS_SINGLE,
                                          tool_turn_semantics=False)
    assert default_output == explicit_off
    # The real guard: both the default path and the explicit-False path must
    # equal the pre-edit golden, not just equal each other.
    assert default_output == _GOLDEN_PRE_EDIT_OUTPUT
    assert explicit_off == _GOLDEN_PRE_EDIT_OUTPUT


def test_tool_turn_semantics_rewrites_success_line_and_adds_stay_note():
    graph = _tool_state_graph()
    output = build_workflow_script(graph, tool_schemas=_TOOL_SCHEMAS_SINGLE,
                                    tool_turn_semantics=True, retry_budget=2)
    assert "on a LATER turn" in output
    assert "stays in" in output.lower() or "stay in" in output.lower()
    # N-total-attempts wording (fix round 1), not "retry at most N times"
    # (N retries after the first, i.e. N+1 total) — see
    # test_retry_note_agrees_with_format_rules_attempt_count.
    assert "2 attempts total" in output.lower()


def test_tool_turn_semantics_no_effect_on_text_only_state():
    """A state with no tools renders identically regardless of the flag."""
    graph = {
        "states": ["A", "TERMINAL"],
        "state_details": [
            {"name": "A", "tools": [], "entry_actions": [], "instruction": "say hi"},
            {"name": "TERMINAL", "tools": [], "entry_actions": [], "instruction": ""},
        ],
        "transitions": [{"from": "A", "to": "TERMINAL", "condition": "", "priority": 0}],
        "initial": "A", "terminal": ["TERMINAL"],
    }
    off = build_workflow_script(graph, tool_turn_semantics=False)
    on = build_workflow_script(graph, tool_turn_semantics=True, retry_budget=3)
    assert off == on
    assert "<tool_call>" not in on
    assert "LATER" not in on


def test_retry_note_wording_at_budgets_1_2_3():
    graph = _tool_state_graph()

    out_1 = build_workflow_script(graph, tool_schemas=_TOOL_SCHEMAS_SINGLE,
                                   tool_turn_semantics=True, retry_budget=1)
    assert "retry at most 1" not in out_1.lower()  # ungrammatical/wrong at budget 1
    assert "do not retry" in out_1.lower()
    assert "only one attempt" in out_1.lower()

    # Fix round 1: wording must say N attempts TOTAL (counting the first), not
    # "retry N times" (which would mean N retries *after* the first, i.e. N+1
    # total) — see the comment above _RETRY_NOTE_WITH_RETRIES in
    # _workflow_script.py and system_prompt.py's _RETRY_RULE_WITH_RETRIES.
    out_2 = build_workflow_script(graph, tool_schemas=_TOOL_SCHEMAS_SINGLE,
                                   tool_turn_semantics=True, retry_budget=2)
    assert "2 attempts total" in out_2.lower()
    assert "retry at most 2 times" not in out_2.lower()

    out_3 = build_workflow_script(graph, tool_schemas=_TOOL_SCHEMAS_SINGLE,
                                   tool_turn_semantics=True, retry_budget=3)
    assert "3 attempts total" in out_3.lower()
    assert "retry at most 3 times" not in out_3.lower()


def test_retry_note_agrees_with_format_rules_attempt_count():
    """Regression for fix round 1: the workflow script's tool-error retry note
    (_workflow_script.py::_retry_note) and the system prompt's FORMAT_RULES
    retry rule (system_prompt.py::_retry_rule) both get concatenated into the
    same system prompt by build_enriched_system_prompt, so they must state the
    same attempt-count semantics for a given retry_budget — otherwise the
    corpus bakes in two contradictory policies a few hundred characters apart.

    This test crosses _workflow_script.py and system_prompt.py; it lives here
    (test_playbook_render.py) rather than a system_prompt-specific test file
    because the other retry-note tests it guards against regressing already
    live in this file (see test_retry_note_wording_at_budgets_1_2_3 above).
    """
    from llm_workflow_agents.data.system_prompt import _retry_rule

    graph = _tool_state_graph()
    for budget in (1, 2, 3):
        script_note = build_workflow_script(
            graph, tool_schemas=_TOOL_SCHEMAS_SINGLE,
            tool_turn_semantics=True, retry_budget=budget,
        )
        prompt_rule = _retry_rule(2, budget)

        if budget <= 1:
            # Neither side should mention a numeral attempt count at budget 1
            # (there is no retry to count) — both use "no retry" framing.
            assert "do not retry" in script_note.lower()
            assert "do not retry" in prompt_rule.lower()
        else:
            # Both must cite the SAME numeral as the total attempt count.
            assert f"{budget} attempts" in script_note.lower()
            assert f"{budget} attempts" in prompt_rule.lower()


def test_tool_turn_semantics_thai_placeholders_all_substituted():
    graph = _tool_state_graph()
    output = build_workflow_script(graph, tool_schemas=_TOOL_SCHEMAS_SINGLE, language="th",
                                    tool_turn_semantics=True, retry_budget=2)
    assert "{name}" not in output
    assert "{to}" not in output
    assert "{n}" not in output
    assert "{fallback}" not in output
    assert "[A]" in output
    assert "[TERMINAL]" in output
    assert "2" in output
