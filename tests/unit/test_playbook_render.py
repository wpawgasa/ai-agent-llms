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
