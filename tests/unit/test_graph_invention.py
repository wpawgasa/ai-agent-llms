"""Tests for Task C gold-graph invention and validation (_graph_invention.py)."""

from __future__ import annotations

import json
import random

import pytest

import llm_workflow_agents.data._graph_invention as gi
from llm_workflow_agents.data._graph_invention import (
    DOMAIN_BRIEFS,
    InventedGraph,
    invent_novel_graphs,
    validate_gold_graph,
)

# Spec §Worked Example gold JSON (id-keyed eval shape), verbatim.
WORKED_EXAMPLE = {
    "nodes": [
        {"id": "S1", "name": "VERIFY_POLICYHOLDER", "tools": ["verify_policy"], "entry_actions": []},
        {"id": "S2", "name": "CLAIM_INTAKE", "tools": ["file_claim"], "entry_actions": []},
        {"id": "S3", "name": "ASSESS_COVERAGE", "tools": [], "entry_actions": []},
        {"id": "S4", "name": "RESOLVE", "tools": [], "entry_actions": []},
        {"id": "S5", "name": "REQUEST_DOCUMENTATION", "tools": [], "entry_actions": []},
        {"id": "S6", "name": "TERMINAL", "tools": [], "entry_actions": []},
    ],
    "edges": [
        {"from_state": "S1", "to_state": "S2", "condition": "policyholder verified", "priority": 0},
        {"from_state": "S2", "to_state": "S3", "condition": "claim filed", "priority": 0},
        {"from_state": "S3", "to_state": "S4", "condition": "coverage assessed", "priority": 0},
        {"from_state": "S3", "to_state": "S5", "condition": "documentation missing", "priority": 1},
        {"from_state": "S5", "to_state": "S3", "condition": "documents received", "priority": 0},
        {"from_state": "S4", "to_state": "S6", "condition": "case closed", "priority": 0},
    ],
    "initial_state": "S1",
    "terminal_states": ["S6"],
}


def test_validate_gold_graph_accepts_worked_example():
    assert validate_gold_graph(WORKED_EXAMPLE) == []


def test_validate_gold_graph_itemizes_violations():
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    bad["edges"].append({"from_state": "S4", "to_state": "S4", "condition": "loop", "priority": 5})
    bad["nodes"][1]["name"] = "verify_policyholder"  # not SCREAMING_SNAKE
    violations = validate_gold_graph(bad)
    assert any("self-loop" in v for v in violations)
    assert any("SCREAMING_SNAKE" in v for v in violations)


def test_validate_gold_graph_rejects_sink():
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    bad["edges"] = [e for e in bad["edges"] if e["from_state"] != "S5"]  # S5 = non-terminal sink
    violations = validate_gold_graph(bad)
    assert any("no outgoing" in v for v in violations)


def test_validate_gold_graph_rejects_duplicate_priority():
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    # Give S3's two outgoing edges the same priority.
    for e in bad["edges"]:
        if e["from_state"] == "S3":
            e["priority"] = 0
    violations = validate_gold_graph(bad)
    assert any("priorit" in v for v in violations)


def test_validate_gold_graph_rejects_schema_violation():
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    del bad["initial_state"]
    violations = validate_gold_graph(bad)
    assert violations and any("schema" in v for v in violations)


def test_invent_repairs_then_accepts(monkeypatch):
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    bad["edges"].append({"from_state": "S2", "to_state": "S2", "condition": "x", "priority": 9})
    responses = iter([json.dumps(bad), json.dumps(WORKED_EXAMPLE)])
    prompts: list[str] = []

    def fake_teacher(model, system_prompt, user_prompt):
        prompts.append(user_prompt)
        return next(responses)

    monkeypatch.setattr(gi, "call_teacher_model", fake_teacher)
    out = invent_novel_graphs(DOMAIN_BRIEFS[:1], count=1, teacher_model="gpt-x", rng=random.Random(1))
    assert len(out) == 1 and isinstance(out[0], InventedGraph)
    assert len(prompts) == 2
    assert "CORRECTIONS REQUIRED" in prompts[1] and "self-loop" in prompts[1]
    assert out[0].graph["initial"] == "VERIFY_POLICYHOLDER"  # name-keyed, ids discarded
    assert "S1" not in json.dumps(out[0].graph)


def test_invent_drops_after_exhausted_repairs(monkeypatch):
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    del bad["initial_state"]
    calls: list[int] = []

    def fake_teacher(model, system_prompt, user_prompt):
        calls.append(1)
        return json.dumps(bad)

    monkeypatch.setattr(gi, "call_teacher_model", fake_teacher)
    out = invent_novel_graphs(DOMAIN_BRIEFS[:1], count=1, teacher_model="gpt-x", rng=random.Random(1))
    assert out == [] and len(calls) == 3  # initial + 2 repairs


def test_invent_survives_teacher_exception(monkeypatch):
    def boom(model, system_prompt, user_prompt):
        raise RuntimeError("network")

    monkeypatch.setattr(gi, "call_teacher_model", boom)
    out = invent_novel_graphs(DOMAIN_BRIEFS[:1], count=1, teacher_model="gpt-x", rng=random.Random(1))
    assert out == []


def test_briefs_wellformed():
    slugs = [b.slug for b in DOMAIN_BRIEFS]
    assert len(DOMAIN_BRIEFS) >= 25
    assert len(set(slugs)) == len(slugs)
    for b in DOMAIN_BRIEFS:
        assert b.slug.islower() and " " not in b.slug
        assert b.title and b.description
    from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY

    assert not set(slugs) & set(DOMAIN_REGISTRY)  # disjoint from registry keys


def test_normalize_preserves_topology(monkeypatch):
    responses = iter([json.dumps(WORKED_EXAMPLE)])
    monkeypatch.setattr(gi, "call_teacher_model", lambda m, s, u: next(responses))
    out = invent_novel_graphs(DOMAIN_BRIEFS[:1], count=1, teacher_model="gpt-x", rng=random.Random(1))
    g = out[0].graph
    assert set(g["states"]) == {
        "VERIFY_POLICYHOLDER", "CLAIM_INTAKE", "ASSESS_COVERAGE",
        "RESOLVE", "REQUEST_DOCUMENTATION", "TERMINAL",
    }
    assert g["terminal"] == ["TERMINAL"]
    trans = {(t["from"], t["to"]): t["priority"] for t in g["transitions"]}
    assert trans[("ASSESS_COVERAGE", "REQUEST_DOCUMENTATION")] == 1
    assert ("REQUEST_DOCUMENTATION", "ASSESS_COVERAGE") in trans
