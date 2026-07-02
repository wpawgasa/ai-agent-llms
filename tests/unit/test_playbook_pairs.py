"""Tests for the Task C playbook-pair orchestrator (generate_playbook_pairs.py)."""

from __future__ import annotations

import random

import llm_workflow_agents.data.generate_playbook_pairs as gpp
from llm_workflow_agents.data.generate_playbook_pairs import (
    TASK_C_LEVEL_WEIGHTS,
    assign_splits,
    build_graph_pool,
    plan_renderings,
)

_TINY_GRAPH = {
    "states": ["START", "WORK", "TERMINAL"],
    "state_details": [
        {"name": "START", "tools": [], "entry_actions": [], "instruction": ""},
        {"name": "WORK", "tools": ["do_thing"], "entry_actions": [], "instruction": ""},
        {"name": "TERMINAL", "tools": [], "entry_actions": [], "instruction": ""},
    ],
    "transitions": [
        {"from": "START", "to": "WORK", "condition": "begin", "priority": 0},
        {"from": "WORK", "to": "TERMINAL", "condition": "done", "priority": 0},
    ],
    "initial": "START",
    "terminal": ["TERMINAL"],
}


def _fake_invention(monkeypatch, n):
    """Patch invent_novel_graphs to return n InventedGraph copies of a tiny valid graph."""
    from llm_workflow_agents.data._graph_invention import DOMAIN_BRIEFS, InventedGraph

    fakes = [InventedGraph(DOMAIN_BRIEFS[i % len(DOMAIN_BRIEFS)].slug, _TINY_GRAPH) for i in range(n)]

    def fake(briefs, count, teacher_model, rng, max_repair_retries=2):
        return fakes[:count]

    monkeypatch.setattr(gpp, "invent_novel_graphs", fake)


def test_build_graph_pool_ratio_levels_determinism(monkeypatch):
    _fake_invention(monkeypatch, 30)
    pool1 = build_graph_pool(100, 0.30, "gpt-x", random.Random(142))
    _fake_invention(monkeypatch, 30)
    pool2 = build_graph_pool(100, 0.30, "gpt-x", random.Random(142))
    key = lambda p: [(e.graph_id, e.source, e.domain, e.complexity_level) for e in p]
    assert key(pool1) == key(pool2)
    assert len(pool1) == 100
    assert pool1[0].graph_id == "G0001" and pool1[99].graph_id == "G0100"
    n_inv = sum(1 for e in pool1 if e.source == "invented")
    assert 25 <= n_inv <= 35
    l3 = sum(1 for e in pool1 if e.complexity_level == "L3")
    assert 15 <= l3 <= 45  # 35% of ~70 registry graphs, wide tolerance


def test_registry_entries_have_tool_schemas(monkeypatch):
    _fake_invention(monkeypatch, 0)
    pool = build_graph_pool(20, 0.0, "gpt-x", random.Random(7))
    for e in pool:
        names_in_states = {t for sd in e.graph["state_details"] for t in sd["tools"]}
        schema_names = {t["function"]["name"] for t in e.tool_schemas}
        assert names_in_states <= schema_names


def test_assign_splits_heldout_axes(monkeypatch):
    _fake_invention(monkeypatch, 30)
    pool = build_graph_pool(120, 0.30, "gpt-x", random.Random(142))
    splits = assign_splits(pool, seed=142)
    assert splits == assign_splits(pool, seed=142)  # deterministic
    for e in pool:
        if e.domain in ("utilities", "surveys"):
            assert splits[e.graph_id] == "test"
    counts = {s: list(splits.values()).count(s) for s in ("train", "validation", "test")}
    assert counts["train"] > counts["validation"] > 0 and counts["test"] > 0
    assert sum(counts.values()) == 120


def test_plan_renderings_register_constraints(monkeypatch):
    _fake_invention(monkeypatch, 30)
    pool = build_graph_pool(120, 0.30, "gpt-x", random.Random(142))
    splits = assign_splits(pool, seed=142)
    plans = plan_renderings(pool, splits, {"en": 0.5, "th": 0.3, "code_switch": 0.2}, random.Random(142))
    by_graph: dict[str, list] = {}
    for p in plans:
        by_graph.setdefault(p.graph_id, []).append(p)
    assert set(by_graph) == {e.graph_id for e in pool}
    for gid, ps in by_graph.items():
        regs = [p.register for p in ps]
        assert 4 <= len(regs) <= 6 and len(set(regs)) == len(regs)
        assert "state_script" in regs
        assert {"sop_document", "prose_narrative"} & set(regs)
        if splits[gid] == "train":
            assert "manager_transcript" not in regs
        assert [p.pair_id for p in ps] == [f"{gid}_r{i + 1}" for i in range(len(ps))]


def test_plan_renderings_knob_language_distribution(monkeypatch):
    _fake_invention(monkeypatch, 30)
    pool = build_graph_pool(120, 0.30, "gpt-x", random.Random(142))
    splits = assign_splits(pool, seed=142)
    plans = plan_renderings(pool, splits, {"en": 0.5, "th": 0.3, "code_switch": 0.2}, random.Random(142))
    langs = [p.language for p in plans]
    assert 0.35 < langs.count("en") / len(langs) < 0.65
    assert 0.10 < langs.count("th") / len(langs) < 0.45
    with_distractors = sum(1 for p in plans if p.distractor_count > 0)
    assert 0.15 < with_distractors / len(plans) < 0.45
    assert {p.paraphrase_density for p in plans} <= {"low", "medium", "high"}
    assert {p.condition_explicitness for p in plans} <= {"explicit", "narrative_order", "listing_order"}


def test_level_weights_sum_to_one():
    assert abs(sum(TASK_C_LEVEL_WEIGHTS.values()) - 1.0) < 1e-9
