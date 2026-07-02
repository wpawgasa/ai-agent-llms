"""Tests for the Task C playbook-pair orchestrator (generate_playbook_pairs.py)."""

from __future__ import annotations

import json
import random

import llm_workflow_agents.data._playbook_render as pr
import llm_workflow_agents.data._playbook_verify as pv
import llm_workflow_agents.data.generate_playbook_pairs as gpp
from tests.unit._task_c_helpers import CompliantTeacher, _patch_all_teachers
from llm_workflow_agents.data.generate_playbook_pairs import (
    TASK_C_LEVEL_WEIGHTS,
    GraphPoolEntry,
    RenderingPlan,
    _drop_undersized_graphs,
    _leg_should_halt,
    assign_splits,
    build_graph_pool,
    generate_playbook_dataset,
    merge_benchmark_runs,
    plan_renderings,
    produce_rendering,
)

_VERIFY_TEACHERS = {"gpt": "gemini-3-flash", "gemini": "gpt-5.4-nano-2026-03-17"}

_TINY_ENTRY = GraphPoolEntry(
    graph_id="G0001",
    source="registry",
    domain="testdomain",
    complexity_level="L1",
    graph={
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
    },
    tool_schemas=[{"type": "function", "function": {"name": "do_thing", "description": "do",
                                                    "parameters": {"type": "object", "properties": {}}}}],
)


def _plan(register="prose_narrative", distractor_count=0):
    return RenderingPlan(
        pair_id="G0001_r1", graph_id="G0001", register=register, language="en",
        distractor_count=distractor_count, paraphrase_density="low", condition_explicitness="explicit",
    )


def _produce_one(monkeypatch, teacher, register="prose_narrative", distractor_count=0,
                 do_back_extraction=False, verifier_response=None):
    monkeypatch.setattr(pr, "call_teacher_model", teacher)
    if do_back_extraction and verifier_response is not None:
        monkeypatch.setattr(pv, "call_teacher_model", lambda m, s, u: verifier_response)
    return produce_rendering(
        _TINY_ENTRY, _plan(register, distractor_count), "train", "gpt-5.4-mini-2026-03-17",
        _VERIFY_TEACHERS, do_back_extraction, random.Random(1),
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


# --- Task 6: produce_rendering ---

def test_row_schema_complete(monkeypatch):
    row, reason = _produce_one(monkeypatch, CompliantTeacher())
    assert reason == "accepted"
    expected = {"pair_id", "graph_id", "source", "domain", "complexity_level", "register",
                "language", "num_states", "num_edges", "distractor_count", "paraphrase_density",
                "condition_explicitness", "verification", "graph", "messages", "split"}
    assert set(row) == expected
    assert json.loads(row["messages"][2]["content"]) == row["graph"]
    assert row["messages"][2]["content"].isascii()
    assert set(row["verification"]) == {"anchor_coverage", "edge_ref_coverage", "back_extraction"}
    assert row["num_states"] == 3 and row["num_edges"] == 2
    assert row["pair_id"] == "G0001_r1"


def test_repair_then_accept(monkeypatch):
    teacher = CompliantTeacher(drop_anchor_on_first_call="WORK")
    row, reason = _produce_one(monkeypatch, teacher)
    assert reason == "accepted" and teacher.calls == 2
    assert "missing state anchor: WORK" in teacher.prompts[1]


def test_drop_after_exhausted_repairs(monkeypatch):
    teacher = CompliantTeacher(always_drop_anchor="WORK")
    row, reason = _produce_one(monkeypatch, teacher)
    assert row is None and reason == "verification" and teacher.calls == 3


def test_distractor_purity_filtered_no_teacher_recall(monkeypatch):
    # Poison the first en distractor with a tool name; draw_distractors must exclude it.
    poisoned = ("Always audit every do_thing invocation in the ledger.",) + pr.DISTRACTOR_LIBRARY["en"]
    monkeypatch.setattr(pr, "DISTRACTOR_LIBRARY", {**pr.DISTRACTOR_LIBRARY, "en": poisoned})
    teacher = CompliantTeacher()
    row, reason = _produce_one(monkeypatch, teacher, distractor_count=1)
    assert reason == "accepted" and teacher.calls == 1  # filtering is free, no re-render
    assert "do_thing invocation in the ledger" not in row["messages"][1]["content"]


def test_back_extraction_failure_drops(monkeypatch):
    empty = json.dumps({"nodes": [], "edges": [], "initial_state": "", "terminal_states": []})
    row, reason = _produce_one(monkeypatch, CompliantTeacher(), do_back_extraction=True,
                               verifier_response=empty)
    assert row is None and reason == "back_extraction"


def test_ids_rederived_per_rendering(monkeypatch):
    # Graph with a branch: START ->(p0) X, START ->(p1) Y, both -> TERMINAL.
    entry = GraphPoolEntry(
        graph_id="G0001", source="registry", domain="d", complexity_level="L1",
        graph={
            "states": ["START", "X", "Y", "TERMINAL"],
            "state_details": [
                {"name": n, "tools": [], "entry_actions": [], "instruction": ""}
                for n in ["START", "X", "Y", "TERMINAL"]
            ],
            "transitions": [
                {"from": "START", "to": "X", "condition": "default", "priority": 0},
                {"from": "START", "to": "Y", "condition": "fallback", "priority": 1},
                {"from": "X", "to": "TERMINAL", "condition": "d", "priority": 0},
                {"from": "Y", "to": "TERMINAL", "condition": "d", "priority": 0},
            ],
            "initial": "START", "terminal": ["TERMINAL"],
        },
        tool_schemas=[],
    )
    playbook_a = ("### START\nBegin. Proceed to X priority 0. Alternatively go to Y priority 1.\n\n"
                  "### X\nDo X. Proceed to TERMINAL.\n\n### Y\nDo Y. Proceed to TERMINAL.\n\n"
                  "### TERMINAL\nDone.")
    playbook_b = ("### START\nBegin. Consider fallback Y priority 1, otherwise proceed to X priority 0.\n\n"
                  "### X\nDo X. Proceed to TERMINAL.\n\n### Y\nDo Y. Proceed to TERMINAL.\n\n"
                  "### TERMINAL\nDone.")

    def make_row(text):
        monkeypatch.setattr(pr, "call_teacher_model", lambda m, s, u: json.dumps({"playbook": text}))
        row, reason = produce_rendering(entry, _plan("prose_narrative"), "train",
                                        "gpt-5.4-mini-2026-03-17", _VERIFY_TEACHERS, False,
                                        random.Random(1))
        assert reason == "accepted", row
        return row

    name_a = {n["id"]: n["name"] for n in make_row(playbook_a)["graph"]["nodes"]}
    name_b = {n["id"]: n["name"] for n in make_row(playbook_b)["graph"]["nodes"]}
    assert name_a["S2"] == "X" and name_b["S2"] == "Y"


# --- Task 7: dataset orchestration ---

def test_leg_should_halt():
    assert not _leg_should_halt(5, 5)      # too few checks
    assert not _leg_should_halt(10, 2)     # ratio 0.20 not > 0.20
    assert _leg_should_halt(10, 3)         # 0.30 > 0.20
    assert _leg_should_halt(20, 5)         # 0.25 > 0.20


def test_drop_undersized_graphs():
    rows = (
        [{"graph_id": "G1"}] * 3
        + [{"graph_id": "G2"}] * 2   # undersized
        + [{"graph_id": "G3"}] * 5
    )
    kept, n_graphs, n_rows = _drop_undersized_graphs(rows, min_renderings=3)
    assert n_graphs == 1 and n_rows == 2
    assert {r["graph_id"] for r in kept} == {"G1", "G3"}


def test_generate_dataset_end_counts(monkeypatch, tmp_path):
    _patch_all_teachers(monkeypatch, CompliantTeacher())
    stats = generate_playbook_dataset(num_graphs=8, invented_ratio=0.5, back_extraction_rate=0.0,
                                      seed=142, output_dir=tmp_path)
    assert stats.renderings_attempted == stats.renderings_accepted + sum(stats.dropped_by_reason.values())
    assert stats.graphs_registry + stats.graphs_invented == 8
    assert stats.output_files and all(f.exists() for f in stats.output_files)
    assert stats.stats_file.exists()
    # Every row of a graph shares one split; splits are valid.
    rows = [json.loads(line) for f in stats.output_files for line in f.read_text().splitlines()]
    by_graph: dict[str, set] = {}
    for r in rows:
        assert r["split"] in ("train", "validation", "test")
        by_graph.setdefault(r["graph_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_graph.values())


def test_seed_determinism(monkeypatch, tmp_path):
    def run(out):
        _patch_all_teachers(monkeypatch, CompliantTeacher())
        generate_playbook_dataset(num_graphs=6, invented_ratio=0.5, back_extraction_rate=0.0,
                                  seed=142, output_dir=out)
        rows = [line for f in sorted(out.glob("pairs_*.jsonl")) for line in f.read_text().splitlines()]
        return sorted(rows)

    assert run(tmp_path / "a") == run(tmp_path / "b")


def test_halt_leg_on_back_extraction(monkeypatch, tmp_path):
    import llm_workflow_agents.data._playbook_verify as pvmod
    _patch_all_teachers(monkeypatch, CompliantTeacher())
    # Verifier always returns an empty graph -> every back-extraction check fails.
    monkeypatch.setattr(pvmod, "call_teacher_model",
                        lambda m, s, u: json.dumps({"nodes": [], "edges": [],
                                                    "initial_state": "", "terminal_states": []}))
    stats = generate_playbook_dataset(num_graphs=40, invented_ratio=0.0, back_extraction_rate=1.0,
                                      seed=142, output_dir=tmp_path)
    assert stats.halted_legs  # at least one leg halted
    assert stats.dropped_by_reason.get("leg_halted", 0) > 0


def test_benchmark_mode(monkeypatch, tmp_path):
    _patch_all_teachers(monkeypatch, CompliantTeacher())
    stats = generate_playbook_dataset(num_graphs=4, invented_ratio=0.0, back_extraction_rate=0.0,
                                      seed=200, output_dir=tmp_path, benchmark_mode=True)
    rows = [json.loads(line) for f in stats.output_files for line in f.read_text().splitlines()]
    assert rows and all("split" not in r for r in rows)
    by_graph: dict[str, set] = {}
    for r in rows:
        by_graph.setdefault(r["graph_id"], set()).add(r["register"])
    assert all(len(regs) == 6 for regs in by_graph.values())
    assert all(f.name.startswith("playbook_pairs_") for f in stats.output_files)


def test_benchmark_single_teacher_no_file_collision(monkeypatch, tmp_path):
    # A single teacher across all language legs must not collapse (source, language)
    # buckets onto one filename and overwrite rows.
    _patch_all_teachers(monkeypatch, CompliantTeacher())
    same = {"en": "gemini-3-flash", "th": "gemini-3-flash", "code_switch": "gemini-3-flash"}
    stats = generate_playbook_dataset(num_graphs=6, invented_ratio=0.5, back_extraction_rate=0.0,
                                      seed=200, output_dir=tmp_path, benchmark_mode=True,
                                      render_teachers=same, invention_teacher="gemini-3-flash")
    assert len(stats.output_files) == len(set(stats.output_files))  # no path collisions
    written = sum(len(f.read_text().splitlines()) for f in tmp_path.glob("*.jsonl"))
    assert written == stats.renderings_accepted  # nothing overwritten/lost


def test_merge_benchmark_runs_dedup(tmp_path):
    def write(path, pair_ids, tag):
        path.write_text("\n".join(json.dumps({"pair_id": p, "tag": tag}) for p in pair_ids) + "\n")

    a = tmp_path / "run_a.jsonl"
    b = tmp_path / "run_b.jsonl"
    write(a, ["p1", "p2", "p3"], "a")
    write(b, ["p2", "p3", "p4"], "b")  # p2,p3 shared
    out = tmp_path / "merged.jsonl"
    count = merge_benchmark_runs([a, b], out)
    merged = [json.loads(line) for line in out.read_text().splitlines()]
    assert count == 4
    by_id = {r["pair_id"]: r["tag"] for r in merged}
    assert by_id == {"p1": "a", "p2": "a", "p3": "a", "p4": "b"}  # first wins


def test_public_export():
    from llm_workflow_agents.data import generate_playbook_dataset as exported
    assert exported is generate_playbook_dataset
