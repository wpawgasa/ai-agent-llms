"""Tests for Task C rendering verification (_playbook_verify.py)."""

from __future__ import annotations

import json
import random

import pytest

import llm_workflow_agents.data._playbook_verify as pv
from llm_workflow_agents.data._playbook_verify import (
    EXTRACTION_SYSTEM_PROMPT,
    assign_state_ids,
    back_extract_check,
    check_distractor_purity,
    check_edge_references,
    find_anchor_occurrences,
    graph_to_eval_shape,
    pick_verifier,
    verify_rendering,
)

GOLD_EVAL = {
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

# Worked-example insurance graph, name-keyed interchange shape.
INSURANCE_GRAPH = {
    "states": ["VERIFY_POLICYHOLDER", "CLAIM_INTAKE", "ASSESS_COVERAGE",
               "RESOLVE", "REQUEST_DOCUMENTATION", "TERMINAL"],
    "state_details": [
        {"name": "VERIFY_POLICYHOLDER", "tools": ["verify_policy"], "entry_actions": [], "instruction": ""},
        {"name": "CLAIM_INTAKE", "tools": ["file_claim"], "entry_actions": [], "instruction": ""},
        {"name": "ASSESS_COVERAGE", "tools": [], "entry_actions": [], "instruction": ""},
        {"name": "RESOLVE", "tools": [], "entry_actions": [], "instruction": ""},
        {"name": "REQUEST_DOCUMENTATION", "tools": [], "entry_actions": [], "instruction": ""},
        {"name": "TERMINAL", "tools": [], "entry_actions": [], "instruction": ""},
    ],
    "transitions": [
        {"from": "VERIFY_POLICYHOLDER", "to": "CLAIM_INTAKE", "condition": "policyholder verified", "priority": 0},
        {"from": "CLAIM_INTAKE", "to": "ASSESS_COVERAGE", "condition": "claim filed", "priority": 0},
        {"from": "ASSESS_COVERAGE", "to": "RESOLVE", "condition": "coverage assessed", "priority": 0},
        {"from": "ASSESS_COVERAGE", "to": "REQUEST_DOCUMENTATION", "condition": "documentation missing", "priority": 1},
        {"from": "REQUEST_DOCUMENTATION", "to": "ASSESS_COVERAGE", "condition": "documents received", "priority": 0},
        {"from": "RESOLVE", "to": "TERMINAL", "condition": "case closed", "priority": 0},
    ],
    "initial": "VERIFY_POLICYHOLDER",
    "terminal": ["TERMINAL"],
}

SOP_RENDERING_A = """# SOP-CL-07: Claims Intake Procedure

1.0 VERIFY_POLICYHOLDER. The representative shall confirm the caller's identity using the verify_policy tool. Upon success, proceed to 2.0 CLAIM_INTAKE.

2.0 CLAIM_INTAKE. Record the incident details and file the claim with file_claim. Continue to 3.0 ASSESS_COVERAGE.

3.0 ASSESS_COVERAGE. Determine whether the incident is covered. When coverage is confirmed, continue to 4.0 RESOLVE. If documentation is missing, transfer to 3.1 REQUEST_DOCUMENTATION and return to ASSESS_COVERAGE upon receipt.

4.0 RESOLVE. Resolve the case and close it (TERMINAL)."""


def test_anchor_regex_prefix_safe():
    graph = {"states": ["RESOLVE", "RESOLVE_ESCALATION"], "state_details": [],
             "transitions": [], "initial": "RESOLVE", "terminal": ["RESOLVE_ESCALATION"]}
    ids = assign_state_ids(graph, "Only RESOLVE_ESCALATION is mentioned here.")
    assert ids is None  # RESOLVE itself never anchored (boundary-safe)


def test_assign_state_ids_worked_example():
    ids = assign_state_ids(INSURANCE_GRAPH, SOP_RENDERING_A)
    assert ids == {
        "VERIFY_POLICYHOLDER": "S1", "CLAIM_INTAKE": "S2", "ASSESS_COVERAGE": "S3",
        "RESOLVE": "S4", "REQUEST_DOCUMENTATION": "S5", "TERMINAL": "S6",
    }


def test_find_anchor_occurrences_sorted():
    occ = find_anchor_occurrences("go to WORK then START", ["START", "WORK"])
    assert [name for _pos, name in occ] == ["WORK", "START"]


def test_edge_ref_window_worked_example():
    result = check_edge_references(SOP_RENDERING_A, INSURANCE_GRAPH)
    assert result.coverage == 1.0 and result.branch_ok
    assert result.failed_edges == []


def test_edge_ref_window_branch_fail():
    graph = {
        "states": ["ALPHA", "BETA", "GAMMA"],
        "state_details": [],
        "transitions": [
            {"from": "ALPHA", "to": "BETA", "condition": "default", "priority": 0},
            {"from": "ALPHA", "to": "GAMMA", "condition": "fallback", "priority": 1},
        ],
        "initial": "ALPHA", "terminal": ["BETA"],
    }
    text = ("### ALPHA\nStart here, then go to BETA.\n\n"
            "### BETA\nnothing relevant here.\n\n"
            "### GAMMA\nthis branch may be reached as a fallback.")
    res = check_edge_references(text, graph)
    assert ("ALPHA", "GAMMA") in res.failed_edges and not res.branch_ok


def test_graph_to_eval_shape_permutes_with_mention_order():
    from llm_workflow_agents.eval.graph_extraction_eval import WorkflowGraph, check_structural_validity

    ids_a = {"VERIFY_POLICYHOLDER": "S1", "CLAIM_INTAKE": "S2", "ASSESS_COVERAGE": "S3",
             "RESOLVE": "S4", "REQUEST_DOCUMENTATION": "S5", "TERMINAL": "S6"}
    ids_b = dict(ids_a, RESOLVE="S5", REQUEST_DOCUMENTATION="S4")

    shape_a = graph_to_eval_shape(INSURANCE_GRAPH, ids_a)
    shape_b = graph_to_eval_shape(INSURANCE_GRAPH, ids_b)
    assert check_structural_validity(WorkflowGraph(**shape_a))
    assert check_structural_validity(WorkflowGraph(**shape_b))
    name_of = {n["id"]: n["name"] for n in shape_b["nodes"]}
    assert name_of["S4"] == "REQUEST_DOCUMENTATION" and name_of["S5"] == "RESOLVE"
    assert [n["id"] for n in shape_a["nodes"]] == ["S1", "S2", "S3", "S4", "S5", "S6"]


def test_verify_rendering_accepts_worked_example():
    report = verify_rendering(SOP_RENDERING_A, INSURANCE_GRAPH, ["verify_policy", "file_claim"])
    assert report.accepted, report.corrections
    assert report.state_ids is not None and report.initial_first


def test_verify_rendering_corrections_itemized():
    graph = {
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
        "initial": "START", "terminal": ["TERMINAL"],
    }
    # WORK mentioned before START; TERMINAL not anchored; do_thing tool not mentioned.
    playbook = "First we handle WORK. Only afterwards do we reach START."
    report = verify_rendering(playbook, graph, ["do_thing"])
    assert not report.accepted
    assert "missing state anchor: TERMINAL" in report.corrections
    assert "missing tool mention: do_thing" in report.corrections
    assert any("initial state must be introduced" in c for c in report.corrections)


def test_state_script_render_passes_verification():
    from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
    from llm_workflow_agents.data._workflow_script import build_workflow_script
    from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
    from llm_workflow_agents.data.generate_workflows import select_subgraph

    wf = select_subgraph(DOMAIN_REGISTRY["banking"], COMPLEXITY_SPECS[ComplexityLevel.L2], random.Random(142))
    gd = wf.to_dict()
    text = build_workflow_script(gd)
    tool_names = sorted({t for sd in gd["state_details"] for t in sd["tools"]})
    report = verify_rendering(text, gd, tool_names)
    assert report.accepted, report.corrections


def test_check_distractor_purity():
    bad = ["Log every do_thing call.", "A clean paragraph about the weather."]
    offending = check_distractor_purity(bad, ["START", "WORK"], ["do_thing"])
    assert offending == ["Log every do_thing call."]


def test_back_extraction_pass_and_prompt(monkeypatch):
    captured: dict[str, str] = {}

    def fake(model, system_prompt, user_prompt):
        captured["sys"] = system_prompt
        captured["user"] = user_prompt
        return json.dumps(GOLD_EVAL)

    monkeypatch.setattr(pv, "call_teacher_model", fake)
    res = back_extract_check("the playbook text", GOLD_EVAL, "gemini-3-flash")
    assert res == {"node_f1": 1.0, "edge_f1": 1.0, "passed": True}
    assert captured["sys"] == EXTRACTION_SYSTEM_PROMPT
    assert captured["user"] == "the playbook text"


def test_back_extraction_fail_thresholds(monkeypatch):
    partial = json.loads(json.dumps(GOLD_EVAL))
    partial["edges"] = partial["edges"][:3]  # 3 of 6 edges -> edge F1 = 0.667 < 0.80
    monkeypatch.setattr(pv, "call_teacher_model", lambda m, s, u: json.dumps(partial))
    res = back_extract_check("x", GOLD_EVAL, "gemini-3-flash")
    assert not res["passed"]
    assert res["edge_f1"] < 0.80


def test_back_extraction_unparseable(monkeypatch):
    monkeypatch.setattr(pv, "call_teacher_model", lambda m, s, u: "sorry, I cannot help")
    res = back_extract_check("x", GOLD_EVAL, "gpt-5.4-nano-2026-03-17")
    assert res == {"node_f1": 0.0, "edge_f1": 0.0, "passed": False}


def test_pick_verifier_cross_family():
    vt = {"gpt": "gemini-3-flash", "gemini": "gpt-5.4-nano-2026-03-17"}
    assert pick_verifier("gpt-5.4-mini-2026-03-17", vt) == "gemini-3-flash"
    assert pick_verifier("gemini-3-flash-preview", vt) == "gpt-5.4-nano-2026-03-17"
    with pytest.raises(ValueError):
        pick_verifier("claude-x", vt)
