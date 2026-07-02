"""Tests for Task C rendering verification (_playbook_verify.py)."""

from __future__ import annotations

import random

from llm_workflow_agents.data._playbook_verify import (
    assign_state_ids,
    check_distractor_purity,
    check_edge_references,
    find_anchor_occurrences,
    graph_to_eval_shape,
    verify_rendering,
)

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
