"""Tests for graph extraction evaluation and constrained decoding."""

from __future__ import annotations

import json

import pytest

from llm_workflow_agents.eval.graph_extraction_eval import (
    GraphExtractionMetrics,
    WorkflowGraph,
    _bfs,
    _extract_json,
    check_mermaid_renderability,
    check_structural_validity,
    compute_edge_f1,
    compute_graph_edit_distance,
    compute_node_f1,
    evaluate_graph_extraction,
    graph_to_mermaid,
    parse_graph_json,
)
from llm_workflow_agents.eval.constrained_decoding import (
    WORKFLOW_GRAPH_SCHEMA,
    build_xgrammar_constraint,
    load_graph_schema,
)


# --- Helpers ---

def _make_graph(
    nodes: list[dict],
    edges: list[dict],
    initial: str = "S1",
    terminals: list[str] | None = None,
) -> WorkflowGraph:
    return WorkflowGraph(
        nodes=nodes,
        edges=edges,
        initial_state=initial,
        terminal_states=terminals or ["S3"],
    )


SIMPLE_GOLD = _make_graph(
    nodes=[{"id": "S1", "name": "Start"}, {"id": "S2", "name": "Mid"}, {"id": "S3", "name": "End"}],
    edges=[
        {"from_state": "S1", "to_state": "S2", "condition": "proceed"},
        {"from_state": "S2", "to_state": "S3", "condition": "done"},
    ],
)


# ============================================================
# JSON Parsing Tests
# ============================================================


class TestExtractJson:

    def test_plain_json(self) -> None:
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_json_in_text(self) -> None:
        text = 'Here is the graph: {"nodes": []} and more text'
        assert _extract_json(text) == '{"nodes": []}'

    def test_nested_json(self) -> None:
        text = '{"a": {"b": 1}}'
        assert _extract_json(text) == '{"a": {"b": 1}}'

    def test_no_json(self) -> None:
        assert _extract_json("no json here") is None

    def test_unclosed_json(self) -> None:
        assert _extract_json('{"a": 1') is None


class TestParseGraphJson:

    def test_valid_graph(self) -> None:
        data = {
            "nodes": [{"id": "S1", "name": "A"}],
            "edges": [],
            "initial_state": "S1",
            "terminal_states": ["S1"],
        }
        graph, valid = parse_graph_json(json.dumps(data))
        assert valid
        assert graph is not None
        assert len(graph.nodes) == 1

    def test_invalid_json(self) -> None:
        graph, valid = parse_graph_json("not json")
        assert not valid
        assert graph is None

    def test_json_in_text(self) -> None:
        data = {"nodes": [], "edges": [], "initial_state": "S1", "terminal_states": ["S1"]}
        text = f"The graph is: {json.dumps(data)} done."
        graph, valid = parse_graph_json(text)
        assert valid

    def test_non_dict_json(self) -> None:
        _, valid = parse_graph_json("[1, 2, 3]")
        assert not valid


# ============================================================
# Node F1 Tests
# ============================================================


class TestComputeNodeF1:

    def test_perfect_match(self) -> None:
        pred = _make_graph(
            nodes=[{"id": "S1"}, {"id": "S2"}, {"id": "S3"}],
            edges=[],
        )
        assert compute_node_f1(pred, SIMPLE_GOLD) == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        pred = _make_graph(
            nodes=[{"id": "S1"}, {"id": "S2"}],  # Missing S3
            edges=[],
        )
        f1 = compute_node_f1(pred, SIMPLE_GOLD)
        # precision=2/2=1.0, recall=2/3, F1=0.8
        assert f1 == pytest.approx(0.8, abs=0.01)

    def test_no_overlap(self) -> None:
        pred = _make_graph(nodes=[{"id": "X1"}], edges=[])
        assert compute_node_f1(pred, SIMPLE_GOLD) == 0.0

    def test_both_empty(self) -> None:
        pred = _make_graph(nodes=[], edges=[])
        gold = _make_graph(nodes=[], edges=[])
        assert compute_node_f1(pred, gold) == 1.0

    def test_pred_empty(self) -> None:
        pred = _make_graph(nodes=[], edges=[])
        assert compute_node_f1(pred, SIMPLE_GOLD) == 0.0


# ============================================================
# Edge F1 Tests
# ============================================================


class TestComputeEdgeF1:

    def test_perfect_match(self) -> None:
        pred = _make_graph(
            nodes=[],
            edges=[
                {"from_state": "S1", "to_state": "S2"},
                {"from_state": "S2", "to_state": "S3"},
            ],
        )
        assert compute_edge_f1(pred, SIMPLE_GOLD) == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        pred = _make_graph(
            nodes=[],
            edges=[{"from_state": "S1", "to_state": "S2"}],
        )
        f1 = compute_edge_f1(pred, SIMPLE_GOLD)
        # precision=1/1=1.0, recall=1/2=0.5, F1=2/3
        assert f1 == pytest.approx(2 / 3, abs=0.01)

    def test_no_overlap(self) -> None:
        pred = _make_graph(nodes=[], edges=[{"from_state": "X", "to_state": "Y"}])
        assert compute_edge_f1(pred, SIMPLE_GOLD) == 0.0

    def test_both_empty(self) -> None:
        pred = _make_graph(nodes=[], edges=[])
        gold = _make_graph(nodes=[], edges=[])
        assert compute_edge_f1(pred, gold) == 1.0


# ============================================================
# GED Tests
# ============================================================


class TestComputeGED:

    def test_identical_graphs(self) -> None:
        ged = compute_graph_edit_distance(SIMPLE_GOLD, SIMPLE_GOLD)
        assert ged == pytest.approx(0.0)

    def test_different_graphs(self) -> None:
        pred = _make_graph(
            nodes=[{"id": "A"}, {"id": "B"}],
            edges=[{"from_state": "A", "to_state": "B"}],
        )
        ged = compute_graph_edit_distance(pred, SIMPLE_GOLD)
        assert ged > 0.0
        assert ged <= 1.0  # normalized

    def test_empty_vs_nonempty(self) -> None:
        pred = _make_graph(nodes=[], edges=[])
        ged = compute_graph_edit_distance(pred, SIMPLE_GOLD)
        assert ged > 0.0


# ============================================================
# Structural Validity Tests
# ============================================================


class TestCheckStructuralValidity:

    def test_valid_graph(self) -> None:
        assert check_structural_validity(SIMPLE_GOLD)

    def test_empty_nodes(self) -> None:
        assert not check_structural_validity(_make_graph(nodes=[], edges=[]))

    def test_missing_initial_state(self) -> None:
        graph = _make_graph(
            nodes=[{"id": "S1"}],
            edges=[],
            initial="MISSING",
            terminals=["S1"],
        )
        assert not check_structural_validity(graph)

    def test_terminal_not_in_nodes(self) -> None:
        graph = _make_graph(
            nodes=[{"id": "S1"}],
            edges=[],
            initial="S1",
            terminals=["MISSING"],
        )
        assert not check_structural_validity(graph)

    def test_unreachable_terminal(self) -> None:
        graph = _make_graph(
            nodes=[{"id": "S1"}, {"id": "S2"}, {"id": "S3"}],
            edges=[{"from_state": "S1", "to_state": "S2"}],
            # S3 is terminal but unreachable
            initial="S1",
            terminals=["S3"],
        )
        assert not check_structural_validity(graph)

    def test_no_terminal_states(self) -> None:
        graph = WorkflowGraph(
            nodes=[{"id": "S1"}],
            edges=[],
            initial_state="S1",
            terminal_states=[],
        )
        assert not check_structural_validity(graph)


# ============================================================
# Mermaid Renderability Tests
# ============================================================


class TestCheckMermaidRenderability:

    def test_valid_graph(self) -> None:
        assert check_mermaid_renderability(SIMPLE_GOLD)

    def test_empty_graph(self) -> None:
        assert not check_mermaid_renderability(_make_graph(nodes=[], edges=[]))

    def test_invalid_node_id(self) -> None:
        graph = _make_graph(
            nodes=[{"id": "S 1"}],  # Space in ID
            edges=[],
        )
        assert not check_mermaid_renderability(graph)

    def test_edge_to_nonexistent_node(self) -> None:
        graph = _make_graph(
            nodes=[{"id": "S1"}],
            edges=[{"from_state": "S1", "to_state": "MISSING"}],
        )
        assert not check_mermaid_renderability(graph)


class TestGraphToMermaid:

    def test_simple_graph(self) -> None:
        mermaid = graph_to_mermaid(SIMPLE_GOLD)
        assert "graph TD" in mermaid
        assert "S1[Start]" in mermaid
        assert "S1 -->|proceed| S2" in mermaid


# ============================================================
# BFS Tests
# ============================================================


class TestBFS:

    def test_simple_path(self) -> None:
        adj = {"A": ["B"], "B": ["C"], "C": []}
        assert _bfs("A", adj) == {"A", "B", "C"}

    def test_disconnected(self) -> None:
        adj = {"A": ["B"], "B": [], "C": []}
        assert _bfs("A", adj) == {"A", "B"}

    def test_cycle(self) -> None:
        adj = {"A": ["B"], "B": ["A"]}
        assert _bfs("A", adj) == {"A", "B"}


# ============================================================
# Full Evaluation Tests
# ============================================================


class TestEvaluateGraphExtraction:

    def test_perfect_extraction(self) -> None:
        pred = {
            "nodes": [{"id": "S1", "name": "Start"}, {"id": "S2", "name": "Mid"}, {"id": "S3", "name": "End"}],
            "edges": [
                {"from_state": "S1", "to_state": "S2", "condition": "proceed"},
                {"from_state": "S2", "to_state": "S3", "condition": "done"},
            ],
            "initial_state": "S1",
            "terminal_states": ["S3"],
        }
        metrics = evaluate_graph_extraction([pred], [SIMPLE_GOLD])
        assert metrics.node_f1 == pytest.approx(1.0)
        assert metrics.edge_f1 == pytest.approx(1.0)
        assert metrics.json_validity == 1.0
        assert metrics.structural_validity == 1.0
        assert metrics.mermaid_renderability == 1.0

    def test_json_string_input(self) -> None:
        pred = json.dumps({
            "nodes": [{"id": "S1", "name": "A"}],
            "edges": [],
            "initial_state": "S1",
            "terminal_states": ["S1"],
        })
        gold = _make_graph(
            nodes=[{"id": "S1", "name": "A"}],
            edges=[],
            initial="S1",
            terminals=["S1"],
        )
        metrics = evaluate_graph_extraction([pred], [gold])
        assert metrics.json_validity == 1.0

    def test_invalid_json_input(self) -> None:
        metrics = evaluate_graph_extraction(["not json"], [SIMPLE_GOLD])
        assert metrics.json_validity == 0.0
        assert metrics.node_f1 == 0.0

    def test_empty_inputs(self) -> None:
        metrics = evaluate_graph_extraction([], [])
        assert metrics.node_f1 == 0.0


# ============================================================
# Constrained Decoding Tests
# ============================================================


class TestConstrainedDecoding:

    def test_schema_has_required_fields(self) -> None:
        assert "nodes" in WORKFLOW_GRAPH_SCHEMA["properties"]
        assert "edges" in WORKFLOW_GRAPH_SCHEMA["properties"]
        assert "initial_state" in WORKFLOW_GRAPH_SCHEMA["properties"]
        assert "terminal_states" in WORKFLOW_GRAPH_SCHEMA["properties"]
        assert set(WORKFLOW_GRAPH_SCHEMA["required"]) == {
            "nodes", "edges", "initial_state", "terminal_states"
        }

    def test_load_graph_schema_default(self) -> None:
        schema = load_graph_schema()
        assert schema == WORKFLOW_GRAPH_SCHEMA

    def test_load_graph_schema_from_file(self, tmp_path) -> None:
        custom = {"type": "object", "properties": {"custom": {"type": "string"}}}
        path = tmp_path / "schema.json"
        path.write_text(json.dumps(custom))
        loaded = load_graph_schema(path)
        assert loaded == custom

    def test_build_xgrammar_constraint(self) -> None:
        schema = build_xgrammar_constraint()
        assert schema == WORKFLOW_GRAPH_SCHEMA

    def test_build_xgrammar_custom_schema(self) -> None:
        custom = {"type": "object"}
        assert build_xgrammar_constraint(custom) == custom


# ============================================================
# Module Import Tests
# ============================================================


class TestModuleImports:

    def test_import_graph_eval(self) -> None:
        from llm_workflow_agents.eval import GraphExtractionMetrics, evaluate_graph_extraction
        assert GraphExtractionMetrics is not None
        assert evaluate_graph_extraction is not None

    def test_import_constrained_decoding(self) -> None:
        from llm_workflow_agents.eval import WORKFLOW_GRAPH_SCHEMA
        assert WORKFLOW_GRAPH_SCHEMA is not None
