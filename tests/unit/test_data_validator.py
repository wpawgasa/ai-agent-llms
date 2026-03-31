"""Tests for data validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.data.data_validator import ValidationResult, validate_dataset


@pytest.fixture
def valid_workflow_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "valid_workflow.jsonl"
    sample = {
        "conversation_id": "L1_001",
        "complexity_level": "L1",
        "domain": "faq_lookup",
        "workflow_graph": {
            "states": ["S1", "S2", "S3"],
            "transitions": [{"from": "S1", "to": "S2", "condition": "proceed"}],
            "initial": "S1",
            "terminal": ["S3"],
        },
        "messages": [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
        ],
    }
    path.write_text(json.dumps(sample) + "\n")
    return path


@pytest.fixture
def valid_graph_pair_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "valid_graph.jsonl"
    sample = {
        "id": "gold_0001",
        "messages": [
            {"role": "system", "content": "Extract graph."},
            {"role": "user", "content": "Workflow prompt"},
            {"role": "assistant", "content": "{}"},
        ],
        "graph": {
            "nodes": [{"id": "S1", "name": "Start"}],
            "edges": [],
            "initial_state": "S1",
            "terminal_states": ["S1"],
        },
    }
    path.write_text(json.dumps(sample) + "\n")
    return path


class TestDataValidator:

    def test_valid_workflow(self, valid_workflow_jsonl: Path) -> None:
        result = validate_dataset(valid_workflow_jsonl, "workflow")
        assert result.valid
        assert result.total_samples == 1
        assert result.valid_samples == 1

    def test_invalid_workflow_missing_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"conversation_id": "test"}) + "\n")
        result = validate_dataset(path, "workflow")
        assert not result.valid
        assert len(result.errors) > 0

    def test_valid_graph_pair(self, valid_graph_pair_jsonl: Path) -> None:
        result = validate_dataset(valid_graph_pair_jsonl, "graph_pair")
        assert result.valid

    def test_graph_pair_orphan_edge(self, tmp_path: Path) -> None:
        path = tmp_path / "orphan.jsonl"
        sample = {
            "id": "test",
            "messages": [{"role": "user", "content": "x"}],
            "graph": {
                "nodes": [{"id": "S1", "name": "A"}],
                "edges": [{"from_state": "S1", "to_state": "S99", "condition": "x"}],
                "initial_state": "S1",
                "terminal_states": ["S1"],
            },
        }
        path.write_text(json.dumps(sample) + "\n")
        result = validate_dataset(path, "graph_pair")
        assert not result.valid
        assert any("S99" in e for e in result.errors)

    def test_unknown_dataset_type(self, tmp_path: Path) -> None:
        path = tmp_path / "any.jsonl"
        path.write_text("{}\n")
        result = validate_dataset(path, "unknown_type")
        assert not result.valid

    def test_tool_call_validation(self, tmp_path: Path) -> None:
        path = tmp_path / "tool.jsonl"
        sample = {"id": "tc_001", "messages": [{"role": "user", "content": "test"}]}
        path.write_text(json.dumps(sample) + "\n")
        result = validate_dataset(path, "tool_call")
        assert result.valid
