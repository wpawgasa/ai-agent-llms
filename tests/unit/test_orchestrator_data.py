"""Tests for orchestrator routing training data generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.data.generate_orchestrator_data import (
    OrchestratorDatasetMetadata,
    OrchestratorSample,
    generate_orchestrator_dataset,
)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "orchestrator_output"


class TestGenerateOrchestratorDataset:

    def test_generates_correct_count(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=50, output_dir=tmp_output_dir, seed=42
        )
        assert isinstance(result, OrchestratorDatasetMetadata)
        assert result.num_samples == 50

    def test_creates_train_val_test_splits(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=100, output_dir=tmp_output_dir, seed=42
        )
        assert len(result.output_files) == 3
        for path in result.output_files:
            assert path.exists()

    def test_split_ratios(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=100, output_dir=tmp_output_dir, seed=42
        )
        stats = result.stats
        assert stats["splits"]["train"] == 85
        assert stats["splits"]["val"] == 10
        assert stats["splits"]["test"] == 5

    def test_routing_distribution(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=500, output_dir=tmp_output_dir, seed=42
        )
        routing = result.stats["routing_distribution"]
        # Default 60/20/20
        assert routing["tool_execution"] > routing["graph_extraction"]
        assert routing["tool_execution"] > routing["self_handle"]
        assert routing["tool_execution"] / 500 > 0.45

    def test_custom_routing_distribution(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=100,
            output_dir=tmp_output_dir,
            seed=42,
            routing_distribution={
                "tool_execution": 0.0,
                "graph_extraction": 1.0,
                "self_handle": 0.0,
            },
        )
        routing = result.stats["routing_distribution"]
        assert routing["graph_extraction"] == 100
        assert routing["tool_execution"] == 0

    def test_multiple_domains_covered(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=200, output_dir=tmp_output_dir, seed=42
        )
        assert result.stats["num_domains"] > 5

    def test_seed_determinism(self, tmp_output_dir: Path) -> None:
        dir1 = tmp_output_dir / "run1"
        dir2 = tmp_output_dir / "run2"
        r1 = generate_orchestrator_dataset(num_samples=20, output_dir=dir1, seed=42)
        r2 = generate_orchestrator_dataset(num_samples=20, output_dir=dir2, seed=42)
        with open(r1.output_files[0]) as f1, open(r2.output_files[0]) as f2:
            assert f1.read() == f2.read()


class TestOrchestratorSampleStructure:

    def test_jsonl_valid(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=20, output_dir=tmp_output_dir, seed=42
        )
        for path in result.output_files:
            with open(path) as f:
                for line in f:
                    sample = json.loads(line)
                    assert "conversation_id" in sample
                    assert "routing_target" in sample
                    assert "messages" in sample
                    assert sample["routing_target"] in (
                        "tool_execution", "graph_extraction", "self_handle"
                    )

    def test_messages_start_with_system(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=20, output_dir=tmp_output_dir, seed=42
        )
        with open(result.output_files[0]) as f:
            for line in f:
                sample = json.loads(line)
                assert sample["messages"][0]["role"] == "system"

    def test_tool_execution_has_routing_annotation(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=50,
            output_dir=tmp_output_dir,
            seed=42,
            routing_distribution={"tool_execution": 1.0, "graph_extraction": 0.0, "self_handle": 0.0},
        )
        with open(result.output_files[0]) as f:
            for line in f:
                sample = json.loads(line)
                # Find assistant message with routing
                assistant_msgs = [m for m in sample["messages"] if m["role"] == "assistant"]
                assert any("[ROUTE: tool_execution]" in m["content"] for m in assistant_msgs)

    def test_graph_extraction_has_delegate_tag(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=50,
            output_dir=tmp_output_dir,
            seed=42,
            routing_distribution={"tool_execution": 0.0, "graph_extraction": 1.0, "self_handle": 0.0},
        )
        with open(result.output_files[0]) as f:
            for line in f:
                sample = json.loads(line)
                assistant_msgs = [m for m in sample["messages"] if m["role"] == "assistant"]
                assert any("<delegate_to_specialist>" in m["content"] for m in assistant_msgs)

    def test_self_handle_has_no_specialist(self, tmp_output_dir: Path) -> None:
        result = generate_orchestrator_dataset(
            num_samples=50,
            output_dir=tmp_output_dir,
            seed=42,
            routing_distribution={"tool_execution": 0.0, "graph_extraction": 0.0, "self_handle": 1.0},
        )
        with open(result.output_files[0]) as f:
            for line in f:
                sample = json.loads(line)
                # Self-handle should have no tool (specialist) messages
                tool_msgs = [m for m in sample["messages"] if m["role"] == "tool"]
                assert len(tool_msgs) == 0


class TestOrchestratorSampleDataclass:

    def test_to_dict(self) -> None:
        sample = OrchestratorSample(
            conversation_id="test_001",
            domain="banking",
            routing_target="tool_execution",
            intent="balance_inquiry",
            messages=[{"role": "system", "content": "test"}],
            num_turns=2,
        )
        d = sample.to_dict()
        assert d["conversation_id"] == "test_001"
        assert d["domain"] == "banking"
        assert d["routing_target"] == "tool_execution"
        assert d["num_turns"] == 2
