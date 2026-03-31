"""Tests for chat template converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.data.chat_template_converter import (
    ConversionStats,
    convert_to_model_format,
)


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a sample JSONL file for conversion testing."""
    path = tmp_path / "input.jsonl"
    samples = [
        {
            "conversation_id": "test_001",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": '[STATE: GREETING → COLLECT_INFO]\n<tool_call>{"name": "lookup", "arguments": {"id": "123"}}</tool_call>',
                    "annotations": {
                        "state_transition": {"from": "GREETING", "to": "COLLECT_INFO"},
                        "tool_calls": [{"name": "lookup", "arguments": {"id": "123"}}],
                    },
                },
                {"role": "tool", "content": '{"status": "ok"}'},
                {"role": "assistant", "content": "Here is the result."},
            ],
        },
        {
            "conversation_id": "test_002",
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Simple question"},
                {"role": "assistant", "content": "Simple answer"},
            ],
        },
    ]
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return path


class TestChatTemplateConverter:
    """Tests for format conversion across all 5 model families."""

    @pytest.mark.parametrize("family", ["qwen", "gemma", "mistral", "nemotron", "glm"])
    def test_convert_all_families(self, sample_jsonl: Path, tmp_path: Path, family: str) -> None:
        output = tmp_path / f"output_{family}.jsonl"
        stats = convert_to_model_format(sample_jsonl, family, output)

        assert isinstance(stats, ConversionStats)
        assert stats.input_samples == 2
        assert stats.output_samples == 2
        assert stats.skipped == 0
        assert output.exists()

    def test_qwen_preserves_tool_calls(self, sample_jsonl: Path, tmp_path: Path) -> None:
        output = tmp_path / "qwen.jsonl"
        convert_to_model_format(sample_jsonl, "qwen", output)

        with open(output) as f:
            sample = json.loads(f.readline())
            messages = sample["messages"]
            # Tool message should be preserved
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            assert len(tool_msgs) == 1

    def test_gemma_uses_model_role(self, sample_jsonl: Path, tmp_path: Path) -> None:
        output = tmp_path / "gemma.jsonl"
        convert_to_model_format(sample_jsonl, "gemma", output)

        with open(output) as f:
            sample = json.loads(f.readline())
            messages = sample["messages"]
            model_msgs = [m for m in messages if m["role"] == "model"]
            assert len(model_msgs) > 0

    def test_glm_tool_becomes_observation(self, sample_jsonl: Path, tmp_path: Path) -> None:
        output = tmp_path / "glm.jsonl"
        convert_to_model_format(sample_jsonl, "glm", output)

        with open(output) as f:
            sample = json.loads(f.readline())
            messages = sample["messages"]
            obs_msgs = [m for m in messages if m["role"] == "observation"]
            assert len(obs_msgs) == 1

    def test_mistral_adds_tool_calls_field(self, sample_jsonl: Path, tmp_path: Path) -> None:
        output = tmp_path / "mistral.jsonl"
        stats = convert_to_model_format(sample_jsonl, "mistral", output)

        assert stats.tool_calls_converted >= 1

    def test_empty_input_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        output = tmp_path / "output.jsonl"
        stats = convert_to_model_format(empty, "qwen", output)
        assert stats.input_samples == 0
        assert stats.output_samples == 0

    def test_invalid_json_line_skipped(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"valid": true, "messages": []}\nnot json\n')
        output = tmp_path / "output.jsonl"
        stats = convert_to_model_format(bad, "qwen", output)
        assert stats.input_samples == 2
        assert stats.skipped == 1
        assert len(stats.errors) == 1
