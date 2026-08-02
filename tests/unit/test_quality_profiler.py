"""Tests for the Task A quality profiler (`quality_profiler.profile_task_a`)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_workflow_agents.data.quality_profiler import profile_task_a


def _base_rec(**overrides) -> dict:
    rec = {
        "conversation_id": "X_001",
        "complexity_level": "L1",
        "domain": "d",
        "workflow_graph": {
            "states": ["A", "TERMINAL"],
            "transitions": [{"from": "A", "to": "TERMINAL", "condition": "", "priority": 0}],
            "initial": "A",
            "terminal": ["TERMINAL"],
        },
        "tool_schemas": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
        "messages": [],
        "ground_truth": {
            "state_sequence": [{"from": "A", "to": "TERMINAL"}],
            "tool_calls": [],
            "tool_chain_dependencies": [],
            "terminal_state": "TERMINAL",
            "terminal_reached": True,
        },
        "language": "en",
    }
    rec.update(overrides)
    return rec


def _write(tmp_path: Path, rec: dict, name: str = "l1_test.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(rec) + "\n")
    return path


def test_profile_task_a_flags_advancing_tool_turn(tmp_path):
    rec = _base_rec(
        messages=[
            {
                "role": "assistant",
                "annotations": None,
                "content": '[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>',
            },
        ],
    )
    path = _write(tmp_path, rec)
    report = profile_task_a(path)
    assert any(d.startswith("X_001: assistant turn 1 issues a <tool_call>") for d in report.defects)
    assert report.distributions["tool_turn_state"]["advancing"] == 1
    assert report.distributions["tool_turn_state"]["self_loop"] == 0
    assert report.distributions["tool_turn_state"]["pct_conformant"] == 0.0


def test_profile_task_a_self_loop_tool_turn_is_conformant(tmp_path):
    # Note: the msg-STATE-seq vs ground_truth check requires the message
    # sequence to equal `ground_truth.state_sequence`. We give a matching
    # single self-loop GT sequence so this sample stays otherwise clean and
    # we can isolate the tool_turn_state distribution.
    rec = _base_rec(
        ground_truth={
            "state_sequence": [{"from": "A", "to": "A"}],
            "tool_calls": [],
            "tool_chain_dependencies": [],
            "terminal_state": "TERMINAL",
            "terminal_reached": False,
        },
        messages=[
            {
                "role": "assistant",
                "annotations": None,
                "content": '[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>',
            },
        ],
    )
    path = _write(tmp_path, rec, "l1_self_loop.jsonl")
    report = profile_task_a(path)
    assert not any("issues a <tool_call>" in d for d in report.defects)
    assert report.distributions["tool_turn_state"]["self_loop"] == 1
    assert report.distributions["tool_turn_state"]["advancing"] == 0
    assert report.distributions["tool_turn_state"]["pct_conformant"] == 100.0


def test_tool_turn_state_ignores_tool_call_with_no_state_marker(tmp_path):
    # A malformed/corrupted assistant turn that contains a <tool_call> but no
    # parsable [STATE: ...] marker at all must NOT be counted in either bucket
    # of tool_turn_state -- it isn't part of the population
    # find_tool_stay_violations actually inspects (parse_assistant_turns
    # returns None for it), so a naive "<tool_call> in content" substring
    # count over all assistant messages would silently misclassify it as a
    # conformant self-loop.
    rec = _base_rec(
        ground_truth={
            "state_sequence": [],
            "tool_calls": [],
            "tool_chain_dependencies": [],
            "terminal_state": "TERMINAL",
            "terminal_reached": False,
        },
        messages=[
            {
                "role": "assistant",
                "annotations": None,
                "content": '<tool_call>{"name": "t", "arguments": {}}</tool_call>',
            },
        ],
    )
    path = _write(tmp_path, rec, "l1_no_marker.jsonl")
    report = profile_task_a(path)
    tts = report.distributions["tool_turn_state"]
    assert tts["self_loop"] == 0
    assert tts["advancing"] == 0
    assert tts["pct_conformant"] == 100.0
