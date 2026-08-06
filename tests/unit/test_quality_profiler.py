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


def test_profile_task_a_ignores_tool_call_examples_in_the_system_prompt(tmp_path):
    # The enriched system prompt teaches the <tool_call> syntax by SHOWING it:
    # FORMAT_RULES carries a placeholder template that is deliberately invalid
    # JSON plus a worked example naming `request_referral`, a tool that belongs
    # to the illustration and not to this row's tool_schemas. Scanning the
    # system message therefore charged 3 phantom hard defects to every row
    # whose prompt had been rebuilt (remediate_task_a_states.py
    # --rebuild-prompts), which would make the v2 "zero hard defects"
    # acceptance gate unsatisfiable. A contract document is not a tool-call site.
    system_content = (
        "You are a helpful agent.\n\n"
        "Rules:\n\n"
        '1. Tool-call format:\n'
        '       <tool_call>{"name": "<tool_name>", "arguments": {<arg_key>: <arg_value>}}</tool_call>\n'
        "   Worked example:\n"
        "       [STATE: VERIFY_PATIENT → VERIFY_PATIENT]\n"
        '       <tool_call>{"name": "request_referral", "arguments": {"patient_id": "P12345"}}</tool_call>'
    )
    rec = _base_rec(
        messages=[
            {"role": "system", "annotations": None, "content": system_content},
            {"role": "user", "annotations": None, "content": "hi"},
            {
                "role": "assistant",
                "annotations": None,
                "content": '[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>',
            },
            {"role": "tool", "annotations": None, "content": "{}"},
            {"role": "assistant", "annotations": None, "content": "[STATE: A → TERMINAL]\nDone."},
        ],
        ground_truth={
            "state_sequence": [{"from": "A", "to": "A"}, {"from": "A", "to": "TERMINAL"}],
            "tool_calls": [],
            "tool_chain_dependencies": [],
            "terminal_state": "TERMINAL",
            "terminal_reached": True,
        },
    )
    path = _write(tmp_path, rec, "l1_rebuilt_prompt.jsonl")
    report = profile_task_a(path)
    assert report.defects == [], report.defects

    # ...but a real assistant-emitted tool call outside tool_schemas is still caught.
    rec["messages"][2]["content"] = (
        '[STATE: A → A]\n<tool_call>{"name": "request_referral", "arguments": {}}</tool_call>'
    )
    path2 = _write(tmp_path, rec, "l1_real_offender.jsonl")
    assert any("not in tool_schemas" in d for d in profile_task_a(path2).defects)
