"""Tests for scripts/clean_task_a_sft.py — clean_record()."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import directly without packaging
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from clean_task_a_sft import clean_record  # noqa: E402


def _base_record(**overrides) -> dict:
    r = {
        "conversation_id": "L1_001",
        "complexity_level": "L1",
        "domain": "account_management",
        "workflow_graph": {},
        "messages": [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "[STATE: S1 → S2] Hi there."},
        ],
        "ground_truth": {
            "state_sequence": ["S1", "S2"],
            "terminal_state": "S2",
        },
    }
    r.update(overrides)
    return r


class TestCleanRecord:

    def test_valid_record_passes_through(self):
        record = _base_record()
        cleaned, reason = clean_record(record)
        assert reason is None
        assert cleaned is not None
        assert cleaned["ground_truth"]["terminal_reached"] is True
        assert len(cleaned["messages"]) == 3

    def test_truncated_row_dropped(self):
        record = _base_record(messages=[{"role": "system", "content": "sys"}])
        cleaned, reason = clean_record(record)
        assert cleaned is None
        assert reason == "truncated_no_non_system_turns"

    def test_role_confused_tool_message_removed_conversation_kept(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
            {"role": "tool", "content": "<tool_call>{}</tool_call>"},  # confused
            {"role": "assistant", "content": "[STATE: S1 → S2] Done."},
        ]
        record = _base_record(messages=msgs)
        cleaned, reason = clean_record(record)
        assert reason is None
        assert cleaned is not None
        roles = [m["role"] for m in cleaned["messages"]]
        assert "tool" not in roles
        assert len(cleaned["messages"]) == 4

    def test_role_confused_is_only_non_system_drops_conversation(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "<tool_call>{}</tool_call>"},
        ]
        record = _base_record(messages=msgs)
        cleaned, reason = clean_record(record)
        assert cleaned is None
        assert reason == "truncated_after_role_confusion_filter"

    def test_empty_terminal_state_flagged_false(self):
        record = _base_record()
        record["ground_truth"]["terminal_state"] = ""
        cleaned, reason = clean_record(record)
        assert reason is None
        assert cleaned["ground_truth"]["terminal_reached"] is False

    def test_none_terminal_state_flagged_false(self):
        record = _base_record()
        record["ground_truth"]["terminal_state"] = None
        cleaned, reason = clean_record(record)
        assert reason is None
        assert cleaned["ground_truth"]["terminal_reached"] is False

    def test_nonempty_terminal_state_flagged_true(self):
        record = _base_record()
        record["ground_truth"]["terminal_state"] = "S_RESOLVED"
        cleaned, reason = clean_record(record)
        assert reason is None
        assert cleaned["ground_truth"]["terminal_reached"] is True

    def test_clean_tool_response_left_alone(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "order status?"},
            {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
            {"role": "tool", "content": '{"status": "shipped"}'},
        ]
        record = _base_record(messages=msgs)
        cleaned, reason = clean_record(record)
        assert reason is None
        # The well-formed tool message should be preserved
        assert any(m["role"] == "tool" for m in cleaned["messages"])
