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
        assert reason == "truncated_after_message_filtering"

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

    def test_malformed_role_message_stripped_conversation_kept(self):
        # Leaked tool-routing syntax (e.g. Harmony-style "to=" channel
        # routing) failing to parse leaves a role field like this, with no
        # content key at all.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "book me a flight"},
            {"role": "assistant", "content": "[STATE: S1 → S2] On it."},
            {"role": "tool", "content": '{"status": "ok"}'},
            {"role": "assistant to=tool"},
            {"role": "assistant to=tool"},
            {"role": "assistant", "content": "[STATE: S2 → S3] Booked."},
        ]
        record = _base_record(messages=msgs)
        cleaned, reason = clean_record(record)
        assert reason is None
        assert cleaned is not None
        roles = [m["role"] for m in cleaned["messages"]]
        assert "assistant to=tool" not in roles
        assert len(cleaned["messages"]) == 5

    def test_malformed_role_garbled_variant_stripped(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant to=book_reservation  garbled token soup"},
            {"role": "assistant", "content": "[STATE: S1 → S2] Done."},
        ]
        record = _base_record(messages=msgs)
        cleaned, reason = clean_record(record)
        assert reason is None
        assert len(cleaned["messages"]) == 3

    def test_malformed_role_is_only_non_system_drops_conversation(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant to=tool"},
        ]
        record = _base_record(messages=msgs)
        cleaned, reason = clean_record(record)
        assert cleaned is None
        assert reason == "truncated_after_message_filtering"

    def test_standard_roles_never_stripped_by_malformed_role_filter(self):
        # Regression guard: the four canonical roles must never be treated
        # as malformed, even when content is missing/empty.
        record = _base_record()
        cleaned, reason = clean_record(record)
        assert reason is None
        assert len(cleaned["messages"]) == len(record["messages"])
