"""Tests for data validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.data._workflow_script import find_tool_placement_violations
from llm_workflow_agents.data.data_validator import (
    detect_thai_corruption,
    validate_dataset,
)


@pytest.fixture
def valid_workflow_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "valid_workflow.jsonl"
    sample = {
        "conversation_id": "L1_001",
        "complexity_level": "L1",
        "domain": "faq_lookup",
        "workflow_graph": {
            "states": ["S1", "S2", "S3"],
            "transitions": [
                {"from": "S1", "to": "S2", "condition": "proceed"},
                {"from": "S2", "to": "S3", "condition": "proceed"},
            ],
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


def _rational_workflow_sample() -> dict:
    """A small, fully coherent workflow sample used as the rationality baseline."""
    return {
        "conversation_id": "L1_001",
        "complexity_level": "L1",
        "domain": "telecom",
        "workflow_graph": {
            "states": ["GREETING", "PROCESS_CHANGE", "TERMINAL"],
            "state_details": [
                {"name": "GREETING", "tools": [], "instruction": "Greet the customer."},
                {"name": "PROCESS_CHANGE", "tools": ["change_plan"],
                 "instruction": "Apply the plan change."},
                {"name": "TERMINAL", "tools": [], "instruction": "Close the conversation."},
            ],
            "transitions": [
                {"from": "GREETING", "to": "PROCESS_CHANGE", "condition": "proceed"},
                {"from": "PROCESS_CHANGE", "to": "TERMINAL", "condition": "proceed"},
            ],
            "initial": "GREETING",
            "terminal": ["TERMINAL"],
        },
        "tool_schemas": [
            {"type": "function", "function": {"name": "change_plan", "description": "Change plan"}},
        ],
        "messages": [
            {"role": "system", "content": "You are a telecom agent."},
            {"role": "user", "content": "Change my plan"},
            {
                "role": "assistant",
                "content": "[STATE: GREETING → PROCESS_CHANGE]\nHappy to help with that.",
            },
            {"role": "user", "content": "Great, go ahead."},
            {
                "role": "assistant",
                "content": "[STATE: PROCESS_CHANGE → PROCESS_CHANGE]\n"
                           "<tool_call>{\"name\": \"change_plan\", \"arguments\": {}}</tool_call>",
            },
            {"role": "tool", "content": "{\"status\": \"success\"}"},
            {
                "role": "assistant",
                "content": "[STATE: PROCESS_CHANGE → TERMINAL]\nAll done!",
            },
        ],
    }


class TestWorkflowRationality:

    def _write(self, tmp_path: Path, sample: dict) -> Path:
        path = tmp_path / "rational.jsonl"
        path.write_text(json.dumps(sample) + "\n")
        return path

    def test_rational_sample_is_valid(self, tmp_path: Path) -> None:
        result = validate_dataset(self._write(tmp_path, _rational_workflow_sample()), "workflow")
        assert result.valid, result.errors

    def test_tool_not_in_schemas_is_invalid(self, tmp_path: Path) -> None:
        sample = _rational_workflow_sample()
        sample["workflow_graph"]["state_details"][1]["tools"] = ["unlock_device"]
        result = validate_dataset(self._write(tmp_path, sample), "workflow")
        assert not result.valid
        assert any("unlock_device" in e and "tool_schemas" in e for e in result.errors)

    def test_gt_tool_not_listed_is_invalid(self, tmp_path: Path) -> None:
        sample = _rational_workflow_sample()
        # State PROCESS_CHANGE no longer lists the tool the GT conversation calls.
        sample["workflow_graph"]["state_details"][1]["tools"] = []
        result = validate_dataset(self._write(tmp_path, sample), "workflow")
        assert not result.valid
        assert any("change_plan" in e and "not listed" in e for e in result.errors)

    def test_missing_instruction_is_invalid(self, tmp_path: Path) -> None:
        sample = _rational_workflow_sample()
        sample["workflow_graph"]["state_details"][1]["instruction"] = ""
        result = validate_dataset(self._write(tmp_path, sample), "workflow")
        assert not result.valid
        assert any("no instruction" in e for e in result.errors)

    def test_unreachable_terminal_is_invalid(self, tmp_path: Path) -> None:
        sample = _rational_workflow_sample()
        # Drop the transition into TERMINAL → terminal unreachable.
        sample["workflow_graph"]["transitions"] = [
            {"from": "GREETING", "to": "PROCESS_CHANGE", "condition": "proceed"},
        ]
        result = validate_dataset(self._write(tmp_path, sample), "workflow")
        assert not result.valid
        assert any("reachable" in e for e in result.errors)


class TestValidatorContinuityAndShape:
    """The validator enforces sequence continuity and conversation shape via
    the shared checkers (same source of truth as the generator repair loop)."""

    def _write(self, tmp_path: Path, sample: dict) -> Path:
        path = tmp_path / "continuity.jsonl"
        path.write_text(json.dumps(sample) + "\n")
        return path

    def test_discontinuous_annotations_are_invalid(self, tmp_path: Path) -> None:
        sample = _rational_workflow_sample()
        # Skip the GREETING → PROCESS_CHANGE hop: every remaining edge is
        # legal, but the sequence no longer chains from the initial state.
        del sample["messages"][1:3]
        result = validate_dataset(self._write(tmp_path, sample), "workflow")
        assert not result.valid
        assert any("initial state" in e for e in result.errors)

    def test_nonterminal_ending_is_invalid(self, tmp_path: Path) -> None:
        sample = _rational_workflow_sample()
        # Drop the closing turn → the conversation ends at PROCESS_CHANGE.
        sample["messages"] = sample["messages"][:-2]
        result = validate_dataset(self._write(tmp_path, sample), "workflow")
        assert not result.valid
        assert any("not a terminal state" in e for e in result.errors)

    def test_mid_message_marker_is_invalid(self, tmp_path: Path) -> None:
        sample = _rational_workflow_sample()
        sample["messages"][-1]["content"] += " [STATE: TERMINAL → TERMINAL] bye"
        result = validate_dataset(self._write(tmp_path, sample), "workflow")
        assert not result.valid
        assert any("mid-content" in e for e in result.errors)

    def test_inbound_assistant_first_is_invalid(self, tmp_path: Path) -> None:
        sample = _rational_workflow_sample()
        sample["messages"].insert(
            1,
            {
                "role": "assistant",
                "content": "[STATE: GREETING → GREETING]\nHow can I help?",
            },
        )
        result = validate_dataset(self._write(tmp_path, sample), "workflow")
        assert not result.valid
        assert any("first message role" in e for e in result.errors)


class TestFindToolPlacementViolations:
    """Unit tests for the shared coherence helper used by the validator and the
    generator's inline repair loop."""

    def _msg(self, state_from: str, state_to: str, tool: str) -> dict:
        return {
            "role": "assistant",
            "content": f"[STATE: {state_from} → {state_to}]\n"
                       f"<tool_call>{{\"name\": \"{tool}\", \"arguments\": {{}}}}</tool_call>",
        }

    def test_coherent_conversation_has_no_violations(self) -> None:
        allowed = {"PROCESS_CHANGE": {"change_plan"}}
        messages = [self._msg("PROCESS_CHANGE", "TERMINAL", "change_plan")]
        assert find_tool_placement_violations(allowed, messages, {"change_plan"}) == []

    def test_tool_in_disallowed_state_is_flagged(self) -> None:
        allowed = {"VERIFY_ACCOUNT": set(), "PROCESS_CHANGE": {"change_plan"}}
        messages = [self._msg("VERIFY_ACCOUNT", "PROCESS_CHANGE", "change_plan")]
        violations = find_tool_placement_violations(allowed, messages, {"change_plan"})
        assert len(violations) == 1
        assert "change_plan" in violations[0] and "VERIFY_ACCOUNT" in violations[0]

    def test_off_schema_tool_is_flagged(self) -> None:
        allowed = {"PROCESS_CHANGE": {"change_plan"}}
        messages = [self._msg("PROCESS_CHANGE", "TERMINAL", "ghost_tool")]
        violations = find_tool_placement_violations(allowed, messages, {"change_plan"})
        assert len(violations) == 1
        assert "ghost_tool" in violations[0] and "tool_schemas" in violations[0]

    def test_unknown_state_is_skipped(self) -> None:
        # State not in the allowed map has no curated expectation → no violation
        # (when the tool is also in schemas).
        allowed = {"PROCESS_CHANGE": {"change_plan"}}
        messages = [self._msg("MYSTERY_STATE", "TERMINAL", "change_plan")]
        assert find_tool_placement_violations(allowed, messages, {"change_plan"}) == []


class TestDetectThaiCorruption:
    """Conservative two-signal detector for teacher-corrupted Thai prose.
    Snippets are taken from the real L5_026 / L5_062 corruption."""

    @staticmethod
    def _rec(content: str, *, language: str = "th", role: str = "assistant") -> dict:
        return {
            "conversation_id": "L5_001",
            "language": language,
            "messages": [
                {"role": "system", "content": "ระบบ"},
                {"role": role, "content": content},
            ],
        }

    def test_latin_glued_into_thai_flagged(self) -> None:
        # From L5_026: Latin "locks" injected mid-word.
        reasons = self._rec("ครับคุณลูกค้า ผมเป๋locksตัวแทน")
        assert any("latin-glued-into-thai" in r for r in detect_thai_corruption(reasons))

    def test_obsolete_consonant_flagged(self) -> None:
        # ฅ (kho khon) — from L5_026 ("ฅุณ") and L5_062 ("ขฅง").
        assert any("obsolete" in r for r in detect_thai_corruption(self._rec("ฅุณโทรมาจากไหน")))
        assert any("obsolete" in r for r in detect_thai_corruption(self._rec("โอนไปที่เบอร์ขฅงคุณ")))

    def test_clean_thai_passes(self) -> None:
        assert detect_thai_corruption(self._rec("สวัสดีค่ะ ดิฉันเป็นผู้ช่วยจากธนาคาร")) == []

    def test_clause_boundary_thai_english_concat_passes(self) -> None:
        # Legitimate Thai+English concatenation at a clause boundary (English word
        # followed by a space, not glued *inside* a Thai word). The placeholder
        # generator emits this shape ("...เรื่องhotel booking") — it must NOT be
        # flagged, or the auto-drop gate would discard all such valid rows.
        assert detect_thai_corruption(
            self._rec("ต้องการความช่วยเหลือเรื่องhotel booking")
        ) == []
        assert detect_thai_corruption(self._rec("ขอสอบถามเรื่องcar rental ค่ะ")) == []

    def test_code_switch_allows_latin_mixing(self) -> None:
        # Identical Latin-in-Thai string: corruption under `th`, legitimate
        # script-mixing under `code_switch` (Signal A gated off).
        s = "ผมเป๋locksตัวแทน"
        assert detect_thai_corruption(self._rec(s, language="th"))
        assert detect_thai_corruption(self._rec(s, language="code_switch")) == []

    def test_code_switch_still_flags_obsolete_consonant(self) -> None:
        # Signal B applies regardless of language.
        assert detect_thai_corruption(self._rec("เบอร์ขฅงคุณ", language="code_switch"))

    def test_english_record_is_noop(self) -> None:
        assert detect_thai_corruption(
            self._rec("Hello, how can I help you today?", language="en")
        ) == []

    def test_system_message_is_ignored(self) -> None:
        # Corruption only in the machine-authored system prompt is not flagged.
        rec = {
            "conversation_id": "L5_001",
            "language": "th",
            "messages": [
                {"role": "system", "content": "ผมเป๋locksตัวแทน"},
                {"role": "assistant", "content": "สวัสดีค่ะ"},
            ],
        }
        assert detect_thai_corruption(rec) == []


class TestValidateDatasetCorruptionWarning:
    def test_corruption_is_warning_not_error(self, tmp_path: Path) -> None:
        # Schema-valid sample (mirrors the valid_workflow fixture) with corrupted
        # Thai in the user turn: stays valid, surfaces a warning.
        sample = {
            "conversation_id": "L5_026",
            "complexity_level": "L1",
            "domain": "faq_lookup",
            "language": "th",
            "workflow_graph": {
                "states": ["S1", "S2", "S3"],
                "transitions": [
                    {"from": "S1", "to": "S2", "condition": "proceed"},
                    {"from": "S2", "to": "S3", "condition": "proceed"},
                ],
                "initial": "S1",
                "terminal": ["S3"],
            },
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "ผมเป๋locksตัวแทน"},
            ],
        }
        path = tmp_path / "corrupt.jsonl"
        path.write_text(json.dumps(sample, ensure_ascii=False) + "\n")
        result = validate_dataset(path, "workflow")
        assert result.valid is True
        assert not any("thai-corruption" in e for e in result.errors)
        assert any("thai-corruption" in w and "L5_026" in w for w in result.warnings)
        assert result.stats["warning_count"] >= 1
