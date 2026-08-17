"""Regression coverage for scripts/mine_model_negatives.py's pure logic.

_classify() and _select_prompts() had no test file before this — both are
reused directly by scripts/investigate_mining_yield.py (Task 4/5), so their
current behavior needs to be locked in before anything downstream depends on
it, and before Task 3 edits main() next to them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from mine_model_negatives import _classify, _select_prompts  # noqa: E402


class TestClassify:
    def test_perfect_match_returns_none(self):
        gt = {
            "tool_calls": [{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            "state_sequence": [{"from": "S1", "to": "S1"}],
        }
        completion = (
            "[STATE: S1 -> S1] "
            '<tool_call>{"name": "book_flight", "arguments": {"dest": "SFO"}}</tool_call>'
        )
        assert _classify(completion, gt) is None

    def test_no_tool_call_when_one_is_expected(self):
        gt = {"tool_calls": [{"name": "book_flight", "arguments": {}}], "state_sequence": []}
        assert _classify("[STATE: S1 -> S1] Sure, one moment.", gt) == "model_no_tool_call"

    def test_wrong_tool_name(self):
        gt = {"tool_calls": [{"name": "book_flight", "arguments": {}}], "state_sequence": []}
        completion = '<tool_call>{"name": "cancel_flight", "arguments": {}}</tool_call>'
        assert _classify(completion, gt) == "model_wrong_tool"

    def test_wrong_arguments_same_tool_name(self):
        gt = {
            "tool_calls": [{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            "state_sequence": [],
        }
        completion = '<tool_call>{"name": "book_flight", "arguments": {"dest": "LAX"}}</tool_call>'
        assert _classify(completion, gt) == "model_wrong_args"

    def test_spurious_call_on_a_turn_needing_none(self):
        gt = {"tool_calls": [], "state_sequence": []}
        completion = '<tool_call>{"name": "book_flight", "arguments": {}}</tool_call>'
        assert _classify(completion, gt) == "model_wrong_tool"

    def test_state_error_when_tool_call_is_correct(self):
        gt = {
            "tool_calls": [{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            "state_sequence": [{"from": "S1", "to": "S2"}],
        }
        completion = (
            "[STATE: S1 -> S1] "
            '<tool_call>{"name": "book_flight", "arguments": {"dest": "SFO"}}</tool_call>'
        )
        assert _classify(completion, gt) == "model_state_error"

    def test_missing_state_annotation_when_one_is_expected(self):
        gt = {"tool_calls": [], "state_sequence": [{"from": "S1", "to": "S2"}]}
        assert _classify("Sure, one moment.", gt) == "model_other"

    def test_no_gt_tools_no_gt_states_and_silent_completion_is_acceptable(self):
        gt = {"tool_calls": [], "state_sequence": []}
        assert _classify("Sure, one moment.", gt) is None


class TestSelectPrompts:
    def _write_grpo_split(self, tmp_path: Path, n_tool: int, n_other: int) -> Path:
        rows = []
        for i in range(n_tool):
            rows.append(self._conv(f"tool-conv-{i}", tool_calls=[{"name": "x", "arguments": {}}]))
        for i in range(n_other):
            rows.append(self._conv(f"other-conv-{i}", tool_calls=[]))
        path = tmp_path / "train.jsonl"
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return tmp_path

    @staticmethod
    def _conv(cid: str, tool_calls: list[dict]) -> dict:
        return {
            "conversation_id": cid,
            "complexity_level": "L2",
            "domain": "banking",
            "workflow_graph": {},
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"hello from {cid}"},
                {
                    "role": "assistant",
                    "content": "[STATE: S1 -> S1] ok",
                    "annotations": {
                        "state_transition": {"from": "S1", "to": "S1"},
                        "tool_calls": tool_calls,
                    },
                },
            ],
            "ground_truth": {
                "state_sequence": [{"from": "S1", "to": "S1"}],
                "terminal_state": "S1",
                "tool_calls": tool_calls,
                "terminal_reached": True,
            },
        }

    def test_respects_tool_share(self, tmp_path):
        data_dir = self._write_grpo_split(tmp_path, n_tool=20, n_other=20)
        picked = _select_prompts(data_dir, "train", n=10, tool_share=0.75, seed=1)
        n_tool_picked = sum(1 for p in picked if json.loads(p["ground_truth"]).get("tool_calls"))
        assert len(picked) == 10
        assert n_tool_picked == 8  # round(10 * 0.75)

    def test_deterministic_given_a_seed(self, tmp_path):
        data_dir = self._write_grpo_split(tmp_path, n_tool=20, n_other=20)
        a = _select_prompts(data_dir, "train", n=10, tool_share=0.5, seed=42)
        b = _select_prompts(data_dir, "train", n=10, tool_share=0.5, seed=42)
        assert a == b

    def test_clamps_to_available_rows_when_n_exceeds_population(self, tmp_path):
        data_dir = self._write_grpo_split(tmp_path, n_tool=2, n_other=2)
        picked = _select_prompts(data_dir, "train", n=100, tool_share=0.75, seed=1)
        assert len(picked) == 4
