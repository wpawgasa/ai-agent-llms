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


class TestExcludedFingerprints:
    def _write_validation_split(self, tmp_path: Path, n: int) -> Path:
        rows = [TestSelectPrompts._conv(f"c{i}", tool_calls=[]) for i in range(n)]
        path = tmp_path / "validation.jsonl"
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return tmp_path

    def test_train_split_has_no_guardrail_exclusion(self, tmp_path):
        from mine_model_negatives import _excluded_fingerprints

        data_dir = self._write_validation_split(tmp_path, n=20)
        excluded = _excluded_fingerprints(
            data_dir,
            split="train",
            heldout=[],
            guardrail_reserved_fraction=0.2,
            guardrail_reserved_seed=42,
        )
        assert excluded == set()

    def test_validation_split_excludes_the_reserved_slice(self, tmp_path):
        from mine_model_negatives import _excluded_fingerprints

        data_dir = self._write_validation_split(tmp_path, n=20)
        excluded = _excluded_fingerprints(
            data_dir,
            split="validation",
            heldout=[],
            guardrail_reserved_fraction=0.2,
            guardrail_reserved_seed=42,
        )
        assert len(excluded) > 0

        from llm_workflow_agents.data.heldout_clean_set import reserve_guardrail_slice

        assert excluded == reserve_guardrail_slice(
            data_dir, split="validation", reserved_fraction=0.2, seed=42
        )

    def test_validation_split_combines_heldout_and_guardrail_exclusions(self, tmp_path):
        from mine_model_negatives import _excluded_fingerprints
        from llm_workflow_agents.data.heldout_clean_set import user_turn_fingerprint

        data_dir = self._write_validation_split(tmp_path, n=20)
        heldout_path = tmp_path / "heldout_test.jsonl"
        heldout_conv = TestSelectPrompts._conv("heldout-c", tool_calls=[])
        with open(heldout_path, "w") as fh:
            fh.write(json.dumps(heldout_conv) + "\n")

        excluded = _excluded_fingerprints(
            data_dir,
            split="validation",
            heldout=[heldout_path],
            guardrail_reserved_fraction=0.2,
            guardrail_reserved_seed=42,
        )
        heldout_fp = user_turn_fingerprint(
            {"messages": [{"role": "user", "content": "hello from heldout-c"}]}
        )
        assert heldout_fp in excluded


class TestSelectPromptsPrefersToolBearingTurns:
    """One row per conversation must be the TOOL-BEARING turn when one exists.

    MEASURED (CLAUDE.md R22 follow-up, 2026-08-30): asking for
    `--tool-share 0.75` realised **0.13 on train and 0.02 on validation**. The
    old dedupe kept whichever turn it met first per conversation and skipped
    the rest, and a conversation's opening assistant turn is almost never a
    tool call — so `tool_rows` was starved before the share was ever applied.

    That single line is why the first mining run reported C2 wrong on 12.8% of
    prompts: the sample was 87% easy no-tool rows at 6% wrong, diluting a
    tool-bearing rate of 67.4%. The scarce, hard, on-distribution negatives DPO
    needed were in the data all along and never got sampled — which is the
    direct cause of R22's null result, where the pairs were too easy and the
    gradient vanished.

    The existing TestSelectPrompts fixtures use single-turn conversations, so
    the defect cannot appear there. These use multi-turn ones.
    """

    @staticmethod
    def _multi_turn_conv(cid: str, tool_calls: list[dict]) -> dict:
        """A conversation whose FIRST assistant turn has no tool call and whose
        SECOND does — the shape that real Task A data overwhelmingly takes."""
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
                    "content": "[STATE: S1 -> S1] let me check that",
                    "annotations": {
                        "state_transition": {"from": "S1", "to": "S1"},
                        "tool_calls": [],
                    },
                },
                {"role": "user", "content": "go ahead"},
                {
                    "role": "assistant",
                    "content": "[STATE: S1 -> S1] calling",
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

    def _write(self, tmp_path: Path, n_tool: int, n_other: int) -> Path:
        rows = [
            self._multi_turn_conv(f"tool-conv-{i}", [{"name": "x", "arguments": {}}])
            for i in range(n_tool)
        ] + [self._multi_turn_conv(f"other-conv-{i}", []) for i in range(n_other)]
        with open(tmp_path / "train.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return tmp_path

    @staticmethod
    def _has_tool_gt(row: dict) -> bool:
        return bool(json.loads(row["ground_truth"]).get("tool_calls"))

    def test_tool_share_is_reachable_on_multi_turn_conversations(self, tmp_path):
        """The defect: 40 conversations each contain a tool turn, yet the old
        sampler returned almost none, because it kept turn 1 and skipped turn 2."""
        data_dir = self._write(tmp_path, n_tool=40, n_other=40)
        picked = _select_prompts(data_dir, "train", n=40, tool_share=0.75, seed=1)
        n_tool = sum(1 for r in picked if self._has_tool_gt(r))
        assert n_tool == 30, (
            f"asked for 0.75 of 40 = 30 tool-bearing rows, got {n_tool}. "
            "The sampler must pick a conversation's tool-bearing turn, not "
            "whichever turn it meets first."
        )

    def test_still_one_row_per_conversation(self, tmp_path):
        """Preferring the tool turn must not reintroduce near-duplicate prompts."""
        data_dir = self._write(tmp_path, n_tool=10, n_other=10)
        picked = _select_prompts(data_dir, "train", n=20, tool_share=0.5, seed=1)
        firsts = [
            next(m["content"] for m in r["prompt_messages"] if m["role"] == "user")
            for r in picked
        ]
        assert len(firsts) == len(set(firsts)), "a conversation appeared twice"

    def test_conversations_without_any_tool_turn_still_contribute(self, tmp_path):
        data_dir = self._write(tmp_path, n_tool=5, n_other=15)
        picked = _select_prompts(data_dir, "train", n=20, tool_share=0.25, seed=1)
        assert sum(1 for r in picked if not self._has_tool_gt(r)) > 0

    def test_still_deterministic_given_a_seed(self, tmp_path):
        data_dir = self._write(tmp_path, n_tool=20, n_other=20)
        a = _select_prompts(data_dir, "train", n=20, tool_share=0.75, seed=42)
        b = _select_prompts(data_dir, "train", n=20, tool_share=0.75, seed=42)
        assert [r["prompt_messages"] for r in a] == [r["prompt_messages"] for r in b]
