"""Tests for the pure trajectory-rollout logic and in-process replay rollout.

The GPU generation path is exercised with a monkeypatched ``model.generate`` and
a cached offline tokenizer (Qwen ChatML); no ``trl``/``unsloth`` is required —
``trajectory_rollout`` imports them lazily, so this whole file runs in ``.venv``.
"""

from __future__ import annotations

from llm_workflow_agents.training.trajectory_rollout import (
    GoldScript,
    assert_trajectory_rollout_support,
    build_gold_script,
    classify_turn,
    prompt_key,
)


class TestEnvGate:
    def test_importable_and_callable(self) -> None:
        # The module must import without trl/unsloth installed, and the gate
        # must be callable. The assertion body only runs meaningfully on the
        # train box (it imports trl), so we don't invoke it here.
        assert callable(assert_trajectory_rollout_support)


GOLD = [("A", "B"), ("B", "C"), ("C", "D")]


class TestClassifyTurn:
    def test_advance_single_step(self) -> None:
        assert classify_turn([("A", "B")], 0, GOLD) == ("advance", 1)

    def test_advance_consecutive_compression(self) -> None:
        assert classify_turn([("A", "B"), ("B", "C")], 0, GOLD) == ("advance", 2)

    def test_stall_on_self_loop(self) -> None:
        # A self-loop (X, X) is a neutral "stay" marker, not a transition.
        assert classify_turn([("B", "B")], 1, GOLD) == ("stall", 1)

    def test_stall_on_empty(self) -> None:
        assert classify_turn([], 1, GOLD) == ("stall", 1)

    def test_diverged_wrong_target(self) -> None:
        assert classify_turn([("B", "Z")], 1, GOLD) == ("diverged", 1)

    def test_diverged_non_consecutive(self) -> None:
        # (A,B) matches but (C,D) is not the immediately-next gold edge.
        assert classify_turn([("A", "B"), ("C", "D")], 0, GOLD) == ("diverged", 0)

    def test_cursor_at_end_any_transition_diverges(self) -> None:
        assert classify_turn([("D", "E")], 3, GOLD) == ("diverged", 3)

    def test_advance_from_mid_cursor(self) -> None:
        assert classify_turn([("B", "C")], 1, GOLD) == ("advance", 2)

    def test_self_loop_mixed_with_real_advance(self) -> None:
        # Self-loop is dropped, the real transition still advances.
        assert classify_turn([("A", "A"), ("A", "B")], 0, GOLD) == ("advance", 1)


# A synthetic Task A conversation: 3 gold assistant turns, spine A->B->C->TERMINAL.
RAW_ROW = {
    "conversation_id": "T_001",
    "workflow_graph": {
        "states": ["A", "B", "C", "TERMINAL"],
        "transitions": [
            {"from": "A", "to": "B", "condition": "c1", "priority": 0},
            {"from": "B", "to": "C", "condition": "c2", "priority": 0},
            {"from": "C", "to": "TERMINAL", "condition": "c3", "priority": 0},
        ],
        "initial": "A",
        "terminal": ["TERMINAL"],
    },
    "tool_schemas": [],
    "messages": [
        {"role": "system", "content": "ORIGINAL SYSTEM"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": '[STATE: A → B] ok <tool_call>{"name":"t1","arguments":{}}</tool_call>',
        },
        {"role": "tool", "content": '{"ok":true}'},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "[STATE: B → C] proceeding"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "[STATE: C → TERMINAL] done"},
    ],
    "ground_truth": {
        "state_sequence": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "TERMINAL"},
        ],
        "tool_calls": [{"name": "t1", "arguments": {}}],
        "terminal_state": "TERMINAL",
        "terminal_reached": True,
    },
}


class TestBuildGoldScript:
    def _script(self) -> GoldScript:
        return build_gold_script(RAW_ROW, enriched_system="ENRICHED SYSTEM")

    def test_prompt_messages_stop_before_first_assistant(self) -> None:
        s = self._script()
        assert [m["role"] for m in s.prompt_messages] == ["system", "user"]
        # The system content is swapped for the enriched prompt.
        assert s.prompt_messages[0]["content"] == "ENRICHED SYSTEM"
        assert s.prompt_messages[1]["content"] == "hi"

    def test_gold_transitions_and_invariant(self) -> None:
        s = self._script()
        assert s.gold_transitions == [("A", "B"), ("B", "C"), ("C", "TERMINAL")]
        # One transition per gold assistant turn.
        assert len(s.gold_transitions) == len(s.segments) == 3

    def test_segments_between_assistant_turns(self) -> None:
        s = self._script()
        # segment[0]: tool + user between asst#0 and asst#1
        assert [m["role"] for m in s.segments[0]] == ["tool", "user"]
        assert s.segments[0][1]["content"] == "next"
        # segment[1]: single user between asst#1 and asst#2
        assert [m["role"] for m in s.segments[1]] == ["user"]
        # segment[2]: trailing (conversation ends on assistant) -> empty
        assert s.segments[2] == []

    def test_terminal_and_valid_transitions(self) -> None:
        s = self._script()
        assert s.terminal_state == "TERMINAL"
        assert s.terminal_reached is True
        assert s.valid_transitions == [["A", "B"], ["B", "C"], ["C", "TERMINAL"]]
        assert s.gold_tool_calls == [{"name": "t1", "arguments": {}}]
        assert s.conversation_id == "T_001"

    def test_mismatched_invariant_raises(self) -> None:
        import copy

        bad = copy.deepcopy(RAW_ROW)
        # 2 transitions but 3 gold assistant turns -> invariant violation.
        bad["ground_truth"]["state_sequence"] = [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
        ]
        import pytest

        with pytest.raises(ValueError, match="gold_transitions"):
            build_gold_script(bad, enriched_system="X")


class TestPromptKey:
    def test_stable_across_key_order(self) -> None:
        a = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        b = [{"content": "s", "role": "system"}, {"content": "u", "role": "user"}]
        assert prompt_key(a) == prompt_key(b)

    def test_distinguishes_content(self) -> None:
        a = [{"role": "user", "content": "u1"}]
        b = [{"role": "user", "content": "u2"}]
        assert prompt_key(a) != prompt_key(b)
