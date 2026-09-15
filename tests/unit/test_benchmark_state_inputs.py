"""Whole-run state metrics must score each conversation against its own ground truth.

The Task A text and voice benchmark strata both number their conversations
L1_001, L1_002, ... ``evaluate_state_machine`` looks ground truth up by
``conversation_id``, so a two-stratum run scored every text prediction against
the voice conversation that shared its id. On the 12B SFT run that reported
state sequence accuracy 0.6476 where the per-row figure is 0.9132.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.agent_benchmark import build_state_machine_inputs
from llm_workflow_agents.eval.state_accuracy import evaluate_state_machine


def _sample(conversation_id: str, modality: str, states: list[str], terminal: str) -> dict:
    messages = [{"role": "user", "content": "hi"}]
    for a, b in zip(states, states[1:]):
        messages.append({
            "role": "assistant",
            "content": f"[STATE: {a} → {b}]",
            "annotations": {"state_transition": {"from": a, "to": b}},
        })
    return {
        "conversation_id": conversation_id,
        "modality": modality,
        "messages": messages,
        "ground_truth": {"terminal_state": terminal},
    }


def _replay_perfectly(sample: dict) -> list[dict]:
    return [
        {"role": m["role"], "content": m["content"]} for m in sample["messages"]
    ]


# Same id, different workflows: the collision only shows when the two ground
# truths disagree.
TEXT = _sample("L1_001", "text", ["GREETING", "VERIFY", "TERMINAL"], "TERMINAL")
VOICE = _sample("L1_001", "voice", ["WELCOME", "BOOK", "CONFIRM", "END_CALL"], "END_CALL")


class TestBuildStateMachineInputs:
    def test_keys_are_unique_when_strata_share_an_id(self):
        preds, gts = build_state_machine_inputs([TEXT, VOICE], [[], []])
        assert len({p.conversation_id for p in preds}) == 2
        assert [p.conversation_id for p in preds] == [g.conversation_id for g in gts]

    def test_each_row_is_scored_against_its_own_ground_truth(self):
        samples = [TEXT, VOICE]
        preds, gts = build_state_machine_inputs(samples, [_replay_perfectly(s) for s in samples])
        metrics = evaluate_state_machine(preds, gts)
        assert metrics.task_completion_rate == 1.0
        assert metrics.state_sequence_accuracy == 1.0
        assert metrics.state_transition_accuracy == 1.0

    def test_keying_by_raw_id_is_refused(self):
        """The pre-fix pairing now fails loudly instead of mis-scoring."""
        preds, gts = build_state_machine_inputs([TEXT, VOICE], [[], []])
        for p, g in zip(preds, gts):
            p.conversation_id = g.conversation_id = "L1_001"
        with pytest.raises(ValueError, match="duplicate ground-truth conversation_id"):
            evaluate_state_machine(preds, gts)

    def test_terminal_state_comes_from_the_sample(self):
        _, gts = build_state_machine_inputs([TEXT, VOICE], [[], []])
        assert [g.terminal_states for g in gts] == [["TERMINAL"], ["END_CALL"]]

    def test_missing_terminal_state_gives_no_terminal_states(self):
        sample = {"conversation_id": "x", "messages": [], "ground_truth": {}}
        _, gts = build_state_machine_inputs([sample], [[]])
        assert gts[0].terminal_states == []

    def test_missing_conversation_id_still_gets_a_unique_key(self):
        preds, _ = build_state_machine_inputs([{"messages": []}, {"messages": []}], [[], []])
        assert len({p.conversation_id for p in preds}) == 2

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="2 samples but 1 predicted"):
            build_state_machine_inputs([TEXT, VOICE], [[]])
