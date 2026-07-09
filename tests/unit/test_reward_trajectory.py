"""Known-answer tests for the trajectory-level Cat A reward + its TRL adapter.

Pure CPU; no trl/unsloth needed (grpo imports them lazily inside functions).
"""

from __future__ import annotations

import json

import pytest

from llm_workflow_agents.training.rewards.reward_business_logic import (
    reward_business_logic_trajectory,
)

# 4-transition gold spine A->B->C->D->E, no gold tools.
GOLD4 = {
    "state_sequence": [
        {"from": "A", "to": "B"},
        {"from": "B", "to": "C"},
        {"from": "C", "to": "D"},
        {"from": "D", "to": "E"},
    ],
    "tool_calls": [],
    "terminal_state": "E",
    "terminal_reached": True,
    "valid_transitions": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]],
}


def _score(traj, meta, gt) -> float:
    (s,) = reward_business_logic_trajectory([None], [traj], [meta], [gt])
    return s


class TestTrajectoryReward:
    def test_partial_coverage_no_tools_known_answer(self) -> None:
        # Traverses A->B, B->C, then diverges to C->Z.
        traj = ["[STATE: A → B]", "[STATE: B → C]", "[STATE: C → Z]"]
        meta = {"stop_reason": "diverged", "n_model_turns": 3, "n_stall_turns": 0}
        # coverage 2/4=0.5; tool F1 empty/empty=1.0; terminal 0 (not gold_complete);
        # legality: (A,B),(B,C) legal + (C,Z) illegal = 2/3, stall_factor 1.0.
        expected = 0.40 * 0.5 + 0.40 * 1.0 + 0.10 * 0.0 + 0.10 * (2 / 3)
        assert abs(_score(traj, meta, GOLD4) - expected) < 1e-9

    def test_full_traversal_scores_one(self) -> None:
        gt = {
            "state_sequence": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
                {"from": "C", "to": "TERMINAL"},
            ],
            "tool_calls": [],
            "terminal_state": "TERMINAL",
            "terminal_reached": True,
            "valid_transitions": [["A", "B"], ["B", "C"], ["C", "TERMINAL"]],
        }
        traj = ["[STATE: A → B]", "[STATE: B → C]", "[STATE: C → TERMINAL]"]
        meta = {"stop_reason": "gold_complete", "n_model_turns": 3, "n_stall_turns": 0}
        assert _score(traj, meta, gt) == 1.0

    def test_order_sensitivity(self) -> None:
        ordered = ["[STATE: A → B]", "[STATE: B → C]"]
        permuted = ["[STATE: B → C]", "[STATE: A → B]"]
        meta = {"stop_reason": "diverged", "n_model_turns": 2, "n_stall_turns": 0}
        # ordered advances cursor to 2 (0.5 coverage); permuted diverges then
        # advances once (cursor 1, 0.25 coverage) -> strictly lower.
        assert _score(ordered, meta, GOLD4) > _score(permuted, meta, GOLD4)

    def test_coverage_capped_at_one(self) -> None:
        gt = {
            "state_sequence": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
            ],
            "tool_calls": [],
            "terminal_state": "C",
            "terminal_reached": True,
            "valid_transitions": [["A", "B"], ["B", "C"]],
        }
        # Emits one extra transition past the end of the gold spine.
        traj = ["[STATE: A → B]", "[STATE: B → C]", "[STATE: C → D]"]
        meta = {"stop_reason": "diverged", "n_model_turns": 3, "n_stall_turns": 0}
        # coverage capped at 1.0; terminal 0 (diverged); legality 2/3.
        expected = 0.40 * 1.0 + 0.40 * 1.0 + 0.10 * 0.0 + 0.10 * (2 / 3)
        assert abs(_score(traj, meta, gt) - expected) < 1e-9

    def test_terminal_renorm_when_not_reached(self) -> None:
        traj = ["[STATE: A → B]", "[STATE: B → C]"]  # 0.5 coverage of GOLD4
        meta = {"stop_reason": "diverged", "n_model_turns": 2, "n_stall_turns": 0}
        s_true = _score(traj, meta, {**GOLD4, "terminal_reached": True})
        s_false = _score(traj, meta, {**GOLD4, "terminal_reached": False})
        # terminal_reached=True keeps a 0-valued terminal component (drags down);
        # False drops it and renormalizes the remaining weights up.
        assert s_false > s_true

    def test_stall_factor_reduces_legality(self) -> None:
        gt = {
            "state_sequence": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
            ],
            "tool_calls": [],
            "terminal_state": "C",
            "terminal_reached": False,  # drop terminal to isolate the legality term
            "valid_transitions": [["A", "B"], ["B", "C"]],
        }
        no_stall = ["[STATE: A → B]", "[STATE: B → C]"]
        with_stall = ["[STATE: A → B]", "(no transition here)", "[STATE: B → C]"]
        m_ns = {"stop_reason": "gold_complete", "n_model_turns": 2, "n_stall_turns": 0}
        m_ws = {"stop_reason": "gold_complete", "n_model_turns": 3, "n_stall_turns": 1}
        assert _score(with_stall, m_ws, gt) < _score(no_stall, m_ns, gt)


class TestTrajectoryAdapter:
    def test_decodes_extra_fields(self) -> None:
        from llm_workflow_agents.training.grpo import _make_trajectory_reward_adapter

        captured: dict = {}

        def fake_reward(prompts, trajectories, metas, gts):
            captured.update(
                prompts=prompts, trajectories=trajectories, metas=metas, gts=gts
            )
            return [0.5] * len(trajectories)

        adapter = _make_trajectory_reward_adapter(fake_reward)
        out = adapter(
            prompts=["p1"],
            completions=[[{"role": "assistant", "content": "whole interleaved stream"}]],
            completion_ids=[[1, 2, 3]],
            ground_truth=[json.dumps({"state_sequence": [{"from": "A", "to": "B"}]})],
            trajectory=[json.dumps(["[STATE: A → B]"])],
            rollout_meta=[json.dumps({"stop_reason": "diverged", "n_model_turns": 1})],
        )
        assert out == [0.5]
        assert captured["trajectories"] == [["[STATE: A → B]"]]
        assert captured["metas"][0]["stop_reason"] == "diverged"
        assert captured["gts"][0]["state_sequence"] == [{"from": "A", "to": "B"}]

    def test_requires_trajectory_field(self) -> None:
        from llm_workflow_agents.training.grpo import _make_trajectory_reward_adapter

        adapter = _make_trajectory_reward_adapter(lambda *a: [])
        with pytest.raises(ValueError, match="trajectory"):
            adapter(prompts=["p"], completions=["c"], ground_truth=["{}"])

    def test_registry_resolves_trajectory_reward(self) -> None:
        from llm_workflow_agents.training.grpo import _resolve_reward_fn

        fn = _resolve_reward_fn("reward_business_logic_trajectory")
        assert fn is reward_business_logic_trajectory
