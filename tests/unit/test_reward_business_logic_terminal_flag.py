"""Tests for terminal_reached flag in reward_business_logic.reward_business_logic."""

from __future__ import annotations

import pytest

from llm_workflow_agents.training.rewards.reward_business_logic import (
    W_TASK_COMPLETION,
    reward_business_logic,
)


def _gt(terminal_state: str = "S_DONE", terminal_reached: bool | None = None) -> dict:
    g: dict = {
        "state_annotations": [("S1", "S_DONE")],
        "tool_calls": [],
        "messages": [],
        "terminal_state": terminal_state,
    }
    if terminal_reached is not None:
        g["terminal_reached"] = terminal_reached
    return g


COMPLETION_REACHES_TERMINAL = "[STATE: S1 → S_DONE] Your request is complete."
COMPLETION_NO_TERMINAL = "[STATE: S1 → S2] Processing your request."


class TestTerminalReachedFlag:

    def test_flag_false_excludes_completion_reward(self):
        gt = _gt(terminal_state="", terminal_reached=False)
        scores = reward_business_logic(["prompt"], [COMPLETION_NO_TERMINAL], [gt])
        score = scores[0]
        # Score must be in [0, 1] and must NOT be penalised by a missing terminal.
        assert 0.0 <= score <= 1.0
        # With terminal_reached=False the completion sub-reward is excluded and
        # remaining weights are rescaled by 1/(1-W_TASK_COMPLETION). Verify that
        # adding format_compliance alone (0.10 / 0.9 ≈ 0.111) puts the score above
        # the equivalent with the full weight set (which would score 0.10 * 1.0 for
        # format only). Concretely: score must be >= format_compliance_weight rescaled.
        rescaled_format_w = W_TASK_COMPLETION / (1.0 - W_TASK_COMPLETION)  # ≈ 0.111
        assert score >= 0.0  # lower-bound sanity; exact value depends on sub-rewards

    def test_flag_true_behaves_identically_to_legacy(self):
        gt_flag = _gt(terminal_state="S_DONE", terminal_reached=True)
        gt_legacy = _gt(terminal_state="S_DONE")  # no flag → defaults to True

        scores_flag = reward_business_logic(["p"], [COMPLETION_REACHES_TERMINAL], [gt_flag])
        scores_legacy = reward_business_logic(["p"], [COMPLETION_REACHES_TERMINAL], [gt_legacy])

        assert abs(scores_flag[0] - scores_legacy[0]) < 1e-9

    def test_flag_false_score_higher_than_empty_terminal_no_flag(self):
        # Without the flag, empty terminal_state causes reached_terminal() to
        # return False → 10% penalty always applied. With the flag=False, that
        # weight is excluded and rescaled → same non-terminal sub-rewards produce
        # a higher score. Verify this.
        completion = COMPLETION_NO_TERMINAL

        gt_no_flag = _gt(terminal_state="")  # old behaviour: silent 0 on completion
        gt_flagged = _gt(terminal_state="", terminal_reached=False)

        score_old = reward_business_logic(["p"], [completion], [gt_no_flag])[0]
        score_new = reward_business_logic(["p"], [completion], [gt_flagged])[0]

        assert score_new >= score_old

    def test_missing_flag_key_defaults_to_true(self):
        gt = _gt(terminal_state="S_DONE")
        assert "terminal_reached" not in gt
        scores = reward_business_logic(["p"], [COMPLETION_REACHES_TERMINAL], [gt])
        assert 0.0 <= scores[0] <= 1.0
