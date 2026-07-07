"""Tests for the pure logic of the RFT Stage-0 headroom probe.

The GPU generation path is not exercised here (no model); these lock the
reduction (summarize_headroom) and the gate classifier (classify_gate) that
decide GO_RFT / GRPO_REVIVAL / NO_GO / MARGINAL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rft_headroom_probe import classify_gate, summarize_headroom  # noqa: E402


class TestSummarizeHeadroom:
    def test_basic_two_prompts(self) -> None:
        # p0: best-of-3 (1.0) beats greedy (0.3) → headroom 0.7, diverse.
        # p1: collapsed group, greedy == best → headroom 0.
        s = summarize_headroom(
            sampled_rewards=[[0.3, 0.5, 1.0], [0.4, 0.4, 0.4]],
            greedy_rewards=[0.3, 0.4],
        )
        assert s["n_prompts"] == 2
        assert s["frontier_frac"] == pytest.approx(0.5)  # only p0 on frontier
        assert s["mean_headroom"] == pytest.approx(0.35)  # (0.7 + 0.0)/2
        assert s["frac_collapsed_groups"] == pytest.approx(0.5)  # p1 std==0
        assert s["rung_histogram"] == {1: 1, 3: 1}
        assert s["per_prompt"][0]["on_frontier"] is True
        assert s["per_prompt"][1]["on_frontier"] is False

    def test_headroom_can_be_negative(self) -> None:
        # greedy beats every sample → negative headroom, not on frontier.
        s = summarize_headroom(sampled_rewards=[[0.2, 0.3]], greedy_rewards=[0.6])
        assert s["per_prompt"][0]["headroom"] == pytest.approx(-0.3)
        assert s["frontier_frac"] == 0.0
        assert s["frac_positive_headroom"] == 0.0

    def test_empty(self) -> None:
        s = summarize_headroom([], [])
        assert s["frontier_frac"] == 0.0
        assert s["mean_headroom"] == 0.0


class TestClassifyGate:
    def _summary(self, frontier, mean_hr, collapsed, med_std):
        return {
            "frontier_frac": frontier,
            "mean_headroom": mean_hr,
            "frac_collapsed_groups": collapsed,
            "median_reward_std": med_std,
        }

    def test_go_rft(self) -> None:
        v, _ = classify_gate(self._summary(0.20, 0.05, 0.7, 0.01))
        assert v == "GO_RFT"

    def test_grpo_revival_takes_precedence_over_nogo_but_not_rft(self) -> None:
        # Not enough frontier for RFT, but GRPO condition met → GRPO_REVIVAL.
        v, _ = classify_gate(self._summary(0.12, 0.02, 0.30, 0.06))
        assert v == "GRPO_REVIVAL"

    def test_no_go(self) -> None:
        v, _ = classify_gate(self._summary(0.05, 0.00, 1.0, 0.0))
        assert v == "NO_GO"

    def test_marginal(self) -> None:
        # frontier in [0.10, 0.15), GRPO condition not met → MARGINAL.
        v, _ = classify_gate(self._summary(0.12, 0.02, 0.80, 0.02))
        assert v == "MARGINAL"

    def test_rft_beats_revival_when_both_true(self) -> None:
        v, _ = classify_gate(self._summary(0.30, 0.10, 0.20, 0.08))
        assert v == "GO_RFT"
