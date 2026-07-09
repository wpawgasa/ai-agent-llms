"""Tests for the pure logic of the trajectory variance micro-probe.

The GPU rollout driver (main) is not exercised here; these lock the reduction
(summarize_trajectory_probe) and the gate classifier (classify_gate) that decide
GO_TRAJECTORY / NO_GO_VARIANCE / NO_GO_TRUNCATION / MARGINAL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from trajectory_variance_probe import (  # noqa: E402
    classify_gate,
    summarize_trajectory_probe,
)


class TestSummarizeTrajectoryProbe:
    def test_basic_two_groups(self) -> None:
        group_rewards = [[0.3, 0.5, 1.0], [0.4, 0.4, 0.4]]
        group_coverages = [[0.2, 0.5, 1.0], [0.5, 0.5, 0.5]]
        group_metas = [
            [{"stop_reason": "gold_complete", "n_model_turns": 6}] * 3,
            [{"stop_reason": "stall", "n_model_turns": 2}] * 3,
        ]
        s = summarize_trajectory_probe(group_rewards, group_coverages, group_metas)
        assert s["n_groups"] == 2
        # group1 has zero std -> collapsed.
        assert s["frac_collapsed_groups"] == pytest.approx(0.5)
        # group0 pstdev([0.3,0.5,1.0])=0.29440, group1=0 -> median = 0.14720.
        assert s["median_reward_std"] == pytest.approx(0.14720, abs=1e-4)
        assert s["mean_coverage"] == pytest.approx(3.2 / 6)
        assert s["mean_within_group_coverage_spread"] == pytest.approx(0.4)
        assert s["mean_model_turns"] == pytest.approx(4.0)
        assert s["stop_reason_histogram"] == {"gold_complete": 3, "stall": 3}
        assert s["rung_histogram"] == {1: 1, 3: 1}

    def test_empty(self) -> None:
        s = summarize_trajectory_probe([], [], [])
        assert s["frac_collapsed_groups"] == 0.0
        assert s["median_reward_std"] == 0.0
        assert s["mean_model_turns"] == 0.0


class TestClassifyGate:
    def _s(self, med_std, collapsed, turns, cov):
        return {
            "median_reward_std": med_std,
            "frac_collapsed_groups": collapsed,
            "mean_model_turns": turns,
            "mean_coverage": cov,
        }

    def test_go_trajectory(self) -> None:
        v, _ = classify_gate(self._s(0.08, 0.30, 5.0, 0.5))
        assert v == "GO_TRAJECTORY"

    def test_no_go_variance(self) -> None:
        v, _ = classify_gate(self._s(0.01, 0.90, 6.0, 0.9))
        assert v == "NO_GO_VARIANCE"

    def test_no_go_truncation(self) -> None:
        v, _ = classify_gate(self._s(0.06, 0.30, 1.5, 0.2))
        assert v == "NO_GO_TRUNCATION"

    def test_marginal(self) -> None:
        v, _ = classify_gate(self._s(0.03, 0.60, 4.0, 0.5))
        assert v == "MARGINAL"

    def test_go_requires_coverage_band(self) -> None:
        # Healthy std/turns but saturated coverage (>0.95) is not GO.
        v, _ = classify_gate(self._s(0.08, 0.30, 5.0, 0.99))
        assert v != "GO_TRAJECTORY"
