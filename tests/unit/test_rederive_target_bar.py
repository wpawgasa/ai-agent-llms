"""Unit tests for scripts/rederive_target_bar.py pure functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rederive_target_bar import (  # noqa: E402
    STATED_OLD_BAR,
    TARGET_STATE_ACC,
    TARGET_TASK_COMPLETION,
    TARGET_TOOL_F1,
    derive_bars,
    expected_perturn_score,
    old_composite,
    required_tool_f1,
    summarize_profile,
)

# (tool_expected, has_state_term, has_task_term)
TOOL_ROW = (True, True, False)
ZERO_TOOL_ROW = (False, True, False)
TERMINAL_ROW = (False, True, True)


class TestOldComposite:
    def test_weights_sum_to_one_at_perfect(self):
        assert old_composite(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_known_value_at_component_targets(self):
        # 0.4*0.85 + 0.4*0.85 + 0.2*0.70 = 0.82
        assert old_composite(
            TARGET_STATE_ACC, TARGET_TOOL_F1, TARGET_TASK_COMPLETION
        ) == pytest.approx(0.82)

    def test_all_zero(self):
        assert old_composite(0.0, 0.0, 0.0) == 0.0


class TestExpectedPerturnScore:
    def test_empty_profile(self):
        assert expected_perturn_score([], 0.9, 0.9, 0.9, 0.9) == 0.0

    def test_perfect_policy_scores_one(self):
        profile = [TOOL_ROW, ZERO_TOOL_ROW, TERMINAL_ROW]
        assert expected_perturn_score(profile, 1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_non_task_row_renormalizes_to_half_half(self):
        # den = 0.4 (tool) + 0.4 (state) -> effective 0.5/0.5
        score = expected_perturn_score([TOOL_ROW], state_acc=1.0, tool_f1=0.0,
                                       abstention_acc=0.0, task_completion=0.0)
        assert score == pytest.approx(0.5)

    def test_terminal_row_uses_full_three_term_weights(self):
        # den = 1.0; only the task term is earned -> 0.2
        score = expected_perturn_score([TERMINAL_ROW], state_acc=0.0, tool_f1=0.0,
                                       abstention_acc=0.0, task_completion=1.0)
        assert score == pytest.approx(0.2)

    def test_abstention_drives_zero_tool_rows_not_tool_f1(self):
        # Zero-tool row must ignore tool_f1 entirely and use abstention_acc.
        a = expected_perturn_score([ZERO_TOOL_ROW], 0.0, 0.0, 1.0, 0.0)
        b = expected_perturn_score([ZERO_TOOL_ROW], 0.0, 1.0, 1.0, 0.0)
        assert a == b == pytest.approx(0.5)

    def test_tool_f1_drives_tool_rows_not_abstention(self):
        a = expected_perturn_score([TOOL_ROW], 0.0, 1.0, 0.0, 0.0)
        b = expected_perturn_score([TOOL_ROW], 0.0, 1.0, 1.0, 0.0)
        assert a == b == pytest.approx(0.5)

    def test_monotonic_in_each_component(self):
        profile = [TOOL_ROW, ZERO_TOOL_ROW, TERMINAL_ROW]
        base = expected_perturn_score(profile, 0.5, 0.5, 0.5, 0.5)
        assert expected_perturn_score(profile, 0.9, 0.5, 0.5, 0.5) > base
        assert expected_perturn_score(profile, 0.5, 0.9, 0.5, 0.5) > base
        assert expected_perturn_score(profile, 0.5, 0.5, 0.9, 0.5) > base
        assert expected_perturn_score(profile, 0.5, 0.5, 0.5, 0.9) > base

    def test_zero_tool_majority_inflates_score_above_old_metric(self):
        # The core claim: with mostly zero-tool rows and good abstention, the
        # per-turn metric sits ABOVE the whole-conversation composite for the
        # same policy — which is why the bar must rise.
        profile = [ZERO_TOOL_ROW] * 6 + [TOOL_ROW] * 4
        perturn = expected_perturn_score(profile, 0.85, 0.85, 1.0, 0.70)
        assert perturn > old_composite(0.85, 0.85, 0.70)


class TestRequiredToolF1:
    def test_inverts_expected_score(self):
        profile = [TOOL_ROW] * 4 + [ZERO_TOOL_ROW] * 6
        target_t = 0.6
        bar = expected_perturn_score(profile, 0.85, target_t, 0.95, 0.70)
        got = required_tool_f1(profile, bar, 0.85, 0.95, 0.70)
        assert got == pytest.approx(target_t, abs=1e-9)

    def test_unreachable_bar_returns_none(self):
        profile = [TOOL_ROW] * 4 + [ZERO_TOOL_ROW] * 6
        assert required_tool_f1(profile, 1.5, 0.85, 0.95, 0.70) is None

    def test_already_met_at_zero_returns_zero(self):
        profile = [ZERO_TOOL_ROW]
        # Bar reachable with no tool-calling ability at all.
        assert required_tool_f1(profile, 0.4, 1.0, 1.0, 1.0) == pytest.approx(0.0)

    def test_no_tool_expected_rows_is_flat_in_t(self):
        profile = [ZERO_TOOL_ROW] * 5
        # Score does not depend on t; an unreachable bar must return None.
        assert required_tool_f1(profile, 0.99, 0.5, 0.5, 0.5) is None


class TestDeriveBars:
    def test_relaxation_ratio_matches_stated_bar(self):
        bars = derive_bars([TOOL_ROW, ZERO_TOOL_ROW])
        assert bars["old_composite_at_component_targets"] == pytest.approx(0.82)
        assert bars["relaxation_ratio"] == pytest.approx(STATED_OLD_BAR / 0.82)

    def test_sweep_covers_all_assumptions(self):
        bars = derive_bars([TOOL_ROW, ZERO_TOOL_ROW], abstention_sweep=(1.0, 0.85))
        assert [r["abstention_acc"] for r in bars["sweep"]] == [1.0, 0.85]

    def test_higher_abstention_gives_higher_bar(self):
        bars = derive_bars([ZERO_TOOL_ROW] * 5 + [TOOL_ROW], abstention_sweep=(1.0, 0.85))
        assert (
            bars["sweep"][0]["component_equivalent_bar"]
            > bars["sweep"][1]["component_equivalent_bar"]
        )

    def test_relaxation_matched_is_below_component_equivalent(self):
        bars = derive_bars([TOOL_ROW, ZERO_TOOL_ROW])
        for row in bars["sweep"]:
            assert row["relaxation_matched_bar"] < row["component_equivalent_bar"]


class TestSummarizeProfile:
    def test_empty(self):
        assert summarize_profile([]) == {"n_rows": 0}

    def test_fractions(self):
        profile = [TOOL_ROW, ZERO_TOOL_ROW, TERMINAL_ROW, ZERO_TOOL_ROW]
        s = summarize_profile(profile)
        assert s["n_rows"] == 4
        assert s["frac_tool_expected"] == 0.25
        assert s["frac_zero_tool"] == 0.75
        assert s["frac_state_term_applies"] == 1.0
        assert s["frac_task_term_applies"] == 0.25
