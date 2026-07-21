"""Unit tests for scripts/passk_tool_emission_probe.py pure functions."""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from passk_tool_emission_probe import (  # noqa: E402
    MATERIAL_RECOVERY,
    WEAK_RECOVERY,
    classify_gate,
    pass_at_k,
    summarize_passk,
)


class TestPassAtK:
    def test_no_correct_samples_is_zero(self):
        assert pass_at_k(8, 0, 8) == 0.0
        assert pass_at_k(8, 0, 1) == 0.0

    def test_all_correct_is_one(self):
        assert pass_at_k(8, 8, 1) == 1.0
        assert pass_at_k(8, 8, 8) == 1.0

    def test_pass_at_1_equals_empirical_rate(self):
        # With k=1 the estimator reduces to c/n.
        assert pass_at_k(8, 2, 1) == pytest.approx(0.25)
        assert pass_at_k(10, 7, 1) == pytest.approx(0.7)

    def test_pass_at_n_is_one_when_any_correct(self):
        # If even one of n is correct, drawing all n always includes it.
        assert pass_at_k(8, 1, 8) == pytest.approx(1.0)

    def test_matches_closed_form(self):
        # 1 - C(n-c, k)/C(n, k)
        n, c, k = 10, 3, 4
        expected = 1 - comb(n - c, k) / comb(n, k)
        assert pass_at_k(n, c, k) == pytest.approx(expected)

    def test_monotonic_in_k(self):
        vals = [pass_at_k(8, 2, k) for k in (1, 2, 4, 8)]
        assert vals == sorted(vals)
        assert vals[0] < vals[-1]

    def test_k_clamped_to_n(self):
        assert pass_at_k(4, 1, 99) == pytest.approx(1.0)

    def test_degenerate_inputs(self):
        assert pass_at_k(0, 0, 4) == 0.0
        assert pass_at_k(8, 4, 0) == 0.0


def _rec(
    c_emitted: int,
    c_name: int,
    *,
    n: int = 8,
    best_f1: float = 0.0,
    mean_f1: float = 0.0,
    greedy: tuple[bool, bool] | None = None,
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "n_samples": n,
        "c_emitted": c_emitted,
        "c_name_match": c_name,
        "best_f1": best_f1,
        "mean_f1": mean_f1,
    }
    if greedy is not None:
        r["greedy_emitted"], r["greedy_name_match"] = greedy
    return r


class TestSummarizePassk:
    def test_empty_records(self):
        s = summarize_passk([])
        assert s["n_anchors"] == 0
        assert s["frac_never_name_match"] == 0.0
        assert s["recovery_name_match"] == 0.0

    def test_curve_climbs_with_k(self):
        s = summarize_passk([_rec(2, 2), _rec(1, 1)])
        curve = s["pass_at_k_name_match"]
        assert curve["pass@1"] < curve["pass@8"]
        assert curve["pass@8"] == pytest.approx(1.0)

    def test_never_and_always_fractions(self):
        records = [_rec(0, 0), _rec(8, 8), _rec(4, 2), _rec(0, 0)]
        s = summarize_passk(records)
        assert s["frac_never_name_match"] == 0.5
        assert s["frac_never_emitted"] == 0.5
        assert s["frac_always_name_match"] == 0.25

    def test_per_sample_means(self):
        s = summarize_passk([_rec(8, 4), _rec(0, 0)])
        assert s["mean_per_sample_emission"] == 0.5
        assert s["mean_per_sample_name_match"] == 0.25

    def test_recovery_against_paired_greedy(self):
        # 4 anchors; greedy name-matches on 1; sampling name-matches on all 4.
        records = [
            _rec(8, 8, greedy=(True, True)),
            _rec(8, 8, greedy=(False, False)),
            _rec(8, 8, greedy=(False, False)),
            _rec(8, 8, greedy=(False, False)),
        ]
        s = summarize_passk(records)
        assert s["n_paired_with_greedy"] == 4
        assert s["greedy_name_match_rate"] == 0.25
        assert s["pass_at_k_name_match"]["pass@8"] == pytest.approx(1.0)
        assert s["recovery_name_match"] == pytest.approx(0.75)

    def test_records_without_greedy_are_not_counted_as_paired(self):
        s = summarize_passk([_rec(8, 8), _rec(0, 0)])
        assert s["n_paired_with_greedy"] == 0
        assert s["greedy_name_match_rate"] == 0.0

    def test_f1_means(self):
        s = summarize_passk([_rec(8, 8, best_f1=1.0, mean_f1=0.5), _rec(0, 0)])
        assert s["mean_best_f1"] == 0.5
        assert s["mean_mean_f1"] == 0.25


class TestClassifyGate:
    def test_sampling_recovers(self):
        verdict, detail = classify_gate({"recovery_name_match": MATERIAL_RECOVERY + 0.1})
        assert verdict == "SAMPLING_RECOVERS"
        assert all(detail["checks"].values())

    def test_weak_signal(self):
        verdict, _ = classify_gate({"recovery_name_match": WEAK_RECOVERY + 0.01})
        assert verdict == "WEAK_SIGNAL"

    def test_absent_from_policy(self):
        verdict, _ = classify_gate({"recovery_name_match": 0.0})
        assert verdict == "ABSENT_FROM_POLICY"

    def test_negative_recovery_is_absent(self):
        verdict, _ = classify_gate({"recovery_name_match": -0.1})
        assert verdict == "ABSENT_FROM_POLICY"

    def test_boundaries_are_inclusive(self):
        assert classify_gate({"recovery_name_match": MATERIAL_RECOVERY})[0] == (
            "SAMPLING_RECOVERS"
        )
        assert classify_gate({"recovery_name_match": WEAK_RECOVERY})[0] == "WEAK_SIGNAL"
