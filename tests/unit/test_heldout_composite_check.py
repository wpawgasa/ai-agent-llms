"""Tests for the pure logic of the SFT-only ceiling check.

The GPU generation path is not exercised here (no model); these lock the
reduction (summarize_heldout_check) and the gate classifier (classify_gate)
that decide PASS / FAIL against the Phase 2 target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from heldout_composite_check import (  # noqa: E402
    TARGET_COMPOSITE,
    classify_gate,
    summarize_heldout_check,
)


class TestSummarizeHeldoutCheck:
    def test_basic_rows(self) -> None:
        s = summarize_heldout_check([0.8, 0.6, 1.0, 0.4])
        assert s["n_rows"] == 4
        assert s["mean_composite"] == pytest.approx(0.7)
        assert s["median_composite"] == pytest.approx(0.7)
        assert s["min_composite"] == pytest.approx(0.4)
        assert s["max_composite"] == pytest.approx(1.0)
        # 0.6 and 0.4 are below the 0.75 target -> 2/4.
        assert s["frac_below_target"] == pytest.approx(0.5)
        assert s["per_row"] == [
            {"row_index": 0, "composite": 0.8},
            {"row_index": 1, "composite": 0.6},
            {"row_index": 2, "composite": 1.0},
            {"row_index": 3, "composite": 0.4},
        ]

    def test_all_above_target(self) -> None:
        s = summarize_heldout_check([0.9, 0.85, 0.76])
        assert s["frac_below_target"] == 0.0

    def test_empty(self) -> None:
        s = summarize_heldout_check([])
        assert s["n_rows"] == 0
        assert s["mean_composite"] == 0.0
        assert s["median_composite"] == 0.0
        assert s["min_composite"] == 0.0
        assert s["max_composite"] == 0.0
        assert s["frac_below_target"] == 0.0
        assert s["per_row"] == []


class TestClassifyGate:
    def _summary(self, mean_composite: float) -> dict:
        return {"mean_composite": mean_composite}

    def test_pass_above_target(self) -> None:
        v, detail = classify_gate(self._summary(0.80))
        assert v == "PASS"
        assert detail["checks"][f"mean_composite >= {TARGET_COMPOSITE}"] is True

    def test_fail_below_target(self) -> None:
        v, detail = classify_gate(self._summary(0.60))
        assert v == "FAIL"
        assert detail["checks"][f"mean_composite >= {TARGET_COMPOSITE}"] is False

    def test_pass_at_exact_boundary(self) -> None:
        # >= is inclusive: exactly the target counts as clearing it.
        v, _ = classify_gate(self._summary(0.75))
        assert v == "PASS"

    def test_fail_on_empty_summary(self) -> None:
        # An empty probe run reduces to mean_composite == 0.0 -> FAIL, not a crash.
        v, _ = classify_gate(self._summary(0.0))
        assert v == "FAIL"
