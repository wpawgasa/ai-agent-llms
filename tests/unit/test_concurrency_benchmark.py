"""Unit tests for concurrency_benchmark degradation logic and helpers."""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.concurrency_benchmark import (
    LevelResult,
    PercentileStats,
    _percentiles,
)


def _make_level(concurrency: int, ttft_p95: float, success_rate: float = 1.0) -> LevelResult:
    return LevelResult(
        concurrency=concurrency,
        ttft_ms=PercentileStats(p50=ttft_p95 * 0.7, p95=ttft_p95, p99=ttft_p95 * 1.2),
        success_rate=success_rate,
    )


def _max_sustainable(levels: list[LevelResult], multiplier: float, max_failure: float) -> int:
    """Mirror of the decision logic inside run_concurrency_sweep."""
    baseline_ttft_p95 = next(lv.ttft_ms.p95 for lv in levels if lv.concurrency == 1)
    best = 1
    for lv in levels:
        threshold = multiplier * baseline_ttft_p95
        if lv.ttft_ms.p95 <= threshold and (1.0 - lv.success_rate) <= max_failure:
            best = lv.concurrency
    return best


class TestPercentileStats:
    def test_empty(self):
        p = _percentiles([])
        assert p.p50 == 0.0
        assert p.p95 == 0.0
        assert p.p99 == 0.0

    def test_single(self):
        p = _percentiles([42.0])
        assert p.p50 == 42.0
        assert p.p95 == 42.0

    def test_sorted_order(self):
        vals = list(range(100, 200))  # 100 values: 100..199
        p = _percentiles(vals)
        assert p.p50 == vals[49]
        assert p.p95 == vals[94]
        assert p.p99 == vals[98]

    def test_unsorted_input(self):
        # _percentiles uses index = max(0, int(n*frac) - 1).
        # For n=3: p50 index = max(0, 1-1) = 0 → sorted[0]=100, p95 index = max(0,2-1)=1 → 200
        p = _percentiles([300.0, 100.0, 200.0])
        assert p.p50 == 100.0   # int(3*0.50)-1=0 → sorted[0]
        assert p.p95 == 200.0   # int(3*0.95)-1=1 → sorted[1]

    def test_to_dict(self):
        p = PercentileStats(p50=1.0, p95=2.0, p99=3.0)
        assert p.to_dict() == {"p50": 1.0, "p95": 2.0, "p99": 3.0}


class TestDegradationRule:
    """Verify max_sustainable_concurrency selection under the 2× TTFT rule."""

    def test_baseline_always_included(self):
        levels = [_make_level(1, 100.0), _make_level(2, 250.0)]
        # At 2×: threshold = 200 ms; level-2 TTFT=250 violates
        assert _max_sustainable(levels, 2.0, 0.01) == 1

    def test_multiple_within_threshold(self):
        levels = [
            _make_level(1, 100.0),
            _make_level(2, 150.0),
            _make_level(4, 190.0),
            _make_level(8, 210.0),  # violates 2× = 200 ms
        ]
        assert _max_sustainable(levels, 2.0, 0.01) == 4

    def test_exact_boundary_passes(self):
        levels = [_make_level(1, 100.0), _make_level(2, 200.0)]
        # 200 == 2×100: should pass (<=)
        assert _max_sustainable(levels, 2.0, 0.01) == 2

    def test_failure_rate_violation(self):
        levels = [
            _make_level(1, 100.0, success_rate=1.0),
            _make_level(2, 150.0, success_rate=0.98),   # 2% failure > 1% limit
        ]
        assert _max_sustainable(levels, 2.0, 0.01) == 1

    def test_failure_rate_just_under_limit(self):
        # 0.5% failure < 1% limit — passes
        levels = [
            _make_level(1, 100.0, success_rate=1.0),
            _make_level(2, 150.0, success_rate=0.995),
        ]
        assert _max_sustainable(levels, 2.0, 0.01) == 2

    def test_tight_multiplier(self):
        levels = [
            _make_level(1, 100.0),
            _make_level(2, 115.0),
            _make_level(4, 125.0),  # violates 1.2× = 120 ms
        ]
        assert _max_sustainable(levels, 1.2, 0.01) == 2

    def test_lenient_multiplier_accepts_all(self):
        levels = [
            _make_level(1, 100.0),
            _make_level(2, 180.0),
            _make_level(4, 280.0),
            _make_level(8, 350.0),
        ]
        assert _max_sustainable(levels, 4.0, 0.01) == 8

    def test_all_fail_from_level_2(self):
        levels = [
            _make_level(1, 100.0),
            _make_level(2, 500.0),   # immediate violation
            _make_level(4, 900.0),
        ]
        assert _max_sustainable(levels, 2.0, 0.01) == 1
