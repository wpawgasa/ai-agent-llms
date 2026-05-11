"""Unit tests for concurrency_benchmark degradation logic and helpers."""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.concurrency_benchmark import (
    ConcurrencySweepResult,
    LevelResult,
    PercentileStats,
    _build_prompts,
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


class TestSweepResultSerialization:
    """The result envelope must surface engine + endpoint so frontier and vLLM
    runs can be told apart (and re-targeted) from the JSON alone."""

    def test_default_engine_is_vllm(self):
        r = ConcurrencySweepResult(
            model="qwen3-32b",
            kv_cache_dtype="auto",
            input_tokens_min=512,
            input_tokens_max=2048,
            output_tokens=128,
            degradation_ttft_multiplier=2.0,
            max_failure_rate=0.01,
        )
        d = r.to_dict()
        assert d["engine"] == "vllm"
        assert d["endpoint"] == ""
        assert d["input_tokens_min"] == 512
        assert d["input_tokens_max"] == 2048

    def test_frontier_engine_round_trip(self):
        r = ConcurrencySweepResult(
            model="anthropic/claude-sonnet-4-6",
            kv_cache_dtype="remote",
            input_tokens_min=512,
            input_tokens_max=2048,
            output_tokens=128,
            degradation_ttft_multiplier=2.0,
            max_failure_rate=0.01,
            engine="bifrost",
            endpoint="http://localhost:23040",
        )
        d = r.to_dict()
        assert d["engine"] == "bifrost"
        assert d["endpoint"] == "http://localhost:23040"
        assert d["kv_cache_dtype"] == "remote"


class TestPromptVariation:
    """_build_prompts must return varied, unique prompts that defeat prefix caching."""

    def test_count(self):
        prompts = _build_prompts(10, 512, 2048, seed=42)
        assert len(prompts) == 10

    def test_lengths_in_range(self):
        snippet = "Describe a step in a business workflow that involves state transitions and tool calls. "
        min_tok, max_tok = 512, 2048
        prompts = _build_prompts(50, min_tok, max_tok, seed=7)
        for p in prompts:
            # char length must be at least min_tok*4 − len(snippet) (min reps=1 floor)
            # and at most max_tok*4 + len(snippet) + 20 (header slack)
            assert len(p) >= min_tok * 4 - len(snippet), f"prompt too short: {len(p)}"
            assert len(p) <= max_tok * 4 + len(snippet) + 20, f"prompt too long: {len(p)}"

    def test_all_unique(self):
        prompts = _build_prompts(20, 512, 2048, seed=99)
        # Unique per-request header guarantees no two prompts share the same prefix.
        headers = {p[:32] for p in prompts}
        assert len(headers) == len(prompts)

    def test_seed_reproducibility(self):
        p1 = _build_prompts(5, 512, 2048, seed=123)
        p2 = _build_prompts(5, 512, 2048, seed=123)
        assert p1 == p2

    def test_different_seeds_differ(self):
        p1 = _build_prompts(5, 512, 2048, seed=1)
        p2 = _build_prompts(5, 512, 2048, seed=2)
        assert p1[0] != p2[0]

    def test_validation_min_gt_max(self):
        with pytest.raises(ValueError, match="Invalid token range"):
            _build_prompts(5, 2048, 512, seed=0)

    def test_validation_min_zero(self):
        with pytest.raises(ValueError, match="Invalid token range"):
            _build_prompts(5, 0, 512, seed=0)
