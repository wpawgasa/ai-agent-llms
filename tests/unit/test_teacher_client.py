"""Tests for the teacher-model client: retry/backoff and client reuse.

These back the parallel generation path — under fan-out a single transient
429/timeout must be retried (not cascade into a placeholder fallback), and each
provider's SDK client must be constructed once and shared across threads.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.data import _teacher_client as tc


@pytest.fixture(autouse=True)
def _no_sleep_and_fresh_clients(monkeypatch):
    """Make backoff instant and reset the client cache between tests."""
    monkeypatch.setattr(tc.time, "sleep", lambda _s: None)
    tc._reset_clients()
    yield
    tc._reset_clients()


class TestIsRetryable:
    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("429 Too Many Requests"),
            RuntimeError("rate limit exceeded"),
            RuntimeError("Resource exhausted (quota)"),
            TimeoutError("request timed out"),
            RuntimeError("503 Service Unavailable"),
            RuntimeError("model is overloaded"),
        ],
    )
    def test_transient_errors_are_retryable(self, exc):
        assert tc._is_retryable(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("invalid request: bad schema"),
            KeyError("missing field"),
            RuntimeError("authentication failed: 401"),
        ],
    )
    def test_permanent_errors_are_not_retryable(self, exc):
        assert tc._is_retryable(exc) is False

    def test_status_code_attribute_is_honored(self):
        exc = RuntimeError("opaque message")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert tc._is_retryable(exc) is True


class TestWithRetries:
    def test_retries_transient_then_succeeds(self, monkeypatch):
        calls = {"n": 0}
        sleeps: list[float] = []
        monkeypatch.setattr(tc.time, "sleep", lambda s: sleeps.append(s))

        def flaky():
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("429 rate limit")
            return "ok"

        result = tc._with_retries("gemini-x", flaky)
        assert result == "ok"
        assert calls["n"] == 3          # 1 initial + 2 retries
        assert len(sleeps) == 2         # one backoff before each retry
        assert all(s >= 0.0 for s in sleeps)

    def test_non_retryable_raises_immediately(self):
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise ValueError("bad schema")

        with pytest.raises(ValueError, match="bad schema"):
            tc._with_retries("gpt-x", boom)
        assert calls["n"] == 1          # no retries

    def test_gives_up_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(tc, "_MAX_RETRIES", 3)
        calls = {"n": 0}

        def always_throttled():
            calls["n"] += 1
            raise RuntimeError("503 unavailable")

        with pytest.raises(RuntimeError, match="503"):
            tc._with_retries("claude-x", always_throttled)
        assert calls["n"] == 4          # 1 initial + 3 retries


class TestClientReuse:
    def test_client_constructed_once(self):
        constructions = {"n": 0}

        def factory():
            constructions["n"] += 1
            return object()

        c1 = tc._get_client("prov", factory)
        c2 = tc._get_client("prov", factory)
        assert c1 is c2
        assert constructions["n"] == 1


class TestRouting:
    def test_unsupported_prefix_raises(self):
        with pytest.raises(ValueError, match="Unsupported teacher model"):
            tc.call_teacher_model("llama-3", "sys", "user")
