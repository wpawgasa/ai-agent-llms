"""Shared teacher model API client for data generation.

Routes calls to the appropriate provider based on model name prefix:
  ``gemini-*``  → Google GenAI  (env: GEMINI_API_KEY)
  ``gpt-*``     → OpenAI        (env: OPENAI_API_KEY)
  ``claude-*``  → Anthropic     (env: ANTHROPIC_API_KEY)

Two concurrency-oriented behaviours back the parallel generation path
(``generate_workflow_dataset(..., max_workers>1)``):

  * **Client reuse** — each provider's SDK client is constructed once
    (lazily, under a lock) and shared across threads. The google-genai,
    openai, and anthropic clients are safe for concurrent requests; only
    construction needs guarding.
  * **Bounded retry with backoff + jitter** — transient failures
    (rate-limit / timeout / connection / 5xx) are retried with exponential
    backoff so a single 429 under fan-out does not cascade into placeholder
    fallbacks. Non-transient errors re-raise immediately, preserving the
    caller's ``teacher_model_fallback`` semantics for genuine failures.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Retry policy (env-overridable). Defaults are tuned for ~8-way fan-out against
# mid-tier provider quotas: enough retries to ride out a short throttle window
# without stalling the whole run.
_MAX_RETRIES = int(os.environ.get("TEACHER_MAX_RETRIES", "5"))
_BACKOFF_BASE_S = float(os.environ.get("TEACHER_BACKOFF_BASE", "1.0"))
_BACKOFF_MAX_S = float(os.environ.get("TEACHER_BACKOFF_MAX", "30.0"))

# Substrings that mark an error as transient/retryable when the SDK does not
# expose a structured status code.
_RETRYABLE_MARKERS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "resource exhausted",
    "resourceexhausted",
    "quota",
    "timeout",
    "timed out",
    "deadline exceeded",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "serviceunavailable",
    "connection",
    "connection reset",
    "internal server error",
    "internalservererror",
    "502",
    "503",
    "504",
    "429",
)
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Client singletons (constructed once, shared across threads)
# ---------------------------------------------------------------------------

_clients: dict[str, Any] = {}
_clients_lock = threading.Lock()


def _get_client(key: str, factory: Callable[[], Any]) -> Any:
    """Return a cached provider client, constructing it once under a lock."""
    client = _clients.get(key)
    if client is not None:
        return client
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = factory()
            _clients[key] = client
        return client


def _reset_clients() -> None:
    """Drop cached clients (test hook)."""
    with _clients_lock:
        _clients.clear()


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """Best-effort classification of a provider error as transient."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _backoff_delay(attempt: int, rng: Any) -> float:
    """Exponential backoff with full jitter, capped at ``_BACKOFF_MAX_S``.

    ``attempt`` is 0-based (delay before the first retry uses attempt=0).
    """
    ceiling = min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * (2 ** attempt))
    return rng.uniform(0.0, ceiling)


def _with_retries(model: str, fn: Callable[[], T]) -> T:
    """Call ``fn`` with bounded exponential backoff on transient errors.

    Re-raises non-transient errors immediately and the last transient error
    after ``_MAX_RETRIES`` exhausted retries.
    """
    import random

    # A local RNG keeps jitter independent of the dataset's seeded RNG.
    jitter_rng = random.Random()
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= _MAX_RETRIES or not _is_retryable(exc):
                raise
            delay = _backoff_delay(attempt, jitter_rng)
            logger.warning(
                "teacher_call_retry",
                teacher_model=model,
                attempt=attempt + 1,
                max_retries=_MAX_RETRIES,
                delay_s=round(delay, 2),
                error=str(exc)[:200],
            )
            time.sleep(delay)
    # Unreachable: the loop either returns or raises, but satisfy type checkers.
    assert last_exc is not None
    raise last_exc


def call_teacher_model(teacher_model: str, system_prompt: str, user_prompt: str) -> str:
    """Call a teacher model and return the raw text response.

    Args:
        teacher_model: Model name, e.g. ``"gemini-2.0-flash"``, ``"gpt-4o"``,
            ``"claude-sonnet-4-6"``.
        system_prompt: System / instruction prompt.
        user_prompt: User turn content.

    Returns:
        Raw text response from the model.

    Raises:
        ValueError: If the model prefix is not recognised.
    """
    if teacher_model.startswith("gemini"):
        return _call_gemini(teacher_model, system_prompt, user_prompt)
    if teacher_model.startswith("gpt"):
        return _call_openai(teacher_model, system_prompt, user_prompt)
    if teacher_model.startswith("claude"):
        return _call_anthropic(teacher_model, system_prompt, user_prompt)
    raise ValueError(
        f"Unsupported teacher model {teacher_model!r}. "
        "Expected prefix: gemini-*, gpt-*, or claude-*."
    )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_gemini(model: str, system_prompt: str, user_prompt: str) -> str:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types as genai_types  # type: ignore[import-untyped]

    client = _get_client("gemini", lambda: genai.Client(api_key=os.environ["GEMINI_API_KEY"]))

    def _do() -> str:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        return response.text

    return _with_retries(model, _do)


def _call_openai(model: str, system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI  # type: ignore[import-untyped]

    client = _get_client("openai", lambda: OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))

    def _do() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    return _with_retries(model, _do)


def _call_anthropic(model: str, system_prompt: str, user_prompt: str) -> str:
    import anthropic  # type: ignore[import-untyped]

    client = _get_client(
        "anthropic", lambda: anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    )

    def _do() -> str:
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip optional markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return raw

    return _with_retries(model, _do)
