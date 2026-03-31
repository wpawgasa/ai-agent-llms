"""Needle-in-a-Haystack evaluation for KV cache quantization.

Tests long-context retrieval accuracy by inserting a unique "needle"
fact at various depths within a long "haystack" context, then asking
the model to retrieve it. Evaluates across multiple context lengths
(2K-32K) to identify where quantization degrades retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768]
DEFAULT_DEPTH_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

NEEDLE_TEMPLATE = "The special magic number for this test is {magic_number}."
QUESTION_TEMPLATE = "What is the special magic number mentioned in the text?"

# Filler text for haystack (repeated to fill context)
HAYSTACK_FILLER = (
    "This is background information about various topics. "
    "The weather has been particularly interesting this season with "
    "fluctuations in temperature and unexpected precipitation patterns. "
    "Scientists have been studying these changes closely to better "
    "understand the underlying climate dynamics. "
)


@dataclass
class NeedleResult:
    """Result for a single needle-in-haystack probe."""

    context_length: int
    depth_position: float  # 0.0 = beginning, 1.0 = end
    needle_found: bool
    model_response: str = ""
    expected_answer: str = ""
    probe_error: bool = False  # True when the API call failed (not a model miss)


@dataclass
class NeedleHaystackResult:
    """Aggregated needle-in-haystack evaluation result."""

    probes: list[NeedleResult] = field(default_factory=list)
    accuracy_by_length: dict[int, float] = field(default_factory=dict)
    accuracy_by_depth: dict[float, float] = field(default_factory=dict)
    overall_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_accuracy": self.overall_accuracy,
            "accuracy_by_length": self.accuracy_by_length,
            "accuracy_by_depth": {str(k): v for k, v in self.accuracy_by_depth.items()},
            "total_probes": len(self.probes),
        }


def _build_haystack(
    context_length_tokens: int,
    needle: str,
    depth: float,
    chars_per_token: int = 4,
) -> str:
    """Build a haystack text with the needle inserted at the specified depth.

    Args:
        context_length_tokens: Target context length in tokens.
        needle: The needle text to insert.
        depth: Position in [0.0, 1.0] — 0.0=start, 1.0=end.
        chars_per_token: Approximate characters per token.

    Returns:
        Haystack string with needle inserted.
    """
    target_chars = context_length_tokens * chars_per_token
    needle_chars = len(needle)
    filler_chars = target_chars - needle_chars

    # Build filler by repeating
    filler_repeats = max(filler_chars // len(HAYSTACK_FILLER) + 1, 1)
    filler = (HAYSTACK_FILLER * filler_repeats)[:filler_chars]

    # Insert needle at depth position
    insert_pos = int(len(filler) * depth)
    # Align to sentence boundary
    if insert_pos > 0:
        space_pos = filler.rfind(". ", 0, insert_pos)
        if space_pos > 0:
            insert_pos = space_pos + 2

    haystack = filler[:insert_pos] + " " + needle + " " + filler[insert_pos:]
    return haystack


def check_needle_found(response: str, magic_number: str) -> bool:
    """Check if the model's response contains the correct magic number."""
    return magic_number in response


def evaluate_needle_in_haystack(
    model_path: str,
    context_lengths: list[int] | None = None,
    depth_positions: list[float] | None = None,
    kv_cache_dtype: str = "auto",
    base_url: str = "http://localhost:8000/v1",
) -> NeedleHaystackResult:
    """Run needle-in-a-haystack evaluation via a running vLLM server.

    Args:
        model_path: Model name for the API.
        context_lengths: Context lengths to test (default: 2K-32K).
        depth_positions: Needle depth positions (default: 0, 0.25, 0.5, 0.75, 1.0).
        kv_cache_dtype: KV cache dtype (for logging).
        base_url: vLLM server URL.

    Returns:
        NeedleHaystackResult with per-probe and aggregated accuracy.
    """
    import openai

    if context_lengths is None:
        context_lengths = DEFAULT_CONTEXT_LENGTHS
    if depth_positions is None:
        depth_positions = DEFAULT_DEPTH_POSITIONS

    client = openai.OpenAI(base_url=base_url, api_key="unused")

    logger.info(
        "evaluating_needle_haystack",
        model=model_path,
        context_lengths=context_lengths,
        kv_cache_dtype=kv_cache_dtype,
    )

    result = NeedleHaystackResult()

    for ctx_len in context_lengths:
        for depth in depth_positions:
            magic_number = f"{ctx_len * 1000 + int(depth * 100)}"
            needle = NEEDLE_TEMPLATE.format(magic_number=magic_number)

            haystack = _build_haystack(ctx_len, needle, depth)
            prompt = f"{haystack}\n\n{QUESTION_TEMPLATE}"

            probe_error = False
            try:
                response = client.completions.create(
                    model=model_path,
                    prompt=prompt,
                    max_tokens=64,
                    temperature=0.0,
                )
                model_response = response.choices[0].text.strip() if response.choices else ""
            except Exception as exc:
                logger.warning(
                    "needle_probe_failed",
                    context_length=ctx_len,
                    depth=depth,
                    error=str(exc),
                )
                model_response = ""
                probe_error = True

            found = check_needle_found(model_response, magic_number)

            result.probes.append(
                NeedleResult(
                    context_length=ctx_len,
                    depth_position=depth,
                    needle_found=found,
                    model_response=model_response,
                    expected_answer=magic_number,
                    probe_error=probe_error,
                )
            )

    # Compute aggregated accuracy
    _compute_aggregated_accuracy(result)

    logger.info("needle_haystack_complete", **result.to_dict())
    return result


def _compute_aggregated_accuracy(result: NeedleHaystackResult) -> None:
    """Compute accuracy breakdowns by context length and depth.

    Probes where probe_error=True are excluded from accuracy calculations
    so transient API failures don't deflate model accuracy scores.
    """
    valid_probes = [p for p in result.probes if not p.probe_error]
    if not valid_probes:
        return

    # By context length
    len_groups: dict[int, list[bool]] = {}
    for p in valid_probes:
        len_groups.setdefault(p.context_length, []).append(p.needle_found)
    result.accuracy_by_length = {
        k: sum(v) / len(v) for k, v in sorted(len_groups.items())
    }

    # By depth
    depth_groups: dict[float, list[bool]] = {}
    for p in valid_probes:
        depth_groups.setdefault(p.depth_position, []).append(p.needle_found)
    result.accuracy_by_depth = {
        k: sum(v) / len(v) for k, v in sorted(depth_groups.items())
    }

    # Overall
    result.overall_accuracy = sum(p.needle_found for p in valid_probes) / len(valid_probes)
