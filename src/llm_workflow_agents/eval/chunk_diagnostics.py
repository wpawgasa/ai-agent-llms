"""Reference-free diagnostics for voice chunking.

Guardrails, never composite terms. Chunk formatting is cheap to install
through fine-tuning, so letting it move the Phase 1 winner would select on the
wrong capability. These metrics diagnose a candidate; they do not rank one.

Not measured against gold: the benchmark scores free generation, so the
model's spoken words differ from the gold spoken words, and once the words
differ the chunk boundaries cannot be aligned. A gold-referenced chunk F1
would compare boundaries in two different sentences and return noise. All
three metrics below read the prediction alone:

- ``first_chunk_p50``/``first_chunk_p90`` is the latency metric: the
  orchestrator starts text-to-speech on chunk 1 while the model is still
  writing chunk 2, so a long opening chunk delays audio proportionally.
- ``chunk_len_p50``/``chunk_len_p90`` and ``chunks_per_turn_p50`` catch what
  format checking cannot see: one long chunk per turn satisfies every format
  rule and still streams badly.
- ``boundary_quality`` is the share of chunks ending at a real pause point
  (Thai sentence-final particles, English terminal punctuation).

Language is per-conversation, not per-run: the benchmark's voice stratum
draws English and Thai at even odds by design (see
``docs/superpowers/specs/2026-08-21-voice-benchmark-and-prompt-switch-design.md``
section 4), so a single ``language`` argument is only correct for one
conversation's chunks. :func:`chunk_diagnostics_by_language` scores each
language group under its own convention and pools the underlying
measurements before any percentile is computed, so a caller holding a mixed
stratum should use it instead of collapsing the stratum to one language.
"""

from __future__ import annotations

import statistics
from typing import Any

from llm_workflow_agents.data.voice_convention import iter_chunks

#: Thai sentence-final particles. They mark a pause point explicitly, which
#: makes boundary quality unusually tractable in Thai.
_THAI_FINALS = ("ค่ะ", "คะ", "ครับ", "นะคะ", "นะครับ", "ค่า")
_EN_FINALS = (".", "?", "!", "…")


def _ends_well(chunk: str, language: str) -> bool:
    text = chunk.strip()
    if not text:
        return False
    if language == "th":
        return text.endswith(_THAI_FINALS)
    if language == "code_switch":
        return text.endswith(_THAI_FINALS) or text.endswith(_EN_FINALS)
    return text.endswith(_EN_FINALS)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])


def _collect_raw(
    completions: list[str], language: str
) -> tuple[list[float], list[float], list[float], int, int]:
    """Return (first_lengths, all_lengths, counts, well_ended, total_chunks).

    One language's worth of raw, unaggregated measurements — the shared core
    of both :func:`chunk_diagnostics` and :func:`chunk_diagnostics_by_language`.
    A turn with no chunks (legal — a silent tool-call-only turn) contributes
    nothing here rather than counting as a poor boundary.
    """
    first_lengths: list[float] = []
    all_lengths: list[float] = []
    counts: list[float] = []
    well_ended = 0
    total_chunks = 0

    for text in completions:
        chunks = iter_chunks(text or "")
        if not chunks:
            continue
        counts.append(len(chunks))
        first_lengths.append(len(chunks[0]))
        for c in chunks:
            all_lengths.append(len(c))
            total_chunks += 1
            if _ends_well(c, language):
                well_ended += 1

    return first_lengths, all_lengths, counts, well_ended, total_chunks


def _summarize(
    first_lengths: list[float],
    all_lengths: list[float],
    counts: list[float],
    well_ended: int,
    total_chunks: int,
) -> dict[str, Any]:
    return {
        "first_chunk_p50": _pct(first_lengths, 0.5),
        "first_chunk_p90": _pct(first_lengths, 0.9),
        "chunk_len_p50": _pct(all_lengths, 0.5),
        "chunk_len_p90": _pct(all_lengths, 0.9),
        "chunks_per_turn_p50": statistics.median(counts) if counts else 0.0,
        "boundary_quality": (well_ended / total_chunks) if total_chunks else 0.0,
        "n_turns_with_chunks": len(counts),
    }


def chunk_diagnostics(
    completions: list[str], language: str = "en"
) -> dict[str, Any]:
    """Return chunk-shape diagnostics over a list of generated turns.

    A turn with no ``<S>...</S>`` chunks (e.g. a silent tool-call-only turn,
    which the voice format contract explicitly permits) is excluded from
    every metric here rather than scored as a poor boundary — see
    ``data/voice_convention.py`` rule 3 and CLAUDE.md R20.

    Every completion here is scored under the SAME ``language`` convention.
    Do not call this directly on a stratum that mixes languages — see
    :func:`chunk_diagnostics_by_language`.
    """
    return _summarize(*_collect_raw(completions, language))


def chunk_diagnostics_by_language(
    completions_by_language: dict[str, list[str]],
) -> dict[str, Any]:
    """Score a mixed-language voice stratum, one convention per language.

    Each language's completions are scored under their OWN boundary-quality
    convention via :func:`_collect_raw` (Thai sentence-final particles vs.
    English terminal punctuation) — never one convention applied to the whole
    stratum. Scoring everything under whichever language happens to have more
    conversations (a majority vote) would silently misapply the wrong
    convention to the minority language and could make a model's chunking
    look bad in a language it never mis-scored; the benchmark's voice
    stratum draws languages at even odds by design, so a majority-vote
    resolution is wrong on non-trivial mixes, not just a rare edge case.

    Combination choice: the underlying per-chunk measurements (chunk
    lengths, first-chunk lengths, chunk counts, well-ended counts) are
    POOLED across languages before any percentile or ratio is computed, not
    averaged after each language's percentiles are computed separately.
    ``_pct`` uses nearest-rank on a sorted list; nearest-rank has no
    algebraic identity under averaging two groups' already-computed
    percentiles, so an average-of-percentiles would not itself be a
    percentile of anything. Pooling first keeps every returned percentile a
    genuine percentile of the full stratum's chunks.

    The set of languages actually pooled is returned under ``"languages"``
    (sorted) so the combination performed is visible in the output, not
    implicit in the number.

    Args:
        completions_by_language: Per-language lists of raw generated turn
            text, e.g. ``{"en": [...], "th": [...]}``.

    Returns:
        The same keys as :func:`chunk_diagnostics`, plus ``"languages"``.
    """
    first_all: list[float] = []
    all_all: list[float] = []
    counts_all: list[float] = []
    well_ended_all = 0
    total_chunks_all = 0

    for language, completions in completions_by_language.items():
        first_lengths, all_lengths, counts, well_ended, total_chunks = _collect_raw(
            completions, language
        )
        first_all.extend(first_lengths)
        all_all.extend(all_lengths)
        counts_all.extend(counts)
        well_ended_all += well_ended
        total_chunks_all += total_chunks

    out = _summarize(first_all, all_all, counts_all, well_ended_all, total_chunks_all)
    out["languages"] = sorted(completions_by_language)
    return out
