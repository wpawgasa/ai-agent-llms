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


def chunk_diagnostics(
    completions: list[str], language: str = "en"
) -> dict[str, Any]:
    """Return chunk-shape diagnostics over a list of generated turns.

    A turn with no ``<S>...</S>`` chunks (e.g. a silent tool-call-only turn,
    which the voice format contract explicitly permits) is excluded from
    every metric here rather than scored as a poor boundary — see
    ``data/voice_convention.py`` rule 3 and CLAUDE.md R20.
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

    return {
        "first_chunk_p50": _pct(first_lengths, 0.5),
        "first_chunk_p90": _pct(first_lengths, 0.9),
        "chunk_len_p50": _pct(all_lengths, 0.5),
        "chunk_len_p90": _pct(all_lengths, 0.9),
        "chunks_per_turn_p50": statistics.median(counts) if counts else 0.0,
        "boundary_quality": (well_ended / total_chunks) if total_chunks else 0.0,
        "n_turns_with_chunks": len(counts),
    }
