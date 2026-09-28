"""Apply declared argument owners before a tool call is scored.

Exact match is the wrong instrument for a free-text argument: the model writes
a faithful paraphrase of what the customer said and scores zero. 120 of the 540
failing arguments on the best Cat A checkpoint are that (CLAUDE.md R31), and no
amount of training fixes a measurement.

This module canonicalizes a *prediction* against the gold call it is paired
with, under a declaration of who owns each argument:

``free_text``   accepted when :func:`free_text_similarity.values_match` says so
                — fact-guarded, so a changed number or identifier still fails.
``system``      supplied, because at serving time the runtime would supply it.
anything else   left exactly as the model wrote it.

Canonicalizing the prediction rather than loosening the comparator is what
keeps this small: every downstream metric — sub-tree F1, name accuracy,
argument exact match, hallucination rate — sees an ordinary tool call and needs
no change at all.

**The declaration lives in a ledger, not in the corpus.** The tool schemas sit
inside frozen benchmark stages (`task_a_benchmark_v4`), and a metric decision
should not require editing frozen data. `data/interim/task_a_argument_sources/
sources.json` is built by `scripts/propose_argument_sources.py` and then
REVIEWED by hand — the rule proposed 22 free-text pairs for v4 and 4 were wrong
in kind. Only pairs that differ from the default are listed, so nothing leaves
scoring by omission.

**This changes the number.** Every Task A benchmark result recorded before
2026-09-28 is on the exact-match rule and is not comparable with one recorded
after it; the result JSON records the ledger, its hash, the backend and the
threshold so a score always says which rule produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from llm_workflow_agents.eval.free_text_similarity import DEFAULT_THRESHOLD, values_match
from llm_workflow_agents.eval.tool_call_f1 import _deep_equals
from llm_workflow_agents.eval.tool_call_failures import align_calls

#: Reviewed declaration for the Task A benchmark; see the module docstring.
#: Anchored at the repo root, not the working directory: a relative default
#: would silently fall back to exact match whenever a runner starts elsewhere,
#: and a metric must not change with the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER = _REPO_ROOT / "data/interim/task_a_argument_sources/sources.json"

FREE_TEXT = "free_text"
SYSTEM = "system"


@dataclass(frozen=True)
class ArgumentScoring:
    """How a tool argument is compared, and the provenance of that decision."""

    sources: Mapping[str, str] = field(default_factory=dict)
    threshold: float = DEFAULT_THRESHOLD
    backend: str = "token_f1"
    ledger: str | None = None
    sha256: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.sources)

    def source_of(self, tool: str, argument: str) -> str | None:
        return self.sources.get(f"{tool}.{argument}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": "declared_sources" if self.enabled else "exact_match",
            "ledger": self.ledger,
            "ledger_sha256": self.sha256,
            "backend": self.backend,
            "threshold": self.threshold,
            "declared_pairs": len(self.sources),
        }


#: Scores every argument by exact match — the rule in force before 2026-09-28.
EXACT_MATCH = ArgumentScoring()


def load_ledger(
    path: Path | str | None,
    threshold: float = DEFAULT_THRESHOLD,
    backend: str = "token_f1",
) -> ArgumentScoring:
    """Read a reviewed sources ledger. A missing path scores by exact match."""
    if path is None:
        return EXACT_MATCH
    path = Path(path)
    if not path.exists():
        return EXACT_MATCH
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    return ArgumentScoring(
        sources=dict(document.get("sources") or {}),
        threshold=threshold,
        backend=backend,
        ledger=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def canonicalize_call(
    gold: dict[str, Any],
    predicted: dict[str, Any] | None,
    scoring: ArgumentScoring,
    counters: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """One predicted call with the declared owners applied.

    A free-text argument judged equivalent is rewritten to the gold value so the
    ordinary sub-tree match accepts it; a system-supplied argument is filled in.
    Every other argument is returned exactly as the model wrote it, and a call
    the model never made stays missing.
    """
    if predicted is None or not scoring.enabled:
        return predicted

    def count(key: str) -> None:
        if counters is not None:
            counters[key] = counters.get(key, 0) + 1

    tool = gold.get("name") or ""
    arguments = dict(predicted.get("arguments") or {})
    for argument, expected in (gold.get("arguments") or {}).items():
        source = scoring.source_of(tool, argument)
        if source == SYSTEM:
            arguments[argument] = expected
            count("system_supplied")
        elif source == FREE_TEXT:
            written = arguments.get(argument)
            if written is None or _deep_equals(written, expected):
                continue
            if values_match(expected, written, threshold=scoring.threshold, backend=scoring.backend):
                arguments[argument] = expected
                count("free_text_accepted")
            else:
                count("free_text_rejected")
    return {**predicted, "arguments": arguments}


def canonicalize_conversation(
    gold_by_turn: list[list[dict[str, Any]]],
    predicted_by_turn: list[list[dict[str, Any]]],
    scoring: ArgumentScoring,
    counters: dict[str, int] | None = None,
) -> list[list[dict[str, Any]]]:
    """Canonicalize a whole conversation, keeping each call on its own turn.

    Gold and predicted calls are pooled across the conversation before pairing,
    the same way the conversation-level tool metric pools them — a model that
    makes the right call one turn late is still answering the same gold call.
    Each rewritten call is then returned to the turn it was predicted on, so
    the per-turn metric sees an unchanged shape.
    """
    if not scoring.enabled:
        return predicted_by_turn

    flat_gold = [call for turn in gold_by_turn for call in turn]
    flat_predicted = [call for turn in predicted_by_turn for call in turn]
    # align_calls hands back the very objects passed in, so identity maps a
    # rewritten call to the turn it came from.
    rewritten: dict[int, dict[str, Any]] = {}
    for gold, predicted in align_calls(flat_gold, flat_predicted):
        if predicted is None:
            continue
        rewritten[id(predicted)] = canonicalize_call(gold, predicted, scoring, counters)
    return [[rewritten.get(id(call), call) for call in turn] for turn in predicted_by_turn]
