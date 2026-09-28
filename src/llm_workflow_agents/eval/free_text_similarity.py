"""Score a free-text tool argument by similarity, without letting facts slip.

`description`, `resolution_summary`, `waiver_reason` and their kind cannot be
exact-matched: the model writes a faithful paraphrase of the customer's words
and scores zero (CLM.md R31 — 120 of 540 failing arguments are this). Similarity
is the right instrument, with two constraints that calibration on 105 real
pairs made non-negotiable.

**Facts are checked exactly; only the prose is compared loosely.** The worst
false accept found was two SMS reminders scoring 0.92 lexical similarity while
differing in policy number, due date and amount — the only parts that matter:

    gold  'กธ.7076205678 ครบชำระเบี้ยฯ 20 มิถุนายน 2569 จำนวน 8,900 บาท'
    other 'กธ.7076201234 ครบชำระเบี้ยฯ 15 มิถุนายน 2569 จำนวน 4,250 บาท'

An embedding model scores that pair HIGHER, not lower, so more semantics makes
this failure worse. Every number and identifier in the reference must therefore
appear in the candidate, or the score is 0 whatever the prose similarity says.

**The default backend is deterministic and offline.** Token F1 over words, with
character bigrams for Thai (which has no word spaces). Calibrated separation:
at threshold 0.40 it accepts 78.1% of genuine paraphrases and 1.0% of other
rows' values. A benchmark metric that depends on a running server or a hosted
API stops being reproducible, and every comparability rule in this project
exists because a score drifted (R24-R31).

**Stopword damping was tried and measured worse.** Removing function words
looked obviously right — "Agent was rude and did not solve the issue" scores
0.421 against "Delivery was three days late and the driver was rude" on
{and, the, was, rude}. But on the 105 calibration pairs, at a matched 1.0%
false-accept rate it keeps 70.5% of paraphrases against 78.1% for the plain
form, so it was reverted. Pairs like that one sit inside the ~1% the threshold
knowingly admits; the operating point is chosen from the distribution, not from
the most annoying example.

**A stronger backend is a registry entry, not a rewrite.** The measured gap the
default cannot close is cross-language — gold in English, the model answering
in the conversation's Thai, which scores 0.00 lexically:

    gold 'Rear-end collision'   model 'รถชนท้าย'

That wants a multilingual sentence encoder. Register one with
:func:`register_backend`, pin its weights, and report which backend produced a
score, because two runs under different backends are not comparable.
"""

from __future__ import annotations

import collections
import re
import unicodedata
from typing import Any, Callable

#: Accepts 78.1% of real paraphrases and 1.0% of unrelated values (calibrated
#: on 105 pairs from the v4 benchmark; see the module docstring).
DEFAULT_THRESHOLD = 0.40

_WORD_SPLIT = re.compile(r"[^0-9a-z\u0e00-\u0e7f]+")
_THAI = re.compile(r"[\u0e00-\u0e7f]")
#: Numbers, and identifier-shaped tokens like POL-4471 or ACC7532.
_FACT = re.compile(r"\d[\d,.:/-]*\d|\d|[A-Za-z]{2,}[-_]?\d[\w-]*")
Backend = Callable[[str, str], float]
_BACKENDS: dict[str, Backend] = {}


def register_backend(name: str, backend: Backend) -> None:
    """Add a similarity backend (e.g. a pinned multilingual encoder)."""
    _BACKENDS[name] = backend


def available_backends() -> list[str]:
    return sorted({"token_f1", *_BACKENDS})


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    words = [w for w in _WORD_SPLIT.split(text) if w]
    if len(words) <= 2 and len(text) > 8:
        joined = _WORD_SPLIT.sub("", text)
        return [joined[i:i + 2] for i in range(len(joined) - 1)]
    out: list[str] = []
    for word in words:
        if _THAI.search(word) and len(word) > 4:
            out.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            out.append(word)
    return out


def token_f1(reference: str, candidate: str) -> float:
    """Harmonic mean of token precision and recall; Thai falls back to bigrams."""
    a, b = collections.Counter(_tokens(reference)), collections.Counter(_tokens(candidate))
    if not a or not b:
        return 1.0 if not a and not b else 0.0
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(b.values()), overlap / sum(a.values())
    return 2 * precision * recall / (precision + recall)


def facts_in(reference: Any, candidate: Any) -> bool:
    """Every number and identifier in *reference* also appears in *candidate*.

    Compared without separators, so ``8,900`` matches ``8900`` — a thousands
    comma is formatting, a different amount is not.
    """
    def normalize(token: str) -> str:
        return re.sub(r"[,.\s:/-]", "", token).casefold()

    haystack = normalize(str(candidate))
    return all(normalize(fact) in haystack for fact in _FACT.findall(str(reference)))


def similarity(reference: Any, candidate: Any, backend: str = "token_f1") -> float:
    """Similarity in [0, 1], or 0.0 when the candidate drops or changes a fact."""
    if not facts_in(reference, candidate):
        return 0.0
    if backend == "token_f1":
        return token_f1(str(reference), str(candidate))
    if backend not in _BACKENDS:
        raise ValueError(f"unknown similarity backend {backend!r}; have {available_backends()}")
    return float(_BACKENDS[backend](str(reference), str(candidate)))


def values_match(
    reference: Any,
    candidate: Any,
    threshold: float = DEFAULT_THRESHOLD,
    backend: str = "token_f1",
) -> bool:
    """Whether a free-text argument should count as correct."""
    return similarity(reference, candidate, backend=backend) >= threshold
