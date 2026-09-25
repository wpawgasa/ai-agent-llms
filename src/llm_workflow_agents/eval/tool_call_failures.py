"""Why a predicted tool call failed to match its ground truth.

`tool_call_f1` scores a call only when the name and every ground-truth argument
deep-equal (`_deep_equals`: same type, exact value). That is the right bar for a
headline metric, but it collapses four very different failures into one number:

    check_availability(date="2024-05-10")   gold date="May 10, 2024"
    check_availability(date="2024-05-11")   gold date="2024-05-10"

Both score zero. The first is a formatting difference — the model read the right
value and wrote it another way; the second is the model getting the date wrong.
Fixing the first is a comparator change; fixing the second needs training data.

This module buckets each failure so the two can be counted separately, because
the fix they imply differs by orders of magnitude in cost (CLAUDE.md R30 records
what skipping that question costs). It decides nothing: `tool_call_f1` is
untouched, and nothing here feeds a score.

The buckets, in the order they are tested:

``no_call``            the gold call has no predicted counterpart at all.
``wrong_name``         a call was made, naming a different tool.
``missing_argument``   the gold argument key is absent from the prediction.
``equivalent_value``   the values differ only by normalization — case,
                       surrounding whitespace, punctuation, a number written as
                       a string, a bool spelled out, or a date written another
                       way. A comparator fix would score these correct.
``value_in_context``   the gold value appears verbatim in what the model was
                       shown, and the model wrote something else: a copy
                       failure, the capability gap.
``value_unknowable``   the gold value appears NOWHERE in what the model was
                       shown. Nothing could have produced it (R28); scoring it
                       measures the corpus.

A paraphrase or translation is NOT a bucket of its own: it lands in
``value_in_context`` when the gold value was there to copy and in
``value_unknowable`` when it was not. Read samples from both buckets before
acting on the counts — that judgement is the one thing this module cannot make.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_DATE_SEPARATORS = re.compile(r"[/\-.\s,]+")
_MONTHS = {
    m: i + 1
    for i, names in enumerate([
        ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
        ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
        ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
        ("december", "dec"),
    ])
    for m in names
}
_BOOLS = {"true": "true", "yes": "true", "1": "true", "false": "false", "no": "false", "0": "false"}


@dataclass(frozen=True)
class ArgumentFailure:
    """One ground-truth argument a predicted call did not reproduce."""

    tool: str
    argument: str
    expected: Any
    actual: Any
    bucket: str

    def describe(self) -> str:
        return f"{self.tool}.{self.argument}: expected {self.expected!r}, got {self.actual!r} [{self.bucket}]"


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # 100 and 100.0 and "100" are the same value written three ways.
        return str(int(value)) if float(value).is_integer() else str(float(value))
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loose(value: Any) -> str:
    """Alphanumerics only, casefolded — no case, spaces, punctuation or underscores.

    So ``ACC-4471``, ``acc 4471`` and ``Acc_4471`` share one form. Dropping
    spaces means ``a b`` also matches ``ab``; that is deliberate. It can only
    make a value look MORE visible in the context, which biases
    :func:`classify_argument` away from calling one unknowable — the safe
    direction, since "unknowable" is an accusation against the benchmark (R28).
    """
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    return re.sub(r"[^0-9a-z\u0e00-\u0e7f]+", "", text)


def _as_date(value: Any) -> str | None:
    """A date written any common way, as YYYY-MM-DD; None if it is not a date."""
    text = unicodedata.normalize("NFKC", _text(value)).casefold().strip()
    parts = [p for p in _DATE_SEPARATORS.split(text) if p]
    if not 2 <= len(parts) <= 3:
        return None
    year = month = day = None
    numbers: list[int] = []
    for part in parts:
        if part in _MONTHS:
            if month is not None:
                return None
            month = _MONTHS[part]
        elif part.isdigit() and len(part) == 4:
            if year is not None:
                return None
            year = int(part)
        elif part.isdigit():
            numbers.append(int(part))
        else:
            stripped = part.rstrip("stndrh")   # 1st, 2nd, 3rd, 4th
            if not stripped.isdigit():
                return None
            numbers.append(int(stripped))
    for number in numbers:
        if number > 12 and day is None:
            day = number
        elif month is None:
            month = number
        elif day is None:
            day = number
        elif year is None:
            year = number
        else:
            return None
    if year is None or month is None or day is None:
        return None
    if year < 100:
        year += 2000
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def values_equivalent(expected: Any, actual: Any) -> bool:
    """True when the two differ only by how the value is written.

    Case, surrounding and repeated whitespace, punctuation, a number written as
    a string, a bool spelled out, and dates in any common order or spelling.
    Deliberately NOT: a different date, a paraphrase, or a translation.
    """
    if isinstance(expected, (dict, list)) or isinstance(actual, (dict, list)):
        return json.dumps(expected, sort_keys=True, ensure_ascii=False) == json.dumps(
            actual, sort_keys=True, ensure_ascii=False
        )
    if _text(expected) == _text(actual):
        return True
    expected_date, actual_date = _as_date(expected), _as_date(actual)
    if expected_date and actual_date:
        return expected_date == actual_date
    expected_loose, actual_loose = _loose(expected), _loose(actual)
    if not expected_loose and not actual_loose:
        return True
    if expected_loose == actual_loose:
        return True
    return _BOOLS.get(expected_loose, object()) == _BOOLS.get(actual_loose, object())


def value_is_visible(value: Any, context: str) -> bool:
    """True when *value* appears in the text the model was shown.

    Compared loosely, so a value the model could have copied but re-cased or
    re-spaced still counts as visible — the question is whether it was THERE,
    not whether the model reproduced it byte for byte.
    """
    needle = _loose(value)
    if not needle or len(needle) < 2:
        return True   # too short to attribute; never call it unknowable
    return needle in _loose(context)


def classify_argument(
    tool: str,
    argument: str,
    expected: Any,
    actual: Any,
    context: str,
) -> ArgumentFailure | None:
    """Bucket one ground-truth argument, or None when the prediction matches."""
    if actual is _MISSING:
        return ArgumentFailure(tool, argument, expected, None, "missing_argument")
    if _deep_equal(expected, actual):
        return None   # the scorer already counts this as correct
    if values_equivalent(expected, actual):
        return ArgumentFailure(tool, argument, expected, actual, "equivalent_value")
    if value_is_visible(expected, context):
        return ArgumentFailure(tool, argument, expected, actual, "value_in_context")
    return ArgumentFailure(tool, argument, expected, actual, "value_unknowable")


# ``value_other`` is declared in the module docstring but is NOT produced here:
# every non-equivalent value is either visible in the context or it is not, and
# those two cover the space. A paraphrase of a visible value lands in
# value_in_context (the source was there to copy); a paraphrase of an absent one
# lands in value_unknowable. Read samples from both before acting on the counts.


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


_MISSING = _Missing()


def classify_call(
    expected_call: dict[str, Any],
    predicted_call: dict[str, Any] | None,
    context: str,
) -> list[ArgumentFailure]:
    """Every way *predicted_call* fails to reproduce *expected_call*.

    An empty list means the call matched on name and on every ground-truth
    argument — the same bar ``_is_subtree_match`` applies.
    """
    tool = expected_call.get("name", "")
    if predicted_call is None:
        return [ArgumentFailure(tool, "", expected_call.get("arguments"), None, "no_call")]
    if predicted_call.get("name") != tool:
        return [ArgumentFailure(tool, "", tool, predicted_call.get("name"), "wrong_name")]

    expected_args = expected_call.get("arguments") or {}
    predicted_args = predicted_call.get("arguments") or {}
    if not isinstance(expected_args, dict) or not isinstance(predicted_args, dict):
        return []
    failures: list[ArgumentFailure] = []
    for key, expected in expected_args.items():
        actual = predicted_args.get(key, _MISSING)
        if actual is not _MISSING and _deep_equal(expected, actual):
            continue
        failure = classify_argument(tool, key, expected, actual, context)
        if failure is not None:
            failures.append(failure)
    return failures


def _deep_equal(a: Any, b: Any) -> bool:
    """The comparison ``tool_call_f1`` itself makes, reproduced here."""
    from llm_workflow_agents.eval.tool_call_f1 import _deep_equals

    return _deep_equals(a, b)


def align_calls(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Pair gold calls with predicted ones by tool name, in order.

    The same rule the fixed chain-propagation metric uses (R27): by index breaks
    as soon as the model makes one extra or one fewer call, which is exactly the
    population under study here.
    """
    remaining = list(predicted)
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for gold in expected:
        match = next((p for p in remaining if p.get("name") == gold.get("name")), None)
        if match is not None:
            remaining.remove(match)
        pairs.append((gold, match))
    return pairs
