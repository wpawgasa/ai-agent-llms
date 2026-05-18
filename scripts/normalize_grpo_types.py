#!/usr/bin/env python3
"""Normalize leaf-value types in the GRPO JSONL corpus.

The synthetic GRPO data emits inconsistent types for the same logical field
across rows — e.g. ``ground_truth.tool_calls[].arguments.amount`` is ``int``
in most rows, ``float`` in some, ``str`` in a few. ``datasets.load_dataset``
with the JSON builder uses pyarrow's block-by-block schema inference and
aborts with ``ArrowInvalid: ... changed from string to number in row 4``.

This script makes the corpus type-consistent so that *any* loader (pyarrow,
``Dataset.from_list``, manual ``json.loads``) sees a unified schema:

  Pass 1 — survey: walk every leaf path in every row and tally non-null types.
  Pick canonical: for each mixed-type path, pick the majority non-null type.
                  Special case: ``{int, float}`` → ``float`` (preserves numeric
                  precision; matches the int/float interop in
                  ``eval/tool_call_f1.py::_deep_equals``).
  Pass 2 — coerce: rewrite outliers to the canonical type when round-trippable;
                   replace with ``None`` and count a failure otherwise.

Usage::

    python scripts/normalize_grpo_types.py \\
        --input-dir  data/output/grpo/task_a \\
        --output-dir data/output/grpo/task_a_normalized

    # Inspect counts without writing output:
    python scripts/normalize_grpo_types.py \\
        --input-dir data/output/grpo/task_a \\
        --output-dir data/output/grpo/task_a_normalized \\
        --dry-run

Scope: surveys every leaf in every row. For Task A GRPO this primarily
normalizes paths under ``ground_truth.tool_calls[].arguments`` and
``messages[].annotations.tool_calls[].arguments``; other subtrees
(``workflow_graph``, ``tool_schemas``, etc.) are type-stable today and pass
through unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

DEFAULT_INPUT = Path("data/output/grpo/task_a")
DEFAULT_OUTPUT = Path("data/output/grpo/task_a_normalized")
DEFAULT_SPLITS = ("train", "validation")

Path_T = tuple[str, ...]


def _type_of(obj: Any) -> str:
    """Return the pyarrow-relevant type name for a node.

    Containers report ``"dict"`` / ``"list"`` (not their element types), so
    the survey detects ``list`` vs ``str`` mismatches at the same path —
    which pyarrow's JSON reader treats as a fatal schema change.
    """
    if isinstance(obj, dict):
        return "dict"
    if isinstance(obj, list):
        return "list"
    return type(obj).__name__


def _walk_nodes(obj: Any, path: Path_T = ()) -> Iterator[tuple[Path_T, str]]:
    """Yield (path, type_name) for every node (both containers and leaves).

    List indices collapse to ``"[]"`` so all list elements share a path key
    (pyarrow infers one element schema per list anyway).
    """
    yield path, _type_of(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_nodes(v, path + (str(k),))
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_nodes(item, path + ("[]",))


def _survey(
    records: list[dict],
) -> tuple[dict[Path_T, Counter], dict[Path_T, set[frozenset[str]]]]:
    """Survey types per path and (for dicts) the set of key signatures observed.

    Returns ``(type_counts, dict_key_signatures)`` where ``dict_key_signatures``
    maps a dict-bearing path to the set of frozensets of its child keys seen
    across rows. Used to detect heterogeneous-key dicts (e.g. tool-call
    ``arguments`` where each call has a different schema) — pyarrow's JSON
    reader can't unify their structs even when leaf types agree.
    """
    type_counts: dict[Path_T, Counter] = defaultdict(Counter)
    dict_keys: dict[Path_T, set[frozenset[str]]] = defaultdict(set)

    def _walk(obj: Any, path: Path_T) -> None:
        t = _type_of(obj)
        if path:
            type_counts[path][t] += 1
        if isinstance(obj, dict):
            if path:
                dict_keys[path].add(frozenset(obj.keys()))
            for k, v in obj.items():
                _walk(v, path + (str(k),))
        elif isinstance(obj, list):
            for item in obj:
                _walk(item, path + ("[]",))

    for rec in records:
        _walk(rec, ())
    return type_counts, dict_keys


HETEROGENEOUS_DICT_THRESHOLD = 4
"""A dict path with more than this many distinct key signatures across rows
is treated as a free-form mapping (e.g., tool-call ``arguments`` where each
call has different keys) and stringified — pyarrow can't unify them even
when leaf types agree."""


def _should_stringify_dict(sigs: set[frozenset[str]]) -> bool:
    """Decide whether a dict-bearing path should be serialized to JSON string.

    Two cases:
      1. Many distinct key signatures (>HETEROGENEOUS_DICT_THRESHOLD) — the
         dict is a free-form mapping (e.g. tool ``arguments``).
      2. Few signatures but **no common required-key core** — i.e. the
         signatures' intersection is empty, indicating disjoint record
         shapes (e.g. a corrupted row where a list-of-tool-calls actually
         contains messages). pyarrow can handle missing optional fields
         (nullable struct fields), but it can't unify structurally
         disjoint records.
    """
    if len(sigs) < 2:
        return False
    if len(sigs) > HETEROGENEOUS_DICT_THRESHOLD:
        return True
    return not frozenset.intersection(*sigs)


def _pick_canonical(
    type_counts: dict[Path_T, Counter],
    dict_keys: dict[Path_T, set[frozenset[str]]] | None = None,
) -> dict[Path_T, str]:
    """Choose canonical type for each path that needs coercion.

    Heuristics, in order:
      1. **Heterogeneous-key dict** (see ``_should_stringify_dict``) → ``str``.
      2. Single non-null type → no coercion.
      3. ``{int, float}`` only → ``float`` (lossless for ints; matches
         ``_deep_equals`` int/float interop at ``eval/tool_call_f1.py:158``).
      4. ``{int, float, str}`` only and numeric count ≥ string count →
         ``float``; otherwise ``str``.
      5. Fallback → majority non-null type.
    """
    canonical: dict[Path_T, str] = {}

    if dict_keys:
        for path, sigs in dict_keys.items():
            if _should_stringify_dict(sigs):
                canonical[path] = "str"

    for path, counts in type_counts.items():
        if path in canonical:
            continue
        non_null = {t: c for t, c in counts.items() if t != "NoneType"}
        if len(non_null) <= 1:
            continue
        types = set(non_null)
        if types.issubset({"int", "float"}):
            canonical[path] = "float"
            continue
        if types.issubset({"int", "float", "str"}):
            numeric = non_null.get("int", 0) + non_null.get("float", 0)
            str_count = non_null.get("str", 0)
            canonical[path] = "float" if numeric >= str_count else "str"
            continue
        canonical[path] = max(non_null.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return canonical


def _coerce_value(value: Any, target: str) -> tuple[Any, bool]:
    """Coerce ``value`` to ``target`` type. Returns (new_value, ok)."""
    if value is None:
        return None, True
    actual = _type_of(value)
    if actual == target:
        return value, True

    if target == "list":
        # Comma-separated strings → list of trimmed tokens (common case for
        # IDs like "SW-X100-01,FB-PRO5-02"). Other scalars → wrap as [value].
        if isinstance(value, str):
            if "," in value:
                parts = [s.strip() for s in value.split(",") if s.strip()]
                return parts or [value], True
            return [value], True
        if isinstance(value, dict):
            return [value], True
        return [value], True

    if target == "str":
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False), True
        return str(value), True

    # Container value but scalar target — drop with failure.
    if isinstance(value, (dict, list)):
        return None, False

    if target == "float":
        if isinstance(value, bool):
            return float(value), True
        if isinstance(value, (int, float)):
            return float(value), True
        if isinstance(value, str):
            try:
                return float(value), True
            except (ValueError, TypeError):
                return None, False
        return None, False

    if target == "int":
        if isinstance(value, bool):
            return int(value), True
        if isinstance(value, int):
            return value, True
        if isinstance(value, float):
            return (int(value), True) if value.is_integer() else (None, False)
        if isinstance(value, str):
            try:
                f = float(value)
            except (ValueError, TypeError):
                return None, False
            return (int(f), True) if f.is_integer() else (None, False)
        return None, False

    if target == "bool":
        if isinstance(value, bool):
            return value, True
        if isinstance(value, int) and value in (0, 1):
            return bool(value), True
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true", True
        return None, False

    if target == "dict":
        return (value, True) if isinstance(value, dict) else (None, False)

    return value, True


def _coerce_walk(
    obj: Any,
    path: Path_T,
    canonical: dict[Path_T, str],
    coercions: Counter,
    failures: Counter,
) -> Any:
    """Recursively coerce ``obj`` so each node matches the canonical type at its path.

    Containers are coerced themselves if the canonical at their path is a
    different shape (e.g., a stray ``str`` at a path whose canonical is
    ``list`` is wrapped/split into a list).
    """
    target = canonical.get(path)
    actual = _type_of(obj)

    # Container coerced to non-container: collapse without recursing.
    if isinstance(obj, (dict, list)) and target is not None and actual != target:
        new_val, ok = _coerce_value(obj, target)
        if ok:
            coercions[path] += 1
        else:
            failures[path] += 1
        return new_val

    if isinstance(obj, dict):
        return {
            k: _coerce_walk(v, path + (str(k),), canonical, coercions, failures)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            _coerce_walk(item, path + ("[]",), canonical, coercions, failures)
            for item in obj
        ]

    # Scalar leaf.
    if target is None or obj is None or actual == target:
        return obj
    new_val, ok = _coerce_value(obj, target)
    if ok:
        coercions[path] += 1
    else:
        failures[path] += 1
    return new_val


def _load_split(src: Path) -> list[dict]:
    if not src.exists():
        sys.exit(f"Error: missing split file: {src}")
    records: list[dict] = []
    with open(src) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"Error: {src}:{lineno}: {exc}")
    return records


def _write_split(dst: Path, records: list[dict]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")


def _format_path(path: Path_T) -> str:
    return ".".join(path) if path else "<root>"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize leaf-value types in GRPO JSONL data.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="DIR",
        help=f"Source directory (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="DIR",
        help=f"Destination directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        metavar="SPLIT",
        help=f"Splits to process (default: {' '.join(DEFAULT_SPLITS)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without writing any files.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Error: input directory not found: {args.input_dir}")

    print(f"Input dir  : {args.input_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Splits     : {args.splits}")
    if args.dry_run:
        print("[dry-run] no files will be written")
    print("=" * 72)

    # Pass 1: load all splits, survey type frequencies + dict key signatures
    # globally so the canonical type for each path is consistent across
    # train/validation.
    split_records: dict[str, list[dict]] = {}
    combined_counts: dict[Path_T, Counter] = defaultdict(Counter)
    combined_dict_keys: dict[Path_T, set[frozenset[str]]] = defaultdict(set)
    for split in args.splits:
        records = _load_split(args.input_dir / f"{split}.jsonl")
        split_records[split] = records
        type_counts, dict_keys = _survey(records)
        for path, counts in type_counts.items():
            combined_counts[path].update(counts)
        for path, sigs in dict_keys.items():
            combined_dict_keys[path] |= sigs
        print(f"  loaded   {split:11s}: {len(records):>6d} rows")

    canonical = _pick_canonical(combined_counts, combined_dict_keys)
    n_heterogeneous = sum(
        1 for p, t in canonical.items()
        if t == "str" and p in combined_dict_keys and _should_stringify_dict(combined_dict_keys[p])
    )
    print(
        f"  canonical paths: {len(canonical)} total "
        f"({n_heterogeneous} heterogeneous dicts → str)"
    )
    print("=" * 72)

    # Pass 2: coerce each split using the global canonical map.
    grand_rows = 0
    grand_coercions: Counter = Counter()
    grand_failures: Counter = Counter()

    for split in args.splits:
        records = split_records[split]
        coercions: Counter = Counter()
        failures: Counter = Counter()
        if canonical:
            records = [
                _coerce_walk(rec, (), canonical, coercions, failures)
                for rec in records
            ]
        if not args.dry_run:
            _write_split(args.output_dir / f"{split}.jsonl", records)

        grand_rows += len(records)
        grand_coercions.update(coercions)
        grand_failures.update(failures)

        n_coerced = sum(coercions.values())
        n_failed = sum(failures.values())
        print(
            f"  {split:11s}: {len(records):>6d} rows   "
            f"coerced: {n_coerced:>4d}   "
            f"failed: {n_failed:>3d}"
        )

    print("=" * 72)
    print(f"Total rows         : {grand_rows}")
    print(f"Distinct mixed paths: {len(canonical)}")
    print(f"Coercions          : {sum(grand_coercions.values())}")
    print(f"Failures           : {sum(grand_failures.values())}")

    if canonical:
        print()
        print("Per-path detail (top 25 by coercion count):")
        top = sorted(canonical.keys(), key=lambda p: -grand_coercions.get(p, 0))[:25]
        for p in top:
            print(
                f"  {grand_coercions[p]:>4d} coerced "
                f"({grand_failures[p]:>2d} failed)  → {canonical[p]:<5s}  "
                f"{_format_path(p)}"
            )

    if args.dry_run:
        print("\n[dry-run] no files written.")
    else:
        print(f"\nNormalized splits written to: {args.output_dir}")


if __name__ == "__main__":
    main()
