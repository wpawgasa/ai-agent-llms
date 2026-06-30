#!/usr/bin/env python3
"""Remove placeholder-fallback conversations from Task A JSONL datasets.

Reusable provenance filter that drops conversations the generator emitted as
*placeholder fallbacks* rather than real teacher output. When teacher generation
fails for a sample, the pipeline backfills a structurally-valid stub
(``generation_source == "placeholder_fallback"``) -- canned filler such as an
assistant turn ``"Handling Order Management in state GREETING."``, tool args
``{"placeholder": "value"}`` and templated user turns. These pass every
deterministic structural validator (STATE/tool/JSON are intact) but must not
train a model on "realistic dialogue", so they need a provenance-aware pass to
remove. This is that pass; it is *not* a structural validator and *not* a
corruption cleaner (for garbled Thai prose see
``clean_corrupted_conversations.py``).

Selection is driven by the ``generation_source`` field. Two mutually exclusive
modes:

  * **drop-list (default):** remove rows whose ``generation_source`` is in the
    drop set (default ``{"placeholder_fallback"}``). Rows missing the field are
    KEPT -- only explicit matches are dropped. Override with ``--drop``.
  * **allowlist:** with ``--keep-only SRC [...]``, remove any row whose
    ``generation_source`` is NOT one of the listed sources. A row missing the
    field is treated as source ``"<missing>"`` and therefore removed (it is not
    in the allowlist); this is reported so the drop is never silent.

Round-trip safe: surviving rows are written back byte-for-byte from the original
file (no JSON re-serialization), so key order / unicode escaping never drift.

Usage:
    # Drop placeholder_fallback rows in place (writes <file>.bak backup first):
    python scripts/remove_placeholder_conversations.py data/output/sft/task_a_merged/l2_merged_20260630.jsonl

    # Preview only -- report what would be removed, write nothing:
    python scripts/remove_placeholder_conversations.py FILE --dry-run

    # Keep ONLY genuine teacher output (drop everything else):
    python scripts/remove_placeholder_conversations.py FILE --keep-only teacher

    # Drop several provenance tags at once:
    python scripts/remove_placeholder_conversations.py FILE --drop placeholder_fallback retry_stub

    # Clean to a new file, keeping the original untouched, with a report sidecar:
    python scripts/remove_placeholder_conversations.py FILE -o cleaned.jsonl --report
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Sources removed by default in drop-list mode.
DEFAULT_DROP_SOURCES: tuple[str, ...] = ("placeholder_fallback",)

# Sentinel source for rows that carry no ``generation_source`` field at all.
MISSING_SOURCE = "<missing>"


def record_source(record: dict) -> str:
    """Return a record's ``generation_source`` (or ``MISSING_SOURCE`` if absent)."""
    src = record.get("generation_source")
    return src if isinstance(src, str) and src else MISSING_SOURCE


def removal_reason(
    record: dict,
    drop: tuple[str, ...],
    keep_only: tuple[str, ...] | None,
) -> str | None:
    """Return why a record should be removed, or ``None`` to keep it.

    ``keep_only`` (allowlist mode) takes precedence when provided: a row is
    removed unless its source is in the allowlist. Otherwise drop-list mode
    removes rows whose source is in ``drop``.
    """
    src = record_source(record)
    if keep_only is not None:
        if src not in keep_only:
            return f"source '{src}' not in keep-only {list(keep_only)}"
        return None
    if src in drop:
        return f"source '{src}'"
    return None


@dataclass
class RemovedRow:
    line_num: int
    conversation_id: str
    source: str
    reason: str


@dataclass
class FilterReport:
    input_path: Path
    output_path: Path | None
    total: int = 0
    kept: int = 0
    removed: list[RemovedRow] = field(default_factory=list)
    malformed_lines: list[int] = field(default_factory=list)

    @property
    def removed_count(self) -> int:
        return len(self.removed)

    def to_dict(self) -> dict:
        return {
            "input": str(self.input_path),
            "output": str(self.output_path) if self.output_path else None,
            "total": self.total,
            "kept": self.kept,
            "removed_count": self.removed_count,
            "malformed_lines": self.malformed_lines,
            "removed": [
                {
                    "line_num": r.line_num,
                    "conversation_id": r.conversation_id,
                    "source": r.source,
                    "reason": r.reason,
                }
                for r in self.removed
            ],
        }


def select_lines(
    raw_lines: list[str],
    drop: tuple[str, ...] = DEFAULT_DROP_SOURCES,
    keep_only: tuple[str, ...] | None = None,
) -> tuple[list[str], FilterReport]:
    """Split raw JSONL lines into (kept_raw_lines, report). Pure / side-effect free.

    Kept lines are returned verbatim (byte-for-byte) so output never re-serializes.
    Malformed (non-JSON) lines are conservatively KEPT and recorded -- this tool's
    job is provenance filtering, not structural validation.
    """
    report = FilterReport(input_path=Path("<lines>"), output_path=None)
    kept: list[str] = []
    for i, raw in enumerate(raw_lines, 1):
        if not raw.strip():
            continue
        report.total += 1
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            report.malformed_lines.append(i)
            kept.append(raw)
            report.kept += 1
            continue
        reason = removal_reason(record, drop, keep_only)
        if reason is not None:
            report.removed.append(
                RemovedRow(
                    line_num=i,
                    conversation_id=str(record.get("conversation_id", f"line {i}")),
                    source=record_source(record),
                    reason=reason,
                )
            )
        else:
            kept.append(raw)
            report.kept += 1
    return kept, report


def filter_file(
    input_path: Path,
    output_path: Path | None = None,
    drop: tuple[str, ...] = DEFAULT_DROP_SOURCES,
    keep_only: tuple[str, ...] | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> FilterReport:
    """Filter one JSONL file. Returns a report; writes output unless ``dry_run``.

    ``output_path=None`` => filter in place (a ``<input>.bak`` copy is written
    first unless ``backup=False``). Otherwise the surviving data goes to
    ``output_path`` and the input is left untouched.
    """
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    kept, report = select_lines(raw_lines, drop, keep_only)
    report.input_path = input_path

    in_place = output_path is None
    target = input_path if in_place else output_path

    if dry_run:
        report.output_path = None
        return report

    # Filtering in place with nothing to remove is a no-op: leave the file
    # untouched (no rewrite, no mtime churn). An explicit -o output is always
    # written so the caller reliably gets a destination file.
    if in_place and report.removed_count == 0:
        report.output_path = None
        return report

    report.output_path = target
    payload = "".join(line + "\n" for line in kept)
    if in_place and backup and report.removed_count:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
    # Atomic-ish write: temp sibling then replace.
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    return report


def _print_report(report: FilterReport, dry_run: bool, quiet: bool) -> None:
    tag = "[DRY RUN] would remove" if dry_run else "removed"
    print(
        f"{report.input_path}: {report.total} rows, kept {report.kept}, "
        f"{tag} {report.removed_count}",
        file=sys.stderr,
    )
    if report.malformed_lines:
        print(
            f"  warning: {len(report.malformed_lines)} malformed line(s) kept "
            f"(not JSON): {report.malformed_lines[:10]}",
            file=sys.stderr,
        )
    if not quiet:
        for r in report.removed:
            print(
                f"  - line {r.line_num} {r.conversation_id}: {r.reason}",
                file=sys.stderr,
            )
    if not dry_run and report.output_path:
        print(f"  -> wrote {report.output_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove placeholder-fallback conversations from Task A JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL file(s) to filter")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (single input only). Default: filter in place with a .bak backup.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--drop",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help=f"generation_source values to remove. Default: {list(DEFAULT_DROP_SOURCES)}.",
    )
    mode.add_argument(
        "--keep-only",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="Allowlist mode: remove every row whose generation_source is NOT listed "
        "(e.g. --keep-only teacher). Rows missing the field are removed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed; write nothing.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a JSON report sidecar next to each input (<input>.placeholder-report.json).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the .bak backup when filtering in place.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the per-file summary, not every removed conversation.",
    )
    args = parser.parse_args(argv)

    if args.output is not None and len(args.inputs) > 1:
        parser.error("-o/--output cannot be used with multiple input files")

    keep_only = tuple(args.keep_only) if args.keep_only is not None else None
    drop = tuple(args.drop) if args.drop is not None else DEFAULT_DROP_SOURCES

    total_removed = 0
    for input_path in args.inputs:
        if not input_path.is_file():
            print(f"error: not a file: {input_path}", file=sys.stderr)
            return 2
        report = filter_file(
            input_path,
            output_path=args.output,
            drop=drop,
            keep_only=keep_only,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        _print_report(report, args.dry_run, args.quiet)
        total_removed += report.removed_count
        if args.report:
            sidecar = input_path.with_suffix(input_path.suffix + ".placeholder-report.json")
            sidecar.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  -> report {sidecar}", file=sys.stderr)

    print(
        f"done: {total_removed} conversation(s) flagged across {len(args.inputs)} file(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
