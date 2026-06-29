#!/usr/bin/env python3
"""Remove teacher-corrupted conversations from Task A JSONL datasets.

Reusable cleaning pass that drops conversations whose prose is
teacher-transcription-corrupted (garbled Thai) while leaving every structurally
clean row untouched. It is *not* a structural validator -- STATE/tool/JSON of a
corrupted row usually stay intact, so the deterministic validators pass them;
this catches the quality defect they miss before the data feeds SFT.

Two detection layers, both conservative (calibrated for near-zero false
positives):

  1. The canonical project detector
     ``llm_workflow_agents.data.data_validator.detect_thai_corruption`` --
     Latin letters glued *inside* a Thai word (Thai-Latin-Thai, gated off for
     ``code_switch``) and obsolete Thai consonants (kho khuat ``ฃ`` / kho khon
     ``ฅ``). See that module for the calibration notes.

  2. A curated, language-agnostic ``KNOWN_THAI_GARBLE`` substring blocklist for
     phonetic corruption the structural signals cannot see (e.g. ``มัสดี``, a
     garble of ``สวัสดี`` "hello"). Every seed entry was confirmed to appear
     ONLY in genuinely corrupted rows across a full L1 corpus (zero false
     positives). This is the extension point: add new confirmed garbles here, or
     pass more at runtime via ``--garble-file`` (one substring per line).

Round-trip safe: surviving rows are written back byte-for-byte from the original
file (no JSON re-serialization), so key order / unicode escaping never drift.

Usage:
    # Clean in place (writes <file>.bak backup first):
    python scripts/clean_corrupted_conversations.py data/output/sft/task_a_merged/l1_merged_20260629.jsonl

    # Preview only -- report what would be removed, write nothing:
    python scripts/clean_corrupted_conversations.py FILE --dry-run

    # Clean to a new file, keeping the original untouched:
    python scripts/clean_corrupted_conversations.py FILE -o cleaned.jsonl

    # Clean several files, dropping a JSON report sidecar each:
    python scripts/clean_corrupted_conversations.py a.jsonl b.jsonl --report

    # Add extra garble substrings discovered later:
    python scripts/clean_corrupted_conversations.py FILE --garble-file extra_garble.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from llm_workflow_agents.data.data_validator import detect_thai_corruption

# --- Curated known-garble blocklist ------------------------------------------
# Phonetic / typo corruption of common Thai words that the calibrated structural
# signals in detect_thai_corruption() cannot catch (no Latin glued in, no
# obsolete consonant). Each entry is a substring that, when present in a non-
# system message, marks the conversation corrupted. Keep entries HIGH-confidence:
# only add a substring once it is confirmed to appear exclusively in corrupted
# rows (no legitimate Thai word contains it). Annotate the intended word.
KNOWN_THAI_GARBLE: tuple[str, ...] = (
    "มัสดี",          # garble of สวัสดี ("hello") -- corrupted greeting
    "เช้าหน้าที่",     # garble of เจ้าหน้าที่ ("officer/agent")
    "สฐานะ",          # garble of สถานะ ("status")
    "เบ็นระบบ",        # garble of เป็นระบบ ("systematic")
    "จริกๆ",          # garble of จริงๆ ("really")
    "ได๊รับ",          # garble of ได้รับ ("received") -- wrong tone mark
    "สวัสดิภาพนะคับ",  # garble of สวัสดิภาพนะครับ ("safety/farewell")
)


def detect_garble_blocklist(record: dict, garble: tuple[str, ...]) -> list[str]:
    """Return reasons a record hits the known-garble blocklist (empty => clean).

    Scans non-system message content only -- ``system`` prompts are
    machine-authored and not a corruption vector (mirrors detect_thai_corruption).
    """
    reasons: list[str] = []
    seen: set[str] = set()
    for msg in record.get("messages", []):
        if msg.get("role") == "system":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for needle in garble:
            if needle in content and needle not in seen:
                seen.add(needle)
                reasons.append(f"known-garble '{needle}'")
    return reasons


def detect_corruption(record: dict, garble: tuple[str, ...] = KNOWN_THAI_GARBLE) -> list[str]:
    """Combined corruption reasons for one record (empty list => keep the row)."""
    return detect_thai_corruption(record) + detect_garble_blocklist(record, garble)


@dataclass
class RemovedRow:
    line_num: int
    conversation_id: str
    language: str
    reasons: list[str]


@dataclass
class CleanReport:
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
                    "language": r.language,
                    "reasons": r.reasons,
                }
                for r in self.removed
            ],
        }


def clean_lines(
    raw_lines: list[str], garble: tuple[str, ...] = KNOWN_THAI_GARBLE
) -> tuple[list[str], CleanReport]:
    """Split raw JSONL lines into (kept_raw_lines, report). Pure / side-effect free.

    Kept lines are returned verbatim (byte-for-byte) so output never re-serializes.
    Malformed (non-JSON) lines are conservatively KEPT and recorded -- this tool's
    job is corruption removal, not structural validation.
    """
    report = CleanReport(input_path=Path("<lines>"), output_path=None)
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
        reasons = detect_corruption(record, garble)
        if reasons:
            report.removed.append(
                RemovedRow(
                    line_num=i,
                    conversation_id=str(record.get("conversation_id", f"line {i}")),
                    language=str(record.get("language", "")),
                    reasons=reasons,
                )
            )
        else:
            kept.append(raw)
            report.kept += 1
    return kept, report


def clean_file(
    input_path: Path,
    output_path: Path | None = None,
    garble: tuple[str, ...] = KNOWN_THAI_GARBLE,
    dry_run: bool = False,
    backup: bool = True,
) -> CleanReport:
    """Clean one JSONL file. Returns a report; writes output unless ``dry_run``.

    ``output_path=None`` => clean in place (a ``<input>.bak`` copy is written
    first unless ``backup=False``). Otherwise the cleaned data goes to
    ``output_path`` and the input is left untouched.
    """
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    kept, report = clean_lines(raw_lines, garble)
    report.input_path = input_path

    in_place = output_path is None
    target = input_path if in_place else output_path

    if dry_run:
        report.output_path = None
        return report

    # Cleaning in place with nothing to remove is a no-op: leave the file
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


def _print_report(report: CleanReport, dry_run: bool, quiet: bool) -> None:
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
                f"  - line {r.line_num} {r.conversation_id} ({r.language}): "
                f"{'; '.join(r.reasons)}",
                file=sys.stderr,
            )
    if not dry_run and report.output_path:
        print(f"  -> wrote {report.output_path}", file=sys.stderr)


def _load_extra_garble(path: Path) -> tuple[str, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(
        s.strip() for s in lines if s.strip() and not s.lstrip().startswith("#")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove teacher-corrupted conversations from Task A JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL file(s) to clean")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (single input only). Default: clean in place with a .bak backup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed; write nothing.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a JSON report sidecar next to each input (<input>.corruption-report.json).",
    )
    parser.add_argument(
        "--garble-file",
        type=Path,
        default=None,
        help="Extra known-garble substrings to add (one per line; '#' comments allowed).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the .bak backup when cleaning in place.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the per-file summary, not every removed conversation.",
    )
    args = parser.parse_args(argv)

    if args.output is not None and len(args.inputs) > 1:
        parser.error("-o/--output cannot be used with multiple input files")

    garble = KNOWN_THAI_GARBLE
    if args.garble_file is not None:
        garble = garble + _load_extra_garble(args.garble_file)

    total_removed = 0
    for input_path in args.inputs:
        if not input_path.is_file():
            print(f"error: not a file: {input_path}", file=sys.stderr)
            return 2
        report = clean_file(
            input_path,
            output_path=args.output,
            garble=garble,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        _print_report(report, args.dry_run, args.quiet)
        total_removed += report.removed_count
        if args.report:
            sidecar = input_path.with_suffix(input_path.suffix + ".corruption-report.json")
            sidecar.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  -> report {sidecar}", file=sys.stderr)

    print(f"done: {total_removed} conversation(s) flagged across {len(args.inputs)} file(s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
