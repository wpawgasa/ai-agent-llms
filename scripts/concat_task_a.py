#!/usr/bin/env python3
"""Concatenate Task A JSONL files of the same complexity level.

Usage:
    python scripts/concat_task_a.py FILE1 FILE2 [FILE3 ...] -o OUTPUT

    # Or let the script auto-detect level and choose all files for it:
    python scripts/concat_task_a.py --level L3 --input-dir data/output/task_a -o data/output/task_a/l3_merged.jsonl

Duplicate conversation_ids are resolved by appending a numeric suffix
(e.g. L3_001 → L3_001_2) so every record in the output is unique.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _level_of(path: Path) -> str | None:
    """Return the complexity level (e.g. 'L3') inferred from the filename."""
    m = re.match(r"(l[1-5])_", path.name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def concat_files(input_files: list[Path], output_path: Path, quiet: bool = False) -> None:
    levels = {_level_of(f) for f in input_files}
    levels.discard(None)
    if len(levels) > 1:
        sys.exit(f"Error: files span multiple complexity levels: {sorted(levels)}")
    if not levels:
        sys.exit("Error: cannot infer complexity level from filenames.")
    level = levels.pop()

    if not quiet:
        print(f"Merging {len(input_files)} file(s) for level {level} → {output_path}")

    seen_ids: dict[str, int] = {}   # id → occurrence count
    records: list[dict] = []

    for path in input_files:
        if not path.exists():
            sys.exit(f"Error: file not found: {path}")
        file_records = 0
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    sys.exit(f"Error: {path}:{lineno}: {exc}")

                conv_id: str = record.get("conversation_id", "")
                if conv_id in seen_ids:
                    seen_ids[conv_id] += 1
                    new_id = f"{conv_id}_{seen_ids[conv_id]}"
                    if not quiet:
                        print(f"  Duplicate '{conv_id}' in {path.name} → renamed to '{new_id}'")
                    record["conversation_id"] = new_id
                else:
                    seen_ids[conv_id] = 1

                records.append(record)
                file_records += 1

        if not quiet:
            print(f"  {path.name}: {file_records} records")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not quiet:
        print(f"Done. {len(records)} records written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate Task A JSONL files of the same complexity level."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        default=[],
        help="Input JSONL files (must all share the same level prefix).",
    )
    group.add_argument(
        "--level",
        metavar="LEVEL",
        help="Complexity level (e.g. L3). Auto-selects all matching files in --input-dir.",
    )

    parser.add_argument(
        "--input-dir",
        metavar="DIR",
        default="data/output/task_a",
        help="Directory to scan when --level is used (default: data/output/task_a).",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="OUTPUT",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    if args.level:
        input_dir = Path(args.input_dir)
        level = args.level.upper()
        prefix = level.lower() + "_"
        input_files = sorted(
            p for p in input_dir.glob(f"{prefix}*.jsonl")
            if p.resolve() != Path(args.output).resolve()
        )
        if not input_files:
            sys.exit(f"No files found for level {level} in {input_dir}")
    else:
        if len(args.files) < 2:
            parser.error("Provide at least 2 input files, or use --level.")
        input_files = [Path(f) for f in args.files]

    concat_files(input_files, Path(args.output), quiet=args.quiet)


if __name__ == "__main__":
    main()
