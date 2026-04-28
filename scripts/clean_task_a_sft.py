#!/usr/bin/env python3
"""Clean Task A SFT JSONL data before fine-tuning.

Three fixes applied in a single pass:
  1. Drop truncated rows (no non-system turns).
  2. Drop role-confused tool messages (role:"tool" with <tool_call> content).
  3. Flag empty terminal_state as ground_truth.terminal_reached=false so the
     GRPO reward function can skip the completion sub-reward on those rows
     rather than silently scoring them 0.

Usage:
    python scripts/clean_task_a_sft.py \\
        --input-dir data/output/sft/task_a \\
        --output-dir data/output/sft/task_a_cleaned

    # Inspect counts without writing output:
    python scripts/clean_task_a_sft.py \\
        --input-dir data/output/sft/task_a \\
        --output-dir data/output/sft/task_a_cleaned \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def clean_record(record: dict) -> tuple[dict | None, str | None]:
    """Return (cleaned_record, drop_reason).

    drop_reason is None when the record is kept (possibly modified).
    """
    msgs = record.get("messages", [])

    # 1. Drop truncated rows (system-only conversations)
    if not any(m.get("role") != "system" for m in msgs):
        return None, "truncated_no_non_system_turns"

    # 2. Strip role-confused tool messages (content may be str or dict/list)
    def _is_role_confused(m: dict) -> bool:
        if m.get("role") != "tool":
            return False
        content = m.get("content")
        return isinstance(content, str) and content.strip().startswith("<tool_call>")

    cleaned_msgs = [m for m in msgs if not _is_role_confused(m)]
    record["messages"] = cleaned_msgs

    # Re-check: if stripping left only a system message, drop the row
    if not any(m.get("role") != "system" for m in cleaned_msgs):
        return None, "truncated_after_role_confusion_filter"

    # 3. Flag empty terminal_state
    gt = record.setdefault("ground_truth", {})
    ts = gt.get("terminal_state")
    gt["terminal_reached"] = bool(ts)

    return record, None


def _clean_file(
    src: Path,
    dst: Path,
    dry_run: bool,
) -> tuple[int, int, int, int, int]:
    """Clean a single JSONL file.

    Returns (total, dropped_truncated, dropped_role_confused_convs,
             stripped_rc_messages, flagged_terminal).
    """
    total = dropped_truncated = dropped_rc_convs = stripped_rc_msgs = flagged_terminal = 0
    out_lines: list[str] = []

    with open(src) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"Error: {src}:{lineno}: {exc}")

            total += 1
            before_len = len(record.get("messages", []))
            cleaned, reason = clean_record(record)

            if cleaned is None:
                if reason == "truncated_no_non_system_turns":
                    dropped_truncated += 1
                else:
                    dropped_rc_convs += 1
                continue

            after_len = len(cleaned.get("messages", []))
            stripped_rc_msgs += before_len - after_len

            if not cleaned.get("ground_truth", {}).get("terminal_reached", True):
                flagged_terminal += 1

            out_lines.append(json.dumps(cleaned, ensure_ascii=False))

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as fh:
            fh.write("\n".join(out_lines))
            if out_lines:
                fh.write("\n")

    return total, dropped_truncated, dropped_rc_convs, stripped_rc_msgs, flagged_terminal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Task A SFT JSONL files before fine-tuning."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Directory containing raw *.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Destination directory for cleaned *.jsonl files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without writing any files.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file progress output.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        sys.exit(f"Error: input directory not found: {input_dir}")

    src_files = sorted(input_dir.glob("*.jsonl"))
    if not src_files:
        sys.exit(f"Error: no *.jsonl files found in {input_dir}")

    if args.dry_run:
        print("[dry-run] No files will be written.")

    grand_total = grand_kept = grand_trunc = grand_rc = grand_rc_msgs = grand_flag = 0

    for src in src_files:
        dst = output_dir / src.name
        total, trunc, rc, rc_msgs, flag = _clean_file(src, dst, dry_run=args.dry_run)
        kept = total - trunc - rc
        grand_total += total
        grand_kept += kept
        grand_trunc += trunc
        grand_rc += rc
        grand_rc_msgs += rc_msgs
        grand_flag += flag

        if not args.quiet:
            parts = [f"{src.name}: {total} in → {kept} kept"]
            if trunc:
                parts.append(f"{trunc} truncated dropped")
            if rc:
                parts.append(f"{rc} role-confused-conv dropped")
            if rc_msgs:
                parts.append(f"{rc_msgs} role-confused msgs stripped")
            if flag:
                parts.append(f"{flag} terminal_reached=False flagged")
            print("  " + ", ".join(parts))

    print()
    print("=" * 60)
    print(f"Input files    : {len(src_files)}")
    print(f"Total records  : {grand_total}")
    print(f"Kept           : {grand_kept}")
    print(f"  Dropped (truncated)                : {grand_trunc}")
    print(f"  Dropped (role-conf conv, all-bad)  : {grand_rc}")
    print(f"  Stripped role-confused tool msgs   : {grand_rc_msgs}")
    print(f"  Flagged terminal_reached=False     : {grand_flag}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
    else:
        print(f"\nCleaned files written to: {output_dir}")


if __name__ == "__main__":
    main()
