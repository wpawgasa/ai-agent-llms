#!/usr/bin/env python3
"""Partition cleaned Task C pairs into train/validation/test.

Splits are pre-assigned by the generator (each row carries a `split` field); this
script NEVER re-derives them. It partitions by the recorded field and fails hard
(nonzero exit, no output written) on any integrity violation:
  - a row missing its `split` field,
  - a graph_id whose rows span more than one split,
  - a held-out-domain row outside test,
  - a held-out-register row in train.

`--seed` is accepted for interface parity with split_task_a_sft.py and recorded in
the summary only; it does not influence partitioning.

Usage:
    python scripts/split_task_c_pairs.py \\
        --input-dir data/output/sft/task_c_cleaned \\
        --output-dir data/output/sft/task_c_splits \\
        --group-key graph_id \\
        --heldout-domains utilities,surveys \\
        --heldout-registers manager_transcript \\
        --seed 142
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SPLITS = ("train", "validation", "test")


def partition_rows(rows: list[dict], group_key: str = "graph_id") -> dict[str, list[dict]]:
    """Bucket rows by their recorded `split` field."""
    buckets: dict[str, list[dict]] = {s: [] for s in _SPLITS}
    for row in rows:
        split = row.get("split")
        if split in buckets:
            buckets[split].append(row)
    return buckets


def validate_partition(
    rows: list[dict],
    group_key: str,
    heldout_domains: list[str],
    heldout_registers: list[str],
) -> list[str]:
    """Return a list of integrity-violation strings (empty = valid)."""
    violations: list[str] = []
    heldout_domain_set = set(heldout_domains)
    heldout_register_set = set(heldout_registers)

    group_to_split: dict[str, str] = {}
    for row in rows:
        pair_id = row.get("pair_id", "<unknown>")
        split = row.get("split")
        if split not in _SPLITS:
            violations.append(f"missing split field: {pair_id}")
            continue

        group = row.get(group_key)
        if group is not None:
            prior = group_to_split.get(group)
            if prior is None:
                group_to_split[group] = split
            elif prior != split:
                violations.append(f"group spans splits: {group}")

        if row.get("domain") in heldout_domain_set and split != "test":
            violations.append(f"held-out domain outside test: {pair_id}")
        if row.get("register") in heldout_register_set and split == "train":
            violations.append(f"held-out register in train: {pair_id}")

    return violations


def _load_rows(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for src in sorted(input_dir.glob("*.jsonl")):
        for lineno, line in enumerate(src.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"Error: {src}:{lineno}: {exc}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition Task C pairs by recorded split field.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-key", default="graph_id")
    parser.add_argument("--heldout-domains", default="utilities,surveys")
    parser.add_argument("--heldout-registers", default="manager_transcript")
    parser.add_argument("--seed", type=int, default=142, help="Recorded in summary only; not used for splitting.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing.")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Error: input directory not found: {args.input_dir}")
    rows = _load_rows(args.input_dir)
    if not rows:
        sys.exit(f"Error: no rows found in {args.input_dir}")

    heldout_domains = [d for d in args.heldout_domains.split(",") if d]
    heldout_registers = [r for r in args.heldout_registers.split(",") if r]
    violations = validate_partition(rows, args.group_key, heldout_domains, heldout_registers)
    if violations:
        print(f"Refusing to write: {len(violations)} integrity violation(s):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        sys.exit(1)

    buckets = partition_rows(rows, args.group_key)
    print("=" * 60)
    print(f"Seed (recorded): {args.seed}")
    for split in _SPLITS:
        print(f"{split:11s}: {len(buckets[split])}")

    if args.dry_run:
        return

    existing = [args.output_dir / f"{s}.jsonl" for s in _SPLITS]
    if any(p.exists() for p in existing) and not args.force:
        print("Split files already exist; use --force to overwrite.", file=sys.stderr)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in _SPLITS:
        rows_sorted = sorted(buckets[split], key=lambda r: r.get("pair_id", ""))
        dst = args.output_dir / f"{split}.jsonl"
        with open(dst, "w", encoding="utf-8") as fh:
            for row in rows_sorted:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
