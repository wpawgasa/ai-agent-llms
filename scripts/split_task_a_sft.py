#!/usr/bin/env python3
"""Deterministically split the cleaned Task A SFT corpus into train/val/test.

Reads every ``*.jsonl`` file in the input directory, shuffles the concatenated
rows with a fixed seed, and writes three JSONL files (``train.jsonl``,
``validation.jsonl``, ``test.jsonl``) into the output directory at the
configured ratio (default 85 / 10 / 5).

The split is reused across:
  - SFT training        (reads ``train.jsonl`` + ``validation.jsonl``)
  - GRPO training       (reads ``train.jsonl`` — filtered to L3-L5 by
                         ``scripts/filter_grpo_data.py``)
  - Held-out evaluation (``test.jsonl`` is reserved for final evaluation)

Default seed is **42**. Output is idempotent for a given (input set, seed,
ratios) — re-running on the same inputs produces byte-identical files.

Usage:
    python scripts/split_task_a_sft.py \\
        --input-dir data/output/sft/task_a_cleaned \\
        --output-dir data/output/sft/task_a_splits

    # Different ratios:
    python scripts/split_task_a_sft.py --train 0.9 --validation 0.05 --test 0.05

    # Inspect counts without writing files:
    python scripts/split_task_a_sft.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DEFAULT_INPUT = Path("data/output/sft/task_a_cleaned")
DEFAULT_OUTPUT = Path("data/output/sft/task_a_splits")
DEFAULT_SEED = 42
DEFAULT_RATIOS = {"train": 0.85, "validation": 0.10, "test": 0.05}


def _load_rows(input_dir: Path) -> list[dict]:
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"Error: no *.jsonl files found in {input_dir}")
    rows: list[dict] = []
    for f in files:
        with open(f) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    sys.exit(f"Error: {f}:{lineno}: {exc}")
    return rows


def _write_split(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically split cleaned Task A SFT corpus into train/val/test.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="DIR",
        help=f"Directory of cleaned *.jsonl files (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="DIR",
        help=f"Destination for {{train,validation,test}}.jsonl (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Shuffle seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument("--train", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--validation", type=float, default=DEFAULT_RATIOS["validation"])
    parser.add_argument("--test", type=float, default=DEFAULT_RATIOS["test"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing splits in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without writing any files.",
    )
    args = parser.parse_args()

    total_ratio = args.train + args.validation + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        sys.exit(
            f"Error: ratios must sum to 1.0, got {total_ratio:.4f} "
            f"(train={args.train}, validation={args.validation}, test={args.test})"
        )

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        sys.exit(f"Error: input directory not found: {input_dir}")

    existing = {
        s: (output_dir / f"{s}.jsonl") for s in ("train", "validation", "test")
    }
    already_present = [s for s, p in existing.items() if p.is_file()]
    if already_present and not args.force and not args.dry_run:
        counts = {s: sum(1 for _ in open(existing[s])) for s in already_present}
        print(f"Splits already exist in {output_dir}: {counts}")
        print("Pass --force to overwrite.")
        return

    rows = _load_rows(input_dir)
    n = len(rows)
    random.Random(args.seed).shuffle(rows)

    n_train = int(n * args.train)
    n_val = int(n * args.validation)
    # Test absorbs the rounding remainder so the three splits sum to n.
    n_test = n - n_train - n_val

    chunks = {
        "train": rows[:n_train],
        "validation": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }

    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Seed        : {args.seed}")
    print(f"Ratios      : train={args.train}  validation={args.validation}  test={args.test}")
    print("-" * 60)
    print(f"  total     : {n}")
    for name, chunk in chunks.items():
        print(f"  {name:11s}: {len(chunk)}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    for name, chunk in chunks.items():
        _write_split(output_dir / f"{name}.jsonl", chunk)

    print(f"\nSplits written to: {output_dir}")


if __name__ == "__main__":
    main()
