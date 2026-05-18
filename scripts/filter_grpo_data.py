#!/usr/bin/env python3
"""Filter the cleaned SFT corpus to produce the GRPO prompt set.

GRPO reuses the SFT corpus (see docs/data_generation_recipes.md): rewards are
recomputed online from policy generations, so GRPO needs prompts — not new
ground-truth labels. The standard refinement is to filter the SFT splits to
the harder complexity levels (L3-L5), where reward shaping has the most
headroom over a saturated SFT checkpoint.

This script reads the deterministic 85/10/5 splits produced by
``scripts/split_task_a_sft.py`` (DVC stage ``task_a_sft_splits``, output
``data/output/sft/task_a_splits/``), filters them to the requested levels,
and writes the result under ``data/output/grpo/task_a/``. The ``test`` split
is intentionally **not** filtered into the GRPO output — it stays reserved
for final evaluation.

Usage:
    python scripts/filter_grpo_data.py \\
        --input-dir data/output/sft/task_a_splits \\
        --output-dir data/output/grpo/task_a

    # Different level set (e.g. just the hardest):
    python scripts/filter_grpo_data.py --levels L4 L5

    # Inspect counts without writing output:
    python scripts/filter_grpo_data.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("data/output/sft/task_a_splits")
DEFAULT_OUTPUT = Path("data/output/grpo/task_a")
DEFAULT_LEVELS = ("L3", "L4", "L5")
# test.jsonl is reserved for final evaluation — never filtered into GRPO.
SPLITS_TO_FILTER = ("train", "validation")


def _filter_file(
    src: Path,
    dst: Path,
    levels: set[str],
    dry_run: bool,
) -> tuple[int, int, Counter]:
    """Filter a single JSONL file to records whose complexity_level is in ``levels``.

    Returns (total_in, kept, kept_by_level).
    """
    total = 0
    kept_by_level: Counter = Counter()
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
            level = record.get("complexity_level")
            if level in levels:
                kept_by_level[level] += 1
                out_lines.append(json.dumps(record, ensure_ascii=False))

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as fh:
            fh.write("\n".join(out_lines))
            if out_lines:
                fh.write("\n")

    return total, sum(kept_by_level.values()), kept_by_level


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter cleaned SFT splits to the GRPO prompt set.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="DIR",
        help=f"Directory containing the SFT splits (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="DIR",
        help=f"Destination directory for filtered splits (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(DEFAULT_LEVELS),
        metavar="LEVEL",
        help="Complexity levels to keep (default: L3 L4 L5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without writing any files.",
    )
    args = parser.parse_args()

    levels = set(args.levels)
    invalid = levels - {"L1", "L2", "L3", "L4", "L5"}
    if invalid:
        sys.exit(f"Error: unknown complexity level(s): {sorted(invalid)}")

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        sys.exit(
            f"Error: input directory not found: {input_dir}\n"
            "  Splits are produced by scripts/split_task_a_sft.py "
            "(DVC stage: task_a_sft_splits)."
        )

    missing = [s for s in SPLITS_TO_FILTER if not (input_dir / f"{s}.jsonl").is_file()]
    if missing:
        sys.exit(
            f"Error: missing split file(s) in {input_dir}: "
            + ", ".join(f"{s}.jsonl" for s in missing)
        )

    if args.dry_run:
        print("[dry-run] No files will be written.")

    print(f"Levels kept: {sorted(levels)}")
    print(f"Input dir  : {input_dir}")
    print(f"Output dir : {output_dir}")
    print("-" * 60)

    grand_total = 0
    grand_kept = 0
    grand_by_level: Counter = Counter()

    for split in SPLITS_TO_FILTER:
        src = input_dir / f"{split}.jsonl"
        dst = output_dir / f"{split}.jsonl"
        total, kept, by_level = _filter_file(src, dst, levels, args.dry_run)
        grand_total += total
        grand_kept += kept
        grand_by_level.update(by_level)

        per_level = ", ".join(
            f"{lvl}={by_level[lvl]}" for lvl in sorted(levels) if by_level[lvl]
        ) or "<empty>"
        print(f"  {split:11s}: {total:>5d} in → {kept:>5d} kept   ({per_level})")

    print("-" * 60)
    print(f"Total in   : {grand_total}")
    print(f"Total kept : {grand_kept}")
    for lvl in sorted(levels):
        if grand_by_level[lvl]:
            print(f"  {lvl}: {grand_by_level[lvl]}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
    else:
        print(f"\nFiltered splits written to: {output_dir}")


if __name__ == "__main__":
    main()
