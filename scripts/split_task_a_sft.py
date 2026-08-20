#!/usr/bin/env python3
"""Deterministically split the cleaned Task A SFT corpus into train/val/test.

Reads every ``*.jsonl`` file in the input directory, shuffles with a fixed
seed, and writes three JSONL files (``train.jsonl``, ``validation.jsonl``,
``test.jsonl``) into the output directory at the configured ratio (default
85 / 10 / 5).

**The shuffle runs per modality, not over the concatenated pool.** That is the
whole reason this file is not three lines long. A positional shuffle over the
concatenation redraws *every* assignment the moment the pool size changes, so
merging the 2,400-conversation voice batch into the 5,543-row text corpus
moved 7 of the 278 pinned held-out candidates into train — measured, not
feared. Those 278 rows are the only link back to cell C2's held-out composite
of 0.7595 (risk R17), and the contamination would have been silent.

Grouping by modality first makes an additive batch additive. The text rows
arrive from the same files in the same sorted order whether or not voice files
sit beside them, so the text group is the same list it was before, the seed is
the same, the group size is the same, and the shuffle and the cut points are
therefore identical. No existing row moves. The ratios also hold *within* each
modality, so the test split carries ~5% of the text rows and ~5% of the voice
rows — which spec section 5 needs, since it requires a held-out score per
modality and a text-only test split would leave the voice score undefined.

The guarantee has an honest limit: adding rows to an **existing** modality
group still reshuffles that group. ``--assert-unmoved DIR`` checks the
guarantee rather than assuming it — it matches conversations against a prior
split directory by ``user_turn_fingerprint`` and fails when any row that was
already assigned lands somewhere else. Run it after every merge.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_INPUT = Path("data/output/sft/task_a_cleaned")
DEFAULT_OUTPUT = Path("data/output/sft/task_a_splits")
DEFAULT_SEED = 42
DEFAULT_RATIOS = {"train": 0.85, "validation": 0.10, "test": 0.05}
SPLIT_NAMES = ("train", "validation", "test")


def _modality_of(row: dict) -> str:
    """A row with no ``modality`` field is text.

    Every one of the conversations that predate the field is a written one, so
    the default is what keeps the existing corpus in a single group.
    """
    return row.get("modality") or "text"


def split_rows(
    rows: list[dict], ratios: dict[str, float], seed: int
) -> dict[str, list[dict]]:
    """Assign every row to a split, one modality group at a time.

    Groups are processed in sorted key order and each group gets its own
    ``random.Random(seed)``, so a group's assignment depends only on that
    group's own contents. Adding a group cannot disturb another one.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_modality_of(row), []).append(row)

    chunks: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for _, group in sorted(groups.items()):
        shuffled = list(group)
        random.Random(seed).shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["validation"])
        # Test absorbs the rounding remainder so the three splits sum to n.
        chunks["train"].extend(shuffled[:n_train])
        chunks["validation"].extend(shuffled[n_train : n_train + n_val])
        chunks["test"].extend(shuffled[n_train + n_val :])
    return chunks


def assert_unmoved(chunks: dict[str, list[dict]], prior_dir: Path) -> list[str]:
    """Return one line per conversation that changed split since ``prior_dir``.

    Matched by ``user_turn_fingerprint``, not by ``conversation_id`` — ids are
    not unique in this corpus (risk R15), so an id-keyed comparison would
    report collisions as movement and miss real movement.

    A conversation absent from the prior splits is new and is not movement.
    """
    from llm_workflow_agents.data.heldout_clean_set import user_turn_fingerprint

    prior: dict[str, str] = {}
    for name in SPLIT_NAMES:
        path = prior_dir / f"{name}.jsonl"
        if not path.is_file():
            sys.exit(f"Error: --assert-unmoved directory has no {name}.jsonl: {prior_dir}")
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    prior[user_turn_fingerprint(json.loads(line))] = name

    moved: list[str] = []
    for name, rows in chunks.items():
        for row in rows:
            was = prior.get(user_turn_fingerprint(row))
            if was is not None and was != name:
                moved.append(
                    f"{row.get('conversation_id')} ({_modality_of(row)}): {was} -> {name}"
                )
    return moved


def _load_rows(input_dirs: list[Path]) -> list[dict]:
    files: list[Path] = []
    for input_dir in sorted(input_dirs):
        files.extend(sorted(input_dir.glob("*.jsonl")))
    if not files:
        dirs_str = ", ".join(str(d) for d in input_dirs)
        sys.exit(f"Error: no *.jsonl files found in {dirs_str}")
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
        action="append",
        default=None,
        help=(
            f"Directory of cleaned *.jsonl files (default: {DEFAULT_INPUT}). Repeat "
            "the flag to read more than one directory, for example the text corpus "
            "and the voice corpus."
        ),
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
    parser.add_argument(
        "--assert-unmoved",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory of prior {train,validation,test}.jsonl. Fail when any "
            "conversation already assigned there lands in a different split. "
            "Run this after merging an additive batch: a silently reassigned "
            "held-out row contaminates the only measurement this project has."
        ),
    )
    args = parser.parse_args()

    total_ratio = args.train + args.validation + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        sys.exit(
            f"Error: ratios must sum to 1.0, got {total_ratio:.4f} "
            f"(train={args.train}, validation={args.validation}, test={args.test})"
        )

    input_dirs: list[Path] = args.input_dir or [DEFAULT_INPUT]
    output_dir: Path = args.output_dir

    for input_dir in input_dirs:
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

    rows = _load_rows(input_dirs)
    n = len(rows)
    ratios = {"train": args.train, "validation": args.validation, "test": args.test}
    chunks = split_rows(rows, ratios, args.seed)

    modality_counts: dict[str, int] = {}
    for row in rows:
        modality = _modality_of(row)
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    print(f"Input dirs  : {sorted(input_dirs)}")
    print(f"Output dir  : {output_dir}")
    print(f"Seed        : {args.seed}")
    print(f"Ratios      : train={args.train}  validation={args.validation}  test={args.test}")
    print(f"Modalities  : {dict(sorted(modality_counts.items()))}  (shuffled per group)")
    print("-" * 60)
    print(f"  total     : {n}")
    for name in SPLIT_NAMES:
        chunk = chunks[name]
        per_modality = {}
        for row in chunk:
            modality = _modality_of(row)
            per_modality[modality] = per_modality.get(modality, 0) + 1
        print(f"  {name:11s}: {len(chunk):6d}  {dict(sorted(per_modality.items()))}")

    if args.assert_unmoved is not None:
        moved = assert_unmoved(chunks, args.assert_unmoved)
        if moved:
            print(
                f"\nError: {len(moved)} conversation(s) changed split relative to "
                f"{args.assert_unmoved}. An additive batch must not reassign a row "
                f"that was already assigned. First few:",
                file=sys.stderr,
            )
            for line in moved[:10]:
                print(f"  - {line}", file=sys.stderr)
            sys.exit(1)
        print(f"\n[assert-unmoved] OK — no row moved relative to {args.assert_unmoved}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    for name in SPLIT_NAMES:
        _write_split(output_dir / f"{name}.jsonl", chunks[name])

    print(f"\nSplits written to: {output_dir}")


if __name__ == "__main__":
    main()
