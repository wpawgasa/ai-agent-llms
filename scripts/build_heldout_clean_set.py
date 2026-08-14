#!/usr/bin/env python3
"""Rebuild the contamination-free held-out set for cross-lineage audits.

The set behind docs/cat_a_corpus_v2_heldout_regression.md was built ad-hoc and
lost; §7's repro pointed at a `<clean-set-dir>` placeholder. Numbers scored on a
differently-built set are NOT comparable to the stored v3 (0.5709) / v4 (0.5120)
composites, so this pins the construction in code.

Rule: keep a v2 test conversation only if it appears in neither v1 train nor v1
validation, matching across corpora on a fingerprint of USER turns (remediation
never touched them; `conversation_id` is not unique).

Default inputs are the two tagged corpora:
    v1  task-a-sft-v1  data/output/sft/task_a_splits  6bb5eb6f7c48356ca05078c537ae68b1.dir
    v2  task-a-sft-v2  data/output/sft/task_a_splits  21e33e25d7bbae63e97a89029b3b4705.dir

v1 is not the working copy, so materialize it out-of-place first:

    .venv-train/bin/python scripts/materialize_dvc_lineage.py \\
        --dir-hash 6bb5eb6f7c48356ca05078c537ae68b1 --out /tmp/v1_splits

    .venv-train/bin/python scripts/build_heldout_clean_set.py \\
        --candidate-split data/output/sft/task_a_splits/test.jsonl \\
        --exclusion-split /tmp/v1_splits/train.jsonl \\
        --exclusion-split /tmp/v1_splits/validation.jsonl \\
        --out-dir data/output/heldout/cat_a_v2_test_not_in_v1 \\
        --expect-clean 206

Then audit any checkpoint against it:

    .venv-train/bin/python scripts/heldout_composite_audit.py \\
        --checkpoint <ckpt> --data-dir data/output/heldout/cat_a_v2_test_not_in_v1 \\
        --split test --n-prompts 206 --seed 42 --output runs/audit/<name>.json

Verify a rebuild reproduces the original set with --verify-against, which
compares per-row ground truth against a stored audit JSON:

    ... --verify-against runs/audit/heldout_ckpt1767_v2corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_workflow_agents.data.heldout_clean_set import build_clean_set  # noqa: E402


def _norm_seq(seq: object) -> list[tuple]:
    return [
        tuple(s.values()) if isinstance(s, dict) else tuple(s) for s in (seq or [])  # type: ignore[union-attr]
    ]


def _verify(out_dir: Path, split: str, n_prompts: int, seed: int, stored_path: Path) -> int:
    """Compare the rebuilt set's sampled ground truth against a stored audit.

    An exact match proves the rebuild selected and ordered the same
    conversations, so a new checkpoint's composite is comparable to the stored
    one. Anything else means the sets differ and the numbers must not be
    compared.
    """
    from preflight_entropy_diag import _decode_gt, _sample_prompts

    prompts = _sample_prompts(out_dir, split, n_prompts, seed)
    fresh = [_decode_gt(p["ground_truth"]) for p in prompts]

    stored = sorted(
        json.loads(stored_path.read_text())["rows"], key=lambda r: r["row_index"]
    )
    if len(fresh) != len(stored):
        print(
            f"[verify] FAIL — rebuilt {len(fresh)} rows vs stored {len(stored)}",
            file=sys.stderr,
        )
        return 1

    mismatches = []
    for i, (f, s) in enumerate(zip(fresh, stored, strict=True)):
        a = (
            _norm_seq(f.get("state_sequence")),
            json.dumps(f.get("tool_calls") or [], sort_keys=True),
            f.get("terminal_state") or "",
        )
        b = (
            _norm_seq(s.get("gt_state_sequence")),
            json.dumps(s.get("gt_tool_calls") or [], sort_keys=True),
            s.get("gt_terminal") or "",
        )
        if a != b:
            mismatches.append(i)

    if mismatches:
        print(
            f"[verify] FAIL — {len(mismatches)}/{len(fresh)} rows differ from "
            f"{stored_path} (first: {mismatches[:5]}). Do NOT compare composites "
            f"scored on this set against the stored ones.",
            file=sys.stderr,
        )
        return 1

    print(f"[verify] OK — {len(fresh)}/{len(fresh)} rows match {stored_path.name}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--candidate-split", type=Path, required=True)
    p.add_argument("--exclusion-split", type=Path, action="append", default=[])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split-name", default="test")
    p.add_argument("--expect-clean", type=int, default=None,
                   help="Fail if the clean count differs (206 for the v2-minus-v1 set).")
    p.add_argument("--verify-against", type=Path, default=None,
                   help="Stored audit JSON to check the rebuild against.")
    p.add_argument("--n-prompts", type=int, default=206)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stats = build_clean_set(
        candidate_split=args.candidate_split,
        exclusion_splits=args.exclusion_split,
        out_dir=args.out_dir,
        split_name=args.split_name,
    )
    print(json.dumps(stats, indent=2))

    if args.expect_clean is not None and stats["n_clean"] != args.expect_clean:
        print(
            f"[build] FAIL — expected {args.expect_clean} clean conversations, "
            f"got {stats['n_clean']}. Check the corpus revisions on both sides.",
            file=sys.stderr,
        )
        return 1

    if args.verify_against:
        return _verify(
            args.out_dir, args.split_name, args.n_prompts, args.seed, args.verify_against
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
