#!/usr/bin/env python3
"""Decompose the mining-yield gap into a split effect and a classifier effect.

`scripts/mine_model_negatives.py`'s first run found C2 wrong on 51 of 399
TRAIN prompts (12.8%). `docs/cat_a_c2_heldout_result.md`'s held-out audit
found C2 wrong on 27 of 71 tool-bearing TEST-derived rows (38.0%) — a
contamination-free set the checkpoint never trained on. CLAUDE.md's `dc84adb`
commit names two untested explanations for the gap:

  1. SPLIT EFFECT   — TRAIN rows were seen during SFT; TEST rows were not.
  2. CLASSIFIER EFFECT — mine_model_negatives.py's `_classify()` may be more
     lenient than the held-out audit's strict composite scorer.

This script measures each while holding the other fixed:

  PROBE 1 (classifier effect, no GPU) --from-audit-json PATH
    Reclassifies a stored scripts/heldout_composite_audit.py output's
    completions with `_classify()` — the SAME rows that produced 38.0%, so
    only the scoring function varies.

  PROBE 2 (split effect, needs GPU) --checkpoint PATH --split validation
    Runs mine_model_negatives.py's own `_select_prompts`/`_classify` code
    path, unchanged except for --split, directly comparable to the known
    12.8% TRAIN figure — only the split varies.

Never point --split at "test" here — see mine_model_negatives.py's identical
guard and CLAUDE.md R18. Probe 1 does not generate anything; it only
reclassifies completions a prior audit already produced, and this script
never writes a chosen/rejected-shaped pairs file, so its output can never be
mistaken for DPO training data.

Usage:
    # Cheap, no GPU:
    python scripts/investigate_mining_yield.py \\
        --from-audit-json runs/audit/heldout_c2_ckpt1767_v2corpus.json \\
        --output runs/audit/mining_yield_investigation.json

    # Needs .venv-train + GPU:
    .venv-train/bin/python scripts/investigate_mining_yield.py \\
        --from-audit-json runs/audit/heldout_c2_ckpt1767_v2corpus.json \\
        --checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767 \\
        --split validation --output runs/audit/mining_yield_investigation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mine_model_negatives import _classify, _select_prompts  # noqa: E402

#: Known figures this script's new measurements get compared against.
_KNOWN_TRAIN_WRONG_RATE = 0.128  # dc84adb: 51/399
_KNOWN_HELDOUT_COMPOSITE_WRONG_RATE = 0.380  # docs/cat_a_c2_heldout_result.md: 27/71


def _gt_from_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the {tool_calls, state_sequence} shape _classify() expects
    from one row of a scripts/heldout_composite_audit.py output JSON.

    The audit stores GT as separate `gt_tool_calls` / `gt_state_sequence`
    fields (see that script's `_components()`), not a single `ground_truth`
    dict — this bridges the two shapes.
    """
    return {
        "tool_calls": row.get("gt_tool_calls") or [],
        "state_sequence": row.get("gt_state_sequence") or [],
    }


def classify_rate_from_audit_json(
    path: Path, tool_bearing_only: bool = True
) -> dict[str, Any]:
    """Probe 1: reclassify a stored audit's completions with _classify().

    Holds the split fixed (these are the exact rows that produced the
    composite "wrong" rate) and varies only the scoring function, isolating
    the classifier effect.
    """
    payload = json.loads(Path(path).read_text())
    rows = payload["rows"]
    if tool_bearing_only:
        rows = [r for r in rows if r.get("n_gt_tools", 0) > 0]

    wrong = 0
    by_kind: dict[str, int] = {}
    for row in rows:
        gt = _gt_from_audit_row(row)
        kind = _classify(row.get("completion", ""), gt)
        if kind is not None:
            wrong += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1

    n = len(rows)
    return {
        "probe": "classifier_effect",
        "source": str(path),
        "tool_bearing_only": tool_bearing_only,
        "n_rows": n,
        "n_wrong": wrong,
        "wrong_rate": wrong / n if n else 0.0,
        "by_kind": by_kind,
    }


def classify_rate_from_split(
    checkpoint: str,
    data_dir: Path,
    split: str,
    n_prompts: int,
    tool_share: float,
    seed: int,
    max_new_tokens: int,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Probe 2: mine_model_negatives.py's own selection + classify code path
    against `split`, unchanged except for which split it reads.

    Directly comparable to the known 12.8% TRAIN figure, which used the
    identical function with split="train" — only the split varies, isolating
    the split effect.
    """
    if split == "test":
        raise SystemExit(
            "refusing to generate against the test split — it feeds the "
            "held-out audit set, the only contamination-free measuring "
            "stick for this lineage"
        )

    from preflight_entropy_diag import _generate_for_checkpoint

    rows = _select_prompts(data_dir, split, n_prompts, tool_share, seed)
    gens = _generate_for_checkpoint(
        checkpoint=checkpoint,
        prompts=rows,
        n_completions=1,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        seed=seed,
        do_sample=False,
    )

    wrong = 0
    by_kind: dict[str, int] = {}
    n_scored = 0
    for row, comp_list in zip(rows, gens):
        completion = (comp_list[0] if comp_list else "") or ""
        gt = json.loads(row["ground_truth"])
        gold = (gt.get("messages") or [{}])[0].get("content") or ""
        if not completion.strip() or not gold.strip():
            continue
        if completion.strip() == gold.strip():
            continue
        n_scored += 1
        kind = _classify(completion, gt)
        if kind is not None:
            wrong += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1

    return {
        "probe": "split_effect",
        "split": split,
        "n_prompts_sampled": len(rows),
        "n_scored": n_scored,
        "n_wrong": wrong,
        "wrong_rate": wrong / n_scored if n_scored else 0.0,
        "by_kind": by_kind,
    }


def print_decomposition_table(
    train_rate: float,
    validation_probe: dict[str, Any] | None,
    heldout_classify_probe: dict[str, Any] | None,
    heldout_composite_rate: float,
) -> None:
    """Print the 4-row comparison table. Pure formatting: no I/O, no model."""
    rows = [
        ("train      (classify, known)", f"{train_rate:.1%}"),
        (
            "validation (classify, new)",
            f"{validation_probe['wrong_rate']:.1%}" if validation_probe else "not run",
        ),
        (
            "held-out   (classify, new)",
            f"{heldout_classify_probe['wrong_rate']:.1%}" if heldout_classify_probe else "not run",
        ),
        ("held-out   (composite, known)", f"{heldout_composite_rate:.1%}"),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-audit-json", type=Path, default=None,
        help="Probe 1: reclassify a stored heldout_composite_audit.py output.",
    )
    ap.add_argument(
        "--include-non-tool-rows", action="store_true",
        help="Probe 1: score all rows, not just the tool-bearing subset.",
    )
    ap.add_argument(
        "--checkpoint", default=None,
        help="Probe 2: run a fresh generation pass against --split.",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-prompts", type=int, default=400)
    ap.add_argument("--tool-share", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-wrong-rate", type=float, default=_KNOWN_TRAIN_WRONG_RATE)
    ap.add_argument(
        "--heldout-composite-wrong-rate", type=float,
        default=_KNOWN_HELDOUT_COMPOSITE_WRONG_RATE,
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.from_audit_json and not args.checkpoint:
        print(
            "Nothing to do: pass --from-audit-json and/or --checkpoint.",
            file=sys.stderr,
        )
        return 1

    result: dict[str, Any] = {
        "train_wrong_rate": args.train_wrong_rate,
        "heldout_composite_wrong_rate": args.heldout_composite_wrong_rate,
    }

    heldout_classify_probe = None
    if args.from_audit_json:
        heldout_classify_probe = classify_rate_from_audit_json(
            args.from_audit_json, tool_bearing_only=not args.include_non_tool_rows
        )
        result["classifier_effect_probe"] = heldout_classify_probe
        print(
            f"[probe1] classifier effect: {heldout_classify_probe['wrong_rate']:.1%} "
            f"on {heldout_classify_probe['n_rows']} rows "
            f"(composite said {args.heldout_composite_wrong_rate:.1%})"
        )

    validation_probe = None
    if args.checkpoint:
        t0 = time.time()
        validation_probe = classify_rate_from_split(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            split=args.split,
            n_prompts=args.n_prompts,
            tool_share=args.tool_share,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
        )
        result["split_effect_probe"] = validation_probe
        print(
            f"[probe2] split effect on --split {args.split}: "
            f"{validation_probe['wrong_rate']:.1%} "
            f"({time.time() - t0:.0f}s, {validation_probe['n_scored']} rows scored)"
        )

    print()
    print_decomposition_table(
        args.train_wrong_rate,
        validation_probe,
        heldout_classify_probe,
        args.heldout_composite_wrong_rate,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
