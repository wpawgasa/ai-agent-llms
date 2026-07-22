#!/usr/bin/env python3
"""SFT-only ceiling check — greedy held-out composite vs the Phase 2 target.

Answers the fork documented in docs/grpo_viability_investigation.md §5.2: if
both the RFT headroom probe and the trajectory variance probe land short of
GO, does the SFT checkpoint's greedy held-out composite already clear the
Phase 2 target on its own? If it does, SFT-only is the correct terminal
state (RL has nothing left to add) rather than a compromise. If it doesn't,
the gap without measurable RL headroom points at a reward/ground-truth issue
to audit (e.g. the placeholder-arg tool-call stubs) before concluding the
policy is capped.

Unlike the RFT/trajectory probes this does one greedy pass only (no
sampling, no headroom) and scores with the *strict* composite
(``_heldout_composite_score`` — the same metric ``_HeldOutEvalCallback``
logs as ``eval/held_out_composite`` during live GRPO training), not the
graded training reward. Target: >= 0.80, re-derived for the per-turn-fair
metric (see the 2026-07-22 Cat A factorial spec §9); this supersedes the
0.75 whole-conversation target in eval/composite_score.py.

Gate:
  PASS : mean_composite >= 0.80  -> ship SFT-only.
  FAIL : mean_composite <  0.80  -> audit reward/GT before assuming a ceiling.

The reduction (summarize_heldout_check) and the gate (classify_gate) are
pure and unit-tested in tests/unit/test_heldout_composite_check.py.

Usage:
    .venv-train/bin/python scripts/heldout_composite_check.py \
        --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000 \
        --data-dir data/output/grpo/task_a --split validation \
        --n-prompts 150 --output runs/preflight/heldout_composite_ckpt1000.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Gate threshold. Re-derived 2026-07-22 for the per-turn-fair metric: 0.80,
# not the 0.75 of eval/composite_score.py's original whole-conversation target
# (docs/superpowers/plans/2026-07-22-cat-a-state-accuracy-factorial.md §9).
TARGET_COMPOSITE = 0.80


def summarize_heldout_check(row_scores: list[float]) -> dict[str, Any]:
    """Reduce per-row composite scores to the probe summary.

    Pure — no model, no I/O. ``row_scores[i]`` is the strict held-out
    composite for held-out row i (one greedy completion each).
    """
    per_row: list[dict[str, Any]] = [
        {"row_index": i, "composite": s} for i, s in enumerate(row_scores)
    ]

    n = len(row_scores)
    return {
        "n_rows": n,
        "mean_composite": statistics.fmean(row_scores) if row_scores else 0.0,
        "median_composite": statistics.median(row_scores) if row_scores else 0.0,
        "min_composite": min(row_scores) if row_scores else 0.0,
        "max_composite": max(row_scores) if row_scores else 0.0,
        "frac_below_target": (
            sum(1 for s in row_scores if s < TARGET_COMPOSITE) / n if n else 0.0
        ),
        "per_row": per_row,
    }


def classify_gate(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify the probe summary into a gate verdict. Pure."""
    mean_composite = summary["mean_composite"]
    passed = mean_composite >= TARGET_COMPOSITE

    verdict = "PASS" if passed else "FAIL"
    return verdict, {
        "checks": {
            f"mean_composite >= {TARGET_COMPOSITE}": passed,
        },
    }


_VERDICT_ACTION = {
    "PASS": (
        "SFT-only already clears the Phase 2 target — RL has nothing to add; "
        "ship SFT-only for Cat A."
    ),
    "FAIL": (
        "Below target despite no measurable RL headroom — audit reward/GT "
        "(e.g. placeholder-arg tool-call stubs) on ~50 rows before assuming "
        "a hard ceiling."
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SFT checkpoint to probe.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-prompts", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from preflight_entropy_diag import _decode_gt, _generate_for_checkpoint, _sample_prompts

    from llm_workflow_agents.training.grpo import _heldout_composite_score

    prompts = _sample_prompts(args.data_dir, args.split, args.n_prompts, args.seed)
    gts = [_decode_gt(p["ground_truth"]) for p in prompts]
    print(
        f"[data] sampled {len(prompts)} prompts from {args.data_dir}/{args.split}.jsonl",
        flush=True,
    )

    t0 = time.time()
    print("[gen] greedy pass (do_sample=False)", flush=True)
    greedy = _generate_for_checkpoint(
        checkpoint=args.checkpoint,
        prompts=prompts,
        n_completions=1,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    completions = [c[0] if c else "" for c in greedy]

    row_scores = [
        _heldout_composite_score([comp], [gt])
        for comp, gt in zip(completions, gts, strict=True)
    ]

    summary = summarize_heldout_check(row_scores)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    verdict, detail = classify_gate(summary)

    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "data": f"{args.data_dir}/{args.split}.jsonl",
            "n_prompts": summary["n_rows"],
            "seed": args.seed,
            "target_composite": TARGET_COMPOSITE,
        },
        "summary": {k: v for k, v in summary.items() if k != "per_row"},
        "gate": {"verdict": verdict, **detail, "action": _VERDICT_ACTION[verdict]},
        "per_row": summary["per_row"],
    }
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(
        f"\n=== held-out composite check — {Path(args.checkpoint).name} ===\n"
        f"  mean_composite   = {summary['mean_composite']:.4f}  (PASS >= {TARGET_COMPOSITE})\n"
        f"  median_composite = {summary['median_composite']:.4f}\n"
        f"  min/max          = {summary['min_composite']:.4f} / {summary['max_composite']:.4f}\n"
        f"  frac_below_target= {summary['frac_below_target']:.3f}\n"
        f"  wall_time_s      = {summary['wall_time_s']}\n"
        f"\n  VERDICT: {verdict} — {_VERDICT_ACTION[verdict]}\n"
        f"[done] wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
