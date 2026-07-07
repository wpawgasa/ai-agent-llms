#!/usr/bin/env python3
"""RFT Stage-0 headroom probe (docs/grpo_viability_investigation.md §4, Stage 0).

The go/no-go gate that decides whether rejection-sampling fine-tuning (RFT) can
work on Cat A BEFORE spending a training run. For each prompt it generates one
greedy completion and N sampled completions, scores all of them with the
existing graded reward (``reward_business_logic``), and measures how much
headroom best-of-N has over greedy:

    headroom(prompt) = max(sampled_rewards) - greedy_reward

RFT can only distill what best-of-N beats greedy on, so ``frontier_frac`` (the
fraction of prompts with headroom above a threshold) is the direct predictor of
whether RFT has anything to learn. The same run also re-measures GRPO's
admissibility condition (``frac_collapsed_groups`` / median ``reward_std``) at
matched sampling settings, so one probe answers both branches.

Gates (docs/grpo_viability_investigation.md §4):
  GO_RFT       : frontier_frac >= 0.15 AND mean_headroom >= 0.03
  GRPO_REVIVAL : frac_collapsed_groups < 0.50 AND median reward_std >= 0.05
  NO_GO        : frontier_frac < 0.10  (kill the single-turn RL/RFT track)
  MARGINAL     : none of the above (inspect the histogram; likely re-probe)

Reuses the GPU-tested generation + scoring path from preflight_entropy_diag.py.
The reduction (summarize_headroom) and the gate (classify_gate) are pure and
unit-tested in tests/unit/test_rft_headroom.py.

Usage:
    .venv-train/bin/python scripts/rft_headroom_probe.py \
        --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000 \
        --data-dir data/output/grpo/task_a --split train \
        --n-prompts 500 --n-completions 8 --temperature 0.8 \
        --output runs/preflight/rft_headroom_ckpt1000.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Gate thresholds (docs/grpo_viability_investigation.md §4)
FRONTIER_MIN = 0.15
HEADROOM_MIN = 0.03
FRONTIER_NOGO = 0.10
COLLAPSED_MAX = 0.50
REWARD_STD_MIN = 0.05
HEADROOM_THRESHOLD = 0.05  # a prompt is "on the frontier" if headroom exceeds this


def summarize_headroom(
    sampled_rewards: list[list[float]],
    greedy_rewards: list[float],
    headroom_threshold: float = HEADROOM_THRESHOLD,
) -> dict[str, Any]:
    """Reduce per-prompt (sampled rewards, greedy reward) to the probe summary.

    Pure — no model, no I/O. ``sampled_rewards[i]`` is the list of N rewards for
    prompt i; ``greedy_rewards[i]`` is its single greedy reward.
    """
    per_prompt: list[dict[str, Any]] = []
    for i, (samps, greedy) in enumerate(zip(sampled_rewards, greedy_rewards)):
        best = max(samps) if samps else 0.0
        headroom = best - greedy
        std = statistics.pstdev(samps) if len(samps) > 1 else 0.0
        n_distinct = len({round(r, 3) for r in samps})
        per_prompt.append({
            "prompt_index": i,
            "greedy_reward": greedy,
            "best_sampled_reward": best,
            "headroom": headroom,
            "reward_std": std,
            "n_distinct_rungs": n_distinct,
            "on_frontier": headroom > headroom_threshold,
        })

    n = len(per_prompt)
    headrooms = [p["headroom"] for p in per_prompt]
    stds = [p["reward_std"] for p in per_prompt]
    return {
        "n_prompts": n,
        "frontier_frac": (sum(p["on_frontier"] for p in per_prompt) / n) if n else 0.0,
        "mean_headroom": statistics.fmean(headrooms) if headrooms else 0.0,
        "median_headroom": statistics.median(headrooms) if headrooms else 0.0,
        "frac_positive_headroom": (
            sum(1 for h in headrooms if h > 0) / n if n else 0.0
        ),
        # GRPO-revival branch metrics (matched-settings re-measurement)
        "frac_collapsed_groups": (
            sum(1 for s in stds if s < 0.01) / n if n else 0.0
        ),
        "median_reward_std": statistics.median(stds) if stds else 0.0,
        "mean_reward_std": statistics.fmean(stds) if stds else 0.0,
        # rung-occupancy histogram: {n_distinct_rungs: n_prompts}
        "rung_histogram": dict(
            sorted(Counter(p["n_distinct_rungs"] for p in per_prompt).items())
        ),
        "per_prompt": per_prompt,
    }


def classify_gate(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify the probe summary into a gate verdict. Pure.

    Precedence: GO_RFT > GRPO_REVIVAL > NO_GO > MARGINAL.
    """
    frontier = summary["frontier_frac"]
    mean_hr = summary["mean_headroom"]
    collapsed = summary["frac_collapsed_groups"]
    med_std = summary["median_reward_std"]

    rft_go = frontier >= FRONTIER_MIN and mean_hr >= HEADROOM_MIN
    grpo_revival = collapsed < COLLAPSED_MAX and med_std >= REWARD_STD_MIN
    no_go = frontier < FRONTIER_NOGO

    if rft_go:
        verdict = "GO_RFT"
    elif grpo_revival:
        verdict = "GRPO_REVIVAL"
    elif no_go:
        verdict = "NO_GO"
    else:
        verdict = "MARGINAL"

    return verdict, {
        "rft_go": rft_go,
        "grpo_revival": grpo_revival,
        "no_go": no_go,
        "checks": {
            f"frontier_frac >= {FRONTIER_MIN}": frontier >= FRONTIER_MIN,
            f"mean_headroom >= {HEADROOM_MIN}": mean_hr >= HEADROOM_MIN,
            f"frac_collapsed_groups < {COLLAPSED_MAX}": collapsed < COLLAPSED_MAX,
            f"median_reward_std >= {REWARD_STD_MIN}": med_std >= REWARD_STD_MIN,
            f"frontier_frac < {FRONTIER_NOGO} (no-go)": no_go,
        },
    }


_VERDICT_ACTION = {
    "GO_RFT": "Proceed to Stage 1 — RFT pilot (best-of-8 SFT).",
    "GRPO_REVIVAL": "GRPO condition met at matched settings — run the merged 50-step diagnostic before RFT.",
    "NO_GO": "No single-turn headroom — kill the single-turn RL/RFT track; escalate to multi-turn env or ship SFT-only.",
    "MARGINAL": "Between gates — inspect rung_histogram; likely re-probe with more prompts or a different base checkpoint.",
}


def _score(completions_per_prompt: list[list[str]], gts: list[dict[str, Any]]) -> list[list[float]]:
    from llm_workflow_agents.training.rewards.reward_business_logic import (
        reward_business_logic,
    )

    out: list[list[float]] = []
    for comps, gt in zip(completions_per_prompt, gts):
        out.append(
            reward_business_logic(
                prompts=[""] * len(comps),
                completions=comps,
                ground_truths=[gt] * len(comps),
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SFT checkpoint to probe.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-prompts", type=int, default=500)
    parser.add_argument("--n-completions", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from preflight_entropy_diag import (
        _decode_gt,
        _generate_for_checkpoint,
        _sample_prompts,
    )

    prompts = _sample_prompts(args.data_dir, args.split, args.n_prompts, args.seed)
    gts = [_decode_gt(p["ground_truth"]) for p in prompts]
    print(f"[data] sampled {len(prompts)} prompts from {args.data_dir}/{args.split}.jsonl",
          flush=True)

    gen_kwargs = dict(
        checkpoint=args.checkpoint,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    t0 = time.time()
    print("[gen] greedy pass (do_sample=False)", flush=True)
    greedy = _generate_for_checkpoint(n_completions=1, do_sample=False, **gen_kwargs)
    greedy_texts = [c[0] if c else "" for c in greedy]
    print(f"[gen] sampled pass (N={args.n_completions}, T={args.temperature})", flush=True)
    sampled = _generate_for_checkpoint(
        n_completions=args.n_completions, do_sample=True, **gen_kwargs
    )

    greedy_rewards = _score([[t] for t in greedy_texts], gts)
    greedy_scalar = [r[0] for r in greedy_rewards]
    sampled_rewards = _score(sampled, gts)

    summary = summarize_headroom(sampled_rewards, greedy_scalar)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    verdict, detail = classify_gate(summary)

    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "data": f"{args.data_dir}/{args.split}.jsonl",
            "n_prompts": summary["n_prompts"],
            "n_completions": args.n_completions,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "headroom_threshold": HEADROOM_THRESHOLD,
        },
        "summary": {k: v for k, v in summary.items() if k != "per_prompt"},
        "gate": {"verdict": verdict, **detail, "action": _VERDICT_ACTION[verdict]},
        "per_prompt": summary["per_prompt"],
    }
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(
        f"\n=== RFT headroom probe — {Path(args.checkpoint).name} ===\n"
        f"  frontier_frac         = {summary['frontier_frac']:.3f}  (GO_RFT >= {FRONTIER_MIN})\n"
        f"  mean_headroom         = {summary['mean_headroom']:.4f}  (GO_RFT >= {HEADROOM_MIN})\n"
        f"  frac_collapsed_groups = {summary['frac_collapsed_groups']:.3f}  (GRPO revival < {COLLAPSED_MAX})\n"
        f"  median_reward_std     = {summary['median_reward_std']:.4f}  (GRPO revival >= {REWARD_STD_MIN})\n"
        f"  rung_histogram        = {summary['rung_histogram']}\n"
        f"  wall_time_s           = {summary['wall_time_s']}\n"
        f"\n  VERDICT: {verdict} — {_VERDICT_ACTION[verdict]}\n"
        f"[done] wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
