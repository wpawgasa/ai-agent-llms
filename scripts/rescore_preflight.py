#!/usr/bin/env python3
"""Re-score a stored preflight run under the current reward function.

Cheap CPU-only validation step. Loads stored completions from a preflight
JSON, re-resolves the ground truths from the validation split (using the
same seed the preflight used so the prompt selection is identical), and
applies ``reward_business_logic`` to compute fresh per-prompt rewards.

Used to A/B the 2026-05-26 active-component-mean preflight against the
2026-05-26 fixed-denominator reward change without re-running generation.

Usage:
    python scripts/rescore_preflight.py \
        --input runs/preflight/postredesign_20x8_ckpt500.json \
        --data-dir data/output/grpo/task_a --split validation --seed 42
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    p.add_argument("--split", default="validation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from preflight_entropy_diag import _decode_gt, _sample_prompts
    from llm_workflow_agents.training.rewards.reward_business_logic import (
        reward_business_logic,
    )

    data = json.loads(args.input.read_text())
    n_prompts = data["config"]["n_prompts"]
    prompts = _sample_prompts(args.data_dir, args.split, n_prompts, args.seed)

    print(f"[load] {args.input}")
    print(f"[data] reloaded {len(prompts)} prompts from {args.data_dir}/{args.split}.jsonl")

    for ckpt, summary in data["by_checkpoint"].items():
        per_prompt = summary["per_prompt"]
        assert len(per_prompt) == len(prompts), (
            f"prompt count mismatch: stored={len(per_prompt)}, reloaded={len(prompts)}"
        )

        new_stds: list[float] = []
        new_means: list[float] = []
        old_stds: list[float] = [pp["reward_std"] for pp in per_prompt]
        old_means: list[float] = [pp["reward_mean"] for pp in per_prompt]

        for pp, p_meta in zip(per_prompt, prompts):
            comps = pp["completions"]
            gt = _decode_gt(p_meta["ground_truth"])
            new_rewards = reward_business_logic(
                prompts=[""] * len(comps),
                completions=comps,
                ground_truths=[gt] * len(comps),
            )
            new_stds.append(
                statistics.pstdev(new_rewards) if len(new_rewards) > 1 else 0.0
            )
            new_means.append(statistics.fmean(new_rewards))

        old_collapsed = sum(1 for s in old_stds if s < 0.01) / len(old_stds)
        new_collapsed = sum(1 for s in new_stds if s < 0.01) / len(new_stds)

        print(f"\n=== {ckpt} (OLD reward vs NEW reward, same {len(prompts)} prompts × "
              f"{len(per_prompt[0]['completions'])} completions) ===")
        print(f"{'metric':<26} {'OLD':>10} {'NEW':>10}  {'delta':>10}")
        for name, old, new in [
            ("mean_reward_std", statistics.fmean(old_stds), statistics.fmean(new_stds)),
            ("median_reward_std", statistics.median(old_stds), statistics.median(new_stds)),
            ("mean_reward", statistics.fmean(old_means), statistics.fmean(new_means)),
            ("frac_collapsed_groups", old_collapsed, new_collapsed),
        ]:
            print(f"{name:<26} {old:>10.4f} {new:>10.4f}  {new - old:>+10.4f}")

        print(f"\nper-prompt std (OLD → NEW):")
        for i, (o, n) in enumerate(zip(old_stds, new_stds)):
            arrow = "→"
            change = ""
            if o < 0.01 and n >= 0.01:
                change = "  [UN-COLLAPSED]"
            elif o >= 0.01 and n < 0.01:
                change = "  [NEWLY COLLAPSED]"
            print(f"  idx={i:2d}  {o:.4f} {arrow} {n:.4f}{change}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
