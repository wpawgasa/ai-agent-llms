#!/usr/bin/env python3
"""Mine on-distribution negatives from the policy's OWN generations.

`scripts/build_preference_pairs.py` produces synthetic negatives by corrupting
gold turns. R18 warns that synthetic-only negatives teach the model to
discriminate against the corruption function rather than to get the task right —
the same trap R15 documents at corpus level, where a structurally uniform edit
was learned as an unconditional habit.

This script closes that gap: generate greedily from the checkpoint on TRAIN
prompts, keep the generations that are actually wrong, and pair each against its
gold turn. Those negatives are on-distribution by construction — they are the
mistakes the model really makes, not ones invented for it.

Selection is biased toward tool-bearing turns on purpose. Measured on the
held-out audit, greedy decoding is wrong on 38.0% of tool-bearing rows versus
8.9% of no-tool rows, so sampling there yields far more negatives per GPU-minute
and concentrates them on the current bottleneck (18 of 71 tool-bearing rows have
the right tool and wrong arguments).

NEVER point --split at the held-out set. Those 206 conversations are the only
contamination-free measuring stick for this lineage; mining negatives from them
and training on the result destroys it. The prefix-fingerprint guard from
`build_preference_pairs.py` is applied here too.

Failure types recorded in `rejected_type`, mirroring the audit's taxonomy:
  model_no_tool_call   gold turn calls a tool, generation emits none
  model_wrong_args     right tool name, different arguments
  model_wrong_tool     a tool call, but not the gold tool
  model_state_error    state annotation disagrees with gold
  model_other          wrong in some other way

Usage:
    .venv-train/bin/python scripts/mine_model_negatives.py \\
        --checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767 \\
        --n-prompts 400 --out data/output/preference/task_a/model_negatives.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_workflow_agents.data.heldout_clean_set import (  # noqa: E402
    load_prefix_fingerprints,
    reserve_guardrail_slice,
    user_turn_fingerprint,
)


def _select_prompts(
    data_dir: Path, split: str, n: int, tool_share: float, seed: int
) -> list[dict[str, Any]]:
    """Pick N unique-conversation rows, over-weighting tool-bearing turns."""
    from llm_workflow_agents.training.grpo import _load_grpo_jsonl

    ds = _load_grpo_jsonl(data_dir, split=split)
    seen: set[str] = set()
    tool_rows: list[dict[str, Any]] = []
    other_rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        row = ds[i]
        # Dedupe by the conversation's opening user turn so we never pick two
        # adjacent turns of one conversation — near-duplicate prompts would
        # inflate the apparent error count without adding information.
        prompt = row["prompt"]
        first_user = next(
            (m.get("content", "") for m in prompt if m.get("role") == "user"), ""
        )
        if first_user in seen:
            continue
        seen.add(first_user)
        gt = json.loads(row["ground_truth"])
        # `_generate_for_checkpoint` keys on "prompt_messages" (the shape
        # `_sample_prompts` emits), not the loader's "prompt".
        entry = {
            "prompt_messages": row["prompt"],
            "ground_truth": row["ground_truth"],
        }
        (tool_rows if gt.get("tool_calls") else other_rows).append(entry)

    rng = random.Random(seed)
    rng.shuffle(tool_rows)
    rng.shuffle(other_rows)
    n_tool = min(len(tool_rows), int(round(n * tool_share)))
    n_other = min(len(other_rows), n - n_tool)
    picked = tool_rows[:n_tool] + other_rows[:n_other]
    rng.shuffle(picked)
    return picked


def _excluded_fingerprints(
    data_dir: Path,
    split: str,
    heldout: list[Path],
    guardrail_reserved_fraction: float,
    guardrail_reserved_seed: int,
) -> set[str]:
    """Union of the held-out contamination guard and, for --split validation
    only, the guardrail-reserved slice dpo.py's R5 callback depends on.

    Gated to validation only: mining from --split train has no overlap risk
    with the guardrail, which only ever reads from validation (dpo.py
    ::_build_heldout_callback, Task 6) — computing the reservation for train
    would just waste a file read and could never exclude anything relevant.
    """
    excluded = load_prefix_fingerprints(heldout or [])
    if split == "validation":
        excluded = excluded | reserve_guardrail_slice(
            data_dir,
            split="validation",
            reserved_fraction=guardrail_reserved_fraction,
            seed=guardrail_reserved_seed,
        )
    return excluded


def _classify(completion: str, gt: dict[str, Any]) -> str | None:
    """Return the failure type, or None when the generation is acceptable."""
    from llm_workflow_agents.training.reward_utils import (
        extract_state_annotations,
        extract_tool_calls,
    )

    gt_tools = gt.get("tool_calls") or []
    pred_tools = extract_tool_calls(completion)
    gt_states = [(s.get("from", ""), s.get("to", "")) for s in gt.get("state_sequence", [])]
    pred_states = extract_state_annotations(completion)

    gt_names = {t.get("name") for t in gt_tools}
    pred_names = {t.get("name") for t in pred_tools}

    if gt_tools and not pred_tools:
        return "model_no_tool_call"
    if gt_tools and pred_tools:
        if not (gt_names & pred_names):
            return "model_wrong_tool"
        gt_args = {json.dumps(t.get("arguments") or {}, sort_keys=True) for t in gt_tools}
        pred_args = {
            json.dumps(t.get("arguments") or {}, sort_keys=True) for t in pred_tools
        }
        if not (gt_args & pred_args):
            return "model_wrong_args"
    if not gt_tools and pred_tools:
        return "model_wrong_tool"  # spurious call on a turn needing none

    if gt_states and pred_states:
        if tuple(pred_states[0]) != tuple(gt_states[0]):
            return "model_state_error"
    elif gt_states and not pred_states:
        return "model_other"

    return None  # matched on every axis we score — not a usable negative


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--heldout", type=Path, action="append", default=None)
    ap.add_argument(
        "--guardrail-reserved-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of --split validation reserved for dpo.py's R5 held-out "
            "guardrail and excluded from mining. Ignored for --split train. "
            "MUST MATCH data.guardrail_reserved_fraction in "
            "configs/training/dpo_cat_a.yaml, which is where "
            "dpo.py::_build_heldout_callback reads its value from. The two "
            "call sites invoke reserve_guardrail_slice independently and "
            "nothing enforces agreement: a mismatch here, in "
            "--guardrail-reserved-seed, or in the corpus path (--data-dir vs "
            "that config's data.heldout_data_source) silently produces two "
            "DIFFERENT reserved sets, reintroducing the guardrail/mining "
            "overlap this flag exists to prevent."
        ),
    )
    ap.add_argument(
        "--guardrail-reserved-seed",
        type=int,
        default=42,
        help=(
            "Seed for the guardrail reservation. MUST MATCH "
            "data.guardrail_reserved_seed in configs/training/dpo_cat_a.yaml — "
            "see --guardrail-reserved-fraction: a differing seed produces a "
            "different reserved set and silently defeats the disjointness "
            "guarantee."
        ),
    )
    ap.add_argument("--n-prompts", type=int, default=400)
    ap.add_argument(
        "--tool-share",
        type=float,
        default=0.75,
        help="Fraction of sampled prompts that must be tool-bearing turns.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.split == "test":
        raise SystemExit(
            "refusing to mine from the test split — it feeds the held-out audit "
            "set, the only contamination-free measuring stick for this lineage"
        )

    from preflight_entropy_diag import _generate_for_checkpoint

    contaminated = _excluded_fingerprints(
        args.data_dir,
        args.split,
        args.heldout or [],
        args.guardrail_reserved_fraction,
        args.guardrail_reserved_seed,
    )
    rows = _select_prompts(
        args.data_dir, args.split, args.n_prompts, args.tool_share, args.seed
    )
    rows = [
        r
        for r in rows
        if user_turn_fingerprint({"messages": r["prompt_messages"]}) not in contaminated
    ]
    print(f"[mine] {len(rows)} prompts selected (tool-share target {args.tool_share})")

    t0 = time.time()
    gens = _generate_for_checkpoint(
        checkpoint=args.checkpoint,
        prompts=rows,
        n_completions=1,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        seed=args.seed,
        do_sample=False,  # greedy: the model's actual best answer, not a sample
    )
    print(f"[mine] generation done in {time.time() - t0:.0f}s")

    from collections import Counter

    pairs: list[dict[str, Any]] = []
    kinds: Counter = Counter()
    skips: Counter = Counter()
    for row, comp_list in zip(rows, gens):
        completion = (comp_list[0] if comp_list else "") or ""
        gt = json.loads(row["ground_truth"])
        gold = (gt.get("messages") or [{}])[0].get("content") or ""
        if not completion.strip() or not gold.strip():
            skips["empty"] += 1
            continue
        if completion.strip() == gold.strip():
            skips["identical_to_gold"] += 1
            continue
        kind = _classify(completion, gt)
        if kind is None:
            skips["generation_acceptable"] += 1
            continue
        kinds[kind] += 1
        pairs.append(
            {
                "prompt": row["prompt_messages"],
                "chosen": [{"role": "assistant", "content": gold}],
                "rejected": [{"role": "assistant", "content": completion}],
                "rejected_type": kind,
                "source": "model",
                "prompt_fingerprint": user_turn_fingerprint(
                    {"messages": row["prompt_messages"]}
                ),
            }
        )

    print(f"\nnegatives mined: {len(pairs)} of {len(rows)} prompts "
          f"({100 * len(pairs) / max(len(rows), 1):.1f}% error rate)")
    for kind, n in kinds.most_common():
        print(f"    {kind:<22} {n:>6}")
    print("  skipped:")
    for reason, n in skips.most_common():
        print(f"    {reason:<22} {n:>6}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(pairs)} model-sourced negatives -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
