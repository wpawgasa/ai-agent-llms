#!/usr/bin/env python3
"""Build the Cat A contrastive preference pair set (DPO/ORPO format).

Why pairs instead of GRPO: R18 / docs/grpo_reward_resolution_investigation.md.
The GRPO reward takes 11 distinct values on 206 real completions and is exactly
1.0 on 81.1% of them, so groups tie and the advantage is zero. A preference
objective needs no reward variance — every pair carries a guaranteed margin.

Output is TRL's *conversational preference* format, one JSON object per line:

    {"prompt":   [{"role": "system", ...}, {"role": "user", ...}],
     "chosen":   [{"role": "assistant", "content": "<gold turn>"}],
     "rejected": [{"role": "assistant", "content": "<corrupted turn>"}],
     "rejected_type": "drop_tool_calls" | "flip_state_transition" | "corrupt_tool_args",
     "source": "synthetic",
     "prompt_fingerprint": "<fingerprint of the prompt's user turns>"}

`rejected_type` and `prompt_fingerprint` are not consumed by TRL; they are
there so the set can be audited, filtered and de-contaminated after the fact.

CONTAMINATION: pairs are refused for any conversation whose user-turn
fingerprint appears in the held-out audit set. Those 206 conversations are the
only contamination-free measuring stick for this lineage, and training on them
destroys it. GRPO train/validation are currently disjoint from it by
construction (the held-out set is drawn from the *test* split, which is
deliberately excluded from data/output/grpo/) — verified 0 overlap on
2026-08-16 — but the guard is enforced anyway rather than assumed.

These negatives are SYNTHETIC. R18 recommends complementing them with negatives
mined from the model's own generations, which are on-distribution; synthetic
negatives alone teach discrimination against this corruption function. Merging
those in is a follow-up; this script emits `source: "synthetic"` on every row so
the two can be distinguished once model-sourced negatives exist.

Usage:
    .venv-train/bin/python scripts/build_preference_pairs.py \\
        --data-dir data/output/grpo/task_a --split train \\
        --heldout data/output/heldout/cat_a_v2_test_not_in_v1/test.jsonl \\
        --out data/output/preference/task_a/train.jsonl

    # Inspect the composition without writing:
    .venv-train/bin/python scripts/build_preference_pairs.py --dry-run --limit 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_workflow_agents.data.heldout_clean_set import (  # noqa: E402
    load_prefix_fingerprints,
    user_turn_fingerprint,
)
from llm_workflow_agents.data.preference_pairs import (  # noqa: E402
    corrupt_tool_args,
    drop_tool_calls,
    flip_state_transition,
)

CORRUPTIONS = ("drop_tool_calls", "flip_state_transition", "corrupt_tool_args")


def _apply(kind: str, text: str, gt: dict[str, Any], seed: int) -> str | None:
    if kind == "drop_tool_calls":
        return drop_tool_calls(text)
    if kind == "flip_state_transition":
        return flip_state_transition(text, gt.get("valid_transitions"), seed=seed)
    if kind == "corrupt_tool_args":
        return corrupt_tool_args(text, seed=seed)
    raise ValueError(f"unknown corruption {kind!r}")


def build(
    data_dir: Path,
    split: str,
    heldout: list[Path],
    seed: int,
    limit: int | None,
    per_type_cap: int | None = None,
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    """Return (pairs, type_counts, skip_reasons)."""
    from llm_workflow_agents.training.grpo import _load_grpo_jsonl

    # Prefix fingerprints, not whole-conversation ones. A per-turn row's prompt
    # holds only the first k user turns, so its fingerprint can never equal the
    # full conversation's — comparing against those would make this guard inert
    # and silently pass every contaminated row.
    contaminated = load_prefix_fingerprints(heldout) if heldout else set()

    ds = _load_grpo_jsonl(Path(data_dir), split=split)
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in CORRUPTIONS}
    types: Counter = Counter()
    skips: Counter = Counter()

    for i in range(len(ds)):
        row = ds[i]
        gt = json.loads(row["ground_truth"])
        messages = gt.get("messages") or []
        if not messages:
            skips["no_gold_message"] += 1
            continue
        gold = messages[0].get("content") or ""
        if not gold.strip():
            skips["empty_gold"] += 1
            continue

        # Keyed on the prompt's user turns, matched against the held-out
        # conversations expanded into all their prefixes (see above).
        fp = user_turn_fingerprint({"messages": row["prompt"]})
        if fp in contaminated:
            skips["heldout_contamination"] += 1
            continue

        # Emit EVERY applicable corruption, then balance below. Taking only the
        # first applicable one skews hard toward flip_state_transition: it
        # applies to nearly every turn, while both tool corruptions need a
        # tool-bearing turn (~36% of rows). Measured, first-applicable gave
        # 76.9% flips — over-weighting the failure C2 has largely solved
        # (spurious self-loops 2.8%) and starving argument fidelity, which is
        # the actual bottleneck (18 of 71 tool-bearing held-out rows).
        applicable = 0
        for kind in CORRUPTIONS:
            rejected = _apply(kind, gold, gt, seed=seed + i)
            if rejected and rejected.strip() and rejected != gold:
                buckets[kind].append(
                    {
                        "prompt": row["prompt"],
                        "chosen": [{"role": "assistant", "content": gold}],
                        "rejected": [{"role": "assistant", "content": rejected}],
                        "rejected_type": kind,
                        "source": "synthetic",
                        "prompt_fingerprint": fp,
                    }
                )
                applicable += 1
        if not applicable:
            skips["no_applicable_corruption"] += 1

    # Balance to equal shares, capped by the scarcest type. The tool
    # corruptions are the scarce ones, so this is effectively "use every
    # tool-bearing row, and match it with an equal number of state flips".
    available = {k: len(v) for k, v in buckets.items() if v}
    if not available:
        return [], Counter(), skips
    per_type = min(available.values())
    if per_type_cap:
        per_type = min(per_type, per_type_cap)

    pairs: list[dict[str, Any]] = []
    for kind, bucket in buckets.items():
        if not bucket:
            skips[f"no_candidates_{kind}"] += 1
            continue
        chosen_idx = rng.sample(range(len(bucket)), per_type)
        pairs.extend(bucket[j] for j in sorted(chosen_idx))
        types[kind] = per_type
        dropped = len(bucket) - per_type
        if dropped:
            skips[f"balance_dropped_{kind}"] += dropped

    rng.shuffle(pairs)
    if limit:
        pairs = pairs[:limit]
        types = Counter(p["rejected_type"] for p in pairs)
    return pairs, types, skips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--heldout",
        type=Path,
        action="append",
        default=None,
        help="Held-out split(s) to exclude by user-turn fingerprint.",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--per-type",
        type=int,
        default=None,
        help="Cap pairs per rejected_type (default: the scarcest type's count).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs, types, skips = build(
        args.data_dir, args.split, args.heldout or [], args.seed, args.limit,
        per_type_cap=args.per_type,
    )

    total = len(pairs)
    print(f"\npairs built: {total}")
    print("  by rejected_type:")
    for kind in CORRUPTIONS:
        n = types.get(kind, 0)
        share = f"{100 * n / total:.1f}%" if total else "-"
        print(f"    {kind:<24} {n:>7}  {share:>7}")
    print("  skipped:")
    for reason, n in skips.most_common():
        print(f"    {reason:<24} {n:>7}")

    if args.dry_run or not args.out:
        print("\n(dry run — nothing written)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {total} pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
