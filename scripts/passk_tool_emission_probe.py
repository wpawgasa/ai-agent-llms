#!/usr/bin/env python3
"""pass@k tool-emission probe — is the missing tool call in the policy at all?

Closes the §4-step-2 item of ``docs/grpo_tool_emission_gap_review.md`` ("add a
pass@k (T≈1.0) probe on the tool-expected rows specifically", overdue since
2026-07-09) and answers the question §8 leaves open.

§8 established with greedy decoding that ckpt-1000 does not emit the expected
tool call — not at the anchor turn (0.335 emission) and not when free-running
across a two-turn window (0.320). Greedy tells you what the *mode* of the policy
does; it cannot tell you whether the correct behaviour exists in the
distribution at all. That distinction picks the next lever:

  - **Sampling recovers the call** -> the behaviour IS in the policy, greedy just
    doesn't surface it. That is an RL / decoding problem, and it partly revives
    the RFT case §6.2 left at MARGINAL — far cheaper than a retrain.
  - **Sampling does not recover it** -> the behaviour is absent from the policy.
    That is a data/SFT problem, and §4 step 4's combined factorial run
    (``response_only`` masking + tool-turn upsampling) is the right lever.

Reuses the anchor machinery from ``free_running_multiturn_probe.py`` verbatim, so
the anchors here are a seed-42 subset of the ones §8 reports, and the greedy
baseline is read back from that probe's artifact (paired, same anchors, no extra
GPU cost) rather than re-measured.

Reports the unbiased Chen et al. (2021) pass@k estimator at k = 1/2/4/8 from
n=8 samples per anchor, for two criteria: *emission* (any tool call) and
*name-match* (the expected tool name). The k-curve matters as much as the endpoint
— a curve that climbs with k means real, recoverable probability mass; a flat one
near zero means the behaviour simply isn't there.

Pure functions (``pass_at_k``, ``summarize_passk``, ``classify_gate``) are
unit-tested in tests/unit/test_passk_tool_emission_probe.py.

Usage:
    .venv-train/bin/python scripts/passk_tool_emission_probe.py \
        --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000 \
        --data-dir data/output/grpo/task_a --split validation \
        --n-anchors 120 --n-samples 8 --temperature 1.0 \
        --greedy-baseline runs/preflight/free_running_probe_ckpt1000.json \
        --output runs/preflight/passk_tool_emission_ckpt1000.json
"""

from __future__ import annotations

import argparse
import gc
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

# k values reported from the n=8 sample budget.
K_VALUES = (1, 2, 4, 8)

# Gate thresholds, applied to pass@8 name-match minus the greedy name-match rate
# on the same anchors. "Materially recovers" reuses §8's +0.20 material bar so
# the two probes' verdicts are stated on a common scale.
MATERIAL_RECOVERY = 0.20
WEAK_RECOVERY = 0.05


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, Codex §2.1). Pure.

    Probability that at least one of k samples drawn without replacement from n
    total samples is correct, given c of the n are correct:
    ``1 - C(n-c, k) / C(n, k)``, computed in product form to avoid overflow.
    """
    if k <= 0 or n <= 0:
        return 0.0
    k = min(k, n)
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    prob_none = 1.0
    for i in range(k):
        prob_none *= (n - c - i) / (n - i)
    return 1.0 - prob_none


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_passk(
    records: list[dict[str, Any]], k_values: tuple[int, ...] = K_VALUES
) -> dict[str, Any]:
    """Reduce per-anchor sample counts to the probe summary. Pure — no model, no I/O.

    Each record carries ``n_samples``, ``c_emitted``, ``c_name_match``,
    ``best_f1``, ``mean_f1`` and (optionally) the paired greedy fields
    ``greedy_emitted`` / ``greedy_name_match``.
    """
    n = len(records)
    emission_curve = {
        f"pass@{k}": _mean(
            [pass_at_k(r["n_samples"], r["c_emitted"], k) for r in records]
        )
        for k in k_values
    }
    name_curve = {
        f"pass@{k}": _mean(
            [pass_at_k(r["n_samples"], r["c_name_match"], k) for r in records]
        )
        for k in k_values
    }

    paired = [r for r in records if "greedy_name_match" in r]
    greedy_name = (
        sum(1 for r in paired if r["greedy_name_match"]) / len(paired) if paired else 0.0
    )
    greedy_emit = (
        sum(1 for r in paired if r["greedy_emitted"]) / len(paired) if paired else 0.0
    )
    k_max = max(k_values)

    return {
        "n_anchors": n,
        "n_samples_per_anchor": records[0]["n_samples"] if records else 0,
        "pass_at_k_emission": emission_curve,
        "pass_at_k_name_match": name_curve,
        "greedy_emission_rate": greedy_emit,
        "greedy_name_match_rate": greedy_name,
        "n_paired_with_greedy": len(paired),
        "recovery_emission": emission_curve[f"pass@{k_max}"] - greedy_emit,
        "recovery_name_match": name_curve[f"pass@{k_max}"] - greedy_name,
        # Anchors where the correct call never appears in any sample — the
        # population a retrain would have to move.
        "frac_never_name_match": (
            sum(1 for r in records if r["c_name_match"] == 0) / n if n else 0.0
        ),
        "frac_never_emitted": (
            sum(1 for r in records if r["c_emitted"] == 0) / n if n else 0.0
        ),
        "frac_always_name_match": (
            sum(1 for r in records if r["c_name_match"] == r["n_samples"]) / n
            if n
            else 0.0
        ),
        "mean_per_sample_emission": _mean(
            [r["c_emitted"] / r["n_samples"] for r in records]
        ),
        "mean_per_sample_name_match": _mean(
            [r["c_name_match"] / r["n_samples"] for r in records]
        ),
        "mean_best_f1": _mean([r["best_f1"] for r in records]),
        "mean_mean_f1": _mean([r["mean_f1"] for r in records]),
    }


def classify_gate(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify the summary into a verdict. Pure.

    SAMPLING_RECOVERS  the call is in the distribution and sampling finds it ->
                       RL / decoding lever is viable, retrain not yet forced.
    WEAK_SIGNAL        some recoverable mass, not enough to bet an RL run on.
    ABSENT_FROM_POLICY sampling finds nothing greedy didn't -> data/SFT problem.
    """
    recovery = summary["recovery_name_match"]
    if recovery >= MATERIAL_RECOVERY:
        verdict = "SAMPLING_RECOVERS"
    elif recovery >= WEAK_RECOVERY:
        verdict = "WEAK_SIGNAL"
    else:
        verdict = "ABSENT_FROM_POLICY"
    return verdict, {
        "checks": {
            f"recovery_name_match >= {MATERIAL_RECOVERY}": recovery >= MATERIAL_RECOVERY,
            f"recovery_name_match >= {WEAK_RECOVERY}": recovery >= WEAK_RECOVERY,
        },
    }


_VERDICT_ACTION = {
    "SAMPLING_RECOVERS": (
        "The correct tool call IS in the policy's distribution — greedy just does "
        "not surface it. Prefer the RL/decoding lever (re-open the RFT case §6.2 "
        "left MARGINAL) over the §4-step-4 SFT retrain."
    ),
    "WEAK_SIGNAL": (
        "Some recoverable probability mass, but too little to bet an RL run on. "
        "Proceed with §4 step 4's factorial SFT run; note the sampled headroom "
        "when re-deriving the target bar."
    ),
    "ABSENT_FROM_POLICY": (
        "Sampling finds nothing greedy did not — the behaviour is absent from the "
        "policy, not merely hidden by decoding. This is a data/SFT problem: run "
        "§4 step 4's combined factorial (response_only + tool-turn upsampling). "
        "An RFT/GRPO pilot cannot reinforce what the policy never emits."
    ),
}


def _load_greedy_baseline(path: Path) -> dict[tuple[Any, int], dict[str, bool]]:
    """Index the §8 probe's per-anchor condition-A results by (conversation_id, target_index)."""
    data = json.loads(Path(path).read_text())
    return {
        (r["conversation_id"], r["target_index"]): {
            "greedy_emitted": r["a_emitted"],
            "greedy_name_match": r["a_name_match"],
        }
        for r in data.get("per_anchor", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-anchors", type=int, default=120)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--greedy-baseline",
        type=Path,
        help="free_running_multiturn_probe.py JSON to pair the greedy rate against.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from free_running_multiturn_probe import (
        _build_prompts,
        _generate,
        _load_conversations,
        _load_model,
        _select_anchors,
    )

    from llm_workflow_agents.training.reward_utils import extract_tool_calls, tool_call_f1

    convs = _load_conversations(args.data_dir, args.split)
    # Same seed as §8 -> these anchors are a nested subset of that probe's 200.
    anchors, census = _select_anchors(convs, args.n_anchors, args.seed)
    print(
        f"[data] {len(convs)} conversations; anchor census: {json.dumps(census)}",
        flush=True,
    )

    greedy_by_anchor: dict[tuple[Any, int], dict[str, bool]] = {}
    if args.greedy_baseline:
        greedy_by_anchor = _load_greedy_baseline(args.greedy_baseline)
        print(
            f"[data] greedy baseline: {len(greedy_by_anchor)} anchors from "
            f"{args.greedy_baseline}",
            flush=True,
        )

    conv_by_id = {c.get("conversation_id"): c for c in convs}
    prompts = [_build_prompts(conv_by_id, a)[0] for a in anchors]

    t0 = time.time()
    model, inner_tok = _load_model(args.checkpoint, args.max_seq_length)

    # One pass per sample index; each pass reseeded so the n draws are independent.
    samples_per_anchor: list[list[str]] = [[] for _ in anchors]
    for s in range(args.n_samples):
        texts = _generate(
            model,
            inner_tok,
            prompts,
            args.max_new_tokens,
            args.max_seq_length,
            args.batch_size,
            f"sample {s + 1}/{args.n_samples} (T={args.temperature})",
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + s,
        )
        for i, text in enumerate(texts):
            samples_per_anchor[i].append(text)

    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    records: list[dict[str, Any]] = []
    for anchor, samples in zip(anchors, samples_per_anchor, strict=True):
        gt_calls = anchor["gt_tool_calls"]
        gt_names = {c.get("name") for c in gt_calls if isinstance(c, dict)}

        parsed = [extract_tool_calls(s) for s in samples]
        f1s = [tool_call_f1(p, gt_calls) for p in parsed]
        record: dict[str, Any] = {
            "conversation_id": anchor["conversation_id"],
            "complexity_level": anchor["complexity_level"],
            "target_index": anchor["target_index"],
            "gt_tool_names": sorted(n for n in gt_names if n),
            "n_samples": len(samples),
            "c_emitted": sum(1 for p in parsed if p),
            "c_name_match": sum(
                1 for p in parsed if gt_names & {c.get("name") for c in p}
            ),
            "best_f1": max(f1s) if f1s else 0.0,
            "mean_f1": _mean(f1s),
            "samples": samples,
        }
        record.update(
            greedy_by_anchor.get((anchor["conversation_id"], anchor["target_index"]), {})
        )
        records.append(record)

    summary = summarize_passk(records)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    verdict, detail = classify_gate(summary)

    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "data": f"{args.data_dir}/{args.split}.jsonl",
            "n_anchors": summary["n_anchors"],
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "greedy_baseline": str(args.greedy_baseline) if args.greedy_baseline else None,
            "material_recovery": MATERIAL_RECOVERY,
        },
        "anchor_census": census,
        "summary": summary,
        "gate": {"verdict": verdict, **detail, "action": _VERDICT_ACTION[verdict]},
        "per_anchor": records,
    }
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    emit = summary["pass_at_k_emission"]
    name = summary["pass_at_k_name_match"]
    print(
        f"\n=== pass@k tool-emission probe — {Path(args.checkpoint).name} ===\n"
        f"  anchors x samples        = {summary['n_anchors']} x "
        f"{summary['n_samples_per_anchor']}  (T={args.temperature})\n"
        f"  pass@k emission          = "
        + ", ".join(f"{k}:{emit[k]:.3f}" for k in emit)
        + "\n"
        f"  pass@k name-match        = "
        + ", ".join(f"{k}:{name[k]:.3f}" for k in name)
        + "\n"
        f"  greedy emission / name   = {summary['greedy_emission_rate']:.3f}"
        f" / {summary['greedy_name_match_rate']:.3f}"
        f"  (paired n={summary['n_paired_with_greedy']})\n"
        f"  recovery (pass@8 - greedy, name-match) = "
        f"{summary['recovery_name_match']:+.3f}  (material >= {MATERIAL_RECOVERY})\n"
        f"  never name-matched in any sample       = "
        f"{summary['frac_never_name_match']:.3f}\n"
        f"  mean best-of-{summary['n_samples_per_anchor']} strict F1              = "
        f"{summary['mean_best_f1']:.3f}\n"
        f"  wall_time_s              = {summary['wall_time_s']}\n"
        f"\n  VERDICT: {verdict} — {_VERDICT_ACTION[verdict]}\n"
        f"[done] wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
