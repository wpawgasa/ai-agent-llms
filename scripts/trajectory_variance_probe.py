#!/usr/bin/env python3
"""Trajectory GRPO Stage-0 variance probe.

The go/no-go gate that decides whether the multi-turn *trajectory* reward
actually manufactures within-group reward variance BEFORE spending a training
run (design spec: docs/superpowers/specs/2026-07-09-multiturn-trajectory-reward-design.md
§4). For each held-out conversation it duplicates the gold script N times, runs
the in-process replay rollout (free-run + truncate-on-divergence), scores each
trajectory with ``reward_business_logic_trajectory``, and measures whether the N
trajectory rewards spread out — the exact ``reward_std`` that collapsed to ~0 on
the per-turn reward and killed four runs.

Gates (design spec §4):
  GO_TRAJECTORY     : median_reward_std >= 0.05 AND frac_collapsed_groups < 0.50
                      AND mean_model_turns >= 3.0 AND 0.05 < mean_coverage < 0.95
  NO_GO_VARIANCE    : median_reward_std < 0.02  (trajectory lattice still collapses)
  NO_GO_TRUNCATION  : mean_model_turns < 2.0    (over-truncation re-collapses variance)
  MARGINAL          : none of the above (inspect histograms; re-probe)

The reduction (summarize_trajectory_probe) and the gate (classify_gate) are pure
and unit-tested in tests/unit/test_trajectory_probe.py. ``main`` is train-box
only (needs the H100 + Unsloth + trl==1.0.0).

Usage (H100 / .venv-train):
    .venv-train/bin/python scripts/trajectory_variance_probe.py \
        --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000 \
        --data-dir data/output/grpo/task_a --split validation \
        --n-conversations 50 --n-completions 8 --temperature 0.8 --top-p 0.95 \
        --output runs/preflight/trajectory_variance_probe.json
"""

from __future__ import annotations

import argparse
import json
import random
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

# Gate thresholds (design spec §4)
REWARD_STD_GO = 0.05
COLLAPSED_MAX = 0.50
TURNS_GO = 3.0
COVERAGE_LO = 0.05
COVERAGE_HI = 0.95
REWARD_STD_NOGO = 0.02
TURNS_NOGO = 2.0
COLLAPSE_STD_EPS = 0.01  # a group is "collapsed" if its reward std < this


def summarize_trajectory_probe(
    group_rewards: list[list[float]],
    group_coverages: list[list[float]],
    group_metas: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Reduce per-conversation (rewards, coverages, metas) to the probe summary.

    Pure — no model, no I/O. ``group_rewards[i]`` is the list of N trajectory
    rewards for conversation i (one per sampled rollout); ``group_coverages[i]``
    and ``group_metas[i]`` are the matching per-rollout coverage fractions and
    rollout meta dicts.
    """
    per_group: list[dict[str, Any]] = []
    all_coverages: list[float] = []
    all_turns: list[int] = []
    stop_reasons: Counter = Counter()

    for i, (rewards, covs, metas) in enumerate(
        zip(group_rewards, group_coverages, group_metas)
    ):
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        spread = (max(covs) - min(covs)) if covs else 0.0
        n_distinct = len({round(r, 3) for r in rewards})
        per_group.append(
            {
                "group_index": i,
                "reward_std": std,
                "coverage_spread": spread,
                "n_distinct_rungs": n_distinct,
                "collapsed": std < COLLAPSE_STD_EPS,
            }
        )
        all_coverages.extend(covs)
        all_turns.extend(int(m.get("n_model_turns", 0)) for m in metas)
        stop_reasons.update(m.get("stop_reason", "unknown") for m in metas)

    n = len(per_group)
    stds = [g["reward_std"] for g in per_group]
    spreads = [g["coverage_spread"] for g in per_group]
    sorted_cov = sorted(all_coverages)

    def _pct(p: float) -> float:
        if not sorted_cov:
            return 0.0
        idx = min(len(sorted_cov) - 1, max(0, int(round(p * (len(sorted_cov) - 1)))))
        return sorted_cov[idx]

    return {
        "n_groups": n,
        "median_reward_std": statistics.median(stds) if stds else 0.0,
        "mean_reward_std": statistics.fmean(stds) if stds else 0.0,
        "frac_collapsed_groups": (
            sum(1 for g in per_group if g["collapsed"]) / n if n else 0.0
        ),
        "mean_coverage": statistics.fmean(all_coverages) if all_coverages else 0.0,
        "coverage_p10": _pct(0.10),
        "coverage_p90": _pct(0.90),
        "mean_within_group_coverage_spread": (
            statistics.fmean(spreads) if spreads else 0.0
        ),
        "mean_model_turns": statistics.fmean(all_turns) if all_turns else 0.0,
        "stop_reason_histogram": dict(sorted(stop_reasons.items())),
        "rung_histogram": dict(
            sorted(Counter(g["n_distinct_rungs"] for g in per_group).items())
        ),
        "per_group": per_group,
    }


def classify_gate(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify the probe summary into a gate verdict. Pure.

    Precedence: GO_TRAJECTORY > NO_GO_VARIANCE > NO_GO_TRUNCATION > MARGINAL.
    """
    med_std = summary["median_reward_std"]
    collapsed = summary["frac_collapsed_groups"]
    turns = summary["mean_model_turns"]
    cov = summary["mean_coverage"]

    go = (
        med_std >= REWARD_STD_GO
        and collapsed < COLLAPSED_MAX
        and turns >= TURNS_GO
        and COVERAGE_LO < cov < COVERAGE_HI
    )
    if go:
        verdict = "GO_TRAJECTORY"
    elif med_std < REWARD_STD_NOGO:
        verdict = "NO_GO_VARIANCE"
    elif turns < TURNS_NOGO:
        verdict = "NO_GO_TRUNCATION"
    else:
        verdict = "MARGINAL"

    return verdict, {
        "checks": {
            f"median_reward_std >= {REWARD_STD_GO}": med_std >= REWARD_STD_GO,
            f"frac_collapsed_groups < {COLLAPSED_MAX}": collapsed < COLLAPSED_MAX,
            f"mean_model_turns >= {TURNS_GO}": turns >= TURNS_GO,
            f"{COVERAGE_LO} < mean_coverage < {COVERAGE_HI}": (
                COVERAGE_LO < cov < COVERAGE_HI
            ),
            f"median_reward_std < {REWARD_STD_NOGO} (no-go)": med_std < REWARD_STD_NOGO,
            f"mean_model_turns < {TURNS_NOGO} (no-go)": turns < TURNS_NOGO,
        },
    }


_VERDICT_ACTION = {
    "GO_TRAJECTORY": "Trajectory reward manufactures variance — build Task 6 trainer wiring and run the 50-step diagnostic.",
    "NO_GO_VARIANCE": "Trajectory lattice still collapses — abandon multi-turn; fall back to RFT/SFT-only (spec §7).",
    "NO_GO_TRUNCATION": "Over-truncation floors trajectory length — relax stall_turn_limit (2->3) / one-legal-edge tolerance and re-probe, else abandon (spec §7).",
    "MARGINAL": "Between gates — inspect histograms; re-probe with more conversations.",
}


# --------------------------------------------------------------------------- #
# GPU driver (train box only).
# --------------------------------------------------------------------------- #


def _sample_conversations(
    data_dir: Path, split: str, n: int, seed: int
) -> list[dict[str, Any]]:
    """Read the per-conversation JSONL and sample ``n`` raw conversation rows."""
    path = Path(data_dir) / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    if len(rows) > n:
        rows = rng.sample(rows, n)
    return rows


def _build_scripts(conversations: list[dict[str, Any]]) -> list[Any]:
    """Build GoldScripts, enriching the system prompt like the GRPO loader."""
    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
    from llm_workflow_agents.training.trajectory_rollout import build_gold_script

    scripts = []
    for raw in conversations:
        msgs = raw.get("messages", [])
        if msgs and msgs[0].get("role") == "system" and raw.get("workflow_graph"):
            enriched = build_enriched_system_prompt(
                raw, msgs[0].get("content") or "", force_rebuild=True
            )
        elif msgs and msgs[0].get("role") == "system":
            enriched = msgs[0].get("content") or ""
        else:
            enriched = ""
        try:
            scripts.append(build_gold_script(raw, enriched))
        except ValueError as exc:  # invariant violation / no assistant turns
            print(f"[skip] {raw.get('conversation_id')}: {exc}", flush=True)
    return scripts


def _load_model_and_tokenizer(checkpoint: str, max_seq_length: int):  # noqa: ANN201
    """Load the SFT checkpoint for inference (mirrors preflight_entropy_diag).

    Uses the *same* proven Gemma-4 workaround as the headroom probe rather than
    grpo.py's ``_unwrap_unsloth_gemma4_kv_zero_proxy``. The unwrap strips the
    KV-zero proxy off ``get_text_config`` returns, but Gemma-4 26B-A4B load
    trips a different surface: ``Gemma4Config.__post_init__`` constructs a
    ``Gemma4TextConfig`` whose transformers 5.x ``validate_token_ids`` iterates
    the config and raw-``getattr``s ``num_kv_shared_layers`` on the proxy,
    hitting its raise. The proxy must stay alive (cache ``__init__`` relies on
    ``hasattr(...) == False``), so we filter that name out of the proxy's
    ``__iter__`` instead — the fix that lets the headroom probe load this same
    checkpoint. Order matters: import unsloth first (it installs the proxy),
    patch after.
    """
    from unsloth import FastLanguageModel

    from preflight_entropy_diag import _patch_unsloth_gemma4_proxy_iter

    _patch_unsloth_gemma4_proxy_iter()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Gemma-4 loads as a multimodal ``Gemma4Processor``, not a bare tokenizer.
    # Its ``apply_chat_template(..., tokenize=True)`` returns a *batched* (nested)
    # id list, so ``run_replay_rollout``'s ``_derive_turn_end_id`` /
    # ``_segment_suffix_ids`` (which expect a flat ``list[int]``) break with
    # "int() argument must be ... not 'list'". The inner ``.tokenizer`` is the
    # correct object for text tokenization (the processor wrapper only adds the
    # image path) and returns flat ids — same resolution the headroom probe uses.
    inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
    return model, inner_tok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SFT checkpoint to probe.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/output/grpo/task_a")
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-conversations", type=int, default=50)
    parser.add_argument("--n-completions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--per-turn-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--stall-turn-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from llm_workflow_agents.training.rewards.reward_business_logic import (
        reward_business_logic_trajectory,
    )
    from llm_workflow_agents.training.trajectory_rollout import (
        TrajectoryRolloutConfig,
        assert_trajectory_rollout_support,
        run_replay_rollout,
    )

    assert_trajectory_rollout_support()

    conversations = _sample_conversations(
        args.data_dir, args.split, args.n_conversations, args.seed
    )
    scripts = _build_scripts(conversations)
    print(f"[data] {len(scripts)} scripts from {args.data_dir}/{args.split}.jsonl",
          flush=True)

    model, tokenizer = _load_model_and_tokenizer(args.checkpoint, args.max_seq_length)
    cfg = TrajectoryRolloutConfig(
        max_turns=args.max_turns,
        per_turn_max_new_tokens=args.per_turn_max_new_tokens,
        max_completion_tokens=args.max_completion_tokens,
        stall_turn_limit=args.stall_turn_limit,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    t0 = time.time()
    group_rewards: list[list[float]] = []
    group_coverages: list[list[float]] = []
    group_metas: list[list[dict[str, Any]]] = []
    mask_audit: dict[str, Any] | None = None

    for gi, sc in enumerate(scripts):
        samples = run_replay_rollout(model, tokenizer, [sc] * args.n_completions, cfg)
        gts = [
            {
                "state_sequence": [
                    {"from": f, "to": t} for (f, t) in sc.gold_transitions
                ],
                "tool_calls": sc.gold_tool_calls,
                "terminal_state": sc.terminal_state,
                "terminal_reached": sc.terminal_reached,
                "valid_transitions": sc.valid_transitions,
            }
            for _ in samples
        ]
        rewards = reward_business_logic_trajectory(
            [None] * len(samples),
            [s.turn_texts for s in samples],
            [s.meta for s in samples],
            gts,
        )
        group_rewards.append(rewards)
        group_coverages.append(
            [s.meta["cursor"] / max(s.meta["gold_len"], 1) for s in samples]
        )
        group_metas.append([s.meta for s in samples])

        if mask_audit is None and samples:  # step-0 mask audit on the first group
            s0 = samples[0]
            mask_audit = {
                "len_match": len(s0.env_mask) == len(s0.completion_ids),
                "model_frac": (
                    sum(s0.env_mask) / len(s0.env_mask) if s0.env_mask else 0.0
                ),
                "ends_on_eos": bool(
                    s0.completion_ids
                    and s0.completion_ids[-1] == int(tokenizer.eos_token_id)
                ),
            }
        if (gi + 1) % 10 == 0:
            print(f"[gen] {gi + 1}/{len(scripts)} conversations", flush=True)

    summary = summarize_trajectory_probe(group_rewards, group_coverages, group_metas)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    verdict, detail = classify_gate(summary)

    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "data": f"{args.data_dir}/{args.split}.jsonl",
            "n_conversations": summary["n_groups"],
            "n_completions": args.n_completions,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "summary": {k: v for k, v in summary.items() if k != "per_group"},
        "mask_audit": mask_audit,
        "gate": {"verdict": verdict, **detail, "action": _VERDICT_ACTION[verdict]},
        "per_group": summary["per_group"],
    }
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(
        f"\n=== trajectory variance probe — {Path(args.checkpoint).name} ===\n"
        f"  median_reward_std     = {summary['median_reward_std']:.4f}  (GO >= {REWARD_STD_GO})\n"
        f"  frac_collapsed_groups = {summary['frac_collapsed_groups']:.3f}  (GO < {COLLAPSED_MAX})\n"
        f"  mean_model_turns      = {summary['mean_model_turns']:.2f}  (GO >= {TURNS_GO})\n"
        f"  mean_coverage         = {summary['mean_coverage']:.3f}  (GO in {COVERAGE_LO}-{COVERAGE_HI})\n"
        f"  stop_reasons          = {summary['stop_reason_histogram']}\n"
        f"  rung_histogram        = {summary['rung_histogram']}\n"
        f"  mask_audit            = {mask_audit}\n"
        f"  wall_time_s           = {summary['wall_time_s']}\n"
        f"\n  VERDICT: {verdict} — {_VERDICT_ACTION[verdict]}\n"
        f"[done] wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
