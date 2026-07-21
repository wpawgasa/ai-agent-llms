#!/usr/bin/env python3
"""Re-derive the Cat A target bar for the per-turn-fair composite metric.

Closes the item flagged in ``docs/grpo_tool_emission_gap_review.md`` §3 and left
outstanding in §6.1: the **0.75** target belongs to
``eval/composite_score.compute_weighted_workflow_score``, a *whole-conversation*
metric, but every current measurement uses ``training/grpo._heldout_composite_score``,
a *per-turn-fair* metric on single-turn sliced rows. Carrying 0.75 across is
apples-to-oranges, so "0.7167 is 0.033 short" has never been a valid pass/fail
statement.

**Why the two metrics are not the same quantity.** In the whole-conversation
metric, ``tool_call_f1`` is one AST-F1 over all calls in the conversation, so a
turn where the model correctly makes *no* call is invisible — it contributes to
neither the predicted nor the gold list. In the per-turn metric each row is scored
separately and ``compute_ast_f1([], []) == 1.0``, so every correct abstention earns
**full marks on the 0.4-weight tool term**. On the real corpus 61.7% of rows are
zero-tool, so the per-turn metric hands out a large mass of easy points the old bar
never contemplated. It is therefore an *easier* metric, and an equivalent bar on it
must sit **above** 0.75 — not below.

**Method.** Rather than guess a translation, this computes it against the real row
population produced by the production slicer (``_load_grpo_jsonl``). For a reference
policy that exactly meets the component targets `.claude/rules/05-eval.md` already
commits to (state_transition_accuracy >= 0.85, tool_call_f1 >= 0.85,
task_completion_rate >= 0.70), it evaluates what ``_heldout_composite_score`` would
score on that population — replicating the scorer's term-applicability and
renormalization rules exactly.

Two bars are reported:

  * **component-equivalent** — the per-turn score of a policy exactly meeting the
    component targets. This is the strict reading.
  * **relaxation-matched** — the above scaled by ``0.75 / 0.82``, the same
    relaxation the original bar embodied (the component targets imply a
    whole-conversation composite of 0.82, but the stated target was 0.75).
    This is the like-for-like successor to 0.75 and the recommended bar.

Because the per-turn tool term depends on how a policy handles *abstention* rows —
something the old component targets never specified — both bars are reported across
a sensitivity sweep of the abstention-accuracy assumption rather than as a single
fake-precise number.

Also solves the inverse: given the bar, what tool-F1 on tool-expected rows would a
policy need? That sizes the target for §4 step 4's factorial run directly.

All computation is pure and CPU-only; unit-tested in
tests/unit/test_rederive_target_bar.py.

Usage:
    .venv-train/bin/python scripts/rederive_target_bar.py \
        --data-dir data/output/grpo/task_a --split validation \
        --output runs/preflight/target_bar_rederivation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Component targets from .claude/rules/05-eval.md (state_accuracy.py /
# tool_call_f1.py dataclass docstrings).
TARGET_STATE_ACC = 0.85
TARGET_TOOL_F1 = 0.85
TARGET_TASK_COMPLETION = 0.70

# The stated whole-conversation composite target the project has been quoting.
STATED_OLD_BAR = 0.75

# Composite weights (identical in both metrics; the per-turn scorer renormalizes
# over whichever subset applies to the row).
W_TOOL, W_STATE, W_TASK = 0.4, 0.4, 0.2

# Abstention-accuracy assumptions to sweep. The old component targets never
# specified how a policy should score on zero-tool rows, but the per-turn metric
# scores them, so the bar is a function of this assumption.
ABSTENTION_SWEEP = (1.0, 0.95, 0.90, 0.85)
RECOMMENDED_ABSTENTION = 0.95


def old_composite(state_acc: float, tool_f1: float, task_completion: float) -> float:
    """Whole-conversation composite — ``compute_weighted_workflow_score``. Pure."""
    return W_STATE * state_acc + W_TOOL * tool_f1 + W_TASK * task_completion


def expected_perturn_score(
    profile: list[tuple[bool, bool, bool]],
    state_acc: float,
    tool_f1: float,
    abstention_acc: float,
    task_completion: float,
) -> float:
    """Expected ``_heldout_composite_score`` for a policy with the given components.

    ``profile`` is one ``(tool_expected, has_state_term, has_task_term)`` triple per
    row. Replicates the scorer's rule exactly: the tool term always applies; the
    state term only where GT expects a transition; the task term only on the
    terminal turn; the score is renormalized over the included weights. Pure.
    """
    if not profile:
        return 0.0
    total = 0.0
    for tool_expected, has_state, has_task in profile:
        # A zero-tool row earns the tool term by correctly abstaining
        # (compute_ast_f1([], []) == 1.0), which is what abstention_acc models.
        tool_term = tool_f1 if tool_expected else abstention_acc
        num, den = W_TOOL * tool_term, W_TOOL
        if has_state:
            num += W_STATE * state_acc
            den += W_STATE
        if has_task:
            num += W_TASK * task_completion
            den += W_TASK
        total += num / den if den else 0.0
    return total / len(profile)


def required_tool_f1(
    profile: list[tuple[bool, bool, bool]],
    bar: float,
    state_acc: float,
    abstention_acc: float,
    task_completion: float,
) -> float | None:
    """Invert ``expected_perturn_score`` for tool-F1. Pure.

    The score is affine in ``tool_f1``, so two evaluations determine the line.
    Returns ``None`` when the bar is unreachable even at tool_f1 = 1.0 (i.e. the
    other components alone cannot carry it).
    """
    lo = expected_perturn_score(profile, state_acc, 0.0, abstention_acc, task_completion)
    hi = expected_perturn_score(profile, state_acc, 1.0, abstention_acc, task_completion)
    if hi < bar:
        return None
    if hi == lo:
        return 0.0 if lo >= bar else None
    return max(0.0, (bar - lo) / (hi - lo))


def derive_bars(
    profile: list[tuple[bool, bool, bool]],
    abstention_sweep: tuple[float, ...] = ABSTENTION_SWEEP,
) -> dict[str, Any]:
    """Compute component-equivalent and relaxation-matched bars across the sweep. Pure."""
    old_at_targets = old_composite(
        TARGET_STATE_ACC, TARGET_TOOL_F1, TARGET_TASK_COMPLETION
    )
    # The original bar (0.75) sat below the composite its own component targets
    # imply (0.82) — a deliberate relaxation. Carry the same ratio across.
    relaxation = STATED_OLD_BAR / old_at_targets

    rows = []
    for abstention in abstention_sweep:
        component_equiv = expected_perturn_score(
            profile,
            TARGET_STATE_ACC,
            TARGET_TOOL_F1,
            abstention,
            TARGET_TASK_COMPLETION,
        )
        rows.append(
            {
                "abstention_acc": abstention,
                "component_equivalent_bar": component_equiv,
                "relaxation_matched_bar": component_equiv * relaxation,
            }
        )

    return {
        "old_composite_at_component_targets": old_at_targets,
        "stated_old_bar": STATED_OLD_BAR,
        "relaxation_ratio": relaxation,
        "sweep": rows,
        "recommended_abstention_acc": RECOMMENDED_ABSTENTION,
    }


def build_profile(data_dir: Path, split: str) -> list[tuple[bool, bool, bool]]:
    """Slice the split with the production slicer and extract term applicability."""
    from llm_workflow_agents.training.grpo import _load_grpo_jsonl

    ds = _load_grpo_jsonl(Path(data_dir), split=split)
    profile: list[tuple[bool, bool, bool]] = []
    for row in ds:
        gt = json.loads(row["ground_truth"])
        seq = gt.get("state_sequence") or []
        trans = [
            (s.get("from", ""), s.get("to", ""))
            for s in seq
            if isinstance(s, dict)
        ]
        terminal = gt.get("terminal_state") or ""
        to_state = trans[-1][1] if trans else ""
        profile.append(
            (
                bool(gt.get("tool_calls")),
                bool(trans),
                bool(terminal and to_state == terminal),
            )
        )
    return profile


def summarize_profile(profile: list[tuple[bool, bool, bool]]) -> dict[str, Any]:
    """Descriptive stats for the row population. Pure."""
    n = len(profile)
    if not n:
        return {"n_rows": 0}
    return {
        "n_rows": n,
        "frac_tool_expected": sum(1 for p in profile if p[0]) / n,
        "frac_zero_tool": sum(1 for p in profile if not p[0]) / n,
        "frac_state_term_applies": sum(1 for p in profile if p[1]) / n,
        "frac_task_term_applies": sum(1 for p in profile if p[2]) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--measured",
        type=float,
        default=0.7167,
        help="Measured per-turn composite to judge against (default: §6.1's 0.7167).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    profile = build_profile(args.data_dir, args.split)
    prof_summary = summarize_profile(profile)
    bars = derive_bars(profile)

    rec = next(
        r for r in bars["sweep"] if r["abstention_acc"] == RECOMMENDED_ABSTENTION
    )
    recommended_bar = rec["relaxation_matched_bar"]
    req_t = required_tool_f1(
        profile,
        recommended_bar,
        TARGET_STATE_ACC,
        RECOMMENDED_ABSTENTION,
        TARGET_TASK_COMPLETION,
    )

    results = {
        "config": {
            "data": f"{args.data_dir}/{args.split}.jsonl",
            "component_targets": {
                "state_transition_accuracy": TARGET_STATE_ACC,
                "tool_call_f1": TARGET_TOOL_F1,
                "task_completion_rate": TARGET_TASK_COMPLETION,
            },
            "measured_perturn_composite": args.measured,
        },
        "row_profile": prof_summary,
        "derivation": bars,
        "recommended": {
            "abstention_acc": RECOMMENDED_ABSTENTION,
            "bar": recommended_bar,
            "verdict_vs_measured": "PASS" if args.measured >= recommended_bar else "FAIL",
            "shortfall": recommended_bar - args.measured,
            "required_tool_f1_on_tool_expected_rows": req_t,
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(
        f"\n=== target-bar re-derivation — {args.data_dir}/{args.split} ===\n"
        f"  rows                     = {prof_summary['n_rows']}\n"
        f"  tool-expected / zero-tool= {prof_summary['frac_tool_expected']:.1%}"
        f" / {prof_summary['frac_zero_tool']:.1%}\n"
        f"  state term applies       = {prof_summary['frac_state_term_applies']:.1%}\n"
        f"  task  term applies       = {prof_summary['frac_task_term_applies']:.1%}\n"
        f"\n  old composite at component targets = "
        f"{bars['old_composite_at_component_targets']:.4f}"
        f"  (stated bar {STATED_OLD_BAR} -> relaxation "
        f"{bars['relaxation_ratio']:.4f})\n"
        f"\n  per-turn bar by abstention assumption:\n"
        + "".join(
            f"    abstention {r['abstention_acc']:.2f}:  "
            f"component-equiv {r['component_equivalent_bar']:.4f}   "
            f"relaxation-matched {r['relaxation_matched_bar']:.4f}\n"
            for r in bars["sweep"]
        )
        + f"\n  RECOMMENDED BAR = {recommended_bar:.4f}"
        f"  (abstention {RECOMMENDED_ABSTENTION}, relaxation-matched)\n"
        f"  measured {args.measured:.4f} -> "
        f"{results['recommended']['verdict_vs_measured']}"
        f"  (shortfall {results['recommended']['shortfall']:+.4f})\n"
        f"  tool-F1 needed on tool-expected rows to clear it = "
        + (f"{req_t:.4f}\n" if req_t is not None else "UNREACHABLE\n")
        + (f"[done] wrote {args.output}\n" if args.output else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
