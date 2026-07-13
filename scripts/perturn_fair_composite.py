#!/usr/bin/env python3
"""Per-turn-fair variant of grpo._heldout_composite_score.

Motivation (see the ckpt-1000 audit, runs/preflight/heldout_composite_audit_*):
the shipped ``_heldout_composite_score`` is a WHOLE-CONVERSATION metric
(``0.4*state + 0.4*tool + 0.2*task``, target 0.75 from
eval/composite_score.compute_weighted_workflow_score) but the GRPO held-out
rows are SINGLE user->assistant turns. Two of the three terms are structurally
unearnable at that granularity, which depressed the as-scored mean to 0.48:

  * task_completion (0.2): "reached terminal state" can only be true on the
    ONE terminal turn of a conversation; on every intermediate turn it is 0
    by construction (and 21/150 rows carry an empty terminal, unearnable at all).
  * state on abstention turns (part of 0.4): when GT has NO transition this
    turn, the strict scorer gives 1.0 only if the model stays silent — but the
    SFT policy was trained to annotate every turn, so 40/41 such rows score 0.

This module keeps the SAME strict per-component scorers and the SAME relative
weights, but only INCLUDES a term when it is applicable to the turn, then
renormalizes over the included weights:

  * tool term (w=0.4): always included.
  * state term (w=0.4): included only when GT has >=1 transition this turn.
  * task term (w=0.2): included only when this IS the terminal turn
    (the GT transition's ``to`` equals a non-empty ``terminal_state``).

This is deliberately conservative: it never GIVES credit the model didn't
earn on an applicable term; it only stops charging the model on terms that
cannot be satisfied on that turn. Pure / CPU-only / unit-tested.
"""

from __future__ import annotations

from typing import Any


def _terminal_turn(gt_state_sequence: list, terminal: str) -> bool:
    """True iff this turn's GT transition lands on the conversation terminal."""
    if not terminal:
        return False
    if not gt_state_sequence:
        return False
    last = gt_state_sequence[-1]
    to = last.get("to") if isinstance(last, dict) else (
        last[1] if isinstance(last, (list, tuple)) and len(last) == 2 else ""
    )
    return to == terminal


def perturn_fair_composite_from_components(
    *,
    state_acc: float,
    tool_f1: float,
    task: float,
    n_gt_trans: int,
    gt_state_sequence: list,
    gt_terminal: str,
) -> tuple[float, dict[str, bool]]:
    """Corrected per-turn composite from already-computed strict components.

    Returns (score, applicability) where applicability records which terms were
    counted. Kept separate from generation so it can re-score saved audit rows.
    """
    W_STATE, W_TOOL, W_TASK = 0.4, 0.4, 0.2

    include_state = n_gt_trans > 0
    include_task = _terminal_turn(gt_state_sequence, gt_terminal)
    # tool term always applies.

    num = W_TOOL * tool_f1
    den = W_TOOL
    if include_state:
        num += W_STATE * state_acc
        den += W_STATE
    if include_task:
        num += W_TASK * task
        den += W_TASK

    score = num / den if den else 0.0
    return score, {
        "state": include_state,
        "tool": True,
        "task": include_task,
    }


def main() -> int:
    import argparse
    import json
    import statistics
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--audit-json",
        type=Path,
        default=Path("runs/preflight/heldout_composite_audit_ckpt1000.json"),
        help="Output of scripts/heldout_composite_audit.py (has per-row components).",
    )
    p.add_argument("--target", type=float, default=0.75)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    d = json.loads(args.audit_json.read_text())
    rows = d["rows"]
    n = len(rows)

    fair, asis = [], []
    per_row = []
    for r in rows:
        s, appl = perturn_fair_composite_from_components(
            state_acc=r["state_acc"],
            tool_f1=r["tool_f1"],
            task=r["task"],
            n_gt_trans=r["n_gt_trans"],
            gt_state_sequence=r.get("gt_state_sequence") or [],
            gt_terminal=r.get("gt_terminal") or "",
        )
        fair.append(s)
        asis.append(r["composite"])
        per_row.append({"row_index": r["row_index"], "fair": s, "asis": r["composite"], **appl})

    mean = statistics.fmean
    out = {
        "n_rows": n,
        "mean_asis": mean(asis),
        "mean_fair": mean(fair),
        "median_fair": statistics.median(fair),
        "frac_fair_below_target": sum(1 for s in fair if s < args.target) / n,
        "terms_included": {
            "state": sum(1 for r in per_row if r["state"]),
            "tool": n,
            "task": sum(1 for r in per_row if r["task"]),
        },
        "target": args.target,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": out, "per_row": per_row}, indent=2))

    print(
        "=== per-turn-fair re-score (from saved completions) ===\n"
        f"  n_rows            = {out['n_rows']}\n"
        f"  mean AS-SCORED    = {out['mean_asis']:.4f}   (whole-conv metric, target {args.target})\n"
        f"  mean PER-TURN-FAIR= {out['mean_fair']:.4f}\n"
        f"  median fair       = {out['median_fair']:.4f}\n"
        f"  frac fair < target= {out['frac_fair_below_target']:.3f}\n"
        f"  terms included: state={out['terms_included']['state']}/{n}  "
        f"task={out['terms_included']['task']}/{n}  (tool always)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
