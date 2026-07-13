#!/usr/bin/env python3
"""Row-level audit companion to scripts/heldout_composite_check.py.

The ceiling check reports only the aggregate composite + per-row scalar. When
it FAILs (mean < 0.75) the action is to audit reward/GT on the low rows before
assuming a hard policy ceiling. This script re-runs the identical greedy pass
(same sampler, seed, checkpoint -> identical completions) but persists, per row:

  - the model completion text,
  - the GT state_sequence / tool_calls / terminal_state,
  - the THREE strict composite components separately
    (state_acc, tool_f1, task) so we can see WHICH term is dragging the
    composite down, and whether it's a genuine policy miss or a reward/GT
    artifact (e.g. GT tool_calls with placeholder args the model can't match).

Component math mirrors grpo._heldout_composite_score exactly:
  composite = 0.4*state_acc + 0.4*tool_f1 + 0.2*task

Usage:
    .venv-train/bin/python scripts/heldout_composite_audit.py \
        --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000 \
        --data-dir data/output/grpo/task_a --split validation \
        --n-prompts 150 --output runs/preflight/heldout_composite_audit_ckpt1000.json
"""

from __future__ import annotations

import argparse
import json
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


def _components(comp: str, gt: dict[str, Any]) -> dict[str, Any]:
    """Strict composite components for one row — same scorers as
    grpo._heldout_composite_score, broken out instead of summed."""
    from llm_workflow_agents.training.reward_utils import (
        extract_state_annotations,
        extract_tool_calls,
        reached_terminal,
        state_sequence_match,
        tool_call_f1,
    )

    gt = gt or {}
    gt_seq = gt.get("state_sequence") or []
    gt_trans = [
        (s.get("from", ""), s.get("to", ""))
        if isinstance(s, dict)
        else tuple(s)
        if isinstance(s, (list, tuple)) and len(s) == 2
        else ("", "")
        for s in gt_seq
    ]
    pred_trans = extract_state_annotations(comp)
    if gt_trans:
        state_acc = state_sequence_match(pred_trans, gt_trans)
    else:
        state_acc = 1.0 if not pred_trans else 0.0

    gt_tools = gt.get("tool_calls") or []
    pred_tools = extract_tool_calls(comp)
    tool_f1 = tool_call_f1(pred_tools, gt_tools)

    terminal = gt.get("terminal_state") or ""
    task = 1.0 if terminal and reached_terminal(comp, terminal) else 0.0

    composite = 0.4 * state_acc + 0.4 * tool_f1 + 0.2 * task
    return {
        "composite": composite,
        "state_acc": state_acc,
        "tool_f1": tool_f1,
        "task": task,
        "n_gt_trans": len(gt_trans),
        "n_pred_trans": len(pred_trans),
        "n_gt_tools": len(gt_tools),
        "n_pred_tools": len(pred_tools),
        "gt_state_sequence": gt_seq,
        "gt_tool_calls": gt_tools,
        "gt_terminal": terminal,
        "pred_trans": pred_trans,
        "pred_tools": pred_tools,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
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

    prompts = _sample_prompts(args.data_dir, args.split, args.n_prompts, args.seed)
    gts = [_decode_gt(p["ground_truth"]) for p in prompts]
    print(f"[data] sampled {len(prompts)} prompts", flush=True)

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

    rows: list[dict[str, Any]] = []
    for i, (comp, gt) in enumerate(zip(completions, gts, strict=True)):
        comps = _components(comp, gt)
        rows.append({"row_index": i, "completion": comp, **comps})

    rows_sorted = sorted(rows, key=lambda r: r["composite"])
    n = len(rows)

    def _mean(key: str) -> float:
        return sum(r[key] for r in rows) / n if n else 0.0

    summary = {
        "n_rows": n,
        "mean_composite": _mean("composite"),
        "mean_state_acc": _mean("state_acc"),
        "mean_tool_f1": _mean("tool_f1"),
        "mean_task": _mean("task"),
        "frac_state_acc_zero": sum(1 for r in rows if r["state_acc"] == 0.0) / n,
        "frac_tool_f1_zero": sum(1 for r in rows if r["tool_f1"] == 0.0) / n,
        "frac_task_zero": sum(1 for r in rows if r["task"] == 0.0) / n,
        "frac_gt_has_no_tools": sum(1 for r in rows if r["n_gt_tools"] == 0) / n,
        "wall_time_s": round(time.time() - t0, 1),
    }

    args.output.write_text(
        json.dumps({"summary": summary, "rows": rows_sorted}, indent=2, ensure_ascii=False)
    )

    print(
        "\n=== held-out composite AUDIT — component means ===\n"
        f"  mean_composite = {summary['mean_composite']:.4f}\n"
        f"  mean_state_acc = {summary['mean_state_acc']:.4f}  (weight 0.4)\n"
        f"  mean_tool_f1   = {summary['mean_tool_f1']:.4f}  (weight 0.4)\n"
        f"  mean_task      = {summary['mean_task']:.4f}  (weight 0.2)\n"
        f"  frac state_acc==0 : {summary['frac_state_acc_zero']:.3f}\n"
        f"  frac tool_f1==0   : {summary['frac_tool_f1_zero']:.3f}\n"
        f"  frac task==0      : {summary['frac_task_zero']:.3f}\n"
        f"  frac GT has 0 tools: {summary['frac_gt_has_no_tools']:.3f}\n"
        f"  wall_time_s = {summary['wall_time_s']}\n"
        f"[done] wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
