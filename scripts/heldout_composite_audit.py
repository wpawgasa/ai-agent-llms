#!/usr/bin/env python3
"""Row-level audit companion to scripts/heldout_composite_check.py.

The ceiling check reports only the aggregate composite + per-row scalar. When
it FAILs (mean < 0.80) the action is to audit reward/GT on the low rows before
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

Modalities are reported separately, never blended (spec section 5). The
aggregate ``mean_composite`` remains in the summary so stored audits stay
readable, but on a mixed sample it is flagged ``mixed_modality`` in the JSON
and labelled MIXED-MODALITY in the printout: it is not comparable to cell C2's
0.7595, which was measured on 206 text rows, and quoting it against the
pre-registered 0.75 bar moves that bar without a decision. Use
``--modality text`` / ``--modality voice`` for a clean single-modality sample.
``voice_format_compliance`` stays a guardrail beside the composite and is
never folded into it.

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


def _corpus_has_voice(data_dir: Path, split: str) -> bool:
    """True if the source split file contains at least one voice conversation.

    Used to catch the silent-failure mode where the corpus does carry voice
    conversations but none reached the audit sample — the same failure shape
    as R18(c)'s silently-inactive R5 guardrail, where "0" looked healthy.
    """
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        return False
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            conv = json.loads(line)
            if (conv.get("modality") or "text") == "voice":
                return True
    return False


def _voice_dropped_warning(
    data_dir: Path, split: str, voice_rows: list[dict[str, Any]]
) -> str | None:
    """Message to print when the corpus carries voice but the guardrail can't fire.

    Returns ``None`` when either the corpus has no voice conversations at all
    (nothing to warn about) or at least one voice row did reach the audit
    sample (the guardrail computes normally). Otherwise returns a loud
    warning: a voice-bearing corpus that yields zero voice rows in the audit
    would otherwise look identical to an all-text corpus — a silently
    inactive guardrail reporting a "healthy" absence, the exact failure shape
    R18(c) documents for the R5 reward-hacking guardrail.
    """
    if voice_rows or not _corpus_has_voice(data_dir, split):
        return None
    return (
        f"[warn] {data_dir}/{split}.jsonl contains at least one voice "
        "conversation, but zero voice rows reached this audit's sampled "
        "prompts — voice_format_compliance did NOT fire. This is the exact "
        "silent-failure shape R18(c) documents for the R5 guardrail: a "
        "missing signal that looks like a healthy 0. Check modality "
        "propagation through _load_grpo_jsonl / _sample_prompts and the "
        "sample size before trusting this run's absence of a voice score."
    )


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


def summarise_by_modality(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Component means per modality, keyed by modality name.

    Spec section 5: "Do not blend the two modalities into one score." A blended
    mean silently moves the pre-registered 0.75 bar and is not comparable to
    cell C2's 0.7595, which was measured on 206 text rows. The aggregate
    ``mean_composite`` stays in the summary for backward compatibility with
    every stored audit, but a mixed sample is labelled as mixed both in the
    JSON (``mixed_modality``) and in the printout, and these figures are
    emitted alongside it.

    A row with no ``modality`` counts as text — every conversation predating
    the field is a written one.
    """
    per_modality: dict[str, dict[str, Any]] = {}
    for modality in sorted({(r.get("modality") or "text") for r in rows}):
        subset = [r for r in rows if (r.get("modality") or "text") == modality]
        per_modality[modality] = {
            "n_rows": len(subset),
            "mean_composite": sum(r["composite"] for r in subset) / len(subset),
            "mean_state_acc": sum(r["state_acc"] for r in subset) / len(subset),
            "mean_tool_f1": sum(r["tool_f1"] for r in subset) / len(subset),
            "mean_task": sum(r["task"] for r in subset) / len(subset),
        }
    return per_modality


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
    parser.add_argument(
        "--modality",
        choices=("text", "voice", "all"),
        default="all",
        help=(
            "Audit one modality only (default: all). Mirrors the same flag on "
            "scripts/build_heldout_clean_set.py. 'all' takes the unchanged "
            "sampling path, so a stored audit stays reproducible; text/voice "
            "draw from the same seeded order and keep the first --n-prompts "
            "rows of that modality."
        ),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from preflight_entropy_diag import _decode_gt, _generate_for_checkpoint, _sample_prompts

    if args.modality == "all":
        prompts = _sample_prompts(args.data_dir, args.split, args.n_prompts, args.seed)
    else:
        # Draw from the whole split in the same seeded order, then keep the
        # first n of the requested modality. Sampling first and filtering
        # afterwards would silently return fewer rows than asked for.
        # Bound: _sample_prompts dedupes to one row per conversation, so the
        # split's conversation count is the most it can ever return. Asking
        # for exactly that avoids a spurious "only N available" warning.
        split_path = Path(args.data_dir) / f"{args.split}.jsonl"
        with open(split_path) as fh:
            n_conversations = sum(1 for line in fh if line.strip())
        pool = _sample_prompts(args.data_dir, args.split, n_conversations, args.seed)
        matching = [p for p in pool if (p.get("modality") or "text") == args.modality]
        prompts = matching[: args.n_prompts]
        print(
            f"[data] --modality {args.modality}: {len(matching)} of {len(pool)} "
            f"rows match; keeping {len(prompts)}",
            flush=True,
        )
        if not prompts:
            print(
                f"FAIL: no {args.modality} rows in {args.data_dir}/{args.split}.jsonl",
                file=sys.stderr,
            )
            return 2
        if len(prompts) < args.n_prompts:
            print(
                f"[warn] asked for {args.n_prompts} {args.modality} rows and found "
                f"{len(prompts)}; the means below rest on a small denominator",
                file=sys.stderr,
            )
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
    for i, (comp, gt, conv) in enumerate(zip(completions, gts, prompts, strict=True)):
        comps = _components(comp, gt)
        rows.append({
            "row_index": i,
            "completion": comp,
            "modality": (conv.get("modality") or "text"),
            **comps,
        })

    rows_sorted = sorted(rows, key=lambda r: r["composite"])
    n = len(rows)

    def _mean(key: str) -> float:
        return sum(r[key] for r in rows) / n if n else 0.0

    per_modality = summarise_by_modality(rows)

    summary = {
        "n_rows": n,
        "requested_modality": args.modality,
        "modalities": per_modality,
        "mixed_modality": len(per_modality) > 1,
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

    # A guardrail, never a composite term. Adding a term would change what
    # cell C2's 0.7595 means.
    from llm_workflow_agents.data.voice_convention import find_voice_violations

    voice_rows = [r for r in rows if r.get("modality") == "voice"]
    if voice_rows:
        clean = sum(
            1 for r in voice_rows
            if not find_voice_violations(
                [{"role": "assistant", "content": r.get("completion", "")}], "voice"
            )
        )
        summary["voice_format_compliance"] = clean / len(voice_rows)
        summary["voice_rows"] = len(voice_rows)
    else:
        warning = _voice_dropped_warning(args.data_dir, args.split, voice_rows)
        if warning:
            print(warning, file=sys.stderr)

    args.output.write_text(
        json.dumps({"summary": summary, "rows": rows_sorted}, indent=2, ensure_ascii=False)
    )

    if summary["mixed_modality"]:
        mix = ", ".join(
            "{} ({})".format(k, v["n_rows"]) for k, v in per_modality.items()
        )
        per_modality_block = (
            "\n  !! MIXED-MODALITY SAMPLE — mean_composite below blends "
            f"{mix}.\n"
            "     It is NOT comparable to cell C2's 0.7595, which was measured on\n"
            "     206 text rows, and it moves the pre-registered 0.75 bar without a\n"
            "     decision. Report the per-modality figures instead (spec section 5);\n"
            "     re-run with --modality text / --modality voice for a clean sample.\n"
        )
    else:
        per_modality_block = ""
    for modality, block in per_modality.items():
        per_modality_block += (
            f"  [{modality}] n={block['n_rows']}  composite={block['mean_composite']:.4f}  "
            f"state_acc={block['mean_state_acc']:.4f}  "
            f"tool_f1={block['mean_tool_f1']:.4f}  task={block['mean_task']:.4f}\n"
        )

    print(
        "\n=== held-out composite AUDIT — component means ===\n"
        f"{per_modality_block}"
        f"  mean_composite = {summary['mean_composite']:.4f}"
        f"{'  (MIXED-MODALITY)' if summary['mixed_modality'] else ''}\n"
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
