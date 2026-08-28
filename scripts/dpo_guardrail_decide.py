#!/usr/bin/env python3
"""Decide whether a chunked DPO run should continue, from outside the trainer.

WHY THIS EXISTS: the R5 held-out guardrail cannot run inside the DPO training
process on this model. `load_in_4bit` reaches only 0.77 GiB of Gemma-4-26B-A4B
— the MoE experts are fused 3-D `nn.Parameter` tensors that bitsandbytes cannot
swap — so training already holds ~46 GiB of weights and a second model copy does
not fit on one GPU. See CLAUDE.md R19 and
docs/dpo_memory_ceiling_investigation.md §8.

`scripts/run_phase2_dpo.sh --chunk-steps N` therefore trains in chunks. Each
chunk is its own process, so the GPU empties between them, and the held-out
score is produced by a separate `scripts/heldout_composite_audit.py` process.
This helper reads what those two processes left on disk and answers one
question: continue, or stop?

It owns no policy. `training.reward_utils.is_reward_hacking` stays the single
stop rule, shared with the in-process callback in `training/dpo.py`, so the two
paths cannot drift on what "reward hacking" means.

Exit codes:
    0   continue training
    10  stop — the training metric is rising while held-out quality falls
    2   the inputs could not be read (argparse, or a missing trainer_state)

Exit 2 matters: a guardrail that cannot read its input must fail loudly rather
than report "all clear". R18(c) records the R5 guardrail sitting silently
inactive across every Gemma-4 GRPO run because a failure was swallowed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llm_workflow_agents.training.reward_utils import (  # noqa: E402
    is_reward_hacking,
)


def read_training_metric(trainer_state: Path) -> list[float]:
    """Per-log-step training metric, in order, from HF's ``trainer_state.json``.

    Mirrors the in-process callback's choice exactly
    (``training/dpo.py``: ``logs.get("rewards/accuracies", logs.get("loss"))``)
    so a chunked run and a straight-through run judge the same signal. Log lines
    carrying neither key — checkpoint saves, eval summaries — contribute
    nothing rather than a zero, which would fake a downward trend.
    """
    if not trainer_state.is_file():
        raise SystemExit(
            f"dpo_guardrail_decide: no trainer state at {trainer_state}. "
            "The chunk did not save a checkpoint, or the path is wrong. "
            "Refusing to report 'continue' on unread input."
        )
    history = json.loads(trainer_state.read_text()).get("log_history", [])
    out: list[float] = []
    for entry in history:
        value = entry.get("rewards/accuracies", entry.get("loss"))
        if value is not None:
            out.append(float(value))
    return out


def read_heldout_scores(audit_dir: Path) -> list[float]:
    """Held-out composites from ``step-N.json`` audit files, in step order.

    Sorted by the integer N, never by filename: a lexical sort puts step-100
    before step-20 and inverts the very trend this guardrail tests for.
    """
    if not audit_dir.is_dir():
        return []
    files = sorted(
        audit_dir.glob("step-*.json"),
        key=lambda p: int(p.stem.rsplit("-", 1)[-1]),
    )
    return [
        float(json.loads(p.read_text())["summary"]["mean_composite"])
        for p in files
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trainer-state",
        type=Path,
        required=True,
        help="path to the newest checkpoint's trainer_state.json",
    )
    ap.add_argument(
        "--audit-dir",
        type=Path,
        required=True,
        help="directory holding this run's step-N.json audit outputs",
    )
    args = ap.parse_args(argv)

    metric = read_training_metric(args.trainer_state)
    heldout = read_heldout_scores(args.audit_dir)

    hacking = is_reward_hacking(metric, heldout)
    print(
        f"dpo_guardrail_decide: n_metric={len(metric)} n_heldout={len(heldout)} "
        f"heldout_last={heldout[-1] if heldout else 'n/a'} "
        f"verdict={'STOP' if hacking else 'continue'}"
    )
    return 10 if hacking else 0


if __name__ == "__main__":
    sys.exit(main())
