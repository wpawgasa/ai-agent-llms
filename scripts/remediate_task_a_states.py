#!/usr/bin/env python3
"""Deterministic CLI for the tool-call stay-convention remediation.

No network, no LLM calls -- authoring the ~17% of conversations that need
new prose is scripts/build_remediation_ledger.py's job (a separate, costly,
explicitly-invoked step). This script only classifies (triage), applies
deterministic + ledger-supplied repairs (apply), re-checks a directory
(verify), and reports before/after deltas (diff).

Usage:
    python scripts/remediate_task_a_states.py triage --input-dir DIR --report PATH
    python scripts/remediate_task_a_states.py apply  --input-dir DIR --output-dir DIR
                                                       [--ledger-dir DIR] [--on-unrepairable drop|keep]
    python scripts/remediate_task_a_states.py verify --input-dir DIR [--strict]
    python scripts/remediate_task_a_states.py diff   --before DIR --after DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_workflow_agents.data.state_convention_repair import (  # noqa: E402
    apply_plan, plan_repair, verify_repaired,
)
from llm_workflow_agents.data._workflow_script import infer_state_tools_from_messages  # noqa: E402


def _iter_records(input_dir: Path):
    for path in sorted(glob.glob(str(input_dir / "*.jsonl"))):
        stem = Path(path).stem
        with open(path) as fh:
            for line_index, line in enumerate(fh):
                if not line.strip():
                    continue
                yield stem, line_index, path, json.loads(line)


_CONTEXT_RADIUS = 6


def _context_window(messages: list[dict], position_after_msg_index: int) -> list[dict]:
    """The messages surrounding an insert point, for the authoring agent.

    Returns up to ``_CONTEXT_RADIUS`` messages either side of the insert
    point, each as ``{"index", "role", "content"}``. ``annotations`` are
    dropped -- the agent authors prose and must never see (or be tempted to
    copy) the structured state/tool metadata. The system message (index 0)
    is always excluded: it is 5-7 KB of workflow contract per row, which
    would dominate the batch prompt for no authoring benefit.
    """
    lo = max(1, position_after_msg_index - _CONTEXT_RADIUS + 1)
    hi = min(len(messages), position_after_msg_index + _CONTEXT_RADIUS + 1)
    return [
        {"index": i, "role": messages[i].get("role"), "content": messages[i].get("content") or ""}
        for i in range(lo, hi)
    ]


def cmd_triage(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    totals = Counter()
    by_move = Counter()
    by_level = Counter()
    by_language = Counter()
    records_out = []
    n = 0
    for stem, line_index, _path, rec in _iter_records(input_dir):
        if args.limit and n >= args.limit:
            break
        n += 1
        plan = plan_repair(rec)
        totals["rows"] += 1
        by_move[plan.move] += 1
        by_level[rec.get("complexity_level", "?")] += 1
        by_language[rec.get("language", "?")] += 1
        inserts = [
            {
                "insert_id": f"{stem}:{line_index}:{i}",
                "position_after_msg_index": ins.position_after_msg_index,
                "role": ins.role,
                "required_marker": ins.required_marker,
                # The agent (Task 12) builds its prompt ONLY from this report --
                # it never re-reads the corpus -- so the surrounding messages it
                # needs to author in-register prose must be embedded here.
                "context_window": _context_window(rec["messages"], ins.position_after_msg_index),
            }
            for i, ins in enumerate(plan.inserts)
        ]
        records_out.append({
            "key": [stem, line_index],
            "conversation_id": rec.get("conversation_id", ""),
            "complexity_level": rec.get("complexity_level", ""),
            "language": rec.get("language", ""),
            "domain": rec.get("domain", ""),
            "move": plan.move,
            "drift_turns": plan.drift_turns,
            "inserts": inserts,
            "infeasible_reason": plan.infeasible_reason,
        })
    report = {
        "input_dir": str(input_dir),
        "totals": dict(totals),
        "by_move": dict(by_move),
        "by_level": dict(by_level),
        "by_language": dict(by_language),
        "records": records_out,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Triage: {totals['rows']} rows -> {dict(by_move)}")
    return 0


def _load_ledger(ledger_dir: Path | None) -> dict[str, dict]:
    if ledger_dir is None:
        return {}
    accepted = ledger_dir / "accepted.jsonl"
    if not accepted.exists():
        return {}
    entries = {}
    for line in accepted.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            entries[entry["insert_id"]] = entry
    return entries


def cmd_apply(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_entries = _load_ledger(Path(args.ledger_dir) if args.ledger_dir else None)

    kept = dropped = 0
    drop_reasons = Counter()
    by_file: dict[str, list[dict]] = {}
    for stem, line_index, path, rec in _iter_records(input_dir):
        plan = plan_repair(rec)
        # assign insert_ids matching the triage convention so a ledger built
        # from a `triage` report lines up with this pass
        for i, ins in enumerate(plan.inserts):
            ins.insert_id = f"{stem}:{line_index}:{i}"
        before_tools = infer_state_tools_from_messages(rec["messages"])
        repaired = apply_plan(rec, plan, ledger_entries=ledger_entries or None)
        if repaired is None:
            dropped += 1
            drop_reasons[plan.infeasible_reason or f"needs-ledger:{plan.move}"] += 1
            continue
        after_tools = infer_state_tools_from_messages(repaired["messages"])
        if before_tools != after_tools:
            dropped += 1
            drop_reasons["tool-from-state-changed"] += 1
            continue
        violations = verify_repaired(repaired)
        if violations:
            dropped += 1
            drop_reasons["post-gate-failed"] += 1
            continue
        kept += 1
        by_file.setdefault(Path(path).stem, []).append(repaired)

    for stem, records in by_file.items():
        out_path = output_dir / f"{stem}.jsonl"
        with open(out_path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Apply: kept {kept}, dropped {dropped} ({dict(drop_reasons)})")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    total_violations = 0
    for _stem, _idx, _path, rec in _iter_records(input_dir):
        violations = verify_repaired(rec)
        if violations:
            total_violations += len(violations)
            print(f"{rec.get('conversation_id')}: {violations}")
    print(f"Total violations: {total_violations}")
    if args.strict and total_violations:
        return 1
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    before = Counter(plan_repair(rec).move for *_ignore, rec in _iter_records(Path(args.before)))
    after = Counter(plan_repair(rec).move for *_ignore, rec in _iter_records(Path(args.after)))
    print("before:", dict(before))
    print("after: ", dict(after))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_triage = sub.add_parser("triage")
    p_triage.add_argument("--input-dir", required=True)
    p_triage.add_argument("--report", required=True)
    p_triage.add_argument("--limit", type=int, default=None)
    p_triage.set_defaults(func=cmd_triage)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--input-dir", required=True)
    p_apply.add_argument("--output-dir", required=True)
    p_apply.add_argument("--ledger-dir", default=None)
    p_apply.add_argument("--on-unrepairable", choices=["drop", "keep"], default="drop")
    p_apply.set_defaults(func=cmd_apply)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--input-dir", required=True)
    p_verify.add_argument("--strict", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_diff = sub.add_parser("diff")
    p_diff.add_argument("--before", required=True)
    p_diff.add_argument("--after", required=True)
    p_diff.set_defaults(func=cmd_diff)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
