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
                                                       [--ledger-dir DIR] [--rebuild-prompts]
                                                       [--on-unrepairable drop] [--limit N]
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
from llm_workflow_agents.data.system_prompt import (  # noqa: E402
    _STAY_RULE_ENABLED, build_enriched_system_prompt,
)


def _iter_records(input_dir: Path):
    for path in sorted(glob.glob(str(input_dir / "*.jsonl"))):
        stem = Path(path).stem
        # utf-8-sig, not the locale default: the corpus is Thai, so under
        # LC_ALL=C an unqualified open() dies with a bare UnicodeDecodeError.
        # The -sig codec also eats a leading BOM, which would otherwise make
        # line 1 look like invalid JSON.
        with open(path, encoding="utf-8-sig") as fh:
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
    # ensure_ascii=False keeps Thai readable in the report, which makes the
    # encoding explicit rather than locale-dependent: under LC_ALL=C an
    # unqualified write_text() would raise UnicodeEncodeError here.
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Triage: {totals['rows']} rows -> {dict(by_move)}")
    return 0


def _load_ledger(ledger_dir: Path | None) -> dict[str, dict]:
    """Read `accepted.jsonl` into {insert_id: entry}.

    Fails loudly and specifically on a bad line rather than with a bare
    `JSONDecodeError` traceback. The ledger is append-only *and* deliberately
    hand-editable -- the design markets it as PR-reviewable -- so a malformed
    line is a plausible operator error, not an impossible one. Skipping the line
    would be worse than dying: a silently dropped entry silently drops a
    conversation from the corpus, which is exactly the failure this whole
    pipeline is trying to make visible.
    """
    if ledger_dir is None:
        return {}
    accepted = ledger_dir / "accepted.jsonl"
    if not accepted.exists():
        return {}
    entries = {}
    # utf-8-sig: authored ledger prose is mostly Thai (LC_ALL=C would otherwise
    # raise a bare UnicodeDecodeError here), and a BOM-prefixed ledger should
    # load rather than be reported as invalid JSON on line 1.
    ledger_text = accepted.read_text(encoding="utf-8-sig")
    for number, line in enumerate(ledger_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"error: {accepted}:{number} is not valid JSON ({exc.msg}). "
                "Fix or remove that line before applying -- refusing to splice a "
                "partially-readable ledger into the corpus."
            ) from exc
        if not isinstance(entry, dict) or not isinstance(entry.get("insert_id"), str):
            raise SystemExit(
                f"error: {accepted}:{number} is not a ledger entry with a string "
                "'insert_id'."
            )
        entries[entry["insert_id"]] = entry
    return entries


def _rebuild_system_prompt(record: dict) -> bool:
    """Regenerate ``messages[0]``'s content from current prompt code (D5).

    A v1 row's system message is frozen at generation time: it carries the v1
    FORMAT_RULES, whose rule-2 worked example shows an ADVANCING transition on a
    tool-call turn -- the exact defect this remediation removes from the
    trajectories -- plus the vague "attempt recovery before escalating"
    tool-error rule with no retry budget. Repairing only the assistant turns
    would ship a corpus that DEMONSTRATES the stay convention while its own
    contract still CONTRADICTS it.

    Passing the whole *record* as the sample matters twice over:
      * ``retry_budget_for_sample`` keys off ``record["complexity_level"]``, so
        an L5 row states 3 attempts and an L1 row states no-retry, matching the
        conversations Task 9's generator actually bakes.
      * ``build_enriched_system_prompt`` regenerates the workflow script from
        ``record["messages"]``. Called AFTER the repair, those are the REPAIRED
        messages, so the script's per-state tool listing describes the corrected
        trajectory rather than the pre-repair one.

    Mutates *record* in place and touches nothing but the system content.
    Returns True if a rebuild happened. A record whose ``messages[0]`` is not a
    ``system`` message is left alone rather than having one synthesised: the
    remediation must not change conversation shape, which ``verify_repaired``'s
    initiator gate scores.
    """
    messages = record.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        return False
    original = messages[0].get("content") or ""
    messages[0]["content"] = build_enriched_system_prompt(
        record, original, force_rebuild=True
    )
    return True


def cmd_apply(args: argparse.Namespace) -> int:
    # Guard BEFORE any work (including creating output_dir): reuse
    # system_prompt._STAY_RULE_ENABLED -- the same resolution
    # system_prompt.build_enriched_system_prompt itself uses -- so this check
    # can never disagree with what --rebuild-prompts would actually render.
    # It is False only when TASK_A_STAY_RULE is the exact string "0"; any other
    # value (unset, "1", "0 ", "00", "false", ...) resolves True there, so this
    # guard stays silent for those too -- it only blocks the one combination
    # that would silently bake the frozen v1 prompt into a v2 corpus.
    if args.rebuild_prompts and not _STAY_RULE_ENABLED:
        print(
            "error: --rebuild-prompts refuses to run with TASK_A_STAY_RULE=0.\n"
            "  TASK_A_STAY_RULE=0 selects the frozen v1 system prompt (wrong "
            "rule-2 worked example, no stay rule, vague no-budget tool-error "
            "rule). Rebuilding system prompts under it would bake that v1 "
            "prompt into what --rebuild-prompts exists to make a v2 corpus -- "
            "the output would state a contract its own repaired trajectories "
            "contradict, and neither `verify --strict` nor the quality "
            "profiler checks system-prompt content, so nothing downstream "
            "would catch it.\n"
            "  To proceed: unset TASK_A_STAY_RULE (or set it to \"1\") and "
            "re-run, or drop --rebuild-prompts if a v1-prompt corpus is "
            "genuinely what you want.",
            file=sys.stderr,
        )
        return 2
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_entries = _load_ledger(Path(args.ledger_dir) if args.ledger_dir else None)

    kept = dropped = 0
    rebuilt = 0
    drop_reasons = Counter()
    by_file: dict[str, list[dict]] = {}
    seen = 0
    for stem, line_index, path, rec in _iter_records(input_dir):
        if args.limit and seen >= args.limit:
            break
        seen += 1
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
        # Deliberately LAST, after every gate. A dropped row never pays the
        # rebuild cost, and each gate above sees the same message structure it
        # would see with the flag off -- none of them read system CONTENT (the
        # finders scan assistant turns; find_shape_violations filters the
        # system role out entirely), so the ordering is about cost and
        # reviewability, not correctness.
        if args.rebuild_prompts and _rebuild_system_prompt(repaired):
            rebuilt += 1
        kept += 1
        by_file.setdefault(Path(path).stem, []).append(repaired)

    for stem, records in by_file.items():
        out_path = output_dir / f"{stem}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    suffix = f", system prompts rebuilt {rebuilt}" if args.rebuild_prompts else ""
    print(f"Apply: kept {kept}, dropped {dropped} ({dict(drop_reasons)}){suffix}")
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
    p_apply.add_argument(
        "--rebuild-prompts",
        action="store_true",
        help=(
            "Regenerate each retained row's system message from current prompt "
            "code (design decision D5), so a v2 row STATES the tool-stay "
            "convention its trajectory now demonstrates: the corrected rule-2 "
            "worked example, the stay rule, and the retry budget for the row's "
            "own complexity_level. Off by default -- omitting it leaves system "
            "messages byte-identical to the input. Refuses to run (exit 2, no "
            "output written) if TASK_A_STAY_RULE=0 is set, since that would "
            "bake the frozen v1 prompt into a v2 corpus; unset it or drop this "
            "flag."
        ),
    )
    # Only 'drop' is offered. 'keep' used to be accepted and then silently
    # ignored (every unrepairable row was dropped anyway), which is worse than
    # not offering it: an operator could ask for retention and get none, with no
    # error. Implementing it properly is also not wanted -- a kept row is by
    # definition one that still violates the convention, so it would poison the
    # v2 corpus this stage exists to produce and would fail the `verify
    # --strict` gate that runs right after. The flag survives as a single-choice
    # argument so the documented DVC/runbook command line stays valid and the
    # policy is explicit at the call site rather than implicit.
    p_apply.add_argument(
        "--on-unrepairable",
        choices=["drop"],
        default="drop",
        help=(
            "What to do with a row that cannot be brought onto the convention. "
            "Only 'drop' is supported: retaining a violating row would defeat "
            "the purpose of the remediation and fail the downstream "
            "`verify --strict` gate."
        ),
    )
    p_apply.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N input rows (smoke runs); default: all.",
    )
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
