#!/usr/bin/env python3
"""Repair the Task A SFT corpus — stage 1, the mechanical half (CLAUDE.md R28).

The benchmark repair ran in two stages. This is the same first stage pointed at
the training corpus, in the same plan/apply shape: ``plan`` decides every repair
once (seeded, reads the benchmark so a fresh identifier never takes a value the
benchmark uses) and writes a ledger; ``apply`` replays it with no randomness, so
a frozen DVC stage can rebuild the repaired corpus exactly.

    python scripts/repair_task_a_corpus.py plan \\
        --input-dir data/output/sft/task_a_splits \\
        --reference data/output/benchmark/task_a_v4 \\
        --reference data/output/benchmark/task_a_voice_v3 \\
        --ledger data/interim/task_a_corpus_repair_ledger/ledger.json

    python scripts/repair_task_a_corpus.py apply \\
        --stratum data/output/sft/task_a_splits data/output/sft/task_a_splits_v4 \\
        --ledger data/interim/task_a_corpus_repair_ledger/ledger.json

Three differences from ``repair_task_a_benchmark.py``:

**The forbidden set flips.** That script kept fresh benchmark identifiers away
from training values. Here the fresh *training* values must avoid the v4
benchmark, or the memorization channel v4 closed reopens from the other side.

**Orphan-result conversations are dropped.** A tool result answering a turn that
only announces the call teaches announce-but-don't-call, and no mechanical edit
repairs one. 41 of 9,932 rows is a cheaper loss than the behaviour.

**Invented facts are counted, not repaired.** The corpus holds 978 confident
ones across 903 conversations against the benchmark's 31, so authoring them is
stage 2 — paid for only if this stage moves the v4 numbers. They are reported
in the ledger summary and by ``apply``; residual ones do not fail the run, and
that tolerance is the ONLY check relaxed here.

What is NOT touched, deliberately: conversations where the agent asks the
customer for a value and the customer supplies it. Those are the counterexample
that keeps asking trained. Moving every unknowable value into session context
without them is exactly the uniform edit R15 shows becoming an unconditional
habit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repair_task_a_benchmark import (  # noqa: E402
    DEFAULT_SEED,
    Row,
    _body,
    _default_render,
    _original_system,
    _reference_identifiers,
    load_stratum,
    plan_rows,
    repair_sample,
    verify_rows,
    write_stratum,
)

from llm_workflow_agents.data.source_traceability import (  # noqa: E402
    find_orphan_tool_results,
    find_unsourced_argument_values,
    find_unsourced_facts,
)

LEDGER_VERSION = 1


def orphan_row_keys(rows: list[Row]) -> list[str]:
    """Keys of rows holding a tool result that answers no tool call."""
    return [key for key, sample in rows if find_orphan_tool_results(_body(sample))]


def residual_finding_counts(rows: list[Row]) -> dict[str, int]:
    """Confident findings left in ``rows`` — what stage 2 would still owe."""
    facts = args = affected = 0
    for _, sample in rows:
        prompt = _default_render(sample, _original_system(sample))
        context = sample.get("session_context")
        row_facts = [
            f for f in find_unsourced_facts(_body(sample), prompt, context)
            if f.confidence == "confident"
        ]
        row_args = [
            f for f in find_unsourced_argument_values(
                _body(sample), sample.get("tool_schemas"), prompt, context
            )
            if f.confidence == "confident"
        ]
        facts += len(row_facts)
        args += len(row_args)
        affected += bool(row_facts or row_args)
    return {"invented_facts": facts, "unsourced_arguments": args, "rows_with_findings": affected}


def plan_corpus(
    rows: list[Row],
    forbidden: set[str] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Decide every stage-1 repair for ``rows``; return the ledger.

    Orphan-result rows are recorded under ``dropped`` and are not planned, so
    ``apply`` removes them rather than repairing them.
    """
    dropped = {key: "orphan_tool_result" for key in orphan_row_keys(rows)}
    kept = [(key, sample) for key, sample in rows if key not in dropped]
    ledger = plan_rows(kept, forbidden=forbidden, seed=seed)
    ledger["version"] = LEDGER_VERSION
    ledger["dropped"] = dropped
    ledger["summary"]["rows_dropped"] = len(dropped)
    ledger["summary"]["rows_in"] = len(rows)
    return ledger


def apply_corpus(rows: list[Row], ledger: dict[str, Any]) -> tuple[list[Row], list[dict[str, Any]]]:
    """Replay ``ledger`` onto ``rows``; return the kept rows and their repairs."""
    dropped = ledger.get("dropped", {})
    kept = [(key, sample) for key, sample in rows if key not in dropped]
    repaired = [repair_sample(sample, ledger["rows"].get(key, {})) for key, sample in kept]
    return kept, repaired


def _cmd_plan(args: argparse.Namespace) -> int:
    rows: list[Row] = []
    for directory in args.input_dir:
        rows.extend(load_stratum(Path(directory)))
    forbidden = _reference_identifiers(args.reference)
    print(f"[plan] {len(rows)} rows, {len(forbidden)} identifiers forbidden from the benchmark")
    ledger = plan_corpus(rows, forbidden=forbidden, seed=args.seed)
    ledger["summary"]["residual_before"] = residual_finding_counts(rows)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ledger["summary"], indent=2))
    print(f"[plan] wrote {args.ledger}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    problems: list[str] = []
    for source, destination in args.stratum:
        rows = load_stratum(Path(source))
        kept, repaired = apply_corpus(rows, ledger)
        write_stratum(kept, repaired, Path(destination))
        print(f"[apply] {source} -> {destination}: {len(kept)} rows "
              f"({len(rows) - len(kept)} dropped)")
        # Invented facts are stage 2's job, so an argument whose value comes
        # from one is expected to survive; every other problem is fatal.
        problems.extend(verify_rows(kept, repaired, ledger))
    fatal = [p for p in problems if "unsourced argument" not in p]
    soft = [p for p in problems if "unsourced argument" in p]
    if soft:
        print(f"[apply] {len(soft)} row(s) keep a confident unsourced argument — "
              f"stage 2 (authored facts) owns these, e.g. {soft[0]}")
    if fatal:
        print(f"[apply] FAILED: {len(fatal)} problem(s)", file=sys.stderr)
        for problem in fatal[:20]:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("[apply] verified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--input-dir", action="append", required=True, type=Path)
    plan.add_argument("--reference", action="append", default=[], type=Path,
                      help="benchmark directory; its identifiers are never reused")
    plan.add_argument("--ledger", required=True, type=Path)
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan.set_defaults(func=_cmd_plan)

    apply_cmd = sub.add_parser("apply")
    apply_cmd.add_argument("--stratum", nargs=2, action="append", required=True, type=Path,
                           metavar=("INPUT_DIR", "OUTPUT_DIR"))
    apply_cmd.add_argument("--ledger", required=True, type=Path)
    apply_cmd.set_defaults(func=_cmd_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
