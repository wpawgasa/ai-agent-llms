#!/usr/bin/env python3
"""Why did this run's tool calls fail? Bucket every one, from a stored log.

No GPU: the benchmark logs keep every ``model_response`` event, which is the
whole prediction side (same source `recompute_chain_propagation.py` reads).

    python scripts/triage_tool_call_failures.py \
        results/exp_a/<run>_v4_auto.log \
        --data data/output/benchmark/task_a_v4 \
        --data data/output/benchmark/task_a_voice_v3 \
        --out runs/audit/tool_failures_<run>.json

Tool F1 is the only lever that can reach a 0.9 composite — with perfect state
AND perfect completion the score caps at 0.878 — and tool F1 is gated by
argument exact match, not tool selection. This says which kind of argument
failure to attack: a comparator change, training data, or benchmark repair
(CLAUDE.md R30 on the cost of guessing).

The context a value is checked against is what the model could actually have
read at that point: the rendered system prompt (so session_context counts),
every user and tool message before the call, and the model's own earlier
replies in that conversation.

CAVEAT, printed with the output: turns the harness never asked the model for —
an outbound opener, a segment after a withheld tool result — emit no
``model_response`` event, so their gold calls appear here as ``no_call``. The
live benchmark treats openers as unscored (R29), so read ``no_call`` as an
upper bound on the model's own omissions.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
from llm_workflow_agents.eval.tool_call_f1 import parse_tool_calls
from llm_workflow_agents.eval.tool_call_failures import (
    align_calls,
    classify_call,
    refine_value_mismatch,
)

SAMPLE_RE = re.compile(r"evaluating_sample\s+conversation_id=(\S+)\s+idx=(\d+)")
RESPONSE_RE = re.compile(r"model_response\s+content=('.*?'|\".*?\")\s+latency_ms=", re.DOTALL)


def parse_log(path: Path) -> dict[int, list[str]]:
    """Map 1-based sample index -> the model's replies, in order."""
    text = path.read_text(errors="replace")
    events: list[tuple[int, tuple[str, Any]]] = []
    for match in SAMPLE_RE.finditer(text):
        events.append((match.start(), ("sample", int(match.group(2)))))
    for match in RESPONSE_RE.finditer(text):
        raw = match.group(1)
        try:
            content = json.loads(raw) if raw.startswith('"') else eval(raw, {"__builtins__": {}})
        except Exception:
            content = raw.strip("'\"")
        events.append((match.start(), ("response", content)))
    events.sort(key=lambda item: item[0])

    by_sample: dict[int, list[str]] = {}
    current: int | None = None
    for _, (kind, value) in events:
        if kind == "sample":
            current = int(value)
            by_sample.setdefault(current, [])
        elif current is not None and isinstance(value, str):
            by_sample[current].append(value)
    return by_sample


def load_samples(data_dirs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in sorted(dict.fromkeys(data_dirs)):
        for path in sorted(glob.glob(str(Path(directory) / "*.jsonl"))):
            with open(path) as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _system_prompt(sample: dict[str, Any]) -> str:
    original = next(
        (m.get("content") or "" for m in sample.get("messages", []) if m.get("role") == "system"), ""
    )
    try:
        return build_enriched_system_prompt(sample, original)
    except Exception:
        return original


def gold_calls_with_context(sample: dict[str, Any]) -> list[tuple[dict[str, Any], str, int]]:
    """Each gold tool call, the text readable before it, and its turn index."""
    prompt = _system_prompt(sample)
    seen: list[str] = [prompt]
    out: list[tuple[dict[str, Any], str, int]] = []
    for index, message in enumerate(sample.get("messages", [])):
        role = message.get("role")
        if role == "assistant":
            calls = (message.get("annotations") or {}).get("tool_calls") or []
            for call in calls:
                out.append((call, "\n".join(seen), index))
        if role != "system":
            seen.append(message.get("content") or "")
    return out


def triage(samples: list[dict[str, Any]], replies: dict[int, list[str]]) -> dict[str, Any]:
    buckets: collections.Counter[str] = collections.Counter()
    # Why a wrong value is wrong, for the two value buckets: the split that
    # decides whether the fix is a data convention, a schema rule or training.
    sub: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    sub_examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    calls_total = matched = 0
    conversations_with_replies = 0

    for index, sample in enumerate(samples, start=1):
        gold = gold_calls_with_context(sample)
        if not gold:
            continue
        said = replies.get(index, [])
        conversations_with_replies += bool(said)
        predicted = [c for reply in said for c in parse_tool_calls(reply)]
        pairs = align_calls([g for g, _, _ in gold], predicted)
        for (expected, actual), (_, context, turn) in zip(pairs, gold):
            calls_total += 1
            failures = classify_call(expected, actual, context)
            if not failures:
                matched += 1
                continue
            for failure in failures:
                buckets[failure.bucket] += 1
                if failure.bucket in ("value_in_context", "value_unknowable"):
                    reason = refine_value_mismatch(failure.expected, failure.actual, context)
                    sub[failure.bucket][reason] += 1
                    key = f"{failure.bucket}/{reason}"
                    if len(sub_examples[key]) < 10:
                        sub_examples[key].append({
                            "conversation": sample.get("conversation_id"),
                            "turn": turn,
                            "detail": failure.describe(),
                        })
                if len(examples[failure.bucket]) < 12:
                    examples[failure.bucket].append({
                        "conversation": sample.get("conversation_id"),
                        "idx": index,
                        "turn": turn,
                        "detail": failure.describe(),
                    })
    return {
        "calls_total": calls_total,
        "calls_matched": matched,
        "calls_failed": calls_total - matched,
        "conversations": len(samples),
        "conversations_with_replies": conversations_with_replies,
        "buckets": dict(buckets.most_common()),
        "examples": {k: v for k, v in examples.items()},
        "value_reasons": {k: dict(v.most_common()) for k, v in sub.items()},
        "value_reason_examples": dict(sub_examples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    samples = load_samples(args.data)
    replies = parse_log(args.log)
    if not replies:
        print("no model_response events — was the run logged at DEBUG?", file=sys.stderr)
        return 1
    report = triage(samples, replies)
    report["log"] = str(args.log)
    report["data"] = args.data

    total, failed = report["calls_total"], report["calls_failed"]
    print(f"{args.log.name}: {total} gold tool calls, {report['calls_matched']} matched, {failed} failed")
    print(f"{'bucket':20} {'count':>7} {'% of failures':>14} {'% of all calls':>15}")
    for bucket, count in report["buckets"].items():
        print(f"{bucket:20} {count:7} {count / max(failed, 1):13.1%} {count / max(total, 1):14.1%}")
    for bucket, reasons in report["value_reasons"].items():
        total_bucket = sum(reasons.values())
        print(f"\n  {bucket} split by why the value differs ({total_bucket}):")
        for reason, count in reasons.items():
            print(f"    {reason:18} {count:6} {count / max(total_bucket, 1):7.1%}")
    print("\nno_call includes turns the harness never asked for (openers, segments after a "
          "withheld result), so it is an upper bound on the model's own omissions.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
