#!/usr/bin/env python3
"""Recompute chain propagation from a stored benchmark log, no GPU needed.

The benchmark result JSONs keep only aggregate metrics, but the run logs keep
every ``model_response`` event, and that is all the prediction side of this
metric needs: the tool calls the model emitted, in order. Use this to rescore
historical runs after the metric definition changed, instead of re-running the
benchmark.

    python scripts/recompute_chain_propagation.py \
        results/exp_a/<run>.log \
        --data data/output/benchmark/task_a_v2 \
        --data data/output/benchmark/task_a_voice

One caveat, stated in the output: turns the harness copied from ground truth
rather than asking the model for (the opener, an unsolicitable assistant turn)
emit no ``model_response`` event, so a call inside one is invisible here and
counts as a missed call. The live benchmark scores those turns as copied.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

from llm_workflow_agents.eval.tool_chain_propagation import evaluate_chain_propagation

SAMPLE_RE = re.compile(r"evaluating_sample\s+conversation_id=(\S+)\s+idx=(\d+)")
RESPONSE_RE = re.compile(r"model_response\s+content=('.*?'|\".*?\")\s+latency_ms=", re.DOTALL)


def parse_log(path: Path) -> dict[int, list[str]]:
    """Map 1-based sample index -> the model's replies for that conversation."""
    text = path.read_text(errors="replace")
    events: list[tuple[int, Any]] = []
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
    paths: list[str] = []
    for directory in sorted(dict.fromkeys(data_dirs)):
        paths.extend(sorted(glob.glob(str(Path(directory) / "*.jsonl"))))
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--data", action="append", required=True)
    args = parser.parse_args()

    samples = load_samples(args.data)
    replies = parse_log(args.log)
    if not replies:
        print("no model_response events found -- was the run logged at debug level?")
        return 1

    predictions: list[dict[str, Any]] = []
    ground_truths: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        messages = [{"role": "assistant", "content": c} for c in replies.get(index, [])]
        predictions.append({"messages": messages})
        ground_truths.append({"messages": sample.get("messages", [])})

    metrics = evaluate_chain_propagation(predictions, ground_truths)
    print(f"log                 {args.log.name}")
    print(f"conversations       {len(samples)} ({len(replies)} found in log)")
    print(f"opportunities       {metrics.total_chains} in {metrics.conversations_with_chains} conversations")
    print(f"accuracy            {metrics.chain_propagation_accuracy:.4f}")
    correct = round(metrics.chain_propagation_accuracy * metrics.total_chains)
    wrong = metrics.total_chains - correct - metrics.missed_calls
    made = metrics.total_chains - metrics.missed_calls
    print(f"carried correctly   {correct}")
    print(f"wrong value         {wrong}  (call made, value not the one the tool returned)")
    print(f"missed calls        {metrics.missed_calls} (includes harness-copied turns)")
    print(f"accuracy when made  {correct / made:.4f}" if made else "accuracy when made  n/a")
    print(f"per depth           {({k: round(v, 3) for k, v in metrics.per_depth_accuracy.items()})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
