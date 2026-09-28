#!/usr/bin/env python3
"""Rescore tool calls with declared argument sources, from a stored run log.

Exact match is the wrong instrument for a free-text argument: the model writes
a faithful paraphrase of the customer's words and scores zero (R31). This
rescores completed runs under the owners declared in a sources ledger:

  free_text     compared by similarity, with every number and identifier in the
                reference required to appear (eval/free_text_similarity)
  system        supplied by the runtime; not scored against the model
  user/derived  exact, as today

    python scripts/rescore_with_argument_sources.py \
        results/exp_a/*_v4_auto.log \
        --data data/output/benchmark/task_a_v4 \
        --data data/output/benchmark/task_a_voice_v3 \
        --sources data/interim/task_a_argument_sources/sources.json

No GPU: the run logs keep every `model_response` event, and the tool metric is
recomputed with the production evaluator, so the BEFORE column reproduces the
recorded `tool_metrics_conversation` rather than approximating it.

This CHANGES the number. It is a different question about the same
generations, so report it beside the recorded score and never in place of it,
and rescore every model before comparing any two (the R27 rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_workflow_agents.eval.argument_scoring import canonicalize_call, load_ledger
from llm_workflow_agents.eval.free_text_similarity import DEFAULT_THRESHOLD
from llm_workflow_agents.eval.tool_call_f1 import (
    TurnGroundTruth,
    TurnPrediction,
    evaluate_tool_calls_conversation,
    parse_tool_calls,
)
from llm_workflow_agents.eval.tool_call_failures import align_calls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_tool_call_failures import gold_calls_with_context, load_samples, parse_log  # noqa: E402

VOICE_WEIGHT = 0.30


def rescore(
    log: Path,
    samples: list[dict[str, Any]],
    scoring: Any,
) -> dict[str, Any]:
    replies = parse_log(log)
    counters: dict[str, int] = {}
    strata: dict[str, dict[str, list[Any]]] = {}

    for index, sample in enumerate(samples, start=1):
        modality = "voice" if sample.get("modality") == "voice" else "text"
        bucket = strata.setdefault(modality, {"gt": [], "before": [], "after": []})
        gold_calls = [call for call, _, _ in gold_calls_with_context(sample)]
        predicted = [c for reply in replies.get(index, []) for c in parse_tool_calls(reply)]
        rewritten = [
            fixed
            for expected, actual in align_calls(gold_calls, predicted)
            if (fixed := canonicalize_call(expected, actual, scoring, counters)) is not None
        ]
        bucket["gt"].append([TurnGroundTruth(turn_id=0, tool_calls=gold_calls)])
        bucket["before"].append([TurnPrediction(turn_id=0, content="", tool_calls=predicted)])
        bucket["after"].append([TurnPrediction(turn_id=0, content="", tool_calls=rewritten)])

    out: dict[str, Any] = {"log": log.name, "counters": counters, "strata": {}}
    for modality, bucket in strata.items():
        # Schemas are deliberately not passed: they differ per conversation, so a
        # single sample's schemas would report a meaningless hallucination rate.
        before = evaluate_tool_calls_conversation(bucket["before"], bucket["gt"])
        after = evaluate_tool_calls_conversation(bucket["after"], bucket["gt"])
        out["strata"][modality] = {"before": before.to_dict(), "after": after.to_dict()}
    return out


def blended_delta(result: dict[str, Any]) -> float:
    """How much the Phase 1 quality blend moves: 0.4 x tool F1, per stratum."""
    delta = 0.0
    for modality, weight in (("text", 1 - VOICE_WEIGHT), ("voice", VOICE_WEIGHT)):
        scores = result["strata"].get(modality)
        if scores:
            delta += weight * 0.4 * (scores["after"]["tool_call_f1"] - scores["before"]["tool_call_f1"])
    return delta


def exact_match_quality(stored: dict[str, Any], blended_delta_value: float) -> float:
    """The recorded run's quality under EXACT MATCH, whichever rule produced it.

    A run already scored under declared sources has the gain baked in, so adding
    the delta would count it twice; subtracting turns the recorded number into
    its exact-match counterfactual and keeps one column comparable across runs
    scored under either rule.
    """
    quality = stored["quality_summary"]["quality"]
    if (stored.get("argument_scoring") or {}).get("rule") == "declared_sources":
        return quality - blended_delta_value
    return quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--backend", default="token_f1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    samples = load_samples(args.data)
    scoring = load_ledger(args.sources, threshold=args.threshold, backend=args.backend)
    results = [rescore(log, samples, scoring) for log in args.logs]

    header = (
        f"{'run':<46} {'text tool F1':>17} {'voice tool F1':>17} {'blended quality':>19}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        result["blended_delta"] = blended_delta(result)
        recorded = Path("results/exp_a") / result["log"].replace(".log", ".json")
        quality = (
            exact_match_quality(json.loads(recorded.read_text()), result["blended_delta"])
            if recorded.exists()
            else None
        )
        cells = []
        for modality in ("text", "voice"):
            scores = result["strata"].get(modality)
            cells.append(
                f"{scores['before']['tool_call_f1']:.4f}->{scores['after']['tool_call_f1']:.4f}"
                if scores
                else "-"
            )
        quality_cell = (
            f"{quality:.4f}->{quality + result['blended_delta']:.4f}"
            if quality is not None
            else f"delta {result['blended_delta']:+.4f}"
        )
        name = result["log"].replace("_v4_auto.log", "")
        print(f"{name:<46} {cells[0]:>17} {cells[1]:>17} {quality_cell:>19}")

    keys = {k for r in results for k in r["counters"]}
    totals = {k: sum(r["counters"].get(k, 0) for r in results) for k in keys} or {
        "free_text_accepted": 0, "free_text_rejected": 0, "system_supplied": 0
    }
    print(
        f"\nfree text accepted {totals.get('free_text_accepted', 0)}, rejected "
        f"{totals.get('free_text_rejected', 0)}; system-supplied "
        f"{totals.get('system_supplied', 0)} "
        f"(backend={args.backend}, threshold={args.threshold})"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"scoring": scoring.to_dict(), "runs": results},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
