#!/usr/bin/env python3
"""Measure where the Cat A SFT token budget actually goes.

Answers two questions that decide the loss-mask recipe and the sequence-length
setting, both of which were previously argued from intuition:

1. What fraction of each training sequence is assistant tokens -- i.e. tokens
   the model is ever asked to produce? Everything else (system contract, customer
   turns, tool results) is context at inference time, so under
   ``loss_mask: all_tokens`` its gradient is spent fitting text the model will
   never generate.
2. How many conversations exceed ``training.max_seq_length``? The ``all_tokens``
   render path truncates RIGHT (``sft.py``: ``truncation=True, max_length=...``
   with a right-side tokenizer), so an over-long conversation loses its ENDING --
   the terminal state transition and the final tool calls.

The system message is rebuilt with ``build_enriched_system_prompt`` because that
is what training does at load time; measuring the raw corpus system message
would understate its share substantially.

Usage:
    source .venv/bin/activate && python scripts/measure_sft_token_budget.py \\
        --split data/output/sft/task_a_splits/train.jsonl \\
        --tokenizer google/gemma-2-2b-it \\
        --limit 250 \\
        --max-seq-lengths 4096 8192 16384

`--tokenizer` defaults to a small Gemma-family SentencePiece model so the script
runs without downloading a 26B checkpoint. Counts are therefore approximate for
the real target model; see the caveat in
`docs/cat_a_loss_mask_and_truncation_analysis.md`. The *shares* are robust --
tokenizer choice does not move a 71%/21% split into a different decision.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_workflow_agents.data.system_prompt import (  # noqa: E402
    build_enriched_system_prompt,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="data/output/sft/task_a_splits/train.jsonl")
    ap.add_argument("--tokenizer", default="google/gemma-2-2b-it")
    ap.add_argument("--limit", type=int, default=250,
                    help="conversations to sample (0 = all)")
    ap.add_argument("--max-seq-lengths", type=int, nargs="*",
                    default=[4096, 8192, 16384])
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def ntok(text: str) -> int:
        return len(tok(text or "", add_special_tokens=False)["input_ids"])

    by_role: collections.Counter[str] = collections.Counter()
    conv_lens: list[int] = []
    sys_lens: list[int] = []
    n = 0

    with open(args.split, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if args.limit and n >= args.limit:
                break
            rec = json.loads(line)
            messages = rec["messages"]
            n += 1
            total = 0
            for i, msg in enumerate(messages):
                if i == 0 and msg.get("role") == "system":
                    # what training actually feeds the model
                    content = build_enriched_system_prompt(rec, msg.get("content") or "")
                    sys_lens.append(ntok(content))
                else:
                    content = msg.get("content") or ""
                count = ntok(content)
                by_role[msg.get("role", "?")] += count
                total += count
            conv_lens.append(total)

    grand = sum(by_role.values())
    if not grand:
        print("no tokens measured -- is the split path right?", file=sys.stderr)
        return 1

    print(f"tokenizer: {args.tokenizer}")
    print(f"sampled {n} conversations, {grand:,} tokens\n")
    print("per-role token share")
    for role, count in by_role.most_common():
        print(f"  {role:10s} {count:9,}  {100 * count / grand:5.1f}%")

    assistant = by_role.get("assistant", 0)
    if assistant:
        print(f"\nassistant share: {100 * assistant / grand:.1f}%")
        print(f"response_only gradient-density multiplier: {grand / assistant:.2f}x")

    conv_lens.sort()
    print("\nconversation length (tokens)")
    print(f"  median {statistics.median(conv_lens):.0f}"
          f"  p90 {conv_lens[int(0.9 * len(conv_lens))]}"
          f"  max {conv_lens[-1]}")
    if sys_lens:
        print(f"  system message alone: median {statistics.median(sys_lens):.0f}"
              f"  max {max(sys_lens)}")

    print("\ntruncation exposure (all_tokens path truncates RIGHT -> loses the ending)")
    for cap in args.max_seq_lengths:
        over = sum(1 for length in conv_lens if length > cap)
        print(f"  max_seq_length {cap:6d}: {over:4d}/{len(conv_lens)} exceed"
              f"  ({100 * over / len(conv_lens):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
