#!/usr/bin/env python3
"""Audit what SFT actually trains on, and whether the benchmark shows the same thing.

Why this exists: the Gemma-4 chat template silently dropped every tool result
from the Task A corpus, so every Gemma-4 SFT run trained without one, and a
first render fix mis-cut 48% of turn boundaries. Neither was visible without
decoding the rendered sequences. See
docs/superpowers/specs/2026-09-17-gemma4-tool-results-design.md.

Two checks, both CPU-only:

  render (V1)  For each conversation, render exactly as SFT does
               (render_response_only_sample, system prompt rebuilt as in
               training) and verify:
                 - every tool result appears in the input and is not trained
                 - every trainable assistant reply is trained whole
                 - no user text and no turn header is trained
                 - no trainable reply fails to be located
  parity (V2)  For each assistant turn, the prompt the SFT model saw during
               training equals the request the benchmark sends in
               --tool-turn-format text, rendered with the same template.

Exits 1 if any check fails.

Usage:
    .venv-train/bin/python scripts/audit_tool_result_rendering.py \\
        --data data/output/sft/task_a_splits/train.jsonl \\
        --tokenizer google/gemma-4-12B-it --tokenizer unsloth/gemma-4-E4B-it \\
        --output runs/audit/tool_result_rendering.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BIG = 10**9


def _training_messages(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Messages exactly as sft._load_split builds them."""
    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt

    msgs = [dict(m) for m in raw.get("messages") or []]
    if msgs and msgs[0].get("role") == "system" and raw.get("workflow_graph"):
        msgs[0] = {
            "role": "system",
            "content": build_enriched_system_prompt(raw, msgs[0].get("content") or "", force_rebuild=True),
        }
    out = []
    for m in msgs:
        item = {"role": m.get("role", ""), "content": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"), ensure_ascii=False)}
        if m.get("loss") is False:
            item["loss"] = False
        out.append(item)
    return out


def _unused_header(tok: Any) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "Zq9probe"}],
        tokenize=False,
    )
    after_user = text[: text.find("Zq9probe")]
    return after_user[after_user.rfind("u") + 1 :]


def _message_char_spans(messages: list[dict[str, Any]], text: str) -> list[tuple[int, int] | None]:
    """Each message's own trimmed content span in the rendered text, in order."""
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for m in messages:
        content = str(m.get("content") or "").strip()
        at = text.find(content, cursor) if content else -1
        if at < 0:
            spans.append(None)
            continue
        spans.append((at, at + len(content)))
        cursor = at + len(content)
    return spans


def audit_render(tok: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Position-based: a message is trained if any token wholly inside its own span is.

    A substring test cannot work here — an assistant reply that quotes the user
    or a tool result verbatim would make that text look trained.
    """
    from llm_workflow_agents.data.tool_turns import TOOL_RESULT_PREFIX, to_text_tool_turns
    from llm_workflow_agents.training import sft

    inner = getattr(tok, "tokenizer", tok)
    c: Counter = Counter()
    failures: list[dict[str, Any]] = []
    for idx, raw in enumerate(rows):
        msgs = _training_messages(raw)
        converted = to_text_tool_turns(msgs)
        out = sft.render_response_only_sample(msgs, tok, BIG)
        text = tok.apply_chat_template(converted, tokenize=False)
        offsets = inner(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        labelled = [l != -100 for l in out["labels"]]
        if len(offsets) != len(labelled):
            c["conversations_failing"] += 1
            failures.append({"row": idx, "conversation_id": raw.get("conversation_id"), "problems": ["token count mismatch"]})
            continue
        _, missed = sft._assistant_char_spans(converted, text, sft._turn_terminator(tok))
        spans = _message_char_spans(converted, text)
        c["conversations"] += 1
        c["unlocated_turns"] += missed
        problems = []

        def trained_inside(span):
            s, e = span
            return any(lab for (a, b), lab in zip(offsets, labelled) if b > a and a >= s and b <= e)

        def fully_trained(span):
            s, e = span
            inside = [lab for (a, b), lab in zip(offsets, labelled) if b > a and b > s and a < e]
            return bool(inside) and all(inside)

        for original, m, span in zip(msgs, converted, spans):
            if original["role"] == "tool":
                c["tool_results"] += 1
                if span is None or not m["content"].startswith(TOOL_RESULT_PREFIX):
                    problems.append("tool result missing from input")
                elif trained_inside(span):
                    problems.append("tool result trained")
                else:
                    c["tool_results_ok"] += 1
            elif m["role"] == "assistant" and m.get("loss", True) is not False and span is not None:
                c["trainable_turns"] += 1
                if fully_trained(span):
                    c["turns_trained_whole"] += 1
                else:
                    problems.append("assistant reply not trained whole")
            elif m["role"] == "assistant" and m.get("loss", True) is False and span is not None:
                if trained_inside(span):
                    problems.append("loss:false turn trained")
            elif m["role"] in ("user", "system") and span is not None and trained_inside(span):
                problems.append(f"{m['role']} text trained")
        if missed:
            problems.append(f"{missed} trainable turn(s) not located")
        if problems:
            c["conversations_failing"] += 1
            if len(failures) < 10:
                failures.append({"row": idx, "conversation_id": raw.get("conversation_id"), "problems": sorted(set(problems))})
    return {"counts": dict(c), "failures": failures, "passed": c["conversations_failing"] == 0}


def audit_parity(tok: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from llm_workflow_agents.data.tool_turns import to_text_tool_turns
    from llm_workflow_agents.eval import agent_benchmark as ab

    c: Counter = Counter()
    failures: list[dict[str, Any]] = []
    for idx, raw in enumerate(rows):
        train_msgs = _training_messages(raw)
        # Only the turns the benchmark asks the model for: it replays an
        # assistant opener that precedes every user turn verbatim, without a
        # model call, so feeding it a ground-truth reply there would shift
        # every later reply by one turn.
        def _model_turns(messages):
            seen_user = False
            for m in messages:
                if m.get("role") == "user":
                    seen_user = True
                elif m.get("role") == "assistant" and seen_user:
                    yield m.get("content") or ""

        gt_replies = _model_turns(raw.get("messages") or [])
        requests: list[list[dict[str, Any]]] = []

        original = ab._call_vllm
        try:
            # Replay with the ground-truth reply at every model turn, so the
            # benchmark's context is the same conversation training saw.
            def fake_call_gt(endpoint, model, messages, temperature, tools=None, **_):
                requests.append(messages)
                return next(gt_replies, ""), [], 0.0, 0.0

            ab._call_vllm = fake_call_gt
            ab._replay_conversation("http://audit", "audit", raw, tool_turn_format="text")
        finally:
            ab._call_vllm = original

        # Training prompt for the k-th model-generated assistant turn.
        asst_positions = [i for i, m in enumerate(train_msgs) if m["role"] == "assistant"]
        for req in requests:
            n_asst_in_req = sum(1 for m in req if m["role"] == "assistant")
            if n_asst_in_req >= len(asst_positions):
                continue
            pos = asst_positions[n_asst_in_req]
            train_prompt = tok.apply_chat_template(
                [{k: v for k, v in m.items() if k in ("role", "content")} for m in to_text_tool_turns(train_msgs[:pos])],
                tokenize=False, add_generation_prompt=True,
            )
            bench_prompt = tok.apply_chat_template(
                [{k: v for k, v in m.items() if k in ("role", "content")} for m in req],
                tokenize=False, add_generation_prompt=True,
            )
            c["turns_compared"] += 1
            if train_prompt == bench_prompt:
                c["identical"] += 1
            else:
                c["different"] += 1
                if len(failures) < 5:
                    k = next((i for i, (a, b) in enumerate(zip(train_prompt, bench_prompt)) if a != b), min(len(train_prompt), len(bench_prompt)))
                    failures.append({
                        "row": idx, "conversation_id": raw.get("conversation_id"),
                        "first_difference_at": k,
                        "training": train_prompt[max(0, k - 80): k + 120],
                        "benchmark": bench_prompt[max(0, k - 80): k + 120],
                    })
    return {"counts": dict(c), "failures": failures, "passed": c["different"] == 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", action="append", required=True)
    parser.add_argument("--limit", type=int, default=0, help="audit the first N conversations (0 = all)")
    parser.add_argument("--checks", default="render,parity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    report: dict[str, Any] = {"data": str(args.data), "conversations": len(rows), "tokenizers": {}}
    ok = True
    for repo in args.tokenizer:
        tok = AutoTokenizer.from_pretrained(repo)
        entry = {}
        if "render" in args.checks:
            entry["render"] = audit_render(tok, rows)
            ok &= entry["render"]["passed"]
            print(f"[render] {repo}: {entry['render']['counts']} passed={entry['render']['passed']}", flush=True)
        if "parity" in args.checks:
            entry["parity"] = audit_parity(tok, rows)
            ok &= entry["parity"]["passed"]
            print(f"[parity] {repo}: {entry['parity']['counts']} passed={entry['parity']['passed']}", flush=True)
        report["tokenizers"][repo] = entry
    report["passed"] = ok
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[done] passed={ok} -> {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
