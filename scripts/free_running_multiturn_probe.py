#!/usr/bin/env python3
"""Free-running multi-turn probe — is the "announce-but-don't-call" gap an eval artifact?

Closes the outstanding recommendation from ``docs/grpo_tool_emission_gap_review.md``
§7.1/§7.3 (commit 69a55b7): the §7.1 forensics confirmed that the teacher corpus
splits narrate-then-call across **two separate assistant turns** for 70.5% of
tool-expected validation rows, and that 100% of those bare-call targets are
preceded (across a user turn) by an assistant turn that narrates the same intent
without calling. Single-turn teacher-forced slicing therefore trains *both* halves
as independently-correct turn-local behaviours, so the measured single-turn
tool-F1 (0.087) may understate true tool-emission competence.

This probe measures the same anchors under two conditions:

  A. **Teacher-forced single turn** (the deployed metric). Prompt = gold history
     up to and including the user turn that precedes the gold bare call, i.e.
     the model sees the *gold* announce turn. Generate once. This is exactly
     what ``_load_grpo_jsonl`` emits as a row, so it reproduces the deployed
     single-turn number on this slice.

  B. **Free-running two-turn window.** Prompt starts two assistant turns earlier
     (at the announce position). The model writes *its own* announce turn (T1),
     the **gold** user reply is appended verbatim, and the model generates again
     (T2). Success = the expected call is emitted anywhere in the T1/T2 window.

B differs from A in exactly one respect — who authored the announce turn and
whether the model is allowed a second turn to act on it. If B's emission rate is
materially higher than A's, the gap is substantially an eval-granularity artifact
and the priority shifts to fixing the single-turn eval + re-deriving the 0.75 bar
(§6.1's outstanding item) rather than an SFT retrain or RFT pilot.

**Anchor scope.** Only anchors whose announce turn is *itself* a valid slicer row
(i.e. ``messages[i-3]`` is a ``user``/``system`` turn) are probed, so both
conditions use in-distribution, slicer-shaped prompts and B changes one variable
rather than two. Anchors whose announce turn follows a ``tool`` response
(414 of 795 in validation) are excluded and reported as ``anchors_excluded_tool_tail``.

**Known limitation.** B's round-2 user message is the gold reply to the *gold*
announce. If the model's own T1 diverges semantically, that reply is a
counterfactual continuation. ``t1_shape`` is recorded per anchor so the caveat can
be quantified rather than assumed away; a user simulator would be needed to remove
it entirely.

The reduction (``summarize_probe``), anchor selection (``find_anchors``) and the
gate (``classify_gate``) are pure and unit-tested in
tests/unit/test_free_running_multiturn_probe.py.

Usage:
    .venv-train/bin/python scripts/free_running_multiturn_probe.py \
        --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000 \
        --data-dir data/output/grpo/task_a --split validation \
        --n-anchors 200 --output runs/preflight/free_running_probe_ckpt1000.json

    # No-GPU anchor census only (validates selection against §7.1's counts):
    .venv-train/bin/python scripts/free_running_multiturn_probe.py \
        --data-dir data/output/grpo/task_a --split validation --anchors-only
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import statistics
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

# A target turn counts as "fused" (narration + call in one turn) rather than a
# bare call if this much non-tool_call, non-STATE prose survives stripping.
# Calibrated against §7.1: reproduces its 332 fused / 794 bare-call split
# (this implementation counts 332 / 795 — a one-row boundary difference).
NARRATION_MIN_CHARS = 15

# Gate thresholds. `MATERIAL_DELTA` is the absolute increase in tool-emission
# rate (free-running window vs teacher-forced single turn) that counts as
# "materially higher" per §7.3's revised next step. `RELIABLE_WINDOW` is the
# absolute free-running rate at which the model can be said to fire "reliably".
MATERIAL_DELTA = 0.20
RELIABLE_WINDOW = 0.60

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
_STATE_RE = re.compile(r"\[STATE:[^\]]*\]")


def strip_tool_calls(text: str) -> str:
    """Remove ``<tool_call>`` blocks and ``[STATE: X → Y]`` markers, return the prose."""
    without_calls = _TOOL_CALL_BLOCK_RE.sub("", text or "")
    return _STATE_RE.sub("", without_calls).strip()


def classify_target_shape(content: str, gt_tool_calls: list) -> str:
    """Classify a gold assistant turn as ``bare_call`` / ``fused`` / ``no_tool``. Pure."""
    if not gt_tool_calls:
        return "no_tool"
    return "fused" if len(strip_tool_calls(content)) >= NARRATION_MIN_CHARS else "bare_call"


def find_anchors(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """Find probe anchors in one conversation. Pure — indices only, no rendering.

    An anchor is a gold assistant turn at index ``i`` such that:
      - it is a valid slicer row (``messages[i-1]`` is ``user``/``system``),
      - its ground truth carries tool calls and its shape is ``bare_call``,
      - ``messages[i-2]`` is an assistant turn that narrates *without* calling
        (the announce turn).

    ``announce_is_sliceable`` records whether the announce turn is itself a
    valid slicer row; ``main`` probes only those so conditions A and B share
    prompt shape.
    """
    msgs = conv.get("messages") or []
    anchors: list[dict[str, Any]] = []
    for i, msg in enumerate(msgs):
        if i < 2 or msg.get("role") != "assistant":
            continue
        if (msgs[i - 1].get("role") or "") not in ("user", "system"):
            continue
        gt_tool_calls = (msg.get("annotations") or {}).get("tool_calls") or []
        if classify_target_shape(msg.get("content") or "", gt_tool_calls) != "bare_call":
            continue

        announce = msgs[i - 2]
        if announce.get("role") != "assistant":
            continue
        if (announce.get("annotations") or {}).get("tool_calls"):
            continue

        announce_prev_role = (msgs[i - 3].get("role") or "") if i >= 3 else "none"
        anchors.append(
            {
                "conversation_id": conv.get("conversation_id"),
                "complexity_level": conv.get("complexity_level"),
                "domain": conv.get("domain"),
                "target_index": i,
                "announce_index": i - 2,
                "user_index": i - 1,
                "announce_prev_role": announce_prev_role,
                "announce_is_sliceable": announce_prev_role in ("user", "system"),
                "gt_tool_calls": gt_tool_calls,
            }
        )
    return anchors


def classify_free_running_outcome(t1_calls: list, t2_calls: list) -> str:
    """Classify the free-running two-turn window. Pure.

    ``fired_at_t1`` means the model fused narration and call into its own
    announce turn (it never needed the second turn); ``fired_at_t2`` means it
    announced first and fired once the user reply landed — the behaviour §7.3
    predicted the single-turn eval would miss.
    """
    if t1_calls:
        return "fired_at_t1"
    if t2_calls:
        return "fired_at_t2"
    return "never_fired"


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return (sum(1 for r in records if r[key]) / len(records)) if records else 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_probe(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-anchor probe records to the summary block. Pure — no model, no I/O.

    Each record carries the condition-A fields (``a_emitted``, ``a_name_match``,
    ``a_f1``, ``a_graded_f1``) and the condition-B fields (``b_outcome``,
    ``b_emitted``, ``b_name_match``, ``b_f1``, ``b_graded_f1``, ``t1_shape``).
    """
    n = len(records)
    outcomes = {
        key: sum(1 for r in records if r["b_outcome"] == key)
        for key in ("fired_at_t1", "fired_at_t2", "never_fired")
    }
    t1_shapes = {
        key: sum(1 for r in records if r["t1_shape"] == key)
        for key in ("announce_no_call", "fused_with_call", "empty")
    }

    single_turn_rate = _rate(records, "a_emitted")
    free_running_rate = _rate(records, "b_emitted")

    # Paired McNemar-style discordance: the probe's whole claim rests on anchors
    # that flip, so report both directions rather than only the net delta.
    recovered = sum(1 for r in records if not r["a_emitted"] and r["b_emitted"])
    lost = sum(1 for r in records if r["a_emitted"] and not r["b_emitted"])

    return {
        "n_anchors": n,
        "single_turn_emission_rate": single_turn_rate,
        "free_running_emission_rate": free_running_rate,
        "emission_rate_delta": free_running_rate - single_turn_rate,
        "single_turn_name_match_rate": _rate(records, "a_name_match"),
        "free_running_name_match_rate": _rate(records, "b_name_match"),
        "single_turn_mean_f1": _mean([r["a_f1"] for r in records]),
        "free_running_mean_f1": _mean([r["b_f1"] for r in records]),
        "single_turn_mean_graded_f1": _mean([r["a_graded_f1"] for r in records]),
        "free_running_mean_graded_f1": _mean([r["b_graded_f1"] for r in records]),
        "free_running_outcomes": outcomes,
        "free_running_outcome_fracs": {
            k: (v / n if n else 0.0) for k, v in outcomes.items()
        },
        "t1_shapes": t1_shapes,
        "n_recovered_by_free_running": recovered,
        "n_lost_by_free_running": lost,
        "frac_recovered_of_single_turn_failures": (
            recovered / sum(1 for r in records if not r["a_emitted"])
            if any(not r["a_emitted"] for r in records)
            else 0.0
        ),
    }


def classify_gate(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify the probe summary into a verdict. Pure.

    ARTIFACT_DOMINANT  free-running fires reliably AND materially beats the
                       single-turn number -> the gap is largely eval granularity.
    PARTIAL_ARTIFACT   materially better but not reliable -> real weakness, but
                       the single-turn number overstates it.
    POLICY_DEFECT      no material gain -> the single-turn number is a fair read.
    """
    delta = summary["emission_rate_delta"]
    window = summary["free_running_emission_rate"]
    material = delta >= MATERIAL_DELTA
    reliable = window >= RELIABLE_WINDOW

    if material and reliable:
        verdict = "ARTIFACT_DOMINANT"
    elif material:
        verdict = "PARTIAL_ARTIFACT"
    else:
        verdict = "POLICY_DEFECT"

    return verdict, {
        "checks": {
            f"emission_rate_delta >= {MATERIAL_DELTA}": material,
            f"free_running_emission_rate >= {RELIABLE_WINDOW}": reliable,
        },
    }


_VERDICT_ACTION = {
    "ARTIFACT_DOMINANT": (
        "The model does fire on its own next turn once a user reply lands — the "
        "single-turn tool-F1 materially understates tool-emission competence. "
        "Shift priority to fixing the single-turn eval + re-deriving the 0.75 bar "
        "(§6.1) rather than an SFT retrain or RFT pilot."
    ),
    "PARTIAL_ARTIFACT": (
        "Free-running recovers a material share of the gap but does not fire "
        "reliably — both an eval-granularity artifact AND a real policy weakness. "
        "Re-derive the target bar first, then size the SFT/RFT fix against the "
        "free-running number, not the single-turn one."
    ),
    "POLICY_DEFECT": (
        "Free-running does not materially beat the teacher-forced single turn — "
        "the gap survives the eval-granularity confound and is a genuine policy "
        "weakness. Proceed to §4 step 4's combined factorial SFT run / RFT gating."
    ),
}


def _load_conversations(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split missing: {path}")
    convs: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                convs.append(json.loads(line))
    return convs


def _select_anchors(
    convs: list[dict[str, Any]], n_anchors: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pick at most one sliceable anchor per conversation, seeded.

    One anchor per conversation keeps the sample from being dominated by a few
    long conversations whose turns share most of their prompt — the same
    dedupe rationale as ``preflight_entropy_diag._sample_prompts``.
    """
    all_anchors = [a for conv in convs for a in find_anchors(conv)]
    sliceable = [a for a in all_anchors if a["announce_is_sliceable"]]

    rng = random.Random(seed)
    by_conv: dict[Any, list[dict[str, Any]]] = {}
    for a in sliceable:
        by_conv.setdefault(a["conversation_id"], []).append(a)

    conv_ids = sorted(by_conv, key=lambda c: str(c))
    rng.shuffle(conv_ids)
    chosen = [rng.choice(by_conv[cid]) for cid in conv_ids][:n_anchors]

    census = {
        "anchors_total": len(all_anchors),
        "anchors_sliceable": len(sliceable),
        "anchors_excluded_tool_tail": sum(
            1 for a in all_anchors if a["announce_prev_role"] == "tool"
        ),
        "conversations_with_sliceable_anchor": len(by_conv),
        "anchors_selected": len(chosen),
    }
    return chosen, census


def _build_prompts(
    conv_by_id: dict[Any, dict[str, Any]], anchor: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Build (condition-A prompt, condition-B round-1 prompt, gold user reply).

    Mirrors ``_load_grpo_jsonl``: the leading system message is re-enriched via
    ``build_enriched_system_prompt`` so the model sees the prompt the benchmark
    and GRPO rollouts see, and messages are stripped to ``{role, content}``.
    """
    from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
    from llm_workflow_agents.training.grpo import _slim_content

    conv = conv_by_id[anchor["conversation_id"]]
    raw_msgs = conv.get("messages") or []
    if (
        raw_msgs
        and raw_msgs[0].get("role") == "system"
        and conv.get("workflow_graph")
    ):
        raw_msgs = [
            {
                "role": "system",
                "content": build_enriched_system_prompt(
                    conv, raw_msgs[0].get("content") or "", force_rebuild=True
                ),
            },
            *raw_msgs[1:],
        ]

    def slim(msgs: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {"role": m.get("role", "") or "", "content": _slim_content(m.get("content"))}
            for m in msgs
        ]

    prompt_a = slim(raw_msgs[: anchor["target_index"]])
    prompt_b = slim(raw_msgs[: anchor["announce_index"]])
    gold_user_reply = slim([raw_msgs[anchor["user_index"]]])[0]
    return prompt_a, prompt_b, gold_user_reply


def _load_model(checkpoint: str, max_seq_length: int):  # noqa: ANN201
    """Load the checkpoint once and return ``(model, inner_tokenizer)``.

    Unlike ``preflight_entropy_diag._generate_for_checkpoint`` this keeps the
    model resident across all three generation rounds — reloading a 26B 4-bit
    checkpoint per round would triple load time for no benefit.
    """
    from unsloth import FastLanguageModel  # noqa: I001

    from preflight_entropy_diag import _patch_unsloth_gemma4_proxy_iter

    _patch_unsloth_gemma4_proxy_iter()

    print(f"[load] {checkpoint}", flush=True)
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    # Gemma-4 hands back a multimodal processor; template rendering and
    # tokenization both live on the inner tokenizer.
    return model, getattr(tokenizer, "tokenizer", tokenizer)


def _generate(
    model,  # noqa: ANN001
    inner_tok,  # noqa: ANN001
    message_lists: list[list[dict[str, str]]],
    max_new_tokens: int,
    max_seq_length: int,
    batch_size: int,
    label: str,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int | None = None,
) -> list[str]:
    """Decode one continuation per message list.

    Greedy by default (what this probe's own conditions use). ``do_sample=True``
    switches to temperature sampling — used by ``passk_tool_emission_probe.py``,
    which reuses this loader/generator rather than duplicating it.
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    rendered = [
        inner_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in message_lists
    ]
    out_texts: list[str] = []
    n_batches = (len(rendered) + batch_size - 1) // batch_size
    for start in range(0, len(rendered), batch_size):
        batch = rendered[start : start + batch_size]
        inputs = inner_tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length - max_new_tokens,
        ).to(model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": inner_tok.pad_token_id or inner_tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        for j, seq in enumerate(out):
            prompt_len = inputs["input_ids"][j].shape[0]
            out_texts.append(inner_tok.decode(seq[prompt_len:], skip_special_tokens=True))
        print(
            f"[gen] {label} batch={start // batch_size + 1}/{n_batches}",
            flush=True,
        )
    return out_texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="SFT checkpoint to probe (omit with --anchors-only).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-anchors", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--anchors-only",
        action="store_true",
        help="Print the anchor census and exit (no GPU, no generation).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    convs = _load_conversations(args.data_dir, args.split)
    anchors, census = _select_anchors(convs, args.n_anchors, args.seed)
    print(
        f"[data] {len(convs)} conversations from {args.data_dir}/{args.split}.jsonl\n"
        f"[data] anchor census: {json.dumps(census)}",
        flush=True,
    )

    if args.anchors_only:
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --anchors-only is set")
    if not args.output:
        parser.error("--output is required unless --anchors-only is set")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from llm_workflow_agents.training.reward_utils import (
        extract_tool_calls,
        graded_tool_call_f1,
        tool_call_f1,
    )

    conv_by_id = {c.get("conversation_id"): c for c in convs}
    built = [_build_prompts(conv_by_id, a) for a in anchors]
    prompts_a = [b[0] for b in built]
    prompts_b1 = [b[1] for b in built]
    gold_replies = [b[2] for b in built]

    t0 = time.time()
    model, inner_tok = _load_model(args.checkpoint, args.max_seq_length)

    # Round 1 — condition A: teacher-forced single turn at the anchor.
    comps_a = _generate(
        model, inner_tok, prompts_a, args.max_new_tokens,
        args.max_seq_length, args.batch_size, "A/teacher-forced",
    )

    # Round 2 — condition B, turn 1: the model writes its own announce turn.
    comps_b1 = _generate(
        model, inner_tok, prompts_b1, args.max_new_tokens,
        args.max_seq_length, args.batch_size, "B/T1-announce",
    )

    # Round 3 — condition B, turn 2: gold user reply lands, model continues.
    prompts_b2 = [
        [*p, {"role": "assistant", "content": t1}, reply]
        for p, t1, reply in zip(prompts_b1, comps_b1, gold_replies, strict=True)
    ]
    comps_b2 = _generate(
        model, inner_tok, prompts_b2, args.max_new_tokens,
        args.max_seq_length, args.batch_size, "B/T2-after-user-reply",
    )

    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    records: list[dict[str, Any]] = []
    for anchor, comp_a, t1, t2 in zip(anchors, comps_a, comps_b1, comps_b2, strict=True):
        gt_calls = anchor["gt_tool_calls"]
        gt_names = {c.get("name") for c in gt_calls if isinstance(c, dict)}

        a_calls = extract_tool_calls(comp_a)
        t1_calls = extract_tool_calls(t1)
        t2_calls = extract_tool_calls(t2)
        b_outcome = classify_free_running_outcome(t1_calls, t2_calls)
        b_calls = t1_calls if b_outcome == "fired_at_t1" else t2_calls

        if t1_calls:
            t1_shape = "fused_with_call"
        elif strip_tool_calls(t1):
            t1_shape = "announce_no_call"
        else:
            t1_shape = "empty"

        records.append(
            {
                "conversation_id": anchor["conversation_id"],
                "complexity_level": anchor["complexity_level"],
                "target_index": anchor["target_index"],
                "gt_tool_names": sorted(n for n in gt_names if n),
                "a_emitted": bool(a_calls),
                "a_name_match": bool(gt_names & {c.get("name") for c in a_calls}),
                "a_f1": tool_call_f1(a_calls, gt_calls),
                "a_graded_f1": graded_tool_call_f1(a_calls, gt_calls),
                "b_outcome": b_outcome,
                "b_emitted": bool(b_calls),
                "b_name_match": bool(gt_names & {c.get("name") for c in b_calls}),
                "b_f1": tool_call_f1(b_calls, gt_calls),
                "b_graded_f1": graded_tool_call_f1(b_calls, gt_calls),
                "t1_shape": t1_shape,
                "completion_a": comp_a,
                "completion_b_t1": t1,
                "completion_b_t2": t2,
            }
        )

    summary = summarize_probe(records)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    verdict, detail = classify_gate(summary)

    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "data": f"{args.data_dir}/{args.split}.jsonl",
            "n_anchors": summary["n_anchors"],
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy (do_sample=False)",
            "seed": args.seed,
            "material_delta": MATERIAL_DELTA,
            "reliable_window": RELIABLE_WINDOW,
        },
        "anchor_census": census,
        "summary": summary,
        "gate": {"verdict": verdict, **detail, "action": _VERDICT_ACTION[verdict]},
        "per_anchor": records,
    }
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    outcomes = summary["free_running_outcome_fracs"]
    print(
        f"\n=== free-running multi-turn probe — {Path(args.checkpoint).name} ===\n"
        f"  anchors                    = {summary['n_anchors']}\n"
        f"  A single-turn emission     = {summary['single_turn_emission_rate']:.3f}\n"
        f"  B free-running emission    = {summary['free_running_emission_rate']:.3f}\n"
        f"  delta                      = {summary['emission_rate_delta']:+.3f}"
        f"  (material >= {MATERIAL_DELTA})\n"
        f"  A / B mean strict tool-F1  = {summary['single_turn_mean_f1']:.3f}"
        f" / {summary['free_running_mean_f1']:.3f}\n"
        f"  B outcomes                 = fired_at_t1 {outcomes['fired_at_t1']:.3f}, "
        f"fired_at_t2 {outcomes['fired_at_t2']:.3f}, "
        f"never_fired {outcomes['never_fired']:.3f}\n"
        f"  recovered / lost           = {summary['n_recovered_by_free_running']}"
        f" / {summary['n_lost_by_free_running']}\n"
        f"  wall_time_s                = {summary['wall_time_s']}\n"
        f"\n  VERDICT: {verdict} — {_VERDICT_ACTION[verdict]}\n"
        f"[done] wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
