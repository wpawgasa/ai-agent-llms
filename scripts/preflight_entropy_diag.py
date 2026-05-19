#!/usr/bin/env python3
"""Pre-flight entropy diagnostic for GRPO restart (step 3 of grpo_diagnosis_gemma4_26b.md).

Compares two SFT checkpoints by generating K rollouts per prompt at GRPO
sampling settings (T=1.0, top_p=0.95) and reporting per-prompt reward_std —
the quantity GRPO actually consumes as advantage signal. If checkpoint-500
shows materially higher reward_std than checkpoint-1656, the SFT-too-good
hypothesis from the diagnosis is confirmed.

Usage:
    python scripts/preflight_entropy_diag.py \
        --checkpoints \
            checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-500 \
            checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1656 \
        --n-prompts 5 --n-completions 4 \
        --output runs/preflight/smoke.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sample_prompts(
    data_dir: Path,
    split: str,
    n_prompts: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Slice validation conversations into GRPO-format rows, then pick N unique prompts.

    Uses the exact slicer GRPO sees (one row per user→assistant boundary),
    then dedupes by conversation_id to avoid picking adjacent turns of the
    same conversation — those would inflate apparent diversity differences
    via prompt similarity, not policy entropy.
    """
    from llm_workflow_agents.training.grpo import _load_grpo_jsonl

    ds = _load_grpo_jsonl(data_dir, split=split)
    # _load_grpo_jsonl doesn't carry conversation_id; key on the rendered
    # system+user prefix instead. Two rows from the same conversation share
    # the same first user message, so this dedupes correctly.
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    chosen: list[dict[str, Any]] = []
    seen_first_user: set[str] = set()
    for i in indices:
        row = ds[i]
        msgs = row["prompt"]
        first_user = next(
            (m["content"] for m in msgs if m["role"] == "user"), ""
        )
        key = first_user[:200]
        if key in seen_first_user:
            continue
        seen_first_user.add(key)
        chosen.append({
            "prompt_messages": row["prompt"],
            "ground_truth": row["ground_truth"],
        })
        if len(chosen) >= n_prompts:
            break

    if len(chosen) < n_prompts:
        print(
            f"[warn] only {len(chosen)} unique conversations available "
            f"(requested {n_prompts}); proceeding with what we have.",
            file=sys.stderr,
        )
    return chosen


def _decode_gt(gt_str: str) -> dict[str, Any]:
    """Decode and reshape the GRPO ground_truth JSON the way _make_reward_adapter does."""
    d = json.loads(gt_str) if isinstance(gt_str, str) else (gt_str or {})
    if not isinstance(d, dict):
        d = {}
    if "state_sequence" in d and "state_annotations" not in d:
        seq = d["state_sequence"]
        if isinstance(seq, list):
            d["state_annotations"] = [
                (s.get("from", ""), s.get("to", "")) if isinstance(s, dict)
                else tuple(s) if isinstance(s, (list, tuple)) and len(s) == 2
                else ("", "")
                for s in seq
            ]
        else:
            d["state_annotations"] = []
    return d


def _generate_for_checkpoint(
    checkpoint: str,
    prompts: list[dict[str, Any]],
    n_completions: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_seq_length: int,
    batch_size: int,
    seed: int,
) -> list[list[str]]:
    """Load `checkpoint`, generate K completions per prompt, return per-prompt lists.

    Each prompt → list[str] of length n_completions. Generation matches GRPO
    sampling (T=1.0, top_p=0.95) so reward_std here predicts what GRPO sees.
    """
    # Unsloth must be imported before torch so its monkey-patches take effect.
    from unsloth import FastLanguageModel  # noqa: I001
    import torch

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

    # Gemma-4 returns a multimodal `Gemma4Processor` from
    # FastLanguageModel.from_pretrained, not a bare tokenizer. The processor's
    # __call__ signature is (images, text, videos, ...) and rejects positional
    # text — and apply_chat_template lives on the inner .tokenizer for the
    # processor case. Resolve once here and use the inner tokenizer for both
    # template rendering and tokenization.
    inner_tok = getattr(tokenizer, "tokenizer", tokenizer)

    # Render prompts once; reuse the tokenized form across the K rollouts.
    rendered: list[str] = []
    for p in prompts:
        rendered.append(
            inner_tok.apply_chat_template(
                p["prompt_messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    completions_per_prompt: list[list[str]] = [[] for _ in prompts]

    # Each "rollout pass" generates one completion for each of the N prompts;
    # do K passes to get K completions per prompt. Each pass uses a different
    # seed so the K samples are independent. Within a pass we batch across
    # prompts (batch_size) to amortize the per-call overhead.
    for k in range(n_completions):
        torch.manual_seed(seed + k)
        for batch_start in range(0, len(rendered), batch_size):
            batch = rendered[batch_start:batch_start + batch_size]
            inputs = inner_tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length - max_new_tokens,
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=inner_tok.pad_token_id or inner_tok.eos_token_id,
                )
            # Strip prompt tokens; decode only the new portion.
            for j, seq in enumerate(out):
                prompt_len = inputs["input_ids"][j].shape[0]
                new_tokens = seq[prompt_len:]
                text = inner_tok.decode(new_tokens, skip_special_tokens=True)
                completions_per_prompt[batch_start + j].append(text)
            print(
                f"[gen] ckpt={Path(checkpoint).name} pass={k+1}/{n_completions} "
                f"batch={batch_start//batch_size + 1}",
                flush=True,
            )

    # Free GPU memory before the next checkpoint loads.
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return completions_per_prompt


def _score_per_component(completion: str, gt: dict[str, Any]) -> dict[str, float]:
    """Compute the 5 reward sub-components for one (completion, gt) pair.

    Mirrors reward_business_logic's body but returns the raw component scores
    so we can see which one (if any) is saturating and squashing the
    effective reward variance.
    """
    from llm_workflow_agents.training.reward_utils import (
        chain_propagation_score,
        extract_state_annotations,
        extract_tool_calls,
        format_compliance_check,
        reached_terminal,
        tool_call_f1,
    )
    from llm_workflow_agents.training.rewards.reward_business_logic import (
        _length_band_score,
        _partial_state_match,
    )

    pred_states = extract_state_annotations(completion)
    gt_states = gt.get("state_annotations", [])
    r_state = _partial_state_match(pred_states, gt_states)

    pred_tools = extract_tool_calls(completion)
    gt_tools = gt.get("tool_calls", [])
    r_tool = tool_call_f1(pred_tools, gt_tools)

    pred_msgs = [{"role": "assistant", "content": completion}]
    gt_msgs = gt.get("messages", [])
    r_chain = chain_propagation_score(pred_msgs, gt_msgs)

    r_format = format_compliance_check(completion)
    r_length = _length_band_score(completion)

    terminal = gt.get("terminal_state", "")
    if not gt.get("terminal_reached", True):
        r_completion = None  # rescaled out
    else:
        r_completion = 1.0 if reached_terminal(completion, terminal) else 0.0

    return {
        "state_transition": r_state,
        "tool_call_f1": r_tool,
        "chain_propagation": r_chain,
        "format_compliance": r_format,
        "task_completion": r_completion,
        "length_band": r_length,
    }


def _score_and_summarize(
    completions_per_prompt: list[list[str]],
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score every (prompt, completion) and reduce to per-prompt + aggregate stats."""
    from llm_workflow_agents.training.rewards.reward_business_logic import (
        reward_business_logic,
    )

    per_prompt: list[dict[str, Any]] = []
    for i, (comps, p) in enumerate(zip(completions_per_prompt, prompts)):
        gt = _decode_gt(p["ground_truth"])
        rewards = reward_business_logic(
            prompts=[""] * len(comps),
            completions=comps,
            ground_truths=[gt] * len(comps),
        )
        components = [_score_per_component(c, gt) for c in comps]
        unique = len({c.strip() for c in comps})
        per_prompt.append({
            "prompt_index": i,
            "rewards": rewards,
            "reward_mean": statistics.fmean(rewards),
            "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "n_unique_completions": unique,
            "completion_lengths": [len(c) for c in comps],
            "completions": comps,
            "components": components,
            "terminal_reached": bool(gt.get("terminal_reached", True)),
        })

    stds = [r["reward_std"] for r in per_prompt]
    means = [r["reward_mean"] for r in per_prompt]
    return {
        "per_prompt": per_prompt,
        "mean_reward_std": statistics.fmean(stds) if stds else 0.0,
        "median_reward_std": statistics.median(stds) if stds else 0.0,
        "mean_reward": statistics.fmean(means) if means else 0.0,
        "frac_collapsed_groups": (
            sum(1 for s in stds if s < 0.01) / len(stds) if stds else 0.0
        ),
        "mean_unique_per_group": (
            statistics.fmean(r["n_unique_completions"] for r in per_prompt)
            if per_prompt else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Two or more SFT checkpoint paths to compare.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--n-completions", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    prompts = _sample_prompts(args.data_dir, args.split, args.n_prompts, args.seed)
    print(f"[data] sampled {len(prompts)} prompts from {args.data_dir}/{args.split}.jsonl",
          flush=True)

    results: dict[str, Any] = {
        "config": {
            "checkpoints": args.checkpoints,
            "n_prompts": len(prompts),
            "n_completions": args.n_completions,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "by_checkpoint": {},
    }

    for ckpt in args.checkpoints:
        t0 = time.time()
        comps = _generate_for_checkpoint(
            checkpoint=ckpt,
            prompts=prompts,
            n_completions=args.n_completions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        summary = _score_and_summarize(comps, prompts)
        summary["wall_time_s"] = round(time.time() - t0, 1)
        # All completions are now stored in summary["per_prompt"][i]["completions"];
        # no need for a separate sample field.
        results["by_checkpoint"][ckpt] = summary

        print(
            f"\n[summary] {ckpt}\n"
            f"  mean_reward_std       = {summary['mean_reward_std']:.4f}\n"
            f"  median_reward_std     = {summary['median_reward_std']:.4f}\n"
            f"  mean_reward           = {summary['mean_reward']:.4f}\n"
            f"  frac_collapsed_groups = {summary['frac_collapsed_groups']:.2f}\n"
            f"  mean_unique_per_group = {summary['mean_unique_per_group']:.2f}"
            f" / {args.n_completions}\n"
            f"  wall_time_s           = {summary['wall_time_s']}",
            flush=True,
        )

    # Print side-by-side comparison table.
    print("\n=== Side-by-side ===")
    header = f"{'metric':<26} " + " ".join(f"{Path(c).name:>22}" for c in args.checkpoints)
    print(header)
    for metric in ("mean_reward_std", "median_reward_std", "mean_reward",
                   "frac_collapsed_groups", "mean_unique_per_group"):
        row = f"{metric:<26} " + " ".join(
            f"{results['by_checkpoint'][c][metric]:>22.4f}" for c in args.checkpoints
        )
        print(row)

    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[done] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
