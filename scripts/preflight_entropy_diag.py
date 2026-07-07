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


def _patch_unsloth_gemma4_proxy_iter() -> None:
    """Filter ``num_kv_shared_layers`` out of ``_Gemma4KVSharedSafeProxy.__iter__``.

    The proxy must stay alive: cache ``__init__`` short-circuits the buggy
    ``layer_types[:-0] == []`` slice via ``hasattr(decoder_config,
    "num_kv_shared_layers") == False``, which only holds while the proxy's
    ``__getattr__`` raises on that name. So we cannot unwrap the proxy.

    But transformers 5.9.0's strict-dataclass validator ``validate_token_ids``
    does ``for name in text_config: getattr(text_config, name)`` — unconditional
    iteration, raw ``getattr``. The proxy's ``__iter__`` forwards to the real
    config's iterator, which yields ``num_kv_shared_layers`` — and the raw
    ``getattr`` on the proxy then triggers the raise. Filtering the name out
    of ``__iter__`` makes the validator skip it without disturbing ``hasattr``.
    """
    try:
        from unsloth_zoo.temporary_patches.gemma4 import _Gemma4KVSharedSafeProxy
    except ImportError:
        return

    _sentinel = "_unsloth_gemma4_proxy_iter_filtered"
    orig_iter = _Gemma4KVSharedSafeProxy.__iter__
    if getattr(orig_iter, _sentinel, False):
        return

    def __iter__(self):  # noqa: ANN001
        return iter(
            name for name in object.__getattribute__(self, "_real")
            if name != "num_kv_shared_layers"
        )

    setattr(__iter__, _sentinel, True)
    _Gemma4KVSharedSafeProxy.__iter__ = __iter__


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
    do_sample: bool = True,
) -> list[list[str]]:
    """Load `checkpoint`, generate K completions per prompt, return per-prompt lists.

    Each prompt → list[str] of length n_completions. With ``do_sample=True``
    (default) generation matches GRPO sampling (T=1.0, top_p=0.95) so reward_std
    here predicts what GRPO sees. With ``do_sample=False`` it decodes greedily
    (temperature/top_p ignored) — used by the RFT headroom probe to score the
    deterministic baseline each prompt's best-of-K is measured against.
    """
    # Unsloth must be imported before torch so its monkey-patches take effect.
    from unsloth import FastLanguageModel  # noqa: I001
    import torch

    # Importing unsloth installs unsloth_zoo's Gemma-4 KV-zero proxy. The
    # proxy is load-bearing for the cache __init__ workaround (its raising
    # __getattr__ on num_kv_shared_layers makes hasattr() return False, which
    # bypasses transformers' buggy layer_types[:-0] == [] slice). But under
    # transformers 5.9.0 the strict-dataclass validator iterates proxy
    # attributes and raw-getattrs each name, hitting the raise. Patch only
    # the proxy's __iter__ to skip num_kv_shared_layers — preserves cache
    # behaviour, bypasses the validator.
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
    """Compute the 4 reward sub-components (2026-05-21 redesign) for one (completion, gt) pair.

    Mirrors the current reward_business_logic body: state_transition (graded),
    tool_call_f1 (graded, placeholder-stripped), task_completion (active when
    terminal_reached), transition_legality (active when valid_transitions
    non-empty). chain_propagation / format_compliance / length_band were
    dropped in the 2026-05-21 redesign and are no longer reported.
    """
    from llm_workflow_agents.training.reward_utils import (
        extract_state_annotations,
        extract_tool_calls,
        graded_tool_call_f1,
        reached_terminal,
        transition_legality_score,
    )
    from llm_workflow_agents.training.rewards.reward_business_logic import (
        _graded_state_match,
        _strip_placeholder_args,
    )

    pred_states = extract_state_annotations(completion)
    gt_states = gt.get("state_annotations", [])
    r_state = _graded_state_match(pred_states, gt_states)

    pred_tools = extract_tool_calls(completion)
    gt_tools = _strip_placeholder_args(gt.get("tool_calls", []))
    r_tool = graded_tool_call_f1(pred_tools, gt_tools)

    terminal = gt.get("terminal_state", "")
    if not gt.get("terminal_reached", True):
        r_completion = None  # rescaled out (active-component mean)
    else:
        r_completion = 1.0 if reached_terminal(completion, terminal) else 0.0

    valid_transitions = gt.get("valid_transitions", [])
    if not valid_transitions:
        r_legality = None  # rescaled out (active-component mean)
    else:
        r_legality = transition_legality_score(pred_states, valid_transitions)

    return {
        "state_transition": r_state,
        "tool_call_f1": r_tool,
        "task_completion": r_completion,
        "transition_legality": r_legality,
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
