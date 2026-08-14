# Cat A corpus v2 — held-out regression: the stay convention gated tool calls behind state accuracy

**Date:** 2026-08-06
**Models:** `sft-gemma4-v4` (`checkpoint-1767`, task-a-sft-v2 corpus) vs `sft-gemma4-v3` (`checkpoint-1770`, v1 corpus)
**Cell:** C0 — `all_tokens` @ 4096, LoRA r=16/α=16, lr 5e-5, 3 epochs. Identical for both runs (verified by diffing `configs/training/sft_cat_a.yaml`, `sft.py`, `lora_targets.py`, `run_phase2_sft.sh` against tag `sft-gemma4-v3`: comment changes only).
**Companion to:** [`cat_a_state_annotation_convention_review.md`](cat_a_state_annotation_convention_review.md) (which motivated the corpus fix) and [`grpo_tool_emission_gap_review.md`](grpo_tool_emission_gap_review.md) (the announce-but-don't-call failure this makes worse).
**Evidence:** `runs/audit/heldout_ckpt1767_v2corpus.json`, `runs/audit/heldout_ckpt1770_v1corpus.json`.

> **STATUS: the task-a-sft-v2 corpus REGRESSES held-out performance.** It did exactly what it
> claimed — 0 stay violations, 100% of tool turns self-loop — and that very completeness is the
> problem. Do not build GRPO on `checkpoint-1767`. See §5 for the mechanism and §6 for options.

---

## 1. Headline

206 held-out prompts, greedy (`do_sample=False`), seed 42, identical decode for both checkpoints.

| Component | v3 (v1 corpus) | v4 (v2 corpus) | Δ |
|---|---|---|---|
| **composite** | 0.5709 | **0.5120** | **−0.0589** |
| state_acc (w 0.4) | 0.5922 | 0.4951 | −0.0971 |
| tool_f1 (w 0.4) | 0.7233 | 0.6877 | −0.0356 |
| task (w 0.2) | 0.2233 | 0.1942 | −0.0291 |

The regression is statistically real, not sampling noise:

- Paired bootstrap (10,000 resamples) on the per-row composite delta: **95% CI [−0.0913, −0.0278]**, excludes zero.
- Sign test on the 71 discordant rows (50 v3-better, 21 v4-better; 135 tied): **two-sided p = 7.7e-4**.

This contradicts the training signal. Best `eval_loss` was **0.17148** for v4 vs **0.17978** for v3, and the
gap was stable at ≈ −0.009 across all four evals. §4 explains why that was never evidence of quality.

---

## 2. The eval was built to be fair, and if anything favours v4

Three things were controlled deliberately; each is worth preserving in future comparisons.

**Contamination.** The naive choice — score both models on the v2 test split — leaks. Matching
conversations across corpora by a fingerprint of their user turns (which remediation does not
touch; `conversation_id` is **not** unique, two ids collide across splits), of v2's 278 test
conversations **63 appear in v3's training set** and 9 in its validation set. Scoring on the full
split would have handed v3 an unearned advantage on 23% of rows. The audit set is the **206
conversations held out from both models**.

**Ground-truth convention.** The default `--data-dir data/output/grpo/task_a` is the v1-derived
set, whose GT still encodes the old convention; scoring v4 against it would penalise it for doing
the right thing. GT here comes from the v2 corpus — the corrected convention.

**Prompt.** `_load_grpo_jsonl` rebuilds the system prompt via `build_enriched_system_prompt`, whose
content depends on `TASK_A_STAY_RULE`. It is unset, and `_resolve_stay_rule_flag()` defaults to
`True`, so **both models were told the stay convention** in the prompt. The prompt therefore
favours v4's training. v4 lost anyway.

---

## 3. Where it broke — stratified by GT turn type

Every audit row is a single assistant turn (`n_gt_trans == 1` for all 206). Of the 206 rows, **98
(47.6%) are self-loop turns**, tracking the 41.6% gold expectation from the convention review, and
**all 71 tool-bearing rows sit inside that stratum** — the convention change and the tool-calling
axis land on the same rows.

| Stratum | Metric | v3 → v4 | |
|---|---|---|---|
| GT self-loop (n=98) | state_acc | 0.3469 → 0.3878 (+0.0408) | ✅ |
| | emits a self-loop | 34.7% → 39.8% (+5.1) | ✅ |
| | tool_f1 | 0.5306 → 0.4558 (−0.0748) | ❌ |
| GT advances (n=108) | state_acc | 0.8148 → 0.5926 (**−0.2222**) | ❌ |
| | spurious self-loop | 3.7% → **26.9% (+23.2)** | ❌ |
| | tool_f1 | 0.8981 → 0.8981 (0.0000) | — |
| GT has tools (n=71) | tool_f1 | 0.3662 → **0.2488** | ❌ |
| | emits NO tool call | 35.2% → **57.7% (+22.5)** | ❌ |

**The fix generalised as an unconditional habit.** v4 gained 5 points of self-loop emission where
the convention wants one, and 23 points where it does not. It learned *"emit self-loops"*, not
*"emit self-loops on tool-calling turns"*.

Corpus-wide, v4 emits fewer tool calls than v3: 0.20 vs 0.29 per row, and 165/206 rows with zero
tool calls vs 148/206.

---

## 4. Read `tool_f1` and `task` with care

Two of the three components are misleading in aggregate at turn granularity. Both distortions apply
equally to each model, so the comparison stands, but the absolute levels do not mean what they look like.

- **`tool_f1` is mostly credit for staying silent.** `tool_call_f1([], []) == 1.0`, and 135 of 206
  rows have no GT tool call. Split: 0.9185 on the 135 no-tool rows vs **0.2488** on the 71 that
  require one. Real tool-calling ability on this set is 0.249, not 0.688.
- **`task` is near-meaningless per-turn.** It scores whether the completion reaches the
  conversation's terminal state, and all 206 rows carry a `gt_terminal` — but these are
  mid-conversation turns where terminating would be *wrong*. It cannot distinguish "correctly not
  terminal" from "failed to terminate", so its 0.2 weight mostly depresses the composite.

`state_acc` is the one clean component: one transition per row, right or wrong.

**Why `eval_loss` pointed the other way.** The two runs evaluate on their own validation splits. A
remediated corpus is a *more predictable* corpus, so a lower cross-entropy on it is expected whether
or not the policy improved. `eval_loss` cannot rank these cells — the same conclusion
`cat_a_loss_mask_and_truncation_analysis.md` §3 reaches for C0/C1/C2.

---

## 5. Mechanism — the corpus made tool calls conditional on state accuracy

P(assistant turn contains `<tool_call>` | its state annotation), over the training splits:

| Given annotation | v1 corpus (v3) | v2 corpus (v4) |
|---|---|---|
| **self-loop** | 66.3% (13,586 / 20,496) | 77.3% (17,252 / 22,328) |
| **advancing** | 10.7% (3,710 / 34,739) | **0.0% (0 / 34,664)** |

That zero is the whole story. The remediation moved *every* tool call onto a self-loop turn, so v4
learned an absolute rule: **an advancing annotation is never followed by a tool call.**

The model emits the annotation first. On self-loop rows it gets that right only ~39.8% of the time —
and on the ~60% where it emits an advancing annotation instead, the learned continuation is prose,
with no path back to a tool call. In v1 a wrong annotation still left a 10.7% escape hatch. The fix
converted two loosely-coupled decisions into a chain whose weakest link is the annotation.

Aggregate tool-call density barely moved (31.3% → 30.3% of assistant turns; the remediation added
1,757 assistant turns and 0 net tool calls), so density alone cannot explain a 22-point emission
collapse. The **conditional structure** can.

### 5.1 What it looks like

Row 171 — GT `[PROCESS_REQUEST → PROCESS_REQUEST]`, GT tool `manage_subscription`:

- **v3** emits `[PROCESS_REQUEST → RESOLVE]`, then the `manage_subscription` call. Wrong annotation, right tool. `tool_f1 = 1.0`.
- **v4** emits `[PROCESS_REQUEST → RESOLVE]`, then narrates *"let me just check for you… done, your account has been upgraded to premium"* — **no tool call at all**, and it fabricates the result. `tool_f1 = 0.0`.

This is the announce-but-don't-call failure of `grpo_tool_emission_gap_review.md`, made worse, and
now with invented tool output.

---

## 6. Recommendation

**Do not run GRPO on `checkpoint-1767`.** GRPO would optimise a policy whose tool emission is gated
behind an annotation it gets wrong ~60% of the time; the reward would mostly be shaped by the
annotation, and reward hacking toward self-loops is the obvious attractor (R5).

Options, in the order worth trying:

1. **Confirm it is the corpus, not the cell.** C0 (`all_tokens` @ 4096) spends 78.8% of gradient on
   tokens the model never emits and right-truncates 56% of rows, removing terminal transitions. Run
   **C2 (`response_only` @ 8192)** on the v2 corpus before concluding the corpus is at fault — the
   convention may be under-trained rather than wrong. Use the explicit `output_dir` key so the cell
   gets its own checkpoint path.
2. **Decouple the corpus.** Relax the 0.0% to allow tool calls on advancing turns where the workflow
   genuinely warrants it. v1's 10.7% was doing real work as an error-recovery path. The convention
   should constrain *annotation*, not forbid a tool call from ever co-occurring with an advance.
3. **Re-audit `data/output/grpo/task_a`.** It is still v1-derived; the `task_a_grpo` stage needs a
   repro before any GRPO run, and whichever corpus wins should be the one it filters from.

Nothing here impugns the remediation's execution — it hit every acceptance gate it set. The gates
just did not include "tool calls must remain reachable when the annotation is wrong."

---

## 7. Reproducing

The clean set is **not** stored — it is rebuilt from the two tagged corpora. The original was
built ad-hoc and lost, which left this section pointing at a `<clean-set-dir>` placeholder for a
week; `scripts/build_heldout_clean_set.py` now pins the construction, and `--verify-against`
proves a rebuild is the same set before any composite is compared to the ones in §1.

```bash
# 1. Materialize the v1 corpus out-of-place (the working copy is v2).
.venv-train/bin/python scripts/materialize_dvc_lineage.py \
    --dir-hash 6bb5eb6f7c48356ca05078c537ae68b1 --out /tmp/v1_splits

# 2. Rebuild the contamination-free set: v2 test conversations absent from v1 train/val,
#    keyed on a fingerprint of user turns (conversation_id is not unique).
#    --expect-clean fails loudly if either side is the wrong revision;
#    --verify-against replays the sampler and compares per-row GT to a stored audit.
.venv-train/bin/python scripts/build_heldout_clean_set.py \
    --candidate-split data/output/sft/task_a_splits/test.jsonl \
    --exclusion-split /tmp/v1_splits/train.jsonl \
    --exclusion-split /tmp/v1_splits/validation.jsonl \
    --out-dir data/output/heldout/cat_a_v2_test_not_in_v1 \
    --expect-clean 206 \
    --verify-against runs/audit/heldout_ckpt1767_v2corpus.json
# -> 278 candidates, 63 in v1 train + 9 in v1 val excluded, 206 clean
# -> [verify] OK — 206/206 rows match heldout_ckpt1767_v2corpus.json

# 3. Then, per checkpoint:
.venv-train/bin/python scripts/heldout_composite_audit.py \
    --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1767 \
    --data-dir data/output/heldout/cat_a_v2_test_not_in_v1 \
    --split test --n-prompts 206 --seed 42 \
    --output runs/audit/heldout_ckpt1767_v2corpus.json

# v3 must be materialized out-of-place — v3 and v4 share the checkpoint path and both
# contain checkpoint-500/-1000/-1500 with different weights (versioning doc §4).
python3 scripts/materialize_dvc_lineage.py --rev sft-gemma4-v3 --out /tmp/sft_v3
```

Wall time ≈ 24 min (v4) and ≈ 17 min (v3) for 206 greedy generations on an H100 80GB.

**A caveat that applies to every number in this document:** the audit loads checkpoints with
`load_in_4bit=True` (`preflight_entropy_diag._generate_for_checkpoint`), while Cat A SFT trains in
bf16. Every composite here is therefore a 4-bit measurement of a bf16 model. It is applied
identically to every checkpoint, so comparisons hold — but the absolute values are not the bf16
serving quality, and the pre-registered ≥0.75 composite bar was not set under this constraint.
