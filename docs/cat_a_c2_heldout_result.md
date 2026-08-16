# Cat A cell C2 — the first held-out result measured on a model that trained on full-length conversations

**Date:** 2026-08-16
**Model:** `checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767` (DVC stage `task_a_sft_gemma4_26b_a4b_c2`, recorded in `7e758da`)
**Cell:** C2 — `response_only` @ `max_seq_length: 8192`, task-a-sft-v2 corpus, LoRA r=16/α=16, lr 5e-5, 3 epochs (1767 steps)
**Baselines:** `sft-gemma4-v3` (`checkpoint-1770`, v1 corpus) and `sft-gemma4-v4` (`checkpoint-1767`, v2 corpus), both `all_tokens` and both — per R16 — actually trained on a 1024-token window.
**Companion to:** [`cat_a_corpus_v2_heldout_regression.md`](cat_a_corpus_v2_heldout_regression.md) (R15), [`sft_max_length_truncation_bug.md`](sft_max_length_truncation_bug.md) (R16), [`cat_a_loss_mask_and_truncation_analysis.md`](cat_a_loss_mask_and_truncation_analysis.md) (R14).
**Evidence:** `runs/audit/heldout_c2_ckpt1767_v2corpus.json`; tables reproduced by `scripts/stratify_heldout_audit.py`.

> **STATUS: C2 clears the pre-registered ≥0.75 composite bar — 0.7595, +0.1887 over v3.**
> Every moving component improves with a bootstrap CI excluding zero. Real tool-calling ability
> (the 71 rows that require a call) went 0.249 → 0.636. **R15's "the v2 corpus regresses" verdict is
> superseded** — see §5. GRPO on this checkpoint is the right next step, but three blockers listed
> in §7 must be cleared first.

---

## 1. Headline

206 held-out prompts, greedy (`do_sample=False`), seed 42, identical decode across all three
checkpoints. The audit set is the same pinned, contamination-free set used for v3 and v4, rebuilt
via `scripts/build_heldout_clean_set.py` and confirmed by `--verify-against` to match the stored
audit on 206/206 rows — so these composites are directly comparable to the ones in R15.

| Component | v3 (v1 corpus) | v4 (v2 corpus) | **C2** | Δ C2 vs v3 |
|---|---|---|---|---|
| **composite** | 0.5709 | 0.5120 | **0.7595** | **+0.1887** |
| state_acc (w 0.4) | 0.5922 | 0.4951 | **0.9369** | +0.3447 |
| tool_f1 (w 0.4) | 0.7233 | 0.6877 | **0.8649** | +0.1416 |
| task (w 0.2) | 0.2233 | 0.1942 | 0.1942 | −0.0291 |

Paired against v3 on the same rows (10,000-resample bootstrap on per-row deltas, exact sign test on
the discordant rows):

| Metric | mean Δ | 95% CI | better / worse / tied | sign test |
|---|---|---|---|---|
| composite | +0.1887 | [+0.1505, +0.2288] | 83 / 6 / 117 | p = 2e-18 |
| state_acc | +0.3447 | [+0.2767, +0.4126] | 75 / 4 / 127 | p = 5.2e-18 |
| tool_f1 | +0.1416 | [+0.0858, +0.1998] | 36 / 6 / 164 | p = 2.8e-06 |

All three CIs exclude zero. The 83-vs-6 split on the composite is the number to trust: this is not a
small mean shift over noisy rows, it is a broad, one-directional improvement.

---

## 2. This run changed two things at once

C2 differs from the v3/v4 runs in **both** axes of the R14 factorial, so it does not isolate either:

1. **Loss mask** — `response_only` instead of `all_tokens`. Only meaningful after `bac1d98`, which
   fixed `render_response_only_sample` returning 2 tokens / 0 unmasked labels on transformers 5.x.
   No prior C2 result was ever valid.
2. **Sequence window** — a real 8192-token window instead of the silent 1024 of R16. Confirmed on
   disk: `training_args.bin` records `max_length=8192, truncation_mode=keep_start` for all four C2
   checkpoints, against `max_length=1024` for `checkpoint-1767` of v4.

This matters for attribution, not for the decision. Per R14's measurement the enriched system prompt
is a median 3,016 tokens, so under the old 1024-token `keep_start` window **v3 and v4 never saw a
single tool-calling turn** — those sit far past token 1024. C2 is therefore the first Cat A
checkpoint whose training data contained the behaviour this evaluation measures. The C1 cell
(`all_tokens` @ 8192) would separate window from loss mask; §7 argues it is not worth blocking on.

The training curve is unremarkable and shows no overfit — eval loss fell monotonically to the final
step (0.5532 → 0.5300 → 0.5237 → 0.5230 at steps 500/1000/1500/1767, best at the last). As R15 and
R14 both note, that number cannot rank cells against each other, because `response_only` averages
loss over a different denominator than `all_tokens`. The ranking rests entirely on this audit.

---

## 3. Stratified by ground-truth turn type

Every audit row is a single assistant turn. 98 of 206 rows are self-loop turns and all 71
tool-bearing rows sit inside that stratum, so the convention axis and the tool-calling axis land on
the same rows.

| Stratum | Metric | v3 | v4 | **C2** | |
|---|---|---|---|---|---|
| GT self-loop (n=98) | state_acc | 0.3469 | 0.3878 | **0.9490** | ✅ |
| | emits a self-loop | 34.7% | 39.8% | **94.9%** | ✅ |
| | tool_f1 | 0.5306 | 0.4558 | **0.7364** | ✅ |
| GT advances (n=108) | state_acc | 0.8148 | 0.5926 | **0.9259** | ✅ |
| | spurious self-loop | 3.7% | 26.9% | **2.8%** | ✅ |
| | tool_f1 | 0.8981 | 0.8981 | **0.9815** | ✅ |

**The R15 failure mode is gone.** v4's defect was that it learned *"emit self-loops"* rather than
*"emit self-loops on tool-calling turns"* — it gained 5 points where the convention wants a
self-loop and 23 points where it does not. C2 gains 55 points where the convention wants one while
*dropping* spurious self-loops on advancing rows below even v3's rate. It learned the rule as a
conditional, which is what the corpus fix was always asking for.

---

## 4. Tool calling

Read the aggregate `tool_f1` with the §4 caveat from R15: `tool_call_f1([], []) == 1.0` and 135 of
206 rows carry no ground-truth tool call, so most of the aggregate is credit for staying silent.
The honest number is the 71-row stratum.

| On the 71 rows that **require** a tool | v3 | v4 | **C2** |
|---|---|---|---|
| tool_f1 (real ability) | 0.3662 | 0.2488 | **0.6362** |
| emits NO tool call | 35.2% | 57.7% | **12.7%** |
| right tool name (any overlap) | 60.6% | 39.4% | **87.3%** |
| exact call (name + args) | 29.6% | 21.1% | **60.6%** |
| **On the 135 rows requiring none** | | | |
| spurious tool call | 8.9% | 8.2% | **1.5%** |

Real tool-calling ability improved 2.6× over v4 and 1.7× over v3, while spurious calls on
no-tool rows fell to 1.5%. Both directions improved at once, which rules out a simple
precision/recall trade — the model is not just calling tools more often. Corpus-wide it emits 0.335
calls per row against v3's 0.286.

### 4.1 Residual failure taxonomy (71 tool-bearing rows)

| Outcome | n | Share |
|---|---|---|
| Perfect — right tool, all arguments exact | 44 | 62.0% |
| Right tool name, imperfect arguments | 18 | 25.4% |
| No tool call emitted | 9 | 12.7% |
| — of which also advanced state ("announce-but-don't-call") | 2 | 2.8% |
| — of which stayed in state but stayed silent | 7 | 9.9% |
| **Wrong / hallucinated tool name** | **0** | **0.0%** |

Tool *selection* is solved on this set. The bottleneck has moved to **argument fidelity** (18 rows),
which is the single largest remaining bucket and is directly shapeable by the GRPO reward's
`tool_call_f1` term. The "announce-but-don't-call" gap that motivated
[`grpo_tool_emission_gap_review.md`](grpo_tool_emission_gap_review.md) survives on only 2 of 71 rows.

---

## 5. What this does and does not establish

**R15's verdict is superseded, and its own §6 predicted this.** R15 concluded that the task-a-sft-v2
corpus regresses held-out performance, but flagged under R16 that the comparison was run between two
models that never trained on the turns the corpus fix targets. Its recommended next step was
literally "run C2 on the v2 corpus before blaming the corpus." That has now happened, and the v2
corpus supports 0.7595 once training can actually see the relevant turns. **Do not carry forward
"the v2 corpus is bad" as a finding.** The correct statement is narrower: the v2 convention is
unlearnable from a 1024-token window, and R15 measured the window, not the corpus.

**What is still not established:**

- **Attribution between the two fixes.** C2 changed loss mask and window together. We cannot say how
  much of +0.1887 is `response_only` and how much is the 8192 window. C1 would answer this.
- **The `task` component.** Unchanged at 0.1942 and should not be read as a failure. It scores
  whether the completion reaches the conversation's terminal state, on rows that are
  mid-conversation turns where terminating would be wrong. It cannot distinguish "correctly not
  terminal" from "failed to terminate" and mostly just depresses the composite by its 0.2 weight.
  `state_acc` remains the one clean component.
- **Absolute serving quality.** As with every composite in R15, the audit harness loads checkpoints
  with `load_in_4bit=True` while Cat A SFT trains in bf16, so this is a 4-bit measurement of a bf16
  model. Applied identically to all three checkpoints, so the comparison holds — but 0.7595 is a
  floor for bf16 serving, and the pre-registered ≥0.75 bar was not set under this constraint.
  Clearing it *under* the 4-bit handicap is the stronger reading.

---

## 6. Samples

**Row 0 — clean success** (healthcare, self-loop tool turn):

```
GT    [STATE: CHECK_ELIGIBILITY → CHECK_ELIGIBILITY]
      verify_coverage{patient_id: "P98765", procedure_code: "72188", provider_id: "PROV001"}

PRED  [STATE: CHECK_ELIGIBILITY → CHECK_ELIGIBILITY]
      <tool_call>{"name": "verify_coverage", "arguments": {"patient_id": "P98765",
                  "procedure_code": "72188", "provider_id": "PROV001"}}</tool_call>
```

State correctly stays, correct tool, all three arguments verbatim. `tool_f1 = 1.000`. The composite
for this row is still only 0.800 because `task = 0` — the §5 artifact, on a row where terminating
would have been wrong.

**Row 37 — the surviving "announce-but-don't-call" failure** (telecom, self-loop tool turn):

```
GT    [STATE: UPGRADE_OFFER → UPGRADE_OFFER]
      change_plan{account_id: "AC987654", new_plan_id: "unlimited_max_100"}

PRED  [STATE: UPGRADE_OFFER → PROCESS_CHANGE]
      "ยินดีอย่างยิ่งครับคุณสมศรี ผมกำลังดำเนินการเปลี่ยนแพ็กเกจเป็น
       'Unlimited Max 100' ให้กับบัญชี AC987654 ของคุณในทันทีครับ"
      (no tool call)
```

The completion narrates performing the plan change — correct account, correct plan, stated as
already in progress — advances the state as though the call had succeeded, and never emits it. This
is the fabricated-tool-result pattern from R15 §5.1. It now affects 2 of 71 rows rather than being
the dominant mode, but it is the failure worth watching through GRPO, because the reward's
`task_completion` term can pay for narration if `tool_call_f1` is not weighted firmly against it.

---

## 7. Recommendation

**Proceed to GRPO on `checkpoint-1767` of C2. Do not block on C1.**

C1 (`all_tokens` @ 8192) is worth running for attribution, but it is an appendix, not a gate. It
changes the plan only if it beats 0.7595, and the mechanism argues against that: R14 measured that
under `all_tokens` roughly 71% of tokens are the near-identical system prompt repeated across all
4,711 training examples, and a further 2.5% are `tool`-role payloads that actively train the model
to *generate* tool results — the hallucination this work exists to suppress. At ~16 s/it × 1767
steps that is ~8 GPU-hours to most likely confirm C2's lead. Run it when the GPU is otherwise idle.

GRPO is the right next step on the merits: the residual failures (§4.1) are argument fidelity and
emission, both of which the `reward_business_logic` terms `tool_call_f1` (0.30) and
`state_transition_correctness` (0.30) shape directly, and tool selection is already solved.

**Three blockers must be cleared before any GRPO launch** (all verified 2026-08-16):

1. **`configs/training/grpo_cat_a.yaml:91` sets `data.source: "data/output/task_a"`, which does not
   exist.** Only `grpo/`, `heldout/` and `sft/` live under `data/output/`. `grpo.py:739` reads this
   key directly, so the run fails at data load.
2. **`data/output/grpo/task_a` is v1-derived** — dated 2026-07-09, while the v2 corpus is
   2026-08-06. Its ground truth encodes the superseded convention, so the reward would penalise the
   behaviour C2 just learned. It feeds both the training rows and the in-run reward-hacking detector
   (`_HeldOutEvalCallback`, `grpo.py:750`). Regenerate with `scripts/filter_grpo_data.py`, whose
   defaults already read `data/output/sft/task_a_splits` (v2) and write `data/output/grpo/task_a`.
3. **`grpo.py:775` derives `output_dir` from the config stem with no override**, and
   `scripts/run_phase2_grpo.sh:143` still writes a fixed, non-run-specific `PATCHED_CFG`. This is
   the exact pair fixed for SFT under R13 and never ported. On a ~9-hour run it costs per-run
   provenance, and any later invocation of the script silently overwrites the frozen config.

Budget note: per R9, Gemma-4 falls back to HF `model.generate()` rollouts rather than Unsloth's
colocated vLLM, at roughly 31 s/step — about 8.6 hours per 1000 GRPO steps.

---

## 8. Reproducing

```bash
# 0. Fetch the C2 checkpoints (all four: 500 / 1000 / 1500 / 1767).
.venv-train/bin/dvc pull task_a_sft_gemma4_26b_a4b_c2

# 1. Materialize the v1 corpus out-of-place (the working copy is v2).
.venv-train/bin/python scripts/materialize_dvc_lineage.py \
    --dir-hash 6bb5eb6f7c48356ca05078c537ae68b1 --out /tmp/v1_splits

# 2. Rebuild the pinned contamination-free set and prove it is the same one.
.venv-train/bin/python scripts/build_heldout_clean_set.py \
    --candidate-split data/output/sft/task_a_splits/test.jsonl \
    --exclusion-split /tmp/v1_splits/train.jsonl \
    --exclusion-split /tmp/v1_splits/validation.jsonl \
    --out-dir data/output/heldout/cat_a_v2_test_not_in_v1 \
    --expect-clean 206 \
    --verify-against runs/audit/heldout_ckpt1767_v2corpus.json
# -> 278 candidates, 63 in v1 train + 9 in v1 val excluded, 206 clean
# -> [verify] OK — 206/206 rows match heldout_ckpt1767_v2corpus.json

# 3. Score C2.  ~17 min for 206 greedy generations on an H100 80GB.
.venv-train/bin/python scripts/heldout_composite_audit.py \
    --checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767 \
    --data-dir data/output/heldout/cat_a_v2_test_not_in_v1 \
    --split test --n-prompts 206 --seed 42 \
    --output runs/audit/heldout_c2_ckpt1767_v2corpus.json

# 4. Reproduce every table above.
.venv-train/bin/python scripts/stratify_heldout_audit.py \
    v3=runs/audit/heldout_ckpt1770_v1corpus.json \
    v4=runs/audit/heldout_ckpt1767_v2corpus.json \
    C2=runs/audit/heldout_c2_ckpt1767_v2corpus.json
```

Step 4 against the two stored v3/v4 audits reproduces every figure in R15 §1 and §3 exactly, which
is what qualifies the script to report the C2 column.
