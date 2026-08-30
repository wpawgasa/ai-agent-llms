# Cat A DPO — the run completes, and changes nothing

**2026-08-30.** The 500-step DPO run finished. Measured against C2 on the
pinned, contamination-free 206-row held-out set, it is a **null result**: the
policy barely moved.

---

## 1. The number

| metric | C2 (SFT) | DPO checkpoint-400 | delta |
|--------|---------|--------------------|-------|
| **composite** | **0.7595** | **0.7566** | **−0.0029** |
| state_acc | 0.9369 | 0.9369 | +0.0000 |
| tool_f1 | 0.8649 | 0.8600 | −0.0049 |
| task | 0.1942 | 0.1893 | −0.0049 |

Per row: **199 of 206 identical**, 3 better, 4 worse.

State accuracy is unchanged to four decimal places. After 500 steps the model
emits the *same text* on 97% of held-out rows. This is not a small regression;
the policy did not move.

Both audits score the same set. It was rebuilt with
`scripts/build_heldout_clean_set.py` and `--verify-against` confirmed
**206/206 rows match** the stored C2 audit, so the two numbers are directly
comparable.

`checkpoint-400` was chosen because it tied for the best in-run guardrail score.

---

## 2. Why: the pairs are too easy, so the gradient vanishes

Training loss per chunk fell 0.4237 → 0.2795 → 0.1563 → 0.1285, and **two
chunks logged 0.0035 and 0.0031**.

A DPO loss near zero means chosen and rejected are already almost perfectly
separated. The implicit margin is large, so the gradient is ~0. The model
learned to *recognise the corruption* without changing how it generates.

This is exactly the trap R18 recorded when it recommended preference learning:
synthetic negatives teach discrimination against the corruption function
rather than the task. The training mix was **29,256 synthetic pairs against 51
mined ones**. Oversampling the mined rows to 20% does not fix a majority that
is trivially separable.

---

## 3. The in-run guardrail could not have caught this

Held-out composite at each chunk, 50 prompts, fixed seed:

| step | 100 | 200 | 300 | 400 | 500 |
|------|-----|-----|-----|-----|-----|
| composite | 0.6280 | 0.6440 | 0.6360 | 0.6440 | 0.6360 |

Flat from step 200 on. At n=50 a change of 0.008 is one row, so single-chunk
moves are noise.

The guardrail fired `STOP` at step 500 — training metric rising while held-out
fell 0.6440 → 0.6360. Technically correct, practically meaningless: step 500
was the last step anyway, and the move is within noise. **The mechanism works;
its resolution at 50 prompts does not support acting on one chunk.** Raise
`eval_held_out_num_prompts`, or read the trend rather than the last point.

Note these guardrail numbers are NOT comparable to 0.7595 — different sample,
different size. They are self-consistent across chunks and nothing more.

---

## 4. What the run did establish

The machinery is now proven end to end, which was not true before:

- chunked training at `max_seq_length: 6144` on length-filtered pairs
- five precomputes of 5,000 rows with **no host-memory growth** (section 11)
- the guardrail scoring real DPO checkpoints in a separate process, five times
- `dpo.py` resuming correctly from a checkpoint after an interrupt
- a held-out set rebuilt from two git tags and verified against a stored audit

**C2 (`checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767`) remains the
best Cat A checkpoint at 0.7595.** DPO neither beat it nor meaningfully harmed
it.

---

## 5. What to do next

**Harder negatives, not more steps.** More steps on separable pairs adds
nothing: the gradient is already ~0.

The blocking question is R18's open one — `scripts/mine_model_negatives.py`
found C2 wrong on only **12.8%** of TRAIN prompts (51 of 399) while the
held-out audit measures **38.0%** on TEST. The classifier-effect explanation is
already refuted (`_classify()` is stricter than the composite, 39.4% vs 38.0%
on the same 71 rows), leaving the split effect unconfirmed. That number governs
whether enough on-distribution negatives exist to make DPO learn anything, so
it should be settled before another DPO run.

Two smaller levers, worth less than the above:

- `beta: 0.1` holds the policy near the reference. Lowering it permits more
  movement — but on separable pairs there is little gradient to amplify.
- `learning_rate: 5.0e-6` on a LoRA adapter is conservative.

Reproduce the comparison:

```bash
.venv-train/bin/dvc fetch -T data/output/sft/task_a_splits   # --rev is not
                                                             # supported here
.venv-train/bin/python scripts/materialize_dvc_lineage.py \
    --dir-hash 6bb5eb6f7c48356ca05078c537ae68b1 --out /tmp/v1_splits
.venv-train/bin/python scripts/build_heldout_clean_set.py \
    --candidate-split data/output/sft/task_a_splits/test.jsonl \
    --exclusion-split /tmp/v1_splits/train.jsonl \
    --exclusion-split /tmp/v1_splits/validation.jsonl \
    --out-dir data/output/heldout/cat_a_v2_test_not_in_v1 \
    --expect-clean 206 \
    --verify-against runs/audit/heldout_c2_ckpt1767_v2corpus.json
.venv-train/bin/python scripts/heldout_composite_audit.py \
    --checkpoint checkpoints/dpo_cat_a/gemma-4-26B-A4B-it/checkpoint-400 \
    --data-dir data/output/heldout/cat_a_v2_test_not_in_v1 \
    --split test --n-prompts 206 --seed 42 \
    --output runs/audit/heldout_dpo_ckpt400.json
```
