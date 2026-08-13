# Every Cat A SFT run trained on a 1024-token window

**Date:** 2026-08-13
**Severity:** invalidates the training-side premise of R14 and R15, and both `sft-gemma4-v3` / `sft-gemma4-v4` as capability baselines.
**Versions:** TRL 1.0.0, transformers 5.12.1
**Fix:** `c5906e7` (`_sft_length_kwargs`), building on `bac1d98` (`response_only` render).
**Companion to:** [`cat_a_loss_mask_and_truncation_analysis.md`](cat_a_loss_mask_and_truncation_analysis.md) (R14) and [`cat_a_corpus_v2_heldout_regression.md`](cat_a_corpus_v2_heldout_regression.md) (R15), both of which need re-reading in light of this.

> **STATUS: `configs/training/*.yaml`'s `max_seq_length` has been inert for training since the
> TRL upgrade.** The trainer truncated every sample to the **first 1024 tokens**. For Task A that
> window is pure system prompt. Fixed; no corrected run has completed yet.

---

## 1. How it surfaced

Cell C2 (`response_only` @ 8192) was launched and logged, at every step:

```
{'loss': '0', 'grad_norm': '0', 'mean_token_accuracy': '0', 'entropy': '1.45', ...}
```

Loss, gradient norm and token accuracy all exactly zero, while `num_tokens` climbed and entropy
stayed ~1.4 — the forward pass was running, but no gradient reached the model. 128 steps of nothing.

Note this was **only** visible because C2 uses `response_only`. Under `all_tokens` the same bug
produces a perfectly healthy-looking loss curve (§4).

---

## 2. Root cause

TRL's `SFTTrainer` builds its own collator (`trl/trainer/sft_trainer.py:891`):

```python
data_collator = DataCollatorForLanguageModeling(
    pad_token_id=pad_token_id,
    max_length=None if self.padding_free else args.max_length,
    truncation_mode=args.truncation_mode,
    ...
)
```

and that collator re-truncates every batch (`sft_trainer.py:183-197`):

```python
if self.max_length is not None and not self.padding_free:
    if self.truncation_mode == "keep_start":
        sl = slice(None, self.max_length)
    ...
    input_ids = [ids[sl] for ids in input_ids]
    labels    = [lbl[sl] for lbl in labels]
```

`SFTConfig.max_length` defaults to **1024** and `truncation_mode` to **`keep_start`** — the first
1024 tokens are kept, the rest discarded.

`sft.py` never set it. It passed the length ceiling only conditionally:

```python
**({"max_seq_length": training_cfg.get("max_seq_length", 8192)}
   if "max_seq_length" in inspect.signature(SFTConfig).parameters else {}),
```

`max_seq_length` was a `SFTConfig` kwarg through TRL 0.22 and was **renamed to `max_length` in
0.23+**. On TRL 1.0.0 the guard is simply False, so the branch is a silent no-op. The accompanying
comment — *"We already truncate in `_render_chat` and pre-pack to max_seq_length, so the trainer
doesn't need it"* — is the wrong assumption: pre-tokenizing does not stop the collator from
truncating again.

**Pre-tokenizing to 4096 or 8192 never mattered. The collator cut to 1024 afterwards.**

---

## 3. Evidence

Four independent confirmations:

| Source | Result |
|---|---|
| `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1767/training_args.bin` (the completed v4 run) | `max_length = 1024`, `truncation_mode = keep_start` |
| C0 log, step 10: `num_tokens 8.192e4` ÷ (10 steps × batch 8) | exactly **1024.0** tokens/sample |
| C2 log, step 50: `num_tokens 4.096e5` ÷ (50 steps × batch 8) | exactly **1024.0** tokens/sample |
| Render 10 real conversations, count unmasked labels in `labels[:1024]` | **0 in 10/10** |

The `num_tokens` arithmetic is the cleanest: it comes from logs already collected during the v4 run
and lands on exactly 1024.0 per sample, not approximately.

---

## 4. Why it stayed invisible for so long

The enriched system prompt is a median **3,016 tokens** (R14), so a 1024-token `keep_start` window
never reaches the first user turn.

- **Under `all_tokens`** (every run before C2): labels fall back to `input_ids`, so the system
  prompt itself is the training target. It is near-identical across all 4,711 examples, making it
  trivially predictable — which produces a *better*-looking curve, not a worse one. This is the
  likely explanation for eval_loss 0.17 and token accuracy 0.9505 sitting alongside a held-out
  composite of only 0.51–0.57. The model was largely learning to recite the contract.
- **Under `response_only`** (C2): system/user/tool tokens are masked to −100, so every retained
  token in the window is masked, there are no valid targets, and the loss is exactly 0.

The bug is silent in the configuration the project has always used, and loud in the one it had
never successfully run — because `response_only` was independently broken until `bac1d98` (it
returned 2 tokens with 0 unmasked labels; see that commit).

---

## 5. Blast radius

**Both existing checkpoints trained on 1024-token windows:**

- `sft-gemma4-v3` (`checkpoint-1770`, v1 corpus)
- `sft-gemma4-v4` (`checkpoint-1767`, task-a-sft-v2 corpus)

Neither ever saw a tool-calling turn during training — those live far past token 1024. Consequences
for prior conclusions:

- **R14** measured that `max_seq_length: 4096` right-truncates 56% of conversations. True of our own
  `_render_chat` call, but it understates reality: the effective window was 1024, truncating
  essentially 100% of rows, and much earlier. The recommended C0/C1/C2 factorial compares cells that
  were all collapsing to the same 1024 tokens — **C0, C1 and C2 at the trainer level were the same
  experiment.**
- **R15** concluded the task-a-sft-v2 corpus regresses held-out performance (composite 0.5709 →
  0.5120). That comparison is *internally* valid — both models carried the identical handicap, the
  audit set was contamination-controlled, and the statistics stand. But it was measured between two
  models that never trained on the turns the corpus fix targets, so it does **not** establish that
  the corpus is bad. The verdict needs re-testing on corrected runs.
- The **held-out audit harness itself is unaffected** — it is inference-side and uses its own
  `--max-seq-length 8192`.

---

## 6. Fix

`training/sft.py::_sft_length_kwargs` passes the configured ceiling under whichever name the
installed TRL accepts (`max_length` on ≥0.23, `max_seq_length` on ≤0.22), and `SFTConfig` is built
with `**_sft_length_kwargs(training_cfg)`.

Regression cover in `tests/unit/test_training.py::TestTrainerMaxLength` constructs a real
`SFTConfig` and asserts `cfg.max_length == 8192` — it fails against the TRL default rather than
asserting on a dict, so a future rename breaks the test instead of silently reverting the bug.

**Expect runs to get much slower, and treat that as the fix working.** The aborted C2 ran at
3.65 s/it on 1024-token samples; real samples average ~4,700 tokens, so roughly 4.6× the tokens per
step plus superlinear attention. Peak VRAM also rises — the 1024-token run sat at 54.9 GB with
~52 GB of that being weights, leaving ~25 GB of headroom for activations at full length.

---

## 7. What to re-run

1. **C2 (`response_only` @ 8192)** on task-a-sft-v2 — relaunched 2026-08-13 with the fix.
2. **C0 on both corpora**, if R15's verdict matters for the GRPO decision. Only a corrected v1-vs-v2
   pair can say whether the stay convention helps; the current answer was measured through a
   1024-token keyhole.
3. **Do not** re-derive `data/output/grpo/task_a` or start GRPO until a corrected SFT baseline
   exists.

## 8. Reproducing the diagnosis

```bash
# What the trainer actually used, from a finished run:
python -c "import torch; a=torch.load('checkpoints/<run>/checkpoint-N/training_args.bin', \
    weights_only=False); print(a.max_length, a.truncation_mode)"

# From any train.log: tokens per sample should equal your configured ceiling, not 1024.
#   num_tokens / (logged_step * per_device_bs * grad_accum)
```

Any future run whose `num_tokens` per sample is a suspiciously round power of two is worth checking
against this.
