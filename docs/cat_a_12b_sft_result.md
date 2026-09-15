# Cat A — gemma-4-12B fine-tuned on corpus v3: result, and a leaked role word

**2026-09-15.** gemma-4-12B trained with the C2 recipe on `corpus/task-a-v3`
improves on held-out data by a margin similar to E4B's. On the Phase 1
benchmark it gains almost nothing overall, because its text score fell while
voice rose. Most of the regressions trace to one defect: **the fine-tuned
model starts 54.6% of its benchmark replies with the literal word `model`**,
because training and inference render the start of an assistant turn
differently for this model's chat template (section 5).

Companion to [cat_a_e4b_sft_benchmark_result.md](cat_a_e4b_sft_benchmark_result.md),
which covers E4B on the same corpus, recipe, benchmark and held-out sets.

---

## 1. What was run

| | |
|---|---|
| Tag | `model/sft-gemma4-12b-c2-on-task-a-v3` |
| DVC stage | `task_a_sft_gemma4_12b` (`frozen: true`) |
| Checkpoint | `checkpoints/sft_cat_a_12b/gemma-4-12B-it/checkpoint-3168`, dir hash `c6d0f566…`, 80 files, 3.00 GB |
| Base | `google/gemma-4-12B-it` (`Gemma4UnifiedForConditionalGeneration`, dense, 48 layers) |
| Corpus | `corpus/task-a-v3`: 8,441 train / 992 validation rows |
| Recipe | `configs/training/sft_cat_a_12b.yaml`, identical to the E4B and C2-v3 configs except `output_dir` |
| Code | `d899b55` (adds the `gemma4_12b` LoRA registry entry) |

bf16, LoRA r16 / alpha 16 on the seven standard projections, lr 5e-5 cosine, 5% warmup, effective batch 8, 3 epochs, `max_seq_length: 8192`,
`loss_mask: response_only`. Trainable parameters: 65,568,768 of 12,025,298,944
(0.55%). A short smoke run preceded the full run to confirm the model loads and
trains under Unsloth 2026.9.4.

---

## 2. Training

3,168 steps in 58,820 s (16.3 h), `train_loss` 0.4272.
W&B: `wpawgasa/huggingface/runs/34lmkm6y`.

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3168 |
|------|-----|------|------|------|------|------|------|
| 12B eval_loss | 0.4429 | 0.4218 | 0.4099 | 0.4035 | 0.4008 | 0.399 | **0.399** |
| E4B eval_loss | 0.4963 | 0.4606 | 0.4459 | 0.4368 | 0.4334 | 0.4315 | 0.4314 |

Evaluation loss fell at every checkpoint and had flattened by the end; the best
checkpoint is the last one. The 12B sits about 0.03 below E4B throughout, which
does not rank the models.

**Gradients ran much larger than E4B's.** Over the first 78 logged points the
median grad norm was 1.63 against E4B's 0.34, with spikes to 47.1 (17 points
above 5; E4B's maximum was 1.06). `sft.py` does not set `max_grad_norm`, so the default
of 1.0 applied and most 12B steps were clipped; E4B's rarely were. Loss fell
smoothly and never went NaN, but the 12B took smaller effective steps than its
learning-rate schedule implies.

---

## 3. Phase 1 benchmark

Same 508 conversations (258 text + 250 voice), 0 stochastic trials, 32K
context, blended as `0.70 × text + 0.30 × voice`. Both 12B rows ran on vLLM
0.24.0 with transformers 5.12.1 and the FlashInfer sampler off, so the 12B
comparison is like for like.

| metric | gemini-3.1-fl | gemini-3.5-fl | **12B SFT** | 12B base | E4B SFT | E4B base |
|---|---|---|---|---|---|---|
| **Blended quality** | 0.8299 | 0.8113 | **0.7186** | 0.7098 | 0.6678 | 0.6131 |
| Text | 0.8179 | 0.7976 | **0.6662** | 0.6964 | 0.6315 | 0.6035 |
| Voice | 0.8579 | 0.8434 | **0.8409** | 0.7409 | 0.7526 | 0.6356 |
| State sequence accuracy | 0.6839 | 0.6880 | 0.6476 | 0.6148 | 0.6125 | 0.5517 |
| Task completion | 0.9390 | 0.9665 | **0.6752** | 0.7559 | 0.4961 | 0.4961 |
| Recovery rate | 0.9910 | 0.9970 | **0.9580** | 0.6096 | 0.6276 | 0.4925 |
| Tool-call F1 | 0.6598 | 0.6055 | 0.6322 | 0.5512 | 0.6112 | 0.5135 |
| Tool name accuracy | 0.7183 | 0.7147 | **0.7937** | 0.6613 | 0.7605 | 0.5616 |
| Argument exact match | 0.4546 | 0.3889 | **0.5358** | 0.3803 | 0.5011 | 0.2927 |
| Chain propagation | 0.3055 | 0.3199 | 0.3084 | 0.2701 | 0.3000 | 0.2371 |
| Full workflow success | 0.3380 | 0.3343 | 0.2351 | 0.2010 | 0.1624 | 0.1079 |
| Voice boundary quality | 0.8977 | 0.8880 | 0.9864 | 0.9199 | 0.9889 | 0.9459 |

- **Tool calling is the best of any model measured:** argument exact match
  0.5358 and tool name accuracy 0.7937.
- **Voice rose +0.100, to 0.8409,** close to gemini-3.5-flash-lite (0.8434).
- **Recovery after an error rose from 0.6096 to 0.9580,** near Gemini's 0.99.
- **Text fell −0.030 and task completion fell from 0.7559 to 0.6752.** Text
  carries 70% of the blend, so this cancels most of the voice gain: blended
  quality moved only +0.009.

The benchmark JSON keeps only per-stratum totals, not per-stratum components,
so the text drop cannot be attributed to a component from the stored result.
The 12B SFT model does jump to the end state too early less often than the
untrained 12B (`terminal_state_reached_continuing` 571 → 493), so early
termination is not the cause.

---

## 4. Held-out audit

Corpus-v3 sets (`derived/task-a-heldout-v3`), one sampled assistant turn per
conversation, `--split test --seed 42`, greedy, 4-bit, text and voice as
separate runs. Ground truth is identical in every row across the 12B and E4B
audits (303 text, 145 voice), so all four models pair row by row.

| | Text composite (303) | Voice composite (145) |
|---|---|---|
| E4B untrained | 0.7127 | 0.7080 |
| E4B SFT | **0.7828** | **0.7894** |
| 12B untrained | 0.7155 | 0.7062 |
| 12B SFT | **0.7809** | **0.7766** |

| 12B: SFT vs untrained | Composite delta | 95% CI | better / worse | Sign test |
|---|---|---|---|---|
| Text | **+0.0653** | [+0.0429, +0.0884] | 55 / 12 | p = 1e-07 |
| Voice | **+0.0703** | [+0.0303, +0.1103] | 27 / 4 | p = 3.4e-05 |

| 12B | Text: untrained | Text: SFT | Voice: untrained | Voice: SFT |
|---|---|---|---|---|
| State accuracy | 0.8581 | 0.9571 | 0.8897 | 0.9586 |
| Tool F1, rows needing a tool (94 / 46) | 0.7021 | 0.7872 | 0.8261 | 0.8043 |
| Emits no tool call on those rows | 14.9% | 3.2% | 4.3% | 6.5% |
| Exact call (name + arguments) | 63.8% | 75.5% | 71.7% | 78.3% |
| Spurious call on rows needing none | 5.7% | 0.0% | 18.2% | 2.0% |
| Stays in state when it should advance | 14.1% | 0.6% | 8.9% | 2.2% |
| Voice format compliance | — | — | 0.8069 | **0.7241** |

- **The held-out gain is real on both sets,** and close to E4B's (+0.0702 text,
  +0.0814 voice). **The benchmark's text drop does not appear on held-out
  text:** held-out text rose +0.0653.
- **Fine-tuned, the two sizes score about the same on held-out data:** 12B SFT
  0.7809 / 0.7766 against E4B SFT 0.7828 / 0.7894. Held-out gives no advantage
  to the larger model at this recipe.
- **Voice format compliance fell** from 0.8069 to 0.7241, the opposite of E4B
  (0.9586 → 0.9724). Section 5 explains why.
- On voice rows needing a tool, tool F1 dipped slightly (0.8261 → 0.8043); on 46
  rows that is within noise.

Held-out composites are a separate scale from the benchmark and are not
comparable to C2's 0.7595.

---

## 5. The leaked `model` role word

**Symptom.** 12B SFT completions start with the bare word `model`:

```
model
[STATE: SELECT_SLOT → CONFIRM_BOOKING]
<S>I will book the 09:00 slot for you now.</S>
```

| Where | 12B SFT | 12B untrained | E4B SFT | E4B untrained |
|---|---|---|---|---|
| Benchmark replies starting with `model` | **3,233 of 5,917 (54.6%)** | 0 | 0 | 0 |
| Held-out text completions starting with `model` | **68 / 303** | 0 | 1 | 0 |
| Held-out voice completions starting with `model` | **37 / 145** | 0 | 0 | 0 |
| Held-out completions starting with `thought` | 0 | 23 text, 23 voice | 0 | 0 |

It accounts for **37 of the 40** voice-format violations in the 12B SFT audit:
the word sits outside every `<S>` chunk, which the format checker rejects. The
untrained 12B's violations are the same shape with `thought` (25 of 28). E4B
has none; its few violations are genuine format slips.

**Cause: training and inference render the start of the reply differently.**
`render_response_only_sample` trains every token of an assistant turn,
including the turn header. Rendered with each model's own tokenizer:

| | Trained assistant span | Generation prompt at inference ends with |
|---|---|---|
| E4B | `<|turn>model\n[STATE: A → B]\nHello there.<turn|>` | `<|turn>model\n` |
| 12B | `<|turn>model\n[STATE: A → B]\nHello there.<turn|>` | `<|turn>model\n<|channel>thought\n<channel|>` |

For E4B the inference prompt ends exactly where training's reply content begins,
so the model continues with content. The 12B's chat template (Google's
`gemma4_unified` template) inserts an empty thinking-channel block after the
header, which never appears in training. After that block the fine-tuned model
writes the header it was trained to produce, and the header's text part leaks
out as `model`. The untrained 12B, which was not trained on the header, instead
leaks `thought` from the channel block on some turns.

**Effect on the scores.**
- **Voice format compliance:** directly, as shown above.
- **Held-out composite:** little. The state and tool scorers find annotations
  and calls anywhere in the text, so a leading `model` line costs nothing.
  Held-out still rose +0.065 / +0.070.
- **Benchmark text and task completion: possibly, not proven.** The benchmark
  replays whole conversations, feeding each reply back as history, so more than
  half of the 12B SFT's replayed assistant turns carry a stray `model` line. That
  is the kind of difference that could affect later turns and reaching the end
  state. The stored results cannot separate this from other causes.

**What would fix it (not done).** Either render training turns so the reply
content starts where the inference prompt ends (for this template, after the
empty channel block), or mask the header tokens out of the loss. Both change
the recipe, so they would need a re-run and a fresh comparison. A serving-side
strip of a leading `model` line would only hide the symptom.

---

## 6. What this result does not show

- **Not comparable to C2's 0.7595**, which is bound to a different 206-row set.
- **No held-out score for C2 on these sets,** so the 26B cannot be placed on
  this scale yet.
- **Single runs with greedy decoding.** The held-out deltas have paired
  confidence intervals; the benchmark numbers do not.
- **The benchmark text drop is not explained.** The leak is a plausible
  contributor, but the stored results cannot attribute it.
- **Clipped gradients.** Most 12B steps were clipped at 1.0 and E4B's were not,
  so the two arms did not take equally sized optimizer steps despite identical
  configs.

---

## 7. Next steps

1. **Fix the header/channel mismatch and re-run the 12B** (section 5), then
   re-benchmark. That tests whether the leak caused the text and completion drop.
2. **Check other templates for the same mismatch** before fine-tuning any model
   whose generation prompt adds content after the turn header.
3. **Consider `max_grad_norm`** for the 12B only after the leak is fixed, so the
   two effects are not confounded.
4. **The E4B go/no-go RL probe** (`scripts/rft_headroom_probe.py`) is running
   and decides whether GRPO or DPO is worth trying on E4B.

---

## 8. Reproduce

```bash
git checkout model/sft-gemma4-12b-c2-on-task-a-v3
dvc pull task_a_sft_gemma4_12b

# Benchmark: merge checkpoint-3168 with the verified merge script
# (.runs/eval_12b_sft/merge_lora_verified.py, not in git), write a model YAML
# from configs/models_exp_a/gemma4_12b.yaml with model.name set to the merged
# path, then in .venv-infer-v024 (vLLM 0.24.0, transformers 5.12.1):
export VLLM_USE_FLASHINFER_SAMPLER=0
./scripts/run_exp_a_single.sh <model YAML> \
    --data data/output/benchmark/task_a \
    --data data/output/benchmark/task_a_voice \
    --stochastic-trials 0 --max-model-len 32768

# Held-out: rebuild the sets per derived/task-a-heldout-v3, then
env -u UNSLOTH_VLLM_STANDBY .venv-train/bin/python scripts/heldout_composite_audit.py \
    --checkpoint checkpoints/sft_cat_a_12b/gemma-4-12B-it/checkpoint-3168 \
    --data-dir data/output/heldout/cat_a_v3_test_not_in_v2 \
    --split test --n-prompts 304 --seed 42 --modality text \
    --output runs/audit/heldout_12b_ckpt3168_v3text.json
# (--checkpoint google/gemma-4-12B-it for the untrained baseline;
#  voice: cat_a_v3_test_voice, --modality voice, --n-prompts 146)
```

Result files: `results/exp_a/sft_cat_a_12b_ckpt3168_auto.json`,
`google_gemma-4-12B-it_auto.json`; `runs/audit/heldout_12b_{base,ckpt3168}_v3{text,voice}.json`.
