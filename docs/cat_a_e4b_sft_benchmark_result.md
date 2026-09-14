# Cat A — gemma-4-E4B fine-tuned on corpus v3: the benchmark result

**2026-09-14.** gemma-4-E4B trained with the C2 recipe on `corpus/task-a-v3`
raises Phase 1 Task A quality from **0.6131 to 0.6678** (+0.055). Tool calling
is where it gained: its argument exact match is the best of every model
measured, Gemini included. Task completion did not move at all, and that one
term is most of what still separates it from gemma-4-12B and the Gemini
Flash-Lite models.

This is the **Phase 1 benchmark**, not the held-out audit. It is **not
comparable to C2's 0.7595**, which is bound to a different 206-row set.

---

## 1. What was run

| | |
|---|---|
| Tag | `model/sft-gemma4-e4b-c2-on-task-a-v3` |
| DVC stage | `task_a_sft_gemma4_e4b` (`frozen: true`) |
| Checkpoint | `checkpoints/sft_cat_a_e4b/gemma-4-E4B-it/checkpoint-3168`, dir hash `4c5c79a2…`, 82 files, 1.70 GB |
| Base | `unsloth/gemma-4-E4B-it` (`Gemma4ForConditionalGeneration`) |
| Corpus | `corpus/task-a-v3`: `train.log` reports 8,441 train / 992 validation rows |
| Recipe | `configs/training/sft_cat_a_e4b.yaml`, identical to `sft_cat_a_c2_corpus_v3.yaml` except `output_dir` |
| Code | `84c9f00` (the warmup fix below) |

The recipe is the C2 cell: bf16, LoRA r16 / alpha 16 on the language layers,
lr 5e-5 cosine with 5% warmup, effective batch 8, 3 epochs,
`max_seq_length: 8192`, `loss_mask: response_only`, packing off. One difference
from C2 on corpus v2: `lora.freeze_router: false`, so the `gate_proj` adapters
trained. Under C2 they stayed exactly zero (CLAUDE.md R24).

This run is the size arm of the comparison planned in
[training_plan_voice_and_e4b.md](training_plan_voice_and_e4b.md). The other arm,
the 26B trained on the same corpus with the same recipe, has **not** been run.
There is not enough disk on this machine for its 51.6 GB of base weights. So the
question that plan asks, *what does shrinking the orchestrator cost?*, is still
open. This document answers a narrower one: what did SFT do for E4B?

---

## 2. Training

One uninterrupted run: 3,168 steps in 22,750 s (6.3 h) on an H100 NVL,
`train_loss` 0.4586. W&B: `wpawgasa/huggingface/runs/hmydpqvs`.

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3168 |
|------|-----|------|------|------|------|------|------|
| eval_loss | 0.4963 | 0.4606 | 0.4459 | 0.4368 | 0.4334 | 0.4315 | **0.4314** |

Evaluation loss fell at every evaluation and was still falling slightly at the
end. There is no sign of overfitting, and the best checkpoint is the last one.
A fourth epoch might still help, but that is not tested.

Two things are absent from the log on purpose:

- **Token accuracy.** Unsloth computes the loss without building full logits,
  so TRL omits `mean_token_accuracy` and `entropy`. Setting
  `UNSLOTH_RETURN_LOGITS=1` restores them at a memory cost. Token accuracy is a
  weak signal here anyway: R16 records 0.9505 token accuracy next to a held-out
  composite of 0.51–0.57.
- **A cross-run ranking.** These `eval_loss` values rank checkpoints within this
  run only. Each run scores its own validation split, and a 4B and a 26B model
  have different losses on the same data.

Early loss confirms the run saw full-length conversations: about 4,500 tokens
per sample, not the 1,024-token window every run before 2026-08-13 used (R16).

---

## 3. The result

Same 508 conversations for every model (`corpus/task-a-benchmark-v1`, 258
text + `corpus/task-a-benchmark-voice-v1`, 250 voice), 0 stochastic trials,
32K context, blended quality `0.70 × text + 0.30 × voice`.

| metric | E4B base | **E4B SFT** | change | 12B base | gemini-3.5-flash-lite | gemini-3.1-flash-lite |
|---|---|---|---|---|---|---|
| **Blended quality** | 0.6131 | **0.6678** | +0.0547 | 0.7098 | 0.8113 | 0.8299 |
| Text | 0.6035 | 0.6315 | +0.0280 | 0.6964 | 0.7976 | 0.8179 |
| Voice | 0.6356 | **0.7526** | **+0.1170** | 0.7409 | 0.8434 | 0.8579 |
| State sequence accuracy | 0.5517 | 0.6125 | +0.0608 | 0.6148 | 0.6880 | 0.6839 |
| Task completion | 0.4961 | 0.4961 | 0.0000 | 0.7559 | 0.9665 | 0.9390 |
| Recovery rate | 0.4925 | 0.6276 | +0.1351 | 0.6096 | 0.9970 | 0.9910 |
| Tool-call F1 | 0.5135 | 0.6112 | +0.0977 | 0.5512 | 0.6055 | 0.6598 |
| Tool name accuracy | 0.5616 | **0.7605** | +0.1989 | 0.6613 | 0.7147 | 0.7183 |
| Argument exact match | 0.2927 | **0.5011** | **+0.2084** | 0.3803 | 0.3889 | 0.4546 |
| Chain propagation | 0.2371 | 0.3000 | +0.0629 | 0.2701 | 0.3199 | 0.3055 |
| Full workflow success | 0.1079 | 0.1624 | +0.0545 | 0.2010 | 0.3343 | 0.3380 |
| Voice boundary quality | 0.9459 | 0.9889 | +0.0430 | 0.9199 | 0.8880 | 0.8977 |

Result files in `results/exp_a/` (DVC dir `82a957ed…`):
`google_gemma-4-E4B-it_auto.json`, `sft_cat_a_e4b_ckpt3168_auto.json`,
`google_gemma-4-12B-it_auto.json`, `gemini_gemini-3.5-flash-lite_frontier.json`,
`gemini_gemini-3.1-flash-lite_frontier.json`.

---

## 4. Discussion

### 4.1 Tool calling is what SFT bought

The two largest moves are both about tool calls. Argument exact match rose from
0.2927 to **0.5011**, which is higher than every other model in the table,
including gemini-3.1-flash-lite at 0.4546. Tool name accuracy rose from 0.5616
to **0.7605**, also the highest. Tool-call F1 (0.6112) now matches
gemini-3.5-flash-lite (0.6055), though it still trails 3.1 (0.6598).

That matters because argument fidelity is the bottleneck the 26B work
identified: C2 had solved tool selection, and 77.5% of its remaining errors
were wrong arguments (R23). On this benchmark, a 4B model fine-tuned on the v3
corpus fills in arguments better than any untrained model tested.

Chain propagation, carrying a value from one tool's result into a later call,
improved less (0.2371 → 0.3000) and stays low for every model. The per-call
argument gain does not yet turn into multi-step correctness.

### 4.2 Voice gained four times as much as text

Voice rose +0.117 and text +0.028. Fine-tuned E4B now scores higher on voice
(0.7526) than the untrained 12B (0.7409), while it still trails the 12B on
text by 0.065.

The likely reason is exposure. The v3 corpus is 29% voice (2,455 of 8,441
training rows), and no untrained model has seen this project's voice format:
state and tool markers outside the `<S>…</S>` chunks, short chunks, and a
closing `[END_CONVERSATION]`. The format is learnable and SFT installed it. The
formatting diagnostics agree:

- **Boundary quality** (chunks that end at a real pause): 0.9459 → 0.9889,
  the best in the table. English went 0.9689 → 1.0000 and Thai
  0.9138 → 0.9738.
- **Chunks got shorter:** first chunk median 53 → 38 characters and
  90th percentile 99 → 63. A shorter first chunk means the orchestrator can start
  speech sooner.

These diagnostics are guardrails, not part of the composite (see
[05-eval.md](../.claude/rules/05-eval.md)). They show SFT installed the format.
They do not show the voice answers are better workflow decisions; the voice
stratum's quality score is what measures that.

### 4.3 Task completion did not move, and it is the gap

Task completion is 0.4961 before and after SFT: 252 of 508 conversations in
both runs. Every other metric moved, so an identical value looks like a broken
metric. It is not.

`task_completion_rate` counts a conversation as complete when its last
predicted state is a terminal state. The two models land on 252 by different
routes. In the logs, 43 conversations reach the terminal state early only
under the base model, and 35 only under the SFT model. The SFT model also
jumps to `TERMINAL` early much less often: the benchmark logged
`terminal_state_reached_continuing` 325 times for SFT against 465 for the
base. So SFT made the model less likely to declare the workflow done too soon,
without making it more likely to finish. The per-conversation scores were not
saved, so this rests on the log evidence, not a re-scoring.

This is now the largest gap in the table:

| | E4B SFT | 12B base | gemini-3.5-flash-lite |
|---|---|---|---|
| Task completion | 0.4961 | 0.7559 | 0.9665 |
| Recovery rate | 0.6276 | 0.6096 | 0.9970 |

Completion is 0.2 of each stratum's score. Closing half the gap to Gemini on
that term alone would be worth about +0.047 blended, close to everything SFT
gained here. Recovery rate tells a related story: E4B-SFT now recovers from an
error as often as the untrained 12B, but both are far below Gemini's 0.99.

Completion is a property of the whole conversation, not of one turn. That
matches what R23 concluded for the 26B: the single-turn signal is exhausted,
and the open direction is multi-turn.

### 4.4 Model size

Fine-tuning closed a little over half the gap between E4B and the untrained
12B: E4B-SFT gained 0.0547, and the gap was 0.0967. The untrained 12B still
wins on text (0.6964 against 0.6315), task completion, and full workflow
success. E4B-SFT wins on voice, tool name accuracy, argument exact match and
tool F1.

These are an untrained 12B against a fine-tuned 4B. The comparison says
nothing about a fine-tuned 12B. The planned size comparison (E4B against 26B,
both fine-tuned) is still the one that answers the size question.

### 4.5 State accuracy

State sequence accuracy rose 0.5517 → 0.6125, level with the untrained 12B
(0.6148). The per-turn transition accuracy is low for every model (0.2192 →
0.2613 for E4B; 0.3494 for gemini-3.1-flash-lite), and the invalid-transition
rate is 0.42–0.50 for all five, including both Gemini models. Read those two as
relative measures only; their absolute level reflects how strictly the
benchmark scores transitions, not how bad the models are.

### 4.6 Latency is not comparable

E4B-SFT has the lowest per-turn latency (425.5 ms average) and E4B the next
lowest, but both were served locally on an H100 while Gemini went through an
API, and the 12B ran on a different vLLM version. Treat the latency columns in
the result files as descriptive, not as a ranking.

---

## 5. What this result does not show

- **It is not the held-out audit.** Nothing here is comparable to C2's 0.7595.
  The corpus-v3 held-out sets are `data/output/heldout/cat_a_v3_test_not_in_v2`
  (304 text rows) and `cat_a_v3_test_voice` (146 voice rows). They have not been
  run on this checkpoint.
- **No confidence intervals.** Each model ran once, greedily, on 508
  conversations. Differences of a few thousandths (for example tool F1 0.6112
  against 0.6055) are within what a re-run could move. The large differences,
  such as argument exact match +0.208 or voice +0.117, are not in doubt.
- **The serving stacks differ.** Both E4B runs used vLLM 0.20.0 with
  transformers 5.6.2. The 12B needed vLLM 0.24.0 with transformers 5.12.1 and
  the FlashInfer sampler turned off (sampling is greedy, so this should not
  change results). The Gemini models ran through their API.
- **The chat template differs slightly.** The base E4B ran from Google's model
  files. E4B-SFT was served with the Unsloth tokenizer and chat template it was
  trained with. The weights are the same.
- **Revisions were not pinned.** The base E4B run served the Hub's latest
  Google revision at the time (`ee0ef60`), and the 12B served `707f0a3`. Neither
  runner config enforces a revision.
- **`hallucinated_tool_rate` is 0.0000 for all five models.** That is not
  credible across models this different; the metric probably does not fire on
  this benchmark. It is left out of the table.

---

## 6. Problems found on the way

None of these changed the result, but each would have cost a later run.

- **transformers 5.17 removed `warmup_ratio`.** `SFTConfig` raised `TypeError`
  after the corpus had rendered, 30 minutes into the first attempt. GRPO and
  DPO had a worse failure: their config filters would have dropped the key and
  trained with no warmup. Fixed in `84c9f00` with
  `_utils.warmup_kwargs`, which passes whichever argument the installed version
  accepts.
- **`freeze_router: true` froze `gate_proj`, not the router.** A substring
  match, plus router names that differ per architecture. C2 on corpus v2 trained
  without `gate_proj` adapters. Fixed in `0e440e8`; see CLAUDE.md R24.
- **The SFT runner never sets `WANDB_PROJECT`.** `logging.wandb_project` only
  switches W&B on, so SFT runs land in the default `huggingface` project. Not
  fixed.
- **`training/merge_adapter.py` is not safe for Gemma-4.** It loads
  `AutoModelForCausalLM`, but the adapter was trained on
  `Gemma4ForConditionalGeneration`, whose text layers have different names.
  PEFT only warns when adapter keys do not match, so the merge probably does
  nothing and saves the base model. This is suspected, not tested. The benchmark
  used a separate merge script that checks all 258 non-zero LoRA B tensors
  loaded and that sampled weights changed.
- **Merged Gemma-4 E-series checkpoints are missing 54 tensors.** transformers
  does not build `k_proj`, `v_proj` or `k_norm` for the 18 layers that reuse
  earlier layers' keys and values (layers 24–41), so `save_pretrained` does not
  write them. vLLM refuses to load the result. The benchmark copied those 54
  tensors unchanged from the base. That is exact, because no adapter can exist
  on a module that was never built. `merge_adapter.py` has the same gap.
- **`test_system_prompt_voice.py::test_real_text_rows_render_byte_identically`
  fails once corpus v3 is on disk.** It compares the first 20 rows of
  whatever corpus is present against a baseline captured from corpus v2, and it
  skips when no corpus is present. The test is fragile; the prompt code is not
  broken.

---

## 7. Next steps

1. **Run the held-out audit** on `checkpoint-3168` against the corpus-v3 sets,
   text and voice reported separately (R20). That is the number that can be
   compared with other fine-tuned checkpoints.
2. **Train the 26B C2-v3 arm on a machine with enough disk.** Only that
   completes the size comparison this run was built for.
3. **Investigate task completion.** Start from the 252 conversations: which end
   on a non-terminal state, and at which complexity level.
   `scripts/run_exp_a_per_level.sh` gives the per-level split this document
   lacks.
4. **Fix the tooling:** put the verified merge (including the 54-tensor
   completion) into `training/merge_adapter.py`, and set `WANDB_PROJECT` from
   the SFT config.

---

## 8. Reproduce

```bash
# Weights and corpus, from the tag
git checkout model/sft-gemma4-e4b-c2-on-task-a-v3
dvc pull task_a_sft_gemma4_e4b
dvc pull data/output/benchmark/task_a data/output/benchmark/task_a_voice

# Merge checkpoint-3168 into its base. The script that produced this result
# lives in .runs/eval_e4b_sft/merge_e4b_sft.py, which git ignores: it checks
# every non-zero LoRA B tensor loaded, checks sampled weights changed, and
# copies the 54 KV-shared tensors from the base. Do NOT use
# training/merge_adapter.py for this model (section 6).

# Benchmark the merged model, as for any local model
source .venv-infer/bin/activate
./scripts/run_exp_a_single.sh <model YAML whose model.name is the merged path> \
    --data data/output/benchmark/task_a \
    --data data/output/benchmark/task_a_voice \
    --stochastic-trials 0 --max-model-len 32768
```

The model YAML used was `configs/models_exp_a/gemma4_e4b.yaml` with only
`model.name` changed.
