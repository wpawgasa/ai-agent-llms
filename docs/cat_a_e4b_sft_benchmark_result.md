# Cat A — gemma-4-E4B fine-tuned on corpus v3: the benchmark result

**2026-09-14.** gemma-4-E4B trained with the C2 recipe on `corpus/task-a-v3`
raises Phase 1 Task A quality from **0.6131 to 0.6678** (+0.055). Tool calling
is where it gained: its argument exact match is the best of every model
measured, Gemini included. Task completion did not move at all, and that one
term is most of what still separates it from gemma-4-12B and the Gemini
Flash-Lite models.

The held-out audit agrees. On the contamination-free corpus-v3 sets, SFT
raises the composite from **0.7127 to 0.7828 on text** and **0.7080 to 0.7894
on voice**, both significant in a row-paired test (section 3.1).

Neither number is **comparable to C2's 0.7595**, which is bound to a different
206-row set.

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
| State sequence accuracy † | 0.7884 | 0.8713 | +0.0829 | 0.8680 | n/a | n/a |
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

### 3.1 Held-out audit (added 2026-09-14)

The Phase 1 benchmark replays conversations that were generated for ranking,
not held out from training. The held-out audit uses corpus-v3 test
conversations that appear in neither the v2 train nor the v2 validation split,
and scores the untrained E4B and `checkpoint-3168` on exactly the same rows.

| | |
|---|---|
| Sets | `cat_a_v3_test_not_in_v2` (text) and `cat_a_v3_test_voice` (voice), tag `derived/task-a-heldout-v3` |
| Rows scored | 303 text, 145 voice (one sampled assistant turn per conversation) |
| Settings | `scripts/heldout_composite_audit.py --split test --seed 42`, greedy, 4-bit loading, text and voice as separate runs |
| Composite | `0.4 × state_acc + 0.4 × tool_f1 + 0.2 × task` |

| | Text: untrained | Text: SFT | Voice: untrained | Voice: SFT |
|---|---|---|---|---|
| **Composite** | 0.7127 | **0.7828** | 0.7080 | **0.7894** |
| State accuracy | 0.8911 | 0.9835 | 0.9172 | 0.9862 |
| Tool F1 (all rows) | 0.8212 | 0.9076 | 0.7908 | 0.9218 |
| Task | 0.1386 | 0.1320 | 0.1241 | 0.1310 |

**The improvement is real.** Rows are paired by `row_index`, and the ground
truth is identical in every one of them across the two audits.
`scripts/stratify_heldout_audit.py` gives:

| | Composite delta | 95% CI (10k bootstrap) | SFT better / worse | Sign test |
|---|---|---|---|---|
| Text | **+0.0702** | [+0.0475, +0.0944] | 60 / 14 (229 tied) | p = 6.2e-08 |
| Voice | **+0.0814** | [+0.0492, +0.1159] | 31 / 10 (104 tied) | p = 0.0015 |

State accuracy and tool F1 are each significant on both sets as well.

**Where it improved.** About 69% of rows need no tool call, and staying silent
earns full tool F1 on them, so the aggregate hides tool-calling ability. The
stratified rows show it:

| | Text: untrained | Text: SFT | Voice: untrained | Voice: SFT |
|---|---|---|---|---|
| Rows needing a tool call | 94 | 94 | 46 | 46 |
| Tool F1 on those rows | 0.6578 | 0.7553 | 0.7971 | 0.8188 |
| Emits no tool call | 6.4% | 1.1% | 8.7% | 0.0% |
| Exact call (name + arguments) | 53.2% | **71.3%** | 58.7% | **84.8%** |
| Spurious call on rows needing none | 10.5% | 2.4% | 21.2% | 3.0% |
| Stays in the same state when it should advance | 25.4% | **1.1%** | 10.0% | **2.2%** |

- **The tool-call stay convention was learned.** Corpus v3 annotates a
  tool-calling turn as staying in its state. The untrained model over-applies
  staying: it wrongly stays put on 25.4% of turns that should advance. SFT cuts
  that to 1.1%, while still staying correctly on 98–100% of turns that should.
- **Arguments improved most.** Exact calls on rows needing a tool rose by 18
  points on text and 26 on voice. This matches the Phase 1 benchmark, where
  argument exact match was the largest gain.
- **Spurious tool calls nearly disappeared.** The untrained model calls a tool
  on 10.5% (text) and 21.2% (voice) of turns that need none.
- **Voice formatting is a guardrail, not part of the composite:** voice format
  compliance 0.9586 → 0.9724.

**The task column is not a completion measure here.** It scores whether a
single sampled turn reaches the conversation's terminal state, so it is low for
any model on mid-conversation turns (CLAUDE.md R17). Its flat value says nothing
about the benchmark's unchanged task completion in section 4.3.

**Remaining error is arguments.** On rows that need a tool call, 28% (text)
and 15% (voice) get the right tool name with wrong arguments, while the right
tool name is chosen 99–100% of the time. The same bottleneck R23 found for the 26B.

**These composites form a new scale.** They are not comparable to C2's 0.7595,
and no fine-tuned 26B has been scored on these sets. To place C2 on this scale,
audit `model/sft-gemma4-c2-on-task-a-v2` against these sets with the same
settings — that needs the 26B weights, so not on this machine.

Audit files: `runs/audit/heldout_e4b_base_v3text.json`,
`heldout_e4b_base_v3voice.json`, `heldout_e4b_ckpt3168_v3text.json`,
`heldout_e4b_ckpt3168_v3voice.json`.

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

State sequence accuracy rose 0.7884 → 0.8713, level with the untrained 12B
(0.8680). Per-turn transition accuracy rose 0.4102 → 0.4853, and the
invalid-transition rate fell 0.1517 → 0.0716.

† **Corrected 2026-09-16.** The originally reported whole-run state metrics
(sequence 0.5517 → 0.6125, transition 0.2192 → 0.2613, invalid 0.42–0.50 for
every model) scored text conversations against voice ground truth, because the
two strata reuse conversation ids. The values above are recomputed from the
logged replies. The Gemini runs cannot be recomputed from their logs, so their
state metrics are shown as n/a. Blended quality, the per-stratum scores, task
completion and tool metrics were not affected. See
[the text-set and id-collision findings](cat_a_benchmark_text_set_convention_mismatch.md),
which also show that E4B SFT's flat task completion hides a text-set effect:
most of its text completion drop is in conversations written before the
tool-call stay rule.

### 4.6 Latency is not comparable

E4B-SFT has the lowest per-turn latency (425.5 ms average) and E4B the next
lowest, but both were served locally on an H100 while Gemini went through an
API, and the 12B ran on a different vLLM version. Treat the latency columns in
the result files as descriptive, not as a ranking.

---

## 5. What this result does not show

- **Nothing here is comparable to C2's 0.7595.** The benchmark in section 3
  is a different measure, and the held-out audit in section 3.1 uses the
  corpus-v3 sets, a different scale from C2's 206-row set.
- **The held-out audit scores single turns.** One sampled assistant turn per
  conversation, 303 text and 145 voice. It does not test whether a whole
  conversation finishes correctly.
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

1. **Do not start single-turn GRPO, DPO or RFT on this checkpoint.** The
   go/no-go probe (`scripts/rft_headroom_probe.py`) returned **NO_GO** on
   2026-09-15, against the gates fixed in advance in
   `docs/grpo_viability_investigation.md` section 4. Settings: checkpoint-3168,
   500 prompts from `data/output/grpo/task_a` train, 8 samples each at
   temperature 0.8, top-p 0.95, seed 42; 4.4 h on the H100 NVL.

   | | E4B SFT | Gate | C2 26B (R23) |
   |---|---|---|---|
   | `frontier_frac` | **0.078** | ≥ 0.15 for RFT; < 0.10 is NO_GO | 0.052 |
   | `mean_headroom` | **0.0183** | ≥ 0.03 | 0.0177 |
   | `frac_collapsed_groups` | **0.746** | < 0.50 for GRPO | 0.876 |
   | `median_reward_std` | **0.0000** | ≥ 0.05 | 0.0000 |

   Best-of-8 beats greedy on only 7.8% of prompts, so RFT has little to distil,
   and 74.6% of groups score all eight samples identically, so GRPO gets no
   gradient from most of a batch. E4B leaves slightly more room than C2 but
   fails every gate. The prompt set is 27.7% voice while C2's was text only, so
   the two rows are not strictly comparable. As for C2, the open direction is
   multi-turn. Raw output: `runs/audit/rft_headroom_e4b_ckpt3168.json`.
2. **Score C2 on the corpus-v3 held-out sets.** The held-out audit is done for
   E4B (section 3.1). Auditing `model/sft-gemma4-c2-on-task-a-v2` on the same
   sets and settings would put the best 26B checkpoint on the same scale. It
   needs the 26B weights, so it has to run on another machine.
3. **Train the 26B C2-v3 arm on a machine with enough disk.** Only that
   completes the size comparison this run was built for.
4. **Investigate task completion.** Start from the 252 conversations: which end
   on a non-terminal state, and at which complexity level.
   `scripts/run_exp_a_per_level.sh` gives the per-level split this document
   lacks.
5. **Fix the tooling:** put the verified merge (including the 54-tensor
   completion) into `training/merge_adapter.py`, and set `WANDB_PROJECT` from
   the SFT config. The verified merge script is now committed as
   `scripts/merge_lora_verified.py`.

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

The held-out audit (section 3.1). The sets are gitignored and rebuilt from two
tagged corpora; `git tag -n60 derived/task-a-heldout-v3` holds the full recipe,
input hashes and file checksums.

```bash
# Rebuild the sets, verifying each against the stored checkpoint-3168 audit
dvc fetch -T data/output/sft/task_a_splits
.venv-train/bin/python scripts/materialize_dvc_lineage.py \
    --dir-hash 21e33e25d7bbae63e97a89029b3b4705 --out /tmp/v2_splits
.venv-train/bin/python scripts/build_heldout_clean_set.py \
    --candidate-split data/output/sft/task_a_splits/test.jsonl \
    --exclusion-split /tmp/v2_splits/train.jsonl \
    --exclusion-split /tmp/v2_splits/validation.jsonl \
    --out-dir data/output/heldout/cat_a_v3_test_not_in_v2 \
    --modality text --expect-clean 304 --n-prompts 304 --seed 42 \
    --verify-against runs/audit/heldout_e4b_ckpt3168_v3text.json
# ... and the same with --modality voice --expect-clean 146 --n-prompts 146
#     --out-dir data/output/heldout/cat_a_v3_test_voice
#     --verify-against runs/audit/heldout_e4b_ckpt3168_v3voice.json

# Audit a checkpoint; use --checkpoint unsloth/gemma-4-E4B-it for the baseline
env -u UNSLOTH_VLLM_STANDBY .venv-train/bin/python scripts/heldout_composite_audit.py \
    --checkpoint checkpoints/sft_cat_a_e4b/gemma-4-E4B-it/checkpoint-3168 \
    --data-dir data/output/heldout/cat_a_v3_test_not_in_v2 \
    --split test --n-prompts 304 --seed 42 --modality text \
    --output runs/audit/heldout_e4b_ckpt3168_v3text.json

# Paired comparison, untrained model first
.venv-train/bin/python scripts/stratify_heldout_audit.py \
    base=runs/audit/heldout_e4b_base_v3text.json \
    sft=runs/audit/heldout_e4b_ckpt3168_v3text.json
```
