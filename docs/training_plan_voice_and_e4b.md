# Training runbook: merge the new corpus, then train E4B and C2 on it

**Runs after `docs/generation_plan_voice_and_hard.md` delivers the two batches.**
Needs the GPU machine and the DVC remote.

## The experiment this is designed to answer

Two runs on **one** corpus, so only model size varies:

| run | model | question it answers |
|-----|-------|---------------------|
| C2-new | gemma-4-26B-A4B | does voice + harder data help the 26B? (versus C2-old's 0.7595) |
| E4B-new | gemma-4-E4B | what does shrinking the orchestrator cost? (versus C2-new) |

Training only E4B would confound model size with the corpus change. R15 and R16 are both cases of that. Each drew a conclusion from a
confounded comparison. Each had to withdraw it.

**Run E4B first.** It is a 4B model, so its run is short. It also exercises the whole chain first.
That chain is the merged corpus, `response_only` masking, the voice
`loss: false` flag, the per-modality split, and the audit.
If something is broken it costs a short run. This ordering is a recommendation,
not a constraint; reverse it if you prefer.

---

## 1. Receive the data

```bash
dvc pull data/output/sft/task_a data/output/sft/task_a_voice
python3 - <<'PY'
import json, glob, collections
c = collections.Counter()
for f in glob.glob("data/output/sft/task_a/*.jsonl"):
    for line in open(f):
        c[json.loads(line).get("complexity_level", "?")] += 1
v = sum(1 for f in glob.glob("data/output/sft/task_a_voice/*.jsonl") for _ in open(f))
print("text by level:", dict(sorted(c.items())), "total", sum(c.values()))
print("voice rows:", v)
PY
```

Expect roughly 7,000 text rows (5,543 existing plus 1,500 new L4/L5) and 3,000
voice rows.

---

## 2. Merge: clean, then split with the guard

```bash
python3 scripts/clean_task_a_sft.py \
    --input-dir data/output/sft/task_a_remediated \
    --input-dir data/output/sft/task_a_voice \
    --output-dir data/output/sft/task_a_cleaned --quiet

python3 scripts/split_task_a_sft.py \
    --assert-unmoved data/output/sft/task_a_splits \
    --no-backup
```

**`--assert-unmoved` is not optional.** R20: the splitter shuffles per modality
so an additive batch is additive, but adding rows to an existing modality group
still reshuffles that group. The guard matches by `user_turn_fingerprint`, since
`conversation_id` is not unique, and exits non-zero if any row changed split.

**If it exits non-zero, stop.** A moved row means the pinned held-out set in
step 3 will not rebuild. The link back to C2's 0.7595 is then gone.

---

## 3. Rebuild the pinned 206-row set and verify it

This is the comparability anchor. Do it before training, not after.

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
```

Must print `[verify] OK — 206/206 rows match`. It is text-only by construction.
**Never blend voice rows into it** — that would break the only comparison to
0.7595 that exists.

---

## 4. Create the two E4B config files

Neither exists yet.

`configs/models_exp_a/gemma4_e4b.yaml` — copy `configs/models/cat_bc/gemma4_e4b.yaml`
and set `category: "A"`, since E4B is filling the orchestrator role here.

`configs/training/sft_cat_a_e4b.yaml` — copy `sft_cat_a_c2.yaml` and change
**only** `output_dir` to `"sft_cat_a_e4b"`. Keep every hyperparameter identical:
`bf16`, `learning_rate: 5.0e-5`, `effective_batch_size: 8`, `num_epochs: 3`,
`max_seq_length: 8192`, `loss_mask: "response_only"`, LoRA rank 16.

Keeping them identical is the point: the comparison is model size, so nothing
else may vary. E4B is a 4B dense model, about 8 GB in bf16. It does not need the headroom the
26B does. But changing the recipe to exploit that would spend the experiment.

`output_dir` must be explicit. Without it, `sft.py` derives the checkpoint path
from the config filename. A run-stamped patched config then moves the output
silently. That is R13's follow-on defect. It wrote 1,770 steps of weights to
the wrong directory.

`loss_mask: "response_only"` is now **mandatory**, not a preference. `all_tokens`
has no per-token masking. It cannot honour the `loss: false` flag a voice
barge-in turn carries. It would train the model to emit the `<unspoken>` marker.
`sft.py` warns with `all_tokens_ignores_loss_flag` if this is wrong.

---

## 5. Run 1 — E4B, the canary

```bash
tmux new-session -d -s sft_e4b -c "$PWD"
tmux send-keys -t sft_e4b "env -u UNSLOTH_VLLM_STANDBY ./scripts/run_phase2_sft.sh \
    --model-config configs/models_exp_a/gemma4_e4b.yaml \
    --sft-config configs/training/sft_cat_a_e4b.yaml 2>&1 | tee /tmp/sft_e4b.log" Enter
```

**Check these before letting it run to completion:**

```bash
L=checkpoints/sft_cat_a_e4b/gemma-4-E4B-it/train.log
grep -a "max_length" $L                       # must be 8192, never 1024 (R16)
grep -a "all_tokens_ignores_loss_flag" $L     # must find nothing
grep -aoE "'loss': '[^']*'" $L | head -3      # must not be 0.0
```

R16 is why the first check matters: every Cat A SFT run before 2026-08-13
silently trained on a 1024-token window because TRL renamed the kwarg. The enriched system prompt alone is a median 3,016 tokens. So a 1024 window
under `response_only` masks everything, and the gradient is zero.

The corpus is larger now. Expect roughly 2,700 steps rather than C2's 1,767,
at 3 epochs and effective batch 8.

---

## 6. Run 2 — C2 on the new corpus

Only after run 1 completes cleanly.

```bash
tmux new-session -d -s sft_c2new -c "$PWD"
tmux send-keys -t sft_c2new "env -u UNSLOTH_VLLM_STANDBY ./scripts/run_phase2_sft.sh \
    --model-config configs/models_exp_a/gemma4_26b_a4b.yaml \
    --sft-config configs/training/sft_cat_a_c2_newcorpus.yaml 2>&1 | tee /tmp/sft_c2new.log" Enter
```

Create `sft_cat_a_c2_newcorpus.yaml` as a copy of `sft_cat_a_c2.yaml` with
`output_dir: "sft_cat_a_c2_newcorpus"`. Do **not** reuse `output_dir:
"sft_cat_a_c2"` — that would overwrite the checkpoint behind the 0.7595 baseline
this whole comparison rests on.

Budget from the previous run: see `checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/train.log`.

---

## 7. Evaluate — four numbers, never blended

```bash
for spec in "sft_cat_a_e4b/gemma-4-E4B-it e4b" \
            "sft_cat_a_c2_newcorpus/gemma-4-26B-A4B-it c2new"; do
  set -- $spec
  CKPT=$(ls -d checkpoints/$1/checkpoint-* | sort -t- -k2 -n | tail -1)
  for mod in text voice; do
    .venv-train/bin/python scripts/heldout_composite_audit.py \
        --checkpoint "$CKPT" \
        --data-dir data/output/heldout/cat_a_v2_test_not_in_v1 \
        --split test --n-prompts 206 --seed 42 --modality $mod \
        --output runs/audit/heldout_$2_$mod.json
  done
done
```

Read it as a 2x2:

| | text composite | reads as |
|---|---|---|
| C2-old (stored) | **0.7595** | the baseline |
| C2-new | ? | **the corpus effect** — did voice + harder data help the 26B? |
| E4B-new | ? | **the size effect** — what shrinking costs, measured against C2-new |

`voice_format_compliance` is reported over voice rows only, as a standalone
guardrail. **It is never folded into the composite** (R20). A mixed-modality
sample is labelled `MIXED-MODALITY` in the output. It is not comparable to
0.7595. Quoting it against the 0.75 bar would move the bar without a decision.

The pinned 206-row set is text-only, so the `--modality voice` audits need a
voice held-out sample built the same way. If none exists yet, report text only
and say so, rather than blending.

---

## 8. Register the results

```bash
dvc add checkpoints/sft_cat_a_e4b/gemma-4-E4B-it \
        checkpoints/sft_cat_a_c2_newcorpus/gemma-4-26B-A4B-it
dvc push
```

Then promote both to frozen `dvc.yaml` stages with declared deps, following
`task_a_dpo_gemma4_26b_a4b` (R22). A bare `.dvc` pointer declares no
dependencies. So the checkpoints can never go stale when the corpus changes.
That is exactly what this work does.

---

## Things that will bite

**A moved split row.** `--assert-unmoved` catches it. If it fires, the pinned
set will not rebuild and every comparison in section 7 is void.

**`max_length: 1024`.** Check it in the log, every run (R16).

**`all_tokens` with voice data.** Trains the model to speak the `<unspoken>`
marker.

**Reusing `output_dir: "sft_cat_a_c2"`.** Overwrites the 0.7595 baseline.

**Blending voice into the pinned 206-row set.** Destroys the only link to the
existing number.
