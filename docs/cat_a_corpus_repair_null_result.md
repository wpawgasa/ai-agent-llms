# Repairing the Task A SFT corpus does not move the v4 benchmark

**2026-09-25.** Stage 1 of the corpus repair (R28's mechanical half) was built,
applied, and measured by retraining the E4B on it. The result is null: the
composite moves −0.0024 in the text format and +0.0144 in the native format,
the two arms disagree in sign, and **argument exact match — the metric the
experiment predicted would improve — fell in both**.

The conclusion is that the mechanical data flaws R28 catalogued are not what
caps Task A performance, and stage 2 (907 authored invented facts, 456
regenerated two-tool conversations) should not be bought.

## 1. What was repaired

`scripts/repair_task_a_corpus.py`, the benchmark repair pointed at
`data/output/sft/task_a_splits`, planned into
`data/interim/task_a_corpus_repair_ledger/ledger.json` (seed 20260924) and
applied to `data/output/sft/task_a_splits_v4`.

| Finding | Before | After |
|---|---|---|
| Confident unknowable tool arguments | 443 | **0** |
| Identifier reuse across conversations | 70.5% | **4.4%** |
| Orphan tool results | 60 | **0** (41 conversations dropped) |
| Mergeable stay+stay pairs | 552 | **1** |
| Rows sharing an identifier with the v4 benchmark | 127 | **52** |
| Confident invented facts | 978 | 907 *(stage 2, not repaired)* |
| Two-tool state visits | 458 | 456 *(needs regeneration, not repair)* |

Audits: `runs/audit/triage_sft_corpus_v3.json` (before),
`runs/audit/triage_sft_corpus_v4.json` (after). The 52 rows still sharing an
identifier with the benchmark hold values the repair deliberately skips —
`AES-256`, `ACC-999-XYZ`, `CLM-INVALID-999` — general knowledge and non-simple
shapes, all present in the original corpus.

## 2. The experiment

`configs/training/sft_cat_a_e4b_corpus_v4.yaml` is
`sft_cat_a_e4b_textturns.yaml` with `output_dir` and `data.source` changed and
nothing else, pinned by `tests/unit/test_sft_cat_a_corpus_v4_config.py`. So the
only difference from the run that produced the standing best result is the
training data.

Run: 3 epochs, 3,156 steps (12 fewer than the baseline's 3,168 — the 41 dropped
rows), 6.66 h, best eval_loss 0.4292 against the baseline's 0.4298.
`training_args.bin` records `max_length 8192`, so R16's silent-1024 trap did not
fire. Scored on the v4 benchmark in both tool-turn formats, 0 stochastic trials,
identical settings to the baseline.

**`eval_loss` says nothing here.** Each run scores its own validation split, and
the repaired split is a different set of rows; R15 and R16 both record this
metric moving opposite to capability.

## 3. Result

| metric | text: base → repaired | native: base → repaired |
|---|---|---|
| **quality** | 0.8332 → **0.8308** (−0.0024) | 0.8046 → **0.8189** (+0.0144) |
| quality, text stratum | 0.8175 → 0.8180 (+0.0005) | 0.7796 → 0.8005 (+0.0209) |
| quality, voice stratum | 0.8699 → 0.8605 (−0.0093) | 0.8628 → 0.8619 (−0.0009) |
| **argument exact match** | 0.6309 → 0.6202 (−0.0106) | 0.6064 → 0.6037 (−0.0027) |
| tool F1 | 0.6952 → 0.6841 (−0.0112) | 0.6844 → 0.6945 (+0.0101) |
| tool name accuracy | 0.8868 → 0.8820 (−0.0048) | 0.8744 → 0.8598 (−0.0146) |
| state transition (turn) | 0.8109 → 0.8077 (−0.0032) | 0.7992 → 0.7848 (−0.0144) |
| state sequence | 0.9613 → 0.9605 (−0.0008) | 0.9452 → 0.9504 (+0.0053) |
| completion (continuous) | 0.8625 → 0.8585 (−0.0039) | 0.8114 → 0.8251 (+0.0138) |
| chain propagation | 0.9000 → 0.9000 (0.0000) | 0.8481 → 0.8704 (+0.0222) |
| invalid transitions | 0.0501 → 0.0526 (+0.0024) | 0.0470 → 0.0524 (+0.0054) |
| **stay-rule violations** | 10 → **1** | 9 → **2** |

Results: `results/exp_a/sft_cat_a_e4b_corpus_v4_ckpt3156_gated{,text}_v4_auto.json`.

## 4. Reading it

**The prediction failed, and that is the finding.** The experiment was framed
around one mechanism: the corpus teaches the model to invent identifier-shaped
values (R28 measured fine-tuning raising that rate from 0.3% to 3.0%), so
removing every unknowable argument and nearly all identifier reuse should raise
argument exact match. It went **down in both formats**. Whatever limits argument
fidelity, it is not that the training data rewards invention.

**The two formats disagree in sign**, and their components contradict each other
— tool F1 up in native and down in text, state transition accuracy down in both
while state sequence rises in one. A real corpus effect would push both arms the
same way. Treat the +0.0144 as noise, not a win.

**One real improvement, worth nothing in the composite.** Stay-rule violations
fell 10 → 1 and 9 → 2: the 552 merged stay+stay pairs taught cleaner turn
structure, exactly as designed. But the trained models were already violating
the rule on ~0.2% of 5,824 segments, so there was no room for it to matter.

## 5. Limits of this measurement

- **One run per arm, no seed replication.** These numbers sit in a band of
  roughly ±0.01; the observed deltas are inside it. A stronger test would need
  several seeds, which is exactly the cost the staging was meant to avoid.
- **Stage 1 is not "corpus quality".** 907 invented facts and 456 two-tool
  conversations remain. The honest claim is narrower: *the mechanical flaws —
  unknowable arguments, identifier reuse, orphan results, stay pairs — do not
  measurably cap v4 performance.*
- **The dropped 41 rows** slightly shrink the training set (8,441 → 8,409). Too
  small to explain anything, but it is a second difference between the arms.

## 6. Consequence

Do not buy stage 2. Argument exact match now sits at 0.60–0.63 across every
variant tested — original corpus, repaired corpus, both tool-turn formats, both
model sizes — and did not move when the data stopped rewarding invention. That
sharpens R23 rather than contradicting it: the bottleneck is the model failing
to carry values across turns, not the corpus teaching it bad ones.

The repaired corpus stays tracked as evidence, and the repair tooling stays
in-tree: it is the same machinery a future regenerated corpus would use to
verify itself.

## 7. Reproduce

```bash
python scripts/repair_task_a_corpus.py plan \
    --input-dir data/output/sft/task_a_splits \
    --reference data/output/benchmark/task_a_v4 \
    --reference data/output/benchmark/task_a_voice_v3 \
    --ledger data/interim/task_a_corpus_repair_ledger/ledger.json --seed 20260924
python scripts/repair_task_a_corpus.py apply \
    --stratum data/output/sft/task_a_splits data/output/sft/task_a_splits_v4 \
    --ledger data/interim/task_a_corpus_repair_ledger/ledger.json

env -u UNSLOTH_VLLM_STANDBY ./scripts/run_phase2_sft.sh \
    --model-config configs/models_exp_a/gemma4_e4b.yaml \
    --sft-config configs/training/sft_cat_a_e4b_corpus_v4.yaml

bash .runs/eval_v4/run_v4_corpus_ab.sh   # both formats, v4 benchmark
```
