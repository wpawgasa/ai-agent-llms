# Cat A: single-turn RL is closed. NO_GO, measured.

**2026-08-31.** `scripts/rft_headroom_probe.py` returned **NO_GO** on the C2
checkpoint. Every pre-registered gate fails, none of them marginally. This
closes R18 (GRPO) and R22 (DPO) with a measurement rather than a third failed
training run.

---

## 1. The numbers

500 prompts x 8 completions, `--split train`, temperature 0.8, top-p 0.95,
against `checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767`.

| gate | measured | threshold | |
|------|----------|-----------|---|
| `frontier_frac` | **0.052** | >= 0.15 for GO_RFT | fail |
| `mean_headroom` | **0.0177** | >= 0.03 for GO_RFT | fail |
| `frac_collapsed_groups` | **0.876** | < 0.50 for GRPO revival | fail |
| `median_reward_std` | **0.0000** | >= 0.05 for GRPO revival | fail |

`frontier_frac` 0.052 is below even the 0.10 NO_GO line, so the verdict is not
a near miss.

Thresholds were fixed in `docs/grpo_viability_investigation.md` §4 **before**
these numbers existed, which is why the verdict cannot be argued after the
fact. Raw output: `runs/audit/rft_headroom_c2.json`.

---

## 2. What each number rules out

**`frontier_frac = 0.052`** — best-of-8 beats greedy on 5.2% of prompts. RFT
can only distil what best-of-N finds, so there is almost nothing to distil.

**`median_reward_std = 0.0000`** — the *median* group has exactly zero
variance. More than half of all prompts produce eight completions that score
identically.

**`frac_collapsed_groups = 0.876`** — 87.6% of groups tie completely. GRPO
derives its advantage from within-group variance, so 87.6% of every batch
contributes exactly zero gradient.

---

## 3. Two of my own earlier claims, corrected

**"DAPO's dynamic sampling is the standard fix we never ran."** True that we
never ran it (verified: zero matches in TRL 1.0.0's `grpo_trainer.py`). False
that it would have helped. Dynamic sampling drops zero-variance prompts and
resamples to refill the batch. At 87.6% collapsed and a median group variance
of exactly zero, it would discard the overwhelming majority and leave a
remnant too small and too weakly differentiated to train on. Naming a
technique is not the same as measuring that it applies.

**"The ties are because the policy repeats itself."** Refuted here and already
refuted in R18, which recorded 8 of 8 genuinely distinct trajectories (lengths
77-113) scoring identically. `frontier_frac = 0.052` confirms it from the
other direction: the model is not hiding better answers that sampling could
surface. Sampling harder does not help because there is nothing to select.

---

## 4. Where the remaining errors actually are

The full-split mine (2,487 prompts, every train conversation the sampler can
reach) gives the largest sample yet of C2's residual failures:

| failure kind | count | share |
|--------------|-------|-------|
| **wrong arguments** | **409** | **77.5%** |
| no tool call | 110 | 20.8% |
| state error | 5 | 0.9% |
| wrong tool name | 4 | 0.8% |

Plus 1,518 identical to gold and 441 otherwise acceptable — a 21.2% error rate.

**Tool selection is solved**: 4 wrong tool names in 2,487 attempts, 0.16%.
**Argument fidelity is the entire remaining problem**, at 77.5% of errors.

This matters for direction. Wrong arguments come from failing to carry values
across a tool chain, which is a **multi-turn** property. A single-turn reward
scoring one turn against its own ground truth cannot see it, which is
consistent with `frontier_frac` being near zero: within a single turn, the
model has no better answer available to find.

---

## 5. Consequences

**C2 (`checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767`, 0.7595)
is the Cat A result.** It clears the pre-registered >= 0.75 bar. Three attempts
to improve on it — GRPO, DPO, and a re-mined DPO — are now closed, the last one
before it was run.

**Do not attempt another single-turn preference or policy-gradient variant on
this prompt distribution.** The probe measures the property all of them depend
on, and it is absent.

**Do not generate more data of the current shape.** The corpus is 67.3% L1-L3
(`chain_depth` <= 2). More easy rows lower the informative fraction further.
`data/output/preference/task_a/model_negatives.jsonl` now holds 528 mined
negatives (backups of the 51- and 67-row versions sit beside it), but they feed
a path that is closed.

**The open direction is multi-turn**, which is what the probe's own verdict
string recommends and what §4's error distribution independently points at.
That is a scoping question, not a next command, and it should start from a
measurement of how often argument errors propagate across a chain rather than
from another training run.

---

## 6. Reproduce

```bash
.venv-train/bin/python scripts/rft_headroom_probe.py \
    --checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767 \
    --split train --n-prompts 500 --n-completions 8 \
    --output runs/audit/rft_headroom_c2.json
```

About 5.7 hours on an H100 NVL. Budget from the measured rate (~48 min per
sampling pass, 8 passes plus greedy) rather than calling it "a short run",
which is what I did and it was wrong by roughly 6x.
