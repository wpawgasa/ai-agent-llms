# Cat A State-Annotation Convention Review

**Date:** 2026-07-29, updated 2026-07-30 (§6.6)
**Scope:** `docs/superpowers/plans/2026-07-22-cat-a-state-accuracy-factorial.md` §12.4 steps 1–2
**Artifacts:** `runs/preflight/heldout_audit_C0_ckpt500_goldenval.json`,
`runs/preflight/composite_decomposition_goldenval.json`,
`runs/preflight/heldout_audit_C0_ckpt1770_goldenval.json`,
`runs/preflight/composite_decomposition_ckpt1770.json`,
`runs/preflight/selfloop_habit_ckpt1770_vs_ckpt500.json`,
`scripts/analyze_composite_decomposition.py`, `scripts/analyze_selfloop_habit.py`
**Checkpoints:** all measurements on the `sft-gemma4-v3` lineage
(`dvc.lock` `d5438dced5a25f54af0d73e8569c6483.dir`, 47 files / 560,269,322 B), staged at
`/tmp/sft_v3`. **Not** `checkpoints/sft_cat_a/gemma-4-26B-A4B-it`, which holds v2 — see §6.6.

---

## 0. TL;DR

The factorial's premise does not survive contact with the data.

§12.4 step 1 asked which composite term produces the 0.5 spike, and warned that if it were
tool-F1 rather than state accuracy, "C1/C2 are both mis-targeted." The answer is **state
accuracy** — so on the surface the factorial is aimed correctly. But decomposing *why* state
fails shows the model is **not choosing wrong destinations**. It is:

- correct on the `from` state **199 times out of 200** on failing rows,
- **87.8%** accurate when the workflow genuinely advances (already above the 0.85 target),
- **12.3%** accurate when the workflow should stay put.

It emits a self-loop on **4.8%** of turns where gold expects **41.6%**. The single behaviour
carrying the entire deficit is: **the model advances the state while calling a tool, instead of
staying put until the tool result arrives.**

That behaviour is what the system prompt tells it to do. Across 150 prompts and 1,920 state
blocks, **0.00%** document a self-loop; every block lists only forward edges, and the audited
turn's block literally reads `- on success: proceed to [CHECK_VISA_REQUIREMENTS]`.

Neither C1 (loss masking) nor C2 (decision-balance reweighting) addresses a rule that was never
stated. See §6 for what does.

**All three cheap paths are now measured, and all three fail.** A perfectly applied rule caps at
0.7609 (§6.1); prompting the rule is null (§6.5); and training 3.5× longer is null — ckpt-1770
(epoch 3.0) emits self-loops on **6.4%** of turns vs the 41.6% gold expects, gate composite 0.6579
against the 0.80 bar (§6.6). The habit is not an under-training artifact. **Corpus regeneration via
`generate_workflows.py` plus a retrain is the remaining path.**

---

## 1. Provenance work done first (why the earlier numbers were unusable)

Before any measurement, the workspace was found to hold the **pre-R12 lineage throughout** —
data and checkpoints — with no lock-file signal:

| Artifact | Workspace held | Should have been |
|---|---|---|
| `data/output/sft/task_a_splits` | 9,131 / 1,074 / 538 | 4,716 / 554 / 279 |
| `data/output/sft/task_a_cleaned` | 10,743 convs, 133 files | 5,549 convs, 5 files |
| `data/output/grpo/task_a` | 22.4% val / 17.2% train malformed roles | 0% |
| `checkpoints/sft_cat_a/…` | `sft-gemma4-v2` | `dvc.lock` claimed v3 |

The 9,131/1,074 split is an exact match for `sft-gemma4-v2`'s `train.log` — i.e. the corpus
ckpt-1000 trained on. The malformed-role rates match CLAUDE.md R12's recorded **pre-fix** figures
to the decimal.

**Resolution:** both SFT directories were deleted and rebuilt from raw via `dvc repro -s -f
task_a_sft_clean` then `task_a_sft_splits`. The rebuild reproduced the recorded hashes exactly
(`dvc.lock` unchanged), all 8 files md5-identical to the copy fetched from the GCS remote, and
0/5,549 conversations carry a malformed role. Tagged **`task-a-sft-v1`** (commit `6a50272`).

Every number below is measured on that tagged corpus.

### 1.1 Eval data source corrected

The audit had been reading `data/output/grpo/task_a`, per spec §9. That directory is **not
separate data** — `filter_grpo_data.py` derives it from `task_a_splits` by filtering to L3–L5,
and its own docstring notes `test.jsonl` is deliberately excluded, "reserved for final
evaluation." `_load_grpo_jsonl(data_dir, split)` is a generic slicer that reads any
conversation-format JSONL.

So the audit now runs directly on `data/output/sft/task_a_splits`, split `validation`. This
removes the stale GRPO directory from the loop entirely and raises the sample ceiling from 289 to
**548** unique conversations — which incidentally supplies the statistical power §8.2 asked for
and §8.3 prescribed unreachably (the sampler dedupes by conversation, so `--n-prompts 2943`
silently caps at the number of unique conversations).

---

## 2. Headline numbers (C0 ckpt-500, `task-a-sft-v1` validation, n=548)

| Component | Measured | Spec §4.4 | Target |
|---|---|---|---|
| gate composite | **0.6527** | 0.7271 | 0.80 |
| state accuracy | **0.5639** | 0.6866 | 0.833 |
| tool-F1 (211 tool-expected rows) | 0.4550 | 0.4623 | 0.85 |
| abstention (337 zero-tool rows) | 0.9228 | 0.9494 | ~0.95 |

Not comparable to §12's 0.6720 — different split, all levels L1–L5 rather than L3–L5, 548 rows
rather than 289. These are the first Cat A numbers measured on hash-verified, tagged data.

---

## 3. §12.4 step 1 answered: the failing term is state

Of the **179 rows at exactly 0.5** (32.7%), §12.7's predicted 1/0 split between the two terms
holds, and resolves 3:1 toward state:

| Pattern | Rows | Share |
|---|---|---|
| **state=0, tool=1** | **135** | **75.4%** |
| tool=0, state=1 | 44 | 24.6% |

Confirmed independently by counterfactual (composite is linear in each per-row term):

```
tool  -> 1.00, state unchanged : 0.7807   fail
state -> 1.00, tool unchanged  : 0.8688   PASS
```

**A perfect tool term cannot clear the bar; a perfect state term can.** State is the binding
constraint.

---

## 4. But the mechanism is not destination selection

§1 of the spec diagnoses "destination-selection, bidirectional." The data says otherwise.

### 4.1 Split by what gold expects

| Gold | Rows | State accuracy |
|---|---|---|
| **ADVANCE** (X → Y) | 320 (58.4%) | **0.8781** |
| **SELF-LOOP** (X → X) | 228 (41.6%) | **0.1228** |

Of the 200 failed stay-rows, **199 (99.5%) had the correct `from` state** and predicted an
advance instead of staying. The model is not lost and does not pick wrong destinations — on
advance turns it is already above the 0.85 component target.

### 4.2 Emission rates

| Source | Self-loop rate |
|---|---|
| Gold (validation) | 41.6% |
| Training text (`[STATE:]` annotations) | 37.1% (20,496 of 55,235) |
| **Model output** | **4.8%** (31 of 640) |

The training corpus is *not* missing self-loops. The model saw 20,496 of them and still emits
almost none.

### 4.3 Where the deficit concentrates

| Gold expects | Rows | State acc | Model says "stay" |
|---|---|---|---|
| advance + no tool | 273 | 0.894 | 0.4% |
| advance + tool | 47 | 0.787 | 2.1% |
| stay + no tool | 64 | 0.297 | 29.7% |
| **stay + tool** | **164** | **0.055** | **5.5%** |

One bucket — 30% of the eval — carries almost the whole deficit: the **tool-execution turns**.

---

## 5. Root cause: an unstated convention the prompt actively contradicts

### 5.1 The gold pattern is a two-turn cycle

Full gold trajectory for `L4_049_9` (travel, L4), the conversation behind audited row 34:

```
 2  VERIFY_TRAVELER → SEARCH_OPTIONS        (no tool)     enter the state
 3  SEARCH_OPTIONS  → SEARCH_OPTIONS        search_flights          <- execute, STAY
 4  SEARCH_OPTIONS  → CHECK_VISA_REQUIREMENTS (no tool)              <- hand off, ADVANCE
 5  CHECK_VISA_REQ  → CHECK_VISA_REQ        check_visa_requirements <- execute, STAY
 6  CHECK_VISA_REQ  → COLLECT_VISA_DOCS     (no tool)                <- hand off, ADVANCE
10  PROCESS_BOOKING → PROCESS_BOOKING       book_reservation
11  PROCESS_BOOKING → PAYMENT_PROCESSING    (no tool)
```

Enter → call the state's tool while staying → advance on the next turn after the result. In this
conversation the correlation is perfect across all 18 turns.

### 5.2 The model collapses the cycle into one turn

Audited row 34, verbatim model output:

```
[STATE: SEARCH_OPTIONS → CHECK_VISA_REQUIREMENTS]
<tool_call>{"name": "search_flights", "arguments": {"origin": "BKK",
  "destination": "NRT", "departure_date": "2025-05-15",
  "passengers": 1, "cabin_class": "economy"}}</tool_call>
"…while the system searches for flights, may I check the visa requirements"
```

The tool call is flawless — correct tool, every argument right (`tool_f1 = 1.0`). It performed
turn 3's *action* with turn 4's *label*. It is not wrong about the destination; it is **one turn
early**, committing to a state before the evidence justifying it exists.

Row 16 (healthcare) is the same shape: the model asks for the patient ID — correct behaviour for
`VERIFY_PATIENT` — while labelling `VERIFY_PATIENT → CHECK_ELIGIBILITY`.

### 5.3 The system prompt instructs exactly this

The audited conversation's own system prompt:

```
### [SEARCH_OPTIONS]
  tool available: search_flights
  - เมื่อสำเร็จ: ดำเนินการต่อที่ [CHECK_VISA_REQUIREMENTS]     ("on success: proceed to …")
```

Scanned 150 system prompts / **1,920 state blocks**: **0 (0.00%)** list themselves as a
destination. Every workflow script documents forward edges only. There is no stated rule, anywhere,
for when to remain in a state.

So the model receives an explicit instruction ("on success, advance") and an unexplained 37%
counter-example rate in the demonstrations. It follows the explicit instruction. Its transitions
are **legal** — `transition_legality` is not the issue.

### 5.4 The convention is not deterministic

Corpus-wide, "tool ⇒ stay" holds only ~4 times in 5:

| Turn type | Self-loop rate (train / validation) |
|---|---|
| Calls a tool | 78.6% / 80.7% |
| Calls no tool | 18.2% / 18.0% |

Tool repetition does not separate the exceptions (stay rate 78.7% on first use vs 85.7% on a
repeated tool), so they are not a class that can be stated as a refined rule. They appear to be
genuine annotation inconsistency.

---

## 6. What this means for the ladder

### 6.1 A stated rule is necessary but NOT sufficient

Simulating a model that perfectly applies "if I call a tool, stay; else advance":

| Scenario | State | Composite |
|---|---|---|
| today | 0.5639 | 0.6527 |
| perfect `tool ⇒ stay` | 0.7792 | **0.7609** |
| bar | — | 0.80 |

It closes roughly half the gap and **still fails**, because 47 of 211 tool-expected rows are
gold-advance and become new errors. The 21% corpus ambiguity is a hard ceiling on any
rule-following policy.

### 6.2 C1 and C2 are mis-targeted, for a different reason than §12.4 anticipated

- **C1** (`response_only` masking) changes which tokens receive gradient. It cannot teach a rule
  that is absent from the input and contradicted by the instruction.
- **C2** (stay/advance decision-balance reweighting) would correct the *base rate* while giving
  the model no way to know *which instance* should stay — the deciding information is not in its
  context.

### 6.3 Recommended order

1. **Prompt experiment (cheap, no retraining).** State the stay-condition in the workflow script
   built by `data/system_prompt.py::build_enriched_system_prompt`, re-run the audit on the
   existing checkpoint. Expected ~0.76 — short of the bar, but it measures the one thing that
   cannot be predicted: whether an SFT'd policy still responds to prompt instruction at all. If it
   ignores the rule, no corpus change helps without retraining either.
   *Caveat:* this introduces a train/eval prompt mismatch. If it works, the rule belongs in the
   SFT prompt and the model should be retrained with it.
2. **Corpus revision to clear the bar.** Make `tool ⇒ stay` deterministic: a tool-calling turn
   annotates `X → X`, the advance moves to the following turn. This is also the behaviourally
   correct choice — the inverse convention (tool + advance in one turn) would score well and would
   bless committing to a state before the result arrives, which is a routing bug in a real
   orchestrator. Would become `task-a-sft-v2` off `task-a-sft-v1`.
3. **Open, gating the relabel:** sample the 47 gold-advance-with-tool rows to check they are
   genuinely inconsistent rather than a legitimate class, and confirm the turn *prose* does not
   already read as a hand-off (in which case annotation and text would disagree after a blanket
   relabel).

### 6.4 The 47 exceptions, read (2026-07-29)

They are **not** a semantically distinct class. They are the same two-turn move with the advance
annotated on the other turn:

```
canonical :  X → X  + tool   →  [tool result]  →  X → Y
exception :  X → Y  + tool   →  [tool result]  →  Y → Y
```

Row 229 (`L2_067_9`) is exactly this: `VERIFY_IDENTITY → AUTHENTICATE` + `verify_identity`, then
the OTP result, then `AUTHENTICATE → AUTHENTICATE`. Same states, same tool, same two turns.

But this shift accounts for only **59.6%** of them. Corpus-wide, for tool-calling turns:

| Tool turn | Following assistant turn | Count | Share |
|---|---|---|---|
| stay | advances | 1,248 | 74.4% |
| stay | stays | 429 | 25.6% |
| ADVANCE | stays | 239 | 59.6% |
| ADVANCE | advances | 162 | 40.4% |

**Consequence for the relabel:** the 162 turns that advance *again* on the next turn cannot be
blanket-relabelled to `X → X` — the following turn begins at `Y`, a state the conversation would
then never have entered, breaking the chain. So `task-a-sft-v2` is **not** a find-and-replace over
annotations; the trajectory must be re-derived per conversation with a consistent rule, which
points at `generate_workflows.py` rather than a cleanup pass.

### 6.5 Prompt experiment result — NULL. A prompt-only fix is not viable.

`STAY_RULE` (as of 2026-08-02 on by default as rule 2 of `FORMAT_RULES`, with `TASK_A_STAY_RULE=0`
as the opt-out that reproduces the frozen v1 prompt; it was opt-in via `TASK_A_STAY_RULE=1` when
this experiment ran) was added stating the policy and explicitly
reinterpreting the workflow script's `- on success: proceed to [Y]` line. Re-audited ckpt-500 on
identical corpus / split / seed / n — the rule is the only variable (guarded at the time by a
strict-prefix assertion in `tests/unit/test_stay_rule_flag.py`, retired on 2026-08-02 when the rule
moved inline as rule 2 and renumbered the rules below it; the equivalent guard is now that file's
byte-identity assertion that `TASK_A_STAY_RULE=0` reproduces the v1 prompt exactly).

| | baseline | STAY_RULE ON |
|---|---|---|
| gate composite | 0.6527 | **0.6496** |
| state accuracy | 0.5639 | 0.5584 |
| self-loop emission (gold: 41.6%) | 4.84% | **7.87%** |
| stay+tool bucket (n=164) accuracy | 0.0549 | 0.0610 |
| tool-F1 (tool-expected) | 0.4550 | 0.4550 |

Paired over the same 548 rows: delta **−0.0031**, 65 rows differ (30 favour the rule, 35 favour
baseline), **Wilcoxon p = 0.917**, paired-t p = 0.697, state McNemar p = 0.755. Against the ~5%
per-row decoding noise floor (§8), this is indistinguishable from noise.

Self-loop emission moved 4.84% → 7.87% — directionally correct, but it needs ~41.6%, so the
instruction closed roughly **8% of the gap**. On the bucket the rule targets directly
(stay + tool-expected) accuracy moved 0.055 → 0.061.

**Conclusion: the SFT'd policy does not respond to an explicit prompt instruction that
contradicts its trained behaviour.** This rules out the cheap path. §6.1 showed a *perfectly
applied* rule would cap at 0.7609 anyway; §6.5 now shows the rule cannot even be applied by
prompting. Both the corpus fix **and** retraining are required — the corpus revision alone is
insufficient if the policy is not retrained on it, and the prompt alone does nothing.

### 6.6 Epoch-3.0 checkpoint audited — the habit does not train out. Corpus regeneration confirmed required.

§6.5 left one cheap falsifier open: every measurement to that point was on ckpt-500 (**epoch 0.85**),
so "the model simply had not trained long enough to absorb the convention" was still live. It is now
closed. `checkpoint-1770` (**epoch 3.0**, 3.5× the training) audited on identical corpus / split /
seed / n (`task-a-sft-v1` validation, n=548, `TASK_A_STAY_RULE=0`) — checkpoint step the only variable.

| | ckpt-500 (ep 0.85) | **ckpt-1770 (ep 3.0)** | gold / bar |
|---|---|---|---|
| gate composite | 0.6527 | **0.6579** | 0.80 |
| state accuracy | 0.5639 | 0.5894 | 0.833 |
| **self-loop emission** | 4.84% | **6.39%** (43/673) | **41.6%** |
| stay+tool bucket (n=164) | 0.0549 | 0.0915 | — |
| tool-F1 (tool-expected, n=211) | 0.4550 | 0.4431 | 0.85 |
| abstention (zero-tool, n=337) | 0.9228 | 0.9050 | ~0.95 |

Paired over the same 548 rows: delta **+0.0052**, 95 rows differ (50 favour ckpt-1770, 45 favour
ckpt-500), **Wilcoxon p = 0.513**, paired-t p = 0.573. The state term alone moves +0.0255 at McNemar
exact **p = 0.054** — borderline, and it does not carry the composite anywhere near the bar. Note the
95/548 (17.3%) row-level disagreement is well above §8's ~5% same-checkpoint decoding-noise floor:
the two checkpoints genuinely behave differently, they just do not differ in aggregate *quality*.

**The decisive number is self-loop emission: 4.84% → 6.39% against the 41.6% gold expects.** Training
2.15 additional epochs closed **4.2%** of that gap — *less* than the §6.5 prompt instruction managed
(8.2%), and both are null. On the bucket that carries the whole deficit (stay + tool) accuracy is
still 0.0915. Meanwhile the model got slightly *better* at advancing (0.8781 → 0.9000) — it is
sharpening the behaviour the corpus rewards, not discovering the unstated one. The `from` state stays
correct on 98.96% of failed stay-rows (191/193), so this remains a convention defect, not confusion.

**Conclusion: the third cheap path is closed.** The self-loop habit is not an
under-training artifact — it does not train out with 3.5× the epochs on this corpus, exactly as it
did not prompt away in §6.5. §6.1 already showed a *perfectly applied* rule caps at 0.7609. All three
routes that avoid regenerating the corpus have now been measured and all three fail.
`task-a-sft-v2` via `generate_workflows.py` (per §6.4) plus a retrain is the remaining path.

*Provenance:* audited from `/tmp/sft_v3` (47 files / 560,269,322 B = `dvc.lock`
`d5438dced5a25f54af0d73e8569c6483.dir`), **not** `checkpoints/sft_cat_a/gemma-4-26B-A4B-it`, which had
silently reverted to the v2 lineage again (ckpt-500 adapter `2f172cb6…`) while `dvc.lock` still names
the v3 hash. **Resolved 2026-07-30, after the audit:** the local cache was also incomplete — 35 of
v3's 47 objects and its `.dir` object were missing — so the only complete *local* copy was the
staging tree in `/tmp/sft_v3`. The tree was verified by recomputing the DVC directory hash to
`d5438dced5a25f54af0d73e8569c6483`, all 47 objects + the `.dir` were written into `.dvc/cache`, and
the working directory was swapped to v3. `dvc status` now reports the SFT stage up to date and the
live path hashes to the `dvc.lock` out. v2 remains fully recoverable — 79/79 objects + `.dir` in
cache under `f89238076f5b09ca0d76e5b9ab98e4ec`.

*Scope of that risk, corrected:* v3 was **never** in danger of permanent loss. `dvc status -c` and
`dvc push` both report the checkpoint already fully present on the `gcs` remote, so the
`sft-gemma4-v3` tag's "pushed to gcs" was accurate and a `dvc fetch` would have restored it. The real
exposure was narrower but still live: the **working directory silently served v2 under a path
`dvc.lock` declared to be v3**, so an audit run against the canonical path would have measured the
wrong lineage and labelled it C0 — which is exactly what §1 caught the first time and what recurred
here. Note also the DVC CLI *is* available in this container at `.venv-train/bin/dvc` (3.67.1); an
earlier note in this section claiming otherwise was wrong.

The ckpt-500 column above is the stored §6.5 baseline; a same-session re-measure of v3
ckpt-500 was attempted as a provenance control but **failed — the container lost GPU access** in the
~3 s between the two arms (`cudaGetDeviceCount` err=100, "no CUDA-capable device is detected"; device
nodes and kernel module still present). Note this is *distinct* from the `Failed to initialize NVML`
that nvidia-smi reports in this dev container, which is cosmetic — it was already failing while the
ckpt-1770 arm ran to completion. It is not required for the conclusion,
which rests on the ckpt-1770 arm alone (6.39% vs 41.6% gold, composite 0.658 vs 0.80 bar), but it
remains the one unclosed control. Artifacts (gitignored):
`runs/preflight/heldout_audit_C0_ckpt1770_goldenval.json`,
`runs/preflight/composite_decomposition_ckpt1770.json`,
`runs/preflight/selfloop_habit_ckpt1770_vs_ckpt500.json`; new reproducible decomposition in
`scripts/analyze_selfloop_habit.py` (validated: it reproduces every §4 figure from the stored artifact).

---

## 7. Reconciliation with contrary evidence

Two facts appeared to contradict the "state accuracy is broken" reading. Both are consistent with
this diagnosis, and neither is undermined by it:

- **SFT token accuracy reaches ~94%.** That is teacher-forced next-token prediction averaged over
  every token, overwhelmingly ordinary conversational text. The state annotation is a few tokens
  per turn; a systematic error confined to one short field does not move that average.
- **The model performs well with real humans taking turns.** Confirmed by the samples: correct
  tools, correct arguments, helpful prose, states reached in the right order. It covers the same
  workflow in *fewer* turns, which is better for a human and worse against a grader that expects
  the two-turn form. The mislabelled state is not user-visible.

The genuine defect that remains, independent of scoring: the model commits to a state transition
before it has the tool result justifying it.

---

## 8. Tooling findings (recorded, not all fixed)

- **Greedy decoding is not bit-reproducible.** Two identical runs — same checkpoint, same args,
  same seed — differ on **14/289 rows** (~5%). Spec §7.3 assumes greedy makes cell-to-cell deltas
  noise-free; it does not. Any paired comparison needs this floor. §12.2's paired result
  (35/289 differing, p≈0.09) is partly noise.
- **`heldout_composite_audit.py` does not match the gate.** It scores a flat
  `0.4·state + 0.4·tool + 0.2·task` with no renormalization, while
  `grpo._heldout_composite_score` includes each term only when applicable and renormalizes. The
  audit's docstring claims they match exactly. Its `mean_state_acc` is also a flat mean over all
  rows, using a `1.0 if not pred_trans else 0.0` fallback that penalises an always-annotating
  model on no-transition rows. Not fixed — `scripts/analyze_composite_decomposition.py` recomputes
  the gate-aligned values post-hoc instead, reusing
  `perturn_fair_composite_from_components`, so no scoring surface moves between cells.
- **`--n-prompts` caps at unique conversations.** `_sample_prompts` dedupes by conversation, so
  §8.3's prescribed `--n-prompts 2943` silently yields far fewer. Switching to the SFT validation
  split raises the honest ceiling to 548.
- **`data/output/grpo/task_a` is stale and self-inconsistent.** `dvc.lock` names two hashes:
  `task_a_grpo.outs` = `9a961d65…` (clean R12 successor, absent from local cache) vs
  `task_a_grpo_gemma4_26b_a4b.deps` = `60831c66…` (pre-R12, what is on disk). The dep was never
  refreshed after the R12 fix landed in `6114866`. Untouched by this work.
- **The July-2 corpus ckpt-1000 trained on is NOT lost.** `8ef8681808e348f82eac36edbbd6c2ae.dir`
  is fully present in the local cache (131 files, 228 MB, 0 members missing), contradicting
  `sft-gemma4-v2`'s tag annotation and factorial spec §4.2. It is **untagged**, so `dvc gc -w`
  would delete it.
