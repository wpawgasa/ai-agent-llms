# Cat A State-Annotation Convention Review

**Date:** 2026-07-29
**Scope:** `docs/superpowers/plans/2026-07-22-cat-a-state-accuracy-factorial.md` §12.4 steps 1–2
**Artifacts:** `runs/preflight/heldout_audit_C0_ckpt500_goldenval.json`,
`runs/preflight/composite_decomposition_goldenval.json`,
`scripts/analyze_composite_decomposition.py`

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
