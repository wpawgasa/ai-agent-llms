# Held-Out Metric Fix + GT Over-Eager Audit — Cat A ckpt-1000

**Date:** 2026-07-13
**Base:** `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000` (Gemma-4 26B-A4B-it, HF-generate path per R9)
**Companion to:** [`grpo_viability_investigation.md`](grpo_viability_investigation.md) — closes the §5.2 fork ("ship SFT-only vs escalate") and confirms §5-item-3 (GT/reward misspecification).

## TL;DR

Ran the SFT-only ceiling check (`scripts/heldout_composite_check.py`). It **FAILed** at
**0.479** (target 0.75). A row-level audit shows the FAIL is **mostly a scoring-harness
artifact, not a policy ceiling**: two of the three composite terms cannot be earned at
the per-turn granularity GRPO rows use. Corrected to **per-turn-fair**, the honest
baseline is **0.674** — still below 0.75, and the residual gap is **one genuine,
addressable weakness** (the policy transitions state and narrates intent correctly but
under-emits the actual `<tool_call>`). Two fixes landed in `training/grpo.py`:

1. **Per-turn-fair held-out metric** (`_heldout_composite_score`) — 0.479 → 0.674.
2. **Schema-aware GT sanitizer** — drops 32 fabricated-required-arg tool calls from the
   reward target so GRPO can't be trained to fabricate-and-fire.

Neither branch of the §5.2 fork as originally framed is correct: **not** a hard ceiling
(state 0.82 / abstention 0.80 are strong), and **not** "SFT already clears 0.75."

## 1. SFT-only ceiling check (as-scored)

`scripts/heldout_composite_check.py`, ckpt-1000, 150 validation prompts, greedy, seed 42, ~14 min.

| Metric | Value | Gate |
|---|---|---|
| mean_composite | **0.4773** | ≥ 0.75 → **FAIL** |
| median_composite | 0.4000 | |
| frac_below_target | 0.733 (110/150) | |

Artifact: `runs/preflight/heldout_composite_ckpt1000.json` (+ `.log`).

## 2. Row-level audit — why 0.48

`scripts/heldout_composite_audit.py` regenerates the same greedy pass but persists every
completion + a per-component breakdown. It reproduces the composite (0.4787 vs 0.4773; the
delta is batch-padding numerics) and decomposes it:

| Term | Weight | Mean | What's happening |
|---|---|---|---|
| **state_acc** | 0.4 | 0.60 | **0.817** where GT expects a transition (109 rows) — strong. **0.024** on the 41 empty-GT-transition turns: 40/41 the model emits its trained `[STATE:…]` annotation when GT says "no transition" → strict scorer zeroes it. **Artifact.** |
| **tool_f1** | 0.4 | 0.55 | **0.796** where GT expects no tool (correct abstention). **0.087** where GT expects a tool (52 rows) — **genuine weakness.** |
| **task** | 0.2 | **0.093** | 90.7% score 0. "Reached terminal" is a whole-conversation check; a single intermediate turn can't satisfy it (only 8/150 rows are terminal turns), and 21 rows carry an empty `terminal_state`. **Artifact.** |

**Root cause:** the 0.75 target is `eval/composite_score.compute_weighted_workflow_score`,
designed as a **whole-conversation** metric. The held-out probe (and the live
`_HeldOutEvalCallback`) apply it to **single user→assistant turns**. The GT itself is sound
and correctly per-turn (median state_sequence length 1); the mismatch is the scoring
granularity.

Artifact: `runs/preflight/heldout_composite_audit_ckpt1000.json` (150 completions + components).

### The disproven hypothesis
The check's FAIL action pointed at "placeholder-arg tool-call stubs." That is **not** the
cause: all 53 GT tool calls carry concrete args (`TXN-48219`, `disputed_amount: 84.5`), zero
placeholders. The real depressants are the two granularity artifacts above + the tool-emission gap.

### The one genuine weakness: "announce-but-don't-call"
Of the 52 rows where a tool is expected, 47 scored 0. Of the 30 that emitted **no** tool:
~15 are **announce-but-don't-call** (correct state transition, correct intent narration,
required args present, but no `<tool_call>` — e.g. gives `ORDER-55221`, transitions to
`LOOKUP_ORDER`, chats, never calls `lookup_order`); ~3–4 are **GT over-eager** (model
correctly withholds — see §4); ~6 borderline verify-first; ~2–3 degenerate (leaks
schema/instruction text, rare). This is a real RL/data target, not an artifact.

## 3. Fix A — per-turn-fair held-out metric (applied)

`training/grpo.py::_heldout_composite_score` now includes each term only when applicable to
the single turn, renormalizing over the included weights: **tool always**, **state only when
GT expects a transition**, **task only on the terminal turn** (GT transition's `to` ==
non-empty `terminal_state`). It never grants unearned credit on an applicable term — it only
stops charging the policy on terms it cannot satisfy on that turn.

| View | Score |
|---|---|
| As-scored (whole-conv metric, per-turn rows) | 0.479 |
| Drop the misapplied task term | 0.575 |
| Per-turn-fair (also stop punishing abstention-turn annotation) | **0.674** |

The patched in-tree function re-scores the saved completions to **0.6737** — bit-identical to
the standalone reference `scripts/perturn_fair_composite.py`.

**Blast radius:** this is also the **live GRPO** metric (`eval/held_out_composite`) and feeds
the R5 reward-hacking auto-stop (`_is_reward_hacking`), which compares deltas, so its logic
still holds — but the corrected metric averages fewer terms on abstention/terminal turns
(slightly noisier per-turn); bump its `lookback` if spurious auto-stops appear. It also changes
"best checkpoint by held-out." The existing known-answer tests are unchanged because their GT
is a terminal-turn-with-transition (all three terms apply → fair == old formula); the divergent
abstention/intermediate cases are locked by new tests.

## 4. Fix B — schema-aware GT sanitizer (applied)

The synthetic corpus's `invalid_tool_inputs` behavior (~15% of turns) produced GT tool calls
that fire an action with a **fabricated value on a required arg** — the anti-pattern §5-item-3
predicted. Schema-verified scope (each conversation's own `tool_schemas`): **69 calls across 42
conversations** carry a null-sentinel (`"UNKNOWN"`, `"N/A"`, `"000000"`) on a required arg or an
out-of-range score. Examples: `apply_for_loan(customer_id="UNKNOWN")`,
`report_fraud(transaction_id="N/A")`, `dispute_bill(account_id="000000")`, `collect_nps(score=11)`.

**Not flagged (deliberately):** placeholders on **optional** fields —
`log_complaint_trend(region="unknown")` (28×, the biggest raw bucket) is a harmless default;
and the dataset's synthetic ID style (`TH12345`, `CUST-123`) when user-provided is not a sentinel.
Both were earlier false positives that the schema check removes.

Implementation (`training/grpo.py`): pure `_gt_tool_call_is_invalid` / `_sanitize_gt_tool_calls`
+ `_required_args_by_tool`, wired into `_load_grpo_jsonl` (the single chokepoint feeding reward
**and** held-out eval), gated by `_SANITIZE_INVALID_TOOL_GT = True` (flip to reproduce a raw run),
logged as `sanitized_invalid_tool_calls`.

**Effect:** loader sanitizes **27 train + 5 val = 32**. The other 37 flagged calls sit on
`tool`-preceded assistant turns the loader already skips (`skipped_tool_preceded_turns`), so they
never reached the reward — reconciliation is exact. **0** invalid calls survive in loaded GT.

**Scope limits (honest):** (a) fix targets the **GT reward target**; the fabricated `<tool_call>`
still exists in raw message *content*, so it can appear as in-context history in a *later* turn's
prompt (never a reward target there) — full content rewrite needs a teacher model, not attempted.
(b) Only the **invalid-value** over-eager pattern is auto-cleaned; the **hypothetical-intent**
pattern (e.g. "file a claim *later*" → GT fires `file_travel_claim`) has valid args and isn't
arg-detectable — flagged for manual review in `runs/preflight/gt_overeager_review_ckpt1000.csv`.

## 5. Changes & artifacts

**Modified:** `src/llm_workflow_agents/training/grpo.py` (`_heldout_composite_score` per-turn-fair;
GT sanitizer + `_load_grpo_jsonl` wiring; `import re`); `tests/unit/test_reward_functions.py` (+3
divergent-case tests).
**New code:** `scripts/heldout_composite_audit.py`, `scripts/perturn_fair_composite.py`,
`tests/unit/test_perturn_fair_composite.py` (5), `tests/unit/test_grpo_gt_sanitize.py` (8).
**Tests:** 130 passing across the grpo/reward/composite selection, 0 regressions.
**Run artifacts (gitignored):** `runs/preflight/heldout_composite_{ckpt1000,audit_ckpt1000,perturnfair_ckpt1000}.json`,
`runs/preflight/gt_overeager_review_ckpt1000.csv`.

## 6. Next move — (2) re-run the headroom/trajectory probes clean

The earlier Stage-0 headroom (`frontier_frac 0.130`) and trajectory-variance probes read
**MARGINAL** while scoring on the artifact-laden metric **and** the un-sanitized reward target.
Both inputs are now fixed, so re-measure before any RFT/GRPO commit:

1. **Re-run `scripts/preflight_entropy_diag.py` headroom probe** (ckpt-1000, and 1500) on the
   sanitized GT + per-turn-fair reward. Cleaning removes reward mass from fabricate-and-fire
   samples; the frontier/headroom lattice can shift.
2. **Re-establish the Stage-1 RFT gate baseline** with the corrected `_heldout_composite_score`.
   The gate (`grpo_viability_investigation.md` §4 Stage 1) is a before/after Δ, and both sides use
   this function, so it stays internally consistent — but the **absolute** baseline is now ~0.674,
   not 0.48. Note the 0.674 was re-scored on **pre-sanitization** validation GT; the ~5 sanitized
   val rows nudge it marginally up, so re-measure post-clean for the true number.
3. **Target the announce-but-don't-call gap** — the genuine, non-artifact headroom a
   tool-F1-weighted RFT/GRPO signal is built to push on.

Do **not** conclude "hard single-turn ceiling" or "ship SFT-only" until steps 1–2 are re-run on
the clean inputs — the prior MARGINAL verdicts were partly artifact-driven.
