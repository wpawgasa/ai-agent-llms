# Cat A State-Accuracy Retrain — Controlled SFT-Recipe Factorial

**Date:** 2026-07-22
**Owner:** Phase 2, Category A (workflow-orchestrating agent)
**Base checkpoint:** `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000` (Gemma-4 26B-A4B-it SFT LoRA adapter; HF-`generate()` rollout path per R9)
**Supersedes the retrain design in:** `docs/grpo_tool_emission_gap_review.md` §4 step 4 (built for a tool-emission diagnosis that §11 overturned)
**Status:** SPEC — not implementation. No code is written and no run is launched by this document. Implementation is a follow-up gated on explicit approval of this spec (see §11).

---

## 0. TL;DR

The investigation's authoritative finding (`grpo_tool_emission_gap_review.md` §11) is that ckpt-1000's dominant deficit is **state-transition accuracy (0.6866 measured vs 0.85 target)**, specifically **destination-selection** — the model always emits `[STATE: X → Y]`, always gets `from` right, and picks the wrong `to` on 87/89 failing rows, in both directions (advances when it should stay, stays when it should advance). Tool-F1 is a secondary problem (0.4623, not the retired 0.087).

The per-turn-fair composite bar is **0.80** (re-derived, §10). Current composite is **0.7271**. Lever analysis (§11.3): taking **state accuracy alone to 0.85 clears the bar** (→0.8085) and is far the cheaper lever (+0.146 vs +0.388 for tools). Four independent probes (§6/§8/§9) established that RL has **zero headroom** (70% of prompt-groups collapse to one reward rung; the failing majority fails *deterministically*), so **this is a data/SFT problem, not an RL problem**.

Therefore this is an **SFT-recipe factorial**, run as a **cost-disciplined single-variable ladder** (C0 control → C1 masking → C2 decision-balance), each cell gated on a greedy held-out audit against the re-derived 0.80 bar. Reward-function changes are **out of scope this round** (a dead RL signal cannot be helped by reweighting the reward; the artifact under test is an SFT checkpoint). Tool-emission is monitored as a guardrail but is **not** a primary lever this round; it is the explicit next-round target, gated on data regeneration.

---

## 1. Target, stated crisply

**Primary objective.** Produce a Cat A SFT checkpoint whose **greedy held-out per-turn-fair composite ≥ 0.80** on the canonical post-R12 validation set (`data/output/grpo/task_a`, split `validation`), achieved **primarily by moving state-transition accuracy from 0.6866 toward 0.85**.

**What "state accuracy" means here, precisely.** It is `mean_state_acc` from `scripts/heldout_composite_audit.py` — `state_sequence_match(pred_transitions, gt_transitions)`, strict, effectively **binary per row** because each sliced GRPO row carries exactly one gold transition (89 zeros / 195 ones at n=284). Because 100% of sliced validation rows carry a transition (§10.2), `mean_state_acc` over all rows **equals** "state accuracy where a transition is expected." The failure is **destination selection** (`to`-state choice), not annotation presence (0/89 omit the marker) and not `from`-tracking (87/89 correct).

**Numeric anchor to clear the bar (from §11.3, empirically validated by `expected_perturn_score` to 0.0013):**

| Path | required component | resulting composite | verdict |
|---|---|---|---|
| **state → 0.85**, tool unchanged (0.4623) | state 0.85 | **0.8085** | PASS ✅ (chosen) |
| state → 0.833, tool unchanged | state 0.833 | ≈0.80 | minimum-to-clear |
| tool → 0.85, state unchanged | tool 0.837 | 0.8026 | PASS but +0.388 gap (rejected as primary) |
| both → target | — | 0.8827 | stretch |

**The primary lever is state accuracy** because the shortfall to clear the bar is +0.146 on state vs +0.388 on tools, and §9 shows the tool behaviour is *deterministically absent* on the majority of failing anchors (an SFT-recipe change cannot manufacture a behaviour the policy never emits; that needs data regeneration).

**Guardrail objective (not primary).** No cell may ship if it **regresses tool-F1 on tool-expected rows below the 0.4623 baseline**, or abstention below 0.9494. `response_only` masking is *hypothesised* to help tool emission as a side effect (it concentrates gradient on the assistant span that contains both `[STATE]` and `<tool_call>`); we measure it but do not depend on it.

---

## 2. Reconciling the reward-function discrepancy (required)

There are **three** distinct scoring surfaces in this system. They must not be conflated. This spec is measured against the **live** definitions, not the stale rules doc.

| # | Surface | Where | Shape | Role in this experiment |
|---|---|---|---|---|
| **(a)** | **Live GRPO training reward** | `training/rewards/reward_business_logic.py` | **4 components**: `state_transition` **0.40** (graded, `_graded_state_match`), `tool_call_f1` 0.40 (`graded_tool_call_f1`), `task_completion` 0.10, `transition_legality` 0.10. Conditionally-active components renormalize. | **Not exercised this round** — no GRPO run (RL headroom is dead). Documented here only so the reader knows the redesign is measured against *this*, not the rules doc. |
| **(b)** | **Held-out eval metric** | `grpo._heldout_composite_score` / `scripts/heldout_composite_check.py` | `0.4·state + 0.4·tool_f1 + 0.2·task`, **per-turn-fair** (each term included only when applicable, renormalized), **STRICT** scorers (`state_sequence_match`, `compute_ast_f1`, `reached_terminal`). | **This is the success metric.** The 0.80 bar and 0.7271 baseline live on this surface. Deliberately numerically independent of (a) so reward-vs-quality divergence stays detectable (R5). |
| **(c)** | **Benchmark composite** | `eval/composite_score.py::compute_weighted_workflow_score` | Whole-conversation `0.4·state + 0.4·tool_f1 + 0.2·task`. | Legacy/Phase-1 surface. The old 0.75 target came from here; **retired** for the per-turn metric (§10). Not used as a gate. |

**The `.claude/rules/03-training.md` "5-component / state 0.30" description is STALE and does not describe any live code.** The live reward is (a): 4 components, state **0.40**. The `length_band` component was removed because it was the only source of within-group GRPO variance and the model gamed completion length instead of correctness (see the module docstring). The `grpo_cat_a.yaml` `reward.weights` block is **informational only** — `grpo.py` reads `reward.function`, not `reward.weights`; live weights are Python constants. Changing a reward weight means editing `reward_business_logic.py`, not YAML.

**Two graded/strict scorer pairs to keep straight:**
- State: training uses `_graded_state_match` (1.0 / 0.5 for `from`-only *or* `to`-only / 0.3 reverse / 0.0). The **audit and gate use the strict `state_sequence_match`** (exact-transition match, binary per single-transition row). A destination miss scores **0.5 under the training reward but 0.0 under the gate** — the gate is the honest, harder number and the one we ship against.
- Tool: training uses `graded_tool_call_f1` (argument-graded); gate uses strict `compute_ast_f1`.

**Consequence for the design:** because the gate is strict-binary on state, partial credit in the training reward is irrelevant to whether a cell ships. This reinforces that reward-side graded-match tuning (a candidate lever below) cannot move the gate directly and is out of scope this round.

---

## 3. Why this is an SFT-recipe factorial (scope decision)

Five independent measurements converge (§6, §8, §9, §11):

- SFT-only greedy composite 0.7167 (n=150) / 0.7271 (n=284) — **FAIL** vs 0.80.
- RFT headroom **MARGINAL**, `frac_collapsed_groups` 0.764, `median_reward_std` 0.000.
- pass@8 shows **53.3% of failing anchors never emit the correct call in 8 samples at T=1.0** — deterministically absent.
- The eval-granularity "artifact" off-ramp is **falsified** (§8): the policy genuinely under-performs regardless of turn granularity.

§9.6 / §11.6 conclusion, adopted here: **"this is a data/SFT problem, not an RL problem."** A GRPO or RFT run cannot reinforce a behaviour the policy emits with zero within-group variance. The artifact the audit grades is an **SFT checkpoint** (ckpt-1000 is step 1000 of a 3,426-step SFT run). Therefore:

- **Each factorial cell = one SFT run + one greedy held-out audit.** No GRPO in the loop.
- **Reward-function / graded-match changes are deferred**, not adopted. Rationale in §7.
- **The GRPO track re-opens only after** a retrained SFT checkpoint shows a non-collapsed group distribution (re-run `scripts/rft_headroom_probe.py`); that is a separate, future decision.

---

## 4. Baseline, corpus, and the two structural facts that force the design

### 4.1 Verified on-disk state (2026-07-22)

| Artifact | Path | Count | mtime |
|---|---|---|---|
| SFT canonical splits | `data/output/sft/task_a_splits/{train,validation,test}.jsonl` | **4,716 / 554 / 279** conversations | 2026-07-14 |
| GRPO audit source | `data/output/grpo/task_a/{train,validation}.jsonl` | 2,563 / **290** conversations | 2026-07-14 |
| GRPO validation, **sliced** (`_load_grpo_jsonl`) | — | **2,943 rows**, **1,126 tool-expected** (§10.2) | derived |
| Pre-R12 backup (for §11.5) | `data/output/grpo/task_a.pre_r12fix_backup_20260714T104107Z` | 290 (largely disjoint) | preserved |

### 4.2 Structural fact #1 — ckpt-1000's training corpus no longer exists on disk (confirmed via W&B + DVC hash)

ckpt-1000's `train.log` proves it trained on **9,131 train / 1,074 eval** conversations (`sft_dataset_render loss_mask=all_tokens`, `Map: 9131`, `Map: 1074`). This was initially inferred from file mtimes and directory naming; it is now **confirmed by direct, hash-verified provenance** (2026-07-22):

- The actual training run is `wandb.ai/wpawgasa/huggingface/runs/uklfswk5`, started **2026-07-02T04:58:27Z**, git commit `ef837079e1d36581c7e589356881c33b3c575ecf`. `output_dir: checkpoints/sft_cat_a/gemma-4-26B-A4B-it`, LoRA rank/alpha/dropout 16/16/0, `learning_rate: 5e-5`, `num_train_epochs: 3`, `seed: 3407` — all match `configs/training/sft_cat_a.yaml`, confirming this run produced ckpt-1000.
- At that exact commit, `dvc.lock`'s `task_a_sft_gemma4_26b_a4b` stage pinned its input `data/output/sft/task_a_cleaned` to hash **`8ef8681808e348f82eac36edbbd6c2ae.dir`** (131 files, 227,958,540 bytes ≈ 228MB).
- The **current** `dvc.lock` pins `data/output/sft/task_a_cleaned` to a **different** hash, **`3d6a4d3eed110546eccb46a94c5d68cd.dir`** (5 files, 109,983,413 bytes ≈ 110MB — 48% the size, restructured from 131 loose files into 5 consolidated merged files), which is the direct input to the current 4,716/554/279 `task_a_splits`.
- **A DVC content hash mismatch is definitional proof the file contents differ** — this is not an inference from mtimes or row counts anymore, it is the same kind of hash comparison DVC itself uses to decide whether a stage needs to re-run. The R12 note's "the corpus grew (4,414→4,716)" narrative compared against a different, stale, now-absent directory and never accounted for this discontinuity.
- The July-2 `task_a_cleaned` blob (hash `8ef8681...`) is **not recoverable in this environment** — no GCS credentials are configured here to `dvc pull`/`dvc fetch` it from `gs://looloo-voicebot-llm-weights-and-data/llm-workflow-agents`. It may still exist in the remote's content-addressable storage if it hasn't been garbage-collected; worth attempting on a machine with the `looloo-ocr` service-account credentials.
- Note a secondary nuance: at the July-2 commit, `dvc.lock` did not yet have a separate `task_a_sft_splits` stage (only `task_a_benchmark`, `task_a_sft`, `task_a_grpo` existed) — so `task_a_splits`'s exact byte-identical state at that time isn't itself independently DVC-pinned. But since `task_a_splits` is deterministically derived from `task_a_cleaned` via `split_task_a_sft.py`, and the input `task_a_cleaned` is proven different, the derived split must differ too — consistent with, and now much more strongly evidenced than, the row-count comparison (9,131 vs 4,716) alone.

**Implication (load-bearing):** there is **no same-data replicate** of ckpt-1000. Its 0.6866 was produced by a model trained on a corpus that is confirmed gone, not just apparently smaller. Any new SFT run trains on the current 4,716-conversation corpus. **A control cell (C0) retrained on the current 4,716-conv corpus with the unchanged `all_tokens` recipe is therefore mandatory** — it is the only legitimate anchor for attributing C1/C2 deltas to *recipe* rather than to the corpus discontinuity. The C0-vs-ckpt-1000 comparison separately *quantifies* the corpus-version confound but is not itself an experimental contrast.

> **OPEN ITEM (blocking, must resolve before launch):** the corpus discontinuity is now hash-confirmed, not just size-inferred. Confirm the current `task_a_splits` (fed by `task_a_cleaned` hash `3d6a4d3e...`) is the *intended* training corpus going forward — i.e. that the consolidation + R12 fix that replaced the July-2 corpus was a deliberate, accepted change, not an accidental loss. If so, C0 on the current 4,716-conv corpus is the correct anchor (this is the default assumption this spec proceeds under). If the July-2 corpus (or something derived from it) was actually the intended target and its replacement was unintentional, that needs resolving — recovering the blob from the DVC remote, if possible — before C0 is meaningful.

> **RESOLVED (2026-07-22).** Verdict: **proceed with C0.** Two independent lines of evidence, both from this session:
> 1. **Provenance is confirmed intentional, not accidental loss.** The July-2 corpus (`task_a_cleaned` hash `8ef8681...`) predates the R12 malformed-role fix and — per CLAUDE.md R12's own contamination analysis of this same corpus lineage — carried ~5.1%/5.7%/4.6% (train/val/test) malformed-role corruption (`assistant to=tool`-style leaked tool-routing syntax). Its replacement by the current, cleaned corpus was a deliberate fix (R12), not data loss.
> 2. **The current corpus was re-verified by actual execution, not just trust.** `dvc repro -s -f` was run for real on `task_a_sft_clean` (`clean_task_a_sft.py` + `clean_corrupted_conversations.py`) and `task_a_sft_splits` this session — output is byte-identical to the previously hash-recorded state (`task_a_cleaned` `3d6a4d3e...`, `task_a_splits` `6bb5eb6f...`; 0 rows altered by either cleaning pass). Tagged `task_a-corpus-v1-2026-07-22` and `task_a-corpus-v2-2026-07-22` (git tags pinning the exact `dvc.lock` state; see `docs/grpo_tool_emission_gap_review.md`'s status banner for the fuller pipeline-fix writeup, including two real DVC bugs this uncovered and fixed: `task_a_sft_gemma4_26b_a4b`'s dep was tracking `task_a_cleaned` instead of the `task_a_splits` it actually reads, and `task_a_sft_clean`/`task_a_sft_splits` had a YAML cmd-formatting bug that meant they'd likely never been successfully run via `dvc repro` before this session).
>
> **Net: ckpt-1000 was trained on a corpus that is both gone *and* confirmed to have had real (if modest, ~5%) structural corruption. C0 is not optional cleanup — it is the first checkpoint trained on the current, cleaned, execution-verified, git-tagged corpus, and the correct baseline for everything downstream in this spec.** Proceed to §6's ladder starting with C0 when GPU time is available.

### 4.3 Structural fact #2 — SFT trains whole-conversation; the gate scores per-turn

`sft.py::_load_split` emits **one row per whole conversation** (`data/output/sft/task_a_splits`). `grpo._load_grpo_jsonl` emits **one row per user→assistant turn** (`data/output/grpo/task_a`, additionally L3-L5-filtered), and the audit scores those per-turn rows single-turn-teacher-forced. **The training unit and the eval unit differ.** This matters for the reweighting lever (§6, factor B): you cannot "upsample an advancing turn" inside a whole-conversation SFT example without net-new per-turn machinery. It also means SFT training is *not* subject to the single-turn slicing artifact of §7.1 (the model sees the full narrate-then-call two-turn structure during training); that artifact lives only on the eval side, where §8 already showed the defect survives it.

### 4.4 Baseline numbers this spec designs against (all from `heldout_composite_audit_ckpt1000_regen.json`, n=284, 106 tool-expected)

| component | value | target | note |
|---|---|---|---|
| **state accuracy** | **0.6866** | 0.85 | primary lever; destination-selection, bidirectional |
| tool-F1 (tool-expected) | 0.4623 | 0.85 | guardrail; 53.8% still score 0 |
| abstention (zero-tool) | 0.9494 | ~0.95 | corroborates the 0.80 bar (§11.2) |
| task completion | 1.0000 | 0.70 | saturated; not a lever |
| **per-turn-fair composite** | **0.7271** | **0.80** | shortfall **−0.073** |

§11.5 ("why 0.817 → 0.687?") is **resolved** and treated as closed: `sanity_check_state_acc_drop.py` returned `DIFFERENT_VALIDATION_SETS_NOT_MODEL_REGRESSION` (only 4.1% conversation overlap between the two eval sets; the empty-GT-transition bucket that made 0.817 a sub-population figure has vanished; corpus composition shifted 94.8%→62.5% advancing). **0.6866 is the trustworthy, current state-accuracy baseline** and there is no moving-target risk from that comparison. The GPU-side confirmatory reproduction on the backup corpus remains a nice-to-have, not a blocker.

---

## 5. Candidate levers — assessed on their merits (accept / reject / modify)

Each candidate from the brief is evaluated freshly against the **destination-selection** diagnosis, not the retired tool-emission one.

### 5.1 `loss_mask: response_only` vs `all_tokens` — **ACCEPT (as the cheap C1 rung), with calibrated skepticism**

- **Mechanism:** `all_tokens` computes loss over every non-pad token including the ~14K-char boilerplate system prompt; `response_only` masks all non-assistant spans to −100 (fully implemented in `sft.py::render_response_only_sample`, never used on Cat A — `configs/training/sft_cat_a.yaml` has `all_tokens`). Concentrating gradient on assistant tokens theoretically sharpens the short `[STATE: X→Y]` + surrounding reasoning span.
- **Skeptical read (inherited from §3, re-assessed for THIS problem):** the argument is *weaker* for destination-selection than it was for the old tool-emission story. The model already emits `[STATE]` and already gets `from` right under `all_tokens` — annotation presence and origin-tracking are not gradient-starved. The wrong-`to` choice is a *context-conditioned decision*, not a rare-span-emission problem, so it is not obvious that de-diluting the boilerplate fixes it. Counter-argument for keeping it: `response_only` still up-weights the *reasoning tokens preceding the transition*, which is where destination is decided, and it is nearly free to test (a config flip on already-shipped code).
- **Verdict:** test it as the **first paid rung** because cost is minimal and the downside is bounded. Do **not** assume it wins. Watch per-epoch eval loss — `response_only` puts fewer supervised tokens per example on a ~4.7K-conversation corpus; the §3 overfitting caveat applies (3 epochs may be too many; enable early-stop on eval loss).

### 5.2 Upsample/reweight state-advancing turns — **MODIFY, then ACCEPT as a gated contingency (C2)**

- **Naive form rejected.** "Upsample advancing turns (`from != to`)" assumes the failing population is one-directional. It is not: §11.4 shows the failure is **bidirectional** (advance-when-should-stay *and* stay-when-should-advance; 87/89 are wrong-`to` in both directions). Upsampling only advancing turns risks **worsening the over-advance failures** by biasing the policy toward motion. This is a real hazard, not a hypothetical.
- **Reframed form accepted.** The lever should **balance the stay-vs-advance decision boundary**, not upsample one class. Concretely: control the ratio of self-loop `(X,X)` targets to advancing `(X,Y)` targets the model is trained on, and (per §9.4's inverted difficulty gradient — L3 fails most, 75.6% never-name-match vs L5 27.3%) **weight by complexity level**, not uniformly.
- **Cost/structure caveat (§4.3):** there is **no upsampling infrastructure anywhere in the codebase** (confirmed: no `WeightedRandomSampler`, no `repeat_factor`/`oversample`). This is net-new. Because SFT is whole-conversation, two insertion strategies exist:
  - **(B-conv, recommended lower-risk):** reweight/replicate *whole conversations* by their stay/advance decision-diversity (e.g. up-weight conversations rich in the under-represented decision type), inside `sft.py::_load_split`. Preserves the whole-conversation training distribution; smaller confound with `response_only`.
  - **(B-perturn, higher-fidelity, higher-confound):** a net-new per-turn SFT slicer (mirroring `_load_grpo_jsonl`) that emits one training row per turn, then balance the stay:advance:level mix directly. This changes the training unit from whole-conversation to per-turn — a large, separate variable that would confound the masking result. **Reject for this round** unless C0/C1 fail and a follow-up spec isolates it.
- **Verdict:** hold C2 as a **contingency gated on C0/C1 not clearing the bar**, implemented as B-conv. Do not build it speculatively.

### 5.3 Change `state_transition` reward weight or `_graded_state_match` partial-credit rules — **REJECT this round**

- The gate (`_heldout_composite_score`) uses **strict** `state_sequence_match`, which is **numerically independent** of the training reward by design (R5). Reweighting `state_transition` (0.40) or loosening `_graded_state_match` moves surface (a), which **does not touch the gate** — a destination miss is 0.0 on the gate regardless of graded partial credit.
- More fundamentally, **there is no GRPO run this round** (§3): RL headroom is dead (§9: 70% collapsed groups; §6.2: `frac_collapsed_groups` 0.764). A reward change cannot help a signal that carries zero within-group variance. Editing `reward_business_logic.py` constants now would be optimizing a component of a stage we are not running.
- **Verdict:** out of scope. Revisit reward shaping only in a **future RL round**, gated on a retrained checkpoint whose group distribution is measurably non-collapsed. Recorded as future work (§10).

### 5.4 Address the tool-emission weakness (upsample tool turns) — **REJECT as a primary lever; keep as guardrail + next-round target**

- Tool-F1 on tool-expected rows is 0.4623 (real; 53.8% score 0), still below the 0.85 component target. But: (1) **state alone clears the bar** (§1) at +0.146 vs +0.388 for tools; (2) §9 shows the missing call is **deterministically absent** on the majority — an SFT-recipe change cannot synthesize it, and §11.6 notes tool-turn upsampling "does not obviously address destination selection"; (3) adding a tool factor **doubles the factorial** and confounds the state result.
- **Verdict:** do **not** make tool emission a factor this round. **Monitor** tool-F1 on tool-expected rows as a ship-blocking guardrail (no regression below 0.4623). Make tool emission the **explicit next-round objective**, addressed by **data regeneration** of the narrate-then-call structure (§7.1's two-turn split; a generator-side change), gated on this round's outcome — not by masking or reweighting.

---

## 6. The factorial: a cost-disciplined single-variable ladder

A full 2×2×level grid is rejected on cost (each cell is a full SFT run + audit) and on the "one variable at a time" mandate. Instead: a **sequential ladder** where each rung changes exactly one factor from the rung below and is **gated** — a rung that clears the bar terminates the ladder (ship); a rung that does not license the next rung.

| Cell | loss_mask | decision-balance reweight | changes vs prior | purpose |
|---|---|---|---|---|
| **C0 (control)** | `all_tokens` | none | corpus-version only (vs ckpt-1000) | anchor; quantifies the 9,131→4,716 confound |
| **C1** | **`response_only`** | none | +masking (one var vs C0) | tests the masking lever cleanly |
| **C2 (contingency)** | (C1's winner) | **stay/advance/level-balanced (B-conv)** | +reweight (one var vs C1) | tests decision-balance if C0/C1 short |

**Everything else is held identical across all cells** (this is the confound control — see §8): base model, LoRA rank/alpha/dropout (16/16/0), LR (5e-5 cosine, 0.05 warmup), `max_seq_length` 4096, `packing: false`, 3 epochs with eval-loss early-stop, seed 3407, **the current `task_a_splits` corpus**, and **SFT-data GT sanitization held OFF** (matching ckpt-1000's recipe; see §8.1).

**Decision rule between cells** (thresholds in §9):
1. Run **C0**. Audit. If composite ≥ 0.80 → **ship C0** (the corpus fix alone sufficed; ladder ends).
2. Else run **C1**. Audit. If ≥ 0.80 → ship C1. If C1 ≤ C0 on state accuracy (masking hurt or was neutral) → **do not carry masking forward**; C2 branches from C0's recipe instead.
3. Else run **C2** from the better of {C0, C1}. Audit. If ≥ 0.80 → ship. If not → escalate to the **next-round options** (§10): SFT-data sanitization as a variable, per-turn SFT slicer (B-perturn), or generator-side data regeneration — each its own controlled follow-up.

**Why C0 cannot be skipped:** without it, a C1 improvement over ckpt-1000's 0.6866 is uninterpretable — it could be entirely the corpus change (4.2). C0 makes every subsequent contrast single-variable.

---

## 7. Confound control (required — how each cell's effect is isolated)

### 7.1 The GT-sanitizer confound (§3)

`ckpt-1000` trained **before** the GT sanitizer existed. The sanitizer (`grpo._sanitize_gt_tool_calls`, `_SANITIZE_INVALID_TOOL_GT=True`) drops fabricated-required-arg tool calls, but it lives **only in the GRPO/eval loader** — `sft.py::_load_split` does **not** sanitize, so SFT training data still contains those calls. Two facts follow:
- **Measurement side is already consistent:** the audit always reads `data/output/grpo/task_a`, which is always sanitized, for **every** cell including ckpt-1000. The gate does not vary with this confound.
- **Training side is held constant:** we keep SFT-data sanitization **OFF in all cells** (C0/C1/C2), matching ckpt-1000's training recipe. This makes sanitization a **non-variable** this round, so a C1−C0 delta is attributable to masking alone, not to a simultaneous data-recipe change. Sanitizing SFT training data is a legitimate future lever — but as its **own** controlled cell in a later round, never folded into this one.

This directly answers §3's warning ("any retrain changes loss recipe AND data at once"): the corpus change is absorbed by C0 (the anchor), and the sanitizer change is neutralized by holding it constant.

### 7.2 The row-slicing exposure-bias confound (§7.1)

The corpus splits narrate-then-call across two assistant turns 70.5% of the time; the single-turn audit can score the bare-call turn as a miss even when the policy is competent across turns. §8 already proved the defect survives this (free-running multi-turn did not recover the call; McNemar p=0.743), so we accept the per-turn audit as the metric of record. To ensure the confound **cancels between cells**: every cell is audited with the **identical protocol** — same slicer (`_load_grpo_jsonl`), same split (`validation`), same seed (42), same `--n-prompts`, same greedy decoding, same `--max-new-tokens 512`. The exposure-bias structure is then a **constant** across cells and drops out of any between-cell difference. We report deltas (C1−C0, C2−C1), never a cell's absolute number against a differently-sliced historical figure.

### 7.3 Determinism / attribution hygiene

- Fixed seeds throughout (SFT `seed: 3407`; audit `--seed 42`), greedy decoding (`do_sample=False`) so completions are reproducible and cell-to-cell deltas are not sampling noise.
- One factor changes per rung (§6). No cell varies two knobs.
- The **same checkpoint step** is audited across cells (see §9 note on step selection) so "which checkpoint" is not a hidden variable.

---

## 8. Power analysis

State accuracy is **binary per row**; tool-F1 is **continuous but strongly bimodal** (≈54% exact zeros). Design is **paired** (identical validation prompts scored on each checkpoint, greedy → deterministic), so paired tests (McNemar for the binary state term; paired-`t`/Wilcoxon for tool-F1) apply and are more powerful than the two-sample bounds below. The two-sample numbers are the conservative floor.

### 8.1 State accuracy — n≈284 is ADEQUATE for the required move

- Single-measurement precision at p=0.687, n=284: SE = √(0.687·0.313/284) = **0.0275** → 95% CI half-width **±5.4pp**.
- Two-sample MDE (independent, 80% power, α=0.05 two-sided): ≈ 2.8·√2·0.0275 = **±10.9pp**.
- **Required improvement to clear the bar: 0.6866 → 0.833 = +14.6pp** (§1). This **exceeds** the unpaired MDE, and the paired MDE is smaller still. **n=284 reliably detects the state-accuracy delta the lever analysis calls for.** No eval growth is needed for the primary metric.

### 8.2 Tool-expected sub-slice — n≈106 is UNDERPOWERED, but growable for free

- Tool-F1 on tool-expected rows (n=106), treating the ≈54/46 zero/nonzero split as near-binary: SD ≈ 0.50, SE ≈ 0.049. Two-sample MDE ≈ 2.8·√2·0.049 = **±19pp**.
- Any tool-F1 movement smaller than ~19pp on this slice is **noise** — inadequate to certify the guardrail or to detect an incidental `response_only` benefit.
- **Fix — no new data required.** The audit samples `--n-prompts` **rows** out of the **2,943** the validation split slices to (1,126 tool-expected). §11's n=290 audit therefore used only ~9% of the available tool-expected population. **Raising `--n-prompts` toward 2,943 grows the tool-expected slice ≈10×** (106 → ~1,126), shrinking its SE by √(1126/106) ≈ 3.3× → **MDE ≈ ±6pp**, at pure GPU cost (no generation of new data). This is the recommended way to power the guardrail.

### 8.3 Recommendation

- **Gate/state metric:** audit at `--n-prompts 290` (matches §11, adequate for state) for the fast per-cell decision.
- **Guardrail/tool metric:** for any cell that reaches the ship decision, **re-audit at `--n-prompts 2943`** (full validation coverage) to certify tool-F1 non-regression at ±6pp before committing the checkpoint. Do not grow generation; grow only the sampled fraction of existing rows.

---

## 9. Success criteria (exact invocations + thresholds)

Two scripts, same args, same seed, same data-dir. Run on the GPU/training host with `.venv-train`.

**(1) Official gate — per-turn-fair composite:**
```
.venv-train/bin/python scripts/heldout_composite_check.py \
    --checkpoint checkpoints/<cell-run-dir>/gemma-4-26B-A4B-it/checkpoint-<STEP> \
    --data-dir data/output/grpo/task_a --split validation \
    --n-prompts 290 --seed 42 \
    --output runs/preflight/heldout_composite_<cell>.json
```
- **SHIP threshold: `mean_composite` ≥ 0.80.**
- ⚠️ `heldout_composite_check.py` hardcodes `TARGET_COMPOSITE = 0.75` (line 52) and `heldout_composite_audit.py`'s docstring still says 0.75. **Update these constants to 0.80 (§10 re-derivation) before the run, or apply the 0.80 threshold manually** — the script's PASS/FAIL string is otherwise wrong. This is a required pre-run code touch.

**(2) Component diagnosis — is state the thing that moved:**
```
.venv-train/bin/python scripts/heldout_composite_audit.py \
    --checkpoint checkpoints/<cell-run-dir>/gemma-4-26B-A4B-it/checkpoint-<STEP> \
    --data-dir data/output/grpo/task_a --split validation \
    --n-prompts 290 --seed 42 \
    --output runs/preflight/heldout_audit_<cell>.json
```
- **`mean_state_acc` ≥ 0.833** (necessary to clear the bar via the state lever; **target-trending 0.85**).
- **Guardrail (from per-row output, rows where `n_gt_tools > 0`): tool-F1 ≥ 0.4623** and abstention (rows where `n_gt_tools == 0`, `state_acc`≈`1.0` proxy) not below 0.9494. Certify at `--n-prompts 2943` (§8.3) before ship.
- Sanity cross-check: `expected_perturn_score` from measured components should predict `mean_composite` within ~0.002 (as it did in §11.2); a large mismatch means a scoring/data drift to investigate before trusting the cell.

**Ship vs iterate:**
- **Ship a cell** iff gate `mean_composite` ≥ 0.80 **AND** tool-F1 guardrail holds (no regression) **AND** eval-loss curve shows no overfitting blow-up. Then (and only then) re-run `scripts/rft_headroom_probe.py` on the shipped checkpoint to see whether RL headroom re-opened — a separate future decision, not a blocker to shipping SFT-only.
- **Iterate to the next rung** if composite < 0.80. If a rung *regresses* state accuracy vs its parent, do not carry that rung's changed factor forward (§6 decision rule).

**Checkpoint-step note:** audit a **fixed step across cells** (e.g. the best-eval-loss checkpoint, or a fixed step count matched to the shorter 4,716-corpus schedule) so "which step" is not a hidden variable (§7.3). ckpt-1000 is step 1000 of a 3,426-step run on the old 9,131 corpus; the new 4,716 corpus at 3 epochs / effective-batch 8 is ≈1,769 optimizer steps, so "step 1000" is **not** the analogous point — select by eval-loss/epoch, not by raw step index.

---

## 10. Effort / cost per cell and recommended run order

**Cost basis (verified):** the 8.6 h/run figure in the brief is the **GRPO/HF-rollout** number (1000 steps × ~31 s/step); it does **not** apply here — there is no GRPO run this round. SFT does no in-loop generation. ckpt-1000's SFT was 3,426 optimizer steps on 9,131 convs at ~1.9 it/s micro-step throughput observed in `train.log`.

| Item | Estimate | Basis |
|---|---|---|
| **One SFT cell (train)** | **~2–3 h** | 3 epochs × 4,716 convs / eff-batch 8 ≈ 1,769 opt steps, seq 4096, Gemma-4 26B-A4B QLoRA on H100/H200; ≈half ckpt-1000's step count |
| Gate + component audit @ n=290 | ~25 min each (~50 min) | §11 ran the audit at n≈290 in 23 min |
| Guardrail re-audit @ n=2943 | ~4 h | ~10× the n=290 audit, decode-bound (HF path) |
| **Per cell, fast decision** | **~3–4 h** | SFT + two n=290 audits |
| **Per shipped cell, with full-power guardrail** | **~7–8 h** | + the n=2943 re-audit (only for the ship candidate) |

**Recommended order (cheapest/most-informative first):**
1. **Pre-flight (no GPU, ~0):** resolve §4.2 corpus provenance (blocking); update the 0.75→0.80 constants (§9). Optionally re-run `sanity_check_state_acc_drop.py` (CPU) once more to reconfirm the baseline population.
2. **C0** (control, ~3–4 h). Highest information per hour: it both anchors the ladder and directly measures the corpus-version effect. **It may clear the bar on its own** (the corpus fix removed role-corruption contamination that reached 22.4% of the pre-fix GRPO validation) — in which case the ladder ends here.
3. **C1** (`response_only`, ~3–4 h). One-variable masking test. Cheap (config flip on shipped code).
4. **C2** (decision-balance, contingency, ~3–4 h **plus** the net-new B-conv reweighting code). Only if C0/C1 fall short. Highest engineering cost (net-new sampler); deferred precisely for that reason.
5. **Ship candidate only:** full-power n=2943 guardrail re-audit (~4 h) before committing.

Total GPU budget: **~7–8 h if C0 clears**, **~11–12 h through C1**, **~15–16 h through C2** — versus a single 8.6 h GRPO run that §9 predicts would learn nothing.

---

## 11. Open questions / judgment calls (for reviewer sign-off before implementation)

1. ~~**BLOCKING — corpus provenance (§4.2)**~~ **RESOLVED (2026-07-22) — proceed with C0.** ckpt-1000 (`wandb.ai/wpawgasa/huggingface/runs/uklfswk5`, 2026-07-02, commit `ef837079`) trained on `task_a_cleaned` hash `8ef8681...` (131 files, ~228MB, 9,131 convs downstream) — a corpus that R12's own contamination analysis shows carried ~5% malformed-role corruption, and which is not recoverable in this environment (no GCS credentials). The current canonical `task_a_cleaned` (hash `3d6a4d3e...`, 5 files, ~110MB, 4,716 convs downstream) was confirmed to be the deliberate, cleaned R12 successor — not an accidental loss — and was re-verified this session by actually running `dvc repro -s -f` on `task_a_sft_clean` + `task_a_sft_splits` (not just trusting a hash-commit): output byte-identical, 0 rows altered. Tagged `task_a-corpus-v1-2026-07-22` / `task_a-corpus-v2-2026-07-22`. `dvc.lock` is now current (no longer stale per R12's outstanding item) and committed on `main`. **ckpt-1000 was trained on a corpus that is both gone and confirmably corrupted — C0 should be run, not just as an anchor, but as the first checkpoint on clean, verified, tagged data.**
2. **Judgment call — reward changes deferred, not adopted (§5.3).** I reject reward-weight and graded-match tuning this round because the gate is reward-independent and there is no RL run to carry a reward change. If the reviewer wants the *training reward* re-shaped now (e.g. to prepare for a future RL round), that is a separate spec.
3. **Judgment call — tool emission demoted to guardrail (§5.4).** State alone clears the bar and is far cheaper; tool emission is deterministically absent (§9) and needs data regeneration, not masking. If the reviewer prioritizes tool-F1, it becomes its own next-round factorial, not an addition to this one.
4. **Judgment call — "upsample advancing turns" reframed to "balance the stay/advance decision" (§5.2).** The naive one-class upsample could worsen the over-advance half of the bidirectional failure; I recommend decision-balance + level-weighting (§9.4) instead, and gate it behind C0/C1 rather than building it speculatively.
5. **Judgment call — B-conv over B-perturn (§5.2).** Whole-conversation reweighting keeps the SFT training unit constant and avoids confounding the masking result; a per-turn SFT slicer is higher-fidelity but a large separate variable, deferred.
6. **Non-blocking — §11.5 GPU reproduction.** Re-running the audit on the backup corpus to reproduce ~0.817 is a nice-to-have; the data-side evidence is already dispositive (§4.4). Not gating.
7. **Skepticism to preserve.** `response_only`'s benefit for *destination-selection* (as opposed to the retired tool-emission story) is unproven and mechanically weaker (§5.1). C1 is worth running because it is cheap, not because it is likely to win. Do not pre-commit to it.

**Implementation is gated on approval of this spec.** Net-new code required if the ladder reaches C2: a decision-balance reweighting path in `sft.py::_load_split` (B-conv), plus the 0.75→0.80 constant updates in the two gate scripts (§9). No training is launched, and no reward/data code is edited, until this spec is approved.
