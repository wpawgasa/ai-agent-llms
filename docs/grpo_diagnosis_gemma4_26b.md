# GRPO Training Diagnosis — Gemma-4-26B-A4B-IT (Cat A)

**Date:** 2026-05-19
**Run analyzed:** [`wpawgasa/huggingface/36ysj7u8`](https://wandb.ai/wpawgasa/huggingface/runs/36ysj7u8) (`glad-glade-34`)
**SFT base:** [`wpawgasa/huggingface/0bsx894b`](https://wandb.ai/wpawgasa/huggingface/runs/0bsx894b) → `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1656`
**Status:** GRPO run **killed at step 93/1000** (9.3%). No checkpoint saved (`save_steps=500`).

---

## TL;DR

The GRPO run failed for two compounding reasons:

1. **The SFT checkpoint is over-trained** — loss plateaued at epoch ~0.6 but training continued for 2.4 more epochs, sharpening the output distribution into a low-entropy mode.
2. **`beta=0.04` is too weak** to constrain a low-entropy reference policy — even small policy drifts produced KL spikes up to 131 and grad norms up to 4,929.

**Recommended fix:** restart GRPO from **`checkpoint-500`** (epoch 0.91, loss 0.385) with `beta=0.1`, `loss_type=grpo`, `max_completion_length=512`.

> **2026-05-19 update.** The pre-flight in Step 3 was run (`scripts/preflight_entropy_diag.py`).
> Results do **not** support over-training as the primary cause: ckpt-500 and ckpt-1656 produce
> nearly identical reward distributions, and the per-component decomposition shows the reward
> function is structurally incapable of producing within-group variance > ~0.05 from anything but
> the `length_band` tie-breaker. **See addendum below.** The "restart from ckpt-500 + beta=0.1"
> action remains defensible as hygiene but will not unblock training on its own.

---

## Evidence: SFT is Over-Trained

SFT loss curve from run `0bsx894b` (3 epochs, 1656 steps, LR 5e-5):

| % of training | Step | Epoch | Loss |
|--------------:|-----:|------:|-----:|
| 10% | 170 | 0.31 | 0.542 |
| 20% | 340 | 0.62 | **0.308 ← elbow** |
| 30% | 500 | 0.91 | 0.385 |
| 50% | 830 | 1.50 | 0.376 |
| 70% | 1160 | 2.10 | 0.249 |
| 100% | 1650 | 2.99 | **0.291** |

- Loss bottomed out at epoch ~0.6 and stayed flat for the remaining 2.4 epochs.
- Mean loss at epoch 2.0 = 0.297; mean of final 50 steps = 0.304 (slightly worse).
- Net result: ~1,400 steps of pure distribution-sharpening with no genuine learning.

A peaked SFT policy is the textbook cause of GRPO collapse: 8 generations from a low-entropy model are near-identical → no within-group reward variance → no useful advantage signal → noise-driven policy jerks.

## Evidence: GRPO Run Symptoms Match

| Symptom | Measured | Mechanism under sharp ref-policy |
|---------|----------|----------------------------------|
| `reward_std` within group | 0.04–0.06 | 8 rollouts ≈ identical → similar rewards |
| Reward trend | slope −0.000168/step (flat) | No usable advantage signal |
| KL @ step 11 | **131.2** | Per-token logprob ratios blow up off sharp ref |
| Grad norm @ step 11 | **4,929** | Same — BNPO importance-ratio term |
| Grad norm @ step 51 | 2,869 | Recurring instability throughout |
| Completion length @ max (256) | 16.1% of samples | Truncation noise on top |

Top KL spikes all occurred during warmup, well before LR reached peak:

| Step | LR | KL | Grad Norm | Loss |
|------|-----|-----|-----------|------|
| 6  | 5e-7 | 79.2 | 736 | 3.17 |
| 11 | 1e-6 | **131.2** | **4,929** | 5.25 |
| 20 | 1.9e-6 | 45.1 | 428 | 1.80 |
| 28 | 2.7e-6 | 24.4 | 599 | 0.98 |
| 51 | 5e-6 | 10.0 | **2,869** | 0.40 |

---

## GRPO Config Issues

| Setting | Used | Issue | Recommended |
|---------|------|-------|-------------|
| `loss_type` | `bnpo` | Numerically sensitive when ref-policy entropy is low | `grpo` |
| `beta` | `0.04` | Insufficient KL constraint | `0.1` (or `0.15`) |
| `max_completion_length` | `256` | 16% truncation rate | `512` |
| `save_steps` | `500` | Lost everything when killed at step 93 | `50` (first run after fix) |
| `use_vllm` | `false` | Correct — Gemma-4 R9 fallback (~12 s/step) | Keep |
| `num_generations` | `8` | OK in isolation, but combined with low reward_std = weak signal | Keep at 8 |
| `warmup_ratio` | `0.05` (50 steps) | LR ramp itself is fine; spikes happen at LR=5e-7 anyway | Keep |

---

## Recommended Action Plan

### Step 1 — Restart GRPO from `checkpoint-500`

The four available SFT checkpoints in `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/`:

| Checkpoint | Step | Epoch | Loss | Suitability |
|------------|-----:|------:|-----:|-------------|
| **`checkpoint-500`** | 500 | 0.91 | 0.385 | ✅ Just past loss elbow; highest entropy among trained checkpoints |
| `checkpoint-1000` | 1000 | 1.81 | 0.287 | Already plateaued — only marginally less sharp than final |
| `checkpoint-1500` | 1500 | 2.72 | ~0.30 | Same as final, effectively |
| `checkpoint-1656` (final) | 1656 | 2.99 | 0.291 | ❌ What this run used — over-sharpened |

Use `checkpoint-500` as the GRPO base.

### Step 2 — Config changes

Minimum diff for the next GRPO run:

```yaml
# was 0.04 — too weak for low-entropy ref policy
beta: 0.1

# was "bnpo" — switch to standard GRPO loss until base is stable
loss_type: grpo

# was 256 — eliminates 16% truncation noise
max_completion_length: 512

# was 500 — checkpoint frequently while validating the fix
save_steps: 50
```

### Step 3 — Optional 2-minute pre-flight: entropy diagnostic

**Status: done (2026-05-19). See "Addendum: pre-flight outcome" below.**

Before launching the full GRPO run, confirm the SFT-too-good hypothesis by comparing
output entropy between `checkpoint-500` and `checkpoint-1656`:

- Generate 16 completions per prompt (temperature=1.0) on ~50 held-out Cat-A prompts
- Measure: pairwise BLEU/edit-distance between completions, or token-level entropy
- Expected: `checkpoint-500` should show 2-3× higher diversity (and higher reward
  variance when scored with the Cat-A reward function)

If the diversity gap is small, the SFT-too-good hypothesis is wrong and we need to
look at the reward function itself (likely culprit: scoring near-constant per prompt).

---

## Going Forward: Shorten Future Cat-A SFTs

The plateau at epoch 0.6 says this dataset doesn't support 3 epochs. For future
Cat-A SFTs on similar data:

- Set `num_train_epochs: 1`, or
- Add `eval_steps: 100` with a held-out split and early-stop on plateau, or
- Reduce LR warmup and use a faster decay so the last steps don't keep peaking
  the distribution after loss has converged

This applies to other Cat-A winners that share a similar data scale, not just
Gemma-4.

---

## Open Questions

- **Why does the reward function produce reward_std ≈ 0.05 even on a sharper-than-ideal
  policy?** ~~The current Cat-A reward (`reward_business_logic`) is a 5-component weighted
  sum. If one component dominates (e.g., format_compliance always passing), the effective
  variance is squashed. Worth printing per-component scores on a sample batch.~~
  **Answered (2026-05-19): see addendum.** 40% of reward weight is pinned to constants
  on every group; only `length_band` (10%) varies smoothly. The 0.05 reward_std is the
  length_band contribution, exactly as predicted.
- **Should we monkey-patch `unsloth.models.vision.VLLM_SUPPORTED_VLM` to enable vLLM
  rollouts?** (See CLAUDE.md Risk R9.) At ~12 s/step with HF generate, a 1000-step
  run is ~8.6h. With vLLM colocate we'd expect ~3-4× speedup. Trade-off: risk of
  failing deeper in vLLM engine init.
- **Is BNPO ever the right choice here?** BNPO's bias-reduction matters more once
  the policy is already stable. May be worth revisiting after the base GRPO run
  is converging cleanly.

---

## Addendum: Pre-Flight Outcome and Revised Diagnosis (2026-05-19)

### What was run

5 prompts × 4 completions × 2 checkpoints via `scripts/preflight_entropy_diag.py`
at GRPO sampling settings (T=1.0, top_p=0.95, max_new=512). Held-out prompts drawn
deterministically from `data/output/grpo/task_a/validation.jsonl` and deduplicated by
first-user-turn. Full output saved to `runs/preflight/smoke_5x4_v2.json`.

Generation cost ~10 min on H100 (two model loads + 40 completions × 2 ckpts).

### Aggregate result — both checkpoints look the same

| metric | ckpt-500 | ckpt-1656 |
|--------|----------:|-----------:|
| mean_reward_std | 0.0620 | 0.0500 |
| median_reward_std | 0.0035 | 0.0180 |
| mean_reward | 0.6733 | 0.6689 |
| frac_collapsed_groups (`std < 0.01`) | 0.60 | 0.20 |
| mean unique completions / group | 3.0 / 4 | 3.4 / 4 |

The doc's prediction was ckpt-500 → "2-3× higher diversity." At N=5 we see the opposite
direction on 3 of 5 metrics (median std, frac collapsed, unique completions). The slight
edge in mean_reward_std for ckpt-500 (0.062 vs 0.050) comes from two outlier prompts —
strip those and ckpt-500 looks *more* mode-collapsed. **The SFT-too-good hypothesis does
not reproduce at this scale.** Caveat: N=5 cannot rule out a real 1.2-1.5× gap; what it
*does* rule out is the 2-3× gap the diagnosis predicted.

### Per-component decomposition — the real bottleneck

Within-group spread (max − min) for each component across each prompt's 4 completions,
ckpt-1656:

| component (weight) | per-prompt spreads | varies? |
|--------------------|----:|:----|
| state_transition (0.30) | `[1.0, 0, 0, 0.5, 0]` | 2 / 5 prompts |
| tool_call_f1 (0.30) | `[0, 0, 0, 0, 0]` | 0 / 5 prompts |
| **chain_propagation (0.10)** | **`[0, 0, 0, 0, 0]`** | **dead — always 1.0** |
| **format_compliance (0.10)** | **`[0, 0, 0, 0, 0]`** | **dead — always 1.0** |
| **task_completion (0.10)** | **rescaled out (all rows had `terminal_reached=False`)** | **dead in 85.7% of train rows** |
| length_band (0.10) | `[0.26, 0.02, 0.26, 0.32, 0.0]` | 4 / 5 prompts |

What this means:

1. **40% of reward weight is pinned to constants.** `chain_propagation_score` returns 1.0
   when `n_pairs <= 1` (`reward_utils.py:63-64`) — and every GRPO row is a single assistant
   turn, so `n_pairs` is always ≤1. `format_compliance_check` returns 1.0 unless tool-call
   tags mismatch or a stack trace appears, which an SFT'd model essentially never produces.
   Both are structurally dead in this setup.
2. **`task_completion` (10%) is rescaled out on 85.7% of training rows.** `_load_grpo_jsonl`
   sets `terminal_reached=True` only on the *final* assistant turn of conversations that
   reach their terminal — but emits one row per `user → assistant` boundary (≈5.86 rows
   per conversation, 14,656 rows total in train). Count verified directly:
   - Train rows: 14,656; terminal_reached=True: 2,089 (14.3%).
   - 2,502 conversations total; 2,089 reach terminal (83.5%). The 1-of-many rows that
     does count is hidden in a sea of rows where this component is silenced.
3. **The 60%-weighted main signal is binary on this data.** `state_transition` GT length
   is 1 on 78% of rows and 0 on 22% (so the score is one of {0, 0.5, 1.0}). Tool-call
   GT length is 0 on 61% of rows and 1 on 39% — so F1 is 1.0 if pred matches exactly,
   0.0 otherwise. Combined, the 60% weight produces ≤6 discrete reward signatures per
   prompt, and on most prompts all 4 completions land in the same bucket.
4. **Only `length_band` (10%) varies smoothly.** Its 0.02-0.32 within-group spread × 0.10
   weight ≈ 0.002-0.032 per-prompt reward std contribution — matching the observed
   `mean_reward_std ≈ 0.05` in this pre-flight and in the killed GRPO run.

The ADR in `reward_business_logic.py:11-24` already documents that the other components
are coarse; the length_band was *added* specifically as a tie-breaker because of this.
The pre-flight confirms it's now load-bearing: nothing else is producing variance.

### Concrete examples

**Prompt 0 — the "binary cliff" pattern.** Ground truth: state = `UNDERSTAND_NEEDS → SEARCH_CATALOG`,
tool = `search_products(...)`. Four completions from ckpt-1656:

```
i=0  state=0.0  tool=0.0  len=271  reward=0.272   "[STATE: UNDERSTAND_NEETS → SEARCH_CATALOG]..."
i=1  state=0.0  tool=0.0  len=252  reward=0.269   "[STATE: UNDERSTAND_NEETS → SEARCH_CATALOG]..."
i=2  state=1.0  tool=0.0  len=405  reward=0.631   "[STATE: UNDERSTAND_NEEDS → SEARCH_CATALOG]..."
i=3  state=0.0  tool=0.0  len=257  reward=0.270   "[STATE: UNDERSTAND_NEETS → SEARCH_CATALOG]..."
```

Three of four completions misspelled `NEEDS` as `NEETS` and got `state=0`. The fourth
spelled it correctly and got `state=1.0`. None of the four produced a tool call. The
0.36-point reward gap is driven by a tokenizer artifact, not policy quality. This is
the binary cliff in action — the only signal GRPO sees is "lucky on spelling".

**Prompt 4 — the "TERMINAL stub" pattern.** All 4 completions are byte-identical:

```
[STATE: TERMINAL → TERMINAL]
Handling booking_payment in state TERMINAL.
```

Reward 0.902 each (max possible is 1.0). This is a trivial stub that exploits the
reward: it produces a syntactically valid state annotation, calls no tools (matching
the 61%-of-rows case where GT has 0 tools), and stays bounded — so state, tool, chain,
and format all pin at 1.0 while length_band gives 0.02. A genuine attempt at the task
that gets one component wrong (like Prompt 0 above) scores ~0.27. **The reward actively
incentivizes stub outputs over real attempts.**

### Revised diagnosis

The killed run failed for **one** primary reason, not two:

> The Cat-A reward function cannot produce within-group reward variance large enough
> for GRPO advantage signal on this single-assistant-turn row format.

SFT over-training (the original TL;DR's first reason) is real but secondary. `beta=0.04`
being too weak (the original second reason) is true mechanically but moot if the reward
provides nothing to optimize against.

### Three Fix Candidates

**(a) One row per conversation (single-turn target).** Emit one row per conversation,
with the final user-preceded assistant turn as the target.

> **Framing correction (2026-05-19):** an earlier draft of this section implied (a) was
> the "next variance lever" after (b)+(c). That is wrong. Cheap-(a) is a *reward-calibration
> / train-deploy-alignment* fix, not a within-group-variance fix. The variance lever is
> (a\*) below — multi-turn rollouts — which is the deferred variant. See the corrected
> retry plan at the end of this addendum.

- **What this DOES fix:**
  - **Train/deploy distribution mismatch.** The model is currently trained on arbitrary
    mid-conversation slices; at deployment it sees the full preceding context and is
    expected to drive to terminal. Per-conversation rows align the two distributions.
  - **Effective weight on workflow completion.** `task_completion` is nominally 10% of
    the reward, but it's rescaled out on 85.7% of rows (`terminal_reached=False`).
    Effective weight on the "did this finish the workflow" objective today is ~1.4%.
    After (a): `task_completion` fires on 83.5% of rows; effective weight ~8.4%.
- **What this does NOT fix:**
  - `chain_propagation` stays dead — single assistant turn still has `n_pairs ≤ 1`.
    Per-conversation rows just pick the *last* turn; they don't make a turn multi-link.
  - **Within-group `reward_std` stays bound by the same components.** `task_completion`
    adds a 10% × {0,1} signal — same binary cliff as state/tool, just gated on a
    different condition. If all 8 rollouts reach (or fail to reach) the terminal,
    `task_completion` contributes zero to the group spread. The variance space has the
    same shape as today; only the *trigger* for one component moves.
  - **The "model produces near-identical completions" problem.** Prompts 1, 2, 4 in
    the smoke had 4-of-4 byte-identical completions. That's an SFT-policy issue, not
    a row-layout one. Same prompt + same policy = same completions, regardless of
    whether the row was emitted per-turn or per-conversation.
- **Code scope:** ~30 lines in `_load_grpo_jsonl` (`training/grpo.py:105-230`).
  Drop the per-turn loop; keep only the last user-preceded assistant turn per conversation.
- **Risk:** 5.86× smaller dataset → ~6× fewer effective optimizer steps for the same
  `training_steps`. Loss of state-machine-adherence signal from intermediate turns
  (most of which were legitimate training examples even though they weren't terminal).

**(a\*) Multi-turn rollouts with simulator (deferred — separate engineering project).**
The model produces an entire conversation trajectory; the reward scores the trace
end-to-end. This is the *real* variance lever, and it's the one that gets confused
with (a) when "row per conversation" is read loosely.

- **What this fixes that cheap-(a) doesn't:**
  - `chain_propagation` becomes computable on every rollout (multi-link trajectories).
  - `task_completion` becomes a per-trajectory signal and varies meaningfully across
    the K rollouts — different trajectories diverge at every turn, with stochasticity
    compounding across turns.
  - Within-group `reward_std` should jump substantially: the variance space is N-fold
    larger (one decision per turn × N turns per trajectory).
  - Best alignment with deployment: the orchestrator IS a multi-turn agent.
- **Code scope:** rollout simulator that mocks user replies (teacher LLM or scripted
  replays) and tool responses (the data's recorded tool responses work for replay;
  novel paths need a fallback). `max_completion_length` 8k–16k. New tested surface area.
  ~1 week of engineering, plus meaningfully more rollout compute (step time grows
  roughly with conversation length).
- **Risk:** simulator bugs that silently corrupt training signal. The simulator itself
  needs unit tests and held-out replay validation.

**(b) Drop saturated components.** `chain_propagation` and `format_compliance` are dead
on this row format. Reweight to:

```
state_transition  0.40  (was 0.30)
tool_call_f1      0.40  (was 0.30)
task_completion   0.10  (unchanged; still rescaled out 85.7% of the time)
length_band       0.10  (unchanged — the tie-breaker we depend on)
```

- **Code scope:** ~5 lines in `reward_business_logic.py` (change weight constants;
  remove the two dead components from the per-row computation; rescale the
  `terminal_reached=False` path).
- **Effect on reward variance:** marginal. The saturated components contributed
  0 variance — dropping them changes nothing about the within-group spread. What
  it *does* do is reweight signal toward the binary state/tool components, so a
  state-correct completion now scores +0.40 above a state-wrong one (vs +0.30
  today). Slightly stronger gradients when they exist.
- **Risk:** none — pure cleanup. Should land regardless of which other fixes ship.

**(c) Graded state/tool components.** Replace the binary cliffs with continuous metrics.

- `tool_call_f1`: the current `compute_ast_f1` (`tool_call_f1.py:174-208`) uses
  `_is_subtree_match` which is all-or-nothing on argument values. Replace with
  per-argument F1: name-match counts as 0.4, then per-key correct counts pro-rata
  for the remaining 0.6. A "right tool, one of two args wrong" completion now
  scores 0.7 instead of 0.0.
- `state_transition`: `_partial_state_match` already has half-credit for `from`-only
  matches. Could add finer-grained scoring (e.g., correct state-pair but wrong
  direction = 0.3; ill-formed annotation but plausible target = 0.2).
- **Code scope:** new `compute_argument_graded_f1` next to `compute_ast_f1` (keep
  the strict one for eval/benchmarking — they should not converge to avoid masking
  the held-out gap). New `_graded_state_match` next to `_partial_state_match`. Plus
  tests. Total ~150 lines + tests; ~half day.
- **Effect on reward variance:** highest leverage. Where today 4 completions all
  score 0 or all score 1 (60% weight), they would now spread across a range based
  on argument-level correctness. Expected mean_reward_std jumps from ~0.05 to ~0.15+.
- **Risk:** divergence between training reward and benchmark eval. Must keep them
  separate and validate the held-out score on the strict metric still improves
  during training.

### Recommendation

Land **(b) first as a no-cost cleanup**, then **(c) as the variance fix where signal
exists**, in this order:

1. **(b) Reweight away from dead components.** Pure config change; immediate.
2. **(c) Graded tool/state scoring.** Unlocks variance on prompts where there's signal
   to extract (per-argument matches, near-miss state transitions). Won't manufacture
   variance on prompts where the policy produces identical completions.
3. *Then* retry GRPO. Keep the doc's earlier action items (restart from ckpt-500 with
   `beta=0.1`, `loss_type=grpo`, `max_completion_length=512`, `save_steps=50`) — they're
   still good hygiene, but only meaningful once (b)+(c) land.
4. **(a) is worth doing for train/deploy alignment and effective task_completion weight**
   (1.4% → 8.4%), independent of the variance discussion. But (a) is **not** the next
   variance lever — see the corrected (a) write-up above. If (b)+(c) don't move
   `frac_reward_zero_std` enough, the next real lever is **(a\*) multi-turn rollouts**,
   which is a separate ~1-week engineering project.

Decision tree if (b)+(c) underperform:
- `frac_reward_zero_std` high + completions visibly diverse: reward still mis-resolved,
  iterate on (c).
- `frac_reward_zero_std` high + completions byte-identical: policy entropy is the
  bottleneck. Two paths: (i) restart from a less-trained SFT checkpoint or rerun SFT
  with `num_train_epochs=1`; (ii) commit to (a\*) multi-turn rollouts. Cheap-(a) won't
  help with this specific failure mode.

Estimated time-to-retry-able: ~half day to land (b)+(c) + tests.

---

## Implementation Outcome (2026-05-19): (b)+(c) Landed

Code changes:

- `reward_business_logic.py`: dropped `W_CHAIN_PROPAGATION` and `W_FORMAT_COMPLIANCE`; new
  weights `state=0.40, tool=0.40, task_completion=0.10, length_band=0.10`. Added
  `_graded_state_match` (1.0 / 0.5 / 0.5 / 0.3 / 0.0 for exact / from-only / to-only /
  reverse / no-overlap). `_partial_state_match` retained for benchmark eval.
- `eval/tool_call_f1.py`: added `compute_argument_graded_f1` and `_pair_match_graded`.
  Per-pair score: `name_weight=0.4` + `0.6 * (matched_args / total_gt_args)`. Aggregated
  via greedy best-first 1:1 assignment + soft F1. `compute_ast_f1` retained for eval.
- `reward_utils.py`: added `graded_tool_call_f1` wrapper. `tool_call_f1` (strict) retained.
- `configs/training/grpo_cat_a.yaml`: weights section updated with rationale comment.
- `tests/unit/test_reward_functions.py`: 13 new known-answer tests covering edge cases
  (exact / partial / reverse / hallucinated / empty), plus dominance assertions
  (`graded >= strict` on partial matches; `graded == strict` on clean cases). All
  549 unit tests pass.

### Verification: re-scoring the smoke completions with the new reward

Same 40 completions, OLD vs NEW reward, side-by-side:

```
                                        ckpt-500                          ckpt-1656
metric                       OLD (saved)      NEW (now)         OLD (saved)      NEW (now)
mean_reward_std                  0.0697           0.0697            0.0485           0.0436
median_reward_std                0.0143           0.0143            0.0121           0.0121
mean_reward                      0.6597           0.5931            0.6839           0.6311
frac_collapsed (<0.01)             0.40             0.40              0.40             0.40
```

The aggregate `mean_reward_std` numbers barely moved at N=5. But the per-prompt
distribution tells a more interesting story — the graded reward redistributes the
variance rather than amplifying it uniformly:

```
ckpt-500 per-prompt reward_std (OLD → NEW)
  idx 0  0.158 → 0.103     (typo cliff: graded gives the NEETS typo credit for matching `to`)
  idx 1  0.004 → 0.004     (collapsed group: nothing to extract)
  idx 2  0.014 → 0.014     (near-collapsed: nothing to extract)
  idx 3  0.172 → 0.228     (mixed group: graded F1 surfaces partial-arg matches)
  idx 4  0.000 → 0.000     (TERMINAL stub: still identical)
```

Two observations:

1. **Where graded scoring has something to discriminate, it produces meaningfully more
   variance.** Prompt 3 (ckpt-500) went from std=0.172 to 0.228 (+33%), driven by
   per-argument tool F1 distinguishing completions that the strict subtree-match
   collapsed to 0.0.
2. **Where the model outputs are nearly identical, graded scoring CAN'T manufacture
   variance.** Prompts 1, 2, 4 are unchanged: when 4 completions agree on every
   component, no metric refinement helps. This is where the SFT entropy concern from
   the original diagnosis still bites.

Also notable: `mean_reward` dropped by ~0.07 on both checkpoints. Expected — the dropped
components were contributing a flat +0.20 (×0.10 each from chain_propagation and
format_compliance), which is exactly the rescaling math. The reward is now actually
discriminating instead of adding constants.

### Calibration win that doesn't show in std

The NEETS typo on idx 0 used to score 0.272 (typo) vs 0.631 (correct spelling), a
0.36-point gap entirely attributable to a tokenizer artifact. After (c2), the typo
scores 0.272 vs 0.519, a 0.25-point gap — still penalized for getting `from` wrong,
but credited for getting `to` right. **The reward gradient now reflects "got partially
right" instead of "spelled it perfectly".** This kind of calibration improvement is
the point even when total variance is unchanged.

### Updated retry plan

The N=5 verification suggests **(b)+(c) alone may not produce dramatically higher
reward_std on this SFT checkpoint** — model output diversity is also a contributing
bottleneck. Two ways to read this:

- *Optimistic:* idx 3 shows the graded reward unlocks signal where it exists; with
  more diverse prompts at higher N the aggregate gain should compound.
- *Pessimistic:* the SFT-over-training concern is real and partially independent —
  even a perfect reward can't grade outputs the policy never produces.

Concrete next step before committing to a 1000-step GRPO run:

1. Restart from ckpt-500 (per original recommendation) with the new reward, run a
   ~50-step diagnostic. Check `frac_reward_zero_std` (was ≈1.0 in the killed run):
   - **Drops below ~0.5:** proceed to full 1000 steps.
   - **Stays high AND completions visibly diverse in W&B samples:** the reward still
     under-resolves — iterate on (c) (e.g., bump `name_weight` down toward 0.2 to
     spread arg-match scores wider, or add finer-grained state credit).
   - **Stays high AND completions byte-identical within groups:** the bottleneck is
     policy entropy, not reward resolution. (b)+(c) cannot fix this. Two paths:
     (i) restart from a less-trained SFT checkpoint or rerun SFT with `num_train_epochs=1`
     and early stopping; (ii) commit to **(a\*) multi-turn rollouts**. Cheap-(a)
     (per-conversation single-turn rows) does NOT fix this failure mode — same prompt,
     same policy, same completions.
2. Keep `beta=0.1`, `loss_type=grpo`, `max_completion_length=512`, `save_steps=50` as
   the doc originally recommended.
3. Land cheap-(a) as a separate, independent improvement once the GRPO run is stable.
   It's a reward-calibration win (effective task_completion weight goes from ~1.4% to
   ~8.4%) and a train/deploy alignment win, but should not be conflated with the
   variance question.

---

## Retry Outcome (2026-05-21): Run `t82y64hc` killed — two new root causes

**Run analyzed:** [`wpawgasa/huggingface/t82y64hc`](https://wandb.ai/wpawgasa/huggingface/runs/t82y64hc) — the first GRPO run launched after (b)+(c) landed. Config `grpo_cat_a.yaml` (`beta=0.1`, `loss_type=grpo`, `max_completion_length=512`, 1000 steps, HF rollouts via the Gemma-4 R9 fallback). **Killed manually at step 100/1000.**

### Symptoms — (b)+(c) did not unblock training

| Symptom | Measured |
|---------|----------|
| Reward trend | flat — 0.40 (step 1) → 0.26 (step 100) |
| `reward_std` within group | ≈ 0.003 |
| `frac_reward_zero_std` | hit **1.0** on steps 18, 50, 89 |
| `train/kl` | spiked 0.7 → **37.9** → **40.2** |
| `train/grad_norm` | spiked to **1126** (clipped to `max_grad_norm=1`) |

The step-100 W&B completions table is conclusive: 8 completions to one prompt scored **0.2570–0.2646** (spread 0.0076). The completions were genuinely different text (188–229 chars) — the reward simply could not tell them apart.

### Root cause 1 — prompt truncation

`max_prompt_length` was never set in `grpo_cat_a.yaml`, so TRL's default **512** applied. The enriched system prompts (workflow script + tool-schema JSON + growing history) are 3,000–5,500+ tokens. The model saw ~10% of its input and could not perceive its workflow state — every rollout collapsed to a generic `[STATE: … → TERMINAL]`. At step 100 every completion emitted `[STATE: RESOLVE → TERMINAL]`, and `RESOLVE` was not even a state in that prompt's workflow.

### Root cause 2 — `length_band` was load-bearing reward hacking

(b)+(c) deliberately kept `length_band` as the tie-breaker. This run proved that tie-breaker is the *only* live signal: the entire 0.0076 within-group spread was `length_band` reacting to completion length — `(229−188 chars)/300 × 0.5 × 0.10 ≈ 0.0068` matches exactly. GRPO normalized that length noise into ±1.5 advantages. `length_band` is task-irrelevant; it cannot teach workflow correctness, and with the model truncation-blinded the state/tool components stayed constant across the group, leaving length as the sole variance source.

### Fix landed (2026-05-21)

- **`max_prompt_length: 7680`** added to `grpo_cat_a.yaml` + `grpo_cat_a_diagnostic.yaml`. 7680 + 512 completion = 8192 = the `max_seq_length` hardcoded in `grpo.py:384`. (Chose to cap the prompt rather than raise `max_seq_length` and pay its KV-cache VRAM cost.)
- **Dropped `length_band`** entirely (removed `_length_band_score`, `LENGTH_BAND_*`). Added **`transition_legality` (weight 0.10)** — `transition_legality_score` in `reward_utils.py` scores the fraction of emitted `[STATE: X→Y]` that are legal edges in the workflow graph. Complementary to `_graded_state_match` (which asks "is it the *expected* transition"): legality asks "does this edge exist at all" → directly penalizes hallucinated states like the `RESOLVE→TERMINAL` collapse.
- Legal-edge set sourced from **ground truth, not the prompt** — `grpo.py::_load_grpo_jsonl` parses `workflow_graph.transitions` into a `valid_transitions` field on each row, so legality scoring stays correct even when a late-turn prompt truncates at 7680.
- **Placeholder-arg sanitizer** — 17% of dataset tool calls carry a frozen `{"placeholder": "value"}` arg stub; `_strip_placeholder_args` blanks it so scoring degrades to name-only matching instead of rewarding the literal placeholder.
- Reward aggregation rewritten as a weighted **active-component mean** — `task_completion` drops out when `terminal_reached=False`, `transition_legality` drops out when `valid_transitions` is empty; remaining weights renormalize. Generalizes the old `terminal_reached` rescale path.
- New weights: `state_transition 0.40 / tool_call_f1 0.40 / task_completion 0.10 / transition_legality 0.10`.

### Verification

70 reward unit tests pass — added `TestTransitionLegality`, `TestStripPlaceholderArgs`, and `TestRewardWithinGroupVariance` (a regression gate asserting within-group spread > 0.2). On a real Task A row, four completions of differing quality scored:

| completion | reward |
|------------|-------:|
| correct transition + correct tool | 1.000 |
| valid-but-wrong edge, no tool | 0.556 |
| hallucinated states | 0.444 |
| empty | 0.444 |

**Within-group spread 0.556 — versus ≈0.003 in the killed run.** Note: on no-tool turns (61% of rows) `graded_tool_call_f1([], [])` returns 1.0, making the tool component a within-group constant; `state + legality` (0.50 combined) carry the variance. Acceptable — the spread is healthy.

### Updated retry plan

1. Run `grpo_cat_a_diagnostic.yaml` (50 steps) from the SFT checkpoint. Check W&B: `train/reward_std` materially above 0.003, `frac_reward_zero_std` near 0, `train/kl` not spiking into the 30–40 range.
2. Keep `beta=0.1`, `loss_type=grpo`, `max_completion_length=512`, the `save_steps` cadence.
3. **Still open, not addressed by this fix:** SFT over-training (policy entropy — same prompt + same policy = same completions, regardless of reward resolution), the one-prompt-per-step batch geometry (`generation_batch_size` 8 ÷ `num_generations` 8 = 1 unique prompt per step), and multi-turn rollouts (a\*). If the diagnostic still shows byte-identical completions within groups, the bottleneck is policy entropy — follow the decision tree in the prior addendum.

---

## Diagnostic Outcome (2026-05-25): Run `5a5w4jqr` — Numerics Tamed, Entropy Collapse Confirmed

**Run analyzed:** [`wpawgasa/huggingface/5a5w4jqr`](https://wandb.ai/wpawgasa/huggingface/runs/5a5w4jqr) (`usual-dew-42`) — the 50-step diagnostic launched after the 2026-05-21 reward redesign (`grpo_cat_a_diagnostic.yaml`, `beta=0.1`, `loss_type=grpo`, `max_prompt_length=7680`, `max_completion_length=512`, HF rollouts via Gemma-4 R9 fallback, ckpt-500 base, length_band dropped, transition_legality added).
**Status:** Ran to completion (50/50, 27.4 min). **No crash, no NaN, no learning.**

### TL;DR

The 2026-05-21 fixes stabilized the numerics but exposed the underlying failure: the policy is in an **entropy-collapsed stub attractor**, and the redesigned reward provides no within-group variance to escape it. **GRPO hyperparameter tuning has hit its ceiling on this SFT base.** Next action moves to the SFT side (re-train with `num_train_epochs=1`) or to (a\*) multi-turn rollouts.

### Headline contrast vs the prior killed run

| Symptom | `t82y64hc` (killed @100) | `5a5w4jqr` (this run, 50 steps) |
|---|---|---|
| KL spikes ≥30 | 4× | **0×** ✅ |
| Grad norm >500 | 7× | 1× (step 19: 784) ✅ |
| Max loss | 4.02 | 0.57 ✅ |
| `frac_reward_zero_std = 1.0` | ~3% of steps | **70% of steps (35/50)** ❌ |
| Byte-identical 8/8 within group | Rare | Steps 6, 7, 42, 46, **50** |
| Reward trajectory | Flat 0.40→0.26 | Flat, oscillating 0.33–1.0 |

The KL/loss-explosion problem is solved. A *different* failure surfaced: the policy converges to a fixed string with zero rollout variance.

### The smoking-gun completions table — step 50

At the final step, all 8 GRPO rollouts for the IT-troubleshoot prompt produced the identical 95-character string at T=1.0, top_p=0.95:

```
[STATE: CONFIRM_ACTION → ESCALATE]
Handling it_troubleshoot_escalation in state CONFIRM_ACTION.
```

Reward 0.333 on every rollout; advantage 0.0 on every rollout. This is the "TERMINAL stub" pattern from the 2026-05-19 addendum (Prompt 4) generalized into a per-domain attractor. The reward gives this stub:
- 0 from `tool_call_f1` (no tool emitted)
- partial from `state_transition` (well-formed annotation, wrong transition)
- 0 from `task_completion` (terminal not reached → rescaled out)
- partial from `transition_legality` (`ESCALATE` is a legal edge somewhere in the graph)

Net 0.333 — low enough that GRPO *should* penalize it, but `reward_std = 0` zeroes out the advantage, so there's no gradient toward escape. The stub is a stable fixed point.

### Mid-run vs end-of-run — the collapse happened during training

Same run, ~step 25 completions table (table file `completions_181_*.json`), same prompt schema — 4 sample rollouts:

| len | reward | content sketch |
|----:|------:|---|
| 344 | 0.556 | `[STATE: ACTIVATE_SERVICE → TERMINAL]` + Thai summary, no tool |
| 465 | 1.000 | Same state + actual `<tool_call>{"name":"activate_roaming",...}</tool_call>` + closing |
| 341 | 0.556 | Same state + Thai summary, no tool |
| 238 | 0.556 | Same state + shorter Thai summary, no tool |

At step 25 the policy still produced diverse outputs of varying lengths, and one rollout in 4 actually emitted a tool call and scored 1.0. By step 50 all 8 rollouts are byte-identical. **GRPO trained the policy *toward* the collapse over 25 steps.** With nonzero `frac_reward_zero_std = 1.0` events at 35/50 steps and `length_band` no longer providing escape variance, the optimizer drifted the policy into the lowest-divergence solution that scored ≥0.33.

### Why removing `length_band` made `frac_reward_zero_std` 25× worse

`length_band` was the only smooth (continuous-valued) reward component. The remaining four components are all discrete:
- `state_transition`: scores in {0.0, 0.3, 0.5, 1.0} per row
- `tool_call_f1` (graded): rational fractions over arg counts; mostly 0.0 or 1.0 in practice
- `task_completion`: rescaled out on ~86% of rows; otherwise {0, 1}
- `transition_legality`: fraction of emitted `[STATE: X→Y]` annotations that are legal edges; granular but typically 0/1 on stub-style outputs

When 8 rollouts cluster (e.g., all emit the same state-stub), every component lands in the same discrete bucket → `reward_std` = exactly 0 → GRPO advantage = 0 → no learning. `length_band` previously masked this by injecting 0.02–0.06 of always-present length noise, *but* that noise was task-irrelevant and was itself the prior collapse mechanism. There is no reward redesign that simultaneously (a) removes length-noise hacking and (b) maintains within-group variance on a policy that produces identical completions. The signal must come from the policy.

### Updated decision-tree position

We are now at the leaf of the prior addendum's tree:

> `frac_reward_zero_std` high + completions byte-identical → **policy entropy is the bottleneck.** (b)+(c) cannot fix this.

The path is (i) less-sharp SFT, or (ii) (a\*) multi-turn rollouts. Cheap-(a), more GRPO hyperparameter tuning, and further reward iteration are all ruled out by this run.

### Recommended next steps (in execution order)

1. **(Cheap, 1 hour) Re-run the pre-flight entropy diagnostic at scale.** `scripts/preflight_entropy_diag.py` with N=20+ prompts × K=8 rollouts, scoring under the *new* reward. Confirm `mean unique completions / group < 3` on ckpt-500 before committing to a re-SFT. If diversity is acceptable but reward_std is still ≈0, the bottleneck is reward resolution, not entropy — return to (c)-iteration.
2. **(Required if step 1 confirms) Re-train SFT with `num_train_epochs=1` + early-stop at the loss elbow.** The 2026-05-19 addendum already located the elbow at step 340 / epoch 0.62 / loss 0.308. ckpt-500 (epoch 0.91) is past it. Config diff to `configs/training/sft_cat_a.yaml`:
   ```yaml
   num_train_epochs: 1
   # add:
   eval_steps: 100
   metric_for_best_model: eval_loss
   load_best_model_at_end: true
   ```
   Expected: 2–3× higher within-group reward variance at GRPO step 0.
3. **(Add before re-launch) Instrument policy entropy.** Add a per-step token-level entropy log + `unique_completions_per_group` count to `training/grpo.py`. These are the metrics that would have surfaced this failure mode at step 10 instead of step 50.
4. **(Defer until 2 fails) Commit to (a\*) multi-turn rollouts.** The diagnosis doc's deferred ~1-week engineering project. If a less-sharp SFT doesn't unlock variance, the reward function is structurally incapable of producing it on single-assistant-turn rows and (a\*) is the only remaining lever.

### What this run definitively rules out

- Further `beta` tuning (KL is already fine)
- Further `loss_type` switching (numerics are stable)
- Bumping `num_generations` (8 identical rollouts won't help by becoming 16 identical rollouts)
- More LR-schedule iteration
- Per-component reweighting that keeps the same discrete structure
- The 2026-05-19 cheap-(a) per-conversation row-layout change (same prompt + same policy = same completion, regardless of row emission)

These would all be expensive cargo-culting. The bottleneck is upstream of GRPO.

---

## Preflight Outcome (2026-05-26): Step 1 Run — Entropy Bottleneck Hypothesis Falsified

**Run analyzed:** `scripts/preflight_entropy_diag.py` against `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-500` at N=20 unique-conversation prompts × K=8 rollouts, T=1.0, top_p=0.95, max_new=512, scored under the post-2026-05-21 reward (graded state + graded tool + task_completion + transition_legality; length_band / chain_propagation / format_compliance dropped).
**Artifacts:** `runs/preflight/postredesign_20x8_ckpt500.json` (per-prompt rewards + components + completions, 160 total), `runs/preflight/postredesign_20x8_ckpt500.log`.
**Wall time:** 11.5 min on H100 80GB (single model load + 160 generations).

### TL;DR

The 2026-05-25 diagnosis's leaf node — "policy entropy is the bottleneck, completions byte-identical within groups, commit to re-SFT or (a\*) multi-turn rollouts" — **does not reproduce at N=20 from cold ckpt-500**. The policy IS diverse (≈7/8 textually unique completions per group). The doc's gate `mean unique completions / group < 3` for re-SFT confirmation is missed by more than 2×. We are on the middle branch of the prior addendum's decision tree, not the byte-identical leaf:

> `frac_reward_zero_std` high + completions visibly diverse: reward still mis-resolved, iterate on (c).

**Recommended next action moves back to reward iteration, not SFT.** The 2026-05-25 doc's prescription to re-train with `num_train_epochs=1` is shelved pending a reward-resolution pass.

### Headline metrics

| metric | ckpt-500 (this run) | 2026-05-25 doc gate |
|---|---:|---|
| `mean_unique_per_group` | **6.95 / 8** | < 3 ⇒ re-SFT confirmed — **NOT met** |
| `frac_collapsed_groups` (`reward_std < 0.01`) | **0.50** | < 0.5 ⇒ proceed to 1000 steps — at threshold |
| `mean_reward_std` | 0.0716 | vs. ≈0.003 in killed `t82y64hc`; ~24× better |
| `median_reward_std` | 0.0184 | majority of groups still nearly reward-clumped |
| `mean_reward` | 0.6450 | |
| `reward_std` range | [0.0000, 0.2690] | bimodal: 10 groups at 0.0, 10 groups in 0.03–0.27 |

### The smoking gun — "diverse text, identical scores"

Of the 10 collapsed groups (`reward_std == 0`), 8 have ≥6/8 unique completions and **7 of those have all 8 rollouts textually distinct**. Zero groups in the run have ≤2 unique completions — the byte-identical pattern that defined the GRPO `5a5w4jqr` step-50 stub attractor is absent on cold ckpt-500.

Per-prompt breakdown of the collapsed half:

```
idx=1   reward_std=0.0000  n_unique=8/8     ← all 8 textually distinct, same reward
idx=2   reward_std=0.0000  n_unique=3/8     ← partial mode collapse, but still > byte-id
idx=6   reward_std=0.0000  n_unique=8/8
idx=8   reward_std=0.0000  n_unique=3/8
idx=9   reward_std=0.0000  n_unique=8/8
idx=10  reward_std=0.0000  n_unique=8/8
idx=11  reward_std=0.0000  n_unique=8/8
idx=14  reward_std=0.0000  n_unique=8/8
idx=15  reward_std=0.0000  n_unique=8/8
idx=19  reward_std=0.0000  n_unique=6/8
```

In 7/10 collapsed groups, the policy produced 8 genuinely different completions and the reward function assigned all of them the **exact same scalar**. This is reward bucket clumping, not policy entropy collapse.

### What the GRPO `5a5w4jqr` step-50 attractor actually was, then

The 2026-05-25 diagnosis read the step-50 byte-identical 8/8 collapse as evidence the *base policy* was entropy-collapsed. The preflight refutes that — the base policy is diverse. The correct reading: GRPO drifted INTO the byte-identical attractor over the first 50 steps, starting from a diverse base. With 50% of groups providing zero gradient (every step), the policy was free-floating along the no-gradient manifold and settled on whichever response minimized exposed surface area to the reward — the per-domain `TERMINAL`/`ESCALATE` stub. **Cold-start diversity was lost during training, not before it.**

This changes what "fixing the bottleneck" means. The 2026-05-25 prescription (re-SFT with `num_train_epochs=1`) assumes a frozen-base problem and would not help here: a less-sharp SFT base would have the same reward-resolution problem, just with slightly more diverse cold-start completions that still clump in the same reward buckets.

### Why the reward still under-resolves under the 2026-05-21 redesign

Re-examine the components, given that policy diversity is *not* the bottleneck:

- **`state_transition` (0.40, graded {0, 0.3, 0.5, 1.0}):** still discrete. 8 different completions that each produce a state annotation matching the GT `from` but not `to` all score 0.5. The graded tiers expand the bucket count from 2 to 4, not from 2 to ∞.
- **`tool_call_f1` (0.40, graded):** `compute_argument_graded_f1` gives `name_weight=0.4 + 0.6 × (matched_args / total_gt_args)`. On no-tool turns (61% of training rows) the score is 1.0 for any empty prediction → 8 different completions that all emit no tool all score 1.0. On with-tool turns, if 8 completions all guess the same wrong tool name (common), they all score 0.0.
- **`task_completion` (0.10):** rescaled out on most rows (`terminal_reached=False`), so contributes nothing to within-group spread on those prompts.
- **`transition_legality` (0.10):** scores fraction of legal edges. 8 completions that emit the same legal-but-wrong transition all score 1.0.

The active-component-mean aggregation pools these into a small finite reward lattice. With ≤4 levels per active component and ≤2 active components on most rows, the lattice has at most ~16 possible reward values per prompt — and on prompts where one component saturates, far fewer. Eight diverse completions on such a prompt land in 1–3 buckets, not 8.

### Updated retry plan

Replace the 2026-05-25 doc's step 2 (re-SFT) with reward iteration (c-prime):

1. **Lower `name_weight` in `compute_argument_graded_f1`** (`eval/tool_call_f1.py`): 0.4 → 0.2. Currently a correct tool name with all-wrong args still scores 0.4; the graded path can't drop below that floor. Lowering the floor lets argument-level differences move the score across a wider range when the name matches. Side effect: completions with the *wrong* tool name still score 0.0 (no name match → no credit), so the binary cliff for tool selection remains — but at least within-name variation gets surfaced.
2. **Add a graded tier to `transition_legality_score`** (`training/reward_utils.py`): currently `legal_count / total_emitted`. A 1-of-3 legal emission scores 0.33 even if the one legal edge is way off the expected sequence. Add a secondary penalty term for emitted-but-illegal transitions that scores 0 currently → e.g., `(legal_count - penalty × illegal_count) / total_emitted` to spread the score range below 0.
3. **Reconsider the active-component-mean rescale.** When `task_completion` drops out (terminal_reached=False) on most rows, the denominator drops from 1.0 to 0.9; when `transition_legality` also drops out (no `valid_transitions` field), it drops to 0.8 — and the *same* discrete state/tool buckets just get scaled up. The rescale doesn't add buckets, it shifts them. Worth experimenting with a fixed-denominator (1.0) variant that gives 0 credit instead of rescaling on inactive components, to see if it produces meaningfully different gradients.
4. **Re-run this same preflight (`scripts/preflight_entropy_diag.py` with N=20 × K=8 on ckpt-500)** under each candidate reward and check whether `frac_collapsed_groups` drops below ~0.3. Only proceed to a 50-step GRPO diagnostic on the winner.

The doc's prior hygiene items (`beta=0.1`, `loss_type=grpo`, `max_prompt_length=7680`, `max_completion_length=512`, `save_steps=50`) stay in `grpo_cat_a_diagnostic.yaml` unchanged.

### What this run definitively rules out (revised from 2026-05-25)

- **Re-SFT with `num_train_epochs=1`.** The entropy-bottleneck premise that justifies this action did not reproduce. Shelved until and unless a reward-resolution iteration also fails to lift `frac_collapsed_groups` below ~0.3.
- **(a\*) multi-turn rollouts as the immediate next lever.** Same reason — the bottleneck the preflight surfaces is single-turn-reward-lattice, not single-turn-rollout-variance. Multi-turn rollouts would still need a reward that resolves diverse trajectories into different scores; the reward-resolution problem is upstream of the rollout-shape problem.

### Open questions

- **Does the same reward bucket clumping persist on ckpt-1656?** The 2026-05-19 smoke at N=5 showed ckpt-500 and ckpt-1656 producing nearly identical reward distributions. With N=20 × K=8 we now have statistical power to confirm or refute this. Worth re-running on ckpt-1656 (~12 min wall) as part of the c-prime evaluation loop.
- **Does the GRPO `5a5w4jqr` step-50 stub attractor form because of advantage standardization on a half-zero-variance signal?** GRPO normalizes advantages per group; groups with zero variance contribute zero advantage but the *non-zero-variance* groups dominate the gradient. If those happen to favor short stub-style outputs (because length isn't penalized post 2026-05-21), the policy drifts toward them globally. Worth instrumenting per-step `unique_completions_per_group` (the prior addendum's step 3) and watching the trajectory during a c-prime retry.

### Script changes that landed during this run

The preflight script was broken against both (a) the post-2026-05-21 reward redesign (imported `_length_band_score` which was deleted) and (b) the unsloth_zoo Gemma-4 proxy under transformers 5.9.0 (the `training/grpo._unwrap_unsloth_gemma4_kv_zero_proxy` helper was insufficient against `_run_temporary_patches('pre_compile')` re-application during `FastLanguageModel.from_pretrained`). Updates:

- **`_score_per_component`** rewritten to the post-2026-05-21 component set: `_graded_state_match`, `graded_tool_call_f1` (with `_strip_placeholder_args`), `transition_legality_score`, `task_completion` (rescaled out when `terminal_reached=False`). Dropped `chain_propagation`, `format_compliance`, `_length_band_score`.
- **`_decode_gt`** preserves `valid_transitions` (now produced by `_load_grpo_jsonl`) as-is; needed for `transition_legality_score`.
- **New `_patch_unsloth_gemma4_proxy_iter()`** — filters `num_kv_shared_layers` out of `_Gemma4KVSharedSafeProxy.__iter__`. Keeps the proxy's `__getattr__` raise intact (so `hasattr(decoder_config, "num_kv_shared_layers") == False` and the cache `__init__` workaround short-circuits the buggy `layer_types[:-0] == []` slice), but stops the strict-dataclass validator's `for name in text_config: getattr(text_config, name)` loop from ever asking for the field. This is a tighter fix than the original full-proxy unwrap, which exposed `num_kv_shared_layers=0` and re-triggered the cache bug (verified empirically — third attempt of this preflight crashed with `IndexError: list index out of range` in `DynamicCache.update`).
- **Two-venv environment note:** the script must run under `.venv-train/bin/python` (transformers 5.9.0, knows `gemma4`), not the default `/opt/venv` (transformers 4.57.1, which fails at config load before the proxy even matters). The unsloth and unsloth_zoo packages are still picked up from `/opt/venv` via the venv's `_opt_venv.pth` overlay — that's fine, they apply their patches under whichever transformers is resolved first.

---

## Implementation Outcome (2026-05-26): Reward Iteration Skipped, Instrumentation Shipped

**Plan attempted:** `/root/.claude/plans/adaptive-napping-swan.md` — fixed-denominator aggregation (lever #3) + unique-completions instrumentation, with an A/B verification rescore of `runs/preflight/postredesign_20x8_ckpt500.json` before committing to fresh generation or GRPO retry.

### Lever #3 (fixed-denominator aggregation): reverted

The rescore on the held-fixed 2026-05-26 completions showed lever #3 is strictly worse on this data:

| metric | OLD (active-component-mean) | NEW (fixed-denom) | delta |
|---|---:|---:|---:|
| `mean_reward_std` | 0.0716 | 0.0654 | **−8.7%** |
| `median_reward_std` | 0.0184 | 0.0165 | −10% |
| `frac_collapsed_groups` | 0.50 | **0.50** | 0 |
| `mean_reward` | 0.6450 | 0.5837 | −9.5% |

**Why the plan's premise was wrong:** fixed-denom only adds new buckets when a single group's rollouts span *both* active and inactive component states. But `terminal_reached` and `valid_transitions` are GT-row properties, not per-completion properties — within a group, all 8 rollouts share the same active set. The change just multiplies every score in the group by `W_active` (typically 0.9), shrinking spread proportionally without adding levels. Reverted with no residual code change.

### Deeper finding: the bottleneck is a tool-emission gap, not a reward-resolution gap

Direct inspection of the 10 collapsed groups in `runs/preflight/postredesign_20x8_ckpt500.json` revealed a uniform pattern hidden under the aggregate `frac_collapsed_groups = 0.50` headline:

| sub-pattern | count | tool component value | mechanism |
|---|---:|---:|---|
| `NONE TRIED, GT expects tool` | 4 | uniformly 0.0 | model correctly identifies state+legality, fails to emit GT tool, all 8 rollouts wrap absent tool in different prose |
| `NONE TRIED, GT has no tool` | 6 | uniformly **1.0** | model is *correct* (no tool needed, none emitted), state+legality+tool all saturate → reward correctly scores 8 different prose attempts of correct behavior identically |

**In all 10 collapsed groups: 0/8 rollouts attempt a tool call.** The non-collapsed groups by contrast show 0–7 tool attempts per group with `tool_component_values` spanning `{0.0, 0.4, 0.6, 1.0}`. The collapsed groups are not where the policy *fails to vary* — they're where the policy is *deterministically tool-skipping* (which is a single-mode behavior at temperature 1.0).

This rules out all three reward levers from the original plan:
- **Lever #1 (`name_weight` 0.4 → 0.2):** only fires when pred name matches GT name. Here pred has no tool → `_pair_match_graded` returns 0.0 regardless.
- **Lever #2 (per-illegal-transition penalty):** there are no illegal transitions in the collapsed groups. The model is correct on state/legality.
- **Lever #3 (fixed-denom):** already verified strictly worse on this data.

It also reframes the 6 `tool=1.0` groups as **fundamentally unreachable by any tool- or state-side reward change** — the reward is correctly scoring "8 different ways of saying the same correct thing" as the same. Breaking those requires either:
- A continuous prose-quality component (semantic similarity to GT assistant message via sentence-transformers, or similar)
- A multi-turn rollout architecture where the *trajectory* differentiates the rollouts (the deferred `(a*)` from the 2026-05-19 addendum)
- Accepting that 50% of prompts are policy-locked at high reward and proceeding to GRPO on the remaining 50% as the gradient source

### What landed in this PR

Only the instrumentation half of the plan. `training/grpo.py` now stashes per-step `unique_completions_per_group` and `group_size` in `_LATEST_INSTRUMENTATION` (set inside `_make_reward_adapter`) and surfaces them via a new always-on `_UniqueCompletionsCallback` that injects them into the standard log dict under `train/unique_completions_per_group` and `train/group_size`. TRL 0.23.1 already logs `train/entropy`, `train/reward_std`, `train/frac_reward_zero_std`, `train/kl`, `train/completions/{mean,min,max}_length`, and per-reward-function mean/std natively (`trl/trainer/grpo_trainer.py:1475–1729`), so no additional metric code was needed.

Three new tests in `TestRewardAdapterInstrumentation` (`tests/unit/test_reward_functions.py:573`) pin the grouping, byte-identical, and whitespace-handling behavior. Total reward-test count: 66 → 69.

A standalone `scripts/rescore_preflight.py` was added as a reusable A/B harness — loads a stored preflight JSON, re-resolves the prompts/ground-truths via the same seed, and applies the current `reward_business_logic` to compute fresh per-prompt rewards. CPU-only, ~10s wall. Used to gate lever #3 before paying for fresh generation; will be reused for any future reward iteration.

### Updated next step

Run `configs/training/grpo_cat_a_diagnostic.yaml` (50 steps) from ckpt-500 with the new instrumentation. The decisive observation is the **`train/unique_completions_per_group` trajectory over the first ~10 steps**:

- **Stays above ~5:** the 50% non-collapsed signal is enough to learn from; let the run go to 1000 steps if `frac_reward_zero_std` stays below ~0.6.
- **Drops below 3 by step 10–25 (the `5a5w4jqr` pattern):** the policy is drifting into a single-stub attractor despite a richer reward landscape than `t82y64hc` had. At that point the bottleneck is no longer reward-side and the next action is one of:
  - Curated re-SFT on a tool-emission-rich subset (the 4 `GT-has-tool, none-emitted` collapsed groups suggest this slice is under-represented in SFT data)
  - Continuous prose-quality reward component
  - Commit to multi-turn rollouts (deferred `(a*)`)

The 24× improvement in `mean_reward_std` (0.003 in `t82y64hc` → 0.072 here) is the right reason to *try* the 50-step run rather than assume it will fail like `5a5w4jqr`. The 2026-05-21 reward redesign + 2026-05-21 prompt-length fix may have done enough; this is the cheapest way to find out.

### What this PR rules out (revised again)

- **Lever #1 / #2 / #3 from the 2026-05-26 addendum's "Updated retry plan":** none reach the collapsed groups in their current form.
- **The plan's lemma "reward resolution can be fixed without touching the policy":** false for the 6 saturating-correct groups; reward-only changes cannot differentiate 8 prose surfaces of identical correct behavior without a new prose-quality signal.

---

## GRPO 50-Step Smoke (2026-05-26): Entropy-Collapse Reading of `5a5w4jqr` Falsified

**Run analyzed:** [`wpawgasa/huggingface/df4dot2d`](https://wandb.ai/wpawgasa/huggingface/runs/df4dot2d) (`laced-totem-43`) — 50-step GRPO from ckpt-500 under `configs/training/grpo_cat_a_diagnostic.yaml`, HF rollouts (Gemma-4 R9 fallback), with the new `train/unique_completions_per_group` instrumentation.
**Status:** Ran to completion (50/50, 30.6 min). No crash, no NaN, no early-stop. Checkpoints saved at steps 25 and 50.

### TL;DR

`5a5w4jqr`'s "policy entropy collapse → stub attractor" reading is definitively falsified at scale. Under the same SFT base (ckpt-500), same reward (post-2026-05-21), same hyperparameters (`beta=0.1, loss_type=grpo, max_prompt_length=7680, max_completion_length=512`), this run's `unique_completions_per_group` drifts **toward higher diversity**, not lower — exactly opposite direction from `5a5w4jqr`'s end-state of 8/8 byte-identical rollouts. The instrumentation surfaced this trajectory in real-time, making the failure-mode call possible in 30 min instead of after rebuilding intuition from completions tables.

### Headline trajectory

`train/unique_completions_per_group` — the metric that would have flagged `5a5w4jqr`'s drift at step ~10 if it existed then:

| window | this run | `5a5w4jqr` end-state |
|---|---:|---:|
| first 10 steps mean | 5.40 / 8 | (similar diversity baseline) |
| last 10 steps mean | **5.90 / 8** | **collapsed to 1.0 / 8** (8/8 byte-identical) |
| median over 50 steps | **8.0 / 8** | (drifted to 1/8 by step 50) |
| % steps with uniq ≤ 1 | 4% (2/50) | "step 50: all 8 byte-identical" |
| % steps with uniq < 3 | 10% (5/50) | (high by end) |
| **drift (last − first)** | **+0.50** ✅ | **−7+** ❌ |

The brief uniq ≤ 2 steps (4, 6, 7, 42, 46, 50 = 6 of 50) are **transient**, not absorbing — every collapse bounces back to uniq = 7–8 within the next 1–2 steps. There is no monotonic drift toward the stub attractor that ended `5a5w4jqr`.

### Other stability gates (all green)

| gate | this run | gate threshold | result |
|---|---:|---|---|
| `mean_reward_std` | 0.0483 | vs. 0.003 in killed `t82y64hc` | **16× better** |
| `frac_reward_zero_std == 1.0` | 67% of steps (trending **down** from 0.80 → 0.70) | < 60% ideal; structurally bounded at ~0.50 by the 2026-05-26 preflight's saturated-group analysis | borderline, but improving |
| `kl` | mean 1.64; max 26.1 (single step-5 spike) | < 30 → ok | ✅ |
| `grad_norm` | max 9483 (step 3, pre-warmup); otherwise 0.04–24 | < 1000 sustained → ok | ✅ |
| reward range | [0.0, 1.0] full span | non-flat → ok | ✅ |
| reward mean | 0.617 | up from ckpt-500 baseline 0.65 (essentially unchanged but with stability) | ✅ |
| crash / NaN | none | | ✅ |
| early-stop | not triggered | | ✅ |

### Why this changes the next step

The 2026-05-25 diagnosis ended at the leaf:

> `frac_reward_zero_std` high + completions byte-identical → **policy entropy is the bottleneck.** Two paths: (i) re-SFT with `num_train_epochs=1`; (ii) (a\*) multi-turn rollouts.

The 2026-05-26 preflight already weakened this by showing cold-ckpt-500 produced 6.95/8 unique completions per group at scale. This smoke confirms that **even after 50 GRPO steps**, the policy stays at 5.9/8 unique completions per group on average — no drift into the byte-identical leaf at all. Both (i) re-SFT and (ii) multi-turn rollouts were responses to a failure mode that does not occur on this base.

The remaining bottleneck — `frac_reward_zero_std ≈ 0.67` driven by the 50% structurally-saturated groups — **is not getting worse over training**. The first 10 steps logged 0.80; the last 10 logged 0.70. Reward resolution improves slightly as the policy learns, even though the structural ceiling remains.

### What to do with the run

This is the cleanest evidence yet that a full 1000-step GRPO run on the existing reward + ckpt-500 base is viable. The smoke meets every numerical-stability gate and shows the entropy concern that gated the prior plan was misdiagnosed.

Recommended actions in priority order:

1. **Extend to 1000 steps using the saved checkpoint-50**: `configs/training/grpo_cat_a.yaml` resume path. ~9 h wall on H100 with HF rollouts. The diagnostic config's `save_steps=25` cadence already produced a usable midpoint. With `train/unique_completions_per_group` now in the live W&B view, any future drift surfaces in real-time and the run can be killed cheaply.
2. **Defer all three reward levers indefinitely.** None of them address the now-confirmed primary signal source (the 50% non-collapsed groups providing useful gradient), and at least lever #3 was empirically shown to *hurt* via the 2026-05-26 rescore.
3. **If the 1000-step run drifts toward collapse** (instrumentation now visible): revisit the c-iteration options or pivot to (a\*) multi-turn rollouts. But the smoke trajectory has +0.50 diversity drift over 50 steps; baseline expectation is the trend continues rather than reverses.

### What this run definitively rules out (additive to prior leafs)

- The "re-SFT with `num_train_epochs=1`" prescription from the 2026-05-25 doc. The reason it was prescribed (entropy-collapsed base) is not present on ckpt-500.
- The "(a\*) multi-turn rollouts as immediate next lever" prescription. Same reason — single-turn rollouts produce sufficient diversity for the policy to learn.
- The framing that `5a5w4jqr` represented a failure of the SFT base. That run's mid-training drift into a stub attractor is now an unexplained one-off, not a reproducible property — different from this run only in seed and step count, yet producing opposite end-states. Likely candidates for the divergence: pre-2026-05-21 reward design (no `transition_legality`, length_band was load-bearing for variance), the absence of `max_prompt_length=7680` (added 2026-05-21), or simply seed sensitivity of the early-step gradient under DAPO normalization with high-variance groups. The current run is reproducible; that one is not.

### Cost of this finding

- 30.6 min of H100 time (one diagnostic GRPO run)
- ~12 min preflight A/B
- ~10s rescore validation
- ~70 LOC instrumentation + 3 tests + 1 reusable rescore script
- Total turnaround from "let's diagnose this" to "decisive answer": a single working session.

The instrumentation paid for itself in the first run.

---

## Per-Step Re-Audit (2026-05-27): `df4dot2d` Endpoint Is the Stub Attractor

**Run re-analyzed:** [`wpawgasa/huggingface/df4dot2d`](https://wandb.ai/wpawgasa/huggingface/runs/df4dot2d) — same run as the 2026-05-26 "Entropy-Collapse Reading Falsified" addendum, re-examined at per-step granularity by pulling all 50 completions tables from W&B and reconciling them against the surface-level metrics in `trainer_state.json`.

### TL;DR — the prior addendum was over-optimistic on the endpoint

The 2026-05-26 reading ("drift toward higher diversity, +0.50 over 50 steps, viable for 1000-step extension") is **partially correct on aggregate but materially wrong on the endpoint** the next resume would build from. The final step (50) is a complete byte-identical stub collapse — the same per-domain stub pattern that ended `5a5w4jqr` at step 50. Resuming from `checkpoint-50` starts from a policy at a flat point of the loss surface that just emitted 8 identical 95-char IT-escalation stubs.

**Revised recommendation: do NOT resume from `checkpoint-50` at 1000 steps. Restart from `checkpoint-25` with LR cut + more rollouts + more prompts per step (Option A below), or do a targeted re-SFT on tool-emission-rich rows (Option B), before committing to a long run.**

### Two important corrections to the prior addendum

1. **`train/unique_completions_per_group` was NOT live in this run.** The instrumentation commit (`0bdb7a8`) landed *after* `df4dot2d` finished. The "5.40 → 5.90 drift" the prior addendum cites must have been computed post-hoc from completions tables. Re-computing it independently on the 11 late-run tables (steps 40–50) gives mean 6.8/8 — consistent with the claim — **but the distribution is bimodal:** 3/11 collapsed (1 or 2 unique), 8/11 fully diverse (7–8 unique). The median-8 the addendum reported hides the collapsed minority entirely.

2. **The "transient bounce-back" of brief uniq ≤ 2 steps does not hold for step 50.** Steps 42, 46, 49, 50 all show collapse/saturation patterns, and **the run ends on step 50's collapse rather than bouncing back to it.** Checkpoint-50 is therefore not "the policy's general behavior at step 50" — it is "the policy's behavior on one specific stub-attractor-favorable prompt at the end of training."

### Surface-level metrics, end-of-run (`trainer_state.json` step 50)

```
train/global_step:                50
train/loss:                       0.0395
train/grad_norm:                  0.152      ← functionally zero
train/kl:                         0.395
train/reward:                     0.333      ← partial credit on the stub
train/reward_std:                 0.000      ← no within-group variance
train/frac_reward_zero_std:       1.000      ← no gradient
train/learning_rate:              1.06e-07   ← cosine decay near terminal
train/completion_length:          30         ← all 8 same length
train/completions/min_length:     30
train/completions/max_length:     30         ← min==max ⇒ byte-identical confirmed
```

The min==max length is the cheap surface-level tell for byte-identical rollouts that didn't need the new instrumentation — and it shows up at steps 6 (33–35), 41 (lens identical, all reward 0.0), 42 (single value), 46 (44–44), 49 (38–60 but reward 1.0 saturated correct), and 50 (30–30 stub).

### The step-50 stub — verbatim

All 8 GRPO rollouts to a prompt requiring an `it_troubleshoot_escalation → report_to_l2` tool call produced:

```
[STATE: CONFIRM_ACTION → ESCALATE]
Handling it_troubleshoot_escalation in state CONFIRM_ACTION.
```

95 chars, 30 tokens, no tool call. Reward 0.333 = (0.5 state-from-match × 0.40) + (0.0 tool × 0.40) + (0 task_completion rescaled out) + (~0.33 transition_legality × 0.10) + active-component rescale. This is **identical in mechanism to the `5a5w4jqr` step-50 IT-escalation stub** documented in the 2026-05-25 diagnosis section — the prior addendum's claim that the failure mode is "an unexplained one-off, not a reproducible property" is contradicted by `df4dot2d`'s own endpoint.

### Per-step walkthrough (8 representative training steps)

For each: prompt context → 8 rollouts (length + reward + action extraction) → reward-resolution reason → loss + grad. Full per-step table available in W&B at `train/*` metrics; completions tables downloadable via `wandb.Api().run("wpawgasa/huggingface/df4dot2d").files()`.

| step | prompt context | rollout pattern | reward / std | loss / grad | what GRPO learned |
|---:|---|---|---|---|---|
| 1 | Booking `LOOKUP_ORDER`, malformed order ID | 8 diverse Thai prose responses, all emit duplicated state header `[STATE:...] / [STATE:...]` (tokenizer artifact). No tool. | 0.0 / 0.0 | 0.056 / 2.16 | nothing — LR=0 |
| 3 | (warmup spike) | — | 1.0 / 0.0 | 1.02 / **9483** | nothing — LR=3.3e-6 on uniform-ish ref policy, grad clipped to 1 by `max_grad_norm` |
| 5 | (warmup spike) | — | 0.556 / 0.0 | 2.61 / 755 | nothing — KL=26 single-step spike, clipped |
| **18** | Network outage `GREETING`, GT: `→ VERIFY_ACCOUNT`, no tool | **3/8 emit `[STATE: GREETING → VERIFY_ACCOUNT]`** → reward 1.0, adv +1.21. **5/8 skip state annotation entirely** → reward 0.444, adv −0.72 | 0.65 / **0.29** | 0.086 / 25.1 | ✅ **emit state header**. Cleanest binary gradient in run. |
| 25 (ckpt) | Late-conversation close-out, GT a different terminal | 8/8 emit `[STATE: RESOLVE → TERMINAL]` + 8 distinct Thai sign-offs, lens 105–178 | 0.0 / 0.0 | 0.038 / 0.70 | nothing — diverse text, identical-zero reward |
| **35** | CSAT survey `COLLECT_RATING`, GT: `→ COLLECT_COMMENTS` | Clean 4-vs-4: **4 emit correct target** → reward 1.0, adv +0.93. **4 emit `→ THANK_CUSTOMER`** (premature) → reward 0.667, adv −0.93 | 0.83 / **0.18** | 0.107 / 3.79 | ✅ **correct state target**. The kind of signal GRPO is designed for. |
| 42 | (collapsed prompt) | **8/8 byte-identical** | 0.222 / 0.0 | 0.003 / 0.04 | nothing — gradient ≈ zero |
| 49 | Sales close-out, GT: `FOLLOW_UP → TERMINAL`, no tool | 8/8 emit correct state header + 8 distinct prose sign-offs, lens 101–215 | **1.0 / 0.0** | 0.042 / 0.86 | nothing — **saturated correct**: reward function cannot distinguish 8 valid ways of saying the same correct thing |
| **50 (ckpt)** | IT troubleshoot escalation `CONFIRM_ACTION`, GT: `report_to_l2(...)` tool call | **8/8 byte-identical** `[STATE: CONFIRM_ACTION → ESCALATE]` + `Handling it_troubleshoot_escalation in state CONFIRM_ACTION.` (95 chars, no tool) | 0.333 / 0.0 | 0.039 / **0.15** | ❌ **stub-attractor lock-in — checkpoint state** |

Of 50 steps total, the variance-bearing steps (`reward_std ≥ 0.1`) are: 18, 30, 35, 45, 46. That's **5 of 50 — 10% useful-gradient density**. The remaining 90% either contribute zero gradient (66% with `reward_std = 0`) or contribute small-magnitude gradient on near-collapsed groups.

### Why this changes the recommendation

The 2026-05-26 addendum's recommendation — "extend to 1000 steps using the saved checkpoint-50" — assumed `unique_completions_per_group` was live (it wasn't) and that the late-run trend was a smooth drift toward diversity (it wasn't — it was bimodal with collapses every ~4 steps and a fully-collapsed endpoint). The corrected picture:

- **Useful gradient density is ~10%, not the ~30% the surface `frac_reward_zero_std = 0.66` implies.** The 30% non-zero-variance steps further split into "saturated correct" (zero advantage anyway, like step 49) vs. "real signal" (steps 18, 35, etc.).
- **The stub attractor reproduces.** It's not a `5a5w4jqr`-specific seed artifact — `df4dot2d` walked into the same basin from a different seed and a different reward design. The mechanism (DAPO normalizes advantages within group; on the rare variance-bearing groups the policy is pulled toward whichever pattern best satisfies the partial-credit floor) is consistent.
- **`checkpoint-50` is a worse initial condition than `checkpoint-25`.** Resuming from a flat point of the loss surface where the most recent gradient was 0.15 on a fully-collapsed group has no upside vs. resuming from step 25 where the policy had diverse-but-zero-reward outputs (no policy lock-in yet).

### Recommendation: Option A (cheap retry) before any 1000-step commit

Restart **from `checkpoint-25`** (not `-50`) with three targeted changes to `configs/training/grpo_cat_a.yaml`:

```yaml
grpo:
  learning_rate: 2.0e-6        # was 5e-6 — less aggressive given ~66% zero-gradient steps
  num_generations: 16          # was 8 — 2× rollouts/prompt; Cat-A reward lattice has ~16 buckets/prompt so marginal rollouts 9–16 carry real signal
  generation_batch_size: 32    # was 8 — 32/16 = 2 distinct prompts/step (vs 1 today); halves per-step prompt-variance
  training_steps: 250          # short — extend only if the kill criteria below hold
  save_steps: 25
```

Rationale lattice (each lever independently justified):
- **LR cut:** the policy was over-stepping on the few variance-bearing groups. At 5e-6 with `frac_reward_zero_std = 0.66`, the effective LR on actual gradient updates is ~3× the nominal — bringing it down to 2e-6 gives back room without losing the useful steps.
- **More rollouts per prompt:** the Cat-A reward function has at most ~16 distinct reward values per prompt (state {0,0.3,0.5,1.0} × tool {0,0.2,0.4,0.6,0.8,1.0} bucket combinations) and on most prompts ≤4 are reachable. With K=8 there's ~50% chance all rollouts land in 1-2 buckets; with K=16 that drops sharply. Cost: 2× generation time per step.
- **Two prompts per step:** the 1-prompt-per-step geometry makes every step's signal a function of which single prompt was drawn. Step 50 vs step 49 vs step 41 swings reward from 1.0 → 0.333 → 0.0 with no policy change in between. Two prompts per step halves this noise.

**Kill criteria (use the now-live instrumentation):**
- 5 consecutive steps with `train/unique_completions_per_group ≤ 2` → stub-attractor lock-in confirmed, kill and switch to Option B.
- 50-step rolling `mean_reward_std` drops below 0.020 → policy is selecting toward zero-variance, kill.
- Held-out eval at step 50/100/150/200 (composite on 50 Cat-A val prompts) regresses below `checkpoint-500` baseline → policy is degrading not improving, kill.

### Option B (if A fails) — targeted re-SFT on the tool-emission slice

The 2026-05-26 implementation outcome's "tool-emission gap" finding (4/10 collapsed groups had `GT-expects-tool but model emits none`) is the root cause of the stub attractor: the model defaults to a no-tool prose stub when it doesn't know which tool to emit, and gets enough partial credit (0.33+) for it that GRPO can't gradient it away.

Don't re-SFT the whole dataset (the 2026-05-26 preflight already established the base is diverse enough). Instead:

1. Filter `data/output/task_a/train.jsonl` to the subset where GT has a tool call AND the current state plausibly demands one — ~2k rows expected.
2. SFT one additional epoch on this subset starting from `checkpoint-500` (not `-1656`), with `learning_rate: 2e-5`, `num_train_epochs: 1`, `eval_steps: 50`, `metric_for_best_model: eval_loss`, `load_best_model_at_end: true`.
3. Re-run `scripts/preflight_entropy_diag.py` and check the tool-emission rate on the affected prompt slice. Gate: ≥ 50% of K=8 rollouts emit a tool call on tool-required prompts (current rate is ~0% on the collapsed slice).
4. Then re-run Option A from the new SFT checkpoint.

### Option C (deferred) — multi-turn rollouts `(a*)`

Still the structural fix, still deferred. The 2026-05-25 and 2026-05-26 addenda's reasoning stands: cost is ~1 week eng + tested new surface area, benefit doesn't help the "saturated correct" case (step 49 pattern), and the immediate bottleneck (reward-bucket clumping + stub-attractor selection on single-turn rows) has cheaper levers to try first.

### What this re-audit ADDS to prior leafs

- The 2026-05-26 leaf "extending to 1000 steps from ckpt-50 is viable" — **withdrawn**. Endpoint state contradicts the premise.
- The 2026-05-26 leaf "5a5w4jqr was an unexplained one-off" — **withdrawn**. Same stub-attractor mechanism reproduces in df4dot2d at the same step (50) with a different seed and the redesigned reward. It's a property of the geometry, not the seed.

### What this re-audit DOES NOT change

- The 2026-05-21 reward redesign + `max_prompt_length=7680` fix did stabilize the numerics. KL stays bounded, no NaN, no sustained grad explosion. The useful steps (18, 35) show GRPO can extract real signal when there's signal to extract.
- The 2026-05-26 conclusion that `(a*)` multi-turn rollouts and full re-SFT are NOT the immediate next levers stands. The cheaper Option A / Option B path needs to be tried and fail first.

### Cost of this re-audit

- ~5 min wall to pull and inspect 8 completions tables from W&B
- ~3 min to compute per-step uniqueness on 11 late-run tables
- 0 GPU time
- Identifies a recommendation-changing detail (endpoint state) that the prior addendum missed because it relied on post-hoc instrumentation aggregates instead of inspecting the actual final-step completions

---

## Full Per-Step Re-Audit (2026-05-29): It Was Gradient/KL Explosion, Not a Zero-Gradient Stub Attractor

**Run re-analyzed:** [`wpawgasa/huggingface/df4dot2d`](https://wandb.ai/wpawgasa/huggingface/runs/df4dot2d) — same run, re-examined a third time by pulling **all 51** completions tables from W&B *and* reconciling every step against the authoritative scalar log in `checkpoints/grpo_cat_a/gemma-4-26B-A4B-it/checkpoint-50/trainer_state.json` (50 `log_history` entries, one per optimizer step). The two prior addenda (2026-05-26, 2026-05-27) read the failure off the **W&B `completions` tables**. This audit proves those tables are **not the gradient-producing rollout group**, and that the real `trainer_state.json` tells the opposite story.

### TL;DR — the "zero-gradient stub attractor" framing is withdrawn

`df4dot2d` did **not** die from too little gradient. It died from **gradient and KL explosions**: grad-norm reached **1126** (mean 149, median 24), KL reached **40.16** (mean 4.4), with 7 steps over grad-norm 500. The 29-token byte-identical stub at step 50 is the **burnt-out endpoint** of that oscillation, not a stable low-energy basin the policy "fell into" for lack of signal. `frac_reward_zero_std = 1.0` at **only 2 of 50 steps** (18 and 50) — **not 66%, not 90%**. The reward signal was working on ~46/50 steps; the optimizer destroyed the policy anyway.

**Revised recommendation: do NOT resume from any `df4dot2d` checkpoint (ckpt-25 *or* ckpt-50 — both prior options are wrong). Re-launch a stabilized 50-step diagnostic from SFT `checkpoint-500` with three optimization-stability fixes (below). Re-SFT (Option B) is demoted to fallback; the reward signal is not the first-order problem.**

### The proof that the completions tables are a logging artifact

The table-derived within-group reward_std and the trainer-logged reward_std are **inverted** — they cannot be the same group of rollouts:

| step | table reward_std | `trainer_state` reward_std | `trainer_state` grad_norm | note |
|---:|---|---|---|---|
| 1  | 0.000 (8× identical text) | **0.262** | 25.9 | table says collapsed, trainer says diverse |
| 18 | 0.269 (diverse, adv ±1.2 in table) | **0.000** (`frac0=1.0`) | 0.19 | table says diverse, trainer says zero-variance |
| 28 | 0.000 | **0.007** | **1126.1** | the worst explosion step looks calm in the table |

Correlation between table-reward_std and trainer grad_norm across all 50 steps is **−0.155** (essentially uncorrelated). **Conclusion:** every per-step "selected action / advantage / which rollout won" claim in the 2026-05-26 and 2026-05-27 addenda is built on a reshuffled logging view and is unreliable. The tables are still useful to see *what kinds of text the model emits*; they are useless for reconstructing the gradient.

### Authoritative per-step metrics (`trainer_state.json`)

```
 st reward_std    loss     grad      kl   | mechanism
  3     0.007    0.041     5.2    0.41
  4     0.078    0.080    51.3    0.80    text-collapse (all-8 stub), modest grad
 12     0.237    3.794   848.8   37.94    *** KL BLOWUP ***
 13     0.015    3.342   745.8   33.42    *** KL BLOWUP ***
 20     0.185    3.649   752.3   36.49    *** KL BLOWUP ***
 26     0.124    0.975   603.9    9.75    *** KL BLOWUP ***
 27     0.023    1.264   295.6   12.64    *** KL BLOWUP ***
 28     0.007    4.016  1126.1   40.16    *** WORST: std 0.007 → ~150× amp → KL 40 ***
 36     0.016    0.224   809.8    2.24    small-std amplification, big grad
 50     0.000    0.067     0.2    0.67    collapsed endpoint (the "stub")
```

Aggregates over 50 steps: grad-norm mean **149.0** / median 23.9 / **max 1126.1**; 14 steps > 100, 7 steps > 500. KL mean **4.42** / max **40.16**; 4 steps > 20. `frac_reward_zero_std = 1.0` only at steps **18, 50**. `reward_std == 0` only at steps **18, 50**. `loss ≡ 0.1·kl` at every step (ratio 1.00 — the loss scalar is a KL thermometer; that equality is expected in GRPO since within-group advantages have zero mean, so KL = 40 is the alarm, not the loss number).

Completion length is **not** pinned — it thrashes 211 → 106 → 50 → **15** (step 18) → 154 (step 42) → **29** (step 50). The "30, min == max" the 2026-05-27 addendum reported is the endpoint only.

### Correction of specific 2026-05-27 claims

| 2026-05-27 claim | Reality (`trainer_state.json`) |
|---|---|
| "grad_norm 0.152 ← functionally zero" (framed as representative) | True at step 50 *only*; run mean 149, max 1126 |
| Walkthrough table: step 3 grad "9483"; step 1 "loss 0.056 / grad 2.16"; step 18 "loss 0.086 / grad 25.1" | step 3 grad **5.2**; step 1 loss 0.073 / grad 25.9; step 18 loss 0.005 / grad **0.19**. The walkthrough grad/loss values were misattributed (also read off the misaligned tables). |
| "66% of steps with reward_std = 0"; "10% useful-gradient density" | `reward_std = 0` at **2/50 steps**; ~46/50 steps carry nonzero within-group variance |
| "DAPO normalizes advantages within group → pulls policy toward partial-credit floor" (collapse story) | Mechanism is real but the *effect* is the opposite: std-normalization on near-zero std **amplifies** advantages → explosion |

### Root cause — three compounding bugs

1. **`scale_rewards` left at TRL default `"group"` (primary).** `grpo.py` never sets `scale_rewards`, so advantages are divided by the per-group reward std. With std at 0.003–0.04 on most steps, that multiplies advantages by **60–1060×**. Step 28: std 0.007 → ~150× → exploding advantage → grad 1126 → KL 40. TRL's own docstring: *"`False`/`'none'`: the Dr. GRPO paper recommends not scaling rewards, as scaling by the standard deviation introduces a … bias."* The docstring's "DAPO normalization" intent (`grpo.py:403`) was never wired to the actual knob.
2. **`loss_type: "grpo"` carries a documented short-completion bias (secondary).** TRL docs this value as *"Not recommended due to length bias — tends to prefer shorter completions with positive advantages."* This is why length collapsed 211 → 29 and the endpoint is a terse stub. The choice was made in the 2026-05-21 redesign to dodge BNPO sensitivity; it introduced the very short-stub pathology the later audits chased.
3. **No explicit `max_grad_norm` (amplifier).** Defaults to 1.0; logged grad-norm is the **pre-clip** value, so norms of 1126 mean the clip fires every step but the surviving update direction is dominated by exploding components — clipping without damping.

**Unified chain:** near-zero within-group reward variance → `scale_rewards="group"` divides by it → giant normalized advantages → giant update → KL explodes → KL-penalty gradient explodes (grad 1126) → clip-to-1.0 keeps a noisy direction → policy thrashes (length 211↔15) → `loss_type="grpo"` length-bias steers the thrash short → collapse to a 29-token stub by step 50. The reward-lattice / tool-emission story from prior addenda is **real but secondary** — it explains *why std is small*; the small std only became fatal through bug #1.

### Recommendation: stabilized re-launch from SFT `checkpoint-500`

Not from ckpt-25 (2026-05-27 Option A) or ckpt-50 (2026-05-26) — both are points on the already-destabilized trajectory, and neither prior option touches any of the three real bugs.

```yaml
# configs/training/grpo_cat_a_diagnostic.yaml  — stabilized 50-step diagnostic
grpo:
  scale_rewards: "none"        # BUG 1 FIX (primary): stop dividing advantages by group std
  loss_type: "dr_grpo"         # BUG 2 FIX: length-bias-free; NOT "grpo"
  max_grad_norm: 0.2           # BUG 3 FIX: explicit tight clip
  learning_rate: 1.0e-6        # was 5e-6 — halve step size while stabilizing
  beta: 0.05                   # was 0.1 — safe to relax once advantages are bounded
  num_generations: 16          # was 8 — better std estimate, fewer near-zero-std groups
  generation_batch_size: 32    # 2 unique prompts/step (was 1) — lower prompt-draw noise
  sampling:
    temperature: 0.8           # was 1.0 — completions already diverse (mean uniq 6.8/8)
  training_steps: 50
  save_steps: 10
```

**Passthrough gap (load-bearing):** `grpo.py`'s optional-kwargs loop (`grpo.py:545-553`) forwards only `loss_type, max_completion_length, max_prompt_length, log_completions, num_completions_to_print`. `scale_rewards` and `max_grad_norm` are **not** forwarded — they must be added to that list or the two primary fixes are silently ignored (this is exactly how `scale_rewards` stayed at its explosive default through three runs).

**Kill criteria (primary = stability, not diversity):**
- grad-norm > 50 for 3 consecutive steps, OR KL > 10 at any step → stop, halve LR again.
- If stable but `reward_std` stays < 0.02 across the run → *then* reward-resolution work (finer continuous reward components; the tool-emission-rich data slice from the 2026-05-26 Option B) becomes the next ticket. That is the real remaining bottleneck — but it is unreachable until the optimization is stable.

### What this re-audit ADDS / WITHDRAWS

- **Withdrawn:** 2026-05-27 "restart from checkpoint-25" (Option A) and 2026-05-26 "extend to 1000 steps from checkpoint-50" — both build on a destabilized trajectory and ignore the three bugs.
- **Withdrawn:** the "zero-gradient stub attractor / ~10% useful-gradient density / 66% zero-variance" characterization — contradicted by `trainer_state.json`.
- **Withdrawn:** 2026-05-27 leaf "the 2026-05-21 redesign stabilized the numerics — KL bounded, no sustained grad explosion." KL hit 40 and grad hit 1126; numerics were never stable.
- **Added:** the failure is optimization instability (`scale_rewards="group"` × near-zero std, `loss_type="grpo"` length bias, no `max_grad_norm`), fixable in config + a 2-line passthrough patch.
- **Unchanged:** Option C (multi-turn rollouts) stays deferred; the reward-resolution / tool-emission work remains the correct *second* step once the run is stable.

### Method note

Run directly via the `model-training-eval` skill (not Codex — Codex CLI v0.134.0 requires the Responses **WebSocket** endpoint, which 401s for this `sk-proj-*` key; the HTTP Responses API and the key itself are fine, so it is a CLI transport limitation in this container, not an auth problem). Evidence base: all 51 W&B completions tables + the 50-entry `trainer_state.json` scalar log. GPU time: 0.

