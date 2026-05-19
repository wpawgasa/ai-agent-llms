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

