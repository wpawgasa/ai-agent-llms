# GRPO Viability Investigation — Gemma-4-26B-A4B Cat A

**Date:** 2026-07-07
**Question:** After four killed GRPO runs and a re-generated + "qualified" dataset, can single-turn online GRPO with the current graded reward actually *learn* on this task — or is a different paradigm required?
**Authoring:** first-principles decision memo produced by **Fable 5** (highest-tier strategic reasoning), then **vetted by Opus** against the repo and this session's three read-only investigations.

---

## Vetting note (what was checked before trusting this memo)

Verified against the repo — all consistent:
- Reward lattice: `training/rewards/reward_business_logic.py` (`_graded_state_match` → {0, 0.3, 0.5, 1.0}; weights 0.40/0.40/0.10/0.10; active-component-mean), `eval/tool_call_f1.py` (binary on no-tool rows).
- RFT feasibility on existing infra: `training/sft.py:589-590` loads `{train,val,test}.jsonl` from `data.source` (so winners → a chatml `train.jsonl` reuse the existing trainer, no new code); `scripts/preflight_entropy_diag.py::_score_and_summarize` already scores every (prompt, completion) with `reward_business_logic` and emits per-prompt `rewards[]` / `reward_std` / `frac_collapsed_groups` (so greedy + headroom is a small extension); `training/merge_adapter.py::merge_and_export` exists; **no** DPO/RFT code exists yet (net-new correctly flagged).
- Stabilized config + wired held-out guardrail: `configs/training/grpo_cat_a_diagnostic.yaml`, `training/grpo.py::_heldout_composite_score`.

Adopted as ground truth (from this session's investigations, not re-derived here): dataset composition stats (61.1% zero-tool, 14.2% terminal, `state_sequence` never ≥2, median 8 reachable reward rungs/prompt). The 77% `frac_reward_zero_std` figure is from `bqbxnqxw`'s production geometry (8 gens, 1 prompt/step) and is **directional** for the 16-gen diagnostic geometry — which is exactly why Stage 0 below re-measures it at matched settings.

---

## 1. Verdict

**No — single-turn online GRPO with the current reward cannot learn a useful policy on this data as configured, and the merged stabilization does not change that.** Conditional revival only if a directly measurable bar is met:

> Online GRPO is admissible only if a preflight on the actual GRPO training distribution (T=0.8, 16 samples/prompt, SFT ckpt-1000) measures **`frac_collapsed_groups` < 0.50 AND median within-group `reward_std` ≥ 0.05**. Every measurement to date fails this bar.

Mechanistic argument (from the advantage math, not run history):
- GRPO's per-group gradient ∝ (r_i − r̄). When the whole 16-sample group lands on one reward rung, the gradient is **exactly zero** — no hyperparameter recovers it. With `scale_rewards="none"`, a 2-rung group with spread 0.02 yields advantages of ±0.01: nonzero but ~50× weaker than a healthy continuous-reward group. The fix converts "explode on ~77% of steps" into "no-op on ~77% and whisper on most of the rest" = the predicted STABLE-BUT-FLAT.
- Step economics: a 1000-step run ≈ 2,000 prompt-groups; at ~77% dead only ~460 carry gradient — ~3% coverage of the 14,682-row corpus, on coarse rungs, at lr 1e-6. Expected held-out movement is below the eval's noise floor.
- The premise is structurally mismatched: the reward was designed for trajectories but the data feeds single turns. `chain_propagation` is dead-constant (no `state_sequence` ≥2), `task_completion` dropped on 85.8% of rows, tool reward binary on 61.1% of rows. Three of five components are inert or binary on the majority of the data **by row-layout construction** — no reward reweighting fixes a support problem.
- "Saturated-correct" groups (8 phrasings all scoring 1.0) are the reward *correctly* reporting "nothing to learn here." Manufacturing variance inside them optimizes an off-target axis; the correct treatment is to *exclude* them — which online GRPO can't do cheaply and offline methods do for free.

## 2. Ranked interventions

| # | Path | Unlocks learning? (mechanism) | Eng | Compute | Risk | Falsifier |
|---|------|-------------------------------|-----|---------|------|-----------|
| **1** | **RFT / rejection-sampling SFT** — sample N=8/prompt, rank by existing reward, SFT on per-prompt winners | **High.** Needs only argmax within group — a 2-rung split suffices; no std, no advantage norm, no optimizer fragility. Extracts signal from every prompt with ≥2 occupied rungs; skips saturated/collapsed for free. | ~1–2 d (extend preflight sampler + keep-top filter; `sft.py` exists) | ~10–20 H100-h full; ~1–2 h for the 500-prompt gate | Distills only best-of-8; can amplify GT noise in winners | Best-of-8 headroom < 15% of 500 prompts → nothing to distill → kill |
| **2** | **DPO/KTO from reward-ranked pairs** (byproduct of #1's samples) | **Med-high.** Pairwise needs 2 rungs; adds the *negative* gradient RFT lacks. KTO covers ≤2-rung prompts via unpaired rows. | ~1–2 d on top of #1 | reuses #1 samples | narrow pairs can degrade fluency/format; needs held-out guard | < ~2,000 usable pairs (gap ≥ 0.1) → underpowered |
| 3 | Stabilized 50-step GRPO diagnostic (merged) | Low as learning; moderate as measurement — subsumed by the 12-min preflight at ~1/10 cost | 0 | ~2–4 h | sunk-cost "just 200 more steps" | `reward_std` > 0.05 sustained + held-out ↑ |
| 4 | Data enrichment targeting the gap (multi-tool/terminal/≥2-transition rows — **generator** change, not re-filter) | Med, slow — attacks the true support problem; helps all methods | ~3–5 d + teacher $ | teacher tokens | drift; "new data same lattice" if specs unverified vs variance | new batch's `frac_collapsed_groups` unchanged ±5pts |
| 5 | (a*) Multi-turn rollout env + trajectory reward | High ceiling, wrong time — makes task_completion/chain live, matches deployment; ~1 wk net-new, 3–5× compute, inherits GRPO fragility + sim risk | ~1 wk+ | high | sim-to-real; compounding surface after 4 kills | build only if #1/#2 improve then plateau below target |
| 6 | (c) Semantic prose-similarity reward | **Low/negative** — continuous variance on an axis the held-out composite excludes → paraphrase hacking (`length_band` precedent) | ~1 d | — | high (documented failure class) | guardrail fires: train ↑, held-out ↓ |
| 7 | (a) row-per-conversation / (b) reweight | alignment hygiene, zero within-group variance; (b) landed marginal | low | — | low | already falsified by unchanged structural stats |
| 8 | SFT hygiene: seed from loss-elbow ckpt-1000 + early-stop / `load_best_model_at_end` | not a standalone unlock (entropy wasn't the bottleneck) but the mandatory base for whichever path wins | hours | — | none | — |

## 3. Recommendation

**Abandon single-turn online GRPO. Adopt Path #1 — rejection-sampling fine-tuning (RFT) using the existing graded reward as an offline ranker — gated by a 1-day headroom probe, with #2 (DPO/KTO) as a second stage built from the same sample corpus. Keep the graded reward; retire the online optimizer.**

Why #1 beats the runner-up (#3, run the already-built GRPO diagnostic):
1. **Same compute, strictly greater extraction.** GRPO consumes the ~32k generations online, paying full price for ~77% dead groups and a coarse-lattice gradient at lr 1e-6. RFT consumes the same completions offline: every prompt with any spread yields a full-strength SFT target; dead/saturated prompts cost only sampling tokens. No GRPO config extracts more information per generated token than argmax-and-distill.
2. **Removes the failure class that killed four runs.** All four post-mortems trace to advantage-normalization × near-zero group variance. RFT has no advantage, no group std, no KL controller, no reference policy — the discrete lattice that is *fatal* to GRPO is *sufficient* for ranking.
3. **The diagnostic answers a question whose answer doesn't change the plan.** STABLE-BUT-FLAT is already predicted; confirming it costs 2–4 H100-h and invites another iteration cycle. The 12-min preflight measures the same decision-relevant quantities plus the RFT headroom GRPO telemetry can't see.
4. **Honest ceiling, honestly priced.** RFT can't exceed best-of-8 of the current policy — but with beta 0.05 / lr 1e-6 neither could this GRPO. If best-of-8 headroom is real (the probe tells us in a day), RFT captures it at near-zero risk; if not, *no* single-turn method works and we reach that verdict in 1 day instead of a fifth kill.

Explicitly rejected: (c) semantic-similarity reward (predicted reward-hack, precedent on file); #5 multi-turn env as the *next* step (right ceiling, wrong sequencing — a week of net-new simulator before a 1-day probe of the cheap path).

## 4. Next-experiment spec — "Headroom Probe → RFT Pilot" (< 1 H100-day, gated)

**Base for all stages:** `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000` (loss-elbow seed; ckpt-500 fallback). First resolve the provenance caveat: verify `data/output/grpo/task_a/{train,validation}.jsonl` hashes vs the W&B `uklfswk5`/`bqbxnqxw` artifacts (validation lags train by 8 days).

### Stage 0 — Headroom probe (~2 H100-h)
- **Harness:** extend `scripts/preflight_entropy_diag.py` (already samples N/prompt and scores per-prompt `rewards[]`, `reward_std`, `frac_collapsed_groups` at `_score_and_summarize`). **Net-new (~50–80 LOC):** (i) one greedy (T=0) completion/prompt, scored; (ii) `headroom = max(rewards) − r_greedy`; (iii) summary `frontier_frac = %prompts with headroom > 0.05`, `mean_headroom`, rung-occupancy histogram.
- **Config:** 500 prompts stratified by (has-tool × terminal_reached × L-band) from `train.jsonl`, N=8, T=0.8 / top_p 0.95 (match `grpo_cat_a_diagnostic.yaml` so numbers transfer). 500 prompts → ±4% CI on frontier_frac.
- **GO (RFT):** `frontier_frac ≥ 15%` AND `mean_headroom ≥ 0.03`.
- **GRPO-revival branch:** `frac_collapsed_groups < 0.50` AND median `reward_std ≥ 0.05` → §1 condition met; run the already-built 50-step diagnostic before committing to RFT.
- **NO-GO (both):** `frontier_frac < 10%` → no single-turn method has headroom; escalate to §5.2 (multi-turn env vs ship SFT-only). Kill the single-turn RL/RFT track.

**Result (2026-07-09, ckpt-1000, 500 prompts, N=8, T=0.8 / top_p 0.95, ~5.6 h wall):** verdict **MARGINAL**. `frontier_frac = 0.130` (GO needs ≥ 0.15 — miss by 2 pts, but well clear of the 0.10 NO-GO floor), `mean_headroom = 0.0448` (GO ≥ 0.03 — pass). GRPO-revival branch firmly ruled out: `frac_collapsed_groups = 0.716`, median `reward_std = 0.000`. Rung-occupancy histogram `{1: 357, 2: 118, 3: 19, 4: 5, 5: 1}` — 71% of prompts collapse all 8 samples to one reward rung; `frac_positive_headroom = 0.132`, so best-of-8 beats greedy on essentially only the same 13% of prompts. Headroom is real and above threshold *where it exists*, but too concentrated. The GRPO death certificate stands (matched-settings collapse confirms §1). Next move per the gate: re-probe a later checkpoint (1500/2000) or raise T→1.0 to widen sampling diversity before committing to the Stage-1 pilot. Artifact: `runs/preflight/rft_headroom_ckpt1000.json` (gitignored).

**Re-probe (2026-07-16, ckpt-1000, same config, on the post-r12fix regenerated corpus, ~4.8 h wall):** verdict **MARGINAL — unchanged**. This is the item #5 / §5.6 re-probe (see [`grpo_tool_emission_gap_review.md`](grpo_tool_emission_gap_review.md) §6.2) triggered by the corpus regeneration that removed the malformed-role contamination (R12). Regenerating the data did **not** shift the lattice the way condition #5 anticipated: `frontier_frac = 0.122` (down from 0.130 — *still* below the 0.15 GO bar, still clear of the 0.10 NO-GO floor), `mean_headroom = 0.0414` (≥ 0.03 — still pass). GRPO-revival branch stays dead and slightly worse: `frac_collapsed_groups = 0.764` (up from 0.716), median `reward_std = 0.000`, rung histogram `{1: 382, 2: 98, 3: 18, 4: 2}` (more concentrated on rung 1 than before). So the clean-data hypothesis is falsified for the RFT/GRPO question — the corruption fix was correct on its own merits but did not unlock single-turn RL headroom. Paired with the SFT-only greedy composite re-measurement on the same regenerated data (**0.7167, still FAIL** vs 0.75 — `grpo_tool_emission_gap_review.md` §6.1), this lands squarely in condition #2's neighborhood (below target, no cheap RL lever): next move is the reward/GT audit on ~50 announce-but-don't-call rows + re-deriving the target bar for the per-turn-fair metric + the outstanding pass@k (T≈1.0) probe on tool-expected rows — **not** a blind RFT pilot. Artifact: `runs/preflight/rft_headroom_ckpt1000_regen.json` (gitignored).

### Stage 1 — RFT pilot (~6–10 H100-h, only on Stage-0 GO)
- **Sampling:** N=8, T=0.8 over a 4,000-prompt stratified subset (bias to tool-bearing + terminal). Serve the *merged* checkpoint via `merge_adapter.py` + `serving/launch_vllm.sh` (R9: plain vLLM, no Unsloth fast_inference for gemma4). Prefix caching on.
- **Filter:** keep top-1/prompt where `max(reward) ≥ max(0.8, r_greedy + 0.05)`; drop saturated (headroom < 0.05) and collapsed. Expected yield ~600–1,500 rows. Write as chatml `train.jsonl` for `sft.py`.
- **Train:** existing `sft.py`, 1 epoch, lr 5e-6, LoRA on ckpt-1000.
- **Gate:** `_heldout_composite_score` (strict, training-reward-independent) on 200 fixed `validation.jsonl` prompts, greedy, before vs after.
  - **GO:** Δcomposite ≥ **+0.02** → scale to full corpus, then build DPO/KTO pairs (Stage 2).
  - **KILL:** Δcomposite < +0.01, or any decline in tool-emission rate on tool-GT rows.
- **Budget cap:** Stage 0 + 1 ≤ 1 H100-day. No stage extended past its gate.

**Update (2026-07-13) — SFT-only ceiling check ran; fork resolved to "neither branch."** See
[`grpo_heldout_metric_and_gt_audit.md`](grpo_heldout_metric_and_gt_audit.md). The greedy held-out
composite FAILed at 0.479, but a row-level audit showed this is mostly a scoring-granularity
artifact (whole-conversation metric applied to per-turn rows); the honest **per-turn-fair** baseline
is **0.674** (still < 0.75). Two fixes landed in `training/grpo.py`: (1) `_heldout_composite_score`
is now per-turn-fair, and (2) a schema-aware GT sanitizer drops 32 fabricated-required-arg tool
calls from the reward target (confirms and partially closes item 3 below). Residual gap is one
genuine weakness — "announce-but-don't-call" tool under-emission. **Re-run Stage 0 (headroom) and
the Stage-1 gate baseline on the clean inputs before any RL commit** — the MARGINAL reads above were
partly artifact-driven.

**Resolved (2026-07-16):** the "re-run Stage 0 on clean inputs" instruction is done (see the
re-probe paragraph under Stage 0 above and `grpo_tool_emission_gap_review.md` §6). Outcome: the
MARGINAL read was **not** artifact-driven — on the fully regenerated, corruption-free corpus the
verdict holds MARGINAL (`frontier_frac 0.122`, `mean_headroom 0.0414`) and the SFT-only greedy
composite is still a FAIL at 0.7167. Do **not** commit to an RL/RFT run on this evidence; proceed to
the reward/GT audit + per-turn target-bar re-derivation first.

## 5. What would change this recommendation
1. **Preflight surprises high** (`frac_collapsed_groups < 0.50` + median `reward_std ≥ 0.05` at diagnostic settings) → §1 condition met, GRPO's death certificate withdrawn; the merged 50-step diagnostic becomes the correct next run (online RL then plausibly beats RFT by also learning from negatives).
2. **No headroom anywhere** (`frontier_frac < 10%` and greedy composite already ≥ target) → SFT policy is at the single-turn ceiling; ship SFT-only for Phase 2 Cat A, or fund the multi-turn env (#5) if Phase 4 E2E demands more.
3. **RFT pilot fails its gate despite real headroom** → winners aren't transferring → GT/reward misspecification (e.g. the ~17% placeholder-args stubs, `_strip_placeholder_args` in `reward_business_logic.py`); audit reward vs human judgment on 50 winner/loser pairs before further training. **(2026-07-13: confirmed & partially closed — see [`grpo_heldout_metric_and_gt_audit.md`](grpo_heldout_metric_and_gt_audit.md). 69 GT tool calls fired an action with a fabricated required arg; 32 row-eligible ones now sanitized at load in `_load_grpo_jsonl`. The hypothetical-intent over-eager pattern remains for manual review.)**
4. **Provenance check fails** (on-disk splits ≠ W&B artifacts) → re-measure all structural stats before trusting these conditions.
5. **Generator-side data revision** (#4) shifts the lattice (`frac_collapsed_groups` down ≥20 pts) → re-run Stage 0 on the new corpus; both the GRPO condition and RFT headroom could flip.
