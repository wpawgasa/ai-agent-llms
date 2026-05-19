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
  policy?** The current Cat-A reward (`reward_business_logic`) is a 5-component weighted
  sum. If one component dominates (e.g., format_compliance always passing), the effective
  variance is squashed. Worth printing per-component scores on a sample batch.
- **Should we monkey-patch `unsloth.models.vision.VLLM_SUPPORTED_VLM` to enable vLLM
  rollouts?** (See CLAUDE.md Risk R9.) At ~12 s/step with HF generate, a 1000-step
  run is ~8.6h. With vLLM colocate we'd expect ~3-4× speedup. Trade-off: risk of
  failing deeper in vLLM engine init.
- **Is BNPO ever the right choice here?** BNPO's bias-reduction matters more once
  the policy is already stable. May be worth revisiting after the base GRPO run
  is converging cleanly.
