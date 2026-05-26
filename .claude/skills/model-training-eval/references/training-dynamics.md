# Training Dynamics Reference

How to read training curves, gradient stats, weight statistics, calibration, and scaling behavior — and what to do with what you read.

## Table of Contents

1. [Reading loss curves](#1-reading-loss-curves)
2. [Gradient analysis](#2-gradient-analysis)
3. [Weight & update statistics](#3-weight--update-statistics)
4. [Activation statistics](#4-activation-statistics)
5. [Learning rate schedules — diagnosing in flight](#5-learning-rate-schedules--diagnosing-in-flight)
6. [Calibration & uncertainty](#6-calibration--uncertainty)
7. [Scaling laws & compute optimality](#7-scaling-laws--compute-optimality)
8. [Per-stratum and failure-mode analysis](#8-per-stratum-and-failure-mode-analysis)
9. [Putting it together: training health checklist](#9-putting-it-together-training-health-checklist)

---

## 1. Reading loss curves

The loss curve is the single most informative training artifact. Most pathology shows up here first.

### Healthy patterns

**Standard supervised training:**

```
loss
 |\
 | \____
 |      \____
 |           \____________________
 |______________________________________  steps
```

Fast initial drop, then progressively slower decrease, then plateau. The plateau is task- and data-dependent — there's an irreducible loss floor (data noise / aleatoric uncertainty / task ceiling).

**Power-law decrease** (LLM pretraining, large-scale CV):

On a log-log plot, loss vs. compute often follows a near-straight line. Deviations from this line are diagnostic:
- Bending up early → too high LR, or under-trained warmup
- Bending down late → entering a new regime (saturation, or capacity ceiling reached)

### Pathological patterns

| Pattern | Likely cause |
|---------|-------------|
| Flat from step 0 | Frozen params, zero LR, broken graph, all-loss-masked |
| Decreases then explodes | LR too high, no gradient clipping, unstable mixed precision |
| Drops sharply then plateaus way above expected | Local optimum (init issue), capacity too small, regularization too strong |
| Oscillates wildly | LR too high, batch size too small for noise, optimizer mismatch |
| Train ↓, val ↑ early | Overfitting (small data) or distribution shift (val pipeline bug) |
| Train ↓ smoothly, val flat | Val is broken (wrong preprocessing, frozen at init, etc.) |
| Sudden drop mid-training | New data shard easier; or schedule transition; or LR warm restart |
| Sudden spike that recovers | Outlier batch; expected occasionally with strong clipping |
| Loss decreases on val but eval metric stays bad | Loss-metric mismatch, or eval format bug |

### Smoothing

Always look at both raw (per-step) and smoothed (EMA or moving average) loss. Smoothing hides outlier batches; raw shows them. Need both.

For very long runs, log-scale x-axis (steps) often reveals patterns invisible in linear scale. For very low loss, log-scale y-axis reveals slow decrease.

### Loss components

When using a composite loss (`L = α·L1 + β·L2 + ...`), log components separately:

- One component dominating gradient → effective single-objective training; reweight or normalize gradients
- One component flat → that head/branch isn't training; investigate why
- Components anticorrelated → trade-off; this is sometimes desired, sometimes a sign of conflicting objectives

For instruction-tuning + KL regularization (RLHF, DPO, etc.), watch KL separately. KL → 0 means policy hasn't moved; KL → very large means catastrophic drift.

---

## 2. Gradient analysis

Gradients are how the loss talks to the parameters. Inspecting them reveals issues invisible in loss alone.

### Global gradient norm

The L2 norm of the concatenated gradient vector across all parameters. Log this every step.

- **Stable order of magnitude (e.g., 0.5–5.0)** — healthy
- **Slowly decreasing** — typical late in training as loss flattens
- **Spiking occasionally** — outlier batches; tighter gradient clipping if frequent
- **NaN** — numerical failure; see debugging-playbook §2
- **→ 0 across many steps** — vanishing gradients; very deep network without skip connections, or stuck in flat region
- **Increasing over time, then exploding** — instability building up; lower LR, tighter clipping, longer warmup

### Per-layer / per-parameter-group gradient norm

This is where dead layers, exploding paths, and gradient flow problems show up.

**What "healthy" looks like:**

- All groups have non-zero gradient
- Order of magnitude similar across groups (within ~2 orders of magnitude)
- Deep layers (close to loss) typically have larger gradients than shallow layers
- For transformers: attention and MLP gradients similar order of magnitude

**Red flags:**

- **One layer has zero gradient** — disconnected from compute graph, frozen, or output not used in loss
- **First layer ≪ last layer** — vanishing gradients; check init, norm layers, skip connections
- **Specific layer ≫ others** — exploding path; one layer's outputs are extreme
- **Embedding gradient very different from rest** — common in transformers; usually fine, sometimes points to bad embedding init

### Update-to-weight ratio

`||Δw|| / ||w||` (norm of the optimizer update, divided by norm of the weights, per layer).

This is a *much* better LR diagnostic than the LR value itself:

- **~1e-3** — healthy for most architectures (Karpathy's rule)
- **<1e-5** — too small; LR likely too low; layer barely training
- **>1e-2** — too large; LR likely too high; risk of instability

Different layers can have different healthy ratios — embeddings often update slower than MLP weights, which often update slower than norm layers. The check is "is this ratio in a reasonable range, given this layer's role."

### Per-parameter-group LR

In transformer training, it's common to use different LRs for embeddings, norms, and main weights. Log per-group LR and verify it matches your configuration.

---

## 3. Weight & update statistics

Weights themselves carry information about training health.

### Weight norm trajectory

`||W||` per layer over training:

- **Slowly increasing then stabilizing** — normal for most layers
- **Continually increasing** — weight decay too low; instability building
- **Decreasing toward zero** — weight decay too high, or layer being "killed" by regularization
- **Sudden change** — schedule transition, checkpoint reload, or numerical event

### Weight distribution

Periodic histograms of weight values reveal:

- **Heavy tails** — common late in training; usually fine, but extreme tails indicate outliers
- **Bimodal** — sometimes a sign of dead neurons (some weights pushed to zero, others active)
- **Saturating at extremes** — bounded activations + un-bounded weights → loss of capacity

### Frobenius-norm trajectory of attention matrices (transformers)

Watch `||Q||`, `||K||`, `||V||`, `||O||` per layer. Imbalances reveal:

- One projection dominating → attention pattern collapse
- All very similar → potential underutilization

---

## 4. Activation statistics

Activations are how data flows through the network. Their statistics reveal flow problems early.

### What to monitor

For a representative batch, periodically log:

- Mean and std of activations after each block / major layer
- Fraction of "dead" units (e.g., for ReLU: fraction always 0; for GELU: fraction always near 0)
- Saturation: for bounded activations (tanh, sigmoid), fraction near the bounds

### Healthy patterns

- Post-norm activations have mean ~0, std ~1 (that's the norm layer's job)
- Pre-norm activations grow with depth — that's normal for pre-norm transformers; norm layers handle it
- Dead-unit fraction is small (<5%) and stable

### Red flags

- **Activations growing unbounded with depth** — norm layer missing, broken, or in wrong precision
- **Many dead units** (>30%) — too high LR caused dying ReLU; or initialization off
- **Activations very small** — vanishing forward signal; bad init, or extremely strong regularization
- **Dead-unit fraction increasing over training** — pathology; some path is permanently shutting down

### Outlier features (LLM-specific)

In trained transformers, a small number of channels develop very large activations (10x – 1000x larger than typical). These are normal in trained LLMs but they:
- Break naive quantization (need outlier-aware quant: SmoothQuant, GPTQ, AWQ)
- Concentrate in specific channels — predict which ahead of time

Tracking max-activation per layer is useful for quantization planning.

---

## 5. Learning rate schedules — diagnosing in flight

Logging the *current* LR every step is essential. The LR you set ≠ the LR the optimizer is using at every moment.

### Schedule sanity checks

- Does warmup actually reach the peak LR you configured?
- Does the schedule decay according to your formula at step k?
- After resume, is the LR consistent with what step k should have?
- For cosine: does it bottom out at the floor LR you set, not zero?
- For warm restarts: do the restart points actually fire?

### How LR affects dynamics

| Symptom | LR diagnosis |
|---------|--------------|
| Loss flat from start | LR too low; warmup too long; or schedule broken |
| Loss decreases then plateaus too early | LR too low at peak |
| Loss decreases then diverges | LR too high; or warmup too short |
| Loss noisy, oscillating | LR too high relative to batch; or batch too small |
| Loss decreases then suddenly stops | LR fell below useful range (cosine bottomed out) |

### LR range test (cyclical learning rates)

For unfamiliar models, run an LR range test: linearly increase LR from very small to very large over a few hundred steps. Plot loss vs LR. Pick a peak LR roughly 10x lower than where loss starts diverging.

This is the cheapest way to set a good LR.

### Batch size and LR

The square-root scaling rule (LR scales with √(batch_size)) is a rough guide. The "linear scaling rule" (LR ∝ batch_size, used at facebook for ResNet) often works up to medium batches. For very large batches, both rules break — LR can't scale forever. For LLMs, optimal LR weakly depends on batch in the typical training regime.

---

## 6. Calibration & uncertainty

A model's confidence should match its accuracy. Confidence without calibration is harmful in deployment.

### Expected Calibration Error (ECE)

Bin predictions by confidence; compute |confidence – accuracy| per bin; weighted average.

- ECE < 1% → well-calibrated
- ECE 5-10% → typical for SOTA classifiers without calibration
- ECE > 15% → poorly calibrated; needs post-hoc fix

### Reliability diagrams

Plot confidence (x) vs. accuracy (y) by bin. Healthy: on the diagonal. Above diagonal: under-confident. Below diagonal: over-confident.

Modern neural networks are typically **over-confident** out of the box. Fine-tuned LLMs are especially over-confident on tasks they're uncertain about.

### Calibration techniques

| Method | Cost | When |
|--------|------|------|
| **Temperature scaling** | Trivial; one parameter | Default for post-hoc calibration |
| **Platt scaling** | Cheap; logistic on logits | When temperature scaling insufficient |
| **Isotonic regression** | Cheap; non-parametric | More flexible, needs more val data |
| **Deep ensembles** | Expensive (N models) | When best calibration needed and compute allows |
| **MC Dropout** | Cheap at train; expensive at inference | When dropout is part of training |
| **Bayesian methods** | Expensive | When formal uncertainty needed |

### Selective prediction

Sort by confidence; report accuracy at coverage = X% (e.g., accuracy on top-80% most-confident predictions). Useful when deployment can defer low-confidence cases.

---

## 7. Scaling laws & compute optimality

### Power-law fits

For a fixed architecture family, loss as a function of compute, data, or parameters often follows a power law:

`L(C) ≈ L_∞ + (C₀ / C)^α`

where L_∞ is the irreducible loss, C₀ is a constant, and α is the scaling exponent.

This means: doubling compute reduces loss above the floor by a constant *factor*, not a constant amount.

### Chinchilla and compute-optimal training

For LLM pretraining, the "Chinchilla rule" (Hoffmann et al., 2022) suggests: optimal training uses ~20 tokens per parameter. For 7B model → ~140B tokens; for 70B → 1.4T tokens.

In practice:
- This is a *starting point*; modern recipes often train longer (more tokens per param) because inference cost matters and a smaller model trained longer is cheaper to deploy
- Different domains have different optimal ratios; data quality matters as much as quantity
- For fine-tuning, scaling laws are different and less well-characterized

### When to commit to a scale

Run a small-scale sweep first:

1. Train models at 3-5 small scales (e.g., 50M, 150M, 500M params), each for compute-optimal duration
2. Fit a scaling law to the resulting losses
3. Extrapolate to your target scale
4. Use the extrapolation to set hyperparameters (LR, warmup, batch) at target scale, often via known scaling rules (e.g., μ-parameterization, μP)

This is far cheaper than full-scale ablations.

### μ-parameterization (μP)

Reparameterizing so that optimal hyperparameters transfer across scales. The key claim: tune at small scale, transfer to large scale without retuning LR. Worth the implementation cost for serious scaling work.

---

## 8. Per-stratum and failure-mode analysis

Average metrics hide structure. Always break metrics down by relevant strata.

### Common strata

| Domain | Useful breakdowns |
|--------|-------------------|
| Classification | Per-class accuracy, confusion matrix, accuracy vs. example difficulty |
| Detection / segmentation | Per-class AP, per-size AP (small/medium/large), per-aspect-ratio |
| LLM | Per-task, per-domain, per-length, per-difficulty, per-language |
| Speech | Per-speaker, per-noise-level, per-accent, per-length |
| RL | Per-task suite, per-difficulty, per-seed |
| Tabular | Per-subpopulation (esp. for fairness), per-time-window |

### Failure-mode analysis

After eval, sort by loss / error; inspect the top-100 worst predictions.

You will find one of:

- **Mislabeled ground truth** — fix the eval set
- **A specific failure pattern** — e.g., model fails on long inputs, fails on a specific category
- **Out-of-distribution examples** — eval covers more than training; expected
- **Genuine model weakness** — improves with more data, different architecture, etc.

Quantify each. The model often has 2-3 dominant failure modes; addressing one can flip the metric significantly.

### Confusion analysis (classification)

A confusion matrix tells you:
- Which classes are confused with which
- Whether errors are symmetric (label noise) or asymmetric (bias)
- Whether classes are too fine-grained (consistent confusion → merge)

For very many classes (1000+), tracking top-confusions per class is more useful than the full matrix.

### Length / size analysis

For sequence models, accuracy vs. input length often reveals issues. Common patterns:
- Drops sharply at training-context-length boundary → length generalization failure
- Improves with length up to a point → benefits from context, plateaus when reasoning saturates
- Drops with length consistently → attention has trouble with long context; may need different positional encoding or sparse attention

### Subgroup robustness

For fairness-sensitive deployments, report worst-group accuracy alongside average. A model with 90% average / 60% worst-group is not a 90% model in practice.

---

## 9. Putting it together: training health checklist

When asked "is my training healthy", check in this order:

**Loss**
- [ ] Step-0 loss matches expected random-init loss
- [ ] Train loss decreases then plateaus, no NaN, no late divergence
- [ ] Val loss tracks train loss to within expected gap; no early divergence
- [ ] Loss-component balance (if composite): no component dominating

**Gradients**
- [ ] Global grad norm stable, no NaN
- [ ] Per-layer grad norms all non-zero, within ~2 orders of magnitude
- [ ] No layer with permanently zero gradient
- [ ] Update-to-weight ratio ~1e-3 across layers

**Weights**
- [ ] Weight norms stable or slowly growing, not exploding
- [ ] No dead-layer pathology
- [ ] No extreme outliers building up (unless expected, e.g., LLM outlier features)

**Activations**
- [ ] Post-norm activations have reasonable mean/std
- [ ] Dead-unit fraction small and stable
- [ ] No unbounded growth with depth

**Optimization state**
- [ ] LR matches schedule at current step
- [ ] Warmup completed as planned
- [ ] Scheduler not reset by an accidental optimizer rebuild

**Throughput**
- [ ] Tokens/sec or samples/sec at expected level for hardware
- [ ] GPU utilization >70% during compute phases
- [ ] Throughput stable over time (no degradation)

**Eval**
- [ ] Eval pipeline gives same answer on same model state across runs
- [ ] Eval-on-train sanity check matches training loss
- [ ] Per-stratum breakdowns show no surprising failure pattern
- [ ] Worst-N failure inspection done

If all of these are green, training is healthy. If a metric is red, jump to the corresponding section in `debugging-playbook.md`.
