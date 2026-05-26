---
name: model-training-eval
description: Design, debug, and analyze training pipelines for any trainable ML model. Use when Claude needs to architect a training pipeline, diagnose training failures (NaN losses, exploding gradients, plateaus, train-val gaps), audit training dynamics (loss curves, gradient flow, calibration), design evaluation protocols, or optimize training efficiency. Covers LLMs, vision, speech, RL, tabular, multimodal, time-series, and classical ML — framework-agnostic. Trigger whenever the user mentions training a model, fine-tuning, pretraining, SFT, RLHF, debugging a training run, loss curves, gradient issues, hyperparameter tuning, overfitting/underfitting, evaluation metrics, benchmark design, dataset splits, data leakage, mixed precision issues, distributed training failures, or anything about why a model "isn't learning." Use even when the user describes symptoms in casual terms ("my loss is weird", "the model isn't converging", "I think I'm overfitting").
---

# Model Training & Evaluation Skill

Design, debug, and reason about training pipelines for any trainable ML model — LLMs, vision, speech, RL, tabular, multimodal, time-series, and classical models — with the rigor of a senior ML practitioner.

This skill treats training as a system: data → optimization → evaluation → analysis. Most "model" problems are really data, optimization, or evaluation problems in disguise. The skill's job is to make that visible and fixable.

## Service Modes

| Mode | Trigger | Output |
|------|---------|--------|
| **Pipeline Design** | "design a training pipeline", "how should I train", "set up SFT/pretraining/RL", "training architecture" | End-to-end pipeline spec |
| **Debug** | "loss is NaN", "not converging", "exploding gradients", "model isn't learning", "training crashed" | Diagnostic walkthrough + fixes |
| **Dynamics Analysis** | "analyze loss curve", "is my training healthy", "interpret gradients", "explain the metrics" | Annotated analysis with signals |
| **Evaluation Design** | "how to evaluate", "design benchmarks", "eval protocol", "metrics for X" | Eval framework with splits, metrics, leakage controls |
| **Optimization** | "train faster", "reduce memory", "improve sample efficiency", "tune hyperparameters" | Concrete optimization plan |
| **Review** | "review my training code", "audit this pipeline", "is this setup correct" | Findings list with severity + fixes |

The modes overlap and can chain (e.g., a debug session often ends in an optimization or design change). Use the mode that fits the user's primary intent; pull in others as needed.

---

## Core Mental Model

Before any concrete work, anchor on this mental model — it shapes every downstream decision:

**A training run is a controlled experiment.** It has a hypothesis (this architecture + data + objective will produce this capability), an intervention (the training loop), and a measurement (evaluation). When the result is wrong, the cause lives in one of four places:

1. **Data** — Wrong distribution, leakage, label noise, tokenization mismatch, augmentation bug, split contamination
2. **Objective** — Wrong loss, mismatched labels, masked-out targets, reward misspecification, regularization too strong/weak
3. **Optimization** — LR too high/low, bad schedule, optimizer mismatch, numerical instability, distributed sync bug
4. **Evaluation** — Test leakage, wrong metric, prompt/format mismatch, judge bias, insufficient samples

**Always diagnose in this order.** Data first, evaluation last. The reason: data bugs masquerade as everything else, and evaluation bugs make you chase phantoms.

A second mental model worth holding: **the model is the easiest thing to change and almost never the actual problem.** Treat "swap the model" as a last resort, not a first move.

---

## Universal Training Workflow

Use this as the scaffold for Pipeline Design mode and as the diagnostic frame for Debug/Analysis modes.

```
1. Frame the problem → 2. Build the data pipeline → 3. Establish baselines → 4. Train with instrumentation → 5. Evaluate rigorously → 6. Iterate
```

### Step 1: Frame the Problem

Capture the following before writing any code. If the user hasn't specified one of these, ask — guessing leads to wasted compute.

- **Task type**: supervised / self-supervised / RL / contrastive / generative / etc.
- **Inputs and outputs**: shapes, dtypes, modalities, max sequence length, label space
- **Success metric**: what *primary* metric decides if this worked, plus 1-3 guardrails
- **Compute envelope**: GPUs/TPUs available, memory per device, time budget, training framework
- **Constraints**: latency at inference, model size limit, deployment target, regulatory/PII constraints
- **What "good" looks like**: a concrete number, behavior, or comparison ("beat the current production model on metric X by Y%")

Output a one-page **Problem Card** capturing the above. This is the contract for the rest of the work.

### Step 2: Build the Data Pipeline

Data quality dominates model quality. Build the data pipeline before the model, not after.

**Data pipeline checklist:**

- [ ] Source data inventory: where it comes from, license, freshness, size, schema
- [ ] Split strategy: random / stratified / time-based / group-based — chosen for the *deployment* distribution
- [ ] Leakage audit: no future info in past splits, no identity leakage across train/test, no label leakage in features
- [ ] Preprocessing pipeline: deterministic, versioned, separately runnable
- [ ] Augmentation strategy: applied only to train, not val/test; consistent with deployment-time inputs
- [ ] Quality checks: distribution stats, null/inf checks, label distribution, duplicate detection
- [ ] Loader: streaming if data > RAM, shuffled, packed (for sequence models), correctly batched
- [ ] Sample inspection: literally look at 20 random examples — every time, every dataset

See `references/pipeline-design.md` for domain-specific data pipeline patterns (LLM packing, CV augmentation, RL replay buffers, tabular CV, etc.).

### Step 3: Establish Baselines

Never start with a fancy model. Establish a ladder of baselines so you can attribute later improvements:

1. **Trivial baseline**: predict majority class / mean / zero / random — to confirm the metric is sane
2. **Simple baseline**: a shallow model or a tiny version of the target architecture
3. **Strong baseline**: published SOTA or current production model, run on the same eval

**Overfit a tiny batch first** (Karpathy's recipe). Take 2-32 samples, turn off regularization, and verify loss → near-zero. If a model cannot memorize a handful of examples, it cannot learn the real task — and the bug is almost certainly in your code, not your hyperparameters.

### Step 4: Train with Instrumentation

Instrument every run so that *if it fails, you can tell why without rerunning*. The cost of logging is trivial; the cost of a re-run on H100s is not.

**Always log:**

- Loss (per batch and smoothed) — and any auxiliary losses separately
- Gradient norm (global and per-parameter-group)
- Weight norm and weight update ratio (||Δw|| / ||w||) — target ~1e-3 for healthy SGD
- Learning rate (especially across schedule transitions)
- Throughput (samples/sec, tokens/sec) and GPU utilization
- Memory (peak per device)
- Validation metrics at a meaningful cadence

**Always save:**

- Run config (every hyperparameter, every flag, frozen as JSON/YAML)
- Code state (commit hash + diff if dirty)
- Environment (framework versions, CUDA, GPU type, seed)
- Periodic checkpoints (not just the last one)

See `references/training-dynamics.md` for how to interpret these signals.

### Step 5: Evaluate Rigorously

Evaluation is what makes the run *mean* something. Common eval failures: contamination, prompt-format mismatch, single-seed reporting, metric-on-wrong-distribution, judge bias.

**Eval design checklist:**

- [ ] Held-out test set never seen during training (incl. via dataset reuse)
- [ ] Eval distribution matches deployment distribution (or known mismatch is documented)
- [ ] Primary metric + guardrails reported together
- [ ] Multiple seeds where stochasticity matters (mean ± std)
- [ ] Per-stratum breakdowns (per-class, per-length, per-subpopulation)
- [ ] Statistical significance vs. baseline (paired bootstrap or t-test)
- [ ] Failure-mode analysis: look at the worst N predictions

See `references/evaluation-frameworks.md` for domain-specific eval protocols.

### Step 6: Iterate

Iterate as a sequence of controlled changes, one variable at a time, each tracked.

**Iteration template per change:**

- Hypothesis (what you expect to happen and why)
- Change (single variable, ideally)
- Result (delta on primary + guardrails)
- Decision (keep / revert / extend)
- Notes (anything surprising — these are gold)

Resist the urge to change three things at once. When you must, plan an ablation to attribute the win/loss.

---

## Debugging Methodology

When a training run misbehaves, use the **diagnostic ladder**. Climb in order — each rung is cheaper to check than the one above.

### The Diagnostic Ladder

| Rung | Check | Why it goes here |
|------|-------|------------------|
| 1 | **Did anything change recently?** Code, data, env, hardware? | Most regressions have an obvious culprit if you look |
| 2 | **Sanity check the data.** Print 10 batches. Inputs make sense? Labels match? Shapes/dtypes/devices right? Mask correct? | 50%+ of "training bugs" are data bugs |
| 3 | **Overfit a tiny batch.** 4-32 samples, no regularization. Does loss → 0? | Isolates code bugs from training/data bugs |
| 4 | **Check the loss numerics.** NaN / Inf / negative-when-shouldn't-be / suspiciously constant? | Cheap to inspect, tells you a lot |
| 5 | **Check gradient flow.** Per-layer grad norms — any zero or exploding? | Reveals dead layers, init issues, exploding paths |
| 6 | **Check optimizer state.** LR actually as configured? Schedule firing? Optimizer betas reasonable? | Schedule bugs are silent and common |
| 7 | **Check distributed sync.** Same loss across ranks? Gradients all-reduced? Effective batch right? | Multi-GPU bugs are invisible until they aren't |
| 8 | **Check eval.** Is the metric well-defined? Eval on train — does it match training loss? | Eval bugs can fake regressions |
| 9 | **Now consider model/hyperparameter changes.** | Last, not first |

See `references/debugging-playbook.md` for the full failure-mode catalog organized by symptom (NaN, plateau, train-val gap, OOM, slow throughput, distributed deadlock, reproducibility break, etc.).

### When You Genuinely Don't Know

If the ladder doesn't reveal the cause:

1. **Bisect the change set.** Find the last known-good commit; binary-search to the regression.
2. **Reduce to a minimal repro.** Smallest model, smallest batch, single GPU, smallest dataset that still shows the bug.
3. **Diff against a working baseline.** A canonical example (HF, official repo) running on your environment. Differences are the search space.
4. **Print everything.** When in doubt, log more. Don't trust your assumptions about what the values are.

---

## Training Dynamics Quick Reference

Use this table when reading a loss curve or training log. For deeper interpretation, see `references/training-dynamics.md`.

| Signal | Healthy | Warning | Common Cause |
|--------|---------|---------|--------------|
| Train loss | Decreasing, then plateaus | NaN, increasing, flat-from-start | LR too high, bad init, dead loss, frozen layer |
| Val loss | Tracks train, eventually plateaus or rises slightly | Diverges early from train | Overfitting, data leak inverse, distribution shift |
| Grad norm | Stable order of magnitude, decreasing | Spikes, NaN, → 0 across all layers | Outlier batch, exploding/vanishing gradients |
| Per-layer grads | All non-zero | Some zero, some huge | Dead layer, no skip connection, bad init |
| ||Δw|| / ||w|| | ~1e-3 | <1e-5 or >1e-2 | LR mis-set for this architecture |
| Activation stats | Stable mean ~0, std ~1 (post-norm) | Drifting, saturating | Norm layer misuse, init mismatch |
| Throughput | Constant | Degrading | Memory fragmentation, data loader stall, thermal |
| GPU util | >85% during compute phase | <50% sustained | Data loader bottleneck, small batch, sync overhead |
| LR schedule | Matches plan | Doesn't decay / spikes | Schedule misconfigured, step count off |

---

## Cross-Domain Principles

These hold across LLMs, CV, RL, tabular, and beyond. They earn their place in the SKILL.md (not a reference) because they prevent the most expensive mistakes.

1. **The deployment distribution is the only one that matters.** Train, validate, and evaluate on data that reflects it. If you can't, document the gap and design probes for it.

2. **Reproducibility is a feature, not a chore.** Pin seeds, framework versions, CUDA, data versions, code state. A run you can't reproduce is a run you can't debug.

3. **Compute spent on iteration speed pays itself back.** Faster eval, smaller dev set, better logging, quicker checkpoints — these compound across experiments.

4. **Most "the model is bad" findings are eval bugs.** Before declaring failure, eval the *previous* known-good model on the new eval to sanity-check.

5. **Make every experiment attributable.** One change at a time, tracked, with a written hypothesis. Heroic 12-change sprints destroy learning.

6. **Save the boring things.** Tokenizers, normalizers, label maps, vocabularies. These vanish silently and break things weeks later.

7. **Watch for silent failures.** Off-by-one masks, swapped channels, wrong device, padding included in loss, augmentation applied to val. These don't crash; they corrode.

8. **Cheap signals beat expensive signals.** Loss on a tiny eval every step beats a perfect eval every epoch. Calibrate the cost/info trade-off.

---

## Output Formats by Mode

### Pipeline Design Mode

```markdown
# Training Pipeline: [Task Name]

## 1. Problem Card
- Task type, inputs/outputs, success metric, compute envelope, constraints

## 2. Data Pipeline
- Sources, splits (with rationale), preprocessing, augmentation, quality checks, loader spec

## 3. Model & Objective
- Architecture choice + rationale, loss function, regularization, init scheme

## 4. Optimization
- Optimizer + hyperparameters, LR schedule, batch size, gradient accumulation, precision

## 5. Distributed / Hardware
- Parallelism strategy (DP/DDP/FSDP/TP/PP), memory budget, checkpointing

## 6. Instrumentation
- Logged metrics, checkpoint cadence, eval cadence, tracking backend

## 7. Evaluation Protocol
- Primary metric, guardrails, splits, statistical methodology

## 8. Risk Register
- Top 3-5 things most likely to go wrong + how we'll detect them

## 9. Iteration Plan
- Baselines → ablations → optimizations, in order
```

### Debug Mode

```markdown
# Debug Report: [Symptom]

## Symptom
[Concise description with evidence — log snippet, metric value, error]

## Diagnostic Walk
- Rung 1: [check] → [result]
- Rung 2: [check] → [result]
- ...

## Root Cause
[The actual cause, with evidence]

## Fix
[Minimal change to resolve. Code or config diff.]

## Prevention
[Logging / assertion / test that would have caught this earlier]
```

### Dynamics Analysis Mode

```markdown
# Training Dynamics Analysis

## Run Summary
[Config, duration, hardware, primary metrics]

## Signal-by-Signal Read
- Loss: [healthy / warning + evidence]
- Gradients: [...]
- Weights & updates: [...]
- Throughput: [...]
- Eval: [...]

## Interpretation
[What the combined signals say about training health]

## Recommendations
[Concrete next steps, ranked by expected impact]
```

### Evaluation Design Mode

```markdown
# Evaluation Framework: [Task]

## Goal
[What this eval is trying to measure and decide]

## Splits
- Train/val/test definition + rationale for split type
- Contamination controls

## Metrics
- Primary: [metric] — why
- Guardrails: [list] — why
- Per-stratum breakdowns: [list]

## Methodology
- Sample size + power justification
- Seeds, runs, aggregation
- Statistical comparison method

## Known Limitations
[Eval gaps and how to mitigate]
```

### Optimization Mode

```markdown
# Optimization Plan

## Current State
[Throughput, memory, time-to-target, primary metric]

## Bottleneck Analysis
[Where time/memory is actually going — profiler evidence]

## Proposed Changes (ordered by impact / cost)
1. [Change] → expected gain → effort → risk
2. ...

## Validation Plan
[How to verify each change doesn't regress quality]
```

### Review Mode

```markdown
# Training Pipeline Review

## Summary
[1-2 sentence verdict]

## Findings
| ID | Severity | Area | Finding | Suggested Fix |
|----|----------|------|---------|---------------|
| 1 | Critical | Data | ... | ... |

## Strengths
[What's working well]

## Recommendations Prioritized
[Top 3-5 actions]
```

---

## When to Read Which Reference

- **`references/pipeline-design.md`** — designing or reviewing a training pipeline for a specific domain (LLM SFT, CV detection, RL policy gradient, tabular GBM, etc.)
- **`references/debugging-playbook.md`** — diagnosing a specific symptom (NaN, plateau, OOM, distributed failure, reproducibility break, etc.)
- **`references/training-dynamics.md`** — interpreting loss curves, gradient stats, calibration, scaling behavior
- **`references/evaluation-frameworks.md`** — designing rigorous evals for a specific domain, including contamination controls and statistical methodology
- **`references/domain-recipes.md`** — quick recipes and known-good configs by domain (LLM, CV, RL, speech, tabular, multimodal, time-series)

Read the relevant reference whenever the user's question is more specific than this SKILL.md covers. The references are deeper and more concrete; this file is the navigation layer.

---

## Quality Gates

Before delivering, check:

- [ ] Mode-appropriate output format used
- [ ] Concrete recommendations, not generic advice
- [ ] Trade-offs and risks named, not glossed
- [ ] Numbers, ranges, or thresholds given where applicable (not just "tune the LR")
- [ ] Domain-specific reference consulted when domain is non-trivial
- [ ] Diagnostic order followed (data → objective → optimization → evaluation)
- [ ] Reproducibility considered (seeds, versions, config capture)
- [ ] Eval distribution alignment with deployment considered
