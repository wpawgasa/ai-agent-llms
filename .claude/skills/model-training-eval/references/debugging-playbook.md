# Debugging Playbook

Symptom → diagnostic walk → likely causes → fixes. Use this when something is wrong; jump to the section that matches the symptom.

## Table of Contents

1. [The diagnostic ladder (full)](#1-the-diagnostic-ladder-full)
2. [Loss is NaN / Inf](#2-loss-is-nan--inf)
3. [Loss is flat or decreases trivially](#3-loss-is-flat-or-decreases-trivially)
4. [Loss diverges (increases) mid-training](#4-loss-diverges-increases-mid-training)
5. [Train loss decreases, val loss doesn't](#5-train-loss-decreases-val-loss-doesnt)
6. [Train and val both look fine, eval metric is bad](#6-train-and-val-both-look-fine-eval-metric-is-bad)
7. [Loss is suspiciously low at step 0](#7-loss-is-suspiciously-low-at-step-0)
8. [Gradient norm spikes or NaNs](#8-gradient-norm-spikes-or-nans)
9. [Gradients are zero in some layers](#9-gradients-are-zero-in-some-layers)
10. [Out-of-memory (OOM) errors](#10-out-of-memory-oom-errors)
11. [Throughput is much lower than expected](#11-throughput-is-much-lower-than-expected)
12. [GPU utilization is low](#12-gpu-utilization-is-low)
13. [Distributed training hangs](#13-distributed-training-hangs)
14. [Loss differs across ranks](#14-loss-differs-across-ranks)
15. [Cannot reproduce previous run](#15-cannot-reproduce-previous-run)
16. [Resumed run behaves differently from continuous run](#16-resumed-run-behaves-differently-from-continuous-run)
17. [Mixed precision instabilities](#17-mixed-precision-instabilities)
18. [Model performs well in training, poorly in production](#18-model-performs-well-in-training-poorly-in-production)
19. [RL-specific: reward stagnates or collapses](#19-rl-specific-reward-stagnates-or-collapses)
20. [LLM-specific: model generates garbage after fine-tuning](#20-llm-specific-model-generates-garbage-after-fine-tuning)

---

## 1. The diagnostic ladder (full)

Climb in order. Each rung is cheaper than the one above.

1. **Did anything change?** Code, data, env, hardware, library versions. `git diff`, `pip freeze` diff, dataset version.
2. **Sanity check the data.** Print 5-10 batches. Decode inputs and labels. Shapes/dtypes/devices correct? Masks correct? Examples readable?
3. **Overfit a tiny batch.** 4-32 samples, regularization off, augmentation off. Loss should approach 0 in <500 steps. If not, code bug.
4. **Inspect loss numerics.** Per-batch loss values — NaN, Inf, suspiciously constant, way too low, way too high?
5. **Inspect gradient flow.** Global grad norm; per-layer grad norms; ratio of update to weight (||Δw||/||w||).
6. **Inspect optimizer state.** Current LR (does it match schedule at this step)? β values? Optimizer state finite?
7. **Inspect distributed sync.** Loss matches across ranks at step 0? Grad all-reduce happening? Effective batch correct?
8. **Inspect evaluation pathway.** Eval on training data — does it match training loss? Metric well-defined?
9. **Only now consider model/HP changes.**

---

## 2. Loss is NaN / Inf

### Diagnostic walk

1. Is it NaN from step 0, or did it become NaN later?
2. Is loss the only NaN, or are gradients/weights also NaN?
3. Are you using fp16? Mixed precision? Any custom autocast?
4. Is there division, log, sqrt, or exp anywhere in the loss?
5. Are there outliers in the input data?

### Likely causes

- **Numerical overflow in fp16.** Logits or attention scores exceed fp16 range. Most common in attention's `softmax(QK^T)` for long sequences.
- **Loss scaler too high.** fp16 dynamic loss scaling allows occasional skipped steps; if every step is skipped, scaler is wrong.
- **Division by zero.** Often in normalization (zero variance batch) or custom loss (zero denominator).
- **log(0) or log(negative).** Cross-entropy with logits that aren't logits but probabilities (use `cross_entropy_with_logits` or apply log_softmax once, never twice).
- **Bad input.** NaN/Inf in the data, or in the labels. Inputs un-normalized, very large outliers.
- **Exploding gradients.** Gradients NaN propagates to weights, which propagates to loss next step.
- **Learning rate too high.** Causes weight explosion in 1-10 steps.

### Fixes

- Switch to bf16 if hardware supports it (A100+, H100, TPU). It has fp32 range and removes most overflow issues.
- Add `eps` to any denominator. `eps=1e-5` to `1e-8` depending on precision.
- Add gradient clipping (`max_norm=1.0` is a safe default for transformers).
- Validate data loader output: assert finite-ness, expected ranges.
- Use `torch.autograd.detect_anomaly()` (slow but pinpoints the offending op) for one diagnostic run.
- For attention: ensure `softmax` is computed in fp32, or use Flash Attention which handles this.
- Reduce LR by 10x and see if NaN goes away — if yes, it was an LR issue.

---

## 3. Loss is flat or decreases trivially

Symptom: loss stays near its initial value, or drops a tiny amount and plateaus far above expected.

### Diagnostic walk

1. Did the overfit-tiny-batch test pass?
2. What's the expected loss at random init? (E.g., CE on K-way classification ≈ log(K))
3. What's the current LR? Did warmup actually fire?
4. Is the model getting *any* gradient signal? Print per-layer grad norms.
5. Is the loss masked out everywhere by accident?

### Likely causes

- **LR too low.** Warmup never reaches the peak; or peak LR is set 100× lower than appropriate.
- **Optimizer mismatch.** Adam with default eps on a model designed for SGD, or vice versa.
- **Frozen parameters by accident.** `requires_grad=False` on the wrong module; `eval()` called when meaning `train()`.
- **All loss masked.** Labels are all `-100` (ignore index), or mask zeros out everything. Decode one batch to verify.
- **Wrong loss function for the task.** E.g., MSE when targets are categorical.
- **Bad initialization.** Layers initialized to constants; norm layer with zero scale; embedding all zeros.
- **Disconnected graph.** Output not actually depending on input (e.g., bug in forward pass returning a constant).

### Fixes

- Inspect per-parameter grad norm. Any parameter with norm > 0 is getting gradient; parameters with norm = 0 aren't being used.
- Increase LR by 10x; if loss starts dropping, LR was the issue.
- Run a "verify gradient" test: change a single input, check if model output changes.
- Confirm `optimizer.param_groups` includes all parameters you think it includes.

---

## 4. Loss diverges (increases) mid-training

Symptom: training was going fine, then loss spikes and either stays high or NaNs.

### Diagnostic walk

1. When did it diverge — step number, wall time?
2. Did anything change around that time — LR schedule transition, new data shard, checkpoint reload, learning rate warm restart?
3. Are gradient norms spiking before loss does?
4. Is there a specific batch causing the spike (reproducible)?

### Likely causes

- **Outlier batch.** A pathological example in the data with very long sequence, extreme values, or label noise causes huge gradients.
- **LR schedule mis-step.** Schedule didn't decrease as planned, or warmup ended into too-high LR for the current weight scale.
- **Loss scaler instability** (fp16) — scaler increased too aggressively.
- **Stale optimizer state** after data distribution shift (e.g., curriculum changes, new data source).
- **Weight decay accumulation.** Some optimizers (especially older AdamW implementations) interact badly with mid-training weight decay changes.

### Fixes

- Tighter gradient clipping (`max_norm=0.5` or `1.0`).
- Investigate the batch at the divergence step — print a sample, check sequence lengths and content.
- Skip-and-continue if a single batch is the problem; long-term, filter the offending example.
- For LLM pretraining, "skip steps when grad norm > threshold" is a standard guardrail in production training.
- Cosine schedule with a longer warmup; restart from a slightly earlier checkpoint with a lower peak LR if mid-training.

---

## 5. Train loss decreases, val loss doesn't

Classic overfitting OR distribution mismatch. Distinguish before fixing.

### Diagnostic walk

1. What's the gap shape — val loss plateaus and stays flat, or val loss decreases briefly then rises?
2. Are val examples genuinely held out from train (no leakage in reverse — train doesn't contain val)?
3. Are train and val from the same distribution? Same preprocessing?
4. Was augmentation accidentally applied to val?
5. Is the model big enough to potentially overfit?

### Likely causes — overfitting

- Model too big for the data
- Regularization too weak (low dropout, no weight decay, weak augmentation)
- Trained too long past optimal

### Likely causes — distribution mismatch (often misdiagnosed as overfitting)

- Train and val sampled differently (e.g., train shuffled, val time-ordered)
- Val preprocessing differs from train (different normalization, tokenization, resize)
- Augmentation applied to val
- Val labels noisier than train (or vice versa)

### Fixes — overfitting

- More data > more regularization. If possible, augment.
- Increase dropout, weight decay
- Stronger augmentation
- Early stopping
- Smaller model (last resort)

### Fixes — distribution mismatch

- Audit val pipeline end-to-end vs. train pipeline
- Visualize a train batch and a val batch — they should be from the same distribution
- For LLMs, verify chat template / tokenization matches between train and val

---

## 6. Train and val both look fine, eval metric is bad

The classic "eval is broken" or "wrong loss for the task."

### Diagnostic walk

1. Is the loss you're training actually correlated with the eval metric?
2. Is the eval running on the same model state as you think?
3. Is the eval format / prompt template same as training?
4. Are you measuring on the right distribution?

### Likely causes

- **Loss-metric mismatch.** Training MSE but evaluating top-1 accuracy. They're correlated but not identical; a model can have great MSE and mediocre top-1.
- **Tokenizer mismatch.** Especially common for LLMs — train tokenizer ≠ eval tokenizer.
- **Generation parameter mismatch.** Eval uses greedy decoding when sampling was intended (or vice versa).
- **Prompt format drift.** Train sees `<|user|>...<|assistant|>...` but eval sends raw text.
- **Eval set contamination.** Model has memorized eval examples from training data; high training-set eval, but real held-out performance is poor.
- **Metric implementation bug.** Off-by-one in the evaluation script.

### Fixes

- Eval the previous known-good model on the same eval. If that drops too, the eval is broken.
- Sanity check: eval on training data. Should be high (with caveat about memorization).
- Decode 10 model predictions manually. Are they obviously bad in a way that explains the metric?

---

## 7. Loss is suspiciously low at step 0

Step 0 loss should be near the random-init loss for the task. If it's way lower, something's leaking.

### Diagnostic walk

1. What's the expected loss at random init? (CE: log(K); MSE: variance of target)
2. Is the model actually freshly initialized, or did a previous checkpoint load?
3. Are labels being inadvertently leaked into the input?
4. Is loss being masked out heavily? (Many ignored tokens → low average loss)

### Likely causes

- **Pretrained weights loaded.** Easy to miss when expecting random init.
- **Label leakage into input.** E.g., for sequence prediction, target included in the input by accident.
- **Loss mask covers most tokens.** Mean loss over very few tokens; check the per-token loss histogram.
- **Loss reduction wrong.** Sum vs. mean confusion.

### Fixes

- Print expected vs. observed step-0 loss in your training log automatically.
- Decode an input and verify it doesn't contain the answer.
- Verify mask sums and number of effective tokens per batch.

---

## 8. Gradient norm spikes or NaNs

### Diagnostic walk

1. Per-layer or global?
2. Spikes vs. NaN: spikes mean optimization instability; NaN means numerical failure.
3. Which step?

### Likely causes

- Long sequence / outlier batch
- LR too high for current weight scale
- Initialization mismatch (some layers tiny, others huge)
- Mixed precision underflow → division by very small loss scale
- A specific operation: softmax saturation, log of very small probability

### Fixes

- Gradient clipping (`max_norm=1.0` is standard for transformers, sometimes 0.5 or 5.0)
- Investigate the specific batch
- Lower LR; longer warmup
- For attention: Flash Attention or fp32 softmax

---

## 9. Gradients are zero in some layers

Symptom: per-layer grad norm shows some layers receiving zero gradient.

### Likely causes

- **`requires_grad=False`** on those parameters
- **Layer not in compute graph** — built but not called in forward
- **Dropout = 1.0** somewhere disabling the path
- **Detach** somewhere breaking the gradient flow
- **Frozen by `eval()`** — but only batch-norm/dropout — parameters still get gradient unless explicitly frozen
- **Conditional branching** — layer used only in some forward passes; check for `if` statements based on training state

### Fixes

- Print `for n, p in model.named_parameters(): print(n, p.requires_grad, p.grad is not None)` after a backward pass.
- Check forward pass returns a tensor that depends on every parameter you expect (run a dummy forward + backward and check `.grad`).

---

## 10. Out-of-memory (OOM) errors

### Diagnostic walk

1. OOM at startup, mid-training, or after N steps (memory leak)?
2. Forward pass, backward pass, or optimizer step?
3. Are you using gradient checkpointing? Mixed precision?
4. What is the *theoretical* memory budget? (Model + grads + optimizer + activations)

### Memory cost (rough)

For an N-parameter model in mixed precision:
- Weights: 2N bytes (bf16) + 4N bytes (fp32 master)
- Gradients: 2N or 4N bytes
- Optimizer (Adam): 8N bytes (m + v in fp32)
- Activations: dominated by batch × seq × hidden² × layers

Adam optimizer state can be larger than the model itself.

### Fixes by category

**Reduce activation memory:**
- Gradient (activation) checkpointing — recomputes activations in backward
- Reduce batch size; use gradient accumulation to maintain effective batch
- Reduce sequence length (truncate or pack better)
- Use Flash Attention (lower activation memory than vanilla attention)

**Reduce model memory:**
- FSDP / ZeRO-3 — shard params + grads + optimizer across devices
- 8-bit optimizers (bitsandbytes) — quantized optimizer state
- LoRA / QLoRA — most of model frozen, train small adapters

**Reduce data loader memory:**
- Lower num_workers if RAM is the issue
- Streaming datasets instead of in-memory

**Mid-training OOM (memory leak):**
- Tensors held in lists across iterations (often loss values not `.item()`-ed)
- Accumulating optimizer state during eval
- `torch.cuda.empty_cache()` only masks issues; find the leak

---

## 11. Throughput is much lower than expected

Symptom: samples/sec or tokens/sec is far below what hardware should achieve.

### Diagnostic walk

1. Profile. `torch.profiler` or NVIDIA Nsight to see actual time breakdown.
2. GPU utilization: high (compute-bound) or low (data-bound or sync-bound)?
3. Is throughput stable or degrading over time?

### Likely bottlenecks

- **Data loader.** Insufficient workers, slow disk, slow preprocessing, no prefetching.
- **Small batch / short sequence.** GPU compute units under-utilized.
- **Communication-bound** (distributed). Gradient all-reduce dominating step time.
- **Inefficient kernels.** Custom Python loops in forward pass; not using fused kernels.
- **CPU bottleneck.** Tokenization, augmentation, host-device copies on the critical path.
- **Thermal throttling.** Long runs degrade as GPUs heat up.

### Fixes

- Profile first, optimize second. Don't guess.
- Increase data loader `num_workers` and `prefetch_factor`.
- Use Flash Attention (often 2-3x speedup for attention-heavy models).
- For DDP, increase batch size if possible; gradient accumulation hides comm cost.
- For long sequences, use sequence parallel or sparse attention.
- Use `torch.compile` (PyTorch 2.x) for graph capture and kernel fusion (caveat: not always faster, sometimes increases compile time significantly).

---

## 12. GPU utilization is low

Almost always means the GPU is waiting on something else.

### Likely causes

- **Data loader bottleneck.** Steps wait for data. Fix: more workers, prefetching, faster preprocessing, in-memory datasets if data is small.
- **Small batch.** Under-utilization of compute. Fix: larger batch + gradient accumulation.
- **Frequent sync points.** `.item()` calls, CPU-side prints, frequent checkpointing on critical path.
- **Distributed sync overhead.** Especially for small models on many devices.
- **Mixed precision unused.** fp32 on hardware optimized for bf16/fp16.

### Diagnose

- `nvidia-smi dmon` — watch GPU util over time
- `torch.profiler` with `record_shapes=True` — see what's running and how long
- Look for "gaps" in the timeline — those are wait states

---

## 13. Distributed training hangs

Symptom: training stalls — no progress, no crash.

### Likely causes

- **Uneven batch counts across ranks.** Some ranks try to do an extra step and wait at barrier.
- **Rank-conditional code.** Rank 0 does eval while others wait at next step; if eval fails or doesn't sync, deadlock.
- **NCCL timeout.** Network issue; some rank stops responding. `NCCL_DEBUG=INFO` reveals.
- **Process imbalance.** A rank crashed silently; others hang waiting for it. Check `dmesg` and OOM killer.
- **Mismatched collective calls.** All ranks must call the same collective in the same order.

### Fixes

- Use `dist.barrier()` carefully — must be called on all ranks or it deadlocks.
- Set `NCCL_TIMEOUT` to a reasonable value (default is very long).
- Drop-last in data loader if batch counts can be uneven.
- Wrap multi-rank logic to ensure symmetric calls.
- Watch each rank's stdout — if one stops emitting logs, it's likely dead.

---

## 14. Loss differs across ranks

In DDP, loss should be approximately equal across ranks (modulo per-rank batch contents).

### Likely causes

- **Sampler not sharded.** Every rank seeing the same data → identical losses (the giveaway is *exact* equality).
- **Wrong sampler.** `RandomSampler` instead of `DistributedSampler`, or `DistributedSampler` not given correct `num_replicas` / `rank`.
- **Gradient all-reduce not happening.** Means parameters drift; losses diverge over time.

### Diagnose

- Print effective batch and rank-specific batch at startup.
- Add an explicit check: all-reduce mean of loss across ranks every N steps; should equal local loss approximately.

---

## 15. Cannot reproduce previous run

### Likely causes

- **Seeds not all set** — Python random, NumPy, PyTorch (both CPU and CUDA), framework-specific. Different operations consume RNG differently.
- **`torch.backends.cudnn.deterministic=False`** — default; deterministic algorithms are slower but reproducible.
- **CUDA non-deterministic operations** — some ops (e.g., `scatter_add`) are inherently non-deterministic on GPU.
- **Different hardware** — different GPU generations can produce numerically slightly different results.
- **Different library versions** — PyTorch, CUDA, NCCL, transformers versions all matter.
- **Data loader order** — multi-worker shuffling depends on seeds + worker behavior.
- **Wall-clock-dependent code** — `time.time()` somewhere influencing behavior.

### Fixes

- Set all seeds; use a `seed_everything` utility.
- `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=":4096:8"`.
- Save full environment (`pip freeze`, `conda env export`) with each run.
- For true bit-exactness across runs, accept the ~10-30% throughput cost.
- For *near* reproducibility (acceptable for most work), seed everything and accept minor numerical drift.

---

## 16. Resumed run behaves differently from continuous run

Common and frustrating. The run *resumes* but training trajectory diverges from what a continuous run would have done.

### Likely causes

- **Optimizer state not resumed.** Resume loads weights but not optimizer (m, v for Adam). Training restarts as if step 0.
- **LR scheduler state not resumed.** Schedule starts over → wrong LR for the step.
- **Data loader state not resumed.** Resume sees examples from the start of the epoch again.
- **RNG state not resumed.** Different augmentations, different dropout patterns.
- **Step count not resumed.** Schedule consults step count to compute LR.

### Fixes

- Save and reload all of: model, optimizer, scheduler, data loader state, RNG states (CPU + each GPU), step counter, run config.
- Test the resume path explicitly: train for N steps, save, resume, train for M more steps; compare to a continuous N+M-step run. Losses should match closely.

---

## 17. Mixed precision instabilities

See also: §2 (NaN losses) and §10 (cross-cutting mixed precision in pipeline-design).

### Symptom-specific guidance

- **Loss explodes early in fp16:** loss scale too low → underflow on gradients → optimizer step does nothing → eventual divergence. Or scale too high → overflow → NaN. Use dynamic scaling.
- **Loss steady in fp16 but val accuracy drops vs fp32:** weight update precision too low. Use fp32 master weights for optimizer.
- **bf16 with no scaling, still NaN:** activation overflow (range OK but precision lower than fp16). Often softmax outputs; use fp32 softmax for affected layers.
- **fp8 training instability:** outlier activations. Need careful scaling; consider per-tensor scaling rather than global.

---

## 18. Model performs well in training, poorly in production

The deployment gap. Almost always one of:

- **Distribution shift** — Production data isn't the data you trained on. Audit the assumed distribution against actual production samples.
- **Pre/post-processing mismatch** — Tokenization, normalization, image resize, audio sampling rate differs between training and production code paths.
- **Stale training data** — Real-world distribution drifted; the model is now learning yesterday's task.
- **Eval contamination during training** — Model memorized eval-set-like examples that aren't in actual production.
- **Format mismatch** — Especially LLMs: production calls in a different prompt format than training saw.
- **Inference-time settings differ** — Decoding strategy, temperature, top-p differ from what eval used.

### Diagnose

- Run a held-out *production sample* through both training and production pipelines. Compare intermediate tensors.
- Verify identical pre/post-processing code paths (ideally shared, not duplicated).

---

## 19. RL-specific: reward stagnates or collapses

### Reward stagnates (won't improve)

- **Reward is too sparse.** Add shaping reward, curriculum, or intrinsic motivation.
- **Exploration insufficient.** Increase entropy bonus, use noisy nets, randomize policy more.
- **Learning rate too low / too high.** RL is more LR-sensitive than supervised.
- **Advantage estimation bug.** Print advantages — should have mean ~0, reasonable variance.
- **Replay buffer too small / stale.** Off-policy methods need diversity.

### Reward collapses (was improving, then drops)

- **Catastrophic forgetting** — Off-policy buffer or distribution shift causes policy to unlearn.
- **Reward hacking.** Policy found an unintended way to maximize. Investigate top-reward trajectories — are they doing the task or gaming it?
- **Critic bias** — Off-policy critic is over-/under-estimating values; actor follows the bad signal.
- **For LLM-RL specifically:** KL to reference too low → policy drifts far from reference and loses coherence; increase KL penalty β.

---

## 20. LLM-specific: model generates garbage after fine-tuning

Symptom: model produces nonsense, repetition, or random tokens after SFT/DPO.

### Likely causes

- **Loss computed on prompt tokens.** Verify loss mask covers only the response portion. Decode `labels` for one example and check.
- **Chat template mismatch at inference.** Train used `<|im_start|>user\n... <|im_end|>` but inference sends raw prompt.
- **Tokenizer doesn't have the expected special tokens.** Or has them but the model wasn't trained on them.
- **LR too high for full fine-tuning.** Standard SFT LR (1e-5 range for full FT) often differs by 100x from LoRA LR (1e-4 range). Mismatched LR destroys the model.
- **Catastrophic forgetting.** Especially with full fine-tuning on small data. Use LoRA, mix in pretraining-style data, or reduce LR/epochs.
- **Loss masking bug results in training on `<pad>`.** Verify pad tokens are masked.

### Diagnose

- Run inference with a prompt that worked pre-fine-tuning. If it's garbage, you broke the base model.
- Check the train/val loss carefully. SFT loss should drop from ~3-5 to ~0.5-1.5; not lower (overfitting) and not staying at 3+ (not learning).
- Decode 10 training labels and verify they look like what the model should produce.

---

## General prevention practices

These eliminate entire classes of bugs:

1. **Decode-a-batch sanity check** — Always include a step in the pipeline that decodes one batch (input, label, mask) at the start of training and prints it. Catches tokenization, mask, format bugs immediately.
2. **Loss-at-init assertion** — Assert step-0 loss is within expected range. Catches init bugs, loaded weights, label leakage.
3. **Throughput baseline** — Record expected tokens/sec or samples/sec for your hardware; alert if a run is far below.
4. **Grad-norm check** — Log global grad norm every step; alert if NaN, very high, or zero.
5. **Eval-on-train spot check** — Run eval pipeline on a small sample of training data; should match training loss closely. Catches eval pipeline bugs without needing to wait for real eval to fail.
6. **Resume-from-checkpoint test** — Once per project, test that resume produces same trajectory as continuous.
7. **Distributed sanity at startup** — Print effective batch, world size, and a hash of a fixed input across ranks; mismatch reveals sync issues immediately.
