# Pipeline Design Reference

Domain-specific training pipeline patterns. Use the section that matches the task; cross-reference for hybrid setups (e.g., multimodal = LLM + CV sections).

## Table of Contents

1. [LLM training](#1-llm-training)
2. [Computer vision](#2-computer-vision)
3. [Reinforcement learning](#3-reinforcement-learning)
4. [Tabular learning](#4-tabular-learning)
5. [Speech & audio](#5-speech--audio)
6. [Multimodal](#6-multimodal)
7. [Time-series & forecasting](#7-time-series--forecasting)
8. [Recommendation systems](#8-recommendation-systems)
9. [Cross-cutting: distributed training](#9-cross-cutting-distributed-training)
10. [Cross-cutting: mixed precision](#10-cross-cutting-mixed-precision)
11. [Cross-cutting: checkpointing & resumption](#11-cross-cutting-checkpointing--resumption)

---

## 1. LLM training

### 1.1 Stages

| Stage | Goal | Data | Compute order |
|-------|------|------|---------------|
| **Pretraining** | Learn general language/world knowledge | Trillions of tokens, web-scale corpus | $10^{22}$–$10^{25}$ FLOPs |
| **Continued pretraining** | Adapt to new domain or language | Billions of tokens, in-domain | 1–10% of pretrain compute |
| **SFT** | Follow instructions / format | 10K–1M examples, instruction-following | Hours–days on small clusters |
| **Preference tuning** (DPO/IPO/KTO) | Align to preferences | 1K–500K preference pairs | Hours |
| **RLHF/RLAIF** | Online optimization to a reward | Prompt set + reward model | Days |
| **Distillation** | Compress | Teacher outputs on prompts | Hours–days |

Pick the lightest stage that meets the goal. A clean SFT usually beats a sloppy RLHF.

### 1.2 Data pipeline specifics

**Tokenization** — Tokenizer choice affects everything downstream. Common mistakes: tokenizing val/test with a different tokenizer, training a new tokenizer when an existing one suffices, ignoring special tokens (BOS/EOS/PAD/system), encoding/decoding round-trip not idempotent.

**Packing** — For pretraining and many SFT setups, concatenate examples up to context length with EOS separators to avoid padding waste. Set `attention_mask` correctly so cross-document attention is blocked if your loss assumes it. Verify packing didn't corrupt examples by decoding a random batch.

**Masking** — For SFT, mask the loss on the prompt portion (only compute loss on the response). Off-by-one in the prompt/response boundary is a classic silent bug — verify by reading the masked labels in a few batches.

**Data mixture** — Multi-source corpora need explicit weighting. Track per-source loss separately; if one source's loss is flat while others decrease, that source may be near-duplicate of pretraining data or too easy/hard.

**Quality filtering** — Common filters: length, language ID, perplexity from a reference model, n-gram overlap with eval sets (decontamination), dedup (MinHash/LSH).

**Decontamination** — Run n-gram overlap (typical: 13-gram or 8-gram with stricter threshold) between training data and every eval benchmark. Document hit rate per benchmark.

### 1.3 Architecture & objective

- **Architecture** — Default to a well-known recipe (Llama/Mistral/Qwen-style) unless there's a reason to deviate. Custom architectures cost months of debugging.
- **Loss** — Next-token CE for pretraining/SFT; DPO/IPO/KTO for preferences; PPO/GRPO for online RL.
- **Position encoding** — RoPE is the default. If extending context, plan a YaRN/linear scaling step and revalidate on long-context evals.
- **Layer norm** — RMSNorm with pre-norm is the default.

### 1.4 Optimization

| Hyperparameter | Pretraining default | SFT default |
|----------------|---------------------|-------------|
| Optimizer | AdamW (β1=0.9, β2=0.95, eps=1e-8) | AdamW (β1=0.9, β2=0.999) |
| Peak LR | 1e-4 to 3e-4 (model-size dependent) | 1e-5 to 5e-5 (full FT), 1e-4 to 5e-4 (LoRA) |
| Schedule | Warmup (1-5% of steps) → cosine to 10% | Warmup (3-10%) → linear or constant |
| Weight decay | 0.1 | 0.0 to 0.01 |
| Grad clipping | 1.0 | 1.0 |
| Batch size (tokens) | 1M–4M tokens/step | 32K–1M tokens/step |
| Sequence length | 2K–32K (pack to fill) | 2K–8K typically |
| Precision | bf16 (fp32 master for some optimizers) | bf16 |

For LoRA/QLoRA, rank 8-64, alpha = rank or 2×rank, target attention + MLP projections. Verify which modules are actually being adapted (silent bugs here are common).

### 1.5 Common LLM pipeline failure modes (cross-ref `debugging-playbook.md`)

- Loss diverges early in pretraining → LR warmup too short, init mismatch, mixed-precision overflow
- SFT loss starts very low → loss is being computed on prompt tokens (mask bug)
- "Model degraded after SFT" → catastrophic forgetting, mix in a small fraction of pretraining-style data
- Long-context eval fails after fine-tuning on short → forgot to maintain length distribution

---

## 2. Computer vision

### 2.1 Task families

| Task | Key choices | Typical baselines |
|------|-------------|-------------------|
| Classification | Backbone + head, aug strength, label smoothing | ViT, ConvNeXt, ResNet |
| Detection | Anchor-based vs anchor-free, NMS, FPN | DETR variants, YOLO, Faster R-CNN |
| Segmentation | Per-pixel CE vs Dice, decoder design | Mask2Former, SAM, U-Net |
| Pose / keypoints | Heatmap regression vs direct | HRNet, ViTPose |
| Generation | Diffusion (DDPM/Rectified Flow), GAN, AR | SD/Flux/DALL-E |
| Self-supervised | Contrastive, masked image modeling, distillation | DINO, MAE, MoCo, SimCLR |

### 2.2 Data pipeline specifics

**Augmentation** — Strength must match dataset size and task. For small datasets, stronger augmentation (RandAugment, MixUp, CutMix) prevents overfitting. For detection/segmentation, augmentations must transform labels consistently (a flip needs to flip the boxes/masks). Verify with overlay visualization.

**Normalization** — Per-channel mean/std must match what the pretrained backbone expects. Mismatched normalization is a common silent killer of fine-tuning.

**Resolution policy** — Multi-scale training improves robustness for detection/segmentation. For classification, training at lower res then fine-tuning at target res saves compute.

**Class imbalance** — Options: class-balanced sampling, focal loss, loss reweighting. Sampling tends to be more stable; loss reweighting can cause loss spikes.

**Label noise** — Label smoothing for classification. For detection/seg, manual relabeling of the worst-loss samples often pays for itself.

### 2.3 Optimization

- AdamW for transformers (ViT/DETR/Mask2Former) at ~1e-4
- SGD + momentum 0.9 still competitive for CNNs at ~0.01–0.1 with cosine schedule
- EMA of weights almost always helps (especially for detection/segmentation)
- Gradient clipping at 1.0–5.0 for transformer-based vision models

### 2.4 Diffusion specifics

- Schedule: Rectified Flow / EDM-style schedules now dominate over original DDPM
- Loss weighting: v-prediction or rectified flow loss; min-SNR weighting helps
- Resolution: train at lower res with a separate upscaler; cascaded training avoids most-of-budget at high res
- Eval: FID + CLIP-Score + human eval. FID alone is misleading; track multiple metrics

---

## 3. Reinforcement learning

### 3.1 Algorithm families

| Family | Best for | Examples | Sample efficiency |
|--------|----------|----------|-------------------|
| Policy gradient (on-policy) | Continuous control, simulation-cheap | PPO, A2C, TRPO | Low |
| Value-based (off-policy) | Discrete action, sample-expensive | DQN, Rainbow, IQN | Medium-High |
| Actor-critic (off-policy) | Continuous, sample-expensive | SAC, TD3, DDPG | High |
| Model-based | Sample-very-expensive | Dreamer, MuZero, TD-MPC2 | Very High |
| LLM RL | Preference / reward optimization on language | PPO-RLHF, GRPO, RLOO, DPO (offline) | N/A — different regime |

### 3.2 Pipeline specifics

**Environment** — Vectorized envs (multiple parallel) are essential for throughput. Verify each env instance is properly seeded and reset. Auto-reset on terminal must propagate to advantage estimation.

**Observation / action normalization** — Running mean/std normalization of obs typically helps. For continuous actions, output a tanh-squashed Gaussian or use a Beta distribution; account for the squashing in the log-prob.

**Reward shaping** — Sparse rewards are training-prohibitive; dense rewards risk reward hacking. Always log un-shaped (true) return alongside training reward.

**Replay buffer** — Size, prioritization (PER), and staleness all matter. For SAC/TD3, target a buffer ≥ 100K transitions; for Atari/DQN, 1M.

**Advantage / target computation** — GAE (λ=0.95, γ=0.99) is standard. Bugs in advantage computation are *the* classic silent killer of policy gradient methods.

### 3.3 RL-specific gotchas

- **Reward hacking** — Policy finds an unintended way to maximize the reward. Mitigation: penalty for distributional drift (KL to reference), curated eval that measures the *real* goal, careful reward design.
- **Non-stationarity** — Policy improves → distribution shifts → critic stale. Critic LR should usually be lower than actor LR for off-policy methods.
- **Determinism failures** — Async envs are non-deterministic by default. Seed every env, every layer of the stack, and document what's *not* deterministic.
- **Eval contamination** — RL "eval" often runs against the *same* environment as training. Use held-out tasks, different seeds, or counterfactual scenarios for true generalization measurement.

### 3.4 LLM RL specifics (RLHF / GRPO / DPO)

- **DPO** — Offline, no reward model needed; simpler but data-hungry. Verify the reference model is correctly frozen.
- **PPO-RLHF** — Reward model + policy + value + reference. KL penalty to reference is essential; tune β carefully. Common bug: KL computed wrong direction.
- **GRPO** — No value network; uses group-relative advantages. Group size matters; small groups have high variance.
- **Reward model** — Critical that the reward model wasn't trained on the same prompts as policy. Calibrate RM accuracy on a held-out preference set; <70% RM accuracy makes downstream RL unstable.

---

## 4. Tabular learning

### 4.1 Method choice

| Data size | Default method | Why |
|-----------|----------------|-----|
| < 10K rows | Logistic regression / random forest | Avoid overfitting |
| 10K–10M rows | Gradient boosting (XGBoost / LightGBM / CatBoost) | Usually best on tabular |
| > 10M rows OR mixed modalities | Deep tabular (TabNet, FT-Transformer, TabPFN, NODE) | Scales, integrates with embeddings |

Tabular deep learning is competitive but rarely *strictly better* than well-tuned GBM. Default to GBM unless there's a reason.

### 4.2 Pipeline specifics

**Splits** — Random splits are dangerous. Use:
- **Time-based** if the deployment context is "predict future"
- **Group-based** if entities (users, hospitals, etc.) appear multiple times — never split a group across train/test
- **Stratified** for imbalanced classification

**Target leakage** — The most common tabular bug. Check by computing feature → target correlation; suspiciously high values warrant investigation. Common leaks: features computed using future info, features that are functions of the target, features unique to one split.

**Feature engineering** — Often dominates model choice. Categorical encoding (target encoding with proper out-of-fold computation, frequency encoding, ordinal for tree models), datetime decomposition, interaction features, lag/window aggregates for time-series tabular.

**Missing values** — Trees handle natively; deep models need imputation. Track whether missingness itself is informative ("missing indicator" feature).

### 4.3 GBM hyperparameters

| Hyperparameter | Sensible default | Notes |
|----------------|------------------|-------|
| n_estimators | 1000+ with early stopping | Use a validation set |
| learning_rate | 0.05 (small data) – 0.1 (large) | Inverse with n_estimators |
| max_depth | 4–8 | Deeper = more overfit risk |
| min_child_weight | 1–10 | Higher = more regularization |
| subsample / colsample | 0.7–0.9 | Bagging-style regularization |
| reg_alpha / reg_lambda | 0–1 | L1/L2 |

Use Bayesian optimization (Optuna) over grid search; usually finds better configs in <100 trials.

### 4.4 Evaluation

- For probabilistic outputs, always report calibration (Brier, log loss) alongside discrimination (AUC, AP).
- For business-critical use, evaluate at the operating point that matches deployment (precision at fixed recall, or vice versa).
- Stratified bootstrap for confidence intervals.

---

## 5. Speech & audio

### 5.1 Task families

| Task | Typical architecture | Loss |
|------|----------------------|------|
| ASR | Conformer / Whisper / wav2vec2 | CTC, attention-CE, RNN-T |
| TTS | Diffusion (e.g., F5-TTS), AR (e.g., Bark), VITS-style | Per-task: spectrogram MSE, flow matching, AR CE |
| Speaker ID | ECAPA-TDNN, x-vector | AAM-Softmax / arcface |
| Audio classification | AST, beats, PaSST | CE |
| Source separation | Conv-TasNet, SepFormer | SI-SNR |

### 5.2 Pipeline specifics

- **Feature extraction** — Mel-spectrogram (80 bins is standard for ASR/TTS) vs raw waveform (wav2vec2-style). Frame rate 25ms / 10ms hop is the convention.
- **Augmentation** — SpecAugment (time/frequency masking) for ASR; noise, RIRs, speed perturbation for robustness.
- **Sequence length** — Audio is long; chunking strategy + overlap matters. For Whisper-style training, 30-second windows are standard.
- **VAD** — For data preprocessing, voice activity detection trims silence; bad VAD removes content. Verify.
- **CTC vs attention** — CTC is fast, monotonic, but assumes conditional independence. Attention models are more accurate but hallucinate. RNN-T balances.

### 5.3 Eval

- WER for ASR — but case, punctuation, normalization rules matter and are often inconsistent across reports.
- For TTS: MOS (human eval is essential), CER on TTS output re-transcribed by ASR, speaker similarity for voice cloning.
- For Thai/multilingual specifically: word vs character vs syllable error rate; Thai has no word delimiters so WER computation requires tokenization choices.

---

## 6. Multimodal

### 6.1 Architectures

| Pattern | Examples | Use when |
|---------|----------|----------|
| **Dual encoder + contrastive** | CLIP, ALIGN, SigLIP | Retrieval, zero-shot classification |
| **Cross-attention fusion** | Flamingo, BLIP-2, LLaVA | Captioning, VQA, instruction following |
| **Native multimodal transformer** | Gemini, GPT-4V, Chameleon, Qwen-VL | End-to-end multimodal generation |
| **Adapter / projection** | LLaVA, MiniGPT-4 | Cheap to train, leverages frozen LLM |

### 6.2 Pipeline specifics

**Pairing quality** — Image-text pairs vary from noisy (web alt-text) to clean (human captions). Filter with CLIP score, perplexity, or learned filter models. Quality usually beats quantity past a threshold.

**Tokenization of non-text** — For image: patchify (ViT-style, 14×14 or 16×16), or use a discrete tokenizer (VQ-VAE-style). For audio similarly. Verify token count budget matches expected sequence length.

**Modality balancing** — In multi-task or multi-modal training, sample mixtures must be tuned; one modality can dominate the gradient.

**Vision encoder freezing** — Freezing the vision tower during instruction-tuning is the default and saves a lot of compute. Unfreezing only helps when the vision tower is undertrained for the target domain.

### 6.3 Specific to VLMs

- Image-token packing: each image consumes hundreds to thousands of tokens depending on resolution and patch size. Verify total tokens per sample.
- Long-image-context: dynamic resolution, image-patching strategies (e.g., AnyRes, Native dynamic resolution) — keep an eye on total tokens to avoid OOM.
- Instruction format: must exactly match the inference-time format. Mismatched chat templates between train and eval are a classic silent failure.

---

## 7. Time-series & forecasting

### 7.1 Task families

| Task | Methods |
|------|---------|
| Univariate forecasting | ARIMA, ETS, Prophet, N-BEATS, NHITS, PatchTST, TimesFM |
| Multivariate forecasting | DeepAR, Temporal Fusion Transformer, Informer, TimeMixer |
| Anomaly detection | Isolation forest, Matrix Profile, autoencoder reconstruction |
| Classification | Rocket / MiniRocket, InceptionTime, time-series transformers |

For forecasting, classical methods often match or beat deep learning on small-to-medium data. Deep learning wins clearly on (a) high-dimensional / many related series, (b) very long history, (c) covariates.

### 7.2 Pipeline specifics

**Splits** — Time-based, always. No future leakage. For cross-validation, use expanding-window or rolling-origin.

**Feature engineering** — Lags, rolling stats, calendar features, holiday flags. For exogenous variables, ensure they are *known at prediction time*.

**Normalization** — Per-series normalization (especially for many heterogeneous series). Avoid global stats that mix scales.

**Stationarity** — Deep models tolerate non-stationarity better than classical, but differencing or trend removal still helps.

**Evaluation** — Beware of metric quirks: MAPE is undefined at zero, sMAPE has asymmetry, WAPE doesn't aggregate well across series. For forecasting competitions, weighted quantile loss and MASE are common.

---

## 8. Recommendation systems

### 8.1 Stages

- **Candidate generation** (retrieval): two-tower, ANN over embeddings, heuristics
- **Ranking**: GBM, DLRM, wide-and-deep, transformer-based
- **Re-ranking**: diversity, freshness, business rules

### 8.2 Pipeline specifics

- **Negative sampling** — Random, in-batch, hard negatives. Hard negatives improve learning but require careful mining to avoid false negatives.
- **Position bias** — Click data is biased by position. Inverse propensity weighting or counterfactual logging mitigates.
- **Cold start** — New users/items have no history; content features + meta-learning help.
- **Offline / online gap** — Offline AUC improvements often don't translate to online lift. Always plan for online A/B testing.
- **Feedback loops** — Recommendation system shapes the data it later trains on. Track this explicitly; periodic exploration prevents collapse.

### 8.3 Evaluation

- Offline: NDCG@k, HitRate@k, MAP@k on a held-out time period
- Online: CTR, dwell time, downstream business metric — these are the ones that count

---

## 9. Cross-cutting: distributed training

### 9.1 Parallelism strategies

| Strategy | When to use | Memory savings | Communication cost |
|----------|-------------|----------------|--------------------|
| **Data Parallel (DDP)** | Model fits on 1 device, want more throughput | None | Gradient all-reduce per step |
| **FSDP / ZeRO-3** | Model too big for 1 device | Sharded params/grads/opt across DP group | Higher than DDP |
| **Tensor Parallel** | Within a single node, attention/MLP sharding | Activations stay sharded | Very high (NVLink-essential) |
| **Pipeline Parallel** | Many layers, want to span nodes | Activations distributed | Lower bandwidth needed |
| **Expert Parallel** | MoE models | Expert weights distributed | Communication via all-to-all |
| **Sequence Parallel** | Long context | Activation memory savings | Modest |

Most production setups combine: e.g., FSDP + Tensor Parallel + Sequence Parallel for large LLM training.

### 9.2 Common distributed bugs

- **Loss differs across ranks** — Logging shows different loss per rank; means data sampler isn't sharded (every rank seeing same data) or gradient sync is broken.
- **Last batch hang** — Uneven batch counts across ranks; use a drop_last or padded sampler.
- **Hangs on validation** — Validation often run only on rank 0; other ranks block at next training step. Use proper barriers.
- **Effective batch ≠ what you think** — Effective batch = per-device batch × DP world size × grad accumulation. Double-check.
- **Slow throughput on multi-node** — Network bottleneck; profile with NCCL_DEBUG=INFO, check for NUMA misbinding, check NIC bandwidth.

### 9.3 Sanity checks

- Same loss at step 0 across ranks (with same seed setup)
- Effective batch printed at training start
- Throughput linear (or near-linear) in node count for DDP up to a reasonable scale

---

## 10. Cross-cutting: mixed precision

### 10.1 Format choice

| Format | When | Watch out for |
|--------|------|---------------|
| **fp32** | Reference, small models | Slow, memory-heavy |
| **fp16** | Older GPUs (V100, T4) | Narrow range — need loss scaling |
| **bf16** | A100, H100, TPU | Lower precision; usually no scaling needed |
| **fp8** (E4M3 / E5M2) | H100, B100/B200 | Requires careful scaling; activation outliers tricky |

Default for modern hardware: bf16 with fp32 master weights for the optimizer. fp8 is production-viable for inference and increasingly for training, but requires care with activation scaling (per-tensor / per-channel) and outlier-tolerant initialization.

### 10.2 Common mixed-precision bugs

- NaN losses in fp16 → loss scale too high, or activation overflow; switch to bf16 or lower scale
- Optimizer not making progress in bf16 → optimizer state must be fp32 (master copy)
- Eval differs from training → eval running in different precision than training; document and standardize
- Norm layer numerics → keep norm computation in fp32 even within bf16 model

---

## 11. Cross-cutting: checkpointing & resumption

A pipeline is only as good as its ability to resume from failure on a multi-day job.

**Checkpoint contents:**
- Model weights
- Optimizer state (often as large as model weights)
- LR scheduler state
- Data loader state (which examples have been seen — essential for true resumption)
- RNG state (CPU + each GPU)
- Step count, epoch count, run config

**Cadence:**
- Frequent enough to recover from crashes (every N hours or every K steps)
- Sparse enough to not bottleneck training (async save, or save subset)

**Sharded checkpoints** for FSDP/ZeRO — each rank saves its shard; reload requires matching shard count or a resharding step.

**Test the resumption path** at least once per project. A resume that loads weights but skips data loader state will silently re-train on the same examples.

---

## Final checklist for any training pipeline

Before launching a serious run:

- [ ] Problem Card filled in
- [ ] Data splits validated against deployment distribution
- [ ] Decontamination run against all eval sets
- [ ] Overfit-tiny-batch test passed
- [ ] Loss masking / label alignment verified by inspecting decoded batches
- [ ] Instrumentation (loss, grad norm, weight update ratio, throughput) wired in
- [ ] Run config saved as artifact
- [ ] Code state captured (commit hash + diff)
- [ ] Checkpoint cadence + resumption tested
- [ ] Evaluation protocol pre-registered (don't choose metrics after seeing results)
- [ ] Distributed sanity checks passed (same loss across ranks, expected throughput)
