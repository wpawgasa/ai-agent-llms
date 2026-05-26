# Domain Recipes Reference

Quick-reference starting points: known-good configurations by domain. These are *defaults to deviate from with reason*, not laws. They cover the 80% case; specific projects will tune.

Use these to skip "what should I set the LR to?" type questions and move straight to interesting decisions.

## Table of Contents

1. [LLM full fine-tuning (SFT)](#1-llm-full-fine-tuning-sft)
2. [LLM LoRA / QLoRA fine-tuning](#2-llm-lora--qlora-fine-tuning)
3. [LLM preference tuning (DPO)](#3-llm-preference-tuning-dpo)
4. [LLM pretraining (small to mid scale)](#4-llm-pretraining-small-to-mid-scale)
5. [Vision classification fine-tuning](#5-vision-classification-fine-tuning)
6. [Object detection from scratch / fine-tuning](#6-object-detection-from-scratch--fine-tuning)
7. [Semantic segmentation](#7-semantic-segmentation)
8. [Diffusion model fine-tuning](#8-diffusion-model-fine-tuning)
9. [Self-supervised vision (DINO/MAE-style)](#9-self-supervised-vision-dinomae-style)
10. [ASR fine-tuning](#10-asr-fine-tuning)
11. [TTS fine-tuning](#11-tts-fine-tuning)
12. [Tabular GBM](#12-tabular-gbm)
13. [Tabular deep learning](#13-tabular-deep-learning)
14. [RL: PPO (continuous control)](#14-rl-ppo-continuous-control)
15. [RL: SAC (continuous control, off-policy)](#15-rl-sac-continuous-control-off-policy)
16. [RL: DQN family (discrete action)](#16-rl-dqn-family-discrete-action)
17. [Time-series forecasting (deep)](#17-time-series-forecasting-deep)
18. [Contrastive embedding (CLIP-style)](#18-contrastive-embedding-clip-style)

---

## 1. LLM full fine-tuning (SFT)

**When**: Have enough compute for full fine-tuning; need maximum quality; data is high quality.

**Defaults:**
- Optimizer: AdamW (β1=0.9, β2=0.999, eps=1e-8)
- Peak LR: 1e-5 to 5e-5 (smaller for larger models)
- Schedule: linear warmup (3-10% of steps) → cosine to 10% of peak
- Weight decay: 0.0 to 0.01
- Batch size: 32K to 1M tokens
- Sequence length: 2K-8K (pack to fill)
- Epochs: 1-3 (data-dependent; over 3 often overfits)
- Precision: bf16 with fp32 master
- Grad clipping: 1.0
- Loss masking: only on response tokens, not prompt
- Chat template: must match inference-time format exactly

**Watch for:**
- Catastrophic forgetting → mix in a small fraction of pretraining-style data
- Low SFT loss (<0.3 mean) → overfitting; reduce epochs or LR
- Eval drop on base capabilities → too high LR or too many epochs

---

## 2. LLM LoRA / QLoRA fine-tuning

**When**: Limited compute / memory; many adapters needed; data is moderate quality.

**Defaults:**
- Rank: 8 (small), 16-32 (typical), 64-128 (when quality-sensitive)
- Alpha: equal to rank, or 2× rank
- Target modules: attention QKVO + MLP gate/up/down (the common pattern). Embeddings and norms typically excluded.
- Dropout: 0.0 to 0.05
- Optimizer: AdamW
- Peak LR: 1e-4 to 5e-4 (10x higher than full FT)
- Schedule: warmup 3-10% → cosine
- Epochs: 1-5
- For QLoRA: NF4 quantization, double-quant, paged AdamW, bf16 compute

**Watch for:**
- Wrong target modules → silently trains nothing useful; print trainable params at startup, verify which layers are adapted
- Merged-LoRA at inference behaves differently than LoRA-applied → numerical precision in merge

---

## 3. LLM preference tuning (DPO)

**When**: Have preference pairs (chosen / rejected); SFT model is a good starting point.

**Defaults:**
- β (KL strength): 0.1 (typical); 0.01 (when SFT is already strong); 0.5+ (when drift is a problem)
- Optimizer: AdamW
- LR: 5e-7 to 5e-6 (very low compared to SFT — DPO is sensitive)
- Schedule: warmup → cosine (warmup more important here than for SFT)
- Epochs: 1 (rarely more)
- Reference model: SFT checkpoint, frozen
- Length normalization: usually on (penalizes length-bias in preferences)

**Watch for:**
- Loss decreasing but eval not improving → β too low; policy drifting in wrong direction
- KL to reference very large → policy lost coherence; increase β or stop earlier
- Length bias: chosen > rejected mostly because longer → normalize or use IPO/SimPO

---

## 4. LLM pretraining (small to mid scale)

**When**: Pretraining a model up to ~10B parameters, hundreds of B tokens.

**Defaults:**
- Optimizer: AdamW (β1=0.9, β2=0.95, eps=1e-8)
- Peak LR: 3e-4 (smaller models), 1e-4 (larger)
- Schedule: warmup (1-5% of total steps) → cosine to 10% of peak
- Weight decay: 0.1
- Batch size: 1M-4M tokens
- Sequence length: 2K-8K initial; long-context extension as a separate stage
- Grad clipping: 1.0
- Precision: bf16 with fp32 master
- Initialization: scaled by layer depth (e.g., μP or DeepNorm)
- Position encoding: RoPE
- Normalization: RMSNorm, pre-norm

**Watch for:**
- Loss divergence in first 1-5% → warmup too short, LR too high, init mismatch
- Loss plateau after ~50% → may be at compute-optimal point (check Chinchilla ratio)
- Throughput degradation over time → memory fragmentation, checkpoint frequency

---

## 5. Vision classification fine-tuning

**When**: Fine-tuning ImageNet-pretrained ViT/ConvNeXt/ResNet to a new dataset.

**Defaults:**
- ViT/transformer: AdamW (β1=0.9, β2=0.999), LR 1e-4, weight decay 0.05
- CNN: SGD + momentum 0.9, LR 0.01, weight decay 1e-4
- Schedule: warmup (5 epochs) → cosine
- Augmentation: RandAugment (n=2, m=9), MixUp (α=0.2) for transformers; weaker for CNN fine-tuning
- Label smoothing: 0.1
- Batch size: 256-1024 (gradient accumulate if memory limited)
- Epochs: 50-300 from scratch; 10-30 for fine-tuning
- EMA of weights: 0.9999 decay, applied at eval
- Stochastic depth (transformer): 0.1-0.2

**Watch for:**
- Normalization mismatch with pretrained backbone (ImageNet mean/std)
- Over-strong augmentation on small datasets
- Final layer LR ~10× higher than backbone for fine-tuning

---

## 6. Object detection from scratch / fine-tuning

**When**: Training a detector (DETR, YOLO, RetinaNet) on COCO or custom data.

**Defaults (DETR-style):**
- Optimizer: AdamW, LR 1e-4 (backbone 1e-5)
- Schedule: warmup → step decay or cosine
- Weight decay: 1e-4
- Batch size: 16-32 (memory-heavy)
- Augmentation: multi-scale, random crop, horizontal flip; respect bbox transforms
- Loss: classification + bbox L1 + GIoU
- EMA: 0.9999
- Epochs: 50-150
- Auxiliary losses (DETR): on intermediate decoder layers; weight 1.0 each

**Watch for:**
- Bbox transformations broken (flip didn't flip the labels) → verify by visualization
- Class imbalance (background dominates) → focal loss or hard negative mining
- Slow convergence (DETR-family): use DN-DETR or DINO-style query denoising

---

## 7. Semantic segmentation

**When**: Per-pixel classification on natural images, medical images, etc.

**Defaults (Mask2Former-style):**
- Optimizer: AdamW, LR 1e-4
- Schedule: warmup (1000 steps) → polynomial decay
- Weight decay: 0.05
- Loss: per-pixel CE + Dice (or focal + Dice for imbalanced classes)
- Augmentation: scale jitter, crop, flip; for medical: elastic deformation
- Batch size: 8-32 (memory-heavy)
- Epochs: 100-500 (small datasets need longer)
- Mixed precision: bf16

**Watch for:**
- Class imbalance (background pixels >>> object pixels) → Dice + per-class loss weights
- Resolution mismatch between train and inference → consistent or test multi-scale
- Annotation noise → Dice loss is more forgiving than CE

---

## 8. Diffusion model fine-tuning

**When**: Adapting a pretrained diffusion model (SD, Flux) to a new domain or concept.

**Defaults (LoRA fine-tuning):**
- Rank: 16-64
- LR: 1e-4 to 1e-5 (Flux is more sensitive than SD)
- Optimizer: AdamW (β2=0.999) or 8-bit AdamW
- Batch size: 1-4 (memory-heavy due to U-Net / DiT size)
- Steps: 1000-10000 (varies)
- Resolution: match training resolution of base model
- Schedule: constant or short warmup
- EMA: 0.999 typical
- Loss weighting: min-SNR-γ=5 (helps for low-noise steps) or rectified flow weighting

**Watch for:**
- Concept drift (model "forgets" base capability) → regularization images, lower LR
- Overfitting (small training set) → fewer steps, stronger weight decay on adapter
- Caption / text encoder issues → either freeze text encoder (default) or train with lower LR
- VAE latent space mismatch → use VAE from the base model, don't swap

---

## 9. Self-supervised vision (DINO/MAE-style)

**When**: Pretraining a vision encoder without labels.

**Defaults (DINO v2-style):**
- Optimizer: AdamW, LR 5e-4 (scaled with batch)
- Schedule: warmup (10 epochs) → cosine
- Weight decay: cosine 0.04 → 0.4
- Batch size: 1024-4096 (large batches help contrastive)
- Augmentation: heavy (random resized crop, color jitter, blur, solarization)
- Teacher EMA: 0.996 → 1.0 schedule
- Centering / sharpening: tuned per setup
- Epochs: 100-800

**MAE-style:**
- Mask ratio: 0.75
- Loss: MSE on masked patches only
- Decoder: lightweight
- LR: 1.5e-4 (typical), longer training

**Watch for:**
- Collapse (all embeddings → same) → centering/teacher EMA wrong, or augmentation too weak
- Eval lag (linear probe accuracy stagnates) → augmentation diversity, longer training

---

## 10. ASR fine-tuning

**When**: Adapting Whisper / wav2vec2 / Conformer to a new domain or language.

**Defaults (Whisper fine-tuning):**
- Optimizer: AdamW, LR 1e-5 to 5e-5
- Schedule: warmup (5-10% of steps) → linear decay
- Batch size: 16-32 audio clips (with packing if available)
- Epochs: 1-5 (data-dependent)
- Augmentation: SpecAugment (time/frequency masking), noise injection
- Loss: cross-entropy (label-smoothed 0.1) for AR; CTC for CTC heads
- Sequence length: 30s windows for Whisper
- Decoder LR: same as encoder for full FT; sometimes higher
- Mixed precision: bf16

**Watch for:**
- Language ID drift (Whisper) → constrain language token at inference
- Catastrophic forgetting of other languages → mix in multilingual data, or use LoRA
- WER computation method (case, punctuation, normalization) → use the same as the base model's published WER

---

## 11. TTS fine-tuning

**When**: Adapting a TTS model (XTTS, F5-TTS, VITS) to a new voice or language.

**Defaults (general):**
- Optimizer: AdamW, LR 1e-4 (lower for full FT; higher for adapter)
- Schedule: warmup → cosine or constant
- Batch size: 8-32 (audio length-dependent)
- Augmentation: limited; usually pitch/speed only
- Mel-spectrogram normalization: match base model
- Hours of data: 10 min (voice cloning) to 10 hours (high-quality voice)
- Epochs/steps: highly variable; monitor with MOS proxy (ASR-CER of synthesized output)

**Watch for:**
- Reference encoder issues for voice cloning → check reference audio quality
- Phoneme/grapheme mismatch → for Thai, ensure phonemizer matches base model
- Over-fitting in voice cloning → small data, stop early
- Vocoder mismatch → use base model's vocoder, don't swap

---

## 12. Tabular GBM

**When**: Tabular data with <10M rows; clear target.

**Defaults (XGBoost / LightGBM / CatBoost):**
- n_estimators: 5000 with early stopping (patience 50-100)
- learning_rate: 0.05 (small data) to 0.1 (large data)
- max_depth: 4-8 (lower for noisy data, higher for clean data)
- min_child_weight (XGB) / min_data_in_leaf (LGB): 5-50
- subsample: 0.7-0.9
- colsample_bytree: 0.7-0.9
- reg_alpha: 0
- reg_lambda: 1
- For CatBoost: use cat_features for categorical, avoid manual encoding
- For LightGBM: feature_fraction_bynode also useful

**Hyperparameter tuning:**
- Optuna with 50-200 trials usually finds near-optimal
- Tune learning rate and tree complexity jointly (they trade off)

**Watch for:**
- Categorical encoding done wrong (target leak) → use out-of-fold target encoding or model-native handling
- Time-leakage → time-based split, no future info in features
- Group leakage → group K-fold
- High-cardinality categoricals → frequency encoding, target encoding (OOF), or model-native

---

## 13. Tabular deep learning

**When**: Many rows (>1M), mixed modalities, want representation learning.

**Defaults (FT-Transformer / TabPFN / etc.):**
- Optimizer: AdamW, LR 1e-4 (transformer-style); higher for shallower models
- Schedule: warmup → cosine
- Weight decay: 1e-5 to 1e-3
- Batch size: 256-4096
- Categorical embeddings: dim ≈ min(50, n_categories // 2)
- Numerical features: BatchNorm or LayerNorm before model
- Loss: BCE for binary, CE for multi-class, MSE/MAE for regression
- Epochs: with early stopping on val

**Watch for:**
- Tabular deep learning loses to well-tuned GBM on most tabular tasks; sanity check against GBM
- Embedding-dim selection → too small = info loss; too large = overfitting
- Missing-value handling differs from GBM (which handles natively)

---

## 14. RL: PPO (continuous control)

**When**: On-policy continuous control; simulation is cheap.

**Defaults:**
- Learning rate: 3e-4 (Adam)
- Discount γ: 0.99
- GAE λ: 0.95
- Clip ε: 0.2
- Entropy bonus: 0.0 to 0.01
- Value loss coefficient: 0.5
- N steps per update: 2048
- Minibatch size: 64
- Update epochs per rollout: 10
- N parallel envs: 8-64
- Observation normalization: yes, running mean/std
- Reward normalization: yes (running std)
- Action distribution: tanh-squashed Gaussian, log-std as separate parameter
- Network: 2 hidden layers, 64-256 units, tanh activation (or LayerNorm + ReLU)

**Watch for:**
- Tanh saturation → log-prob becomes huge negative; clip
- Reward scale very large or small → normalize
- KL between policies blowing up → smaller LR, larger clip

---

## 15. RL: SAC (continuous control, off-policy)

**When**: Sample-efficient continuous control; off-policy is acceptable.

**Defaults:**
- Learning rate (actor / critic / alpha): 3e-4
- Discount γ: 0.99
- Soft target update τ: 0.005
- Initial entropy coefficient α: 0.2 (or learn it automatically to match target entropy)
- Target entropy: -dim(action) (default heuristic)
- Replay buffer size: 1M
- Batch size: 256
- Warmup steps (random actions): 10K-100K
- Update-to-data ratio: 1 (standard) to 20 (DroQ / REDQ for very sample-efficient)
- Network: 2 hidden layers, 256 units, ReLU

**Watch for:**
- Critic overestimation → twin critics (in SAC by default) and target nets reduce
- Entropy collapse → α too small or learned-α schedule wrong
- Stale buffer at very high UTD ratios → use Q-network ensembles or layer norm

---

## 16. RL: DQN family (discrete action)

**When**: Discrete action space; Atari-like or grid environments.

**Defaults (Rainbow DQN):**
- Learning rate: 6.25e-5 (Adam)
- Discount γ: 0.99
- Target network update: every 8K steps (hard copy) or τ=0.005 (soft)
- Replay buffer: 1M frames
- Batch size: 32
- Train every: 4 environment steps
- ε-greedy: anneal from 1.0 to 0.01 over first 1M steps
- N-step returns: n=3
- Prioritized replay: α=0.5, β annealing
- Network: dueling head, double-DQN trick, distributional (C51 or QR-DQN)
- For pixel inputs: standard Nature-CNN architecture; for vector: 2-layer MLP

**Watch for:**
- Reward clipping (-1, 0, +1) common for Atari but problematic for tasks with magnitude information
- Frame skip and frame stack assumed for Atari (skip 4, stack 4)
- Distributional support range should match observed return range

---

## 17. Time-series forecasting (deep)

**When**: Many series or long horizon; standard methods (ETS/ARIMA) hit ceiling.

**Defaults (PatchTST / NHITS / TFT family):**
- Optimizer: AdamW, LR 1e-3 to 1e-4
- Schedule: warmup → cosine or constant + decay-on-plateau
- Batch size: 32-256 (series-dependent)
- Sequence length / lookback: 96-720 (depends on seasonality)
- Forecast horizon: per task
- Normalization: per-series (Z-score over recent window) — RevIN-style is standard
- Loss: MSE for point forecasts; quantile loss for quantile forecasts; weighted quantile for probabilistic
- Patch size (PatchTST): 16
- Dropout: 0.1-0.3
- Augmentation: limited (time-series augmentation is fragile)
- Epochs: 50-200 with early stopping

**Watch for:**
- Look-ahead bias in features (using future at training time) → time-based split + careful feature engineering
- Train-test distribution shift (concept drift) → expanding-window CV, recent data weighted more
- Calibration of probabilistic forecasts → reliability diagrams; not just point accuracy

---

## 18. Contrastive embedding (CLIP-style)

**When**: Aligning two modalities (text+image, etc.) or training general embeddings.

**Defaults (CLIP-style):**
- Optimizer: AdamW (β1=0.9, β2=0.98 — different from typical!)
- Peak LR: 5e-4 (image encoder), 5e-5 (text encoder) for cosine
- Schedule: warmup (5-10% of steps) → cosine
- Weight decay: 0.1
- Batch size: 4096-65536 (contrastive loss benefits from huge batches)
- Temperature: learned, initialized to log(1/0.07) ≈ 2.66, clipped to [0, log(100)]
- Augmentation: heavy random crop for images; minimal for text
- Loss: symmetric InfoNCE (image→text and text→image)
- Epochs: 5-32 typically

**Variants:**
- SigLIP: replaces softmax with sigmoid; doesn't need huge batch
- Hard negative mining: improves quality at small batch
- LiT: text encoder frozen (pretrained); only image trained

**Watch for:**
- Temperature collapse → clip the learned temperature
- Batch size matters a lot for InfoNCE; SigLIP if batch limited
- Modality dominance (one tower trained too fast) → per-tower LR
- Eval mismatch (zero-shot classification quirks) → use canonical prompts, ensemble

---

## How to use this reference

These are starting points, calibrated for the typical case. Deviate from them deliberately:

- **Smaller dataset than typical** → lower LR, more regularization, fewer epochs (overfit risk), stronger augmentation
- **Larger dataset than typical** → higher LR (within stability), longer training, weaker augmentation
- **Specialized domain** → consider domain-pretrained starting point; recipes for general data may need adjustment
- **Production deployment constraints** → optimize compute / latency / memory accordingly; the recipe is for quality, deployment may demand trade-offs

When in doubt, **run an LR range test** (linear LR ramp over a few hundred steps, plot loss vs LR) — it's the cheapest way to land a sensible LR for any model on any data.
