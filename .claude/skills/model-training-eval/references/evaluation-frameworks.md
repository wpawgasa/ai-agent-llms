# Evaluation Frameworks Reference

How to design rigorous evaluations across domains. Evaluation is the only way to know if training worked — and bad evaluation is more dangerous than no evaluation, because it produces confident wrong answers.

## Table of Contents

1. [Universal principles](#1-universal-principles)
2. [Splits & contamination control](#2-splits--contamination-control)
3. [Metric selection](#3-metric-selection)
4. [Statistical rigor](#4-statistical-rigor)
5. [LLM evaluation](#5-llm-evaluation)
6. [Computer vision evaluation](#6-computer-vision-evaluation)
7. [Reinforcement learning evaluation](#7-reinforcement-learning-evaluation)
8. [Tabular evaluation](#8-tabular-evaluation)
9. [Speech & audio evaluation](#9-speech--audio-evaluation)
10. [Multimodal evaluation](#10-multimodal-evaluation)
11. [Production evaluation (online)](#11-production-evaluation-online)
12. [Eval design checklist](#12-eval-design-checklist)

---

## 1. Universal principles

### What evaluation is for

Evaluation answers questions, not "is this good?" Common questions:

1. **Is this better than X?** Compare to a baseline. Requires statistical significance.
2. **Is this good enough to deploy?** Compare to a threshold. Requires correct deployment-distribution sampling.
3. **What does this model do well/poorly?** Per-stratum analysis. Requires diverse, labeled subsets.
4. **Does this generalize?** Out-of-distribution evaluation. Requires explicit OOD sets.
5. **Is this safe to deploy?** Adversarial / robustness / fairness evaluation. Requires targeted test sets.

Different questions require different eval designs. A "great eval" answers a specific question well; "great eval" in the abstract is meaningless.

### The eval-pipeline-as-a-product

Treat the eval pipeline with the same rigor as the training pipeline:

- Version-controlled, reproducible
- Tested (assertion: known-good model achieves known-good score)
- Logs intermediate outputs for inspection, not just final score
- Reports confidence intervals and per-stratum breakdowns by default

When eval is implemented as "a Jupyter cell that runs once", it has bugs.

### Pre-registration

Decide the eval design *before* seeing results. Specifically:

- Choose metrics in advance
- Specify the comparison and the success threshold
- Define the population of interest

Otherwise, the "garden of forking paths" problem applies: with enough metrics and enough strata, some will look favorable by chance. Pre-registration prevents this.

---

## 2. Splits & contamination control

### Split types and when to use which

| Split type | Use when |
|------------|----------|
| **Random** | Examples are i.i.d. and the test sample reflects deployment |
| **Stratified** | Class imbalance — want each split to have the same class distribution |
| **Time-based** | Deployment is "predict future"; train on past, test on future |
| **Group-based** | Entities (users, patients, hospitals) appear multiple times — split by entity |
| **Geographic** | Generalization to new locations matters |
| **Source-based** | Multi-source data; held-out sources test cross-source generalization |
| **Adversarial** | Specifically curated to break the model — for robustness measurement |

A split type encodes an assumption about how the model will be used. Choose the type whose assumption matches reality.

### Common contamination types

1. **Direct contamination** — Test examples appear verbatim in training data
2. **Near-duplicate contamination** — Near-identical test examples in training (paraphrases, formatting variants)
3. **Solution contamination** — Test answers appear in training (e.g., math problem solutions in web data)
4. **Distributional contamination** — Test data drawn from a source that overlaps training source
5. **Reverse contamination** — Training contains *evaluations* — e.g., GitHub repos with the eval set checked in
6. **Time-leakage** — Training contains events that "happened after" the cutoff used for test data

### Decontamination procedures

For LLM-scale corpora:

1. Compute n-gram (typical: 13-gram for strict, 8-gram with threshold for permissive) overlap between training documents and each eval example.
2. Remove training documents with overlap above threshold.
3. Document hit rate per eval benchmark — if hit rate is very high before decon, the benchmark may not be reliable even after decon.
4. Some benchmarks publish "contamination indicators" (canary strings) — search for them in training data.

For smaller-scale supervised learning:

1. Hash-based exact match between train and test
2. Embedding-similarity check (cosine > 0.95 = near-duplicate)
3. Group/entity check (no shared user/group across splits)

### The "decontaminated benchmark is still leaked" problem

Even after decon:
- The benchmark may have leaked into base model that was used for pretraining
- The benchmark may have leaked into training data of *other* models that produced synthetic data used in your training
- Eval prompts themselves may have leaked

Treat any popular benchmark with caution. New, private, or recent benchmarks (post-training-cutoff) are more trustworthy.

---

## 3. Metric selection

### The primary metric trap

Picking the wrong primary metric leads to optimizing the wrong thing. Common failure modes:

- **Accuracy on imbalanced data** — a 99%-negative class makes 99% accuracy trivially achievable
- **MAPE on time-series with near-zero values** — undefined or wildly misleading
- **F1 with arbitrary threshold** — moves with threshold; AUC or AP captures the curve
- **BLEU for "good writing"** — measures n-gram overlap, not quality
- **MSE for ranking** — doesn't care about rank, only absolute error
- **Loss on val set as the only signal** — different from any downstream metric

### Choosing a primary metric

The primary metric should:

1. **Be aligned with deployment success.** If the deployment goal is "high precision, can tolerate low recall", report precision-at-fixed-recall. Not F1.
2. **Be hard to game.** Robust to overfitting to the metric itself.
3. **Have a meaningful unit.** "AUC 0.85" is fine; "loss 0.342" is not, alone.
4. **Be reproducibly computable.** Same model + same eval set → same number, regardless of implementation.

### Guardrail metrics

Primary metric tells you "did the main thing work"; guardrails tell you "did anything else break."

Examples:
- LLM SFT: primary = downstream eval score; guardrails = perplexity on held-out, safety eval score, latency
- CV classifier: primary = top-1; guardrails = calibration ECE, inference latency, worst-class accuracy
- RL policy: primary = mean return; guardrails = success rate, safety constraint violations, average action smoothness

Report all guardrails alongside the primary. A primary improvement at the cost of a guardrail violation is usually not actually an improvement.

### Metric implementations differ

A surprising number of "standard" metrics have multiple implementations that produce different numbers:

- BLEU: sentence-BLEU vs corpus-BLEU; different smoothing methods
- ROUGE: max vs average across references; stemming yes/no
- WER: case sensitive vs not; punctuation normalization; text normalization
- mAP: COCO-style (multiple IoU thresholds) vs Pascal-style (single threshold)
- F1: macro vs micro vs weighted

When comparing results across papers/teams, verify implementations match. When in doubt, use the canonical reference implementation (e.g., `sacrebleu` for BLEU, COCOeval for mAP).

---

## 4. Statistical rigor

### Reporting

Always report:

- **Mean** ± **standard deviation** (or confidence interval) across multiple seeds, where stochasticity matters
- **N** (sample size — both number of examples and number of runs)
- **Test for significance** when claiming "X is better than Y"

A single-seed result with no confidence interval is informal evidence, not a finding.

### How many seeds

Rough guidance:

- **Eval is large + deterministic + model is deterministic**: 1 seed suffices
- **Eval has sampling variance (e.g., LLM generation)**: 3-5 seeds
- **RL or other high-variance training**: 5-10 seeds minimum; serious papers do 30+
- **Comparing close models**: more seeds; need to distinguish from noise

### How many test examples

A useful frame: how small a difference do you need to detect?

For binary classification accuracy:
- To detect a 1% difference at p<0.05 with 80% power, you need ~6000 examples
- To detect a 5% difference, ~250 examples
- To detect 10%, ~80 examples

For continuous metrics (regression MSE, NLL), variance of the metric estimator depends on data distribution, but order-of-magnitude similar reasoning applies.

If your test set has 100 examples and you're comparing models that differ by 1%, you cannot detect that difference reliably.

### Significance tests

| Test | When |
|------|------|
| **Paired bootstrap** | General-purpose; same examples, two models, sample-with-replacement and recompute metric N times |
| **Paired t-test** | Continuous-metric, sample-level, normally-distributed-ish |
| **McNemar's test** | Binary classification, paired predictions |
| **Wilcoxon signed-rank** | Non-parametric, paired, when normality assumption is iffy |
| **Bootstrap difference confidence interval** | Almost always usable; gives interpretable interval |

For paired comparisons (same examples evaluated by two models), use *paired* tests. Unpaired tests waste statistical power.

### Multiple comparisons

If you test 20 hypotheses at p<0.05, one will look significant by chance. Correct with Bonferroni, Benjamini-Hochberg, or pre-specify a single primary comparison.

### Variance reduction

For RL specifically: use **paired evaluation** (same seeds for both methods being compared) and **same set of evaluation episodes** (e.g., same set of held-out tasks). This reduces variance by factors of 5-20x compared to independent eval.

---

## 5. LLM evaluation

### Categories of LLM eval

| Category | Method | Reliability |
|----------|--------|-------------|
| **Knowledge / reasoning benchmarks** (MMLU, GPQA, etc.) | Multiple-choice scoring | Easy to contaminate; check carefully |
| **Code benchmarks** (HumanEval, MBPP, LiveCodeBench, SWE-Bench) | Execution-based | More robust; still has contamination risk |
| **Math benchmarks** (GSM8K, MATH, AIME) | Final-answer match or step-by-step | Contamination is rampant; LiveAIME-style is fresher |
| **Instruction-following** (IFEval, MT-Bench) | Rule-based or LLM judge | Rule-based reliable; judges biased |
| **Long-context** (RULER, LongBench) | Per-task | Watch for length distribution mismatch |
| **Open-ended generation** | Human eval / LLM judge / reference-based | Human eval gold standard; expensive |
| **Safety / red-team** | Adversarial probes | Specialized; specific to model deployment |
| **Capability-specific** (function calling, multilingual, etc.) | Custom benchmarks | Often the most informative for deployment |

### Multiple-choice scoring methods

1. **Letter-output match** — Model outputs "A"/"B"/"C"/"D"; compare to gold
2. **Logprob comparison** — Compute logprob of each choice continuation; pick highest
3. **Per-token logprob with length normalization** — Above, normalized by length

These can give different numbers. Logprob-based is more sample-efficient and tests "did the model know it" more cleanly than generation-based, which also tests format-following.

For instruction-tuned models, format-following matters too; both should be reported.

### Open-ended generation evaluation

Hardest case. Options:

1. **Reference-based** (BLEU, ROUGE, BERTScore) — Bad for open-ended generation; many valid outputs.
2. **Human evaluation** — Gold standard; expensive; requires careful instruction design and inter-rater reliability checks.
3. **LLM-as-judge** — Cheap, scalable; biased toward verbose responses, biased toward the judge's own model family.
4. **Pairwise comparison** — A vs B, judge picks winner. Often more reliable than absolute scoring.
5. **Rule-based** (IFEval-style) — When you can decompose "good" into checkable rules.

When using LLM-as-judge:
- Use a strong judge (different family from the model being evaluated, if possible)
- Use pairwise comparison rather than absolute scoring
- Randomize position (A/B order) — judges have position bias
- Validate the judge against human ratings on a sample
- Report inter-judge agreement if using multiple

### LLM eval pitfalls

- **Prompt-template drift** — Eval uses one chat template; model trained with another. Differences in tokenization of `<|im_start|>` vs no special tokens can shift scores 5-10%.
- **Stop sequence misconfiguration** — Generation cut off mid-answer; scored as wrong but model knew it.
- **Tokenizer differences for scoring** — Logprob scoring depends on exact tokenization of the choices.
- **Few-shot example selection** — Some benchmarks specify few-shots; using different ones changes scores. Use canonical few-shots.
- **Decoding settings** — Temperature, top-p, top-k affect generation. Greedy is most reproducible; sampling reveals robustness.
- **Length bias** — Many evals favor longer responses (judges; F1 with brevity penalty); track length distribution.

### Holistic LLM evaluation

Modern LLM evaluation is moving toward:

- Suites of complementary benchmarks (no single number is sufficient)
- Open-ended evaluation with verified judges
- Process-level eval (steps, not just final answer)
- Behavior eval (refusal patterns, safety, calibration of "I don't know")
- Capability + safety + usefulness as a tri-axial evaluation

---

## 6. Computer vision evaluation

### Classification

- Top-1, Top-5 accuracy
- Per-class accuracy and confusion matrix
- Calibration (ECE, reliability)
- Robustness on shifted versions (ImageNet-C, ImageNet-R, ImageNet-Sketch, ImageNet-A)

### Detection / instance segmentation

- mAP at standardized IoU thresholds (COCO uses average over IoU 0.5–0.95 in 0.05 increments)
- AP per class
- AP per size (small/medium/large objects)
- AR (average recall) at fixed proposals
- For real-time: latency at target hardware

Watch out: COCO's "mAP" averages over IoU thresholds; Pascal VOC's "mAP" is at IoU=0.5 only. They are not directly comparable.

### Semantic segmentation

- mIoU (mean Intersection over Union) per class
- Pixel accuracy (less informative for imbalanced classes)
- Boundary IoU for thin structures

### Generation / diffusion

- FID, FID-CLIP (lower better; sample-size dependent — report N)
- CLIP-Score (text-to-image alignment)
- Inception Score (less reliable)
- Human eval — preference rates against a baseline
- Diversity metrics (LPIPS to nearest neighbor)

FID is unreliable below ~10K samples; below ~1K, it's nearly random. Report sample sizes.

### Robustness

- Performance under shift: corruption types (noise, blur, weather), rendition types (sketches, paintings)
- Adversarial robustness if relevant (PGD-attacked accuracy)
- Distribution-shift datasets (WILDS, ImageNet-Sketch)

---

## 7. Reinforcement learning evaluation

### Returns

- **Mean episode return** ± std over many seeds (5-30 depending on variance)
- **Median return** — often more representative when distribution is skewed
- **Success rate** (binary) — for goal-conditioned tasks
- **Best-X-percent** — when the policy has high variance and only "good" episodes matter
- **Worst-X-percent** — when safety / robustness matters

### Sample efficiency

- Returns vs steps / environment interactions
- Returns vs wall time
- Crossover point (when does method X surpass method Y?)

### Generalization

- Held-out environment configurations / tasks
- Same environment, different seeds
- New environment with shared structure

Many "RL results" don't generalize because eval uses the *same* environment as training. Real generalization requires held-out tasks.

### Ablations

For RL, ablations matter more than in supervised learning because the algorithmic moving parts are many. Ablate:

- Discount factor γ
- GAE λ
- Entropy bonus
- Network architecture
- Reward shaping components

Without ablations, attributing a result to a specific change is impossible.

### Sample-efficiency reporting

Always report performance per environment step (or per gradient step) rather than only final performance. Otherwise, "method X achieves better final performance" might mean "method X was trained 10x longer."

---

## 8. Tabular evaluation

### Classification

- AUC (ranking metric, threshold-independent)
- AP / PR-AUC (especially for imbalanced data)
- Calibration (Brier score, ECE)
- Performance at target operating point (precision at fixed recall, e.g.)

### Regression

- MAE (mean absolute error) — robust to outliers
- RMSE — penalizes large errors
- MAPE — careful with near-zero targets
- R² — proportion of variance explained
- Per-quantile error (errors aren't uniform; matters for risk applications)

### Cross-validation

- **K-fold** for general use; K=5 or 10
- **Stratified K-fold** for imbalanced classification
- **Group K-fold** when entities appear multiple times
- **Time-series CV** (rolling-origin or expanding-window) for time-aware tasks

Always cross-validate at the *outer* level for hyperparameter selection; nested CV when honest performance estimates are needed.

### Fairness / subgroup analysis

For deployment with diverse populations:

- Per-subgroup accuracy
- Demographic parity / equalized odds metrics
- Worst-group performance
- Calibration per subgroup

A model with great average performance and poor worst-group performance often is not deployable.

### Feature-importance audit

Especially for high-stakes use:

- SHAP or permutation importance — identify what features the model relies on
- Verify it makes domain sense
- Spurious correlations (model relies on an ID column, a leaky timestamp, etc.) are a red flag

---

## 9. Speech & audio evaluation

### ASR

- **WER** (word error rate) — standard. Specify normalization rules.
- **CER** (character error rate) — for languages without word boundaries (Chinese, Thai, Japanese)
- **Per-domain WER** (read speech, conversational, noisy, accented)
- **Real-time factor** (processing time / audio duration)

Thai-specific: Word boundaries are inferred via tokenization (e.g., PyThaiNLP). Different tokenizers produce different WER. CER or syllable-error-rate is often more reliable.

### TTS

- **MOS** (mean opinion score, 1-5) from human raters — gold standard, expensive
- **MCD** (mel-cepstral distortion) — automatic but limited
- **CER via ASR** (transcribe TTS output, measure CER) — checks intelligibility
- **Speaker similarity** (cosine of speaker embeddings) — for voice cloning
- **Naturalness** vs **similarity** as separate axes

### Speaker verification

- EER (equal error rate)
- minDCF (minimum detection cost function)
- Per-subgroup robustness (accent, age, gender, language)

### Source separation

- SI-SNR / SI-SDR (scale-invariant)
- PESQ (perceptual)
- STOI (intelligibility)
- Per-noise-condition breakdown

---

## 10. Multimodal evaluation

### Vision-language (CLIP-style)

- Zero-shot classification across many datasets
- Retrieval (text → image, image → text) — Recall@K
- Robustness to distribution shift (different image styles)

### VQA / captioning / instruction VLMs

- VQA: exact-match or LLM-judged
- Captioning: CIDEr, SPICE, METEOR, BLEU — all have known biases; LLM-judged or human eval more reliable for open-ended
- Instruction VLMs: standard LLM evals adapted to multimodal inputs (MMMU, MathVista, MM-Vet, RealWorldQA)

### Multimodal pitfalls

- **Modality dominance** — Model can answer from text alone (or image alone); doesn't actually use both modalities
- **Visual hallucination** — Model describes things not in the image
- **Compositional failure** — Identifies objects but fails on relations / counts / colors

Specific benchmarks (POPE for hallucination, BLINK for visual perception, etc.) target these.

---

## 11. Production evaluation (online)

Offline eval is necessary but not sufficient. Production behavior often differs.

### A/B testing

- Randomized assignment at user (or session) level
- Predetermined sample size based on minimum detectable effect
- Predetermined duration (full business cycle if seasonal effects)
- Predefined success criteria

Common pitfalls:
- **Peeking** — checking results early and stopping → false positives
- **Insufficient power** — too small / too short to detect realistic effects
- **Novelty / primacy effects** — users behave differently for a new model than for a familiar one in the short term

### Shadow deployment

Run new model alongside production but don't act on its outputs. Logs predictions for offline comparison. Useful for:

- Estimating disagreement rate vs production
- Detecting catastrophic failures before they reach users
- Building up an eval set drawn from real production traffic

### Online metrics vs offline metrics

Offline metrics often correlate weakly with business metrics:

- LLM perplexity weakly predicts user satisfaction
- Classification accuracy weakly predicts revenue
- Recommendation NDCG weakly predicts user engagement

Always plan for A/B testing on the business metric for high-stakes decisions, even if offline looks great.

### Production monitoring

- Distribution shift detection (input distribution vs training distribution)
- Prediction distribution drift
- Performance on labeled production samples (when labels become available)
- Latency, error rate, throughput

---

## 12. Eval design checklist

Before declaring an evaluation framework "done":

**Splits**
- [ ] Split type matches deployment scenario
- [ ] Contamination audit performed and documented
- [ ] No leakage (forward through time, across groups, across sources)
- [ ] Split sizes large enough to detect target effect size

**Metrics**
- [ ] Primary metric is aligned with deployment success
- [ ] Guardrails are defined and reported alongside primary
- [ ] Metrics implemented from canonical reference; not re-implemented
- [ ] Per-stratum breakdowns defined (per-class, per-domain, etc.)

**Methodology**
- [ ] Multiple seeds where stochasticity matters
- [ ] Sample size justified
- [ ] Comparison method (significance test, CI) specified in advance
- [ ] Multiple-comparison correction applied if many comparisons
- [ ] Variance-reduction techniques (paired eval, same episodes) used for stochastic eval

**Reproducibility**
- [ ] Eval pipeline runs end-to-end deterministically
- [ ] All hyperparameters / settings / seeds logged
- [ ] Known-good model produces known-good score (regression test)
- [ ] Decoding settings (for generation) explicit

**Failure analysis**
- [ ] Top-N worst predictions inspected
- [ ] Failure modes named
- [ ] Worst-stratum performance considered, not just average

**Documentation**
- [ ] Eval card: what's measured, what isn't, known limitations
- [ ] Distribution match with deployment documented
- [ ] How to reproduce: dataset version, model version, command
