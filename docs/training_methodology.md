# Training Methodology Notes

Design discussion of three coupled choices in this project's training pipeline:

1. Embedding the workflow script (DSL) in the **system prompt**.
2. Emitting `[STATE: X → Y]` **state annotations** in assistant responses.
3. Running **GRPO after SFT** rather than SFT alone.

These are not orthogonal — the first two are what make GRPO's reward signal cheap, and the third is what justifies the cost of (1) and (2). Read together.

---

## 1. Caveats of Training With Workflow Script in System Prompt + State Annotations in Response

The choice the project has made (see `data/templates/workflow_prompt_template.txt`, Task A samples, and the rewards in `training/rewards/reward_business_logic.py`).

### 1.1 Prompt-as-program generalization gap

The model can memorize the *specific* workflow graphs seen in L1–L5 rather than learning to *read* the workflow schema from the system prompt. Symptoms: high `state_transition_accuracy` on held-out turns of seen workflows, sharp drop on unseen topologies.

**Mitigations**
- Randomize node names per sample.
- Paraphrase the DSL serialization (JSON / YAML / Mermaid variants of the same graph).
- Inject distractor states the conversation must not enter.
- Hold out *entire workflow topologies* — not just unseen turns of seen workflows — in eval. Without topology-level holdout, `state_transition_accuracy` overstates capability.

### 1.2 Loss masking interacts badly with prompt-conditioned tasks

`training.loss_mask: response_only` (introduced in commit `4271fe9`) means the model gets *no* gradient signal on parsing the workflow definition. It learns to *emit* annotated outputs, not to ground them in the system prompt.

**Mitigation**: mix in a small fraction of "describe the workflow" / "list reachable states from X" examples where the workflow text appears in the *response*, so the model has gradient on workflow comprehension itself.

### 1.3 Evaluation circularity

`eval/state_accuracy.py` parses the same `[STATE: X → Y]` tags the model was trained to emit. You're measuring annotation production, not workflow adherence. A model can emit a perfectly formatted, valid-against-the-graph tag and then take an action inconsistent with that state.

**Mitigation**: add a *consistency* check — does the tool call / response semantically match the claimed state? This is also the cleanest defense against R5 reward hacking; the format-compliance and transition-correctness rewards both fire on tag shape alone otherwise.

### 1.4 Autoregressive anchoring of state tags

Once `[STATE: A → B]` is emitted, the rest of the turn is conditioned on it. Wrong tag → wrong response, with high confidence. Ordering matters:
- **State-first**: model commits before "thinking."
- **State-last**: post-hoc rationalization that may not match the action above it.

Pick one consistently and test the other as an ablation.

### 1.5 Scaffolding dependence

If the annotation is *always* present in training, behavior degrades when it's absent (e.g., a downstream consumer strips tags then re-prompts). Conversely, parsers that expect the tags break silently when the model drops them under distribution shift.

**Decision needed**: are the tags an *internal* scratchpad (strip before user) or a *contract* (must always be present and valid)? These require different training regimes. Document the choice and align eval, serving, and reward design with it.

### 1.6 Teacher contamination on state tracking

GPT-4o / Sonnet make their own state-tracking mistakes — off-by-one transitions, "convenient" terminals, missed adversarial branches. Synthetic Task A inherits these as ground truth.

**Mitigation**: run `data/data_validator.py` for *graph-level* consistency before training:
- Every transition exists in `workflow_graph.edges`.
- Declared terminal is reachable.
- No orphan nodes.

Validate beyond schema-level validity.

### 1.7 Token-level fragility

`[STATE: foo → bar]` tokenizes differently across Qwen / Gemma / Mistral. Arrows (`→` vs `->`), spacing, and case all shift gradient distribution.

**Mitigation**: normalize aggressively in `data/chat_template_converter.py`. Verify the tag tokenizes into a stable, short sequence per model — otherwise the format-compliance reward becomes a tokenizer artifact.

### 1.8 Long-context drift / KV cache footprint

Workflow script in every system prompt wastes KV cache — especially relevant to Phase 3's concurrency story. Long shared system prompts inflate the per-session footprint that TurboQuant is supposed to shrink.

**Mitigation**: if the workflow is hoisted into a cached prefix at serving time, ensure eval matches serving topology. Otherwise `max_concurrent_batch_4096ctx` numbers won't transfer.

### 1.9 Prompt-extraction surface

Production workflows in system prompts are exfiltratable by adversarial users. If real customer workflows go into Cat A deployments, include adversarial-probing examples (already 15% in `generate_workflows.py` behavior mix) that explicitly try to dump the system prompt, with refusals as gold.

### Priority for this project

The two highest-impact caveats for the current trajectory:
- **§1.1 topology holdout** — without it, Phase 1 winner selection is over-optimistic.
- **§1.3 evaluation circularity** — without a behavior-vs-annotation consistency check, GRPO rewards in Phase 2 are gameable.

---

## 2. With vs Without Workflow DSL and State Annotations

Two fundamentally different architectures:

| Dimension | With DSL + annotations | Without |
|---|---|---|
| Workflow lives in | System prompt (runtime data) | Model weights (training-time) |
| New workflow at inference | Edit system prompt — zero retraining | Requires fine-tune or many-shot in-context |
| Output trace | Parseable `[STATE: X → Y]` | Opaque — infer state from behavior |
| Reward signal | Direct (regex + graph check) | Outcome-only (task completion, tool F1) |
| Eval | Per-transition accuracy | Trajectory-level only |
| Multi-tenant | One model, N workflows | One model per workflow |
| Context cost | Workflow DSL in every prompt | Zero prompt overhead |
| Failure mode | Wrong annotation, right behavior (or reverse) | Silent drift, no audit trail |
| Training data | Workflow-paired examples | Many trajectories per workflow |

### What "without" buys you

A model that has *internalized* a fixed workflow is usually better at *that* workflow:
- No parsing overhead.
- No scaffolding-dependence.
- Shorter prompts → larger effective context.
- Cleaner outputs.
- No `[STATE: ...]` tokens stealing probability mass from real response tokens.
- Rewards become harder to game — model can't satisfy a regex without producing a coherent end-to-end response.

For a single-purpose deployment (one fixed customer-support flow) it's the right call. This is closer to how production assistants like ChatGPT-style agents are actually trained — workflow knowledge baked into weights, no DSL prompt.

### What "without" costs

- **N-way training cost**: every new workflow becomes a new training run. The 3-category Phase 2 plan turns into N-category as workflows multiply.
- **No A/B on workflow edits** without retraining.
- **No interpretable trace** for debugging production failures.
- **Expensive reward signal**: outcome-only rewards are sparse, high-variance, and require longer rollouts to converge. The 5-component composite reward in `reward_business_logic.py` collapses to ≈R5 (task completion) alone — exactly the signal known to suffer worst from reward hacking and credit assignment in long horizons.

### Middle ground: DSL but no annotations

Model still reads the workflow at runtime (preserves multi-tenancy) but produces clean outputs. State is tracked by a separate parser or a probing head.

- **Loses**: interpretability of the trace.
- **Kills**: annotation-as-crutch (§1.5) and evaluation circularity (§1.3).
- **Worst hybrid**: annotations but no DSL — weight-baked workflow plus a scaffold that pretends it's data-driven. Don't pick this.

### Decision for this project

The "with" choice is correct *given* the research questions:

- Comparing Cat A models on workflow-following ability across L1–L5 complexity requires runtime-swappable workflows.
- GRPO on a single H100 in a 1000-step budget needs a cheap per-step signal.

The "without" approach would force one of:
1. Only training on a single workflow — defeats the benchmark.
2. Outcome-only rewards — GRPO converges poorly in budget.
3. Much larger per-workflow datasets — exceeds data generation budget.

The cost paid for this choice is exactly the §1.1–§1.9 caveats. Tractable with topology-level holdouts (§1.1) and a behavior-vs-annotation consistency reward (§1.3). The "without" path's costs are not tractable in Phase 2's budget.

---

## 3. SFT → GRPO: Why, and When SFT Alone Is Enough

### 3.1 Why GRPO after SFT — four reasons

**Loss ≠ task metric.** SFT minimizes token-level cross-entropy on *one* reference trajectory. The actual objective is `0.3·state + 0.3·tool_f1 + 0.2·chain + 0.1·format + 0.1·completion`. SFT weights every token equally — a wrong state tag and a wrong filler word contribute the same to loss but very differently to reward. GRPO gives credit assignment over reward components.

**Exposure bias.** SFT is teacher-forced: the model conditions on *the teacher's* prior tokens. At inference, it conditions on *its own* prior tokens, which drift off-distribution and compound errors. A near-zero-loss SFT model can still fail multi-turn rollouts because it never trained on its own mistakes. GRPO trains on on-policy rollouts — exactly the distribution served at inference.

**Teacher ceiling.** Perfect SFT = perfect mimicry, including teacher mistakes. Task A teacher (GPT-4o / Sonnet) gets state tracking wrong some fraction of the time. GRPO with a programmatic reward can *surpass* the teacher because the reward checks correctness, not similarity. This is the main reason RL is run even when SFT has "converged."

**Mode shaping.** SFT is forward-KL-ish — covers all teacher modes including bad ones. GRPO with `β=0.04` KL penalty is reverse-KL-ish — concentrates mass on high-reward modes while staying near the SFT policy. SFT can't selectively *suppress* a behavior present in training data.

### 3.2 When SFT alone is enough

Run held-out eval after SFT and look at *reward components individually*, not aggregate loss:

| Post-SFT held-out signal | GRPO verdict |
|---|---|
| All components near ceiling on held-out **topologies** | Skip GRPO. Low headroom, high reward-hacking risk. |
| Loss low, `task_completion` mediocre | GRPO helps — this is exposure bias / teacher noise. |
| Loss low, `format_compliance` high, `state_transition_accuracy` low | GRPO won't fix this without a topology-grounded reward; fix data first. |
| Reward variance across rollouts low | GRPO won't have signal to learn from. |

### 3.3 Cost side — GRPO is not free

- **~10× wall-clock per gradient step** (4 rollouts per prompt + reward eval).
- **Reward hacking risk (R5)** — model learns format-valid annotations that satisfy the regex without following the workflow.
- **KL drift / mode collapse** — too much GRPO and response diversity collapses.
- **Gemma-4 specifically**: on HF-generate fallback (R9), rollouts ~31 s/step → ~8.6 h per 1000-step run. Each run is a real cost.

### 3.4 Decision rule

**GRPO's value ≈ (reward_ceiling − SFT_held_out_metric) discounted by reward-hacking risk.**

When the gap is small, the discount dominates and you should skip it.

**Operational procedure for this project**:

1. After SFT, run `training/pilot_check.py` style held-out eval on a small slice — read per-component scores, not loss curves.
2. If Cat A `state_transition_accuracy` on held-out *topologies* is < ~80% OR `task_completion` < ~65% → run GRPO. Worth the ~8 h.
3. If both are saturating → skip GRPO for that category, document, redirect H100 hours to Phase 3.
4. If only one component is weak → consider targeted data augmentation before GRPO. SFT on better data is usually cheaper than RL.

The "always do GRPO" reflex from the RLHF literature assumes headroom exists. The right answer here depends on the post-SFT numbers.

---

## Cross-References

- Reward design: `.claude/rules/03-training.md`, `src/llm_workflow_agents/training/rewards/`
- Eval circularity context: `.claude/rules/05-eval.md`, `src/llm_workflow_agents/eval/state_accuracy.py`
- Data generation choices: `.claude/rules/02-data-generation.md`, `data/templates/`
- Risk register (R3, R5, R9): `CLAUDE.md`
- Gemma-4 GRPO operational notes: `docs/grpo_diagnosis_gemma4_26b.md`
