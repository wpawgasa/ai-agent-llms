# Is RL viable for Cat A? Research, and a correction to our diagnosis

**2026-08-30.** Two RL attempts on the Cat A tool-calling model produced no
learning. GRPO had zero reward variance (R18). DPO had a vanishing gradient
(R22). This document records what the literature says about both failures, what
the standard fixes are, and which of them we never ran.

The short version: **RL is viable here. We never tried the standard fix for
either failure, and our diagnosis started at the wrong level.**

---

## 1. The correction: both failures are DATA problems

We diagnosed both failures at the level of the objective — "the reward
saturates", "the pairs are too easy" — and responded by changing the algorithm
(GRPO to DPO). The defect is one level below that.

| rung | GRPO | DPO |
|------|------|-----|
| **data** | **63.9% of 27,056 turn rows carry no GT tool call.** `tool_call_f1([], [])` returns 1.0, so those rows collect 0.40 free; C2's 0.9369 state accuracy usually collects the other 0.40. | **29,256 synthetic pairs against 51 mined ones.** The negatives come from a corruption function, not from the policy's own errors. |
| objective | Behaves as specified. Scored directly it returns 1.0 for a gold completion, 0.0 for a bogus one, 0.44 for prose. | Behaves as specified. |
| evaluation | Clean. The C2 baseline was re-scored on the rebuilt 206-row set and verified 206/206 against the stored audit. | Same. |

**The reward function is not the bug. The prompt distribution is.** Most
prompts pose no question the policy can get wrong, so most groups tie and most
pairs separate trivially.

This is why swapping GRPO for DPO did not help. It changed the objective while
the defect sat in the data.

---

## 2. GRPO: zero-variance groups, and the fix we never ran

GRPO computes the advantage inside a group. If every sample in a group receives
the same reward, the advantage is zero and the gradient is zero. Our run logged
`reward 1.0, reward_std 0, frac_reward_zero_std 1` at every step.

This is the most documented failure mode in GRPO. The standard remedy is
**dynamic sampling**, introduced by DAPO: over-sample, drop every prompt whose
pass rate is 0 or 1, and resample until the batch is full of prompts that carry
a gradient.

**We never ran it, and could not have.** Verified against the installed code:

```
$ grep -c "dynamic_sampling\|zero.variance\|resample_until" \
    .venv-train/.../trl/trainer/grpo_trainer.py
0
```

TRL 1.0.0's `GRPOTrainer` does not implement dynamic sampling, and neither does
our `grpo.py` — it has no pass-rate filter of any kind.

**What we used instead was a proxy.** `data.tool_bearing_ratio`
(`grpo.py::_tool_bearing_mix_indices`) filters on the **ground truth**: does
this row contain a tool call? It never asks the question that matters: does the
policy get this row right only sometimes? The measurement shows how weak the
proxy is — the informative-prompt share moved only 19.4% to 38.0%, even at a
pure filter.

The literature prioritises prompts near a **pass rate of 0.5**, where the
advantage is largest.

---

## 3. DPO: vanishing gradient on separable pairs

Our training loss reached 0.003. That is the signature, not an anomaly. When
chosen and rejected are easily separable, the implicit margin is already large,
the logistic saturates, and the gradient vanishes. The model learns to
recognise the corruption function without changing how it generates.

This is documented behaviour of the DPO loss, and it is worse when the two
responses differ only in a short span — which is exactly our case, since
`chosen` and `rejected` differ only in the trailing assistant turn.

---

## 4. Why the standard recipe does not transfer unchanged

C2 is right on roughly **78% of tool-bearing turns** and **94% of no-tool
turns**. The literature calls these **saturated problems**: the model answers
correctly on nearly every rollout, so there is little signal left to extract.

Most published RLHF targets tasks with a wide, genuinely uncertain quality
gradient. Ours is a narrow structured-output task that SFT has largely solved.
The remaining errors are concentrated in one place, measured in R17 on the 71
tool-bearing held-out rows:

- 44 of 71 perfect
- **18 right tool, wrong arguments**
- 9 silent
- **0 wrong tool names**

Tool selection is solved. **Argument fidelity is not.** Our reward spreads a
0.30 weight across whole-turn tool F1 and never isolates arguments.

---

## 5. What others do, and what applies to us

| technique | what it fixes | applies here |
|-----------|---------------|--------------|
| Dynamic sampling / zero-variance filtering (DAPO) | tied groups give no gradient | **directly** — this is our exact failure |
| Pass-rate prompt selection near 0.5 | saturated prompt distribution | **directly** |
| Failure-prefix conditioning | saturated problems with rare errors | plausible, unproven for us |
| RAFT / rejection-sampling fine-tuning | handles all-correct and all-wrong prompts by construction | **strong baseline** — simpler than GRPO and competitive with PPO/GRPO |
| Composite, capability-aware rewards for function calling (format, then selection, then arguments) | reward that does not target the residual failure | **directly** — our errors are argument-level |

---

## 6. Ordered plan

1. **Measure per-prompt pass rate.** Sample K=8 completions on a few hundred
   train prompts. Count prompts with pass rate strictly between 0 and 1.
   This is a data measurement, not an RL change. It needs no trainer edit and
   answers "does a learnable signal exist" with a number.
2. **Fix the prompt distribution**, not the algorithm. Keep prompts in that
   band. Dynamic sampling is the same fix applied online; TRL will not provide
   it, so it must be a filter around the rollout.
3. **Re-target the reward at argument fidelity**, where the residual errors
   are.
4. **Use RAFT as the baseline.** It tolerates zero-variance prompts natively.

Step 1 is the gate. It costs one short GPU run and decides whether steps 2 to 4
are worth starting.

---

## 6b. Does this mean we need to generate more data?

**Not yet. Selection first.** The RL problem was never volume.

| | |
|---|---|
| train turn rows | 27,056 |
| **tool-bearing rows** | **9,771 (36.1%)** |
| no-tool rows | 17,285 (63.9%) |

C2 is wrong on roughly 22% of tool-bearing turns (67 negatives from 333
mostly-tool-bearing prompts, 2026-08-30). Applied to 9,771 rows that is on the
order of **2,000 rows where C2 already errs**, before generating anything. We
fed the trainer all 27,056 rows and 63.9% of them cannot produce a gradient.

Generating before measuring the pass rate would repeat the mistake this
document records: acting at the wrong level. Run step 1 of section 6 first.

### If generation is needed, the lever is complexity, not volume

The corpus is skewed shallow:

| level | share | `chain_depth` | `num_tools` |
|-------|-------|---------------|-------------|
| L1-L3 | **67.3%** | 0-2 | 1-4 |
| L4-L5 | 32.7% | 3-4 | 6-7 |

The residual failure is **argument fidelity** — 18 of 71 right-tool-wrong-args,
0 wrong tool names (R17). Wrong arguments come from carrying values across a
tool chain, and `chain_depth` is that knob. Two-thirds of the corpus has a
chain depth of 2 or less.

So the target is **L4/L5-heavy generation**, not more conversations.

**Validate a generated batch by pass rate, not by volume.** More easy data makes
the RL signal worse. The acceptance test is "does C2 fail on it sometimes",
which is only knowable by generating and then measuring.

### Voice stays out of this experiment

Voice is a different capability axis — chunking, barge-in, `<S>` markers. It
does not make tool arguments harder, which is where the errors are. Three
reasons from R20 not to mix it in:

1. The held-out audit must report text and voice **separately and never
   blended**. The pinned 206-row set behind C2's 0.7595 is text-only; blending
   destroys the only measuring stick tied to that number.
2. Adding rows to an existing modality group reshuffles that group's split
   assignment. A merge needs `split_task_a_sft.py --assert-unmoved` and a
   rebuild of the pinned set.
3. Voice requires the `response_only` loss recipe, because `all_tokens` cannot
   honour the per-message `loss: false` flag on barge-in turns.

**One open, measurable question:** under the tool-call stay convention almost
every self-loop turn is a silent tool-call turn, so a voice corpus may carry a
*higher* tool-bearing turn density than text. That is one CPU pass to check
once a voice corpus exists. It is not a reason to entangle voice with the RL
work now.

---

## 7. Prevention

Log `frac_reward_zero_std` and a pass-rate histogram **before** the first
optimizer step, not after 50. Both failed runs would have been abandoned in
minutes instead of hours. The number was available the whole time.

---

## 8. Sources

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476)
- [From GRPO to DAPO and GSPO: What, Why, and How](https://huggingface.co/blog/NormalUhr/grpo-to-dapo-and-gspo)
- [A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce](https://arxiv.org/pdf/2504.11343)
- [Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning](https://arxiv.org/html/2601.20829)
- [Beyond Variance: Prompt-Efficient RLVR via Rare-Event Amplification and Bidirectional Pairing](https://arxiv.org/html/2602.03452v2)
- [Prompt Replay: Speeding up GRPO with On-Policy Reuse of High-Signal Prompts](https://arxiv.org/html/2603.21177v1)
- [Advancing SLM Tool-Use Capability using Reinforcement Learning](https://arxiv.org/pdf/2509.04518)
- [R2IF: Aligning Reasoning with Decisions via Composite Rewards for Interpretable LLM Function Calling](https://arxiv.org/html/2604.20316)
- [Linear Preference Optimization: Decoupled Gradient Control via Absolute Regularization](https://arxiv.org/html/2508.14947)

Project evidence referenced above: CLAUDE.md R17, R18, R22;
`docs/grpo_reward_resolution_investigation.md`;
`docs/cat_a_dpo_null_result.md`; `docs/cat_a_c2_heldout_result.md`.
