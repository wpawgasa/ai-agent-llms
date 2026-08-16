# GRPO reward resolution — why Cat A GRPO cannot learn, and what to do instead

**Date:** 2026-08-16
**Policy under test:** `checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767` (cell C2, held-out composite 0.7595 — see [`cat_a_c2_heldout_result.md`](cat_a_c2_heldout_result.md))
**Runs:** per-turn 50-step diagnostic; trajectory 10-step probe @ T=0.8; trajectory 10-step probe @ T=1.4
**Commits:** `886bbe5`, `95cb100`, `fec93f1`, `a9685bd`, `65cf697`, `35bff48`

> **STATUS: do not start a 1000-step GRPO run in either mode.** Both sit at
> `frac_reward_zero_std = 1.0` or near it. The optimizer is healthy; the reward cannot separate
> rollouts. §7 argues the fix is preference learning on contrastive pairs, not more reward shaping.

---

## 1. Headline

The GRPO optimizer is **not** the problem. Across the 50-step per-turn diagnostic on the C2 base:

| Gate (from `grpo_cat_a_diagnostic.yaml`) | Kill threshold | Observed |
|---|---|---|
| `grad_norm` | >50 for 3 consecutive steps | **0.008 – 0.392** |
| `kl` | >10 at any step | **0.62 – 1.94** |

Two to three orders of magnitude clear. The `scale_rewards: none` + `dr_grpo` + `max_grad_norm: 0.2`
stabilization from the 2026-05-29 re-audit works; none of `df4dot2d`'s explosion (grad-norm 1126,
KL 40) reappears.

The reward is the problem. Every logged step reported `reward 1.0, reward_std 0,
frac_reward_zero_std 1`. GRPO derives advantages from **within-group** reward variance, so a tied
group produces exactly zero gradient regardless of how long it runs.

---

## 2. The reward function is not broken

The obvious hypothesis — a bug making everything score 1.0 — is wrong. Scored directly on real
rows, `reward_business_logic` discriminates correctly:

| Completion | reward |
|---|---|
| GT-perfect (right transition + right tool call) | **1.0** |
| Bogus state + hallucinated tool | **0.0** |
| Plausible prose, no state annotation | **0.44** |

It is the *prompt distribution* that saturates it.

---

## 3. Why the per-turn reward saturates

Measured on `data/output/grpo/task_a` (27,056 turn rows from 2,558 conversations):

- **63.9%** of turns carry no ground-truth tool call (9,771 of 27,056 are tool-bearing).
- `tool_call_f1([], []) == 1.0`, so those rows collect **0.40 of the reward for free**.
- C2's state accuracy is **0.9369**, so they usually collect the other 0.40 too.

⇒ roughly **59% of prompts score exactly 1.0**, and every generation in the group ties.

Scored on 206 real C2 completions, the per-turn reward takes only **11 distinct values** and lands
on exactly 1.0 for **81.1%** of them. That is the resolution ceiling stated plainly.

---

## 4. The tool-bearing mix helps, but does not fix it

`_tool_bearing_mix_indices` (`fec93f1`) rebalances the training set toward turns where the policy
has headroom. Verified on the real corpus: 36.1% → 50.0% at ratio 0.5, → 70.0% at 0.7.

But scoring C2's real held-out completions, a greedy reward below 1.0 — the proxy for a prompt whose
group *can* vary at all — occurs on **38.0%** of tool-bearing rows and **8.9%** of no-tool rows:

| tool ratio | expected informative prompts (lower bound) |
|---|---|
| natural (0.361) | 19.4% |
| 0.5 | 23.5% |
| 0.7 | 29.3% |
| 1.0 (pure filter) | 38.0% |

Even a pure tool-only slice leaves ~62% of groups tied, because the graded reward also saturates on
tool-bearing rows (mean **0.886** there, against a strict held-out `tool_f1` of **0.636**). The mix
is a partial mitigation; it is not the fix. No default ratio is baked in, and a pure filter carries
the R15 hazard of teaching "always call a tool".

**Methodological note, recorded because it nearly produced a wrong answer:** the first pass measured
*cross-row* reward std (0.149 → 0.173) and read it as the result. That is the wrong statistic — GRPO
cares about variance across the N samples of **one** prompt, not across different prompts. Every
number in this section rests on the within-group proxy instead.

---

## 5. Trajectory mode does not fix it either

`reward_business_logic_trajectory` is the project's own designed answer; its docstring claims
aggregating over T turns "turns the per-turn discrete reward lattice into a near-continuous
distribution, restoring the within-group variance GRPO's advantage needs". The path was dead
(transformers 5.x `BatchEncoding` drift, §6.2) and was unblocked and wired in `a9685bd` / `65cf697`.

Measured, 10 steps, `num_generations=8`:

| step | uniq/8 | reward | `reward_std` |
|---|---|---|---|
| 1 | 1/8 | 0.1115 | 0 |
| **5** | **8/8** | 0.05152 | **0** |
| 9 | 2/8 | 0.1071 | 0 |

**Step 5 is decisive: 8 of 8 genuinely distinct trajectories (lengths 77–113) all scoring exactly
the same reward.** With full diversity available the reward still cannot separate them. That rules
out both easy explanations — the policy is not too deterministic, and the per-turn lattice is not
the cause, because this *is* the aggregated trajectory reward.

**The docstring's claim should be read alongside this result.** Aggregation did not restore variance
on this policy and this corpus.

### 5.1 Exploration is a real lever, but a weak one

Identical config except two sampling knobs (temperature 0.8 → 1.4, top_p 0.95 → 1.0):

| step | uniq/8 (0.8 → 1.4) | `reward_std` @ 0.8 | `reward_std` @ 1.4 |
|---|---|---|---|
| 1 | 1/8 → 1/8 | 0 | 0 |
| 5 | 8/8 → 8/8 | **0** | **0.02283** |
| 9 | 2/8 → 6/8 | **0** | **0.00221** |

`frac_reward_zero_std` moved 1 → 0 at steps 5 and 9. So at 0.8 the rollouts were varying
**lexically** while landing on the same workflow outcome — same state path, same early stall,
different wording — and the reward was correctly reporting one outcome.

But 0.00221–0.02283 is at or below the diagnostic's own "reward under-resolves" threshold of 0.02.
With `scale_rewards: none` that is a typical advantage of ±0.002–0.02 against a reward in [0, 1] —
non-zero, and nowhere near enough to move a 26B policy in 1000 steps. Raising temperature further
buys variance by degrading the policy being improved. It is not a foundation.

---

## 6. Defects found and fixed along the way

### 6.1 Silent per-row reward corruption (trajectory mode)

The loader logged `rows=2558` but `indexed_scripts=2420`. Gold scripts live in a dict keyed by
`prompt_key`, so only the **last** colliding conversation survives and every earlier one replays a
*different* conversation's gold segments, scored against its transitions and tool calls. Real
corpus: **76 colliding keys, 138 rows (5.4%) affected, one key colliding 8 ways.**

Worse than merely wrong: mis-scored rows depress reward, which **inflates** variance — the exact
quantity being measured. Left in, it would have biased the trajectory experiment toward a false
pass. Deduped in `65cf697` so rows and index agree exactly.

### 6.2 transformers 5.x `BatchEncoding` drift — twice

`apply_chat_template(tokenize=True)` returns a `BatchEncoding` on transformers 5.x where 4.x
returned `list[int]`. This broke two call sites in two different ways:

- `sft.py::render_response_only_sample` — **silent**: `list(mapping)` yields the KEYS, so every
  per-turn delta came out empty and samples rendered as 2 tokens with 0 unmasked labels. Total loss
  of training signal, no crash. Fixed in `bac1d98`.
- `trajectory_rollout.py::_derive_turn_end_id` — **loud**: `int(ids[-1])` raising on a
  `tokenizers.Encoding`, which is what had disabled the whole trajectory path.

The second happened *because* the first was fixed in place rather than shared. Normalization now
lives once, in `training/_utils.py::normalize_chat_template_ids`, used by both (`a9685bd`). Full
unit suite went 1391 passed / 5 failed → **1433 passed / 0 failed**.

### 6.3 The R5 guardrail has been silently inactive on every Gemma-4 GRPO run

Every held-out eval failed with `grpo_heldout_eval_failed — num_kv_shared_layers is 0 … 
layer_types[:-0] == []`, and **training continued regardless** — so the reward-hacking auto-stop
was never actually armed while the run looked healthy.

Root cause: `_unwrap_unsloth_gemma4_kv_zero_proxy()` already existed for this, but ran *before*
`FastLanguageModel.from_pretrained`, which re-applies unsloth_zoo's temporary patches and
reinstalls `_Gemma4KVSharedSafeProxy` on top — undoing the unwrap exactly when it starts to matter.
Anything that later builds or validates a config then hits the proxy's deliberate `AttributeError`.

Fixed two ways in `65cf697`: re-arm the unwrap after model load, and pass an explicit
`generation_config` at both `model.generate()` sites so transformers never takes the
`self.config._get_generation_parameters()` branch that re-validates the config.
**Not yet independently confirmed** that the guardrail now runs — verify before relying on it.

### 6.4 Config/geometry drift

- **TRL 1.0.0 removed `max_prompt_length`** from `GRPOConfig`, killing the run at construction after
  the model and both splits had loaded. `_filter_grpo_config_kwargs` now drops unsupported keys and
  **logs each one by name** (`95cb100`) — silence is what made R16 cost months. Checked before
  dropping it: prompt lengths are median 4,408 / p90 6,109 / max 7,738, so the 7,680 bound was
  non-binding (1/400 rows) and TRL 1.0.0 no longer truncates prompts at all.
- **VRAM geometry assumptions expired.** The inherited `generation_batch_size=32` /
  `per_device_train_batch_size=8` pair was annotated "matches prior run's train-side VRAM", but that
  run was on the pre-R16 1024-token lineage. At real prompt lengths, 32 concurrent sequences OOM'd
  *during generation* (8.51 GiB requested, 6.84 free). Corrected to 16/4, then 8/2 for trajectory
  mode. Reducing `generation_batch_size` also halves unique prompts per round, which works against
  reward variance — a real cost of the fix, not a free win.

---

## 7. Recommendation

**Stop trying to extract variance from this reward.** Three independent attempts — richer prompt mix
(§4), trajectory aggregation (§5), higher temperature (§5.1) — each moved the needle a little and
none produced a usable learning signal. The common cause is that the reward is a coarse, quantized
function of a workflow outcome, and a strongly-SFT'd policy nearly always produces the *same*
outcome with different words.

**Use preference learning instead.** The failure modes are already measured per row on the held-out
audit, and they map directly onto contrastive pairs:

- **chosen** — the gold assistant turn (correct `[STATE: X → Y]`, correct tool call)
- **rejected** — the same turn corrupted in a way C2 actually fails:
  - narrates the action but emits no `<tool_call>` (the announce-but-don't-call gap)
  - advances the state where the convention wants a self-loop, or vice versa
  - right tool name, wrong arguments — **18 of 71** tool-bearing rows, the current bottleneck

This needs no reward variance at all: every pair carries a guaranteed margin. DPO/ORPO is also
cheaper per step than GRPO here, because there is no generation in the loop — which matters
given R9 forces HF `generate()` rollouts on Gemma-4 (~31 s/step).

**Two constraints on building it:**

1. **Mine negatives from the model's own errors, not only synthetic corruptions.** Synthetic
   negatives teach the model to discriminate against *your corruption function*; its real errors are
   on-distribution and teach it to discriminate against itself. R15 is the cautionary precedent — a
   structurally uniform edit got learned as an unconditional habit.
2. **Never source negatives from the held-out audit set.** Those 206 conversations are the only
   contamination-free measuring stick for the C2 lineage; training on them destroys it. Exclude by
   `user_turn_fingerprint` (`data/heldout_clean_set.py`), the same key
   `build_heldout_clean_set.py` uses.

**Do not inject edited completions into GRPO groups.** GRPO weights samples by logprobs under the
current policy; hand-edited completions are off-policy and would bias the gradient. If pairs are the
tool, use a preference objective built for them.

---

## 8. Reproducing

```bash
# Per-turn diagnostic (50 steps) — optimizer gate, saturated reward
./scripts/run_phase2_grpo.sh --skip-filter \
    --grpo-config configs/training/grpo_cat_a_c2_diagnostic.yaml \
    --sft-checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767

# Trajectory mode (10 steps) @ T=0.8, then the exploration probe @ T=1.4
./scripts/run_phase2_grpo.sh --skip-filter \
    --grpo-config configs/training/grpo_cat_a_c2_trajectory.yaml \
    --sft-checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767
./scripts/run_phase2_grpo.sh --skip-filter \
    --grpo-config configs/training/grpo_cat_a_c2_trajectory_temp.yaml \
    --sft-checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767
```

Read `reward`, `reward_std`, `frac_reward_zero_std` and
`train/unique_completions_per_group` from the step logs. `uniq == N` with `reward_std == 0` is the
signature of the resolution ceiling: the policy varied and the reward could not tell.

All runs used `PYTORCH_ALLOC_CONF=expandable_segments:True`. Every checkpoint directory is distinct
per config (`output_dir`), because `grpo.py` auto-resumes from the highest checkpoint it finds and a
shared directory silently continues another run's trajectory (R13).
