# Cat A SFT: where the token budget goes, and why 4096 was too small

**Measured 2026-08-04** against `data/output/sft/task_a_splits/train.jsonl` (the
canonical, DVC-tracked split), with the system message rebuilt exactly as training
rebuilds it. Reproduce with:

```bash
source .venv/bin/activate && python scripts/measure_sft_token_budget.py --limit 250
```

Two findings. The first settles the `loss_mask` recipe question that
`fine_tuning_recipes.md` had left open as "a hypothesis to A/B test". The second is
a sequence-length bug that plausibly affects every Cat A number published so far,
and is the more important of the two.

---

## 1. 79% of the gradient trains tokens the model never emits

Per-role token share, 250 conversations, 1,042,188 tokens:

| role | tokens | share | emitted by the model at inference? |
|---|---:|---:|---|
| **system** | 744,479 | **71.4%** | no — supplied in context |
| assistant | 220,846 | **21.2%** | **yes** |
| user | 51,074 | 4.9% | no — the customer speaks |
| tool | 25,789 | 2.5% | no — the environment returns |

Under `loss_mask: all_tokens`, **78.8% of gradient is spent fitting text the model
is never asked to produce.** `response_only` masks it to `-100`, a **4.72×**
increase in gradient density on assistant tokens at identical compute.

The system prompt dominates because it is the *enriched* contract —
`build_enriched_system_prompt` injects `FORMAT_RULES`, the retry-budget rule and a
regenerated workflow script — at a median of **3,016 tokens per sample**, largely
byte-identical across all 4,716 training examples.

### Why this reframes the null results

`docs/cat_a_state_annotation_convention_review.md` §6.6 records that 3.5× epochs
moved composite to 0.6579 at p=0.513. That experiment tripled compute while holding
the 21% allocation fixed. `response_only` at 1× epochs puts more gradient on
assistant tokens than three `all_tokens` epochs do — it is a different intervention
from "train longer", not a weaker version of it.

### The tool-result span is actively harmful

2.5% of gradient trains the model to *generate* `{"status": "success", ...}`
payloads. That is training toward tool-result hallucination — precisely the failure
mode `.claude/agents/corpus-remediator.md` forbids in authored prose, that
`docs/grpo_tool_emission_gap_review.md` documents, and that the tool-stay
convention work exists to suppress. `response_only` removes that gradient for free.

### It also aligns SFT with GRPO

`grpo.py::_load_grpo_jsonl` splits conversations into prompt/completion rows where
one assistant turn is the completion, so the RL stage already scores assistant
tokens only. Under `all_tokens`, SFT and GRPO optimize different objectives over
the same corpus.

---

## 2. `max_seq_length: 4096` silently truncated 56% of conversations — from the END

Measured conversation lengths:

| | tokens |
|---|---:|
| median | 4,264 |
| p90 | 6,055 |
| max | 9,108 |
| system message alone (median) | 3,016 |

| `max_seq_length` | conversations exceeding |
|---:|---|
| **4096** | **139/250 (56%)** |
| 8192 | 2/250 (1%) |
| 16384 | 0/250 (0%) |

`configs/training/sft_cat_a.yaml:35` sets `max_seq_length: 4096`, and
`checkpoints/sft_cat_a/gemma-4-26B-A4B-it/frozen_sft_config_20260722T051246Z.yaml`
confirms 4096 was used for the run that produced weights.

**The truncation is right-sided.** `training/sft.py:674-675` passes
`truncation=True, max_length=max_seq_length_for_tokenize`, and the tokenizer's
`truncation_side` is `right`. So on 56% of training conversations, the model was
shown the 3,000-token contract and the opening turns, and never saw the ending:

- the terminal `[STATE: X → TERMINAL]` transition
- the final tool calls and their results
- the closing turns

### Why this matters for the diagnosis in flight

The measured Cat A symptoms are low state-transition accuracy, low task-completion
rate, and terminal states not being reached. Those are properties of conversation
*endings* — the exact region that was being cut. This does not explain the
self-loop-emission gap (that is a convention defect, established independently at
§4.2/§5.3 of the convention review), but it is a live confound for any
terminal-state or completion metric.

**Consequence for existing baselines:** ckpt-500, ckpt-1770 and the pre-registered
≥0.75 composite bar were all measured against a partly truncated training set.
Treat them as a floor, not a clean comparison point, when judging the v2 retrain.

### Note the asymmetry between the two render paths

`render_response_only_sample` truncates **left**, deliberately — its docstring says
*"Truncates from the left so the final assistant turn always survives"*
(`sft.py:37-38`). The `all_tokens` path truncates right. So the two recipes were
never differing only in loss masking; they also differed in which end of an
over-long conversation survives. **Any A/B of the two recipes at 4096 would have
confounded the loss mask with the truncation direction.**

Raise `max_seq_length` regardless of the recipe chosen. Left-truncation is a
safety net, not a substitute for a window that fits the data.

---

## 3. Recommendation

**Adopt `response_only` with `max_seq_length: 8192`** for Cat A. 8192 covers 99% of
conversations; 16384 covers all of them but costs VRAM for the 1% tail, and the
existing config comment at `sft_cat_a.yaml:37-43` documents why packing is off.

Run three cells against the fixed v2 corpus, one variable at a time:

| cell | `loss_mask` | `max_seq_length` | isolates |
|---|---|---|---|
| C0 | `all_tokens` | 4096 | the existing baseline, reproduced |
| C1 | `all_tokens` | 8192 | the truncation effect alone |
| C2 | `response_only` | 8192 | the recommended recipe |

**C1 is not optional.** Without it a C2 win cannot be attributed between the loss
mask and the end of truncation, and those imply different follow-on work.

Give each cell an explicit `output_dir` in its config. Per CLAUDE.md R13, that key
is the supported way to keep factorial cells apart; without it C1 overwrites C0.

### Two operational cautions

**`eval_loss` cannot rank the cells.** `sft.py:283-291` already warns on this
(`sft_loss_mask_response_only`): `all_tokens` averages over easy boilerplate,
`response_only` over harder assistant tokens only. Expect C2's absolute `eval_loss`
to look *worse* while task metrics improve. Compare cells only on
`eval/agent_benchmark.py` — `weighted_workflow_score`, state-transition accuracy,
tool-call F1. `metric_for_best_model: eval_loss` remains valid *within* a run.

**Effective batch shrinks in gradient-carrying tokens.** At a fixed sequence
budget, C2 backprops through ~21% as many tokens per step. If gradient noise rises,
raise `effective_batch_size` rather than the learning rate — on a 26B MoE under
QLoRA, LR increases interact badly with the frozen router (`freeze_router: true`).

**Cost of `response_only`:** O(N²) tokenizer calls at dataset-prep time, per-turn
incremental `apply_chat_template`. One-time, minutes at 4,716 examples. Already
documented at `fine_tuning_recipes.md:132`.

---

## Measurement caveat

Counts come from `google/gemma-2-2b-it` — a Gemma-family SentencePiece tokenizer
small enough to run locally, used as a proxy for `gemma-4-26B-A4B-it`. Absolute
token counts will shift somewhat with the real tokenizer.

The **shares** are robust: no plausible tokenizer difference converts a 71%/21%
split into a different recipe decision. The **truncation finding** is likewise
robust — a median 3,016-token system prompt against a 4,096 window leaves ~1,080
tokens for a median 20-turn conversation, and no vocabulary difference closes that
gap. Re-run the script with `--tokenizer` pointed at the real checkpoint before
quoting exact figures in a writeup.
