# Tool-Emission Gap ("Announce-but-Don't-Call") — Recommendations + Fable Review

**Date:** 2026-07-14
**Base:** `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000` (Gemma-4 26B-A4B-it, HF-generate path per R9)
**Companion to:** [`grpo_heldout_metric_and_gt_audit.md`](grpo_heldout_metric_and_gt_audit.md) (source of the "announce-but-don't-call" finding this doc acts on) and [`grpo_viability_investigation.md`](grpo_viability_investigation.md) (RFT headroom probe this doc reprioritizes).

## TL;DR

Cat A tool-calling is under the 0.75 target (honest per-turn-fair baseline: 0.674) because of one
concentrated, genuine weakness: on the 52/150 held-out rows where a tool call is expected, the
model narrates the correct state transition and intent but under-emits the actual `<tool_call>`
tag 47/52 times ("announce-but-don't-call"; tool-F1 0.087 on these rows vs 0.796 on correct
abstentions). An initial recommendation set proposed an SFT loss-masking fix as the top lever. A
Fable 5 adversarial review found the diagnosis plausible but unproven, surfaced a real confound
(ckpt-1000 predates the GT sanitizer), and reordered the plan. **Net effect: do not retrain SFT
yet** — run cheap forensics and the already-overdue RFT headroom re-probe first.

## 1. The gap, restated

From the held-out audit (`grpo_heldout_metric_and_gt_audit.md` §2): of 52 tool-expected rows, 47
score 0 tool-F1. Of the ~30 that emit no tool call at all, ~15 are "announce-but-don't-call" —
correct state transition, correct intent narration, required arguments present in context, but no
`<tool_call>` tag ever emitted (e.g. gives `ORDER-55221`, transitions to `LOOKUP_ORDER`, chats,
never calls `lookup_order`). This is flagged in that doc as "a real RL/data target, not an
artifact."

## 2. Initial recommendation set (pre-review)

Given in priority order:

1. **A/B `loss_mask: response_only` vs the current `all_tokens`** on Cat A SFT. The frozen config
   that produced ckpt-1000 (`.runs/sft_cat_a/sft_cat_a.yaml`) predates the `loss_mask` flag
   entirely and so defaulted to `all_tokens` — loss computed over every token including the long,
   boilerplate-heavy system prompt, not just assistant tokens. `docs/fine_tuning_recipes.md`
   already implements `response_only` and calls it "expected to be the better recipe... not a
   settled win, treat it as a hypothesis to A/B test" — never actually tried on Cat A.
2. **Upsample/reweight tool-bearing turns** — the Cat A corpus is 61.1% zero-tool rows at the row
   level, a plain class-imbalance problem independent of the masking question.
3. **Re-run the Stage-0 RFT headroom probe** on the now-sanitized GT + per-turn-fair scorer (the
   original `frontier_frac = 0.130` MARGINAL verdict predates both fixes and is explicitly flagged
   in `grpo_heldout_metric_and_gt_audit.md` §6 as not yet re-measured) — placed *after* 1–2 on the
   theory that if SFT changes move the needle, the RFT calculus changes too.
4. **RFT (rejection-sampling fine-tuning)** as the next RL-ish lever if 1–3 don't close the gap,
   holding off on expensive generator-side data regeneration.

## 3. Fable 5 review — findings

Full review run via the `model-routing` skill (Fable, adversarial/skeptical framing). Key points:

- **The loss_mask diagnosis is plausible but unproven.** The model already learned to emit
  `[STATE: X → Y]` correctly — an equally rare, templated span. If `all_tokens` dilution were
  suppressing rare assistant spans, it should suppress both, not just tool calls. Cheaper,
  unruled-out alternatives:
  - **Row-slicing exposure bias**: the corpus is single-turn rows sliced from multi-turn teacher
    conversations. If narration and the tool call sometimes land in *separate* assistant turns in
    the source data, slicing may have produced gold rows that are announce-only by construction —
    i.e. the corpus may be *training* this exact behavior. Check by rendering ~10 tool-expected
    training rows.
  - **Decoding truncation**: 15 "intends but never emits" cases is the classic signature of
    `max_new_tokens` or a stop sequence firing right after narration. Check `finish_reason` on
    those rows — a 10-minute check that could dissolve the whole hypothesis.
  - **Template rendering**: verify the system prompt (carrying the tool schemas) actually folded
    correctly into Gemma's chat template for this checkpoint (Gemma templates reject bare
    `system` roles per `03-training.md`).
  - **A real confound**: ckpt-1000 trained *before* the GT sanitizer existed (which drops 32
    fabricated-required-arg tool calls). Any retrain now changes both the loss recipe **and** the
    training data at once unless explicitly controlled — an uncontrolled retrain can't attribute
    its delta to masking alone.
- **Reordering: promote the headroom re-probe to step zero, not step three.** It is nearly free,
  already flagged as overdue in the project's own docs, and its result changes the value of
  everything downstream in both directions. Gating it behind two GPU retrains was backwards. Run
  it now, in parallel with the forensics above.
- **New probe to add**: pass@k (T≈1.0) specifically on the 52 tool-expected held-out rows — the
  actual RFT-feasibility number (does the model ever emit the tag under sampling on the rows that
  matter?), not measured by any existing probe.
- **Recs 1+2 shouldn't run sequentially.** LoRA reruns on the ~1000-conversation corpus are cheap
  on an H100; combine mask-recipe and upsampling into one factorial run on sanitized data rather
  than two single-variable reruns that waste GPU time for the same information.
- **Missing entirely from the original plan:**
  - **Statistical power**: n=52 tool-expected rows → ~±13pp binomial CI. Any A/B delta under
    ~15pp on this slice is noise. Expand the held-out tool-expected slice before trusting deltas
    from any experiment below.
  - **Overfitting risk**: `response_only` puts fewer supervised tokens per example on an already
    tiny (~1000-conversation) corpus; 3 epochs may now be too many — watch per-epoch val loss.
  - **Single-turn teacher-forced eval ≠ multi-turn free-running deployment**: nobody has measured
    whether announce-but-don't-call compounds across turns in an actual multi-turn rollout.
- **The 0.674 vs 0.75 comparison is not valid as stated.** 0.75 was calibrated against the old
  whole-conversation composite; 0.674 is a different, renormalized per-turn metric. Carrying the
  old target onto the new metric without re-deriving the bar is apples-to-oranges — re-derive it
  before treating 0.674 as "close" or "far."

## 4. Revised action plan (supersedes §2's ordering)

1. **Forensics first (near-zero cost, no GPU training):**
   - Render ~10 tool-expected training rows; check whether narration/tool-call ever get split
     across separate assistant turns by the single-turn slicing.
   - Check `finish_reason`/truncation on the 15 announce-but-don't-call held-out rows.
   - Verify Gemma chat-template system-prompt folding is correct for ckpt-1000.
2. **In parallel: re-run the RFT headroom probe** (`scripts/preflight_entropy_diag.py`) on the
   sanitized GT + per-turn-fair scorer, and add a **pass@k (T≈1.0) probe on the 52 tool-expected
   rows** specifically. This is cheap and was already overdue per
   `grpo_heldout_metric_and_gt_audit.md` §6 — it should not wait on any SFT experiment.
3. **Fix eval power before trusting any delta**: expand the held-out tool-expected slice beyond
   n=52 (re-slice existing conversations; no new generation needed).
4. **If a masking/data experiment is still warranted after 1–3**, run `response_only` +
   tool-turn upsampling as **one combined factorial run on sanitized data** (not two sequential
   single-variable retrains), and re-derive the target bar for the per-turn-fair metric before
   judging pass/fail.
5. **RFT** remains the next-stage lever if 1–4 don't close the gap, gated by the headroom/pass@k
   results from step 2. Hold off on generator-side data regeneration (teacher-model cost, multi-day)
   as the last resort.

**Do not retrain Cat A SFT until step 1's forensics are done** — an uncontrolled retrain right now
would confound the loss-mask change with the GT-sanitizer change and produce an unattributable
result.

## 5. Forensics results (2026-07-14)

Run in an environment with no GPU (`torch.cuda.is_available() == False`) and only the LoRA
adapter for ckpt-1000 on disk (not merged weights) — checks requiring generation are noted as
blocked below.

### 5.1 New finding, not in the original plan: corrupted training rows

Rendering training rows for check 5.2 surfaced message objects with **no `content` key**, e.g.
`{'role': 'assistant to=tool'}`. Quantified across `data/output/sft/task_a_cleaned/splits/`:

| Split | Conversations affected | Malformed messages |
|---|---|---|
| train | 167 / 4414 (3.8%) | 2,143 |
| validation | **65 / 519 (12.5%)** | — |
| test | 20 / 261 (7.7%) | — |

These look like leaked tool-routing/channel syntax from a teacher model's raw completion (e.g.
`assistant to=book_reservation`, `assistant to=tool_call?`, one row contains garbled multilingual
token soup) that a parser failed to turn into a `<tool_call>` block, and instead of being
discarded, was written into the corpus with an empty completion. **62.9% occur immediately after
a `tool`-role message** — exactly the point where the model must decide whether to narrate or
call the next tool (15.4% follow another malformed row — a cascading failure; 11.8% follow
`assistant`; 9.9% follow `user`).

`src/llm_workflow_agents/data/data_validator.py` only checks that `messages[0]` is role `system`
(line 131-132) — it never validates the role enum for the rest of the conversation, so this is
invisible to existing validation and reaches the "cleaned" splits untouched.

Verified how this is actually trained on: Gemma-4's chat template
(`checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000/chat_template.jinja:219`) sets
`role = 'model' if message['role'] == 'assistant' else message['role']` — it does not validate or
reject unknown roles, it renders them literally as `<|turn>assistant to=tool\n` with empty
content. Under `loss_mask: all_tokens` (masks only padding), **the model was trained with full
gradient weight to reproduce these degenerate empty turns**, disproportionately right after tool
results. Separately, `training/grpo.py:324,328`'s per-turn extraction only selects turns literally
equal to `"assistant"` and preceded by `user`/`system` — so these rows can never become a GRPO or
held-out-eval *target* directly, but they still sit in conversation history feeding later turns'
prompts during full-conversation SFT.

**This is a real, independent bug** — worth fixing (filter/repair at the data layer, add a role-enum
check to `data_validator.py`) regardless of how the loss-mask/upsampling questions resolve, and
before any SFT retrain, since it currently contaminates ~4-12% of every split including 12.5% of
the held-out validation set the audit's own baseline was measured against.

**Correction (2026-07-14, prompted by a user question):** `data/output/sft/task_a_cleaned/splits/`
(inspected above) is **not part of the DVC pipeline** — `dvc.yaml`'s `task_a_sft` stage only
declares `data/output/sft/task_a_cleaned` as an output (no `splits/` subdir); the canonical,
DVC-tracked split directory is produced by a separate `task_a_sft_splits` stage into
`data/output/sft/task_a_splits`. The `task_a_cleaned/splits/` directory is untracked by git/DVC
(covered by the `/task_a_cleaned` gitignore pattern), dated 2026-05-13, and roughly half the size
of the current canonical `task_a_splits` (dated 2026-07-07, ~2× the rows). It was the right
directory to inspect for *this specific checkpoint* — `.runs/sft_cat_a/sft_cat_a.yaml`, the frozen
config that actually trained ckpt-1000, points to it — but that itself means **ckpt-1000 was
trained on a non-reproducible, out-of-pipeline data snapshot**, a separate issue from the role
corruption.

Checking further confirms the corruption is **not** an artifact of that stale snapshot — it's in
the current canonical pipeline output too, and worse in the data the held-out audit actually draws
from:

| Dataset | Train | Validation | Test |
|---|---|---|---|
| `task_a_cleaned/splits` (stale) | 167/4414 (3.8%) | 65/519 (12.5%) | 20/261 (7.7%) |
| `task_a_splits` (canonical) | 465/9131 (5.1%) | 61/1074 (5.7%) | 25/538 (4.6%) |
| `data/output/grpo/task_a` (audit's actual source, L3-L5 filtered) | 431/2502 (17.2%) | **65/290 (22.4%)** | — |

Root cause pinned down precisely by reading `scripts/clean_task_a_sft.py`: the "264 role-confused
tool messages" the `dvc.yaml` stage description advertises stripping is a **different** bug —
`_is_role_confused()` only strips messages where `role == "tool"` **and** content starts with
`<tool_call>` text. It has no logic for messages whose `role` field itself is corrupted (e.g.
`"assistant to=tool"`) — those pass straight through untouched, at every stage
(`clean_task_a_sft.py` → `split_task_a_sft.py` → `filter_grpo_data.py`, none of which validate the
role enum). Fixing this means patching `clean_task_a_sft.py`'s `clean_record()` to also drop/strip
any message whose role isn't in `{system, user, assistant, tool}`, then re-running the DVC pipeline
(`task_a_sft` → `task_a_sft_splits` → `task_a_grpo`) to regenerate everything downstream — not
patching a static snapshot file.

**Second correction (2026-07-14, same follow-up):** checked `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/train.log`
directly rather than trusting `.runs/sft_cat_a/sft_cat_a.yaml`'s `data.source` field. The log's
actual dataset-mapping counts are unambiguous:

```
2026-07-02 04:57:53 [info] sft_dataset_render  loss_mask=all_tokens
Map: 100%|██████████| 9131/9131  [train]
Map: 100%|██████████| 1074/1074  [eval]
```

**9,131 train / 1,074 eval exactly matches the canonical `task_a_splits`, not the stale
`task_a_cleaned/splits` (4,414/519) the frozen config claims.** So ckpt-1000 was **not** trained on
the wrong/orphaned snapshot — it trained on the correct, canonical, DVC-tracked data. The actual
bug is that **`.runs/sft_cat_a/sft_cat_a.yaml`'s `data.source` field is itself wrong** — a
provenance/reproducibility bug in whatever mechanism froze that config, independent of the data-
corruption bug in §5.1. Two separate, now-fixed-in-understanding issues:

1. **Provenance bug**: the frozen run config for ckpt-1000 does not accurately record what data was
   used. Needs its own fix (regenerate/correct the frozen snapshot mechanism) before trusting any
   future frozen config as ground truth for "what did this checkpoint train on."
2. **Data-corruption bug (§5.1)** stands, but the correct contamination rate for what ckpt-1000
   *actually* trained on is the canonical `task_a_splits` numbers — **5.1% train / 5.7% validation
   / 4.6% test** — not the 3.8%/12.5%/7.7% originally reported from the stale directory. The held-out
   audit's own data source (`data/output/grpo/task_a`, 17.2%/22.4%) is unaffected by this correction,
   since it was already derived from the canonical `task_a_splits`.

### 5.2 Check 1 — row-slicing exposure bias (as originally scoped)

Scanned 10,322 well-formed narration-only assistant turns (no `<tool_call>`, intent language
present); 3,934 are followed by a separate later assistant turn with a real tool call. Inspecting
examples, this is legitimate multi-turn dialogue (agent asks for a missing required argument, user
supplies it, *then* the tool call happens) — not evidence of a slicing bug. **Inconclusive** for
the audit's specific 52 tool-expected rows: the actual sampled-row artifact
(`runs/preflight/heldout_composite_audit_ckpt1000.json`) is gitignored and not present on disk, so
direct cross-reference wasn't possible in this environment.

### 5.3 Check 2 — finish_reason/truncation on the 15 announce-but-don't-call rows

**Blocked.** The audit artifact isn't on disk and there's no GPU here to regenerate it via
`scripts/heldout_composite_audit.py`. Needs to run on the actual training host.

### 5.4 Check 3 — Gemma-4 chat template system-prompt folding

**Cleared, not a bug.** The template explicitly handles `messages[0]['role'] in ['system',
'developer']` (`chat_template.jinja:179,186`) — no rejection, unlike Gemma-3. Confirmed structurally
via direct rendering. System prompts run up to ~13.9K chars in the corpus — not itself a bug, but
supporting context for the original loss-dilution hypothesis (§2 rec 1).

### 5.5 Updated next steps

1. **Fix the role corruption** (5.1) independent of everything else — it's fast, well-scoped, and
   currently contaminates the held-out set the 0.674 baseline was measured against.
2. Re-run the held-out composite audit *after* the fix, to see whether 0.674 moves at all before
   attributing anything to loss-mask or upsampling.
3. Checks 2 and the headroom/pass@k re-probes (§4 step 2) still require the actual GPU host —
   carry them over as-is.

