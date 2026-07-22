# Tool-Emission Gap ("Announce-but-Don't-Call") — Recommendations + Fable Review

**Date:** 2026-07-14 (original) — **diagnosis superseded 2026-07-22, see §11**
**Base:** `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000` (Gemma-4 26B-A4B-it, HF-generate path per R9)
**Companion to:** [`grpo_heldout_metric_and_gt_audit.md`](grpo_heldout_metric_and_gt_audit.md) (source of the "announce-but-don't-call" finding this doc acts on) and [`grpo_viability_investigation.md`](grpo_viability_investigation.md) (RFT headroom probe this doc reprioritizes).

> **STATUS (2026-07-22): the title's diagnosis is retired — read §11 first.** This doc started as
> an investigation into "announce-but-don't-call" tool emission (tool-F1 **0.087**, §1). §§7–11
> (dated 2026-07-21/22) progressively overturned that: on the full regenerated sample tool-F1 is
> actually **0.4623** (5.3× higher than the 0.087 headline), and the **dominant deficit is
> state-transition accuracy** (0.6866 vs a 0.85 target) — specifically destination selection
> (right `from`, wrong `to`), not tool emission. The per-turn-fair pass bar is **0.80** (§10), not
> the 0.75 used throughout §§1–6.
>
> **§11.5's open question is resolved** (2026-07-22): the earlier "state accuracy fell 0.817 →
> 0.687" comparison is not a model regression — it compares two largely disjoint, non-comparable
> eval samples (only 4.1%/8.9% conversation overlap; the "GT expects no transition" exclusion
> bucket that made 0.817 a restricted sub-population figure has since vanished from the corpus).
> See the resolution note appended to §11.5 and `scripts/sanity_check_state_acc_drop.py`. **0.6866
> is the trustworthy current baseline.**
>
> The re-scoped retrain design is
> [`docs/superpowers/plans/2026-07-22-cat-a-state-accuracy-factorial.md`](superpowers/plans/2026-07-22-cat-a-state-accuracy-factorial.md)
> — an SFT-recipe factorial targeting destination-selection accuracy, not tool emission or GRPO
> (RL headroom is dead per §9). That spec also surfaced a new, load-bearing finding, since
> **confirmed via W&B + git/DVC provenance** (2026-07-22): ckpt-1000's actual training run
> (`wandb.ai/wpawgasa/huggingface/runs/uklfswk5`, started 2026-07-02T04:58:27Z, git commit
> `ef837079`) consumed a `data/output/sft/task_a_cleaned` snapshot DVC-pinned at that commit to
> hash `8ef8681808e348f82eac36edbbd6c2ae.dir` (131 files, ~228MB). The **current** `task_a_cleaned`
> is a different DVC hash, `3d6a4d3eed110546eccb46a94c5d68cd.dir` (5 files, ~110MB — 48% the size,
> restructured from many loose files into 5 consolidated merged files). **A DVC hash mismatch is
> not circumstantial — it means the file contents differ.** The corpus ckpt-1000 trained on has
> been replaced (via an intervening consolidation + the R12 fix) and is not recoverable in this
> environment (no GCS credentials to `dvc pull` the old blob). This is a confirmed corpus-version
> discontinuity, not a naming/definition artifact — blocking for the factorial spec's control cell
> (§4.2 there).
>
> **Blocker resolved, decision made (2026-07-22): retrain SFT, starting with the spec's C0 control
> cell.** ckpt-1000's now-gone corpus is confirmed to have carried real corruption (R12's own
> contamination analysis of this same corpus lineage: ~5.1%/5.7%/4.6% train/val/test malformed-role
> messages) — the current corpus's replacement of it was a deliberate fix, not accidental loss.
> The current corpus was further re-verified this session by *actually running* the DVC pipeline
> (`dvc repro -s -f` on `task_a_sft_clean` + `task_a_sft_splits`, not just a hash-commit): output
> byte-identical, 0 rows altered by either cleaning pass. This run also surfaced and fixed two real
> DVC bugs — `task_a_sft_gemma4_26b_a4b` was tracking `task_a_cleaned` instead of the
> `task_a_splits` it actually reads (`sft.py::_load_split`), and `task_a_sft_clean`/
> `task_a_sft_splits` had over-indented `cmd:` continuation lines that YAML folds into literal
> newlines — a shell then treats those as command separators, so these stages likely never
> successfully ran via `dvc repro` before this session, only via manual script runs + `dvc commit`.
> Both fixed; `task_a_grpo` has the same cmd bug, flagged but not fixed (out of scope). The corpus
> is tagged `task_a-corpus-v1-2026-07-22` / `task_a-corpus-v2-2026-07-22` (git tags pinning the
> exact `dvc.lock` state — see "Recovering a tagged corpus version" below). See the factorial
> spec's §4.2/§11 item 1 for the full resolution writeup.
>
> Everything below this banner is preserved as the dated investigation trail. §§1–6's numbers and
> framing (0.087, 0.75 target, 0.674 baseline, "one concentrated weakness") are **historical, not
> current** — read them for the reasoning path, not as present fact.

## TL;DR (original, 2026-07-14 — superseded, see status banner above and §11)

Cat A tool-calling is under the 0.75 target (honest per-turn-fair baseline: 0.674) because of one
concentrated, genuine weakness: on the 52/150 held-out rows where a tool call is expected, the
model narrates the correct state transition and intent but under-emits the actual `<tool_call>`
tag 47/52 times ("announce-but-don't-call"; tool-F1 0.087 on these rows vs 0.796 on correct
abstentions). An initial recommendation set proposed an SFT loss-masking fix as the top lever. A
Fable 5 adversarial review found the diagnosis plausible but unproven, surfaced a real confound
(ckpt-1000 predates the GT sanitizer), and reordered the plan. **Net effect: do not retrain SFT
yet** — run cheap forensics and the already-overdue RFT headroom re-probe first.

## 1. The gap, restated (original framing — superseded, see §11)

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

### 5.6 Fix applied + data regenerated (2026-07-14)

`clean_task_a_sft.py::clean_record` patched (TDD, 4 new tests, 12/12 pass) to drop any message
with a role outside `{system, user, assistant, tool}`. Regenerating surfaced a **third, separate
bug**: the current raw `data/output/sft/task_a` (5 consolidated "merged" files, unrelated to this
investigation — it looks like it was replaced/de-garbled by someone at some point) already carries
**zero** instances of the role corruption, but `task_a_cleaned`/`task_a_splits`/`grpo/task_a` had
never been regenerated from it — confirmed by a `dvc.lock` hash mismatch between what `task_a_sft`
recorded as `task_a_cleaned`'s output vs. what `task_a_sft_splits` recorded as its dependency on
the same path. So the corruption wasn't just unpatched — it was baked into stale downstream
directories that had drifted from an already-fixed upstream.

Regenerated all three stages (`clean_task_a_sft.py` → `split_task_a_sft.py --force` →
`filter_grpo_data.py`) from current raw data. Verified 0% malformed-role conversations in all
three post-regen. **New split sizes: train 4,716 / validation 554 / test 279** (up from
4,414/519/261 — the corpus grew since the old directories were last built; GRPO/held-out
validation is now 290 rows, L3-L5 filtered). Old directories preserved as
`*.pre_r12fix_backup_20260714T104107Z` (not deleted) for rollback/comparison.

**Consequence for the 0.674 baseline:** it no longer applies as a live number. The held-out
validation set it was measured against has been fully replaced — different conversations, a
different corpus scale — not just cleaned in place. **Re-measure `heldout_composite_check.py`
against the new data before treating 0.674 as current.**

**Still outstanding:**
- `dvc.lock` needs `dvc commit`/`dvc push` on a machine with DVC configured (unavailable in this
  session) so the lock file and GCS remote reflect this regeneration.
- Checks 2 and the headroom/pass@k re-probes (§4 step 2) still require the actual GPU host, and
  should now run against **this new data**, not the old sampled artifacts.

## 6. Re-measurement on the regenerated data (2026-07-16, GPU host)

Run on the actual GPU host (H200 143GB; ckpt-1000 is a LoRA adapter loaded via the HF-generate
path per R9 — Gemma-4 is vLLM-incompatible under Unsloth). Both probes below draw from the
**regenerated** `data/output/grpo/task_a` (validation = 290 conversations, the post-r12fix corpus),
not the old sampled artifacts.

### 6.1 Held-out composite check — the §5.6 re-measurement

Ran `scripts/heldout_composite_check.py` (`--split validation --n-prompts 150 --seed 42`) against
the new data. This closes the §5.6 action item ("re-measure before treating 0.674 as current").

| Metric | Value |
|---|---|
| **verdict** | **FAIL** (`mean_composite < 0.75`) |
| mean_composite | **0.7167** |
| median_composite | 1.0000 |
| min / max | 0.0000 / 1.0000 |
| frac_below_target | 0.373 (56/150 rows) |
| n_rows | 150 (of 290-row validation) |
| wall_time_s | 699 |

Artifact: `runs/preflight/heldout_composite_ckpt1000.json` (gitignored).

**Read:** the old **0.674** baseline is superseded — on the regenerated held-out set the SFT-only
greedy composite is **0.7167**, up ~4pp from the stale number but **still short of the 0.75 target**
(gate FAILs). The distribution is strongly bimodal (median 1.0, a hard 0.0 floor, 37% of rows below
target), consistent with the "announce-but-don't-call" tail dragging an otherwise-saturated mean
under the bar. **Caveat carried forward from §3:** 0.75 was calibrated against the old
whole-conversation composite; this is the renormalized per-turn-fair metric, so "0.033 short of 0.75"
is not a clean pass/fail until the bar is re-derived for this metric. The clean conclusion is that
SFT-only did **not** clear the target as-stated, so the reward/GT audit and the headroom/pass@k
re-probes stay on the table rather than shipping SFT-only.

### 6.2 RFT headroom re-probe — MARGINAL (unchanged)

Per §4 step 2, re-ran `scripts/rft_headroom_probe.py` on the regenerated data with the **same config
as the original 2026-07-09 MARGINAL run** (`--split train --n-prompts 500 --n-completions 8
--temperature 0.8 --seed 42`, ~4.8 h wall on the HF path) so `frontier_frac` / `mean_headroom` are
directly comparable. Artifact: `runs/preflight/rft_headroom_ckpt1000_regen.json` (gitignored).

| Metric | Prior (07-09) | New (07-16, regen) | Gate |
|---|---|---|---|
| **verdict** | MARGINAL | **MARGINAL** | — |
| frontier_frac | 0.130 | **0.122** | GO_RFT ≥ 0.15 (miss); NO_GO < 0.10 (clear) |
| mean_headroom | 0.0448 | **0.0414** | GO_RFT ≥ 0.03 (pass) |
| frac_collapsed_groups | 0.716 | **0.764** | GRPO_REVIVAL < 0.50 (fail) |
| median_reward_std | 0.000 | **0.000** | GRPO_REVIVAL ≥ 0.05 (fail) |
| mean_reward_std | 0.0432 | 0.0345 | — |
| rung_histogram | {1:357, 2:118, 3:19, 4:5, 5:1} | {1:382, 2:98, 3:18, 4:2} | — |

**Read:** regenerating the corpus did **not** move the RFT calculus. Verdict is still **MARGINAL** —
`frontier_frac` actually ticked *down* 0.130 → 0.122 (still 3pp under the 0.15 GO bar, still well
clear of the 0.10 NO_GO floor), `mean_headroom` still passes at 0.0414. The GRPO-revival branch stays
firmly dead: 76.4% of prompt-groups collapse all 8 samples to a single reward rung (up from 71.6%),
median `reward_std = 0.000`. The rung histogram is if anything slightly *more* concentrated on rung 1
than before. So the data-corruption fix (§5.6) removed a real contaminant but did not unlock
single-turn RL headroom — best-of-8 still beats greedy on only ~12% of prompts, and that headroom is
too concentrated to clear the RFT GO gate.

**Net standing after §6:** SFT-only greedy composite = 0.7167 (FAIL vs 0.75, §6.1) with no measurable
RL headroom to close the gap (MARGINAL RFT, dead GRPO). This is exactly the fork the gate warns
about — below target *and* no cheap RL lever — so the next move per §4/§5.5 is the **reward/GT audit
on ~50 rows** (announce-but-don't-call tail) and the target-bar re-derivation for the per-turn metric,
not a blind RFT pilot or SFT retrain. The pass@k (T≈1.0) probe on the tool-expected rows remains the
other outstanding GPU-host item.

## 7. §4 step 1 forensics (2026-07-21, no-GPU session) — 2 of 3 checks run, one reframes the diagnosis

Ran the three cheap, no-training checks §4 step 1 called for, against the canonical
post-R12-fix `data/output/grpo/task_a` (290-conv validation), using the actual production
slicer (`_load_grpo_jsonl`, `training/grpo.py:236`) and ckpt-1000's real `chat_template.jinja`
— not reconstructions. 2/3 completed; the third needs the GPU host.

### 7.1 Row-slicing exposure bias — CONFIRMED, reframes the "genuine weakness" claim

Sliced validation into 2,943 single-turn rows; 1,126 are tool-expected (GT carries
`tool_calls`). Classified each target turn's shape:

| Target shape | Count | % of tool-expected |
|---|---|---|
| Bare call (tool_call only, no narration in that turn) | 794 | **70.5%** |
| Fused (narration + tool_call in the same turn) | 332 | 29.5% |

Of the 794 bare-call targets, **100%** are immediately preceded (across a user turn) by an
assistant turn that narrates intent *without* calling. So the teacher corpus itself encodes
narrate-then-call as **two separate assistant turns** the majority of the time — pattern:
`assistant narrates & asks a clarifying/confirming question → user replies → assistant fires
a bare tool call`.

**This confirms the row-slicing hypothesis from §3.** The gold corpus trains *both*
"narrate without calling" (the announce turn — a legitimately zero-tool gold target on its
own row) and "call without narrating" (the bare-call turn) as equally correct, turn-local
behaviors. Combined with the 61.1% zero-tool row imbalance (§2 of the companion doc), the
policy has strong incentive to reproduce the announce-style turn, and on a single-turn
teacher-forced held-out eval it sometimes does so exactly where GT expects the bare call
instead. **A material share of the measured "announce-but-don't-call" gap is therefore a
data-structure + single-turn-eval-granularity artifact, not a pure policy defect** — this is
the exact confound the Fable review (§3) warned would make an uncontrolled retrain
unattributable. It does not mean the weakness is fake; it means the *single-turn* tool-F1
number likely understates true tool-emission competence, and a free-running multi-turn probe
is needed before sizing any fix.

### 7.2 Chat-template system-prompt folding — CLEAN, no bug found

Rendered a real sliced GRPO validation prompt (17-message multi-turn history, enriched
system prompt with workflow script + tool descriptions) through ckpt-1000's actual
`chat_template.jinja` via a minimal sandboxed Jinja environment (mirrors HF's
`apply_chat_template` semantics: `trim_blocks`/`lstrip_blocks`, `raise_exception`,
`strftime_now` globals). Renders cleanly: the system message folds into a single
`<|turn>system … <turn|>` block, no `TemplateError`, no dropped content, and
`add_generation_prompt` opens a clean `<|turn>model` slot for the completion. Rules out
silent schema/system-prompt loss as a contributor. Note: `03-training.md`'s "Gemma rejects
bare `system` role" caveat does not apply to this checkpoint's template — Task A carries tool
info as system-prompt text (not the `tools=` kwarg), and this template explicitly special-cases
`messages[0]['role'] in ['system', 'developer']` (`chat_template.jinja:179-195`).

### 7.3 finish_reason / truncation — BLOCKED, needs the GPU host

Requires either live generation metadata for the 15 announce-but-don't-call held-out rows,
or the artifact `runs/preflight/heldout_composite_audit_ckpt1000.json`. Neither exists in a
no-CUDA session — `runs/preflight/` here holds only a provenance txt, no completions.
**Unresolved**, carries forward as the one remaining §4-step-1 item.

**Revised next step:** given §7.1, don't read the single-turn tool-F1 gap as a clean policy
ceiling. On the next GPU-host session, pair the blocked §7.3 finish_reason check with a
**free-running multi-turn probe**: let the model continue past an announce-style turn and
check whether it fires the bare call on its own next turn once a user reply arrives. If it
does so reliably, the true deployed-eval tool-emission rate is materially higher than 0.087
tool-F1 and the priority shifts from "fix the policy" to "fix the single-turn eval + re-derive
the 0.75 bar" (§6.1's outstanding item) rather than an SFT retrain or RFT pilot.

> **Resolved 2026-07-21 — see §8.** The probe was run and the hypothesis **failed**: free-running
> emission (0.320) does not beat teacher-forced single-turn (0.335), 68% of anchors never fire even
> given a second turn, and the paired difference is noise (McNemar p = 0.743). The "fix the eval,
> not the policy" off-ramp is closed. §7.3's own finish_reason/truncation check is still unrun.


---

## 8. Free-running multi-turn probe (2026-07-21, GPU host) — the artifact hypothesis is FALSIFIED

Ran the probe §7.3 called for, on the H100 host against ckpt-1000 and the canonical
post-R12-fix `data/output/grpo/task_a` validation split (290 conversations, freshly
`dvc pull`ed). New harness: `scripts/free_running_multiturn_probe.py`
(pure functions unit-tested in `tests/unit/test_free_running_multiturn_probe.py`, 28 tests).
Artifact: `runs/preflight/free_running_probe_ckpt1000.json` (gitignored).

**Data-freshness note:** the on-disk `data/output/grpo/task_a` at session start was dated
Jul 9 — i.e. **pre**-R12-regeneration — and an initial anchor census against it produced
1,609 rows / 615 tool-expected, matching nothing in §7.1. After the `dvc pull` the census
reproduces §7.1 exactly (see below). Anyone re-running these probes should verify the split's
mtime before trusting a number.

### 8.0 In plain terms (non-specialist summary)

**The problem.** The model talks about using a tool but never actually uses it — like a waiter who
says "I'll go put your order in!" and then just stands there.

**The hopeful theory.** Maybe the model isn't broken — maybe the *test* was unfair. We only ever
gave it one turn to speak. In the conversations it learned from, the pattern is often two turns:
the waiter says "I'll put that in," the customer says "great, thanks," and *then* the waiter goes.
So maybe the model was waiting for its natural turn and our one-turn test cut it off before it
could act. If true, the model would be mostly fine and we'd only need to fix the test — cheap.

**What we did.** Ran the same 200 situations two ways. **Way A** is the old test: one turn, act
now. **Way B** is the fair version: let the model say its "I'll go do that" line, hand it a real
customer reply, then give it *another* turn to act. Way B was deliberately generous — it counted
as a win if the model used the tool at *any* point across those two turns.

**What happened.** No difference. 33.5% success the old way, 32.0% the generous way — slightly
*worse*, and the small wobble was coin-flip randomness (some cases got better, about the same
number got worse). The clearest detail: of the 136 cases where it never used the tool, **all 136**
still correctly wrote down which step of the process it was on. It isn't confused about what should
happen. It knows. It just asks the customer for the information by hand instead of calling the tool
that fetches it.

**So.** The hopeful theory is dead. The model genuinely has this habit — it's not the test's fault.
We can't fix this cheaply by changing how we measure; it needs retraining.

**One extra thing.** The number this whole investigation was built around — **0.087**, the score
that made this look catastrophic — didn't hold up. Measuring the same thing on current data gives
about **0.26**. Still a real problem, just noticeably less dire than we'd been assuming. The old
number came from a smaller sample on data we've since cleaned up, so stop quoting it until someone
re-measures properly (§8.4).

### 8.1 Design — two paired conditions over the same anchors

An **anchor** is a gold assistant turn that is a valid `_load_grpo_jsonl` row, whose GT carries
tool calls, whose shape is *bare call*, and whose `messages[i-2]` is an assistant turn that
narrates without calling. Both conditions are greedy (`do_sample=False`, `max_new_tokens=512`).

| | Prompt | Measures |
|---|---|---|
| **A — teacher-forced** | gold history through the user turn preceding the gold bare call (the model sees the *gold* announce) | the deployed single-turn metric |
| **B — free-running** | starts at the announce position: model writes **its own** announce turn (T1) → **gold** user reply appended verbatim → model generates again (T2) | does it fire on its own next turn once a user reply lands? |

B differs from A in exactly one respect: who authored the announce turn, and whether the model
gets a second turn to act on it. **B's success criterion is strictly more generous than A's** —
a call anywhere in the two-turn window counts, versus A's single turn at one exact position.

**Anchor census** (reproduces §7.1: 795 bare-call anchors vs the doc's 794, 100% announce-preceded):

| | Count |
|---|---|
| bare-call anchors, total | 795 |
| … whose announce turn is itself a sliceable row (probed population) | 381 |
| … whose announce follows a `tool` response (**excluded**) | 414 |
| conversations contributing ≥1 sliceable anchor | 228 |
| anchors sampled (≤1 per conversation, seed 42) | **200** |

### 8.2 Results — free-running does not recover the call

| Metric | A (teacher-forced) | B (free-running) |
|---|---|---|
| **tool-emission rate** | **0.335** (95% CI 0.273–0.403) | **0.320** (95% CI 0.259–0.388) |
| name-match rate | 0.325 | 0.295 |
| mean strict tool-F1 | 0.258 | 0.175 |
| mean graded tool-F1 | 0.306 | 0.259 |

**delta = −0.015** (material bar: ≥ +0.20) → **VERDICT: POLICY_DEFECT.**

Free-running B outcomes: `fired_at_t1` 11.0% (22) — the model fused the call into its own
announce turn, *earlier* than gold; `fired_at_t2` 21.0% (42); **`never_fired` 68.0% (136)**.

The paired churn is **noise, not a wash of two real effects**: 17 anchors recovered by
free-running, 20 lost, exact two-sided McNemar on the 37 discordant pairs **p = 0.743**. Only
12.8% of A's failures are recovered by B, and B loses about as many. Per level, the delta never
points consistently one way (L3 +0.072, L4 −0.067, L5 −0.087).

### 8.3 The failure is behavioral, and it is specific

Among the 136 `never_fired` anchors, **136/136 (100%) still emit a `[STATE: X → Y]` marker** and
only 4/136 name the expected tool in prose. The model reliably performs the state-annotation half
of the turn and substitutes a **clarifying question to the user** for the tool call. In free-running
T1 the model reproduced the announce style — narration, no call — on **178/200 (89%)** of anchors.

Representative `never_fired` anchor (`L3_059_8`, gold tool `qualify_lead`); T2 transitions correctly,
then asks the user for exactly the fields the tool takes:

> `[STATE: GREETING → QUALIFY_PROSPECT]`
> …เพื่อที่จะแนะนำแพ็กเกจที่เหมาะสมกับคุณที่สุด ขออนุญาตสอบถามข้อมูลเบื้องต้น… ไม่ทราบว่าคุณชื่ออะไร
> และติดต่อจากบริษัทไหนคะ รวมถึงงบประมาณที่วางไว้และกรอบเวลาที่อยากเริ่มใช้งาน…

### 8.4 What this does and does not overturn

**Falsified:** §7.3's hypothesis that the model "fires the call on its own next turn once a user
reply arrives, and the single-turn eval just can't see it." It does not — not even under a
two-turn window and a more generous success criterion. §7.1's row-slicing finding remains factually
correct about the *corpus* (the teacher does split narrate-then-call across two turns), but that
structure does **not** rescue the *policy*: the model has internalized the announce half and
under-emits the call half regardless of turn granularity. The priority does **not** shift to
"fix the single-turn eval" on the strength of this probe.

**Not reproduced, and worth flagging:** the §1 headline **0.087** tool-F1 does not appear on this
slice — condition A, the single-turn metric, scores **0.258** mean strict tool-F1 and a 0.335
emission rate on 200 bare-call anchors. The gap is real but **materially smaller than the headline
number implies**. Candidate explanations, none yet tested: the §1 figure came from n=52 rows on the
*pre*-regeneration corpus and mixed bare-call with fused targets, whereas this is n=200 bare-call-only
anchors on the regenerated corpus. **The 0.087 figure should not be quoted again until re-measured
on current data.**

### 8.5 Caveats

- **Scope:** only the 381 sliceable anchors were probed; the 414 whose announce follows a `tool`
  response were excluded to keep A and B on matching prompt shapes. The result generalizes to
  roughly half the bare-call anchor population.
- **Counterfactual continuation:** B's round-2 user message is the gold reply to the *gold*
  announce. For the 178/200 anchors where the model's T1 was its own announce, that reply is
  approximately-valid rather than exact. Removing this needs a user simulator. Note the caveat
  cuts *toward* B — a better-matched reply could only help B, and B still lost.
- **Greedy only.** No pass@k. A sampled probe could still find the call in the distribution's
  tail; that is the outstanding pass@k item from §4 step 2, not addressed here.
- **§7.3's finish_reason/truncation check remains unrun.** It was not needed to reach this
  verdict — 100% of never-fired completions terminate with coherent prose well inside the
  512-token budget, which is not the signature of truncation — but it is still formally open.

### 8.6 Net standing

§4 step 1's forensics are now complete enough to act on. The gap survives the eval-granularity
confound, so the §7.3 off-ramp ("fix the eval, not the policy") is closed. Standing position is
unchanged from §6's fork and now better-evidenced: SFT-only composite 0.7167 (FAIL vs 0.75),
RFT headroom MARGINAL, GRPO revival dead, and the tool-emission weakness confirmed as policy
behaviour. **Next move per §4 step 4:** the combined factorial run (`response_only` masking +
tool-turn upsampling) on sanitized data, with the target bar re-derived for the per-turn-fair
metric first — plus the still-outstanding pass@k probe, which is now the cheapest remaining way
to learn whether the call exists in the sampling distribution at all before committing to a retrain.

---

## 9. pass@k tool-emission probe (2026-07-21, GPU host) — WEAK_SIGNAL, and it explains §6.2

Ran the pass@k probe §4 step 2 called for (overdue since 2026-07-09) and that §8 left as the
open question: greedy says the model doesn't emit the call, but is the behaviour in the policy's
distribution *at all*? Harness: `scripts/passk_tool_emission_probe.py` (pure functions unit-tested
in `tests/unit/test_passk_tool_emission_probe.py`, 20 tests). Artifact:
`runs/preflight/passk_tool_emission_ckpt1000.json` (gitignored).

Config: 120 anchors × 8 samples, T=1.0, top_p=0.95, seed 42, ~72 min wall. Anchors are drawn by
§8's `_select_anchors` at the same seed, so they are a **strict subset** of §8's 200; the greedy
baseline is read back from §8's artifact (paired, n=120 matched, zero extra GPU cost) rather than
re-measured. Reported with the unbiased Chen et al. (2021) pass@k estimator.

### 9.1 In plain terms

Greedy decoding is the model's "default answer." Sampling at T=1.0 eight times asks: *if it rolls
the dice, does the right tool call ever come up?* That distinguishes "the model knows this but
usually doesn't say it" (fixable cheaply by nudging how we sample, or by RL) from "the model
doesn't know this" (needs retraining on better data). The answer: **for about half the cases it
never comes up, not once in eight tries.** Those cases can't be fixed by rolling the dice
differently.

### 9.2 Results

| k | pass@k emission | pass@k name-match |
|---|---|---|
| 1 | 0.297 | 0.287 |
| 2 | 0.363 | 0.350 |
| 4 | 0.428 | 0.412 |
| 8 | **0.483** | **0.467** |

Paired greedy baseline (n=120): emission 0.317, name-match 0.300.
**recovery = pass@8 − greedy = +0.167** against a +0.20 material bar → **VERDICT: WEAK_SIGNAL.**

The curve is still climbing at k=8 (no saturation), so more samples would keep finding a little
more — but the endpoint is not the interesting part.

### 9.3 The decisive finding — the population is tri-modal, and the majority is a hard zero

Distribution of `c_name_match` (how many of 8 samples hit the correct tool):

| c | anchors | share |
|---|---|---|
| **0** | **64** | **53.3%** |
| 1–7 | 36 | 30.0% |
| **8** | **20** | **16.7%** |

**53.3% of anchors never produce the correct call in any of 8 samples at T=1.0.** (All 64 are also
greedy failures — 0/64 had a greedy name-match, as expected.) A further 16.7% get it every time.
Only the 30% middle band is stochastic.

**This explains §6.2's MARGINAL RFT verdict from an independent measurement.** GRPO/RFT learns from
*within-group* reward variance: a prompt whose k samples all score the same rung produces zero
advantage and zero gradient. Here **70.0%** of prompt-groups would collapse that way (53.3% all-wrong
+ 16.7% all-right). §6.2 measured `frac_collapsed_groups = 0.764` and `median_reward_std = 0.000` on
the composite reward — a different metric, on a different split, landing on the same number. The
two probes corroborate each other: **RL has no signal here because the failing majority fails
deterministically.** An RFT/GRPO pilot cannot reinforce a behaviour the policy never emits.

### 9.4 An anomaly worth chasing: the difficulty gradient is inverted

| Level | n | never name-matched | always name-matched |
|---|---|---|---|
| L3 | 45 | **75.6%** | 11.1% |
| L4 | 42 | 50.0% | 14.3% |
| L5 | 33 | **27.3%** | 27.3% |

The *simplest* complexity level fails most and the hardest fails least — the opposite of the
expected ordering, and consistent with §8's per-level emission (L5 0.526 vs L3 0.253). This is not
explained by anything in this doc. Untested hypotheses: L5 conversations are tool-dense enough to
prime emission, whereas L3's tools (`qualify_lead`, lookups) are the ones most naturally satisfied
by *asking the user* — exactly the substitution §8.3 observed. Worth a targeted look before the
factorial run, since it may mean the fix should be weighted by level rather than applied uniformly.

### 9.5 Caveats

- n=120 anchors (subset of §8's 200); single temperature (T=1.0); k=8. A higher temperature or
  larger k would raise pass@k somewhat — but cannot rescue the 53.3% hard-zero band, which is the
  load-bearing number.
- **Name-match, not full correctness.** pass@8 name-match 0.467 only asks whether the right tool
  was named; mean best-of-8 strict F1 is **0.383**, so argument correctness is a further loss on
  top. The hard-zero population is if anything under-counted.
- Still restricted to §8's sliceable-anchor scope (381 of 795); the 414 tool-tailed anchors are
  unmeasured by both probes.

### 9.6 Net standing — the lever is now determined

§4 step 1 and step 2 are both closed. The picture is consistent across four independent probes:
SFT-only composite 0.7167 (FAIL, §6.1); RFT headroom MARGINAL with 76.4% collapsed groups (§6.2);
the eval-granularity artifact hypothesis falsified (§8); and now the missing behaviour shown to be
**deterministically absent** on the majority of failing anchors (§9), which is *why* the RL signal
was dead all along.

**This is a data/SFT problem, not an RL problem.** Next move is §4 step 4's combined factorial run
(`response_only` masking + tool-turn upsampling) on sanitized data — with two amendments this
section adds: (1) re-derive the target bar first, now doubly warranted given §8.4's finding that the
0.087 headline doesn't reproduce; (2) consider weighting the fix by complexity level per §9.4. The
RFT/GRPO track should be considered closed until a retrained checkpoint shows a non-collapsed
group distribution.

---

## 10. Target-bar re-derivation (2026-07-21) — the honest bar is 0.80, not 0.75

Closes the item §3 raised and §6.1 carried forward: "0.75 was calibrated against the old
whole-conversation composite; this is the renormalized per-turn-fair metric, so '0.033 short of
0.75' is not a clean pass/fail until the bar is re-derived." Harness:
`scripts/rederive_target_bar.py` (pure, CPU-only, no GPU; 21 unit tests in
`tests/unit/test_rederive_target_bar.py`). Artifact: `runs/preflight/target_bar_rederivation.json`.

### 10.1 In plain terms

We've been grading against a passing mark of 0.75 that was written for a **different test**. The
old test scored a whole conversation at once. The current test scores one turn at a time — and it
hands out **free full marks** every time the model correctly does nothing, which is 62% of turns.
Easier test, so the passing mark has to be **higher**, not the same. Re-deriving it properly moves
the mark from 0.75 to **0.80** — which means the model is failing by *more* than we thought, not
less. This was not the answer anyone was hoping for.

### 10.2 Why the two metrics are not the same quantity

In `compute_weighted_workflow_score` (whole-conversation), `tool_call_f1` is a single AST-F1 over
all calls in the conversation, so a turn where the model correctly makes **no** call is *invisible*
— it appears in neither the predicted nor the gold list. In `_heldout_composite_score` (per-turn)
each row is scored on its own and **`compute_ast_f1([], []) == 1.0`**, so every correct abstention
earns full marks on the 0.4-weight tool term.

Row population from the production slicer (`_load_grpo_jsonl`, validation, 2,943 rows):

| | share |
|---|---|
| tool-expected rows | 38.3% |
| **zero-tool rows (free 1.0 on the tool term if the model abstains)** | **61.7%** |
| state term applies | **100.0%** |
| task term applies | 9.8% |

So for ~90% of rows the per-turn metric reduces to **0.5·tool + 0.5·state**, with the tool half
gifted on the zero-tool majority. It is a **materially easier** metric, and an equivalent bar on it
must sit **above** 0.75.

### 10.3 Method

Rather than guess a translation, evaluate what `_heldout_composite_score` scores on the *real* row
population for a reference policy that exactly meets the component targets `.claude/rules/05-eval.md`
already commits to (state ≥ 0.85, tool-F1 ≥ 0.85, task completion ≥ 0.70), replicating the scorer's
term-applicability and renormalization rules exactly. Those component targets imply a
whole-conversation composite of **0.82**, while the stated bar was **0.75** — a deliberate
relaxation of ×0.9146. The same relaxation is carried across.

The per-turn tool term depends on how a policy handles *abstention* rows, which the old component
targets never specified — so the bar is reported across a sensitivity sweep rather than as a single
fake-precise number.

| assumed abstention accuracy | component-equivalent bar | relaxation-matched bar |
|---|---|---|
| 1.00 | 0.8919 | 0.8158 |
| **0.95** | **0.8770** | **0.8021** |
| 0.90 | 0.8620 | 0.7884 |
| 0.85 | 0.8471 | 0.7748 |

### 10.4 Recommended bar: 0.80

Take **0.80** (relaxation-matched at 0.95 abstention, rounded). It sits mid-range across the
sweep — the full plausible band is 0.775–0.816, so 0.80 is not sensitive to the one judgement call
in the derivation.

**Consequence — the gap gets bigger, not smaller:**

| | vs old bar 0.75 | vs re-derived bar 0.80 |
|---|---|---|
| measured (§6.1) | 0.7167 | 0.7167 |
| shortfall | −0.033 | **−0.085** |

The re-derivation **2.6×'s the shortfall**. Every "only 0.033 short" framing in §6.1 and earlier is
superseded. This cuts against the direction everyone was hoping the re-derivation would go, which is
precisely why it needed doing before the factorial run rather than after.

### 10.5 The fix is not single-lever — a new finding

Inverting the derivation gives the tool-F1 a policy needs on tool-expected rows to clear 0.80:

| state acc | abstention | required tool-F1 | |
|---|---|---|---|
| 0.850 | 0.950 | 0.448 | both components at target |
| 0.900 | 0.950 | 0.320 | strong state |
| 0.850 | 0.900 | 0.526 | target state, weaker abstention |
| 0.800 | 0.900 | 0.654 | both slightly under target |
| **0.817** | **0.796** | **0.773** | **ckpt-1000's measured components** |

An internal consistency check confirms the model is below target on the *other* two components, not
just tools: the measured composite 0.7167 is only reachable with a plausible tool-F1 if state and
abstention are around their audited values (0.817 / 0.796 → implied tool-F1 0.337, consistent with
§8's 0.258 on the hardest bare-call slice). Assuming the *targets* (0.85 / 0.95) instead would imply
a tool-F1 of 0.012, far below anything measured.

**So raising tool-F1 alone to ~0.45 would not clear the bar** — with state and abstention where they
are, it would take ~0.77, which is close to the component target itself and far beyond what §9
suggests is reachable. §4 step 4's factorial run should therefore target **state accuracy and
abstention alongside tool emission**, not tool emission alone. That is a materially different plan
from the one §2–§9 have been converging on.

### 10.6 Caveats and the next cheap step

- The relaxation ratio (×0.9146) assumes the original 0.75 was a deliberate softening of the
  component targets rather than an unrelated round number. If it was arbitrary, the
  component-equivalent bar (**0.877**) is the defensible one and the shortfall is larger still.
- The abstention assumption is a judgement call; the ±0.02 band it induces does not change any
  verdict here.
- **ckpt-1000's component values are stale** — 0.817 / 0.796 come from the pre-regeneration n=150
  audit (`grpo_heldout_metric_and_gt_audit.md` §2). §10.5's bottom row, and the "not single-lever"
  conclusion, should be re-confirmed by re-running `scripts/heldout_composite_audit.py` on the
  regenerated corpus (~15 min GPU). That is the cheapest remaining step and it directly sizes the
  factorial run's targets.
- Bars apply to `_heldout_composite_score` on Task A Cat A rows only; Cat B/C keep their own targets.

**Recommended action:** adopt **0.80** as the Cat A per-turn-fair bar, mark 0.75 as retired for this
metric wherever it is quoted, and re-measure the component breakdown before setting the factorial
run's success criteria.

---

## 11. Component re-measurement (2026-07-21, GPU host) — the diagnosis was wrong; state, not tools

Ran `scripts/heldout_composite_audit.py` on the regenerated corpus at **n=290 requested / 284 rows,
106 tool-expected** (greedy, seed 42, 23 min). This closes §10.6's action item *and* §4 step 3's
power item — the tool-expected slice is now n=106, up from the n=52 the Fable review flagged as
±13pp. Artifact: `runs/preflight/heldout_composite_audit_ckpt1000_regen.json`.

### 11.1 In plain terms

We went looking for confirmation and found the opposite. The thing we've spent this entire
investigation blaming — the model not calling tools — is **five times better than the number in
this doc's own summary** (0.46, not 0.087). What's actually broken is something nobody was looking
at: the model gets the **workflow step wrong** about a third of the time. It always writes down a
step transition, and it always gets the *starting* step right — it just picks the wrong
*destination*, either advancing when it should stay put or staying when it should advance. That is
now the single biggest thing standing between this checkpoint and the bar.

### 11.2 Measured components — two large corrections

| component | measured now | previously believed | target |
|---|---|---|---|
| state accuracy (where a transition is expected) | **0.6866** | 0.817 (§2 audit) | 0.85 |
| **tool-F1 (tool-expected rows)** | **0.4623** | **0.087** (§1 headline) | 0.85 |
| abstention (zero-tool rows) | **0.9494** | 0.796 (§2 audit) | — (0.95 assumed in §10) |
| task completion (genuine terminal rows) | **1.0000** | 0.093 (§2, artifact) | 0.70 |

**Correction 1 — the tool gap is 5.3× smaller than this doc has claimed.** §8.4 suspected 0.087
wouldn't reproduce; it doesn't. On the full tool-expected population it is **0.4623**. The §1/TL;DR
framing — "one concentrated, genuine weakness... under-emits the tool tag 47/52 times" — is
**obsolete**, an artifact of the pre-regeneration n=52 slice. 53.8% of tool-expected rows still
score zero tool-F1, so the weakness is real, but it is not *the* story.

**Correction 2 — state accuracy is now the dominant deficit** at 0.6866 against a 0.85 target, and
*lower* than previously believed rather than higher.

Two confirmations worth noting: abstention measured **0.9494** against the **0.95** §10 assumed for
the recommended bar, so **0.80 stands empirically**; and the per-turn-fair composite here is
**0.7271** (n=284) against §6.1's 0.7167 (n=150) — consistent, still **FAIL** vs 0.80 (shortfall
0.073).

**§10's model of the metric is validated:** `expected_perturn_score` predicts **0.7284** at these
measured components against an actual **0.7271** — a 0.0013 error. The derivation machinery in §10
is empirically sound, which is what licenses the lever analysis below.

### 11.3 Lever analysis — this supersedes §10.5

§10.5 concluded "the fix is not single-lever" from the *stale* component values. With measured
values that is **wrong**. Either component, taken to its target, clears the bar on its own:

| scenario | per-turn composite | |
|---|---|---|
| measured now | 0.7284 | fail |
| **state → 0.85**, tool unchanged (0.4623) | **0.8085** | **PASS** |
| **tool → 0.85**, state unchanged (0.6866) | **0.8026** | **PASS** |
| state → 0.80, tool unchanged | 0.7840 | fail |
| both → target | 0.8827 | PASS |

**State is much the cheaper lever: +0.146 to close, versus +0.388 on tools.** Required-to-clear-0.80
is state 0.833 at current tool-F1, or tool-F1 0.837 at current state accuracy.

### 11.4 What the state failure actually is

`state_acc` is strictly binary (89 zeros, 195 ones — no partial credit). Of the 89 failures:

- **0/89 omit the annotation** — the model always emits `[STATE: X → Y]`.
- **87/89 get the `from` state right and only the `to` state wrong.** 2/89 have the reverse.

It fails in both directions — gold says stay and it advances (`SEARCH_OPTIONS → SEARCH_OPTIONS`
predicted as `→ CHECK_VISA_REQUIREMENTS`), and gold says advance and it stays (`PROCESS_PAYOUT →
CONFIRM_SETTLEMENT` predicted as `→ PROCESS_PAYOUT`). This is a **destination-selection** problem —
the model tracks where it *is* and mis-chooses where to *go* — not a formatting or annotation-habit
problem.

**The two failures are correlated but not the same event.** State accuracy is **0.349 on
tool-expected rows** versus **0.888 on zero-tool rows**, and on tool-expected rows:

| | tool fired ok | tool missed |
|---|---|---|
| **state correct** | 24 | 13 |
| **state wrong** | 25 | 44 |

φ = **0.274**; 41.5% fail both, 35.8% are discordant. So a shared population fails both — the gold
turn is "advance the state *and* fire the tool", and the model instead stays put and asks the user —
but the coupling is loose enough that the two are partly independent levers rather than one relabelled.

### 11.5 Open question before the retrain

**Why did state accuracy fall 0.817 → 0.687?** The 0.817 came from the pre-regeneration n=150 audit;
this is the regenerated corpus at n=284. Both cannot be dismissed as sampling noise at these sizes.
Either the R12 regeneration changed the state-transition distribution, or the earlier figure was
computed on an easier slice. Since state accuracy is now the primary lever, **this should be resolved
before the factorial run** — otherwise the run may be tuned against a moving target. Cheap check: run
the same audit against the `*.pre_r12fix_backup_20260714T104107Z` directories preserved in §5.6.

> **RESOLVED (2026-07-22, no-GPU session).** Ran `scripts/sanity_check_state_acc_drop.py` — a
> pure-CPU comparison using the real production slicer (`_load_grpo_jsonl`) against the current
> canonical `data/output/grpo/task_a` and the preserved `*.pre_r12fix_backup_20260714T104107Z`
> snapshot, both `validation` split. Verdict: **`DIFFERENT_VALIDATION_SETS_NOT_MODEL_REGRESSION`**.
>
> - **Dominant factor:** the two validation sets are almost entirely disjoint conversation sets —
>   only 12/290 current conversations (4.1%) also appear among the backup's 135 *unique*
>   conversation IDs (8.9% of backup). `split_task_a_sft.py --force` (per CLAUDE.md R12)
>   regenerated the train/validation/test partition from scratch on a grown raw corpus, so
>   validation-set *membership* changed independently of any per-row cleaning. 0.817 (n=150) and
>   0.6866 (n=284) are not two measurements of the same held-out sample.
> - **Compounding factor:** the original 0.817 was explicitly restricted (by
>   `grpo_heldout_metric_and_gt_audit.md` §2) to the 109/150 rows with a non-empty GT
>   `state_sequence`, excluding 41/150 "empty GT transition" rows scored separately as an artifact.
>   In the *current* corpus, **0% of rows have an empty GT transition** (vs 25.1% in the backup) —
>   that exclusion bucket has essentially vanished, so even setting aside the conversation-set
>   mismatch, "0.817" and "0.6866" were never computed over comparable denominators.
> - **Tertiary observation:** corpus composition also differs sharply — backup rows are 94.8%
>   "advancing" (`from≠to`) vs current rows 62.5% advancing (37.5% self-loop), and total sliced
>   row count differs 1.83× (1,609 → 2,943) on unrelated conversation sets.
> - A tangential, unchased finding: the backup's `validation.jsonl` has 290 lines but only 135
>   unique `conversation_id`s (155 duplicate rows) — a pre-R12 data-quality artifact in a file
>   nothing trains or evaluates against today, noted in case it resurfaces elsewhere.
>
> **Conclusion: the drop is not evidence ckpt-1000 regressed. 0.6866 (n=284, current canonical
> corpus) is the trustworthy baseline to design the factorial run against — no moving-target risk
> from this comparison.** The GPU-side confirmatory reproduction (re-run the audit *against the
> backup corpus* with ckpt-1000 to see whether it recovers ~0.817 there) remains a nice-to-have for
> a future GPU-host session but is not blocking, since the data-side evidence above is already
> dispositive that the two samples are structurally incomparable regardless of what the model
> scores on either. Artifact: `runs/preflight/sanity_check_state_acc_drop.json`.

### 11.6 Revised plan

1. ~~**Resolve §11.5**~~ **Done — see the resolution above (2026-07-22).**
2. ~~**Re-scope the factorial run around state-transition accuracy**~~ **Done — see
   [`docs/superpowers/plans/2026-07-22-cat-a-state-accuracy-factorial.md`](superpowers/plans/2026-07-22-cat-a-state-accuracy-factorial.md)
   (2026-07-22).** A cost-disciplined SFT-recipe ladder (control → `response_only` masking →
   gated stay/advance decision-balance reweighting), explicitly not a GRPO/reward factorial since
   §9 already shows RL headroom is dead. Tool-turn upsampling is demoted to a guardrail + explicit
   next-round target (data regeneration), not folded into this round — it doesn't obviously address
   destination selection and would confound the state-accuracy result. That spec also surfaced a
   new **blocking** finding, since **confirmed via W&B + git/DVC provenance** (2026-07-22, prompted
   by the user pointing at the actual training run log): ckpt-1000's training run
   (`wandb.ai/wpawgasa/huggingface/runs/uklfswk5`, started 2026-07-02T04:58:27Z, git commit
   `ef837079`) consumed `data/output/sft/task_a_cleaned` DVC-pinned at that commit to hash
   `8ef8681808e348f82eac36edbbd6c2ae.dir` (131 files, ~228MB; `train.log` confirms `Map: 9131`
   train / `1074` eval rows downstream of it). The **current** `task_a_cleaned` is a *different*
   DVC hash, `3d6a4d3eed110546eccb46a94c5d68cd.dir` (5 files, ~110MB, feeding the current
   4,716/554/279 `task_a_splits`) — 48% the byte-size and restructured from 131 loose files into 5
   consolidated merged files. This is a **hash-verified**, not inferred, corpus-version
   discontinuity: R12's "the corpus grew (4,414→4,716)" narrative compared against a different,
   stale, now-absent directory and never accounted for this. The July-2 blob is not recoverable in
   this environment (no GCS credentials to `dvc pull` it). Corpus provenance must be confirmed
   before any GPU spend on the factorial run; see the spec's §4.2 and §11 item 1.
3. **Retire the 0.087 figure and the §1/TL;DR framing** wherever quoted — superseded by 0.4623.
   Done at the document level via the top-of-doc status banner (2026-07-22); §§1–6 are left
   textually unchanged as the dated historical record rather than silently rewritten.
4. Bar remains **0.80** (§10), now empirically corroborated by the measured 0.9494 abstention.

### 11.7 Recovering a tagged corpus version

The current, execution-verified `task_a_cleaned`/`task_a_splits` state is pinned by two git tags
(same data, `v2` additionally confirmed by actually running the pipeline rather than a hash-commit
— see §11.6 item 1's resolution note above): `task_a-corpus-v1-2026-07-22`,
`task_a-corpus-v2-2026-07-22`.

**In place** (restores `dvc.lock` on the current branch, then syncs data to match):
```
git checkout <tag> -- dvc.lock
dvc checkout data/output/sft/task_a_cleaned data/output/sft/task_a_splits
```
Leaves `dvc.lock` modified in the working tree — commit it to roll forward permanently, or
`git checkout HEAD -- dvc.lock` to undo.

**Isolated** (doesn't touch the current branch — safer for just inspecting an old version):
```
git worktree add /tmp/task_a-<tag> <tag>
cd /tmp/task_a-<tag>
dvc checkout
```

Both need the data blocks in the **local** DVC cache (`.dvc/cache`) by hash. That's satisfied here
since this session created and ran both tags. On a fresh clone or a different machine, `dvc pull`
first — which needs the `looloo-ocr` GCS service-account credentials this environment doesn't have
(the same limitation that made the July-2/ckpt-1000 corpus unrecoverable, §4.2 of the factorial
spec).
