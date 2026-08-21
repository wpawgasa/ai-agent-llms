# Voice Mode Switch and Voice Benchmark Data

**Date:** 2026-08-21
**Status:** Approved design. Not implemented.
**Scope:** `data/system_prompt.py`, `data/voice_convention.py`, the placeholder
generator, benchmark data generation, and Phase 1 scoring.

## Terms

This document uses one word for each concept. The list is binding.

| Keep | Drop |
|---|---|
| voice conversation | voice sample, TTS conversation |
| chunk | segment, span, piece |
| turn | message, reply |
| stratum | slice, subset, portion |
| check | verify, validate |
| fail | break, blow up |
| delete | remove, drop |
| marker | token, tag |

"Modality" means text or voice. Nothing else. "Stratum" means the text half or
the voice half of one evaluation.

## 1. Problem

Two gaps block voice work at evaluation time.

### 1.1 The system prompt has no voice mode

`src/llm_workflow_agents/data/system_prompt.py` holds no voice logic. The word
"voice" does not appear in it. The voice branch never touched it.

The earlier spec stated a decision it never delivered. It said: "One model
serves both modalities. The system prompt selects the mode." Nothing selects
the mode.

`build_enriched_system_prompt` builds the prompt for four consumers:

- `eval/agent_benchmark.py`
- `scripts/heldout_composite_audit.py`
- GRPO rollouts
- SFT training

A teacher-authored rich prompt covers about 30 percent of samples, and it does
carry voice instructions. The other 70 percent carry none.

This matters most for Phase 1. Phase 1 ranks **pre-trained** candidates. No
candidate has seen this convention. A voice conversation with no voice
instruction measures whether a model guesses an unstated rule.

### 1.2 The benchmark holds no voice data

`data/output/benchmark/task_a` holds text conversations only.

### 1.3 A provenance defect found while scoping this work

The benchmark artifact and its pipeline stage disagree. Measured on
2026-08-21:

| Property | `dvc.yaml` stage says | The files hold |
|---|---|---|
| Source | placeholder, no API key | 250 teacher, 8 unlabelled |
| Size | 1,000 conversations | 258 conversations |
| Model | none | `gemini-3-flash-preview`, `gemini-3-5-flash` |

`docs/data_generation_recipes.md` agrees with the files. Only the stage
description is wrong.

`dvc repro task_a_benchmark` would therefore delete 258 teacher conversations
and write 1,000 placeholder ones. That artifact is the data behind the current
Cat A ranking. Do not run that stage until someone corrects it. This design
does not fix it; it avoids it. See section 8.

### Decisions already made

1. The system prompt selects voice mode. Fix that first.
2. Phase 1 quality blends text and voice into one number.
3. Voice weight is 0.30.
4. The placeholder generator learns barge-in.
5. The voice benchmark corpus is additive. The text corpus is frozen.
6. Chunk diagnostics are reference-free and never enter the composite.

## 2. The voice mode switch

### One source for the rules

Add `VOICE_FORMAT_RULES` to `src/llm_workflow_agents/data/voice_convention.py`.
It states the five format rules and both length limits, once.

Three consumers render from it. None keeps a copy.

1. The new voice block in `system_prompt.py`.
2. `_VOICE_RULES` in `generate_workflows.py` (the teacher prompt).
3. `_RICH_VOICE_OVERRIDE` in `generate_workflows.py` (rich prompt authoring).

This retires a known defect. `tests/unit/test_voice_prompt_contract.py` checks
only that certain strings appear. A reviewer inverted one prompt's meaning from
"outside" to "inside" and every assertion still passed. One source removes the
drift that test was meant to catch. The test becomes an identity check.

### The block itself

`build_enriched_system_prompt` reads `sample.get("modality")`. It appends the
voice block only for a voice sample.

Number the voice rules `V1` to `V6`. Do not continue the existing sequence. The
base `FORMAT_RULES` numbering must not shift.

Two properties matter more than the block's wording.

**A text sample must render byte-identically to today.** Not nearly. Exactly.
That property keeps every stored result comparable, including the 0.7595
composite in risk R17. It gets a test that renders a real corpus row before and
after the change and compares bytes.

**The block sits inside the rebuilt region.** `build_enriched_system_prompt` is
idempotent through its `"Workflow script"` marker. `force_rebuild=True` strips
back to the bare persona line. Put the voice block after that marker. A rebuild
then renders it from current code. A block placed before the marker would
survive as a stale copy, which is the defect risk R13 records.

### Cost

The block adds about 250 tokens. The median prompt is 3,016 tokens. Under the
`response_only` recipe those tokens are context, not gradient.

## 3. Barge-in from the placeholder generator

The placeholder generator writes no `<unspoken>` marker today.

A barge-in needs three coordinated edits, not one insertion.

1. Select an assistant turn. It must sit mid-conversation and hold two or more
   chunks. It must not be the opening turn or a terminal turn.
2. Cut inside one chunk at a word boundary. Insert `<unspoken>`. Keep the text
   after it. The caller never heard that text.
3. Insert an interrupting user turn. Then insert a recovery assistant turn.

The recovery turn opens with an acknowledgement from
`ACKNOWLEDGEMENTS[language]`. It annotates the **same state** as the interrupted
turn. `find_barge_in_violations` already enforces both rules. An interruption
completes nothing, so the workflow must not advance.

Selection reads the sample's seeded random generator. Output stays reproducible.
The benchmark is a pipeline stage with a fixed seed, and the placeholder is the
reproducible path.

Two details follow from existing code.

**The ground truth stays correct.** The insertion runs before
`_extract_ground_truth`. The added self-loop turn therefore appears in
`state_sequence`.

**The flags set themselves.** `apply_barge_in_loss_flag` runs on the final
messages on every path. It sets `loss: False` on the marker-bearing turn. It
also sets the sample's `barge_in` field from what the messages hold.

### One fix while in this code

`code_switch` currently draws English acknowledgements. Code-switched
conversations are Thai-primary. Draw the Thai list for `code_switch`.

## 4. The voice benchmark corpus

### Freeze the text corpus

Do not regenerate `data/output/benchmark/task_a`. It is the text stratum.

### Add a voice stratum

Add `scripts/generate_benchmark_voice_data.sh`. Model it on
`scripts/generate_voice_data.sh`. It writes `data/output/benchmark/task_a_voice`
and gets its own pipeline stage. The existing stage does not change.

Three properties are chosen for measurement quality, not convenience.

**Size: about 250 conversations, 50 per level.** The text stratum holds 258.
Equal sizes give the two strata equal precision. Size no longer sets the
weighting, because the weight moved to a config file (section 5).

**Language: match the text stratum.** The text stratum mixes English and Thai.
It does so by passing no `language` argument, which draws English or Thai per
sample at even odds. The voice runner must do the same. Do not use the
20/50/30 Thai weighting that `generate_voice_data.sh` uses for the training
corpus.

A Thai-weighted voice stratum would confound modality with language. Then a
voice-versus-text gap could be a Thai-versus-English gap. Modality must be the
only difference between the strata.

**Source: teacher-generated, matching the text stratum.** The text stratum comes
from a teacher model. Placeholder conversations are structurally uniform. A
teacher text stratum against a placeholder voice stratum would confound modality
with generation source.

Use ONE teacher model for the whole voice stratum. The text stratum used two
(`gemini-3-flash-preview` for L1 to L3, `gemini-3-5-flash` for L4 and L5), which
is itself a defect and must not be copied. Record the model used in the stage
description.

The cost is about 250 teacher conversations.

### A consequence for section 3

A teacher-generated voice stratum carries teacher-written barge-ins. So section
3 does not serve the benchmark.

Section 3 still earns its place. It gives the offline path barge-in for smoke
tests. It also serves `generate_benchmark_data.sh`. It serves anyone who
regenerates without an API key.

## 5. Scoring

One benchmark run scores both strata and groups the rows by modality.

The existing text rows predate the `modality` field. A row without the field
counts as text. `filter_by_modality` and the held-out audit already use that
rule.

### Correction, measured 2026-08-21

An earlier draft of this section named the wrong formula and the wrong home for
the weight. Both are corrected here.

The live formula is `compute_weighted_workflow_score` in
`src/llm_workflow_agents/eval/composite_score.py`:

```
score = 0.4 x state_transition_accuracy + 0.4 x tool_call_f1
      + 0.2 x task_completion_rate
```

`configs/benchmark/selection_weights.yaml` states a different formula
(0.5 / 0.5). **No code reads that file.** A grep across `src/` and `scripts/`
returns no consumer. It is aspirational configuration. Do not put the voice
weight there: a knob nothing reads is the defect risk R16 and risk R18c both
record.

### The blend

Score each stratum with the existing formula. Then blend the two scores:

```
score_text  = compute_weighted_workflow_score(state_text,  tool_text)
score_voice = compute_weighted_workflow_score(state_voice, tool_voice)
score       = w x score_voice + (1 - w) x score_text
```

Blending at the score level keeps the per-stratum formula untouched. Within one
stratum, the number means exactly what it meant before.

Define `DEFAULT_VOICE_WEIGHT = 0.30` once, in `composite_score.py`. Expose it as
a `--voice-weight` flag on the benchmark runner. Leave
`selection_weights.yaml` alone; wiring that file into the scoring path is
separate work, named in section 9.

Compute a weighted mean of two per-stratum means. Do NOT compute one mean over
pooled rows. The two differ. A pooled mean takes its weight from the row counts,
so the weight drifts whenever anyone regenerates the data. Nobody notices,
because the result still looks like a number.

The blended number ranks the models. The two per-stratum numbers print beside
it. Without them, nobody can tell whether a winner is better at state machines
or better at emitting `<S>`.

The per-stratum quality mapping does not change. Within one stratum, "quality"
means what it always meant.

### The change must be inert until voice data exists

With no voice corpus present, the blended number must equal today's number
exactly. A float-identity test enforces this.

This makes the change safe to merge. Results move when a person adds the voice
corpus. Results do not move because someone merged a branch. Risk R16 records
what a silently-active change costs.

### The cost of blending

Once the voice corpus exists and `w` is 0.30, the composite no longer means what
it meant when Cat A's winner was selected. A comparable ranking needs Phase 1
re-run across the candidate set. That is 15 candidates.

This cost follows from blending. It does not follow from anything else in this
design.

## 6. Chunk diagnostics

### Why not chunk accuracy against ground truth

The benchmark scores free generation. The model's spoken words differ from the
gold spoken words. Once the words differ, the boundaries cannot be aligned. A
"chunk F1 against gold" would compare boundaries in two different sentences. It
would return a number, and the number would be noise.

Teacher-forced scoring would make the comparison valid. This harness does not
use it.

### Three reference-free metrics

**First-chunk length, p50 and p90.** This is the latency metric. The
orchestrator starts text-to-speech on chunk 1 while the model still writes chunk
2. A 150-character opening chunk delays audio about ten times as long as a
15-character one. Nothing measures this today.

**Chunk length and chunk count distribution.** Compare against the corpus
distribution, not against per-row gold: median 42 characters, p90 85, and 2
chunks per turn. This catches what format compliance cannot see. One
155-character chunk per turn passes every format rule and streams badly.

**Boundary quality.** The share of chunks ending at a real pause point. Thai
marks this explicitly through sentence-final particles (ค่ะ, คะ, ครับ). English
uses terminal punctuation. A `code_switch` conversation accepts either. This is
the closest honest proxy for splitting where a person would breathe.

### All three are guardrails

None enters the composite. `voice_format_compliance` follows the same rule.

Chunk formatting is cheap to install through fine-tuning. Letting it move the
Phase 1 winner would select on the wrong capability. These metrics diagnose a
candidate. They do not rank one.

They cost no extra inference. They read generations the benchmark already
produces.

## 7. Tests

| File | Checks |
|---|---|
| `tests/unit/test_system_prompt_voice.py` | A voice sample gets the block. A text sample renders byte-identically to today, on a real corpus row. `force_rebuild` regenerates the block. |
| `tests/unit/test_voice_prompt_contract.py` | Rewritten as an identity check: all three consumers render from `VOICE_FORMAT_RULES`. |
| `tests/unit/test_data_generation.py` | Placeholder barge-in: exactly one marker, never in the last turn, recovery opens with an acknowledgement, recovery holds the state. Output is reproducible at a fixed seed. `code_switch` draws Thai. |
| `tests/unit/test_benchmark_scoring.py` | The blend at `w` = 0, 0.3 and 1. Float-identity with no voice rows. Grouping when the `modality` field is absent. The guardrail never enters the composite. |
| `tests/unit/test_chunk_diagnostics.py` | Known-answer tests for all three metrics, including a turn with no chunks. |
| `tests/unit/test_generate_benchmark_voice_data_sh.py` | Emitted kwargs bind to the real signature. Per-level counts sum to the target. |

Write the tests first.

## 8. Risks

| # | Risk | Detection |
|---|---|---|
| 1 | Someone runs `dvc repro task_a_benchmark` and deletes the text stratum. | Out of scope here and unfixed. Named in section 1.3. Fix the stage description in separate work. |
| 2 | The voice block shifts text-sample bytes. | The byte-identity test on a real corpus row. |
| 3 | The blend moves the ranking on merge, before anyone adds voice data. | The float-identity test. |
| 4 | Placeholder barge-in breaks seed reproducibility. | The existing seed-determinism tests, plus a fixed-seed barge-in test. |
| 5 | The two strata differ by more than modality. | Section 4 pins size, language and source. Review the generated stratum against those three properties before use. |

Risk 1 is the one to act on soonest. It is pre-existing, it is one line of
description against 258 conversations, and this design cannot prevent it.

## 9. Out of scope

- Fixing the `task_a_benchmark` stage description. Separate work, named here.
- Wiring `configs/benchmark/selection_weights.yaml` into the scoring path. No
  code reads it today. Separate work, named in section 5.
- Re-running Phase 1 across the 15 candidates.
- Changing the composite weights outside the new `voice_weight` key.
- Chunk-boundary comparison against gold under teacher forcing.
- Any training run.
