# Voice Conversation Generation for Cat A

**Date:** 2026-08-20
**Status:** Approved design. Not implemented.
**Scope:** `data/` generation, `training/sft.py` masking, held-out audit protocol.

## Terms

This document uses one word for each concept. The list is binding.

| Keep | Drop |
|---|---|
| voice conversation | voice sample, TTS conversation, call |
| chunk | segment, span, piece |
| turn | message, reply, response |
| check | verify, validate, confirm |
| fail | break, blow up |
| start | launch, kick off |
| delete | remove, drop, purge |
| marker | token, tag, signal |

"Modality" means text or voice. Nothing else.

## 1. Problem

Cat A models must serve a voicebot as well as a chat system. The current Task A
corpus holds only written conversations. A written turn is long. A spoken turn is
short.

A voicebot needs one more thing. The orchestrator reads the model output as a
stream. It looks for chunk markers. It sends each chunk to the text-to-speech
engine in order. The model must therefore split its own speech into chunks.

Two production prompts show the target format. They are
`data/templates/monomax_prompt_parts.json` and
`data/templates/oceanlife_prompt_parts.json`. Neither file has a code consumer
today. Both are reference material.

### Decisions already made

1. One model serves both modalities. The system prompt selects the mode.
2. The corpus carries `<S>` chunks, `[END_CONVERSATION]`, and `<unspoken>`
   barge-in recovery. It does not carry turn-limit markers.
3. The voice conversations arrive as an additive batch. The existing text rows do
   not change.
4. The voice batch is weighted toward Thai.
5. An interrupted turn carries a per-turn loss flag.

### The reversal this records

`_RICH_PROMPT_SYSTEM` in `src/llm_workflow_agents/data/generate_workflows.py:1281`
states a rule today. It forbids `<S>`, `<F>`, `[END_CONVERSATION]`, and
`[TRANSFER]`. It calls them deployment concerns. This design reverses that rule
for voice samples. The rule stays for text samples.

## 2. The voice format contract

An assistant turn in a voice conversation has this shape:

```
[STATE: OFFER_PACKAGE → OFFER_PACKAGE]
<S>เข้าใจค่ะ</S><S>ขออนุญาตตรวจสอบสิทธิ์ให้สักครู่นะคะ</S>
<tool_call>{"name": "check_eligibility", "arguments": {"customer_id": "C8842"}}</tool_call>
```

Five rules define the format.

1. The `[STATE: X → Y]` line is the first line. It sits outside every chunk. The
   agent never speaks it.
2. Each `<tool_call>` block sits outside every chunk. The agent never speaks it.
3. Every spoken character sits inside a chunk. Delete the state line. Delete each
   tool call. Each remaining non-whitespace character must sit between an `<S>`
   and its `</S>`. A turn with no spoken text at all is legal and carries no
   chunk. A turn that only calls a tool is silent on the line. The production
   reference states this: "Format spoken text with `</S>`; emit no delimiter
   when there is no speech." Do not invent filler speech to give such a turn a
   chunk. Uniform filler before every tool call is the kind of structural edit
   risk R15 records being learned as an unconditional habit.
4. Chunks do not nest. The count of `<S>` equals the count of `</S>`. Each `<S>`
   precedes its `</S>`.
5. `[END_CONVERSATION]` follows the last chunk of a terminal turn. It sits
   outside the chunks. It never appears on a turn that holds a tool call.

Rules 1 and 2 keep the format cheap. `_STATE_ANNOTATION_RE`,
`extract_state_annotations`, `extract_tool_calls`, `eval/state_accuracy.py`, and
`eval/tool_call_f1.py` need no change. Rule 3 makes rules 1 and 2 measurable.

### Length limits

The limits come from measurement, not from taste. The two reference prompts hold
45 chunks across 19 turns.

| Statistic | Measured |
|---|---|
| Chunk length, median | 42 characters |
| Chunk length, p90 | 85 characters |
| Chunk length, maximum | 117 characters |
| Chunks per turn, median | 2 |
| Chunks per turn, maximum | 5 |

The limits follow from those numbers.

| Level | Target | Violation above |
|---|---|---|
| Chunk | 100 characters | 160 characters |
| Turn | 3 chunks | 5 chunks |

The limits count characters, not words. Thai and code-switched text have no
reliable word boundary. A turn above either violation limit goes back to the
teacher model.

### Text samples

A text sample must hold no `<S>`. It must hold no `[END_CONVERSATION]`. It must
hold no `<unspoken>`. The checker enforces both directions. Without the second
direction the modality field is advisory.

## 3. Generation control

### The modality preset table

Add `MODALITY_PRESETS` beside the three preset tables at
`src/llm_workflow_agents/data/generate_workflows.py:137-176`.

| Preset | text | voice |
|---|---|---|
| `default` | 1.00 | 0.00 |
| `voice_mix` | 0.70 | 0.30 |
| `voice_heavy` | 0.40 | 0.60 |
| `voice_only` | 0.00 | 1.00 |

The `default` preset is text-only. This choice protects reproducibility. Each
sample draws from a child random generator seeded from `seed`. A new draw shifts
that stream. Every existing config would then produce different output.

Therefore `_select_modality` runs only when the preset is not `default`. The
default path consumes no randomness. `tests/unit/test_data_generation.py:116`
already checks seed determinism. It will catch a mistake here.

### How the batch reaches 30 percent

Generate the voice batch with `voice_only`. Do not re-roll the corpus. The
existing text rows must stay byte-identical to the rows cell C2 trained on.

### Language

Do not add a language parameter. `scripts/generate_sft_data.sh` already produces
an even language split. It runs three legs of equal size.

Add `scripts/generate_voice_data.sh`. It runs the same three legs at 20 / 50 / 30
for en / th / code_switch.

### Changes to thread the modality through

1. Add `ConversationSample.modality: str = "text"`. Add it to `to_dict()`. Add it
   to the stats sidecar.
2. Replace the `_TEACHER_SYSTEM_PROMPT` constant with
   `_teacher_system_prompt(modality)`. The text branch returns today's bytes. The
   voice branch holds a chunked OUTPUT FORMAT example and the five rules of
   section 2.
3. Split `_RICH_PROMPT_SYSTEM` the same way. Invert its marker rule for voice.
   Rich prompts are 30 percent of samples. They consist mostly of quoted dialogue
   lines. For voice those quoted lines must carry chunks. `9_instructions` in both
   reference files shows this shape.
4. Teach `_generate_placeholder_conversation` to render voice turns. A fallback
   keeps the modality of its sample. A text-shaped fallback inside the voice batch
   would contradict its own label.
5. Add `--modality-preset` to `scripts/generate_sft_data.sh`. Pass it through
   `scripts/generate_sft_until_target.py`.

`tests/unit/test_generate_sft_data_sh.py:83` binds the shell kwargs against the
real signature. It covers step 5 already.

Point 2 needs a reason. The current constant shows an OUTPUT FORMAT example
without chunks. A voice block appended to the user prompt would contradict that
example. The teacher model would then choose between them.

## 4. Checks and barge-in

### The new module

Add `src/llm_workflow_agents/data/voice_convention.py`. Model it on
`src/llm_workflow_agents/data/state_convention.py`, which is 118 lines.

It exports three functions.

- `find_voice_violations(messages, modality)` returns a list of strings. For a
  voice sample it checks the five rules, both length limits, the
  `[END_CONVERSATION]` position, and the barge-in recovery. For a text sample it
  checks the three absence rules of section 2.
- `strip_voice_markup(text)` deletes the chunk markers and the control markers.
  The held-out audit needs it. The reward functions need it. Both compare a voice
  completion against a text-convention ground truth. The markup must not count as
  a difference.
- `iter_chunks(text)` returns the chunks of one turn.

Call `find_voice_violations` inside `_find_violations` at
`generate_workflows.py:1755`. Put it beside `find_tool_stay_violations`. Voice
violations then use the existing repair path. They also use the existing
placeholder fallback. This design adds no new control flow.

### Barge-in is generated, not injected

Do not post-process a finished conversation. Cutting one turn is not enough. The
next two turns must also change. The caller turn must become an interruption. The
assistant turn must acknowledge the interruption and repeat the lost content.

A mechanical injection would produce a conversation where the caller answers a
question the caller never heard. The repair loop exists to catch that kind of
incoherence.

Therefore some voice samples carry a `barge_in` flag. The teacher model writes all
three turns.

Add the parameter `barge_in_rate: float = 0.25` to `generate_workflow_dataset`.
Draw the flag only for a voice sample. A text sample consumes no randomness for
this draw. Section 3 states the reason.

### Checking the recovery

The checker does not read meaning. It checks four facts.

1. The `<unspoken>` marker appears exactly once in the conversation.
2. The marker sits in an assistant turn that is not the last turn.
3. The next assistant turn starts with an acknowledgement from a known list.
4. The next assistant turn annotates the same state as the interrupted turn.

Fact 4 has a reason. A barge-in completes nothing. The workflow does not advance.

Fact 3 needs a list. Define `ACKNOWLEDGEMENTS: dict[str, tuple[str, ...]]` in
`voice_convention.py`. Key it by language. Seed the Thai entry from the five
examples in `4_barge_in_block` of the two reference files. Write the English
entries to match. The teacher prompt states the same list, so the checker and the
teacher model never disagree.

### The loss flag

The interrupted turn carries `"loss": false`.

The orchestrator writes `<unspoken>` into the model's own past turn. The model
does not write it. `render_response_only_sample` at
`src/llm_workflow_agents/training/sft.py:133` masks by role alone. Every
assistant turn is a target today. Training on the marker would teach the model to
emit the marker. Risk R15 records what happened the last time a uniform edit
became an unconditional habit.

Four call sites must honour the flag.

1. `render_response_only_sample` reads `msg.get("loss", True)`. It masks the span
   to `-100` when the value is false. The default keeps all 5,549 existing rows
   unchanged.
2. `training/grpo.py::_load_grpo_jsonl` skips such a turn as a training row. It
   keeps the turn in the prompt prefix.
3. `scripts/build_preference_pairs.py` never uses such a turn as a chosen turn.
   `scripts/mine_model_negatives.py` never uses it as a gold turn.
4. `scripts/clean_task_a_sft.py` and `src/llm_workflow_agents/data/data_validator.py`
   must keep the key. Both keep it today. `clean_record` filters the message
   list. It does not rebuild the message dicts. `data_validator` holds no
   allowlist of message keys. This point therefore needs a regression test, not
   a code change. Risk R12 made the cleaner delete an unknown role, so the
   guard is worth its one test.

One limit is real. The `all_tokens` recipe cannot express a per-turn opt-out.
Under that recipe the code ignores the flag and logs a warning. The `response_only`
recipe produced the only valid Cat A result, so this limit is acceptable. See
risks R16 and R17. The warning must be loud. An `all_tokens` run on voice data
teaches the model to emit `<unspoken>`.

## 5. Corpus and evaluation

### Size

The corpus holds 5,549 conversations today. Solve `V = 0.3 × (5549 + V)`. The
result is 2,378. Round it to **2,400**.

Split the 2,400 by language at 20 / 50 / 30.

| Language | Conversations |
|---|---|
| en | 480 |
| th | 1,200 |
| code_switch | 720 |

The merged corpus holds about 7,949 conversations. Then run the existing chain.

1. Run `scripts/clean_task_a_sft.py`.
2. Run `scripts/split_task_a_sft.py --force`.
3. Run `scripts/filter_grpo_data.py`.

The splits land near train 6,750, validation 795, test 400.

### The DVC stages

The text chain runs `task_a_sft_generate`, then `task_a_sft_remediate`, then
`task_a_sft_clean`, then `task_a_sft_splits`.

Add the stage `task_a_sft_generate_voice`. It runs
`scripts/generate_voice_data.sh`. It writes `data/output/sft/task_a_voice`.

The voice batch skips `task_a_sft_remediate`. That stage replays an authoring
ledger. The ledger names specific existing conversations. The voice batch also
needs no repair, because `require_tool_stay` is true at generation time.

The voice batch therefore joins the chain at `task_a_sft_clean`. Two scripts must
accept more than one input directory. They are `scripts/clean_task_a_sft.py` and
`scripts/split_task_a_sft.py`. Change `--input-dir` to accept a list of paths in
both. A single path keeps today's behaviour.

Do not change `task_a_sft_generate`. That stage holds the text corpus. Risk 1 of
section 7 depends on the text corpus staying fixed.

### Report two held-out numbers

Do not blend the two modalities into one score.

The pinned 206-row set of risk R17 does not change. It is the only link to cell
C2's composite of 0.7595. Build a second set for voice. Add the flag
`--modality {text,voice,all}` to `scripts/build_heldout_clean_set.py`. The
default is `all`, so the current behaviour does not change.

The audit reports the text score on the 206 rows. It reports the voice score on
the voice rows. A blended score would move the pre-registered bar of 0.75 without
a decision.

One contamination rule applies. Compute `user_turn_fingerprint` on the output of
`strip_voice_markup`.

This change is a no-op today. `user_turn_fingerprint` hashes user turns only, and
only an assistant turn carries markup. The change is defensive. It keeps the
fingerprint correct if a user turn ever carries a marker.

### Voice format compliance is a guardrail

Report it beside `state_acc`, `tool_f1`, and `task`. Do not weight it into the
composite. A new composite term changes what 0.7595 means.

The metric is the share of voice assistant turns that satisfy section 2.

## 6. Tests

The strategy follows `.claude/rules/08-testing.md`.

| File | Checks |
|---|---|
| `tests/unit/test_voice_convention.py` | Each rule of section 2, in both directions. Boundary values at 100 and 160 characters. Boundary values at 3 and 5 chunks. |
| `tests/unit/test_data_generation.py` | The modality draw. Seed determinism under `default`. Placeholder voice output passes `find_voice_violations`. |
| `tests/unit/test_response_only_loss_flag.py` | A `loss: false` span is all `-100`. An absent key behaves as `true`. |
| `tests/unit/test_generate_sft_data_sh.py` | The new flag. This file already binds shell kwargs against the real signature. |

Write the tests first. Follow `superpowers:test-driven-development`.

The seed determinism test is the important one. A shifted random stream would
invalidate every existing config, and it would do so silently.

## 7. Risks

| # | Risk | Detection |
|---|---|---|
| 1 | The voice rows reduce text ability. | The text score on the pinned 206 rows. This split exists for this risk. |
| 2 | The teacher model fails the format. Placeholder rows then fill the voice batch. | Read `repair_fallbacks` and `generation_source` in the stats sidecar. Gate the merge on them. |
| 3 | An `all_tokens` run teaches the model to emit `<unspoken>`. | The loud warning of section 4. |
| 4 | `dvc.lock` drifts. | Open already under risk R12. This environment has no `dvc` command. It stays a manual step. |

Gate the work on risk 2. A failed teacher run yields 2,400 deterministic
placeholder rows under a voice label. Risk R15 records what a structurally uniform
corpus teaches a model.

## 8. Out of scope

- Turn-limit markers `[TURN_LIMIT_WRAPUP]` and `[TURN_LIMIT_FINAL]`.
- The `[TRANSFER]` marker and call redirection.
- A separate voice checkpoint. One model serves both modalities.
- Any change to the composite score weights.
- Any GRPO or DPO run. This work produces data and checks only.
