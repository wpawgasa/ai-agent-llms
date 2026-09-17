# Gemma-4 never saw a tool result in training — design for the fix

**2026-09-17.** Status: **design, not implemented.** Decisions D1–D4 at the end
need an owner before implementation starts.

## 1. The problem

### 1.1 Tool results are dropped from every Gemma-4 training sequence

The Task A corpus writes a tool call as plain text inside an assistant
message's `content` (`[STATE: X → X]\n<tool_call>{json}</tool_call>`) and the
result as a following `{"role": "tool", "content": "{json}"}` message, with no
structured `tool_calls` field and no `tool_call_id`.

The Gemma-4 chat template (`gemma4_unified` and the E-series share this logic)
renders a `tool` message **only** from inside an assistant message that carries
structured `tool_calls` (the "forward-scan consecutive role:tool messages"
branch) or `tool_responses`. Its main loop skips `role == "tool"` otherwise.
So every tool result in our corpus is silently discarded.

Measured with `apply_chat_template` on `data/output/sft/task_a_splits/train.jsonl`:

| Tokenizer | Conversations with tool results | Tool messages | `<|tool_response>` blocks rendered |
|---|---|---|---|
| google/gemma-4-12B-it | 191 | 796 | **0** |
| unsloth/gemma-4-E4B-it | 191 | 796 | **0** |
| Qwen/Qwen2.5-0.5B-Instruct (control) | — | 171 | 171 (100%) |

What the model is trained on after a tool call is therefore the next assistant
turn, directly:

```
[STATE: CHECK_ELIGIBILITY → CHECK_ELIGIBILITY]
<tool_call>{...}</tool_call><turn|>
<|turn>model
[STATE: CHECK_ELIGIBILITY → VERIFY_DOCUMENTS]
จากการตรวจสอบเบื้องต้น ระบบแจ้งว่า...   ← narrates a result that is not in the sequence
```

### 1.2 What that produced

- **Run-on after a tool call.** Share of held-out completions that emit a tool
  call and keep writing (a new `[STATE:]` line after `</tool_call>`):

  | | Text | Voice |
  |---|---|---|
  | 12B untrained | 15.2% | 14.3% |
  | 12B SFT | **50.5%** | **40.0%** |
  | E4B untrained | 56.4% | 38.1% |
  | E4B SFT | **65.3%** | **75.5%** |

  The corpus does this on 0.13% of tool-call turns (44 of 32,805) and never
  writes a second `[STATE:]` after a call. Fine-tuning taught it.
- **Invented tool results.** Those run-on completions narrate outcomes they
  never received ("received request number AUTH-88120", "reference
  LN-2023-99812").
- **The missing `[` in `STATE: X → Y]`.** With nothing between `</tool_call>`
  and the next turn, the tokenizer merges `>` and `[` into one token (`>[`);
  spans cut on either side of it lose a character.
- **Plausibly, the argument errors.** 77.5% of the 26B C2 model's mined
  mistakes were failing to carry a value from a tool result into the next call
  (CLAUDE.md R23). A model that never saw a tool result cannot learn to copy
  from one. This is a hypothesis for the retrain to test, not a finding.

### 1.3 Inference sees a different format again

`eval/agent_benchmark.py` synthesizes structured `tool_calls` from the model's
`<tool_call>` text, sends `tool` messages with `tool_call_id`, and passes
`tools=`. The template then renders, for the same conversation:

```
<|turn>model
<|tool_call>call:check_benefit_eligibility{benefit_type:<|"|>unemployment<|"|>,...}<tool_call|>
<|tool_response>response:check_benefit_eligibility{value:<|"|>{"status": ...}<|"|>}<tool_response|>
[STATE: CHECK_ELIGIBILITY → CHECK_ELIGIBILITY]<turn|>
```

plus a native `<|tool>declaration:...<tool|>` block in the system turn from
`tools=`. None of this appears in training: native call syntax, native
response blocks, the tool declaration block, and the `[STATE:]` line moved
after the call and its result. Three train/inference mismatches in one turn.

### 1.4 The render fix `fc7bc1d` is not the fix

It stopped training the turn header, but its probe-render span detection is off
by one token wherever content and template tokens merge: on 344 assistant turns
checked, **166 (48%) had their leading `[` masked**, and some tool-call turns
lost their closing `>`. It does nothing about §1.1. Branch
`fix/response-only-masks-turn-header` (`fc7bc1d`, `bc15531`) must not be
merged, and the `sft_cat_a_12b_rerender` run trained on it is not a valid test
of anything.

### 1.5 Scope

- **Confirmed:** gemma-4-12B-it and gemma-4-E4B-it tokenizers, therefore every
  Gemma-4 12B and E4B SFT run on this corpus.
- **Very likely, unverified here:** the 26B-A4B C2 model (0.7595). Its
  tokenizer is not cached on this machine. On the machine that has it:
  render any training conversation containing a `tool` message and count
  `<|tool_response>`; zero confirms.
- **Same rendering path, so same condition:** the held-out audit, the RFT
  headroom probes, DPO pair rendering and GRPO rollouts all call
  `apply_chat_template` on the same message shape. Their conclusions,
  including R23's NO_GO, were measured on models and prompts without tool
  results — consistent with training, not with the benchmark.

## 2. Options

**A. Native format everywhere.** Convert tool calls to structured `tool_calls`
and results to `tool` messages with ids for training too. *Rejected.* The
template renders native calls **before** `content`, so `[STATE: X → X]` would
follow the call and its result. That inverts the annotate-then-call convention
the corpus, the stay rule (FORMAT_RULES rule 2), `parse_state_transitions` and
the held-out scorer are built on, and changes the model's output syntax to
`<|tool_call>call:name{...}` with custom string quoting.

**B. Text format everywhere (recommended).** Keep `<tool_call>{json}</tool_call>`
in assistant content, and give each tool result to the template as a
`user` message with a fixed prefix — a message every chat template renders.
Apply the same transform at training, at every probe and audit, and at
benchmark inference, and stop sending `tools=` for models in this format. The
benchmark already has this transform for Gemini (`_downgrade_tool_turns_to_text`).

**C. Hand-render tool results inside the model turn.** *Rejected.* A bespoke
per-template format that vLLM's chat template cannot reproduce at inference.

## 3. Design (option B)

### 3.1 One transform, one module

New `src/llm_workflow_agents/data/tool_turns.py`:

```python
TOOL_RESULT_PREFIX = "[Tool result]: "

def to_text_tool_turns(messages: list[dict]) -> list[dict]:
    """Assistant tool calls as <tool_call> text; tool results as prefixed user turns."""
```

- An assistant message with structured `tool_calls` gets them appended to
  `content` as `<tool_call>{json}</tool_call>` lines, serialized **byte-identically
  to the corpus** (verify separators and `ensure_ascii` against real rows), and
  loses the `tool_calls` field.
- A `tool` message becomes `{"role": "user", "content": TOOL_RESULT_PREFIX + content}`;
  `tool_call_id` and `name` are dropped.
- Idempotent; keeps the per-message `loss` key (R20); leaves every other
  message untouched.
- `_downgrade_tool_turns_to_text` in the benchmark becomes a call to this
  function, so Gemini's path and the text path cannot drift.

Verified on 60 real conversations: with this transform all 267 tool results are
visible in the Gemma-4 render (0 before), and 1,695 of 1,722 turns extend their
own prefix; the other 27 are all two consecutive assistant turns, which the
template merges into one model turn (handled by §3.3).

### 3.2 Every render site uses it

`apply_chat_template` is called in nine files; each must render
`to_text_tool_turns(messages)`:

- `training/sft.py` (render), `training/grpo.py` (`_load_grpo_jsonl` prompts),
  `training/dpo.py`, `training/trajectory_rollout.py`, `training/_utils.py`
- `scripts/preflight_entropy_diag.py` (held-out audit and headroom probes),
  `scripts/free_running_multiturn_probe.py`, `scripts/trajectory_variance_probe.py`,
  `scripts/build_dpo_smoke_fixtures.py`

Guard: a test that pushes one conversation containing a `tool` message through
each loader and asserts the tool result's text appears in the rendered prompt.
That test is what would have caught this.

**Side effect to decide (D2):** `_load_grpo_jsonl` admits an assistant turn as
a row only when the previous message is `user` or `system`, and skips the rest
(`skipped_tool_preceded_turns=28649` on train). After the transform those turns
are preceded by a `user` message and become rows — exactly the turns that
consume tool results. This grows the GRPO and probe pools.

### 3.3 Loss masking by character offsets

Replace the probe-render span detection:

1. Render the transformed conversation once as text.
2. Tokenize it with `return_offsets_mapping=True` (both Gemma tokenizers are
   fast tokenizers; verified).
3. For each trainable assistant message, find its rendered content by
   sequential search from a cursor (the template trims model content, so search
   the trimmed string).
4. Measure the turn terminator once per tokenizer: render `[user, assistant "A"]`
   and take the text after `A` (`<turn|>\n` for Gemma-4).
5. Label every token that overlaps `[content_start, content_end + terminator)`
   when the terminator follows the content; content alone when the template
   continued the model turn instead.

A token straddling a boundary (`>[`) is labelled, so content is never cut. A
content string that cannot be located labels nothing and increments a logged
counter; it is never guessed. Bump `_RENDER_CACHE_VERSION` and hash the new
module into the cache key.

### 3.4 Benchmark

- New `--tool-turn-format {native,text}` for local engines. `text` sends
  `to_text_tool_turns(context)`, omits `tools=`, and relies on the existing
  `parse_tool_calls` over `content` (already the fallback path today).
- Record the format in the result JSON.
- `skip_unsolicitable_assistant_turn` still works: a tool result is now a
  `user` turn, which it already accepts.

## 4. Validation before any retrain

- **V1 — render audit (full corpus, both tokenizers):** tool results present in
  100% of conversations; every trained span decodes to exactly the message
  content plus terminator; zero spans starting after `[` or ending before `>`;
  no user or tool-result text in any label. Ship as
  `scripts/audit_tool_result_rendering.py`, not a one-off.
- **V2 — parity:** for the same turn, the SFT training prompt and the
  benchmark's text-mode request render to the same string up to the generation
  prompt.
- **V3 — smoke SFT (~50 steps, E4B):** loss non-zero, tokens per sample
  consistent with the 8192 window (R16).
- **V4 — retrain E4B first** (~6.3 h; 12B is ~16.3 h), then measure against the
  first E4B run, both in text mode:
  - run-on after tool calls (target: near the corpus's 0.13%)
  - malformed `STATE:` without `[` (target: 0)
  - argument exact match on chain-dependent turns (`prompt_filters.py`)
  - benchmark v2 text + voice, held-out audit, and the strict-reward headroom
    probe (`--reward strict --prompt-filter chain_dependent`)
- **V5 — 12B**, same protocol, if V4 improves.

## 5. Consequences

- Every Gemma-4 SFT result on this corpus — E4B, 12B, and very likely C2's
  0.7595 — is a measurement of a model trained without tool results. The
  numbers stand as measurements of those checkpoints, not of the recipe.
- R23's conclusion that argument fidelity is out of reach of single-turn RL was
  measured on those models; revisit it after V4.
- Benchmark comparisons between an untrained model and a model trained in text
  format must run both in the same `--tool-turn-format`.
- Supersedes `fix/response-only-masks-turn-header`; its header-masking goal is
  kept by §3.3.

## 6. Open decisions

- **D1 — benchmark default.** Keep `native` as default (existing Phase 1
  rankings unchanged) and pass `text` explicitly for this corpus's models, or
  switch the default and re-run the ranked set.
- **D2 — GRPO/probe pools.** Admit the post-tool-result turns §3.2 unlocks
  (recommended: they are where chain-carried arguments live).
- **D3 — result prefix.** `[Tool result]: ` alone, or include the tool name
  (`[Tool result: book_reservation]: `), which helps turns with several calls
  but must then be derivable identically at training and inference.
- **D4 — order.** E4B then 12B here; whether and where to re-run C2 26B.
