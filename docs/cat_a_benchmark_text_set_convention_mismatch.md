# Cat A benchmark — the text set predates the tool-call stay rule, and a conversation-id collision

**2026-09-16.** Branch `fix/benchmark-conversation-id-collision`.

Two findings from rebuilding the Phase 1 Task A benchmark conversation by
conversation:

1. **Most of the fine-tuned models' text task-completion drop comes from the
   text set itself.** It was generated in June, before the tool-call stay rule
   (a turn that calls a tool stays in its state; the advance moves to a later
   turn). Corpus v2 and v3 follow the rule; 81 of the 258 text conversations do
   not. A model trained on the rule falls a step behind on those and runs out of
   replayed turns before reaching the terminal state.
2. **Whole-run state metrics scored text conversations against voice ground
   truth.** Both strata number their conversations `L1_001`, `L1_002`, … and
   `evaluate_state_machine` looked ground truth up by id. Fixed in this branch.
   The ranking number (`quality_summary.quality`) was never affected.

---

## 1. Method: rebuilding every conversation from the logs

`agent_benchmark.py` stores only totals, but its DEBUG log records every model
reply (`model_response`) under the conversation it belongs to
(`evaluating_sample idx=…`). `.runs/eval_12b_template/rebuild_bench.py`
replays the benchmark's own message walk: ground-truth user and tool messages,
ground-truth preamble turns before the first user message, and the logged reply
at every other assistant turn. It then re-scores with the benchmark's own
functions.

**Validation:** all five local runs reproduce their stored `quality_text`,
`quality_voice` and `quality` to six decimal places (E4B base and SFT, 12B base
and SFT, 12B SFT with the patched template). The two Gemini logs cannot be
rebuilt: they hold 5,663 replies against 5,959 assistant turns, so replies do
not line up with turns.

---

## 2. The text drop is task completion

Text-stratum components for the 12B, untrained → SFT:

| Component | Weight | Untrained | SFT | Effect on text score |
|---|---|---|---|---|
| State accuracy (best of turn / sequence) | 0.4 | 0.8707 | 0.8721 | +0.001 |
| Tool-call F1 | 0.4 | 0.5003 | 0.5357 | +0.014 |
| Task completion | 0.2 | 0.7403 | **0.5155** | **−0.045** |

The sum is the whole −0.030 text drop. Completion by stratum:

| | 12B base | 12B SFT | E4B base | E4B SFT |
|---|---|---|---|---|
| Text (258) | 0.740 | **0.516** | 0.535 | **0.450** |
| Voice (250) | 0.772 | 0.840 | 0.456 | 0.544 |

The fine-tuned models finish more voice conversations and fewer text ones, for
both sizes. Most failed text conversations end one state short of the terminal
state. At the turn where ground truth enters the terminal state, the 12B SFT
model is still at an earlier state 40% of the time, against 17% untrained: it
fell behind earlier in the conversation. Its state sequence accuracy stays level
because it visits the right states, a turn late.

**Ruled out, with data:** the leaked `model` word (the patched template removes
60% of it and completion does not recover); the terminal move landing on a tool
turn (no dataset does that); a different closing structure (text benchmark and
corpus-v3 text close the same way: the terminal-entry reply asks a question
10.9% vs 9.8%, is the last assistant turn 90.3% vs 89.2%); terminal replies
claiming a result with no tool output (18 conversations, and completion falls
just as much without them); malformed `[STATE:` markers (90 of 5,959 replies).

---

## 3. The text set predates the stay rule

`P(tool call | advancing turn)`, from the `[STATE:]` markers:

| Set | Advancing turns that call a tool |
|---|---|
| Benchmark text (June) | **7.8%** |
| Benchmark voice (August) | 0.0% |
| Corpus v3 train, text | 0.0% |
| Corpus v3 train, voice | 0.0% |

`scripts/remediate_task_a_states.py triage` (no LLM, read-only) on the text set:

| Move needed | Conversations | What the fix does |
|---|---|---|
| `none` | 177 | — |
| `relabel` | 25 | Changes `[STATE:]` labels in place; prose, tool calls, length unchanged |
| `insert_handoff_turn` | 32 | Inserts authored assistant hand-off turns (+ user padding) |
| `append_closing_pair` | 24 | Appends an authored user + assistant closing pair |

`verify` on the voice set: 0 violations.

Completion by triage group:

| Text conversations | n | 12B base | 12B SFT | E4B base | E4B SFT |
|---|---|---|---|---|---|
| `none` | 177 | 0.802 | 0.661 | 0.525 | **0.593** |
| `relabel` | 25 | 0.680 | 0.360 | 0.760 | 0.320 |
| `insert_handoff_turn` | 32 | 0.688 | 0.156 | 0.562 | 0.094 |
| `append_closing_pair` | 24 | 0.417 | 0.083 | 0.333 | 0.000 |
| All 81 needing a change | 81 | 0.605 | **0.198** | 0.556 | **0.136** |

- **E4B:** fine-tuning improved completion on the 177 compliant conversations;
  its whole text drop is in the 81 that are not.
- **12B:** the 81 account for about 60% of its lost completions. It also drops
  on the compliant 177 (0.802 → 0.661), which the rule mismatch does not
  explain. That part is open.

---

## 4. Relabel-only rescore: the relabel is invisible to the score

`remediate_task_a_states.py apply` with no ledger relabels the 25 and drops the
56 that need authored turns. It also drops three `none` rows
(`INS_PREMIUM_001`–`003`, the hand-added Thai insurance file) for an unrelated
shape rule, consecutive assistant prose turns. Those need no relabel and are
kept unchanged below, giving 202 conversations. `verify`: 0 violations. Of the
199 rows `apply` kept, 174 are byte-identical to the originals and 25 differ
only in `[STATE:]` labels and `ground_truth.state_sequence`.

Because the relabel does not change prose, tool calls or message count, each
run's logged replies are still a valid replay, so no model was re-run
(`.runs/eval_12b_template/rescore_relabel.py`):

| Run | Text: A, 258 original | B, 202 original keys | C, 202 relabelled keys | Blended A | Blended C |
|---|---|---|---|---|---|
| 12B base | 0.6964 (0.740) | 0.7113 (0.787) | 0.7113 (0.787) | 0.7098 | 0.7202 |
| 12B SFT | 0.6662 (0.516) | 0.6947 (0.624) | 0.6947 (0.624) | 0.7186 | 0.7385 |
| 12B SFT, patched template | 0.6743 (0.500) | 0.6941 (0.584) | 0.6941 (0.584) | 0.7230 | 0.7368 |
| E4B base | 0.6035 (0.535) | 0.6095 (0.554) | 0.6095 (0.554) | 0.6131 | 0.6174 |
| E4B SFT | 0.6315 (0.450) | 0.6602 (0.559) | 0.6602 (0.559) | 0.6678 | 0.6879 |

Text score, task completion in brackets; voice unchanged.

**B and C are identical for every run.** The relabel did apply: on the 25
relabelled conversations the ground-truth transitions change in all 25, and the
12B SFT per-turn transition accuracy moves 0.3631 → 0.4647. But the score's
state component takes the better of per-turn and sequence accuracy, and
sequence accuracy collapses repeated states, so moving an advance one turn
later leaves it unchanged (0.7987 both). Completion checks only the final state,
and tool F1 ignores labels.

So:
- **A relabel-only text set does not change the benchmark score.** Every gain
  from A to C comes from dropping the 56 conversations that need authored turns.
- **Those 56 are where the rule mismatch costs completion.** Under the stay rule
  they need extra turns; replay stops at the ground truth's length.
- **Dropping them is not neutral.** 44 of the 56 are L4–L5 or L2–L3, so the
  202-conversation set is easier.

A comparable fix therefore needs the authored inserts (`build_remediation_ledger.py`
on 56 conversations, 143 inserts: 87 assistant, 56 user), reviewed as test
content and frozen as a new, versioned text stratum beside the current one,
which stays frozen (R21). Every model, Gemini included, would need re-running.

---

## 5. The conversation-id collision

`evaluate_state_machine` built `{conversation_id: ground_truth}` and looked each
prediction up in it. Across the 508-conversation run, 250 voice ids repeat text
ids, so the last-loaded voice ground truth replaced the text one and every text
prediction was scored against a different conversation. `stochastic_map` was
keyed the same way.

**Affected:** the whole-run `metrics.state_metrics` block of every two-stratum
result JSON: state sequence accuracy, state transition accuracy and
invalid-transition rate. Also `metrics.weighted_workflow_score`, which uses
them, but nothing ranks on it.

**Not affected:** `quality_summary` (`quality`, `quality_text`,
`quality_voice`), which scores each stratum separately, and ids are unique
within a stratum. Also unaffected: task completion, recovery rate (every
terminal state is `TERMINAL`), tool metrics and chain propagation, which pair by
position.

Corrected values, from the rebuilt runs:

| Run | Sequence accuracy | Transition accuracy | Invalid transitions |
|---|---|---|---|
| E4B base | 0.5517 → **0.7884** | 0.2192 → **0.4102** | 0.4727 → **0.1517** |
| E4B SFT | 0.6125 → **0.8713** | 0.2613 → **0.4853** | 0.4224 → **0.0716** |
| 12B base | 0.6148 → **0.8680** | 0.2769 → **0.5073** | 0.4469 → **0.0933** |
| 12B SFT | 0.6476 → **0.9132** | 0.3845 → **0.6375** | 0.4139 → **0.0655** |
| 12B SFT, patched template | 0.6551 → **0.9256** | 0.3778 → **0.6677** | 0.4289 → **0.0708** |

Gemini's values cannot be corrected from its logs (section 1). The stored result
JSONs are left as they are.

**Fix:**
- `state_accuracy.evaluate_state_machine` raises `ValueError` on a duplicate
  ground-truth id instead of letting the last one win silently.
- New `agent_benchmark.build_state_machine_inputs(samples, predicted_messages)`
  keys each row as `"<row index>:<conversation_id>"`. `__main__` uses it, and
  stochastic trials attach to predictions by row position.

**Tests:**
- `tests/unit/test_benchmark_state_inputs.py` (7 tests): unique keys across
  strata sharing an id, each row scored against its own ground truth, raw-id
  pairing refused, terminal states from the sample, missing ids, length
  mismatch.
- `tests/unit/test_eval_metrics.py::TestEvaluateStateMachine::test_duplicate_ground_truth_ids_raise`.
- Unit suite: 1,864 passed, 17 failed. The same 17 fail on unmodified `main`;
  all construct TRL `DPOConfig`, `GRPOConfig` or `SFTConfig` objects and none
  touch evaluation.

---

## 6. Reproduce

```bash
# Rebuild and validate against stored results (CPU only)
python .runs/eval_12b_template/rebuild_bench.py .runs/eval_12b_template/bench_rebuild \
    base12=google_gemma-4-12B-it_auto sft12=sft_cat_a_12b_ckpt3168_auto

# Triage the text set, then build the relabel-only set
python scripts/remediate_task_a_states.py triage --input-dir data/output/benchmark/task_a \
    --report .runs/eval_12b_template/bench_remediation/text_triage.json
python scripts/remediate_task_a_states.py apply --input-dir data/output/benchmark/task_a \
    --output-dir .runs/eval_12b_template/bench_remediation/task_a_relabel_only --on-unrepairable drop

# Rescore, and corrected whole-run state metrics
python .runs/eval_12b_template/rescore_relabel.py \
    .runs/eval_12b_template/bench_remediation/task_a_relabel_only sft12=sft_cat_a_12b_ckpt3168_auto
python .runs/eval_12b_template/corrected_state_metrics.py
```

The `.runs/` scripts are not in git.
