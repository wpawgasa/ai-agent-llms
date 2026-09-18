# Phase 1 benchmark: tool results for calls the model never made

**Date:** 2026-09-17 · **Branch:** `fix/benchmark-tool-results` · **Code:** `eval/agent_benchmark.py::_replay_conversation`

## 1. Summary

The Phase 1 Task A benchmark replays each ground-truth conversation and asks
the model for every assistant turn. It inserted the ground-truth tool result
after every tool-calling turn **whether or not the model made the call**.

- In the **native** format the Gemma-4 chat template renders a tool result only
  under the assistant message whose `tool_calls` it answers. The template is
  right: a result with no call does not exist. But the replay carried on as if
  the model had seen it, and asked it to answer that result.
- In the **text** format (`--tool-turn-format text`) the result became a
  `[Tool result]: ` user turn, so the model **saw** the output of a call it
  never made. Models that miss calls got free information.

Two further defects made the native format score Gemma-4 too low for reasons
that were not the model's fault:

- **Copied ground-truth turns lost their tool calls.** Turns the replay cannot
  solicit (an opener, or a second consecutive assistant turn) are copied from
  ground truth. The corpus writes their calls as `<tool_call>` text, so they had
  no `tool_calls` field and the template dropped the result that followed.
- **Empty replies after a real call.** The Gemma-4 template renders a merged
  assistant message's text *after* the tool responses and then closes the model
  turn. A request that ends on that result carries no generation cue, and the
  untrained model replies with nothing.

## 2. Evidence

Every request of the finished runs was rebuilt offline from the run logs and
rendered with the served template
(`.runs/eval_textturns/replay_render_audit.py`, `classify_dropped.py`). All 508
conversations matched the logged call counts for every run.

### Ground-truth results the template did not render (v2 text + voice, 1,853 results the model was about to answer)

| Model | Not rendered |
|---|---|
| E4B untrained | 998 (54%) |
| E4B SFT ckpt-3168 | 608 (33%) |
| 12B untrained | 742 (40%) |
| 12B SFT ckpt-3168 | 567 (31%) |

### What preceded each unrendered result (old code, 2026-09-15 runs)

| Preceding message | E4B base | E4B SFT | 12B base | 12B SFT |
|---|---|---|---|---|
| Model reply, no tool call | 822 | 522 | 691 | 564 |
| Empty model reply | 155 | 16 | 62 | 0 |
| Model reply, call the replay could not parse | 23 | 80 | 0 | 12 |
| Copied ground-truth turn | 2 | 0 | 0 | 2 |

With the 2026-09-17 consecutive-turn skip (1b7b079) the replay copies ~327
ground-truth turns per run, and the copied-turn row grows to **320** per run.

### Empty replies

Untrained E4B: **490** empty replies of 6,046 calls (old code), **387** with the
2026-09-17 code. 405 of the old ones followed a tool result the model had
correctly called. Rendering one of these requests with the served Google
template (`ee0ef60`) ends in:

```
...<|tool_call>call:qualify_lead{...}<tool_call|><|tool_response>response:qualify_lead{...}<tool_response|>[STATE: QUALIFY_PROSPECT → QUALIFY_PROSPECT]
Thanks for providing those details, Mark. ...<turn|>
```

The model's own pre-call text is placed after the result and the turn is
closed; no `<|turn>model` follows. Sending the text and the call as two
assistant messages renders in the true order and leaves the turn open
(`.runs/eval_textturns/render_probe.py`, both the Google and the Unsloth
template). ChatML (Qwen2.5) renders the merged form correctly and the split form
as two separate assistant turns, so the split is Gemma-4 only.

### Scores under each variant (v2 text + voice, blended quality)

| Model | Native, old code | Native, 2026-09-17 code | Text, 2026-09-17 code |
|---|---|---|---|
| E4B untrained | 0.6182 | 0.6766 | 0.7944 |
| E4B SFT ckpt-3168 | 0.6881 | 0.7263 | 0.8486 |
| E4B SFT tool-result ckpt-3168 | — | — | 0.8466 |

None of these is a clean measurement: native under-scores Gemma-4 (dropped
results for copied turns, empty replies), text over-scores models that miss
calls (free results). Do not rank on any of them.

## 3. Fix

1. **A result is shown only for a call that was made.** The replay keeps the
   ids of the calls made by the latest assistant turn (model reply or copied
   ground-truth turn) and admits one ground-truth result per call. The list is
   reset at every assistant turn, so an earlier call never admits a later
   result. A result with no call is withheld (`tool_result_withheld_no_call`).
2. **The turn that answers a withheld result is scored as a miss.** It cannot be
   asked for — the context ends in the model's own reply — and copying it from
   ground truth would credit the model with a turn it could not produce. It is
   recorded as an empty prediction (`missed_turn_after_withheld_tool_result`).
   This is a scoring decision: a missed call costs the call turn and the turn
   that depends on its result.
3. **Copied ground-truth turns keep their calls.** In native format their
   `<tool_call>` text is lifted into structured `tool_calls`, so the result that
   follows renders.
4. **`--split-tool-call-content`** sends a reply holding text and a call as two
   assistant messages, text first. `run_exp_a_single.sh` turns it on
   automatically when `serving.tool_call_parser` is `gemma4`
   (`--split-tool-call-content auto|on|off`).

Result JSONs now record `split_tool_call_content` and
`tool_result_gating: calls_made_only`. Tests:
`tests/unit/test_replay_tool_result_gating.py`.

## 4. E4B re-measured under the fix (2026-09-17)

All three E4B models re-run in the native format with the gating and the
Gemma-4 split (`*_gated_textv2_auto.json`), v2 text + voice, 0 stochastic
trials:

| Metric | Untrained | SFT (no tool results in training) | SFT with tool results |
|---|---|---|---|
| Blended quality | 0.7260 | 0.7433 | **0.7999** |
| text / voice | 0.7205 / 0.7387 | 0.7391 / 0.7531 | 0.7844 / 0.8362 |
| state_sequence_accuracy | 0.869 | 0.871 | 0.934 |
| state_transition_accuracy | 0.509 | 0.657 | 0.705 |
| task_completion_rate | 0.730 | 0.581 | 0.785 |
| task_completion_rate_continuous | 0.423 | 0.520 | 0.726 |
| invalid_transition_rate | 0.103 | 0.038 | 0.034 |
| tool_call_f1 | 0.590 | 0.705 | 0.700 |
| argument_exact_match | 0.350 | 0.568 | 0.570 |
| chain_propagation_accuracy | 0.291 | 0.336 | 0.330 |
| full_workflow_success | 0.224 | 0.246 | 0.324 |
| empty replies / results withheld | 2 / 693 | 6 / 433 | 1 / 310 |

**Training with tool results is worth +0.0566 blended quality**, and the gain
is in workflow following, not tool calling: completion 0.581 → 0.785 and
continuous completion 0.520 → 0.726, while `tool_call_f1` and
`argument_exact_match` are flat. The model that saw tool results in training
also misses fewer calls, so fewer results are withheld (310 against 433).

Under the pre-fix text format the same two checkpoints scored 0.8486 and
0.8466 — the free results flattered the weaker model enough to hide the
difference entirely, which is why the first read of this comparison was "the
fix changes nothing".

Chain propagation is ~1 in 3 for all three models: the R23 argument-fidelity
gap is untouched by this fix.

## 5. Consequences

- **Every Task A benchmark result before this fix is not comparable** with
  results after it: all local models, both formats, and Gemini (the gating
  applies to the frontier path too).
- The text format is no longer inflated by free results, but it is still a
  format the untrained models were not trained on. Native with split is the
  primary format for Gemma-4.
- **Still open:** the pre-existing consecutive-turn skip (1b7b079) copies the
  ground-truth turn into the predictions, where it is scored as a perfect turn
  (~327 turns per run, nearly the same set for every model). It lifts every
  absolute score by a similar amount, so it barely reorders models.
