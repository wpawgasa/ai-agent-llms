# Phase 1 / Task A Metrics — Definitions and Interpretation

This document describes every metric computed by `eval/agent_benchmark.py` for the workflow-orchestrator benchmark, what each one actually measures, how it can fail, and how to read a result file.

It mirrors the code at `src/llm_workflow_agents/eval/{state_accuracy,tool_call_f1,tool_chain_propagation,composite_score}.py` as of the commits that added per-level evaluation. Targets in the tables are aspirational thresholds for a fully fine-tuned specialist (Phase 2 output), not for pre-trained frontier models.

---

## 1. Where the numbers come from

A benchmark run produces one JSON file under `results/exp_a/` with this shape:

```json
{
  "model": "...",
  "engine": "vllm" | "bifrost",
  "complexity_level": "L1" | ... | "L5" | "mixed",
  "num_samples": 40,
  "stochastic_trials": 2,
  "metrics": {
    "weighted_workflow_score": 0.553,
    "full_workflow_success":   0.000,
    "state_metrics":           { ... },
    "tool_metrics":            { ... },   // per-turn (strict, zip-aligned)
    "tool_metrics_conversation": { ... }, // per-conversation (lenient, set-aligned)
    "chain_metrics":           { ... },
    "latency_per_turn_avg_ms": 993.5,
    "ttft_avg_ms":             808.0
  }
}
```

The composite scoring is in `eval/composite_score.py`. The component metrics are produced by three modules described below.

---

## 2. State-machine metrics (`state_accuracy.py`)

Computed from the `[STATE: X → Y]` annotations the model emits. The harness replays the conversation turn-by-turn against the ground truth.

| Metric | Formula | Aggregation | Target | What it tells you |
|---|---|---|---|---|
| `state_transition_accuracy` | `correct_transitions / total_transitions` | per-turn, micro-averaged across all turns of all conversations | ≥0.85 | Strict per-turn correctness. Penalises any deviation, including legal alternative paths. |
| `state_sequence_accuracy` | LCS-recall of the *visited-state* sequence against GT | per-conversation, then averaged | (none, reporting) | Lenient version. Collapses self-loops, ignores ordering of equivalent transitions. Counts a conversation as correct if it covers the GT states in the right order, even if turn counts differ. |
| `task_completion_rate` | `1 if pred_terminal in gt_terminals else 0` | per-conversation, then averaged | ≥0.70 | Did the model end the conversation in *any* terminal state listed in the workflow graph? |
| `invalid_transition_rate` | `transitions_not_in_graph / total_transitions` | micro-averaged | ≤0.05 | Did the model invent edges that don't exist in the GT state machine? |
| `recovery_rate` | `recovered_errors / total_errors_seen` | micro-averaged | ≥0.60 | When a tool returned an error, did the model re-prompt / retry / branch correctly instead of cascading the failure? |
| `consistency_pass5` | `1 if all stochastic trials reached the same terminal else 0` | per-conversation, then averaged | ≥0.40 | Stability under temperature=0.7. Note: name says "5" but `stochastic_trials` is configurable; with `stochastic_trials=2` this is effectively pass^2. |

### Interpretation guide

- **`state_transition_accuracy` is the strictest metric in this group.** It zeros out on any single bad turn. It's normal for it to be much lower than `state_sequence_accuracy` — a 0.40 / 0.85 split means "the model gets the shape right but takes detours."
- **`state_sequence_accuracy`** is the more useful "did the model basically do the right thing?" signal.
- **`recovery_rate` is undefined when `total_errors_seen == 0`.** The code returns 0.0 in that case (`max(total_errors, 1)` denominator) — it does *not* mean the model failed; it means nothing went wrong to recover from. Read it together with the synthetic ~20% tool-error rate from data generation: if the recovery_rate is ~0 across many samples, the model is genuinely cascading failures.
- **`task_completion_rate` is lenient by design** — it accepts any GT-listed terminal. If a workflow has both `RESOLVED` and `ESCALATE_SUPERVISOR` as terminals, ending in either counts.

---

## 3. Tool-call metrics (`tool_call_f1.py`)

Computed by parsing `<tool_call>{JSON}</tool_call>` from `content` (and structured `tool_calls` if the API returned them — they get appended as text tags by the streaming wrapper). Two separate aggregations are computed per run:

- **`tool_metrics`** — strict per-turn alignment. `zip(pred.messages, gt.messages)` then compare.
- **`tool_metrics_conversation`** — set-based per-conversation. Collect every tool call in the trajectory and compare against the GT bag.

Both produce the same dataclass:

| Metric | Formula | Target | What it tells you |
|---|---|---|---|
| `tool_name_accuracy` | `correct_tool_names / total_tools_compared` | ≥0.90 | Did the model pick the right function? Names only. |
| `argument_exact_match` | `exact_dict_match / total_tools_compared` using `_deep_equals` | ≥0.75 | Did the arguments match the GT *exactly* — same keys, same string values, no extras? |
| `tool_call_f1` | BFCL-style AST F1: precision/recall over (name, normalised-args) sub-trees | ≥0.85 | More forgiving than `argument_exact_match` — accepts subtree matches and tolerates extra-but-valid arguments. |
| `hallucinated_tool_rate` | `predicted_tools_not_in_schema / total_predicted` | ≤0.03 | Did the model invent function names that aren't in the provided `tool_schemas`? |

### Per-turn vs conversation: why both?

- **Per-turn (`tool_metrics`)** is brittle: if the GT has 4 turns with tool calls and the model produces 3 (or 5), zip-alignment slides everything by one and most turns mismatch. Useful only when turn counts already line up — i.e. the model didn't collapse multi-turn negotiation.
- **Per-conversation (`tool_metrics_conversation`)** ignores turn ordering. As long as the right *set* of (name, args) tuples appeared somewhere, it scores. This is what `weighted_workflow_score` actually reads (via the `max(turn, conv)` rule in `composite_score.py`).

A common pattern: `tool_metrics.tool_call_f1 = 0.10` while `tool_metrics_conversation.tool_call_f1 = 0.45`. That gap is *not* model error — it's structural disagreement between the model's turn-collapsing and zip-based per-turn alignment.

### `argument_exact_match` is harsher than the name suggests

It calls `_deep_equals`, which requires:
- Same key set (no extra keys, no missing keys)
- Same value type for every key
- For strings: byte-exact equality (no case-fold, no whitespace strip)
- For numbers: numeric equality
- For nested dicts/lists: recursive `_deep_equals`

So `{"product": "premium"}` vs `{"product": "premium package"}` is a miss, and `{"customer_id": "12345"}` vs `{"customer_id": "12345", "reason": "upgrade"}` is also a miss (extra key on the prediction side). Frontier models in particular drift on optional-field inclusion. Read this metric alongside `tool_call_f1`, which is more forgiving.

---

## 4. Chain-propagation metrics (`tool_chain_propagation.py`)

Tests whether outputs from earlier tool calls correctly populate inputs of later ones — e.g. `lookup_user(email)` returns `{"user_id": "U123"}` and a later `update_subscription(user_id="U123", ...)` must reuse that exact value.

| Metric | Formula | Target | Notes |
|---|---|---|---|
| `chain_propagation_accuracy` | `correctly_propagated_chains / total_chains_in_GT` | ≥0.70 | Per-chain binary success. |
| `per_depth_accuracy` | Same, broken down by chain depth (2, 3, 4, …) | — | Useful for L4/L5 diagnostics. |
| `total_chains` | Count of GT chains in the run | — | If 0, the dataset has no multi-tool chains; the metric is N/A. |

`total_chains == 0` is normal for L1 (single-tool calls only) and means `chain_propagation_accuracy` should be ignored, not interpreted as failure.

---

## 5. Composite metrics (`composite_score.py`)

### `weighted_workflow_score` — the headline number

```
0.4 × max(state_transition_accuracy, state_sequence_accuracy)
+ 0.4 × max(tool_metrics.tool_call_f1, tool_metrics_conversation.tool_call_f1)
+ 0.2 × task_completion_rate
```

Range: [0, 1]. Target: ≥0.75 for a fine-tuned specialist.

This metric is **continuous** (degrades smoothly) and **lenient** (uses the better of the strict and lenient state/tool metrics). Use this for ranking across models and complexity levels.

### `full_workflow_success` — strict perfection check

A conversation counts as successful if and only if **all four** hold:
1. Every per-turn state transition exactly matches GT (`state_transition_accuracy == 1.0`)
2. Zero invalid transitions for the entire conversation
3. Reached a GT terminal state
4. Every tool-call turn scores `compute_ast_f1 == 1.0`

The result is the fraction of conversations that pass all four gates.

This is a **binary, all-or-nothing** metric. A 19-turn conversation that's 95% correct scores 0, the same as one that's 0% correct. **Do not use this as a ranking metric** — use it as a diagnostic answering "did the eval pipeline even produce a perfect trajectory once?" (For example, on Gemini-3.1-Flash-Lite-Preview run 2026-05-06, L1 hits 0.625 while L2–L5 are all 0 — confirming the metric works on simple flows but is mathematically pinned to zero on multi-turn complexity.)

The 0.55 target in `.claude/rules/05-eval.md` is set for **fine-tuned models only**. Treat `full_workflow_success` for pre-trained frontier as a perfection canary, not a quality measure.

---

## 6. Latency

| Metric | Definition |
|---|---|
| `latency_per_turn_avg_ms` | Mean wall-clock per assistant turn (request → final SSE chunk). |
| `latency_per_turn_median_ms` | Median across turns. Resilient to cold-start / streaming stalls. |
| `ttft_avg_ms` | Mean time to first token of the streamed response. |

Use median latency for cross-model comparison; mean is skewed by the first turn's TLS handshake and any retries. TTFT is a good proxy for perceived responsiveness in conversational UX.

---

## 7. How to read a result file

```bash
jq '.complexity_level, .num_samples, .metrics.weighted_workflow_score' \
  results/exp_a/<MODEL>_frontier_l3.json
```

Recommended interpretation order:

1. **`weighted_workflow_score`** — headline. If this is below 0.5 across L1, the model probably can't follow the format; below 0.5 only at L4–L5 means it scales poorly.
2. **`tool_metrics_conversation.tool_call_f1`** — the tool capability number. If this is high (>0.6) but per-turn is low (<0.3), the model's getting the right calls but in a different turn ordering than GT. That's usually a harness problem, not a model problem.
3. **`state_metrics.state_sequence_accuracy`** — does the model follow the right *shape*? If 1.0, it knows the workflow.
4. **`state_metrics.state_transition_accuracy`** — strict turn-level correctness. Read together with `state_sequence_accuracy`. A wide gap (say 0.30 / 0.95) means the model takes detours but arrives correctly.
5. **`task_completion_rate`** — lenient terminal-reached signal.
6. **`recovery_rate`** — only meaningful if there are tool errors in the trajectory; ~20% of GT tool calls return error payloads.
7. **`full_workflow_success`** — diagnostic only. Expect 0 on anything beyond L1.
8. **`chain_propagation_accuracy`** — only meaningful for L3+ where multi-tool chains exist. L1 always shows 0 because `total_chains == 0`.

---

## 8. Known caveats and structural pitfalls

These are *harness*-level issues that affect metric values without reflecting model quality. Be aware of them when reading numbers.

### `tool_metrics` (per-turn) inflates pessimism on multi-turn

`zip(pred.messages, gt.messages)` is the alignment strategy. When the model collapses or expands the conversation by even one turn, every subsequent comparison slides off-by-one. This is why `tool_metrics_conversation` is the metric that drives the composite score.

### `argument_exact_match` punishes paraphrase

`_deep_equals` requires byte-exact string equality. Real-world models pass `"premium"` when GT expects `"premium package"`, or include `"reason": "upgrade"` as an extra optional argument the GT didn't list. Both are "wrong" by this metric. Use `tool_call_f1` (subtree match) for the gentler reading.

### `consistency_pass5` is misnamed

The metric requires *all* `stochastic_trials` to agree on the terminal state. With `stochastic_trials=2` (the cost-controlled default for frontier models), it's pass^2, not pass^5. The name is historical from when the spec called for 5 trials.

### `chain_propagation_accuracy = 0` does not always mean failure

If `total_chains == 0` (e.g. L1 with single-tool conversations), the metric returns 0 because of the `max(total_chains, 1)` denominator. Always cross-reference `total_chains > 0` before interpreting this as a model deficiency.

### `state_transition_accuracy` is *micro*-averaged, not per-conversation

Total transitions across all conversations divided by total correct. A few long-and-bad conversations dominate over many short-and-good ones. If you want "what fraction of conversations had a perfect state trace?" use `state_sequence_accuracy` (which is per-conversation then macro-averaged).

### Latency depends heavily on the serving path

`vllm` runs go through a self-hosted server with controlled cold-start; `bifrost`/frontier runs go over the open internet to a third-party API. Don't compare ms-for-ms across engines — use TTFT to reason about perceived latency rather than raw wall-clock per turn.

---

## 9. Per-level expectations (calibration)

For a competent pre-trained 3B–35B model **without fine-tuning**, expect the headline `weighted_workflow_score` to roughly degrade as:

| Level | Domain | Tools / chains | Realistic frontier range |
|---|---|---|---|
| L1 | faq_lookup, single-tool | 1 tool / 0 chains | 0.70–0.85 |
| L2 | order_status_cancel | 2 tools / depth 1 | 0.55–0.75 |
| L3 | booking_payment | 4 tools / depth 2 | 0.40–0.60 |
| L4 | it_troubleshoot | 6 tools / depth 3 | 0.35–0.55 |
| L5 | multi_dept_workflow | 7 tools / depth 4 | 0.25–0.45 |

A fine-tuned specialist trained against the corresponding SFT corpus is expected to clear ≥0.75 on its target level; the 0.85 / 0.85 / 0.75 component targets in `.claude/rules/05-eval.md` are calibrated for that case.

If a frontier model lands well below the L1 range (say `weighted_workflow_score < 0.50` on L1), that's almost always either (a) a wrong `tool_call_parser` setting in the model config, (b) the model emitting tool calls in a non-OpenAI format the harness fails to parse, or (c) provider-side errors (HTTP 400s, timeouts) that the harness recovers from with empty turns. Inspect the run log for `http_error_during_call` warnings before concluding the model is bad.

---

## 10. Quick reference — all metrics in one table

| Group | Metric | Type | Target | Read first? |
|---|---|---|---|---|
| Composite | `weighted_workflow_score` | continuous | ≥0.75 | yes |
| Composite | `full_workflow_success` | binary per-conv | (diagnostic) | no — diagnostic only |
| State | `state_transition_accuracy` | continuous (turn) | ≥0.85 | secondary |
| State | `state_sequence_accuracy` | continuous (conv) | — | yes |
| State | `task_completion_rate` | continuous (conv) | ≥0.70 | yes |
| State | `invalid_transition_rate` | continuous | ≤0.05 | yes |
| State | `recovery_rate` | continuous | ≥0.60 | only if errors present |
| State | `consistency_pass5` | continuous (conv) | ≥0.40 | for stability check |
| Tool (turn) | `tool_call_f1` | continuous | ≥0.85 | secondary |
| Tool (turn) | `argument_exact_match` | continuous | ≥0.75 | strict-only diagnostic |
| Tool (turn) | `tool_name_accuracy` | continuous | ≥0.90 | yes |
| Tool (turn) | `hallucinated_tool_rate` | continuous | ≤0.03 | yes |
| Tool (conv) | `tool_call_f1` | continuous | ≥0.85 | yes — drives composite |
| Tool (conv) | other fields | same as turn | — | secondary |
| Chain | `chain_propagation_accuracy` | continuous | ≥0.70 | only for L3+ |
| Latency | `latency_per_turn_median_ms` | ms | ≤2000 (L1–L3), ≤5000 (L4–L5) | yes |
| Latency | `ttft_avg_ms` | ms | — | yes |
