# Task A SFT Data — Quality Review & Fine-Tuning Suitability

**Scope:** `data/output/sft/task_a/`
**Reviewed:** 2026-04-28
**Files:** 113 JSONL files, 4,450 conversations, ~42,699 assistant turns
**Schema validity:** 100%

## Format

Every row carries 13 top-level keys, all populated:

- `conversation_id`, `complexity_level`, `domain`
- `num_states`, `num_tools`, `chain_depth`
- `workflow_graph` — `states`, `state_details`, `transitions`, `initial`, `terminal`
- `workflow_script` — natural-language flow description
- `tool_schemas` — OpenAI function-calling JSON
- `messages` — system / user / assistant / tool turns
- `user_behavior`, `language`
- `ground_truth` — `state_sequence`, `tool_calls`, `tool_chain_dependencies`, `terminal_state`

Assistant turns use the spec'd annotations:

- `[STATE: X → Y]` state-transition tag
- `<tool_call>{"name":..., "arguments":{...}}</tool_call>` Hermes-style tool-call tag
- `role: "tool"` returns realistic JSON, including ~20% error payloads (`{"error":"Service temporarily unavailable"}`)

## Coverage

| Axis | Distribution |
|---|---|
| Levels | L1=1010, L2=960, L3=860, L4=810, L5=810 |
| Languages | en=1550, th=1450, code_switch=1450 |
| Domains | 18 (account_management, banking, healthcare, telecom, emergency, insurance, …) |
| Behaviors | cooperative 2302 (51.7%), adversarial_probing 1118 (25.1%), digressing 525 (11.8%), invalid_tool_inputs 505 (11.3%) |
| Complexity scaling | L1: avg 3.6 states / 1 tool / chain_depth=0 → L5: avg 25.6 states / 7 tools / chain_depth=4 |
| Conversation length | avg 20.0 turns, max 106 |

### Domain breakdown

| Domain | Count | % |
|---|---|---|
| emergency | 424 | 9.5% |
| billing_payments | 297 | 6.7% |
| account_management | 296 | 6.7% |
| telecom | 279 | 6.3% |
| travel | 279 | 6.3% |
| ecommerce | 263 | 5.9% |
| product_info | 258 | 5.8% |
| order_management | 244 | 5.5% |
| utilities | 243 | 5.5% |
| sales | 235 | 5.3% |
| healthcare | 221 | 5.0% |
| government | 219 | 4.9% |
| complaints | 211 | 4.7% |
| technical_support | 209 | 4.7% |
| surveys | 205 | 4.6% |
| banking | 204 | 4.6% |
| scheduling | 200 | 4.5% |
| insurance | 163 | 3.7% |

`emergency` is slightly overrepresented (~9.5%) due to generation batching; all other domains fall in the 3.7–6.7% range.

## Quality Issues

| Issue | Count | % | Severity |
|---|---|---|---|
| `[STATE:]` missing on assistant turn (no `<tool_call>` either) | 135 | 0.3% of asst turns | Low — rare; relax `format_compliance_check` to accept tool-only turns or patch these |
| `terminal_state == ""` / `null` (didn't reach terminal) | 481 | 10.8% of convs | Medium — concentrated in L4/L5 + adversarial; GRPO terminal-bonus must handle empty terminal |
| Truncated rows (no non-system turns) | 5 | 0.11% | Trivial — filter |
| `role: "tool"` content starts with `<tool_call>` (role confusion) | 264 messages in 66 convs | 1.5% of convs | **Real bug** — gpt-5-4-nano early-run artifact; filter `role:"tool"` messages whose content begins with `<tool_call>` |
| Behavior mix vs spec (60/15/10/15) | actual 51.7/25.1/11.8/11.3 | — | Adversarial ~2× over-target — intentional (see note below) |
| Per-level volume vs spec target (200/level) | 810–1010/level | — | 4× above spec target; more data is strictly beneficial |

**Note on behavior mix:** The 51.7% cooperative / 48.3% non-cooperative split (vs the spec'd 60/15/10/15) results from blending the `adversarial` generation preset (45% cooperative) with `cooperative_only` runs. This is intentional for SFT: the extra adversarial-probing exposure teaches error-recovery behavior that is disproportionately hard and consequential. The cooperative anchor (51.7%) is still strong enough to prevent an over-defensive policy.

## Verdict — Suitable for SFT on Workflow Automation + Tool Calling

### Strengths for the stated objective

- **Reward alignment.** Schema directly maps onto `eval/state_accuracy.py`, `eval/tool_call_f1.py`, and `training/rewards/reward_business_logic.py`. The `ground_truth` fields correspond 1:1 to reward components (state transitions, AST tool match, chain propagation, terminal reach).
- **Multi-turn structure.** 20-turn average trains sustained context and state tracking, not just single-step tool selection.
- **Curriculum.** L1→L5 progression supports difficulty scheduling for SFT.
- **Multilingual.** en / th / code_switch coverage matches Thai-deployment targets while preserving English tool/state token vocabulary inside the annotations.
- **Robustness signal.** Realistic tool errors + adversarial users build error-recovery behavior, exactly what `error_recovery_rate` in `tool_call_f1.py` measures.
- **Parser compatibility.** Hermes-style `<tool_call>` tags work for `qwen3_coder`, `hermes`, `mistral`, and `gemma` parsers (per `.claude/rules/01-configs.md`), so the same dataset trains all Cat A candidates.

### Recommended Pre-Training Cleanup Pass

1. **Drop pipeline failures** — 5 truncated rows + 264 role-confused tool messages (in 66 conversations). Filter any `role:"tool"` message whose content begins with `<tool_call>`.
2. **Handle empty `terminal_state`** — either drop or repair the 481 cases (10.8%) depending on whether the model should learn "graceful timeout" behavior. Otherwise the GRPO terminal-bonus signal is mis-shaped on those rows.
3. **(Optional) Format-compliance consistency** — add `[STATE:]` to the 135 tool-only assistant turns, or relax `format_compliance_check` in `training/reward_utils.py` to accept tool-only turns as compliant.
4. **(Optional) Domain rebalancing** — `insurance` is slightly under-represented (3.7%) vs `emergency` (9.5%). For domain-agnostic training this is acceptable; resample if domain-balance is required.

After this cleanup pass, the dataset is appropriate for the Phase 2 SFT stage of the project.

## Reproducibility

The numbers above were produced by parsing all 113 `*.jsonl` files in `data/output/sft/task_a/`, validating the 13-key schema, regex-extracting `[STATE:]` and `<tool_call>` annotations from assistant content, and JSON-parsing tool-role content. Behavior counts, language counts, and per-level totals come from the corresponding top-level fields. Terminal-state alignment was checked against `ground_truth.terminal_state`. Role-confusion detection matched `role:"tool"` messages whose `content` starts with `<tool_call>` after stripping whitespace.
