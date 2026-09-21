# Task A benchmark: traceability triage

**Date**: 2026-09-21
**Branch**: `feature/task-a-data-quality-checkers`
**Risk register**: CLAUDE.md R28

## Why this exists

A benchmark conversation is only fair if the model could have produced every
value its gold turns contain. Investigating tool-call omissions (R27 follow-up)
showed that it often could not:

- Conversation `L1_006` scores the model against `collect_csat(interaction_id="INT-5541")`.
  The value appears nowhere in the conversation or the rendered system prompt.
  The model asked for it, as rule 7 of the served prompt tells it to, and was
  still scored wrong.
- Gold assistant prose states specific facts no tool returned — a discount code
  `PREM20`, a 20% discount. The fine-tuned 12B then invented a 15% discount of
  its own. Fine-tuning raised the rate at which the 12B emits identifiers absent
  from its whole context from **0.3% to 3.0%**; most invented values
  (`RES-99887`, `INT-99821`) occur in no training row, so the model learned the
  *shape* of an identifier, not a list of them.
- **72.7%** of identifier occurrences in the SFT corpus reuse a value that
  appears in more than one conversation (`TX-101` in 196 rows).

This work builds the checks and runs them over the benchmark. **It repairs
nothing.** The report is the work list for the repair steps.

## What was built

### `src/llm_workflow_agents/data/source_traceability.py`

Stdlib-only, like `state_convention.py`, and reuses its `parse_assistant_turns`.

| Check | Finds | Confidence |
|---|---|---|
| `find_unsourced_argument_values` | A required tool argument whose value appears nowhere before the call — not in the user's words, an earlier tool result, the rendered prompt, or the session context | `confident` if the value is identifier-shaped **or** the argument is named like an identifier (`*_id`, `*_code`, `*_number`); otherwise `needs_review` |
| `find_unsourced_facts` | An identifier, thousands-separated amount or percentage stated first in assistant prose | `confident` for identifiers; `needs_review` for amounts and percentages, which may be arithmetic on sourced values |
| `find_multi_tool_states` | States that offer more than one tool, and states where a conversation uses more than one | — |
| `collect_identifiers` / `find_identifier_reuse` | Identifier values shared across rows; values in the rendered prompt (state names, tool names, enum values) are excluded as vocabulary | — |
| `find_mergeable_stay_pairs` | Two adjacent assistant turns that both stay in the same state | — |

Matching is normalization-aware: case and separators are ignored
(`INT-5541` matches `int 5541`), and numbers must match a whole number
(`5` is not sourced by `15,000`; `1000000` is sourced by `1,000,000`).

**Deliberately not flagged:** an advancing prose turn followed by a tool turn
(`[W → X]` then `[X → X]`). That pair is how the stay convention enters a tool
state, and the decision on 2026-09-21 was to keep it as two turns.

### `scripts/triage_task_a_quality.py`

Runs every check and writes a per-row defect list plus a summary. Rows are keyed
by `<file>:<line>`, never by `conversation_id`, which repeats across the text and
voice strata (R25). The prompt each row is checked against is the one the model
is served, rebuilt with `build_enriched_system_prompt` — the stored system
message is the authored prompt, not what the model saw.

```bash
python scripts/triage_task_a_quality.py \
    --data data/output/benchmark/task_a_v2 \
    --data data/output/benchmark/task_a_voice \
    --reference data/output/sft/task_a_splits \
    --out runs/audit/triage_benchmark_v2.json
```

## Results on the benchmark (508 conversations)

| Finding | Count |
|---|---|
| Unsourced tool arguments, confident | **35** |
| Unsourced tool arguments, needs review | 160 |
| Unsourced facts in prose, confident | **33** |
| Unsourced facts in prose, needs review | 104 |
| Conversations with at least one confident finding | **52** (10.2%) |
| States offering more than one tool | 483 |
| State visits using more than one tool | 19 |
| Mergeable stay+stay pairs | 59 |
| Identifier values reused across benchmark rows | 109 (31.1% of occurrences) |
| Benchmark rows sharing an identifier with the SFT corpus | **362** (71.3%) |

### What `needs_review` actually contains

A breakdown of the 168 `needs_review` arguments from the first run, before the
argument-name rule moved 8 of them to `confident`:

| Kind | Count | Example | Verdict |
|---|---|---|---|
| Short English | 60 | `origin='Bangkok'` from a Thai request | mostly translation |
| Free text, 4+ words | 46 | `description='website kept crashing during checkout process'` | paraphrase by design |
| ISO date | 29 | `departure_date='2024-11-25'` | normalization; the year may be invented |
| Enum-like | 13 | `category='technical'` | mapping onto a schema enum |
| Thai text | 11 | `amount='หนึ่งร้อยบาท'` | paraphrase |
| Number | 9 | `coverage_amount='1000000'` | normalization or invention |

So `needs_review` is mostly **not** defects. The earlier estimate of 11.9% of
calls being unobtainable came from a plain substring check and was inflated by
exactly these cases. The confident set is the defect list.

### False positives found and fixed during the run

- `AUTHENTICATE_2FA` is a state name. It is identifier-shaped and inflated
  "shared with training" by 51 rows. Identifiers that appear in the rendered
  prompt are now excluded as vocabulary.
- `AES-256` is an encryption standard. It is in `DEFAULT_FACT_ALLOWLIST`.
- `customer_id='C-8821'` was `needs_review` only because a one-letter prefix is
  not identifier-shaped. Arguments named like identifiers are now `confident`.

## Known limits

- Plain digit runs in prose (`15000 บาท`) are not checked, to keep years, times
  and scores out of the report. Thousands-separated amounts are checked.
- Compact matching can over-match across word boundaries (`int5541` inside a
  longer run). This errs toward calling a value sourced, so it can hide a
  defect but not invent one.
- `DEFAULT_FACT_ALLOWLIST` is short and hand-kept. Extend it when a real
  general-knowledge token shows up as `confident`.

## Testing

- `tests/unit/test_source_traceability.py` — 34 tests
- `tests/unit/test_triage_task_a_quality.py` — 6 tests
- Coverage of `source_traceability.py`: 90%
- Full unit suite: 2,042 passed, 2 skipped

## Next steps

1. Repair the mechanical classes on the benchmark: re-randomize identifiers
   per conversation (and away from training values), move confident unsourced
   argument values into a session-context block, split the 19 multi-tool state
   visits, merge the 59 stay+stay pairs.
2. Session context needs a rendering path in `build_enriched_system_prompt`
   that emits nothing when the field is absent, so existing rows stay
   byte-identical (`tests/fixtures/text_prompt_baseline.json`).
3. The 33 confident invented facts need a per-conversation edit, ledger-driven
   as in R25.
4. Freeze the result as a new stratum and re-score every model, both Gemini
   runs included.
