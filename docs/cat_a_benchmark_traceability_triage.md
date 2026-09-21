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

## Repair pilot — mechanical stage (2026-09-21)

**Branch**: `feature/task-a-benchmark-repair-pilot`

Three repairs, planned once into a ledger and replayed deterministically — the
R25 pattern — so the repaired strata never depend on code that changes later.

### What was built

- `system_prompt.render_session_context` and `SESSION_CONTEXT_HEADER`. The
  served prompt states a sample's `session_context` after the tool schemas and
  before the rules. A sample without one renders byte-identically; the
  committed fixture test on real rows still passes.
- `src/llm_workflow_agents/data/benchmark_repair.py` — plan and apply for each
  repair:
  - **Identifier remap.** Each simple entity identifier (`CUST-882`) gets a
    fresh value of the same shape that appears nowhere in the training corpus
    or the benchmark. Skipped: general-knowledge tokens (`AES-256`),
    identifiers with structure (`PLAN_50GB`, `CARD-ENDING-4455`), and any whose
    digits are also spoken on their own ("the one ending 882").
  - **Stay merges.** A run of self-loop turns in one state ending in a tool
    call becomes one turn. Advance-then-stay pairs are never merged. Runs that
    touch a barge-in turn are never merged.
  - **Session context.** Each confident unsourced tool argument becomes a
    session-context fact.
  - Planning order is remap, merge, then session context: merging moves prose
    into the calling turn, and a calling turn cannot source its own value.
- `scripts/repair_task_a_benchmark.py plan|apply`. `plan` is the only step that
  is seeded or reads the training corpus; it also drops any merge that would
  add a format violation. `apply` replays the ledger and verifies every row: no
  new format violations (shape, continuity, tool stay, voice), ground truth
  still aligned with the turns, no confident unsourced argument left, no
  remapped value surviving.

```bash
python scripts/repair_task_a_benchmark.py plan \
    --input-dir data/output/benchmark/task_a_v2 \
    --input-dir data/output/benchmark/task_a_voice \
    --reference data/output/sft/task_a_splits \
    --ledger data/interim/task_a_benchmark_repair_ledger/ledger.json
python scripts/repair_task_a_benchmark.py apply \
    --stratum data/output/benchmark/task_a_v2 data/output/benchmark/task_a_v3 \
    --stratum data/output/benchmark/task_a_voice data/output/benchmark/task_a_voice_v2 \
    --ledger data/interim/task_a_benchmark_repair_ledger/ledger.json
```

### Result

Plan (seed 20260921): 420 of 508 rows changed; 943 identifiers remapped; 88
skipped (70 structured, 9 digits spoken elsewhere, 6 embedded in a longer
token, 3 general knowledge); 59 merge runs, none dropped for violations; 35
session-context values. Apply: every row passes verification, and a second
apply is byte-identical.

**Two defects caught and fixed before commit:**

- The first plan remapped `AES-256` to `AES-616`: the general-knowledge
  allowlist was applied to the facts check but not to the remap. The remap now
  skips it (`general_knowledge`).
- Five conversations came out naming one thing two ways. `INV-5544` was
  remapped where it stood alone, while `INV-5544-SETUP` was skipped as a
  structured identifier. The planner and the rewriter disagreed about token
  boundaries, and verification used the rewriter's rule, so it could not see
  the problem. The planner now skips an identifier that also occurs inside a
  longer token (`embedded_in_longer_token`), and verification checks for the
  old value as a plain substring. Both have regression tests.

## Repair pilot — authored stage (2026-09-21)

The 31 confident invented facts were resolved one by one in
`data/interim/task_a_benchmark_repair_ledger/facts.json` (32 edits, approved by
the user), replayed after the mechanical ledger by `apply --facts-ledger`.

**26 of the 31 were hidden unknowable tool arguments.** The teacher invented a
value in prose ("I recommend plan PLAN-UNL-50") and the gold call then used it.
The argument check counted those calls as sourced, because the prose had said
the value first. Stating the value in session context fixes the prose and the
call together.

| Action | Count | Examples |
|---|---|---|
| Session context | 27 | 13 recommended upgrade plans, 2 survey reward codes, 12 on-file identifiers (contract, claim, procedure, prescription, pharmacy, interaction, fix) |
| Rewrite to a format mask | 2 | the agent's "example" `RX566609002` was exactly the ID the user read out next — the teacher leaked the answer |
| Remove a sentence | 1 | the agent said "My customer ID is CID-1274." — a role slip; the ID moved to session context |
| Accept as a format example | 2 | "e.g. RX + 6 digits", after which the user gives a different, real ID |

`apply_fact_edits` refuses to apply when a fact is not where the ledger says it
is (fact values are post-remap, so drift would otherwise be silent), when a
session-context field already holds a different value, or when a rewrite
target does not occur exactly once. With `--facts-ledger`, verification also
requires that no confident invented fact remains except the accepted examples.

The session-context header now reads "use these values where a tool call or the
conversation needs them": a reward code is told to the customer, not passed to
a tool.

### Final state

| | v2 + voice | v3 + voice v2 |
|---|---|---|
| Confident unsourced tool arguments | 35 | **0** |
| Confident invented facts | 33 | **2** (both accepted format examples) |
| Mergeable stay+stay pairs | 59 | **0** |
| Identifier occurrences reused across rows | 31.1% | **1.4%** |
| Rows sharing an identifier with training | 362 | **27** |

Frozen as DVC stage `task_a_benchmark_repair_v3` (guard:
`tests/unit/test_dvc_benchmark_repair_v3_stage.py`); the ledgers are tracked by
`data/interim/task_a_benchmark_repair_ledger.dvc`. Run the benchmark as
`--data data/output/benchmark/task_a_v3 --data data/output/benchmark/task_a_voice_v2`.

### Left as they are, by decision

- **19 multi-tool state visits and 483 multi-tool state offers.** Splitting a
  state inside an existing conversation needs an advancing turn the
  conversation does not contain, and converting the offers to router states
  would rewrite the conversations. They make a turn ambiguous; they do not make
  anything unscorable. One tool per state is a rule for future generation.
- **Advance-then-stay pairs** stay two turns (decided 2026-09-21).

### Next

Every model must be re-scored on the new strata — the scores are a separate
scale from v2. Four local models in two formats and both Gemini runs.

## v4 — replacing the multi-tool conversations (2026-09-21)

**Branch**: `feature/single-tool-states-and-benchmark-v4`

The 19 v3 conversations that call two different tools in one state (18 text,
1 voice) could not be repaired in place, so they were replaced by 20 newly
generated conversations — 19 matched one-to-one, plus one extra English L3
sales conversation.

### Generator changes

- **One tool per state** — `generate_workflow_dataset(single_tool_states=True)`.
  Each of the 25 multi-tool states in the registry declares `tool_mode`:
  6 `sequence` (every tool, in order: one state per tool, chained) and 19
  `choice` (the customer's request decides: a text-only router listing every
  option, one state per tool). Read off the instructions' "and"/"or".
  `sequence_order` fixes the order where the listed order is unnatural, without
  changing the listed tools.
- **Routers** — the subgraph keeps every route of an included router (also one
  pulled in through an optional branch), and each branch continues as the spine
  branch does.
- **Pinned routes** — `required_states=(...)` attaches a state even behind an
  optional branch, states it to the teacher as a required route, and makes the
  repair loop reject a draft that skips it. Needed because teacher
  conversations do not follow the walked path: the teacher writes its own
  route, and never took banking's optional fraud branch in a service call.
- **Teacher rules** — tool arguments must come from the user or an earlier tool
  result (ask first if missing); no invented identifiers, codes, amounts or
  dates in prose.
- **Orphan tool results** — `find_orphan_tool_results` flags a tool result after
  a turn that only announces the call. Found in the first generated
  conversation; now part of the generator's repair loop, the triage report and
  the gate.

All of it is off by default. With the flags off, placeholder generation over
five domains hashes identically to `main`.

### Generation and gate

`scripts/generate_benchmark_replacements.py` generates each slot from its
original's stratum and teacher. A candidate is accepted only after the v3
repairs and a gate: zero traceability findings, zero format violations, at
least one tool call, and **the part of the workflow the original was about** —
the last step of the split sequence, or a branch of the split router.

The first pass used a looser gate (enter the original state) and filled 18 of
20. Reading them showed that three L2 sales conversations quoted but never sent
the proposal, and one ecommerce conversation took a side exit instead of its
router, so the gate was tightened and those four regenerated, together with the
two fraud slots under pinned routes. Final: 20 of 20, each on its first attempt
in the final run; about 36 attempts and 108 candidates across both runs.

**One exception, by decision:** the L1 survey replacement collects CSAT only.
Split into CSAT then NPS, the two-rating survey no longer fits L1's 3-4 state
budget. Recorded in the manifest as `accepted_exception`.

### Result

| | v3 | v4 |
|---|---|---|
| Conversations | 508 | 509 (259 text + 250 voice) |
| State visits using two tools | 19 | **0** |
| Confident unsourced arguments | 0 | 0 |
| Confident invented facts | 2 (accepted) | 2 (same) |
| Orphan tool results | 4 | 4 (pre-existing; none in the replacements) |
| Rows sharing an identifier with training | 27 | 25 |

The replacements alone are clean on every check, including zero identifiers
shared with training. Frozen as `task_a_benchmark_v4` (outs `task_a_v4`,
`task_a_voice_v3`); the replacements, the manifest and every rejected
candidate's reasons are DVC-tracked in
`data/interim/task_a_benchmark_v4_replacements`.

### Left open

- The 457 states that *offer* two tools in untouched conversations stay as they
  are; one tool per state is enforced for future generation.
- The 4 pre-existing orphan tool results, 3 of them in the hand-added
  `l3_insurance_premium_payment_th` file.
- Every model must be re-scored on v4; it is a separate scale from v3.
