# Task A `task-a-sft-v2`: tool-call state convention — design

**Date:** 2026-07-31
**Status:** Approved (plan-mode, user-confirmed decisions D1–D5)
**Supersedes nothing; extends** `docs/cat_a_state_annotation_convention_review.md`

## Context

Cat A fine-tuning is stalled on state-transition accuracy, not tool emission. `docs/cat_a_state_annotation_convention_review.md` (commits `993adb7` → `f91b728`) pins the cause: the corpus teaches an **unstated and inconsistent** convention for what state a tool-calling turn should annotate. Gold expects a self-loop on 41.6% of turns; ckpt-500 emits one on 4.8%; accuracy on stay+tool-expected turns is 0.055. Three cheap fixes were measured and are null:

- Perfect rule application caps composite at 0.7609 (§6.1).
- Prompt-only `STAY_RULE` injection: self-loop emission 4.84%→7.87%, Wilcoxon p=0.917 (§6.5).
- 3.5× epochs (ckpt-1770): composite 0.6579, p=0.513 (§6.6).

§6.3 concludes the corpus itself must change. This spec designs that change plus a generator redesign so the defect cannot recur.

**Target convention:**
1. A turn emitting `<tool_call>` annotates `[STATE: X → X]` (stay).
2. A `role:"tool"` message returns the result.
3. On success, the *next* assistant turn advances: `[STATE: X → Y]`.
4. On error, the next turn stays `X→X` and may retry the same tool.
5. After N failed attempts, stop retrying and take a fallback path instead of continuing to retry.

## Measured baseline (`data/output/sft/task_a`, `task-a-sft-v1`, 5,549 conversations)

64,964 assistant turns, 20,433 tool-call turns:

| | count | share |
|---|---|---|
| tool-call turns self-looping `X→X` (conformant) | 16,100 | 78.8% |
| tool-call turns forward-annotated `X→Y` (defect) | 4,333 | 21.2% |
| tool-call turns with no annotation | 0 | 0% |

Per conversation: 3,476 (62.6%) already conformant, 1,143 (20.6%) deterministically repairable with zero authored text, 930 (16.8%) need 1–2 short authored messages to stay legal.

Error/retry reality: 18.8% of tool results are errors; retry streaks are `{1 attempt: 3,784, 2: 26}`; only 10 of 5,549 conversations escalate after an error. **There is no retry-exhaustion arc anywhere in the corpus** — requirement 5 of the convention has zero corpus support and cannot be repaired into existing rows, only generated fresh.

Domain registry constraints (`domain_registry.py`, 18 domains): every domain has exactly one terminal, literally `"TERMINAL"`; 8/18 domains have no escalation-like state; 10/18 have no `tool_error` edge; `validate_domain()` enforces exactly one spine successor per non-terminal state and full reachability to a terminal.

## Decisions (user-confirmed)

- **D1 (the 930 authoring cases):** a `claude -p` agent authors the missing prose into a reviewable decision ledger; a deterministic 3-layer gate accepts or drops each entry. Rejected alternatives: drop outright (loses 16.8%, skews curriculum) or drop-and-backfill from the generator (cleaner provenance but needs a partial teacher-API regen).
- **D2/D5 (prompt policy):** fix the wrong worked example in `FORMAT_RULES` rule 2, promote the existing (but currently opt-in) `STAY_RULE` to default-on via an inverted `TASK_A_STAY_RULE=0` opt-out (preserves ckpt-500/1770 byte-identical comparability), and rebuild system prompts on v2 rows so the corpus states the rule it now demonstrates.
- **D3 (retry-arc synthesis):** do **not** retrofit retry-exhaustion into v1 rows — there is nothing there to repair. Teach the fixed generator the arc and generate a dedicated 500–800-conversation slice (L3–L5, all language legs) merged into v2.
- **D4 (fallback target):** per-level retry budget (`ComplexitySpec.retry_budget`) with an in-state "handoff" behavior when the graph has no `tool_error` edge, rather than adding a shared `HANDOFF` state to `domain_registry.py`. Avoids invalidating all 18 canonical graphs, Task C graph-pair targets, and `task_a_benchmark`.

## Core algorithm: whole-trajectory requeue

Per-turn relabelling is unsafe — defects come in runs (957 defective tool turns are themselves followed by another tool turn). The repair unit is the whole assistant-turn trajectory. Preserve every tool call's `from`-state exactly (making tool-placement legality invariant *by construction*, not merely re-checked), force tool turns to `X→X`, and push each displaced advance onto a FIFO queue drained by the next prose turn:

```
cur = labels[0].from
for each assistant turn in order:
    if turn has <tool_call>:
        require cur == turn.from          # else: stacked tool -> split or insert
        emit (cur, cur)
        if turn.to != turn.from: pending.append(turn.to)
    else:
        if pending: emit (cur, pending.popleft()); cur = emitted.to
                    if turn.to != turn.from: pending.append(turn.to)
        else:       emit (cur, turn.to);          cur = turn.to
require pending == []                     # else: tail deficit -> append closing pair
require every emitted (a,b) with a != b is a declared edge
require last emitted .to in terminals
```

`[STATE:]` marker text, `annotations.state_transition`, and `ground_truth.state_sequence` are all written from the same `emit` list, in lockstep — this is what keeps `quality_profiler`'s exact list-equality check (defect #7) satisfied.

### Move ladder (ranked by cost; measured yield)

| # | Move | Convs | Authored text |
|---|---|---|---|
| 1 | `relabel` — queue drains cleanly | 608 | none |
| 2 | `split_fused_tool_turn` — `[STATE: X→Y]\nprose\n<tool_call>` → two turns | +535 | none |
| 3 | `pullback_fuse` — move `<tool_call>` onto the preceding `X→X` prose turn | measure first | none |
| 4 | `insert_handoff_turn` — stacked bare tool turns; insert one assistant message | 443 | 1 msg |
| 5 | `append_closing_pair` — tail deficit; append `user` ack + terminal `assistant` | 599 | 2 msgs |
| 6 | `drop` — post-gate failure or agent refusal | residual | — |

Move 2 is legal because `find_shape_violations` (`_workflow_script.py:262-270`, verified by direct read) permits consecutive assistant turns when the second, after marker-stripping, starts with `<tool_call>`. Move 5 must prepend a `user` turn — 574 of 599 deficits end on an assistant turn, so appending an assistant alone trips the consecutive-prose rule.

**Rejected moves:** marker-only inserted turns (0 of 64,964 corpus turns are marker-only — would teach empty-content turns); blanket 2-turn relabel for every defect (unsafe on the 650+307 turns whose successor also calls a tool); skip-edges (0 of the tail deficits have a declared shortcut); terminating one state short (adds ~600 non-terminal rows, trips continuity).

## Invariants any repair must preserve (all independently re-checked post-repair)

| Invariant | Enforced at |
|---|---|
| Exactly one `[STATE:]` marker per assistant turn, at content start | `_workflow_script.py::find_continuity_violations` |
| Chain: `turn[n].to == turn[n+1].from`; first `from` = initial; last `to` ∈ terminals | same |
| Self-loops `X→X` always legal | `generate_workflows.py:69`; contract text |
| Consecutive assistant turns only if the second is pure-tool-call | `_workflow_script.py::find_shape_violations` |
| Tools callable only from the state listing them; calls attributed to the turn's `from` | `_workflow_script.py::find_tool_placement_violations` |
| `ground_truth.state_sequence` == message markers, exactly | `quality_profiler.py` hard defect #7 |
| Inline content markers override `annotations` | `generate_workflows.py::_backfill_annotations` |
| `_TEACHER_SYSTEM_PROMPT` keeps "ALLOWED TRANSITIONS" / "TOOL PERMISSIONS PER STATE" / "never invent a transition" | `tests/unit/test_teacher_prompt_contract.py` |
| Deterministic regardless of `max_workers` | `tests/unit/test_data_generation.py::test_concurrent_output_matches_serial` |

## Architecture

### Part A — Remediation (new corpus: `data-output/sft/task_a_remediated`, tagged `task-a-sft-v2`)

- `src/llm_workflow_agents/data/state_convention.py` — `TurnLabel`, `parse_assistant_turns`, `find_tool_stay_violations`. Single gate reused by the generator repair loop, `quality_profiler`, `data_validator`, the remediation post-gate, and the generation-loop qualification bar.
- `src/llm_workflow_agents/data/state_convention_repair.py` — `plan_repair`, `apply_plan`, `verify_repaired`, `rederive_ground_truth`. Pure; never mutates input; `rederive_ground_truth` is a byte no-op on already-conformant rows.
- `scripts/remediate_task_a_states.py` — `triage` / `apply` / `verify` / `diff` CLI. Deterministic, no network.
- `scripts/build_remediation_ledger.py` — the `claude -p` driver for the 930 authoring cases. **The agent never writes corpus rows** — it writes a decision ledger (one JSON line per required insert) that `apply` replays deterministically. Mirrors the one existing subprocess call site, `generate_sft_until_target.py::verify_batch_with_agent`, exactly (command construction, envelope unwrap, never-raises contract, `shutil.which` preflight). Three gate layers (entry / record / file) all deterministic; resumable via an append-only `accepted.jsonl` checkpoint; `--dry-run` and `--limit` for smoke testing.
- `.claude/agents/corpus-remediator.md` — new agent, `tools: Bash, Read, Grep, Glob, Write` (the only agent needing `Write`, scoped to a scratch dir by the playbook).
- `docs/task_a_state_convention_remediation_playbook.md` — the operator runbook.
- DVC: new stage `task_a_sft_remediate` between `task_a_sft_generate` and `task_a_sft_clean`.

Acceptance gates (all 5 files, post-remediation): 100% of tool-call turns self-loop; 0 `profile_task_a` hard defects; `ground_truth.state_sequence` == markers on 100% of rows; GT-inferred tool→state map unchanged on every retained row; no level loses >5% of rows; arrow-glyph distribution preserved; retry-arc distribution explicitly unchanged (remediation manufactures no retries).

### Part B — Generator redesign (prevents recurrence)

- `system_prompt.py`: fix rule 2's wrong worked example; promote `STAY_RULE` into `FORMAT_RULES` as default-on; rewrite rule 7 to state the retry-then-fallback arc; invert `TASK_A_STAY_RULE` to opt-out.
- `generate_workflows.py`: insert `find_tool_stay_violations` into the repair-loop's `or`-chain at position 3 (after the two referential checks, before continuity/shape — justified because v1 has 4,333 stay violations and 0 continuity defects, so the checks are independent and this ordering costs nothing).
- `_workflow_script.py::build_workflow_script`: new `tool_turn_semantics` (default `False` — required for Task C byte-identity) and `retry_budget` params; rewrites the "on success: proceed to [Y]" line that review §5.3 names as actively contradicting the convention.
- `config/schema.py::ComplexitySpec`: `retry_budget` and `retry_exhaustion` fields, per-level defaults (L1–L2: 1/none, L3–L4: 2/error_path, L5: 3/error_path). `error_path` resolves to `handoff_in_state` at sample time when the selected subgraph has no `tool_error` edge — no registry change needed.
- `quality_profiler.py` / `data_validator.py`: new hard defect for tool-call turns that don't self-loop, using the `f"{cid}: ..."` message contract `defective_conversation_ids` already parses.
- `generate_sft_until_target.py` / `generate_sft_data.sh`: new `--retry-budget`, `--retry-exhaustion`, `--require-tool-stay` flags; observability (not re-implementation) of the stay-conformance drop rate, since the profiler now gates it directly.

## Model routing

Per `.claude/skills/model-routing`: library/algorithm code, the agent driver, agent/playbook authoring, prompt edits, and the `ComplexitySpec`/determinism-touching work go to Opus (silent-corpus-corruption risk, concurrency/trust-boundary risk, judgment-dense authoring). CLI plumbing, template edits, validator wiring, and script flags go to Sonnet (pattern-following against a fully specified contract). DVC edits are Sonnet with human review. No Fable item is proposed.

## Out of scope for this change

- Adding a shared `HANDOFF` state to `domain_registry.py` (deferred; would invalidate Task C targets and `task_a_benchmark`).
- Retrofitting retry-exhaustion arcs into existing v1 rows.
- The actual GPU retrain and re-audit (tracked as follow-on work; this repo environment has no GPU).

## Risks

See the risk register in `docs/superpowers/plans/2026-07-31-task-a-tool-stay-convention-plan.md` (§Risk register) — duplicate/cid collisions (9 of 5,549 conversation_ids repeat), prose/label drift on requeued turns, agent-authored text quality, ledger/corpus drift after a regen, curriculum skew from drops, Task C invalidation, checkpoint-comparability breakage, generator-determinism breakage, and the DVC-lineage-drift failure mode that has already happened twice in this project's history (`docs/cat_a_state_annotation_convention_review.md` §1, §6.6).
