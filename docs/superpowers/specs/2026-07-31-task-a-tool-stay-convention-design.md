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

**Note (code-verified during implementation, Task 1):** the 4,333 turn-level figure above was measured by an earlier exploratory script; `find_tool_stay_violations`'s code-verified count on the same corpus is **4,322** — the 11-turn gap is exactly accounted for by turns whose structured `annotations.tool_calls` field claims a tool call but whose `content` has no literal `<tool_call>` tag (the separate, pre-existing "announce-but-don't-call" defect in `docs/grpo_tool_emission_gap_review.md`), which this module deliberately does not count (content is authoritative, per `_backfill_annotations`'s own rule). Treat 4,322 as the corrected reference figure; the per-conversation bucket percentages above are unaffected (they were never turn-count-derived).

**Note (code-verified during implementation, Task 2, after the `split_fused_tool_turn` removal above):** the 930/1,143 authoring-vs-free-repair split above was measured against the original (now-removed) move ladder, which made the split bucket disjoint from `relabel` by construction. With `split_fused_tool_turn` removed, the code-verified partition of all 5,549 conversations is `none` 3,476 / `relabel` 608 / `insert_handoff_turn` 1,150 / `append_closing_pair` 315 / `drop` 0 — i.e. **authoring cases rise from 930 to 1,465** (+71%), and zero-authored-text repairs fall from 1,143 to 608. This is the expected, accepted cost of the placement-safety fix, not a defect: the tool-placement invariant went from 360 violating conversations (6.5% of the corpus) to zero. Task 11/12/14's cost estimates (~$5–8, 1–2h at 4 workers for ~930 conversations) scale roughly linearly with authoring-case count — budget ~$8–13, 1.5–3h for 1,465. This does not change decision D1 (agent-authors, drop-on-reject) or the acceptance gates (still zero rows dropped, all 5,549 partition cleanly).

Error/retry reality: 18.8% of tool results are errors; retry streaks are `{1 attempt: 3,784, 2: 26}`; only 10 of 5,549 conversations escalate after an error. **There is no retry-exhaustion arc anywhere in the corpus** — requirement 5 of the convention has zero corpus support and cannot be repaired into existing rows, only generated fresh.

Domain registry constraints (`domain_registry.py`, 18 domains): every domain has exactly one terminal, literally `"TERMINAL"`; 8/18 domains have no escalation-like state; 10/18 have no `tool_error` edge; `validate_domain()` enforces exactly one spine successor per non-terminal state and full reachability to a terminal.

## Decisions (user-confirmed)

- **D1 (the 930 authoring cases):** a `claude -p` agent authors the missing prose into a reviewable decision ledger; a deterministic 3-layer gate accepts or drops each entry. Rejected alternatives: drop outright (loses 16.8%, skews curriculum) or drop-and-backfill from the generator (cleaner provenance but needs a partial teacher-API regen).
- **D2/D5 (prompt policy):** fix the wrong worked example in `FORMAT_RULES` rule 2, promote the existing (but currently opt-in) `STAY_RULE` to default-on via an inverted `TASK_A_STAY_RULE=0` opt-out (preserves ckpt-500/1770 byte-identical comparability), and rebuild system prompts on v2 rows so the corpus states the rule it now demonstrates.
- **D3 (retry-arc synthesis):** do **not** retrofit retry-exhaustion into v1 rows — there is nothing there to repair. Teach the fixed generator the arc and generate a dedicated 500–800-conversation slice (L3–L5, all language legs) merged into v2.
- **D4 (fallback target):** per-level retry budget (`ComplexitySpec.retry_budget`) with an in-state "handoff" behavior when the graph has no `tool_error` edge, rather than adding a shared `HANDOFF` state to `domain_registry.py`. Avoids invalidating all 18 canonical graphs, Task C graph-pair targets, and `task_a_benchmark`. **Final per-level values: L1–L4 = budget 2 / `error_path`, L5 = budget 3 / `error_path`.**
  - *Superseded sub-decision (final review wave, 2026-08-03):* D4 originally set L1/L2 to budget 1 / `"none"`, which renders the "do NOT retry it" form of the rule. The final whole-branch review measured that the existing corpus already retries at those levels — 200 of 1,251 retained L1 conversations and 783 of 1,305 L2 — so a budget-1 prompt would have shipped paired with data that demonstrates a retry. L1/L2 were raised to 2 / `error_path` to match L3/L4: complexity level describes graph *shape*, not whether a failed call may be retried.

## Core algorithm: whole-trajectory requeue

Per-turn relabelling is unsafe — defects come in runs (957 defective tool turns are themselves followed by another tool turn). The repair unit is the whole assistant-turn trajectory. Preserve every tool call's `from`-state exactly (making tool-placement legality invariant *by construction*, not merely re-checked), force tool turns to `X→X` — **fused or bare, always as a single message, never split** — and push each displaced advance onto a FIFO queue drained by the next prose turn:

```
cur = labels[0].from
for each assistant turn in order:
    if turn has <tool_call>:
        require cur == turn.from          # else: stacked tool -> needs an inserted hand-off
        emit (cur, cur)                   # prose + <tool_call>, if fused, stay in ONE message
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

**Why a fused turn is never split.** An earlier version of this algorithm classified a fused turn (`[STATE: X→Y]\nprose\n<tool_call>`) as a distinct `split_fused_tool_turn` move: keep the prose at the (wrong) advancing label `X→Y` and move only the `<tool_call>` into a new self-loop turn at the *destination* state `Y→Y`. Implementation-time measurement (Task 2 of the implementation plan) found this re-attributes the tool call from `X` to `Y` in `infer_state_tools_from_messages`'s accounting on ~66% of the sites where it applies (777 of 1,227 shipped split sites; 360 conversations, 6.5% of the corpus) — directly violating this section's own "preserve every tool call's `from`-state exactly" rule and the "GT-inferred tool→state map unchanged" acceptance gate. The fix: **a fused turn relabels exactly like a bare one** — the whole message (prose and `<tool_call>` together, content otherwise unchanged) becomes a single self-loop at its original `from`-state, and the displaced advance is queued exactly as for a bare tool turn. If this leaves an *already-bare* tool turn immediately following unable to fire (its own `from` no longer matches `cur`, since the fused turn's relabel didn't advance `cur`), that is a stacked-tool infeasibility resolved by `insert_handoff_turn` below — the same move that already handles two consecutive bare tool turns.

### Move ladder (ranked by cost)

| # | Move | Authored text |
|---|---|---|
| 1 | `relabel` — queue drains cleanly (covers both bare and fused tool turns) | none |
| 2 | `pullback_fuse` — move `<tool_call>` onto the preceding `X→X` prose turn (measured negligible yield: 14 candidates, 1.2% of the 1,150-conversation insert bucket — implemented ladder skips it) | none |
| 3 | `insert_handoff_turn` — stacked tool turns (bare-after-bare, or bare-after-relabelled-fused); insert one assistant message | 1 msg |
| 4 | `append_closing_pair` — tail deficit; append `user` ack + terminal `assistant` | 2 msgs |
| 5 | `drop` — post-gate failure or agent refusal | — |

Move 3 must bridge `cur` to the stacked turn's own `from`-state (not a bare self-loop at `cur`) — a self-loop at `cur` would leave the trajectory exactly as stuck as before. Move 4 must prepend a `user` turn before the closing `assistant` turn — most tail deficits end on an assistant turn, so appending an assistant turn alone would trip the "no two consecutive assistant prose turns" shape rule.

**Rejected moves:** splitting a fused turn to preserve its prose's claimed destination (see above — breaks tool-placement attribution at scale); marker-only inserted turns (0 of 64,964 corpus turns are marker-only — would teach empty-content turns); blanket 2-turn relabel for every defect (unsafe when the successor turn also calls a tool); skip-edges (tail deficits rarely have a declared shortcut); terminating one state short (adds non-terminal rows, trips continuity).

Exploratory bucket counts cited in earlier drafts of this design (608 `relabel` / 535 `split_fused_tool_turn` / 443 `insert_handoff_turn` / 599 `append_closing_pair`) predate this simplification and are superseded by the code-verified triage run (implementation plan, Task 4 Step 5) — that run is the authoritative source for current bucket sizes.

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
- `config/schema.py::ComplexitySpec`: `retry_budget` and `retry_exhaustion` fields, per-level defaults (L1–L4: 2/error_path, L5: 3/error_path — L1/L2 raised from 1/none in the final review wave, see D4). `error_path` resolves to `handoff_in_state` at sample time when the selected subgraph has no `tool_error` edge — no registry change needed.
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
