# Task A Tool-Stay Convention: Remediation Playbook

Operator runbook for producing `task-a-sft-v2` from `task-a-sft-v1`. Design rationale
lives in `docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md`;
this document is the sequence of commands and the decision points between them.

**All figures below were re-measured against the shipped code on 2026-08-02**
(branch `task-a-tool-stay-convention`, `state_convention_repair.py` at commit
`6341f6d`). Several supersede numbers still printed in the design spec and in the
implementation plan's task briefs — the deltas are called out where they occur.
Where this document and an older one disagree, this document is the later
measurement.

---

## 1. The convention, and why

Cat A fine-tuning is stalled on **state-transition accuracy, not tool emission**.
`docs/cat_a_state_annotation_convention_review.md` pins the cause: the v1 corpus
teaches an unstated and inconsistent convention for which state a tool-calling turn
should annotate. Gold expects a self-loop on 41.6% of turns; ckpt-500 emits one on
4.8%; accuracy on stay+tool-expected turns is 0.055. Three cheaper fixes were measured
and are null — perfect rule application caps composite at 0.7609 (§6.1), prompt-only
`STAY_RULE` injection moves self-loop emission 4.84% → 7.87% at Wilcoxon p=0.917
(§6.5), and 3.5× epochs gives composite 0.6579 at p=0.513 (§6.6). The corpus itself
has to change.

**Target convention:**

1. A turn emitting `<tool_call>` annotates `[STATE: X → X]` (stay).
2. A `role: "tool"` message returns the result.
3. On success, the *next* assistant turn advances: `[STATE: X → Y]`.
4. On error, the next turn stays `X → X` and may retry the same tool.
5. After N failed attempts, stop retrying and take a fallback path.

### Measured baseline

`data/output/sft/task_a`, `task-a-sft-v1`, **5,549 conversations**, 64,964 assistant
turns, **20,369** tool-call turns:

| | count | share |
|---|---|---|
| tool-call turns self-looping `X→X` (conformant) | **16,047** | 78.8% |
| tool-call turns forward-annotated `X→Y` (defect) | **4,322** | 21.2% |
| tool-call turns with no annotation | 0 | 0% |

All four figures are code-verified via `parse_assistant_turns` /
`find_tool_stay_violations`; the violation count is also reproducible as
`remediate_task_a_states.py verify --input-dir data/output/sft/task_a` (prints
`Total violations: 4322`).

> **Deviation.** The design spec's table reads 20,433 / 16,100 / 4,333, measured by an
> earlier exploratory script that counted a turn as tool-calling when its structured
> `annotations.tool_calls` said so. `state_convention.py` counts the literal
> `<tool_call>` tag in `content`, because content is authoritative per
> `_backfill_annotations`. The 64-turn difference is the separate
> "announce-but-don't-call" defect (`docs/grpo_tool_emission_gap_review.md`), which
> this module deliberately does not count. The 21.2% / 78.8% split is identical either
> way, and the per-conversation buckets in the next table were never turn-derived, so
> nothing downstream changes.

**Error/retry reality:** 18.8% of tool results are errors; retry streaks are
`{1 attempt: 3,784, 2: 26}`; only 10 of 5,549 conversations escalate after an error.
**There is no retry-exhaustion arc anywhere in v1** — convention requirement 5 has
zero corpus support and cannot be repaired into existing rows, only generated fresh
(decision D3; that is the separate retry slice in §10, not this remediation).

### Current move census

Reproduce with the triage command in §8 step 1:

| Move | Conversations | Share | Authored text |
|---|---:|---:|---|
| `none` — already conformant | 3,476 | 62.6% | — |
| `relabel` — queue drains cleanly | 619 | 11.2% | none |
| `insert_handoff_turn` — stacked tool turns | 833 | 15.0% | yes |
| `append_closing_pair` — tail deficit | 620 | 11.2% | yes |
| `drop` — planner found it infeasible | **1** | 0.0% | — |

**Authoring queue: 1,453 conversations carrying 3,842 individual inserts.**

> **Deviation from earlier documents.** The design spec's Task-2 note records the
> split as `insert_handoff_turn` 1,150 / `append_closing_pair` 315, and the Task 11
> brief was written against a still older `930 conversations / ~1,020 inserts`
> figure. Both are stale. The 930 figure predates the removal of
> `split_fused_tool_turn`; the 1,150/315 split predates commits `9736933`, `2e739bb`,
> `97c33a7`, and `7824c90`, which made one plan accumulate every bridge in a
> conversation instead of returning at the first.
>
> The census above also supersedes the `608 / 866 / 599 / 1,465 convs / 3,902 inserts`
> figures this document carried until the retry-after-error fix (§4.1). That fix moved
> 33 conversations out of `insert_handoff_turn` — 11 collapsed to a free `relabel`, 21
> became `append_closing_pair`, and 1 became the corpus's only `drop` — and removed 60
> inserts from the paid queue.

Insert composition of the 3,842:

| Kind | Role | `required_marker` | Count |
|---|---|---|---:|
| Hand-off bridge | `assistant` | non-empty | 1,583 |
| Closing-pair terminal turn | `assistant` | non-empty (`… → TERMINAL`) | 620 |
| Shape-padding acknowledgement | `user` | `""` | **1,019** |
| Closing-pair opener | `user` | `""` | 620 |

The 1,019 shape-padding user acks are the largest single surprise relative to the
brief, which does not mention them at all. They exist because a bridge is assistant
*prose* spliced next to another assistant turn, and `find_shape_violations` rejects
two consecutive assistant prose turns — most pad the trailing edge (the stacked turn
is fused, so prose follows prose), a handful pad the leading edge (the bridge lands
directly after an assistant turn). Together with the closing-pair openers, **1,639 of
3,842 requests (43%) are customer voice, not agent voice.**

Distribution of inserts per conversation (max **14**):

| inserts | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversations | 242 | 777 | 101 | 198 | 19 | 66 | 30 | 12 | 6 | 2 |

By language — the queue is **not** English-majority:

| Language | corpus rows | authoring rows | at risk | inserts |
|---|---:|---:|---:|---:|
| `code_switch` | 1,895 | 525 | 27.7% | 1,443 |
| `th` | 1,801 | 511 | 28.4% | 1,372 |
| `en` | 1,853 | 417 | 22.5% | 1,027 |

---

## 2. Invariant table

Every one of these is independently re-checked after repair by
`state_convention_repair.py::verify_repaired`, which the `apply` subcommand runs on
each candidate row before writing it.

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

One invariant is **not** checkable inside `verify_repaired` and lives in the CLI
instead: *"the repair did not change which state any tool is called from."* It needs
both the before and after messages, so `cmd_apply` compares
`infer_state_tools_from_messages(before)` against `infer_state_tools_from_messages(after)`
and drops the row under `tool-from-state-changed` if they differ. On the current
corpus that counter is 0 — the whole-trajectory requeue makes it hold by construction.

---

## 3. The queue algorithm, worked

```
cur = labels[0].from
for each assistant turn in order:
    if turn has <tool_call>:
        require cur == turn.from          # else: stacked tool -> insert a hand-off bridge
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

`[STATE:]` marker text, `annotations.state_transition`, and
`ground_truth.state_sequence` are all written from the same emit list, in lockstep.

### 3.1 A real `relabel` row — `L2_011_4`

`l2_merged_20260630` line 508, en, L2, `government`. Reproduce with the dump script in
§8 step 1b. Only assistant turns 9 and 10 change; nothing is inserted.

```
                                      BEFORE                        AFTER
 [6] assistant +tool  CHECK_ELIGIBILITY → CHECK_ELIGIBILITY   (unchanged)
 [7] tool             {"status": "documents_required", ...}
 [8] assistant        CHECK_ELIGIBILITY → VERIFY_DOCUMENTS    (unchanged)
...
[18] assistant +tool  SUBMIT_FORM → RESOLVE          -->      SUBMIT_FORM → SUBMIT_FORM
[19] tool             {"status": "submitted", "inquiry_id": "INQ-449182", ...}
[20] assistant        RESOLVE → RESOLVE              -->      SUBMIT_FORM → RESOLVE
[21] user             "No, that's all. Thanks..."
[22] assistant        RESOLVE → TERMINAL                      (unchanged)
```

Message 18 is a fused tool turn that illegally advanced. It relabels **in place** to a
self-loop at its own `from`-state — prose and `<tool_call>` stay in the same message,
untouched — and `RESOLVE` goes on the queue. Message 20 is the next prose turn, so it
drains the queue and carries the advance. Its prose ("Great news! Your housing benefit
eligibility has been fully verified and…") reads correctly as an arrival at `RESOLVE`
either way, which is why this is free.

`ground_truth.state_sequence` is re-derived in lockstep:

```
BEFORE  … {"from":"SUBMIT_FORM","to":"RESOLVE"}, {"from":"RESOLVE","to":"RESOLVE"}, …
AFTER   … {"from":"SUBMIT_FORM","to":"SUBMIT_FORM"}, {"from":"SUBMIT_FORM","to":"RESOLVE"}, …
```

`verify_repaired` on the result returns `[]`.

### 3.2 A real `append_closing_pair` row — `L1_104_2`

`l1_merged_20260629` line 747, en, L1, `complaints`. The conversation ends on a tool
call, so the queue cannot drain and two messages are appended.

```
                                      BEFORE                        AFTER
 [6] assistant +tool  LISTEN_COMPLAINT → ACKNOWLEDGE_ISSUE  -->  LISTEN_COMPLAINT → LISTEN_COMPLAINT
 [7] tool             {"status": "success", "complaint_id": "CMP-90812"}
 [8] assistant        ACKNOWLEDGE_ISSUE → TERMINAL          -->  LISTEN_COMPLAINT → ACKNOWLEDGE_ISSUE
 [9] user             ——                                    -->  «authored ack»
[10] assistant        ——                                    -->  ACKNOWLEDGE_ISSUE → TERMINAL, «authored close»
```

```
BEFORE  [GREETING→LISTEN_COMPLAINT, LISTEN_COMPLAINT→LISTEN_COMPLAINT,
         LISTEN_COMPLAINT→ACKNOWLEDGE_ISSUE, ACKNOWLEDGE_ISSUE→TERMINAL]
AFTER   [GREETING→LISTEN_COMPLAINT, LISTEN_COMPLAINT→LISTEN_COMPLAINT,
         LISTEN_COMPLAINT→LISTEN_COMPLAINT, LISTEN_COMPLAINT→ACKNOWLEDGE_ISSUE,
         ACKNOWLEDGE_ISSUE→TERMINAL]
```

Note the `user` turn at index 9. Most tail deficits end on an assistant turn, so
appending the terminal assistant turn alone would trip the "no two consecutive
assistant prose turns" shape rule. That is why the move is a *pair*, and why 620 of
the 1,639 user-role inserts exist.

### 3.3 A real retry-after-error row — `L1_009_2`

`l1_merged_20260629` line 101, code_switch, L1, `account_management`. **Zero inserts** —
this is the shape §4.1's fix produces, and it is the convention's rules 1–4 in one row:

```
 [6] assistant +tool  VERIFY_IDENTITY → AUTHENTICATE  -->  VERIFY_IDENTITY → VERIFY_IDENTITY
                      (bare verify_identity call; AUTHENTICATE goes on the queue)   rule 1
 [7] tool             {"error": "Service temporarily unavailable"}                  rule 2
 [8] assistant +tool  AUTHENTICATE → AUTHENTICATE     -->  VERIFY_IDENTITY → VERIFY_IDENTITY
                      (fused: apology prose + retry call — stays put)               rule 4
 [9] tool             {"status": "success", …}                                      rule 2
[10] assistant        AUTHENTICATE → AUTHENTICATE     -->  VERIFY_IDENTITY → AUTHENTICATE
                      "ระบบผ่านขั้นตอนนี้เรียบร้อยแล้วค่ะ" — the queued advance lands here  rule 3
```

The advance queued off message 6 survives the failed attempt and drains on the first
prose turn *after the success* — which is precisely the turn whose existing prose says
"we've passed this step". Nothing is authored, nothing is invented, and the row is a
clean demonstration of stay → error → retry-in-place → advance-on-success.

**This row used to be `insert_handoff_turn` with two authored messages**, and this
section used to explain why the authored bridge had to advance `VERIFY_IDENTITY →
AUTHENTICATE` immediately after the error while somehow not narrating success. That
was the bug, rationalised into documentation — see §4.1.

Inserts that share a `position_after_msg_index` come out in **list order**;
`apply_plan` guarantees this by sorting on `(-position, -list_index)` and inserting
back-to-front. (`L2_071_4`, `l2_merged_20260630` line 365, is a surviving one-insert
`insert_handoff_turn` if you want to trace that path instead.)

---

## 4. Move ladder and rejected moves

| # | Move | Authored text | Conversations |
|---|---|---|---:|
| 1 | `relabel` — queue drains cleanly (bare and fused tool turns alike) | none | 619 |
| 2 | `pullback_fuse` — move `<tool_call>` onto the preceding `X→X` prose turn | none | **not implemented** |
| 3 | `insert_handoff_turn` — stacked tool turns; insert an assistant bridge (+ ack padding) | 1–13 msgs | 833 |
| 4 | `append_closing_pair` — tail deficit; append `user` ack + terminal `assistant` | 2+ msgs | 620 |
| 5 | `drop` — planner infeasibility, post-gate failure, or agent refusal | — | 1 planner-side |

Move 3 must bridge `cur` to the stacked turn's own `from`-state, **not** a bare
self-loop at `cur` — a self-loop would leave the trajectory exactly as stuck as before.

### 4.1 The retry-after-error exception (found at §8 step 4, 2026-08-03)

**A queued advance is never drained across a tool error.** When a stacked tool turn
follows an errored result, it is a retry: it stays at `cur`, the queued advance stays
queued, and no bridge is authored. `state_convention_repair._errored` is the predicate;
`plan_repair` short-circuits on it before the stacked-tool branch.

Without this, `plan_repair` authored a bridge asserting `X → Y` directly after
`{"error": …}` and the retry then fired *from Y* — teaching advance-on-failure, the
exact inverse of rule 4, on **107 of 912 advancing bridges (11.7%)**. It was also an
impossible authoring request: `corpus-remediator.md` forbids narrating success after an
error, so the agent could only refuse or hallucinate. Both this document (old §3.3) and
the agent definition had rationalised the contradiction into prose instead of treating
it as a bug — worth remembering when a spec starts explaining why something
uncomfortable is fine.

Caught by `--dry-run` before any spend: it was the *first* request in batch 0.

Two consequences worth knowing:

- **The tool-attribution gate had to be narrowed from equality to a subset relation.**
  Relabelling a retry back to the state it actually retried from correctly *removes* the
  wrongly-advanced attribution, which `before != after` read as corruption and dropped
  100 conversations — precisely the rows demonstrating rule 4. The property that
  actually keeps the corpus safe is "the repair never invents a call-site", i.e.
  `after ⊆ before`. Measured: of the 102 rows whose map changes, 97 only *lose* pairs
  (kept) and 5 genuinely *move* a tool between states (still dropped).
  `remediate_task_a_states._gains_tool_attribution`, with a unit test.
- **One conversation became unrepairable** — `L4_058_2` (`l4_merged_20260630` line 155),
  a four-deep stacked-tool chain with an error in the middle. `cur` stays at
  `CLAIM_INTAKE` while the rest of the row is labelled as if already past
  `ASSESS_COVERAGE`, and bridges jump `cur` straight to the stacked turn's `from`,
  skipping the queue. It fails the deficit gate rather than shipping wrong: 1 of 5,549.

Net across all levels: **6 drops (0.11%)** with a full ledger — 5 tool-attribution, 1
infeasible — against 3 before the fix. Per level: L1 0, L2 0, L3 2, L4 4, L5 0.

**`pullback_fuse` was measured and deliberately skipped.** It yielded 14 candidates —
1.2% of what was then a 1,150-conversation insert bucket. Not worth a distinct code
path, so the implemented ladder goes straight from `relabel` to `insert_handoff_turn`.

**Rejected moves:**

- **Splitting a fused turn** (`split_fused_tool_turn`, the original move 2). Keep the
  prose at its claimed destination `X→Y` and move only the `<tool_call>` into a new
  self-loop at `Y→Y`. Measurement killed it: it re-attributes the tool call from `X`
  to `Y` in `infer_state_tools_from_messages`' accounting on ~66% of applicable sites
  (777 of 1,227 shipped split sites, across 360 conversations = 6.5% of the corpus),
  directly violating the "preserve every tool call's `from`-state exactly" rule and
  the "GT-inferred tool→state map unchanged" acceptance gate. Removing it is what
  raised the authoring bucket from 930 to ~1,450 conversations, and it drove
  tool-placement violations from 360 conversations to zero. That is the trade, and it
  was accepted.
- **Marker-only inserted turns.** 0 of 64,964 corpus turns are marker-only; adding
  them would teach empty-content turns.
- **Blanket 2-turn relabel for every defect.** Unsafe whenever the successor turn also
  calls a tool — which is the common case, since defects come in runs.
- **Skip-edges.** Tail deficits rarely have a declared shortcut.
- **Terminating one state short.** Adds non-terminal rows and trips continuity.

---

## 5. Drift

`RepairPlan.drift_turns` marks prose turns whose destination changed because the
trajectory was re-derived rather than relabelled in place. It holds 1-based
*assistant-turn* ordinals, not message indices — it is a human-review pointer, not an
addressing handle.

**Read this before designing an audit around it: `drift_turns` turned out to be
low-signal.** Measured on the current corpus:

| Move | conversations | with non-empty `drift_turns` |
|---|---:|---:|
| `relabel` | 619 | **619 (100%)** |
| `insert_handoff_turn` | 833 | 678 |
| `append_closing_pair` | 620 | 620 |
| **total** | 2,072 | **1,917** |

It fires on **every single `relabel` plan**, because the queue-draining turn *always*
has a changed `from` — that is the mechanism, not an anomaly. An audit protocol phrased
as "read every conversation with non-empty `drift_turns`" therefore degenerates into
"read the entire repaired corpus," which is not a usable gate. Do not write one.

### The audit protocol that is actually worth running

Sample-based, ~30 conversations, roughly 45 minutes:

```bash
source .venv/bin/activate && python3 - <<'PY'
import json, random, collections
R = json.load(open("data/interim/task_a_state_triage/report.json"))
pool = [r for r in R["records"] if r["drift_turns"]]
random.seed(20260802)
buckets = collections.defaultdict(list)
for r in pool:
    buckets[(r["language"], r["move"])].append(r)
picked = []
for k in sorted(buckets):
    random.shuffle(buckets[k])
    picked += buckets[k][:4]          # 3 langs x 3 moves x 4 = up to 36
for r in picked:
    print(r["key"], r["conversation_id"], r["language"], r["move"],
          "drift_turns:", r["drift_turns"])
PY
```

Then read each one before/after with the dump script from §8 step 1b, and judge one
question only:

> **Does the prose of each drifted turn still make sense at its new destination?**

Concretely, for each flagged turn: it now claims to arrive somewhere different than it
originally did. Does its text still describe *that* arrival? §3.1's message 20 is the
common, benign shape — a turn that reports a tool result and moves on reads correctly
at either destination, because the prose describes the *result*, not the *state*. The
failure shape to hunt for is a turn whose prose names the destination explicitly
("Let's move on to scheduling your appointment") while the requeue has re-pointed it
somewhere else.

Stratify across all three language legs — the drift pool is 1,917 rows and the
language mix of the authoring queue is 28% code_switch / 28% th / 23% en, so an
English-only sample is not representative.

**Gate:** if more than ~3 of 30 read badly, stop and tighten `plan_repair` (Tasks 2/3)
rather than proceeding to §8 step 4. This is a quality gate, not a formality — the
authoring pass costs real money and its output is only as good as the trajectory it
is authored into.

> **Align before/after by prose, not by turn ordinal.** `insert_handoff_turn` adds
> messages, so after-turn *n* is not before-turn *n* — an ordinal-aligned dump makes
> correct repairs look like wild mislabels on all 833 insert rows. `apply_plan` never
> edits existing prose, so prose is a stable join key; authored inserts are the
> after-turns with no match in the before-list.

### Audit result (run 2026-08-03, 36 conversations, 3 languages × 3 moves)

**0 of 36 read badly.** The gate passes with a wide margin, and the corpus-wide
measurement below explains why — the failure shape the protocol hunts for is
structurally impossible here, not merely rare.

Over all 1,917 drifted conversations, 5,260 drifted turns:

| Shape | turns | share |
|---|---:|---:|
| `from` changed, destination identical | 2,114 | 40.2% |
| destination changed | 3,146 | 59.8% |
| — of those, new destination == that turn's **own old `from`** | **3,146** | **100.0%** |
| — of those, re-pointed anywhere else | **0** | **0%** |

Every destination change in the corpus is the same move: **the label is pulled back by
exactly one state.** There is no case of a turn being re-pointed to an unrelated
destination, which is precisely the "prose names the destination, requeue sent it
elsewhere" failure §5 was written to catch.

Reading the sample explains the direction. The v1 labels ran consistently *one state
ahead of the prose*: a turn saying "the claim is approved — next I'll process the
payout" was labelled `→ PROCESS_PAYOUT`, when the payout is explicitly the *next* turn's
work. The requeue pulls it back to `→ APPROVE_OR_DENY`, which is what the prose actually
reports completing. Across the sample the repaired label was **more** faithful to the
prose than the original in every destination-change case, not less. Same pattern in all
three languages (`L4_087` th/code_switch insurance chains, `L3_045_6`/`L3_078_6` en
sales, `L5_051_7` th telecom).

**What the audit *did* surface** is not a trajectory bug but an authoring risk, now
fixed in `.claude/agents/corpus-remediator.md`: in **568 of 620** `append_closing_pair`
conversations (91.6%) the last existing assistant turn **already says goodbye**, so the
authored closing pair lands after a farewell. Left unaddressed, the agent would have
written a second full sign-off on ~550 rows.

---

## 6. Ledger contract and the three gate layers

**The agent never writes corpus rows.** `scripts/build_remediation_ledger.py` drives
`claude -p --agent corpus-remediator` and collects a *decision ledger*: one JSON line
per requested insert. `remediate_task_a_states.py apply` then replays that ledger
deterministically. This keeps `dvc repro` free of LLM spend, makes every authored
sentence PR-reviewable as a diff, and makes a rejected entry degrade to `drop` in
isolation instead of poisoning a batch.

### Entry schema

```json
{"insert_id": "l1_merged_20260629:101:0", "conversation_id": "L1_009_2",
 "role": "assistant", "content": "[STATE: VERIFY_IDENTITY → AUTHENTICATE]\n…",
 "rationale": "<=200 chars", "agent_model": "<model id>", "schema_version": 1}
```

A refusal is the same shape with `"refuse": true` and no `content`. Field names are
load-bearing: `remediate_task_a_states.py::_load_ledger` keys on `insert_id` and
`apply_plan` reads `entry["content"]`. Do not rename without changing both.

### Layer 1 — entry gate (`validate_entry`, per insert)

Empty list == accept. Rejects on:

| Check | Rule |
|---|---|
| request integrity | the *request* carries a valid `insert_id`/`conversation_id`/`language`/`role`, and `required_marker` is non-empty iff `role == "assistant"`. A driver-side bug fails loudly here instead of silently disabling the checks that depend on the missing field |
| refusal | `refuse: true` → `["agent refused this insert"]` |
| entry shape | the ledger line is a JSON **object**; `insert_id`, `conversation_id`, `role`, `content`, `rationale`, `agent_model` are all present, all strings, all non-empty |
| identity | `entry.insert_id == request.insert_id` and `entry.conversation_id == request.conversation_id` |
| role | `entry.role == request.role` |
| schema_version | `1` when present |
| tool calls | neither `"<tool_call>"` nor `"</tool_call>"` in `content` |
| marker prefix (assistant) | `content.startswith(required_marker)` — **byte for byte, arrow glyph included** |
| marker newline (assistant) | the character right after the marker is `"\n"` (100% of the 64,964 markers in the corpus are followed by one) |
| meaningful-content floor (both roles) | the prose must **contain ≥10 meaningful characters** — letters or digits of the Latin or Thai script, NFKC-folded. Stated positively on purpose: padding, punctuation, symbols, combining marks and *every* invisible codepoint count zero, so there is no blocklist to keep current. Assistant: measured after the marker (a 25–60 char marker plus a newline otherwise clears the 20-char floor while saying nothing). User: measured on the whole content (the 20-char floor counts spaces, so `"x"` + 19 spaces otherwise passed — and 1,639 of the 3,842 inserts are acks). Calibrated: 10 costs **0** false rejections across the 93,064 assistant/user corpus turns that clear the other floors; the shortest real turn carries 12 |
| second marker | no `"[STATE:"` in `content[len(required_marker):]` |
| marker (user) | no `"[STATE:"` anywhere when `required_marker == ""` |
| length | `20 <= len(content) <= 600`, counted **including** the marker |
| language | th / code_switch rows must contain a Thai **letter** (ก–ฮ and the spacing vowels ะ า ำ เ–ๅ); `en` rows must contain none. **Not** the whole Thai block: ฿, ๆ, ๏, ๚, ๛ and ๐–๙ are excluded, so a baht sign cannot pass an English sentence off as Thai and an `en` row may quote a ฿ price |
| never-text characters | no character whose Unicode **category** is `Cc`/`Cf`/`Co`/`Cs`/`Cn` (controls other than `\n`/`\t`, format characters, private use, surrogates, unassigned). Category-based, not a codepoint list. This is the *secondary* defence only — invisible codepoints outside those categories (U+3164 HANGUL FILLER and U+115F/U+1160/U+FFA0 are `Lo`, U+2800 is `So`, U+FE00–FE0F/U+E0100/U+17B4/U+17B5/U+034F are `Mn`) are stopped by the meaningful-content floor instead, which never counts them |
| special tokens | no chat-template sentinel: any `<|…|>` form (`<|im_end|>`, `<|eot_id|>`, `<|user|>`), `<start_of_turn>`/`<end_of_turn>`, `<s>`/`</s>`/`<bos>`/`<eos>`, `[INST]`/`[gMASK]`, `<extra_id_N>`, `<<SYS>>`, `<think>`/`</think>`, `<tool_response>`, `<unusedN>` (Gemma), `<reserved_special_token_N>` (Llama 3), and Mistral v3's `[TOOL_CALLS]`/`[AVAILABLE_TOOLS]`/`[TOOL_RESULTS]` — Mistral-Small-3.1-24B is a live Cat A candidate and this is a tool-calling corpus. Baked into `content` these are re-read as turn boundaries at template time. 0 false positives across all 106,992 assistant/user corpus turns; `[STATE: …]` and ordinary bracketed prose are untouched |
| copy guard | `content`'s prose is not a copy of a `context_window` message, for copies ≥40 characters. Compared on the **meaningful skeleton** (NFKC, meaningful characters only, casefolded), so one invisible character or a changed comma cannot defeat it. The ≥40 gate stays on the visible prose so the guard cannot fire *less* often than before |
| duplicate prose (batch level, in `_reject_duplicate_content`, not `validate_entry`) | no two accepted entries in one batch share ≥40 characters of prose with the same **meaningful skeleton** — the "generic ack repeated across rows" defect. Same normalisation as the copy guard, so an invisible-character variant is still a duplicate. First occurrence stands |

The arrow-glyph rule bites in practice: of the 2,203 markered requests, **2,125 carry
a Unicode `→` and 78 carry an ASCII `->`**. `_marker()` copies the arrow from the
source turn, so both survive into the queue. An agent that normalises one to the other
fails the prefix check and loses the row.

The language check is gated because 2,815 of the 3,842 authored inserts are th or
code_switch and silent English drift is the single most likely quality failure — and
the only one no other check can see. It is a *script-presence* check, not a fluency or
ratio check, which is what keeps it safe for `code_switch`: that register mixes Thai
grammar with English technical nouns, and any such entry contains Thai characters and
passes. Measured against the corpus, requiring ≥1 Thai character rejects 1 of 29,968
`th` turns and 207 of 31,856 `code_switch` turns (0.65%, all of them fully-English
turns, which the agent file explicitly forbids authoring); requiring **zero** Thai in
an `en` entry rejects 0 of 31,164 `en` turns. `data_validator.detect_thai_corruption`
is deliberately *not* used here: it detects *garbled* Thai (Latin glued into a Thai
word, obsolete `ฃ`/`ฅ`), so a wholly-English entry on a `th` row — the failure mode
that matters — trips neither of its signals and passes it clean.

**The check counts Thai *letters*, not the Thai Unicode block** (fixed 2026-08-02 after
review). Matching the whole `U+0E00–U+0E7F` block defeated the rule in both directions:
a single `฿` in an otherwise all-English answer satisfied it, which — measured by
injecting exactly that into every row — let **all 2,815** th/code_switch inserts pass,
and the same block match rejected `en` rows for a baht price, the one legitimate English
use of a Thai-block character. The narrowed class costs nothing on real content: across
all **93,059** corpus turns of ≥20 characters, the letter class and the old block class
agree on every single one (**0 disagreements**), so the 1-of-29,968 `th` and
207-of-31,856 `code_switch` figures above are unchanged. Post-fix the injection is
caught 2,815/2,815, and 0 of 1,027 `en` rows are rejected for quoting `฿1,200`.

Every row of this table corresponds to a numbered rule in
`.claude/agents/corpus-remediator.md` § "Hard formatting rules", and the agent's own
self-check calls `validate_entry` directly rather than re-implementing it, so the
three cannot drift.

### Layer 2 — record gate (`apply_plan` + `cmd_apply`, per conversation)

- `apply_plan` returns `None` if **any** of the conversation's `insert_id`s is missing
  from `accepted.jsonl`. A conversation is all-or-nothing: 13 accepted inserts and one
  rejected still drops the row.
- `cmd_apply` then re-checks `infer_state_tools_from_messages(before) == (after)` and
  drops under `tool-from-state-changed` if not.
- `verify_repaired(repaired)` runs the full invariant table from §2; any violation
  drops the row under `post-gate-failed`.

### Layer 3 — file gate (`verify --strict`, per directory)

`remediate_task_a_states.py verify --input-dir … --strict` re-runs `verify_repaired`
over every written row and exits non-zero on any violation. Run it on the final output
directory before anything downstream consumes it.

### Never-raises and resumability

`run_agent_batch` never raises: it preflights with `shutil.which("claude")`, catches
`subprocess.TimeoutExpired`, and reports a non-zero exit as a batch-level rejection.
Every request in a failed batch is written to `rejected.jsonl` with a reason, so a
crashed or throttled run loses money but never corrupts state.

Resumability is `accepted.jsonl`: it is append-only, and `load_accepted_ids` reads it
at startup to skip work already done. **Re-running the full command after a partial
run is safe and is the intended recovery path.** `rejected.jsonl` is *not* consulted
for resume — a rejected insert is retried on the next run. If you want a rejection to
stick, you must remove the request or accept the drop.

Three details make that safe under a hard kill. Appends are a single `write()` of the
whole blob followed by `fsync`, and only the main thread writes, so concurrent batches
never interleave lines. A kill can still leave a torn final line; `load_accepted_ids`
detects a trailing fragment that does not parse and truncates it before the next
append, so a good line is never glued onto a partial one. And `accepted.jsonl` is
written *before* `rejected.jsonl` for each batch, so the only record a crash between
the two can lose is a rejection log line — whose insert is retried anyway.

A third file, `progress.json`, is rewritten (atomically, via `os.replace`) after every
batch with the run id, the arguments, batch/accept/reject counts and every batch-level
error. It is a monitoring artefact only — nothing reads it back, and deleting it does
not affect resume. Each batch also keeps its request file and the agent's raw ledger
under `<ledger-dir>/batches/<run-id>/batch_NNNNN/` for post-hoc review.

---

## 7. Acceptance criteria

All nine must hold across all five output files before tagging `task-a-sft-v2`.

| # | Gate | How to check |
|---|---|---|
| 1 | 100% of tool-call turns self-loop | `verify --strict` prints `Total violations: 0` |
| 2 | 0 `profile_task_a` hard defects | `python -m llm_workflow_agents.data.quality_profiler <file>` |
| 3 | `distributions.tool_turn_state.pct_conformant` reads `100.0` | same, `--json` |
| 4 | `ground_truth.state_sequence` == message markers on 100% of rows | covered by gate 1 (`verify_repaired`'s final check) |
| 5 | GT-inferred tool→state map unchanged on every retained row | `apply` reports `tool-from-state-changed: 0` |
| 6 | **No level loses >5% of its rows** | §9 table |
| 7 | Arrow-glyph distribution preserved | §9 command |
| 8 | Retry-arc distribution explicitly unchanged (remediation manufactures no retries) | §9 command |
| 9 | All 5,549 rows partition cleanly into kept ∪ dropped, no duplicates | `apply`'s `kept + dropped == 5549` |

Gate 6 is the binding constraint and it is tight — see §9.

---

## 8. Runbook

```bash
source .venv/bin/activate
```

### Step 1 — Triage

```bash
python scripts/remediate_task_a_states.py triage \
  --input-dir data/output/sft/task_a \
  --report data/interim/task_a_state_triage/report.json
```

Runs in ~4 s, writes ~19 MB. Expected output:

```
Triage: 5549 rows -> {'none': 3476, 'relabel': 619, 'insert_handoff_turn': 833, 'append_closing_pair': 620, 'drop': 1}
```

If those four numbers differ, the input corpus changed — stop and reconcile against §1
before continuing, because every cost and budget figure downstream is derived from
them.

### Step 1b — The before/after dump script (used by §3 and §5)

Write this once; §5's audit and every worked example above use it.

```bash
cat > /tmp/worked.py <<'PY'
import json, re, sys
sys.path.insert(0, "src")
from llm_workflow_agents.data.state_convention_repair import (
    plan_repair, apply_plan, verify_repaired)

STEM, LINE = sys.argv[1], int(sys.argv[2])
with open(f"data/output/sft/task_a/{STEM}.jsonl") as fh:
    for i, line in enumerate(fh):
        if i == LINE:
            rec = json.loads(line); break

plan = plan_repair(rec)
for k, ins in enumerate(plan.inserts):
    ins.insert_id = f"{STEM}:{LINE}:{k}"
print(rec["conversation_id"], rec.get("language"), rec.get("complexity_level"),
      rec.get("domain"), "| move:", plan.move, "| drift_turns:", plan.drift_turns)

SR = re.compile(r"\[STATE: [^\]]*\]")
def dump(msgs, label):
    print(f"\n=== {label} ===")
    for i, m in enumerate(msgs):
        if m["role"] == "system":
            print(f"  [{i}] system: <workflow contract, elided>"); continue
        c = m["content"] or ""
        mk = SR.search(c)
        tc = " +tool" if "<tool_call>" in c else ""
        body = SR.sub("", c).strip().replace("\n", " ")[:70]
        print(f"  [{i}] {m['role']:9s} {mk.group(0) if mk else '':44s}{tc:6s} {body}")

dump(rec["messages"], "BEFORE")
print("\nBEFORE state_sequence:", json.dumps(rec["ground_truth"]["state_sequence"]))
ledger = {ins.insert_id: {"content": (ins.required_marker + "\n«AUTHORED»")
                          if ins.required_marker else "«AUTHORED ACK»"}
          for ins in plan.inserts}
out = apply_plan(rec, plan, ledger_entries=ledger or None)
if out is None:
    print("\napply_plan returned None"); raise SystemExit
dump(out["messages"], "AFTER")
print("\nAFTER  state_sequence:", json.dumps(out["ground_truth"]["state_sequence"]))
print("\nverify_repaired(after):", verify_repaired(out))
PY

python /tmp/worked.py l2_merged_20260630 508   # the §3.1 relabel example
python /tmp/worked.py l1_merged_20260629 747   # the §3.2 closing-pair example
python /tmp/worked.py l1_merged_20260629 101   # the §3.3 handoff example
```

### Step 2 — Deterministic apply (no ledger yet)

Establishes the floor: what the corpus looks like with zero authored text.

```bash
python scripts/remediate_task_a_states.py apply \
  --input-dir data/output/sft/task_a \
  --output-dir /tmp/task_a_deterministic_only \
  --on-unrepairable drop
```

Runs in ~13 s. Expected:

```
Apply: kept 4094, dropped 1455 ({'needs-ledger:insert_handoff_turn': 833, 'needs-ledger:append_closing_pair': 620, 'tool-from-state-changed': 1, 'deficit of 4 states at end (only 1 supported)': 1})
```

4,094 = 3,476 `none` + 619 `relabel` − 1 tool-attribution drop. Confirm the output is clean:

```bash
python scripts/remediate_task_a_states.py verify --input-dir /tmp/task_a_deterministic_only --strict
# Total violations: 0     (exit 0)
python scripts/remediate_task_a_states.py diff \
  --before data/output/sft/task_a --after /tmp/task_a_deterministic_only
# before: {'none': 3476, 'relabel': 619, 'insert_handoff_turn': 833, 'append_closing_pair': 620, 'drop': 1}
# after:  {'none': 4094}
```

> `--on-unrepairable` accepts only `drop`; `keep` and `truncate` were removed rather
> than left as flags that argparse took and `cmd_apply` ignored. A row that cannot be
> brought onto the convention is still violating it, so retaining it would fail Step 7.

**Do not ship this directory.** It fails acceptance gate 6 badly (§9) — L4 would lose
41.6% of its rows. It exists to prove the deterministic half is sound before spending
money on the other half.

### Step 3 — Drift audit

Run §5's protocol now, before the paid step. It is the last cheap opportunity to find
a trajectory bug.

### Step 4 — Smoke the agent pass (~20 inserts, whole conversations)

> Steps 4–5 require `scripts/build_remediation_ledger.py`, which now exists (Task 12).
> The commands below are its documented contract and are exercised by
> `tests/unit/test_build_remediation_ledger.py`, but they have **not been executed
> against the live API** — that spend is still gated on an explicit go-ahead. Add
> `--dry-run` to render the first batch's prompt and request file without calling
> `claude` at all; that path is free and is the right way to review the prompt before
> paying for anything.

```bash
python scripts/build_remediation_ledger.py \
  --input-dir data/output/sft/task_a \
  --triage-report data/interim/task_a_state_triage/report.json \
  --ledger-dir data/interim/task_a_remediation_ledger \
  --limit 20

python3 -m json.tool < data/interim/task_a_remediation_ledger/accepted.jsonl
```

**`--limit` trims on conversation boundaries, so it never splits a conversation.**
It normally undershoots, but it takes the first conversation whole even when that
conversation alone exceeds the limit — so it can overshoot by up to one conversation's
worth of inserts (at most 14, the largest authoring queue on a single conversation).
On the current queue `--limit 20` authors **18 inserts across 8 complete conversations,
in 2 batches**. It used to take a flat slice of exactly 20, which cut conversation
`l1_merged_20260629:528` in half — and a conversation with one insert missing is dropped
whole by `apply_plan`, so the smoke run was silently reporting on a conversation that
could never have been applied. The run prints the conversation count; if it ever reports
a partially-authored conversation, that is a bug.

**Read all 18 entries by hand.** Deterministic gates cannot catch the failure modes
that matter here. Check specifically:

- Any `th` / `code_switch` row answered in English → stop, the agent file's language
  section is not landing.
- Any bridge following an `"error"` result that narrates success → stop, that is the
  worst failure mode in the set (it teaches tool-result hallucination).
- Any `user` ack that reads as agent voice, or that is a generic "Thank you!" repeated
  across rows.
- Any closing pair whose two halves do not read as one exchange.

Also inspect `rejected.jsonl`. A rejection rate above ~5% at this stage means the agent
file needs work, not that you should raise the budget.

### Step 5 — Full ledger run (costly — get explicit go-ahead)

**~$8–13 and 1.5–3 h at 4 workers**, for 3,842 inserts across 1,453 conversations.
This supersedes the Task 11 brief's "~$5–8 / 1–2 h", which was priced against the
pre-Task-2 count of ~930 conversations / ~1,020 inserts.

```bash
python scripts/build_remediation_ledger.py \
  --input-dir data/output/sft/task_a \
  --triage-report data/interim/task_a_state_triage/report.json \
  --ledger-dir data/interim/task_a_remediation_ledger
```

Resumable — re-run the identical command after any interruption; `accepted.jsonl`
makes it skip completed inserts.

> **Batches are packed on conversation boundaries** (`build_batches`), so no
> conversation is ever split across two agent calls — flat slicing at the default size
> of 10 would have split ~250 of the 1,453. That matters most for closing pairs (the
> `user` ack and the terminal turn must read as one exchange) and for the 100+
> conversations needing 6–14 bridges. `--batch-size` is therefore a soft cap: a
> conversation with more inserts than the cap gets a batch to itself. Measured on the
> full queue at the default 10: **437 batches, 0 conversations split**, sizes 2–14.

### Step 6 — Final apply, with the ledger

```bash
python scripts/remediate_task_a_states.py apply \
  --input-dir data/output/sft/task_a \
  --output-dir data/output/sft/task_a_remediated \
  --ledger-dir data/interim/task_a_remediation_ledger \
  --rebuild-prompts \
  --on-unrepairable drop
```

> `--rebuild-prompts` is the D2/D5 system-prompt rebuild (so v2 rows *state* the rule
> they now demonstrate). It regenerates each **retained** row's `messages[0]` from
> current prompt code, after every repair gate has passed: the corrected rule-2 worked
> example, the default-on stay rule, and the retry budget for that row's own
> `complexity_level` (L1–L4 two attempts, L5 three; a row with no `complexity_level`
> degrades to the no-retry wording). It also refreshes
> the embedded workflow script from the **repaired** messages. Omitting the flag leaves
> system messages byte-identical to the input.
>
> `--on-unrepairable` accepts only `drop`. A row that cannot be brought onto the
> convention is by definition still violating it, so retaining it would poison the v2
> corpus and fail Step 7's `verify --strict`.
>
> `--rebuild-prompts` hard-fails (exit 2, nothing written) if `TASK_A_STAY_RULE=0` is
> set in the environment — that value would bake the frozen v1 prompt into this v2
> corpus. Unset it before running this step.

### Step 7 — Verify

```bash
python scripts/remediate_task_a_states.py verify \
  --input-dir data/output/sft/task_a_remediated --strict
python scripts/remediate_task_a_states.py diff \
  --before data/output/sft/task_a --after data/output/sft/task_a_remediated
for f in data/output/sft/task_a_remediated/*.jsonl; do
  python -m llm_workflow_agents.data.quality_profiler "$f"
done
```

`verify --strict` must print `Total violations: 0` and exit 0. `diff`'s `after:` line
must be `{'none': N}` with no other key. The profiler must show
`tool_turn_state.pct_conformant: 100.0` and zero hard defects on every file.

---

## 9. If the drop rate exceeds budget

Acceptance gate 6 — *no level loses more than 5% of its rows* — is the binding
constraint, and it is tighter than it looks. A conversation is kept only if **every one
of its inserts** is accepted.

| Level | rows | authoring rows | at risk | max drop (5%) | min conversation success | mean inserts/conv | implied per-insert acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 1,251 | 16 | 1.3% | 62 | — (no constraint) | 2.31 | — |
| L2 | 1,305 | 331 | 25.4% | 65 | 80.4% | 2.18 | 0.904 |
| L3 | 1,189 | 421 | 35.4% | 59 | 86.0% | 2.77 | 0.947 |
| L4 | 907 | 377 | **41.6%** | 45 | **88.1%** | 3.06 | **0.959** |
| L5 | 897 | 320 | 35.7% | 44 | 86.3% | 2.58 | 0.944 |

**L4 is the gate.** It needs ~96% per-insert acceptance, assuming independence. Budget
the smoke run in step 4 against that number, not against a general sense of "mostly
fine". L1 is effectively free — only 16 of its 1,251 rows need authoring at all.

### Diagnose before acting

Both scripts take the "after" directory as an argument, so you can point them at
`/tmp/task_a_deterministic_only` from step 2 as well as at the final output.

```bash
# Gate 6 -- which levels/languages actually lost rows
source .venv/bin/activate && python3 - data/output/sft/task_a_remediated <<'PY'
import collections, glob, json, sys
AFTER = sys.argv[1]
def census(d):
    c = collections.Counter()
    for p in glob.glob(f"{d}/*.jsonl"):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                c[(r.get("complexity_level"), r.get("language"))] += 1
    return c
b, a = census("data/output/sft/task_a"), census(AFTER)
print(f"{'level':6s} {'lang':12s} {'before':>7s} {'after':>7s} {'lost%':>7s}")
for k in sorted(b):
    lost = 100 * (b[k] - a.get(k, 0)) / b[k]
    flag = "  <-- OVER 5%" if lost > 5 else ""
    print(f"{k[0]:6s} {k[1]:12s} {b[k]:7d} {a.get(k,0):7d} {lost:6.1f}%{flag}")
PY

# Gate 7 (arrow glyphs) and gate 8 (retry arcs) -- both must be ~unchanged
source .venv/bin/activate && python3 - data/output/sft/task_a_remediated <<'PY'
import collections, glob, json, sys
def stats(d):
    arrows, retries = collections.Counter(), collections.Counter()
    for p in glob.glob(f"{d}/*.jsonl"):
        for line in open(p):
            if not line.strip(): continue
            r = json.loads(line)
            n_err = 0
            for m in r["messages"]:
                c = m.get("content") or ""
                if m["role"] == "assistant":
                    arrows["unicode" if "→" in c else ("ascii" if "->" in c else "none")] += 1
                if m["role"] == "tool" and '"error"' in c:
                    n_err += 1
            retries[n_err] += 1
    return arrows, retries
for d in ("data/output/sft/task_a", sys.argv[1]):
    a, r = stats(d)
    tot = sum(a.values())
    print(d)
    print("  arrows:", {k: f"{100*v/tot:.1f}%" for k, v in a.items()})
    print("  error-count-per-conversation:", dict(sorted(r.items())))
PY
```

Reference values on v1, for comparison: arrows `{'unicode': '97.0%', 'ascii': '3.0%'}`
and error-count-per-conversation `{0: 1995, 1: 3306, 2: 220, 3: 23, 4: 4, 5: 1}`. The
deterministic-only output of step 2 reproduces the arrow split exactly (97.0% / 3.0%),
which is the expected result — relabelling copies the source turn's arrow glyph.

Also read `rejected.jsonl`'s reason histogram — the fix depends entirely on *why*:

```bash
source .venv/bin/activate && python3 -c "
import collections, json, pathlib, sys
p = pathlib.Path('data/interim/task_a_remediation_ledger/rejected.jsonl')
if not p.exists():
    sys.exit('no rejected.jsonl yet -- run step 4 or 5 first')
c = collections.Counter()
for line in p.read_text().splitlines():
    if line.strip():
        for reason in json.loads(line)['reasons']:
            c[reason.split(chr(39))[0].strip()] += 1
for k, v in c.most_common(): print(f'{v:6d}  {k}')
"
```

### Then choose one

- **Reason is a formatting gate** (`does not start with required marker`, `outside
  [20, 600]`, second `[STATE:`) — the agent file is unclear, not the corpus. Fix
  `.claude/agents/corpus-remediator.md`, delete the offending lines from
  `accepted.jsonl`, and re-run step 5. The resume checkpoint makes this cheap: only
  the removed ids are re-authored. This is the common case and the right first move.
- **Reason is `agent refused`, concentrated in one domain or level** — the trajectory
  is probably wrong, not the prose. Go back to §5's drift audit on that slice, and
  tighten `plan_repair` (Tasks 2/3) rather than pressuring the agent to author over a
  bad plan.
- **Reason is `timeout` / `claude exited N`** — infrastructure. Lower `--max-workers`,
  raise `--timeout`, re-run. Costs nothing but wall time.
- **Genuinely unauthorable rows at a level that blew the gate** — do **not** proceed to
  `dvc repro`. Backfill that level/language with the fixed generator (Tasks 9/10,
  `--require-tool-stay`) before splitting, so the curriculum keeps its shape. Loosening
  a move to rescue rows is the last resort, and only after re-running §5's audit on
  whatever the loosened move produces.

Never widen the acceptance gate to make a run pass. Gate 6 exists because a level that
quietly loses a third of its rows changes what the curriculum teaches, and that
failure is invisible in every aggregate metric.

---

## 10. Lineage

- **`task-a-sft-v1`** = `data/output/sft/task_a` as of commit `93e0cf7`.
- **`task-a-sft-v2`** = `data/output/sft/task_a_remediated` + the D3 retry slice,
  merged via `scripts/concat_task_a.py`.

The D3 retry slice is a separate, teacher-API-funded generation of 500–800 L3–L5
conversations across all three language legs, carrying the retry-then-fallback arc
that v1 contains **zero** instances of. It cannot come from remediation — there is
nothing in v1 to repair into it.

DVC: the new `task_a_sft_remediate` stage sits between `task_a_sft_generate` and
`task_a_sft_clean` (Task 13). After `dvc repro`:

```bash
dvc status
dvc commit && dvc push
```

**Compare the reproduced directory hash against `dvc.lock` before `dvc push` and
before tagging.** This exact silent-lineage-drift failure has happened twice in this
project — see `docs/cat_a_state_annotation_convention_review.md` §1 and §6.6, and
CLAUDE.md Risk R12, where `dvc.lock`'s recorded output hash for `task_a_cleaned` from
one stage did not match its recorded dependency hash for the same path in the next
stage, and three downstream directories stayed stale and corrupted for weeks. Two
stages disagreeing about one directory's contents is the signature; check for it
explicitly rather than assuming `dvc repro` succeeded because it exited 0.
