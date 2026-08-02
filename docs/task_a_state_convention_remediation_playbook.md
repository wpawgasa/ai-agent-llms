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
| `relabel` — queue drains cleanly | 608 | 11.0% | none |
| `insert_handoff_turn` — stacked tool turns | 866 | 15.6% | yes |
| `append_closing_pair` — tail deficit | 599 | 10.8% | yes |
| `drop` — planner found it infeasible | **0** | 0% | — |

**Authoring queue: 1,465 conversations carrying 3,902 individual inserts.**

> **Deviation from earlier documents.** The design spec's Task-2 note records the
> split as `insert_handoff_turn` 1,150 / `append_closing_pair` 315, and the Task 11
> brief was written against a still older `930 conversations / ~1,020 inserts`
> figure. Both are stale. The 930 figure predates the removal of
> `split_fused_tool_turn`; the 1,150/315 split predates commits `9736933`, `2e739bb`,
> `97c33a7`, and `7824c90`, which made one plan accumulate every bridge in a
> conversation instead of returning at the first. The **total** authoring-case count
> (1,465) is unchanged since the spec's Task-2 note; only its split between the two
> moves and the per-conversation insert count moved.

Insert composition of the 3,902:

| Kind | Role | `required_marker` | Count |
|---|---|---|---:|
| Hand-off bridge | `assistant` | non-empty | 1,630 |
| Closing-pair terminal turn | `assistant` | non-empty (`… → TERMINAL`) | 599 |
| Shape-padding acknowledgement | `user` | `""` | **1,074** |
| Closing-pair opener | `user` | `""` | 599 |

The 1,074 shape-padding user acks are the largest single surprise relative to the
brief, which does not mention them at all. They exist because a bridge is assistant
*prose* spliced next to another assistant turn, and `find_shape_violations` rejects
two consecutive assistant prose turns. 1,059 pad the trailing edge (the stacked turn
is fused, so prose follows prose) and 15 pad the leading edge (the bridge lands
directly after an assistant turn). **43% of the authoring queue is customer voice,
not agent voice.**

Distribution of inserts per conversation (max **14**):

| inserts | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversations | 237 | 785 | 100 | 203 | 19 | 70 | 31 | 12 | 6 | 2 |

By language — the queue is **not** English-majority:

| Language | corpus rows | authoring rows | at risk | inserts |
|---|---:|---:|---:|---:|
| `code_switch` | 1,895 | 531 | 28.0% | 1,471 |
| `th` | 1,801 | 512 | 28.4% | 1,382 |
| `en` | 1,853 | 422 | 22.8% | 1,049 |

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
assistant prose turns" shape rule. That is why the move is a *pair*, and why 599 of
the 1,673 user-role inserts exist.

### 3.3 A real `insert_handoff_turn` row — `L1_009_2`

`l1_merged_20260629` line 101, code_switch, L1, `account_management`. Two inserts, both
at `position_after_msg_index` 7:

```
 [6] assistant +tool  VERIFY_IDENTITY → AUTHENTICATE   -->  VERIFY_IDENTITY → VERIFY_IDENTITY
                      (bare verify_identity call; AUTHENTICATE goes on the queue)
 [7] tool             {"error": "Service temporarily unavailable"}
     ↳ INSERT 0  assistant  [STATE: VERIFY_IDENTITY → AUTHENTICATE]  «authored bridge»
     ↳ INSERT 1  user       «authored ack»
 [8] assistant +tool  AUTHENTICATE → AUTHENTICATE   (fused: apology prose + retry call, unchanged)
```

Message 8 is a *fused* tool turn whose `from` is `AUTHENTICATE`, but `cur` is still
`VERIFY_IDENTITY` because message 6's relabel did not advance. That is the stacked-tool
infeasibility: an authored assistant turn has to bridge `VERIFY_IDENTITY →
AUTHENTICATE`. And because message 8 is fused (prose before its `<tool_call>`), the
bridge would sit prose-next-to-prose, so a user ack is padded in behind it.

**This row is also the error case.** The tool result the bridge follows is an error,
and the very next turn retries. The bridge's marker still *advances* — that advance was
displaced off message 6 and has to land somewhere — but the prose must not narrate
success. 108 of the 1,630 bridges are in this position. See the agent definition's
"situation 2" for the good/bad wording contrast.

Inserts that share a `position_after_msg_index` come out in **list order**;
`apply_plan` guarantees this by sorting on `(-position, -list_index)` and inserting
back-to-front.

---

## 4. Move ladder and rejected moves

| # | Move | Authored text | Conversations |
|---|---|---|---:|
| 1 | `relabel` — queue drains cleanly (bare and fused tool turns alike) | none | 608 |
| 2 | `pullback_fuse` — move `<tool_call>` onto the preceding `X→X` prose turn | none | **not implemented** |
| 3 | `insert_handoff_turn` — stacked tool turns; insert an assistant bridge (+ ack padding) | 1–13 msgs | 866 |
| 4 | `append_closing_pair` — tail deficit; append `user` ack + terminal `assistant` | 2+ msgs | 599 |
| 5 | `drop` — planner infeasibility, post-gate failure, or agent refusal | — | 0 planner-side |

Move 3 must bridge `cur` to the stacked turn's own `from`-state, **not** a bare
self-loop at `cur` — a self-loop would leave the trajectory exactly as stuck as before.

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
  raised the authoring bucket from 930 to 1,465 conversations, and it drove
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
| `relabel` | 608 | **608 (100%)** |
| `insert_handoff_turn` | 866 | 666 |
| `append_closing_pair` | 599 | 599 |
| **total** | 2,073 | **1,873** |

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

Stratify across all three language legs — the drift pool is 1,873 rows and the
language mix of the authoring queue is 28% code_switch / 28% th / 23% en, so an
English-only sample is not representative.

**Gate:** if more than ~3 of 30 read badly, stop and tighten `plan_repair` (Tasks 2/3)
rather than proceeding to §8 step 4. This is a quality gate, not a formality — the
authoring pass costs real money and its output is only as good as the trajectory it
is authored into.

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
| refusal | `refuse: true` → `["agent refused this insert"]` |
| required fields | `insert_id`, `role`, `content` all present |
| identity | `entry.insert_id == request.insert_id` |
| role | `entry.role == request.role` |
| tool calls | `"<tool_call>" not in content` |
| marker prefix (assistant) | `content.startswith(required_marker)` — **byte for byte, arrow glyph included** |
| second marker | no `"[STATE:"` in `content[len(required_marker):]` |
| marker (user) | no `"[STATE:"` anywhere when `required_marker == ""` |
| length | `20 <= len(content) <= 600`, counted **including** the marker |
| language | th / code_switch rows must contain Thai script (U+0E00–U+0E7F) |

The arrow-glyph rule bites in practice: of the 2,229 markered requests, **2,151 carry
a Unicode `→` and 78 carry an ASCII `->`**. `_marker()` copies the arrow from the
source turn, so both survive into the queue. An agent that normalises one to the other
fails the prefix check and loses the row.

> The language check is specified here as part of the agent's output contract because
> 1,894 of 2,853 authored inserts are th or code_switch and silent English drift is
> the single most likely quality failure. The `validate_entry` draft in the
> implementation plan's Task 12 does **not** yet include it — Task 12 must add it.

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
Triage: 5549 rows -> {'none': 3476, 'relabel': 608, 'insert_handoff_turn': 866, 'append_closing_pair': 599}
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
Apply: kept 4084, dropped 1465 ({'needs-ledger:insert_handoff_turn': 866, 'needs-ledger:append_closing_pair': 599})
```

4,084 = 3,476 `none` + 608 `relabel`. Confirm the output is clean:

```bash
python scripts/remediate_task_a_states.py verify --input-dir /tmp/task_a_deterministic_only --strict
# Total violations: 0     (exit 0)
python scripts/remediate_task_a_states.py diff \
  --before data/output/sft/task_a --after /tmp/task_a_deterministic_only
# before: {'none': 3476, 'relabel': 608, 'insert_handoff_turn': 866, 'append_closing_pair': 599}
# after:  {'none': 4084}
```

> **Known cosmetic defect.** `--on-unrepairable` is parsed but never read by
> `cmd_apply` — the code always drops. Passing `keep` does nothing. Harmless today
> because `drop` is the intended behaviour at every call site in this playbook, but do
> not rely on `keep`.

**Do not ship this directory.** It fails acceptance gate 6 badly (§9) — L4 would lose
41.6% of its rows. It exists to prove the deterministic half is sound before spending
money on the other half.

### Step 3 — Drift audit

Run §5's protocol now, before the paid step. It is the last cheap opportunity to find
a trajectory bug.

### Step 4 — Smoke the agent pass (20 inserts)

> Steps 4–5 require `scripts/build_remediation_ledger.py`, which is **Task 12 of the
> implementation plan and is not yet written**. The commands below are the contract
> Task 12 must satisfy; they have not been executed. Everything in steps 1–3, 6 and 7
> has been.

```bash
python scripts/build_remediation_ledger.py \
  --input-dir data/output/sft/task_a \
  --triage-report data/interim/task_a_state_triage/report.json \
  --ledger-dir data/interim/task_a_remediation_ledger \
  --limit 20

python3 -m json.tool < data/interim/task_a_remediation_ledger/accepted.jsonl
```

**Read all 20 entries by hand.** Deterministic gates cannot catch the failure modes
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

**~$8–13 and 1.5–3 h at 4 workers**, for 3,902 inserts across 1,465 conversations.
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

> **Set `--batch-size` deliberately.** At the default 10, **249 of the 1,465
> conversations have their inserts split across a batch boundary**, so the agent sees
> only part of the conversation's authoring job. That matters most for closing pairs
> (the `user` ack and the terminal turn must read as one exchange) and for the 100+
> conversations needing 6–14 bridges. Either raise `--batch-size` or have Task 12 pack
> batches on conversation boundaries rather than by flat slicing.

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
> they now demonstrate). The flag is **not yet implemented** in
> `remediate_task_a_states.py`'s argparse — omit it until it is, or the command will
> exit 2.

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
