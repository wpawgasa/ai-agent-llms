# Task A `task-a-sft-v2` Tool-Stay Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate the Task A SFT corpus (`data/output/sft/task_a`, `task-a-sft-v1`) so every tool-calling assistant turn annotates a self-loop `[STATE: X → X]` and the advance happens on a later turn, and redesign the teacher-driven generator so new data is born conforming — producing `task-a-sft-v2`.

**Architecture:** A pure, dependency-free library (`state_convention.py` + `state_convention_repair.py`) implements a whole-trajectory requeue algorithm that relabels tool turns deterministically wherever safe — a tool call's state attribution never changes, even for a fused prose+tool_call turn — and classifies the remainder as needing 1–2 short authored messages. A `claude -p` agent (`corpus-remediator`) authors only that missing text into a reviewable, replayable decision ledger — it never edits corpus JSONL directly. Three deterministic gate layers validate every agent output before it can enter the corpus. In parallel, the teacher-facing prompts, the generator's repair loop, the workflow-script renderer, and the quality profiler are updated so newly generated conversations conform by construction and the defect cannot recur.

**Tech Stack:** Python 3, pytest, existing `llm_workflow_agents.data` package, DVC, the `claude` CLI (already used once in this repo at `scripts/generate_sft_until_target.py::verify_batch_with_agent`).

## Global Constraints

- Project convention: prefix Python invocations with `source .venv/bin/activate &&`; use `uv`, not `pip`, for any dependency install.
- Every new/changed defect message that feeds `defective_conversation_ids` MUST be formatted `f"{cid}: ..."` (`quality_profiler.py`'s parser splits on the first `:`).
- `_TEACHER_SYSTEM_PROMPT` must retain the exact strings `"ALLOWED TRANSITIONS"`, `"TOOL PERMISSIONS PER STATE"`, and `"never invent a transition"` (`tests/unit/test_teacher_prompt_contract.py` asserts this).
- All new RNG draws inside the generator must use the per-sample `random.Random(child_seed)` instance — never a module-level `random` call — or `tests/unit/test_data_generation.py::test_concurrent_output_matches_serial` breaks.
- `_workflow_script.py::build_workflow_script`'s new `tool_turn_semantics` parameter must default to `False` — Task C's `_playbook_render.py` calls it and must render byte-identical output.
- Never mutate an input `dict`/`list` in `state_convention_repair.py`'s public functions — always return a new object (JSONL round-tripping and the DVC diff tooling depend on this).
- Arrow glyph (`→` vs `->`) must be preserved per-row exactly as found — never normalize it.
- Real corpus fixtures for tests come from `data/output/sft/task_a/l{1..5}_merged_*.jsonl` (read-only; never write into this directory from a test).

---

## Task 1: `state_convention.py` — turn parsing and the stay-violation gate

**Files:**
- Create: `src/llm_workflow_agents/data/state_convention.py`
- Test: `tests/unit/test_state_convention.py`

**Interfaces:**
- Consumes: `llm_workflow_agents.data._workflow_script._STATE_RE` (pattern: `r"\[STATE:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:→|->)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]"`).
- Produces: `TurnLabel` dataclass, `parse_assistant_turns(messages: list[dict]) -> list[TurnLabel | None]`, `find_tool_stay_violations(messages: list[dict]) -> list[str]`. These three names are imported by Tasks 2, 4, 7, and 8 — do not rename.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_state_convention.py
from llm_workflow_agents.data.state_convention import (
    TurnLabel,
    parse_assistant_turns,
    find_tool_stay_violations,
)


def _msgs(*contents_and_roles):
    return [{"role": r, "content": c, "annotations": None} for r, c in contents_and_roles]


def test_bare_tool_turn_has_empty_prose_prefix():
    msgs = _msgs(("user", "hi"), ("assistant",
        '[STATE: LOOKUP → LOOKUP]\n<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'))
    labels = parse_assistant_turns(msgs)
    assert labels[0] is None
    label = labels[1]
    assert label.from_state == "LOOKUP" and label.to_state == "LOOKUP"
    assert label.has_tool_call is True
    assert label.prose_prefix == ""
    assert label.tool_names == ("lookup",)


def test_fused_tool_turn_splits_prose_from_tail():
    msgs = _msgs(("assistant",
        '[STATE: A → B]\nLet me check that for you.\n'
        '<tool_call>{"name": "check", "arguments": {"x": 1}}</tool_call>'))
    label = parse_assistant_turns(msgs)[0]
    assert label.prose_prefix == "Let me check that for you."
    assert label.tail.startswith("<tool_call>")
    assert label.arrow == "→"


def test_parse_preserves_ascii_arrow():
    msgs = _msgs(("assistant", "[STATE: A -> A]\nok"))
    assert parse_assistant_turns(msgs)[0].arrow == "->"


def test_prose_turn_advancing_is_not_a_violation():
    msgs = _msgs(("assistant", "[STATE: A → B]\nAll set, thanks!"))
    assert find_tool_stay_violations(msgs) == []


def test_advancing_tool_turn_yields_one_violation():
    msgs = _msgs(("assistant",
        '[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'))
    violations = find_tool_stay_violations(msgs)
    assert len(violations) == 1
    assert "turn 1" in violations[0]
    assert "[A -> B]" in violations[0]
    assert "[A -> A]" in violations[0]


def test_conformant_tool_turn_yields_no_violation():
    msgs = _msgs(("assistant",
        '[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'))
    assert find_tool_stay_violations(msgs) == []


def test_ignores_non_assistant_roles():
    msgs = _msgs(("tool", '<tool_call>{"name": "t", "arguments": {}}</tool_call>'))
    assert find_tool_stay_violations(msgs) == []


def test_turn_with_no_marker_is_none_and_not_a_violation():
    msgs = _msgs(("assistant", "no marker here"))
    labels = parse_assistant_turns(msgs)
    assert labels == [None]
    assert find_tool_stay_violations(msgs) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_state_convention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_workflow_agents.data.state_convention'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_workflow_agents/data/state_convention.py
"""Single source of truth for the tool-call state-annotation convention.

Convention: an assistant turn that emits <tool_call> must annotate a
self-loop [STATE: X -> X]. The advance to a new state happens on a LATER
turn, after the tool result has been seen. See
docs/cat_a_state_annotation_convention_review.md and
docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md.

Pure stdlib, no heavy imports, so quality_profiler, data_validator,
generate_workflows, and the remediation scripts can all import this without
import cycles (mirrors the posture of _workflow_script.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm_workflow_agents.data._workflow_script import _STATE_RE

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class TurnLabel:
    """One assistant turn's state annotation and tool-call shape."""

    msg_index: int
    turn_index: int          # 1-based ordinal among assistant turns
    from_state: str
    to_state: str
    arrow: str                # "→" or "->", verbatim from the row
    has_tool_call: bool
    prose_prefix: str         # text between the marker and the first <tool_call>; "" if none
    tail: str                 # from the first "<tool_call>" onward, verbatim; "" if none
    tool_names: tuple[str, ...]


def _extract_tool_name(raw_json: str) -> str | None:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    name = parsed.get("name")
    return name if isinstance(name, str) else None


def parse_assistant_turns(messages: list[dict[str, Any]]) -> list[TurnLabel | None]:
    """Parse every assistant message into a TurnLabel; None if unparsable.

    Returns one entry per assistant message, in order (not one per input
    message) -- callers that need to map back to ``messages`` use
    ``msg_index``.
    """
    labels: list[TurnLabel | None] = []
    turn_index = 0
    for msg_index, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        turn_index += 1
        content = msg.get("content") or ""
        m = _STATE_RE.search(content)
        if not m:
            labels.append(None)
            continue
        arrow = "->" if content[m.start():m.end()].find("->") != -1 else "→"
        after_marker = content[m.end():]
        tc_match = _TOOL_CALL_RE.search(after_marker)
        if tc_match:
            prose_prefix = after_marker[: tc_match.start()].strip()
            tail = after_marker[tc_match.start():].strip()
            tool_names = tuple(
                name
                for tc in _TOOL_CALL_RE.finditer(after_marker)
                if (name := _extract_tool_name(tc.group(1))) is not None
            )
        else:
            prose_prefix, tail, tool_names = "", "", ()
        labels.append(
            TurnLabel(
                msg_index=msg_index,
                turn_index=turn_index,
                from_state=m.group(1),
                to_state=m.group(2),
                arrow=arrow,
                has_tool_call=tc_match is not None,
                prose_prefix=prose_prefix,
                tail=tail,
                tool_names=tool_names,
            )
        )
    return labels


def find_tool_stay_violations(messages: list[dict[str, Any]]) -> list[str]:
    """Return one message per tool-call turn that advances instead of staying.

    Reads inline content markers only (via parse_assistant_turns), consistent
    with _backfill_annotations' rule that content is authoritative over the
    structured annotations dict. An empty list means every tool-call turn in
    this conversation already conforms.
    """
    violations: list[str] = []
    for label in parse_assistant_turns(messages):
        if label is None or not label.has_tool_call:
            continue
        if label.from_state != label.to_state:
            violations.append(
                f"assistant turn {label.turn_index} issues a <tool_call> but "
                f"annotates an advancing transition [{label.from_state} -> "
                f"{label.to_state}]; a tool-execution turn must annotate "
                f"[{label.from_state} -> {label.from_state}] and advance on "
                f"a later turn"
            )
    return violations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_state_convention.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Validate against the real corpus**

```bash
source .venv/bin/activate && python3 - <<'PY'
import glob, json
from llm_workflow_agents.data.state_convention import find_tool_stay_violations

total_turns = total_violations = 0
for f in sorted(glob.glob("data/output/sft/task_a/l*_merged_*.jsonl")):
    for line in open(f):
        if not line.strip():
            continue
        rec = json.loads(line)
        total_violations += len(find_tool_stay_violations(rec["messages"]))
print("violations:", total_violations)  # expect 4333, per the design doc's measured baseline
PY
```

Expected output: `violations: 4333` (matches the design spec's measured baseline exactly — if it differs, the parsing logic has a bug; do not proceed until it matches).

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/state_convention.py tests/unit/test_state_convention.py
git commit -m "feat(data): add tool-call stay-convention parser and violation gate"
```

---

## Task 2: `state_convention_repair.py` — the whole-trajectory requeue algorithm

**Files:**
- Create: `src/llm_workflow_agents/data/state_convention_repair.py`
- Test: `tests/unit/test_state_convention_repair.py`

**Interfaces:**
- Consumes: `state_convention.parse_assistant_turns`, `state_convention.TurnLabel` (Task 1).
- Produces: `TurnPlan`, `InsertRequest`, `RepairPlan` dataclasses; `plan_repair(record: dict) -> RepairPlan`. `RepairPlan.move` is one of `"none" | "relabel" | "insert_handoff_turn" | "append_closing_pair" | "drop"`. Task 3 consumes `RepairPlan` to implement `apply_plan`/`verify_repaired`; Task 4's CLI consumes `plan_repair` for `triage`.

### Algorithm being implemented

```
cur = labels[0].from_state
emitted = []                      # list of (from, to) pairs, in order
pending = deque()                 # advances displaced off tool turns
for label in labels (skip None):
    if label.has_tool_call:
        if cur != label.from_state:
            → infeasible: "stacked tool turn requires an inserted hand-off"
        emitted.append((cur, cur))
        if label.to_state != label.from_state:
            pending.append(label.to_state)
    else:
        if pending:
            nxt = pending.popleft()
            emitted.append((cur, nxt)); cur = nxt
            if label.to_state != label.from_state:
                pending.append(label.to_state)
        else:
            emitted.append((cur, label.to_state)); cur = label.to_state
if pending:
    → deficit: needs an appended closing pair (one per remaining item, but
      the corpus never has more than one outstanding item at end -- assert
      len(pending) == 1 and treat >1 as infeasible/drop)
```

`move` is decided from what happened during the walk:
- `"none"` — every `label.to_state == label.from_state` already (no tool turn ever needed a requeue) AND every prose turn's original `(from, to)` matches its `emitted` pair unchanged.
- `"relabel"` — the walk completed with `pending` empty at the end and no infeasibility. This covers **every** displaced tool turn, fused or bare: a fused turn's prose and `<tool_call>` stay together in the same message, relabelled as a single self-loop at its *original* `from_state`. This is deliberate, not a simplification: the only way to guarantee a repair never changes which state a tool call is attributed to (`infer_state_tools_from_messages`) is to never move the `<tool_call>` to a different message at a different state. An earlier version of this design proposed a distinct `split_fused_tool_turn` move that physically separated the prose (kept at the wrong advancing label) from the tool call (moved to a new self-loop turn at the *destination* state) — that move was removed after implementation measurement showed it re-attributes the tool call to the destination state on ~66% of its sites, breaking the tool-placement invariant on 6.5% of the real corpus. See `docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md`'s "Core algorithm" section for the full account.
- `"insert_handoff_turn"` — a stacked-tool infeasibility occurred (`cur != label.from_state` for a tool turn with no preceding prose turn to carry the requeued advance). This now also covers the case where a fused turn's relabel-in-place doesn't advance `cur`, so an *already-bare* tool turn immediately following it is stacked exactly as if the fused turn had no prose at all.
- `"append_closing_pair"` — `pending` non-empty at the end (`len(pending) == 1`), AND the closing transition `(cur, deficit)` is a declared edge, AND `deficit` is a terminal (otherwise `"drop"`).
- `"drop"` — `len(pending) > 1` at any point, or an emitted non-self pair `(a, b)` with `a != b` is not a declared edge in `record["workflow_graph"]["transitions"]`, or the final emitted `to` is not in `record["workflow_graph"]["terminal"]`, or the closing-pair transition described above is undeclared/non-terminal.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_state_convention_repair.py
import copy
import json

from llm_workflow_agents.data.state_convention_repair import plan_repair
from llm_workflow_agents.data.state_convention import find_tool_stay_violations


def _record(messages, transitions, terminal=("TERMINAL",), initial="A"):
    return {
        "conversation_id": "T_001",
        "workflow_graph": {
            "states": sorted({t for pair in transitions for t in pair} | {initial, *terminal}),
            "transitions": [{"from": a, "to": b, "condition": "", "priority": 0} for a, b in transitions],
            "initial": initial,
            "terminal": list(terminal),
        },
        "messages": messages,
        "ground_truth": {"terminal_reached": True},
    }


def _amsg(content):
    return {"role": "assistant", "content": content, "annotations": None}


def test_conformant_record_yields_move_none():
    msgs = [
        {"role": "user", "content": "hi", "annotations": None},
        _amsg('[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: A → TERMINAL]\nDone!"),
    ]
    plan = plan_repair(_record(msgs, [("A", "TERMINAL")]))
    assert plan.move == "none"
    assert plan.turns == []


def test_case_a_two_turn_relabel():
    # tool turn wrongly claims A->B; the *next* turn stays at B->B.
    # Correct: tool turn A->A, next turn A->B.
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    assert plan.move == "relabel"
    by_index = {t.source_turn_index: t for t in plan.turns}
    assert by_index[0].from_state == "A" and by_index[0].to_state == "A"
    assert by_index[2].from_state == "A" and by_index[2].to_state == "B"


def test_fused_tool_turn_relabels_in_place_keeping_prose_and_tool_together():
    # A fused turn (prose + <tool_call> in one message) that wrongly advances
    # A->B relabels to a SINGLE self-loop A->A, keeping the prose and the
    # tool call together in the same message -- exactly like a bare tool
    # turn. This is deliberate: physically splitting the tool call into its
    # own message at the destination state would change which state the
    # tool is attributed to (infer_state_tools_from_messages), breaking the
    # "a tool's from-state never changes" invariant. See
    # docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md.
    msgs = [
        _amsg(
            '[STATE: A → B]\nChecking that now.\n'
            '<tool_call>{"name": "t1", "arguments": {}}</tool_call>'
        ),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    # Same shape as test_case_a_two_turn_relabel, just with a fused (prose +
    # tool_call) first turn instead of a bare one -- proving plan_repair
    # treats them identically.
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    assert plan.move == "relabel"
    tp = plan.turns[0]
    assert tp.from_state == "A" and tp.to_state == "A"
    assert tp.content_op == "relabel"


def test_stacked_tool_turn_after_a_fused_turn_needs_insert_handoff():
    # Because the fused turn above relabels WITHOUT advancing cur, a second,
    # already-bare tool turn immediately following it (originally marked
    # B->B, i.e. looking self-consistent on its own) cannot fire -- cur is
    # still A, not B -- so this is a stacked-tool infeasibility needing a
    # hand-off, exactly as if the fused turn had carried no prose at all.
    msgs = [
        _amsg(
            '[STATE: A → B]\nChecking that now.\n'
            '<tool_call>{"name": "t1", "arguments": {}}</tool_call>'
        ),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg('[STATE: B → B]\n<tool_call>{"name": "t2", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → TERMINAL]\nDone."),
    ]
    plan = plan_repair(_record(msgs, [("A", "B"), ("B", "TERMINAL")]))
    assert plan.move == "insert_handoff_turn"
    assert plan.inserts[0].role == "assistant"
    assert "B" in plan.inserts[0].required_marker


def test_tail_deficit_requires_append_closing_pair():
    msgs = [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ]
    plan = plan_repair(_record(msgs, [("A", "TERMINAL")]))
    assert plan.move == "append_closing_pair"
    assert len(plan.inserts) == 2  # a user ack + the assistant closing turn
    assert plan.inserts[0].role == "user"
    assert plan.inserts[1].role == "assistant"
    assert "TERMINAL" in plan.inserts[1].required_marker


def test_bare_stacked_tool_turns_need_insert_handoff():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t1", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg('[STATE: B → TERMINAL]\n<tool_call>{"name": "t2", "arguments": {}}</tool_call>'),
    ]
    plan = plan_repair(_record(msgs, [("A", "B"), ("B", "TERMINAL")]))
    assert plan.move == "insert_handoff_turn"
    assert len(plan.inserts) == 1
    assert plan.inserts[0].role == "assistant"
    assert "B" in plan.inserts[0].required_marker


def test_undeclared_edge_makes_plan_infeasible_and_drops():
    msgs = [
        _amsg('[STATE: A → Z]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ]
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    assert plan.move == "drop"
    assert plan.infeasible_reason is not None


def test_tool_from_state_never_changes():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {"x": 1}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    tool_turn = next(t for t in plan.turns if t.source_turn_index == 0)
    assert tool_turn.from_state == "A"  # unchanged even though it now self-loops


def test_arrow_glyph_preserved():
    msgs = [
        _amsg('[STATE: A -> B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B -> B]\nAll set."),
    ]
    plan = plan_repair(_record(msgs, [("A", "B")], terminal=("B",)))
    assert all(t.arrow == "->" for t in plan.turns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_state_convention_repair.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_workflow_agents/data/state_convention_repair.py
"""Deterministic repair of the tool-call stay convention (see
state_convention.py and docs/superpowers/specs/2026-07-31-task-a-tool-stay-
convention-design.md for the algorithm and invariant table).

plan_repair() is pure and never mutates its input. apply_plan() (Task 3)
consumes a RepairPlan and returns a new record.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_workflow_agents.data.state_convention import parse_assistant_turns

Move = Literal[
    "none",
    "relabel",
    "insert_handoff_turn",
    "append_closing_pair",
    "drop",
]


@dataclass(frozen=True)
class TurnPlan:
    """One post-repair assistant turn, mapped back to its source message."""

    source_turn_index: int | None   # the message's index in messages[] (label.msg_index); None if inserted
    from_state: str
    to_state: str
    arrow: str
    content_op: Literal["relabel", "insert", "verbatim"]


@dataclass(frozen=True)
class InsertRequest:
    """One message the corpus-remediator agent must author (Task 6/13)."""

    insert_id: str                       # "<file_stem>:<line_index>:<ordinal>", filled by the CLI (Task 4)
    position_after_msg_index: int         # insert immediately after this index in messages[]
    role: Literal["user", "assistant"]
    required_marker: str                  # e.g. "[STATE: B -> B]"; "" for a user-role insert


@dataclass
class RepairPlan:
    key: tuple[str, int] = (("", -1))     # (file_stem, line_index); filled by the CLI, not here
    conversation_id: str = ""
    move: Move = "none"
    turns: list[TurnPlan] = field(default_factory=list)
    inserts: list[InsertRequest] = field(default_factory=list)
    drift_turns: list[int] = field(default_factory=list)
    infeasible_reason: str | None = None


def _marker(from_state: str, to_state: str, arrow: str) -> str:
    return f"[STATE: {from_state} {arrow} {to_state}]"


def plan_repair(record: dict[str, Any]) -> RepairPlan:
    messages = record["messages"]
    wg = record["workflow_graph"]
    declared_edges = {(t["from"], t["to"]) for t in wg["transitions"]}
    terminals = set(wg["terminal"])
    conversation_id = record.get("conversation_id", "")

    labels = [l for l in parse_assistant_turns(messages) if l is not None]
    if not labels:
        return RepairPlan(conversation_id=conversation_id, move="drop",
                           infeasible_reason="no parsable assistant turns")

    cur = labels[0].from_state
    pending: deque[str] = deque()
    turns: list[TurnPlan] = []
    drift_turns: list[int] = []
    any_relabelled = False

    for label in labels:
        if label.has_tool_call:
            if cur != label.from_state:
                return RepairPlan(
                    conversation_id=conversation_id, move="insert_handoff_turn",
                    inserts=[InsertRequest(
                        insert_id="", position_after_msg_index=label.msg_index - 1,
                        role="assistant", required_marker=_marker(cur, label.from_state, label.arrow),
                    )],
                )
            if label.to_state != label.from_state:
                # Relabel the WHOLE message -- prose and <tool_call>, if
                # fused, stay together unchanged -- to a self-loop at its
                # ORIGINAL from_state. Never split a fused turn: moving the
                # tool call to a different message at the destination state
                # would change infer_state_tools_from_messages' attribution
                # for that tool, breaking the "tool from-state never
                # changes" invariant. The displaced advance is pushed onto
                # `pending` and drained by a later non-tool turn, exactly
                # like a bare tool turn.
                pending.append(label.to_state)
                any_relabelled = True
                turns.append(TurnPlan(label.msg_index, cur, cur, label.arrow, "relabel"))
            else:
                turns.append(TurnPlan(label.msg_index, cur, cur, label.arrow, "verbatim"))
        else:
            if pending:
                nxt = pending.popleft()
                if nxt != label.to_state or label.from_state != cur:
                    drift_turns.append(label.turn_index)
                turns.append(TurnPlan(label.msg_index, cur, nxt, label.arrow, "relabel"))
                cur = nxt
                if label.to_state != label.from_state and label.to_state != nxt:
                    pending.append(label.to_state)
            else:
                turns.append(TurnPlan(label.msg_index, cur, label.to_state, label.arrow, "verbatim"))
                cur = label.to_state

    if len(pending) > 1:
        return RepairPlan(conversation_id=conversation_id, move="drop",
                           infeasible_reason=f"deficit of {len(pending)} states at end (only 1 supported)")

    for tp in turns:
        if tp.from_state != tp.to_state and (tp.from_state, tp.to_state) not in declared_edges:
            return RepairPlan(
                conversation_id=conversation_id, move="drop",
                infeasible_reason=f"undeclared transition [{tp.from_state} -> {tp.to_state}]",
            )

    if pending:
        deficit = pending[0]
        if (cur, deficit) not in declared_edges:
            return RepairPlan(conversation_id=conversation_id, move="drop",
                               infeasible_reason=f"undeclared closing transition [{cur} -> {deficit}]")
        if deficit not in terminals:
            return RepairPlan(conversation_id=conversation_id, move="drop",
                               infeasible_reason=f"deficit target '{deficit}' is not a terminal")
        return RepairPlan(
            conversation_id=conversation_id, move="append_closing_pair",
            inserts=[
                InsertRequest(insert_id="", position_after_msg_index=len(messages) - 1,
                              role="user", required_marker=""),
                InsertRequest(insert_id="", position_after_msg_index=len(messages),
                              role="assistant", required_marker=_marker(cur, deficit, labels[-1].arrow)),
            ],
        )

    if cur not in terminals:
        return RepairPlan(conversation_id=conversation_id, move="drop",
                           infeasible_reason=f"final state '{cur}' is not a terminal")

    if not any_relabelled:
        return RepairPlan(conversation_id=conversation_id, move="none")

    return RepairPlan(conversation_id=conversation_id, move="relabel", turns=turns, drift_turns=drift_turns)
```

Note: `TurnPlan.source_turn_index` is the assistant message's actual index in `messages[]` (`label.msg_index`) — not an index into an assistant-only list. This matters because Task 1's `parse_assistant_turns` returns one entry per *input* message (`None` for non-assistant messages), so `messages[source_turn_index]` is always the correct message to rewrite directly; `apply_plan` (Task 3) must index `messages` with it, never a filtered assistant-only list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_state_convention_repair.py -v`
Expected: PASS (9 tests: the original 7 plus `test_fused_tool_turn_relabels_in_place_keeping_prose_and_tool_together` and `test_stacked_tool_turn_after_a_fused_turn_needs_insert_handoff`, which together replace the removed `test_stacked_tool_turns_split_fused_head`).

- [ ] **Step 5: Measure `pullback_fuse` yield against the real corpus (decides whether Task 3 needs a 3rd move)**

```bash
source .venv/bin/activate && python3 - <<'PY'
import glob, json
from llm_workflow_agents.data.state_convention_repair import plan_repair

counts = {}
pullback_candidates = 0
for f in sorted(glob.glob("data/output/sft/task_a/l*_merged_*.jsonl")):
    for line in open(f):
        if not line.strip():
            continue
        rec = json.loads(line)
        plan = plan_repair(rec)
        counts[plan.move] = counts.get(plan.move, 0) + 1
        # A pullback_fuse candidate: an insert_handoff_turn case where the
        # turn immediately preceding the stacked tool call is itself a
        # prose-only self-loop in the same state (so the tool call could be
        # moved onto it instead of inserting new text).
        if plan.move == "insert_handoff_turn":
            msgs = rec["messages"]
            pos = plan.inserts[0].position_after_msg_index
            if pos >= 0 and msgs[pos].get("role") == "assistant":
                pullback_candidates += 1
print(counts)
print("pullback_fuse candidates:", pullback_candidates)
PY
```

There is no fixed target bucket table to compare against here: the design spec's original exploratory estimate (535 `split_fused_tool_turn` / 443 `insert_handoff_turn` / 599 `append_closing_pair`) was measured against a different, since-removed move ladder that classified fused and bare tool turns differently, and its own authored-text buckets were shown (by Task 2's implementer) to double-count conversations needing more than one kind of fix — they are not a partition (443 + 599 = 1,042 against a 930-conversation headline, a 112 overlap). The only numbers worth checking now: (1) every one of the 5,549 conversations lands in exactly one bucket (`sum(counts.values()) == 5549`, `drop` count included) — if not, there's a real bug; (2) `none` should land at or very near 3,476 (the fraction with zero forward-annotated tool turns is a corpus-content fact this refactor cannot change). Record the actual measured counts in the playbook (Task 11) as the new authoritative baseline — do not chase the old numbers. **Already measured** (Task 2's fix-round re-run, code-verified): `none` 3,476 / `relabel` 608 / `insert_handoff_turn` 1,150 / `append_closing_pair` 315 / `drop` 0. Note `relabel` did **not** grow relative to the old spec's 608 figure — the entire old split bucket redistributed into the two authored-text moves instead (they are the only moves that can absorb a stacked-tool case once splitting is off the table), so authoring cases rise from 930 to 1,465 (see the design spec's Task-2 note). If `pullback_fuse candidates` is a meaningful fraction (>15%) of the `insert_handoff_turn` bucket, add a `pullback_fuse` move to `plan_repair` before Task 3 (move the tool call onto the preceding self-loop turn, zero authored text, and — like `relabel` — it must never change which state the tool call is attributed to); Task 2's implementer already measured this at 11 candidates (1.6%, well below threshold) and skipped it — no further action needed here.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/state_convention_repair.py tests/unit/test_state_convention_repair.py
git commit -m "feat(data): add whole-trajectory requeue algorithm for stay-convention repair"
```

---

## Task 3: `apply_plan`, `verify_repaired`, `rederive_ground_truth`

**Files:**
- Modify: `src/llm_workflow_agents/data/state_convention_repair.py`
- Test: `tests/unit/test_state_convention_repair.py` (append)

**Interfaces:**
- Consumes: `RepairPlan`, `TurnPlan`, `InsertRequest` (Task 2); `find_tool_stay_violations` (Task 1); `_workflow_script.find_continuity_violations`, `find_shape_violations`, `find_tool_placement_violations`, `infer_state_tools_from_messages` (existing, `_workflow_script.py`); `generate_workflows._backfill_annotations`, `generate_workflows._extract_ground_truth`, `generate_workflows.WorkflowGraph`, `generate_workflows.WorkflowState`, `generate_workflows.WorkflowTransition` (existing).
- Produces: `apply_plan(record: dict, plan: RepairPlan, ledger_entries: dict[str, dict] | None = None) -> dict | None` (returns `None` if the plan needs inserts not present in `ledger_entries`), `verify_repaired(record: dict) -> list[str]`, `rederive_ground_truth(record: dict) -> dict`. Task 4's CLI and Task 13's ledger driver both call `apply_plan` and `verify_repaired` directly — do not rename.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_state_convention_repair.py
from llm_workflow_agents.data.state_convention_repair import (
    apply_plan, verify_repaired, rederive_ground_truth,
)


def test_rederive_ground_truth_is_noop_on_conformant_record():
    msgs = [
        _amsg('[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: A → TERMINAL]\nDone!"),
    ]
    rec = _record(msgs, [("A", "TERMINAL")])
    rec["ground_truth"] = {
        "state_sequence": [{"from": "A", "to": "A"}, {"from": "A", "to": "TERMINAL"}],
        "tool_calls": [{"name": "t", "arguments": {}}],
        "tool_chain_dependencies": [[{"name": "t", "arguments": {}}]],
        "terminal_state": "TERMINAL",
        "terminal_reached": True,
    }
    before = copy.deepcopy(rec["ground_truth"])
    after = rederive_ground_truth(rec)
    assert after == before


def test_apply_plan_relabel_updates_marker_annotations_and_ground_truth():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    rec = _record(msgs, [("A", "B")], terminal=("B",))
    plan = plan_repair(rec)
    repaired = apply_plan(rec, plan)
    assert repaired is not rec  # never mutates input
    assert "[STATE: A → A]" in repaired["messages"][0]["content"]
    assert repaired["messages"][0]["annotations"]["state_transition"] == {"from": "A", "to": "A"}
    assert "[STATE: A → B]" in repaired["messages"][2]["content"]
    assert repaired["ground_truth"]["state_sequence"] == [
        {"from": "A", "to": "A"}, {"from": "A", "to": "B"},
    ]
    assert verify_repaired(repaired) == []


def test_apply_plan_does_not_mutate_input():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    rec = _record(msgs, [("A", "B")], terminal=("B",))
    original = copy.deepcopy(rec)
    plan = plan_repair(rec)
    apply_plan(rec, plan)
    assert rec == original


def test_apply_plan_relabels_fused_turn_without_splitting_or_reattributing_tool():
    msgs = [
        _amsg(
            '[STATE: A → B]\nChecking that now.\n'
            '<tool_call>{"name": "t1", "arguments": {}}</tool_call>'
        ),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: B → B]\nAll set."),
    ]
    rec = _record(msgs, [("A", "B")], terminal=("B",))
    plan = plan_repair(rec)
    repaired = apply_plan(rec, plan)
    assert verify_repaired(repaired) == []
    assistant_msgs = [m for m in repaired["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) == 2  # unchanged from input -- no split, no inserted turn
    assert assistant_msgs[0]["content"].startswith("[STATE: A → A]")
    assert "Checking that now." in assistant_msgs[0]["content"]
    assert "<tool_call>" in assistant_msgs[0]["content"]
    assert assistant_msgs[1]["content"].startswith("[STATE: A → B]")
    from llm_workflow_agents.data._workflow_script import infer_state_tools_from_messages
    assert infer_state_tools_from_messages(msgs) == infer_state_tools_from_messages(repaired["messages"])


def test_apply_plan_returns_none_without_required_ledger_entry():
    msgs = [
        _amsg('[STATE: A → B]\n<tool_call>{"name": "t1", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg('[STATE: B → TERMINAL]\n<tool_call>{"name": "t2", "arguments": {}}</tool_call>'),
    ]
    rec = _record(msgs, [("A", "B"), ("B", "TERMINAL")])
    plan = plan_repair(rec)
    assert plan.move == "insert_handoff_turn"
    assert apply_plan(rec, plan, ledger_entries=None) is None


def test_verify_repaired_catches_state_sequence_mismatch():
    msgs = [
        _amsg("[STATE: A → TERMINAL]\nDone."),
    ]
    rec = _record(msgs, [("A", "TERMINAL")])
    rec["ground_truth"] = {"state_sequence": [{"from": "A", "to": "WRONG"}],
                            "tool_calls": [], "tool_chain_dependencies": [],
                            "terminal_state": "TERMINAL", "terminal_reached": True}
    violations = verify_repaired(rec)
    assert any("state_sequence" in v for v in violations)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_state_convention_repair.py -v -k "apply_plan or verify_repaired or rederive"`
Expected: FAIL with `ImportError: cannot import name 'apply_plan'`

- [ ] **Step 3: Write the implementation**

Append to `src/llm_workflow_agents/data/state_convention_repair.py`:

```python
import copy

from llm_workflow_agents.data._workflow_script import (
    find_continuity_violations,
    find_shape_violations,
    find_tool_placement_violations,
    infer_state_tools_from_messages,
)
# _marker() is already defined earlier in this same module (Task 2's Step 3) --
# apply_plan() below is appended to the same file, not a new module.


def apply_plan(
    record: dict[str, Any],
    plan: RepairPlan,
    ledger_entries: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return a NEW record with ``plan`` applied, or None if required ledger
    entries are missing. Never mutates ``record``."""
    if plan.move == "none":
        return copy.deepcopy(record)
    if plan.move == "drop":
        return None

    record = copy.deepcopy(record)
    messages = record["messages"]

    if plan.move == "relabel":
        for tp in plan.turns:
            if tp.content_op == "verbatim":
                continue
            msg_index = tp.source_turn_index  # a message index -- see Task 2's note
            new_marker = _marker(tp.from_state, tp.to_state, tp.arrow)
            content = _STATE_RE_SUB(messages[msg_index]["content"], new_marker)
            messages[msg_index]["content"] = content
            messages[msg_index]["annotations"] = {
                **(messages[msg_index].get("annotations") or {}),
                "state_transition": {"from": tp.from_state, "to": tp.to_state},
            }

    if plan.move in ("insert_handoff_turn", "append_closing_pair"):
        if ledger_entries is None:
            return None
        # inserts are applied back-to-front to keep earlier positions valid
        for req in sorted(plan.inserts, key=lambda r: -r.position_after_msg_index):
            entry = ledger_entries.get(req.insert_id)
            if entry is None:
                return None
            messages.insert(req.position_after_msg_index + 1, {
                "role": req.role, "content": entry["content"], "annotations": None,
            })

    record["ground_truth"] = rederive_ground_truth(record)
    return record


def _STATE_RE_SUB(content: str, new_marker: str) -> str:
    from llm_workflow_agents.data._workflow_script import _STATE_RE
    return _STATE_RE.sub(new_marker, content, count=1)


def rederive_ground_truth(record: dict[str, Any]) -> dict[str, Any]:
    """Re-derive ground_truth from (possibly repaired) messages, merging into
    the existing dict so fields like terminal_reached survive. A byte no-op
    on an already-conformant record."""
    from llm_workflow_agents.data.generate_workflows import (
        WorkflowGraph, WorkflowState, WorkflowTransition,
        _backfill_annotations, _extract_ground_truth,
    )

    messages = record["messages"]
    _backfill_annotations(messages)
    wg = record["workflow_graph"]
    states = [WorkflowState(id=name, name=name) for name in wg["states"]]
    transitions = [
        WorkflowTransition(from_state=t["from"], to_state=t["to"],
                            condition=t.get("condition", ""), priority=t.get("priority", 0))
        for t in wg["transitions"]
    ]
    workflow = WorkflowGraph(
        states=states, transitions=transitions,
        initial_state=wg["initial"], terminal_states=list(wg["terminal"]),
    )
    fresh = _extract_ground_truth(messages, workflow)
    gt = dict(record.get("ground_truth") or {})
    gt.update(fresh)
    return gt


def verify_repaired(record: dict[str, Any]) -> list[str]:
    """The post-repair gate. Empty list == accept."""
    messages = record["messages"]
    wg = record["workflow_graph"]
    violations: list[str] = []
    violations += find_tool_stay_violations(messages)
    violations += find_continuity_violations(messages, wg["initial"], set(wg["terminal"]))
    violations += find_shape_violations(messages, record.get("conversation_initiator", "user"))

    schema_names = {t["function"]["name"] for t in record.get("tool_schemas", [])} or None
    allowed_by_state = infer_state_tools_from_messages(messages)  # self-consistency: never tightens
    violations += find_tool_placement_violations(allowed_by_state, messages, schema_names)

    declared = {(t["from"], t["to"]) for t in wg["transitions"]}
    from llm_workflow_agents.data.state_convention import parse_assistant_turns
    labels = [l for l in parse_assistant_turns(messages) if l is not None]
    for label in labels:
        if label.from_state != label.to_state and (label.from_state, label.to_state) not in declared:
            violations.append(f"undeclared transition [{label.from_state} -> {label.to_state}]")

    expected_seq = [{"from": l.from_state, "to": l.to_state} for l in labels]
    actual_seq = (record.get("ground_truth") or {}).get("state_sequence")
    if actual_seq != expected_seq:
        violations.append(
            f"ground_truth.state_sequence {actual_seq} does not match message "
            f"markers {expected_seq}"
        )
    return violations
```

Note on `allowed_by_state` in `verify_repaired`: it is inferred from the *post-repair* messages themselves, so `find_tool_placement_violations` here can never fire from this check alone (a tool is always "allowed" in the state the repair itself attributed it to) — its real job is structural (catching a `None`/malformed messages list). The **actual** invariant — "the repair did not change which state any tool is called from" — is a stronger, separate check: compare `infer_state_tools_from_messages(original_messages)` against `infer_state_tools_from_messages(repaired_messages)` for equality. Add this as an explicit assertion in Task 4's CLI (`apply` subcommand), not inside `verify_repaired` itself, since `verify_repaired` only sees one record at a time by contract in this task's tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_state_convention_repair.py -v`
Expected: PASS (all tests from Task 2 and Task 3)

- [ ] **Step 5: Validate against 20 real conformant + 20 real relabel rows**

```bash
source .venv/bin/activate && python3 - <<'PY'
import glob, json
from llm_workflow_agents.data.state_convention_repair import plan_repair, apply_plan, verify_repaired

seen = {"none": 0, "relabel": 0}
for f in sorted(glob.glob("data/output/sft/task_a/l*_merged_*.jsonl")):
    for line in open(f):
        if not line.strip():
            continue
        rec = json.loads(line)
        plan = plan_repair(rec)
        if plan.move in seen and seen[plan.move] < 20:
            repaired = apply_plan(rec, plan)
            v = verify_repaired(repaired)
            if v:
                print("FAIL", rec["conversation_id"], plan.move, v)
            seen[plan.move] += 1
    if all(v >= 20 for v in seen.values()):
        break
print(seen)
PY
```

Expected: no `FAIL` lines printed; `seen` shows 20/20/20 (or fewer if a bucket has <20 total rows in the files scanned so far — acceptable, just confirms zero failures on what was checked).

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/state_convention_repair.py tests/unit/test_state_convention_repair.py
git commit -m "feat(data): add apply_plan/verify_repaired/rederive_ground_truth to the repair library"
```

---

## Task 4: `scripts/remediate_task_a_states.py` — triage/apply/verify/diff CLI

**Files:**
- Create: `scripts/remediate_task_a_states.py`
- Test: `tests/unit/test_remediate_task_a_states.py`

**Interfaces:**
- Consumes: `state_convention_repair.plan_repair/apply_plan/verify_repaired` (Tasks 2–3), `state_convention.find_tool_stay_violations` (Task 1).
- Produces: a `triage` report JSON at a path the operator chooses (schema below); an `apply`-produced output directory of JSONL files; `key = (file_stem, line_index)` as the row identity used throughout (per Risk R2 — 9 of 5,549 `conversation_id`s repeat, so `conversation_id` is a cross-check field only, never a dict key). Task 13's ledger driver consumes the `triage` report's `records[].inserts[].insert_id` format: `f"{file_stem}:{line_index}:{ordinal}"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_remediate_task_a_states.py
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "remediate_task_a_states.py"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _record(cid, messages, transitions, terminal=("TERMINAL",), initial="A"):
    return {
        "conversation_id": cid,
        "workflow_graph": {
            "states": sorted({t for pair in transitions for t in pair} | {initial, *terminal}),
            "transitions": [{"from": a, "to": b, "condition": "", "priority": 0} for a, b in transitions],
            "initial": initial, "terminal": list(terminal),
        },
        "messages": messages,
        "ground_truth": {"terminal_reached": True},
        "tool_schemas": [],
        "conversation_initiator": "user",
    }


def _amsg(content):
    return {"role": "assistant", "content": content, "annotations": None}


def test_triage_reports_bucket_counts(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    conformant = _record("A_001", [
        _amsg('[STATE: A → A]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: A → TERMINAL]\nDone!"),
    ], [("A", "TERMINAL")])
    needs_relabel = _record("A_002", [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: TERMINAL → TERMINAL]\nAll set."),
    ], [("A", "TERMINAL")])
    _write_jsonl(input_dir / "l1_merged_test.jsonl", [conformant, needs_relabel])

    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "triage", "--input-dir", str(input_dir),
         "--report", str(report_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["totals"]["rows"] == 2
    assert report["by_move"]["none"] == 1
    assert report["by_move"]["relabel"] == 1
    assert report["records"][1]["conversation_id"] == "A_002"
    assert report["records"][1]["key"] == ["l1_merged_test", 1]


def test_apply_writes_repaired_output_and_drops_unrepairable(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    good = _record("B_001", [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
        {"role": "tool", "content": "{}", "annotations": None},
        _amsg("[STATE: TERMINAL → TERMINAL]\nAll set."),
    ], [("A", "TERMINAL")])
    unrepairable = _record("B_002", [
        _amsg('[STATE: A → Z]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ], [("A", "TERMINAL")])
    _write_jsonl(input_dir / "l1_merged_test.jsonl", [good, unrepairable])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "apply", "--input-dir", str(input_dir),
         "--output-dir", str(output_dir), "--on-unrepairable", "drop"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out_lines = (output_dir / "l1_merged_test.jsonl").read_text().splitlines()
    assert len(out_lines) == 1
    kept = json.loads(out_lines[0])
    assert kept["conversation_id"] == "B_001"
    assert "[STATE: A → A]" in kept["messages"][0]["content"]


def test_verify_strict_exits_nonzero_on_violation(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    still_broken = _record("C_001", [
        _amsg('[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'),
    ], [("A", "TERMINAL")])
    _write_jsonl(bad_dir / "l1_merged_test.jsonl", [still_broken])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--input-dir", str(bad_dir), "--strict"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_remediate_task_a_states.py -v`
Expected: FAIL — `scripts/remediate_task_a_states.py` does not exist.

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Deterministic CLI for the tool-call stay-convention remediation.

No network, no LLM calls -- authoring the ~17% of conversations that need
new prose is scripts/build_remediation_ledger.py's job (a separate, costly,
explicitly-invoked step). This script only classifies (triage), applies
deterministic + ledger-supplied repairs (apply), re-checks a directory
(verify), and reports before/after deltas (diff).

Usage:
    python scripts/remediate_task_a_states.py triage --input-dir DIR --report PATH
    python scripts/remediate_task_a_states.py apply  --input-dir DIR --output-dir DIR
                                                       [--ledger-dir DIR] [--on-unrepairable drop|keep]
    python scripts/remediate_task_a_states.py verify --input-dir DIR [--strict]
    python scripts/remediate_task_a_states.py diff   --before DIR --after DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_workflow_agents.data.state_convention_repair import (  # noqa: E402
    apply_plan, plan_repair, verify_repaired,
)
from llm_workflow_agents.data._workflow_script import infer_state_tools_from_messages  # noqa: E402


def _iter_records(input_dir: Path):
    for path in sorted(glob.glob(str(input_dir / "*.jsonl"))):
        stem = Path(path).stem
        with open(path) as fh:
            for line_index, line in enumerate(fh):
                if not line.strip():
                    continue
                yield stem, line_index, path, json.loads(line)


def cmd_triage(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    totals = Counter()
    by_move = Counter()
    by_level = Counter()
    by_language = Counter()
    records_out = []
    n = 0
    for stem, line_index, _path, rec in _iter_records(input_dir):
        if args.limit and n >= args.limit:
            break
        n += 1
        plan = plan_repair(rec)
        totals["rows"] += 1
        by_move[plan.move] += 1
        by_level[rec.get("complexity_level", "?")] += 1
        by_language[rec.get("language", "?")] += 1
        inserts = [
            {
                "insert_id": f"{stem}:{line_index}:{i}",
                "position_after_msg_index": ins.position_after_msg_index,
                "role": ins.role,
                "required_marker": ins.required_marker,
            }
            for i, ins in enumerate(plan.inserts)
        ]
        records_out.append({
            "key": [stem, line_index],
            "conversation_id": rec.get("conversation_id", ""),
            "complexity_level": rec.get("complexity_level", ""),
            "language": rec.get("language", ""),
            "domain": rec.get("domain", ""),
            "move": plan.move,
            "drift_turns": plan.drift_turns,
            "inserts": inserts,
            "infeasible_reason": plan.infeasible_reason,
        })
    report = {
        "input_dir": str(input_dir),
        "totals": dict(totals),
        "by_move": dict(by_move),
        "by_level": dict(by_level),
        "by_language": dict(by_language),
        "records": records_out,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Triage: {totals['rows']} rows -> {dict(by_move)}")
    return 0


def _load_ledger(ledger_dir: Path | None) -> dict[str, dict]:
    if ledger_dir is None:
        return {}
    accepted = ledger_dir / "accepted.jsonl"
    if not accepted.exists():
        return {}
    entries = {}
    for line in accepted.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            entries[entry["insert_id"]] = entry
    return entries


def cmd_apply(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_entries = _load_ledger(Path(args.ledger_dir) if args.ledger_dir else None)

    kept = dropped = 0
    drop_reasons = Counter()
    by_file: dict[str, list[dict]] = {}
    for stem, line_index, path, rec in _iter_records(input_dir):
        plan = plan_repair(rec)
        # assign insert_ids matching the triage convention so a ledger built
        # from a `triage` report lines up with this pass
        for i, ins in enumerate(plan.inserts):
            ins.insert_id = f"{stem}:{line_index}:{i}"
        before_tools = infer_state_tools_from_messages(rec["messages"])
        repaired = apply_plan(rec, plan, ledger_entries=ledger_entries or None)
        if repaired is None:
            dropped += 1
            drop_reasons[plan.infeasible_reason or f"needs-ledger:{plan.move}"] += 1
            continue
        after_tools = infer_state_tools_from_messages(repaired["messages"])
        if before_tools != after_tools:
            dropped += 1
            drop_reasons["tool-from-state-changed"] += 1
            continue
        violations = verify_repaired(repaired)
        if violations:
            dropped += 1
            drop_reasons["post-gate-failed"] += 1
            continue
        kept += 1
        by_file.setdefault(Path(path).stem, []).append(repaired)

    for stem, records in by_file.items():
        out_path = output_dir / f"{stem}.jsonl"
        with open(out_path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Apply: kept {kept}, dropped {dropped} ({dict(drop_reasons)})")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    total_violations = 0
    for _stem, _idx, _path, rec in _iter_records(input_dir):
        violations = verify_repaired(rec)
        if violations:
            total_violations += len(violations)
            print(f"{rec.get('conversation_id')}: {violations}")
    print(f"Total violations: {total_violations}")
    if args.strict and total_violations:
        return 1
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    before = Counter(plan_repair(rec).move for *_ignore, rec in _iter_records(Path(args.before)))
    after = Counter(plan_repair(rec).move for *_ignore, rec in _iter_records(Path(args.after)))
    print("before:", dict(before))
    print("after: ", dict(after))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_triage = sub.add_parser("triage")
    p_triage.add_argument("--input-dir", required=True)
    p_triage.add_argument("--report", required=True)
    p_triage.add_argument("--limit", type=int, default=None)
    p_triage.set_defaults(func=cmd_triage)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--input-dir", required=True)
    p_apply.add_argument("--output-dir", required=True)
    p_apply.add_argument("--ledger-dir", default=None)
    p_apply.add_argument("--on-unrepairable", choices=["drop", "keep"], default="drop")
    p_apply.set_defaults(func=cmd_apply)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--input-dir", required=True)
    p_verify.add_argument("--strict", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_diff = sub.add_parser("diff")
    p_diff.add_argument("--before", required=True)
    p_diff.add_argument("--after", required=True)
    p_diff.set_defaults(func=cmd_diff)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `--on-unrepairable keep` is accepted by the parser per the design's CLI surface but Step 3 above only implements `drop` behavior end-to-end (`keep` degrades to the same drop path since a `None` `apply_plan` result cannot be "kept" as-is). If a future task needs `keep` (writing the pre-repair record through verbatim, flagged), extend `cmd_apply`'s `if repaired is None` branch — not required for this plan's acceptance gates, which specify `--on-unrepairable drop` as the default used in production.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_remediate_task_a_states.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run `triage` against the real corpus and sanity-check the bucket table**

```bash
source .venv/bin/activate && mkdir -p data/interim/task_a_state_triage && \
python scripts/remediate_task_a_states.py triage \
  --input-dir data/output/sft/task_a \
  --report data/interim/task_a_state_triage/report.json
```

This is the authoritative, code-verified version of the exploratory numbers used throughout the design doc, superseding them — record the actual `by_move` counts in the playbook (Task 11) rather than comparing against any earlier estimate (Task 2's implementer already found the original bucket table was not a partition and does not correspond to the final, simplified move ladder — see the design spec's "Core algorithm" section). The only checks that matter: every one of the 5,549 conversations lands in exactly one bucket, and `none` lands at or very near 3,476.

- [ ] **Step 6: Commit**

```bash
git add scripts/remediate_task_a_states.py tests/unit/test_remediate_task_a_states.py
git commit -m "feat(scripts): add deterministic triage/apply/verify/diff CLI for stay-convention remediation"
```

**Do not commit `data/interim/task_a_state_triage/report.json`** to git in this task — it is a large generated artifact; it gets DVC-tracked in Task 14.

---

## Task 5: Prompt edits — fix the wrong example, promote `STAY_RULE`, rewrite the retry rule, invert the flag

**Files:**
- Modify: `src/llm_workflow_agents/data/system_prompt.py`
- Modify: `tests/unit/test_stay_rule_flag.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `FORMAT_RULES` (module constant, unchanged name/type: `str`), `STAY_RULE` (unchanged name/type: `str`), `build_enriched_system_prompt` (unchanged signature). Task 8 (`generate_workflows.py`) and Task 6 (`quality_profiler`/`data_validator`) do not import from this module, but the *rendered prompt text* these functions produce is what Task 1's `find_tool_stay_violations` is checking corpora against — no code coupling, just a semantic dependency already satisfied.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_stay_rule_flag.py (full replacement)
import os

import pytest

from llm_workflow_agents.data import system_prompt


BASELINE_V1_FORMAT_RULES = system_prompt.FORMAT_RULES  # captured below once fixed; see Step 3 note


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("TASK_A_STAY_RULE", raising=False)
    yield


def test_default_prompt_contains_stay_rule():
    assert "do NOT advance" in system_prompt.FORMAT_RULES or "does NOT advance" in system_prompt.FORMAT_RULES


def test_rule_2_worked_example_is_a_self_loop():
    # The historical bug: rule 2's example showed an ADVANCING transition on
    # a tool-call turn, actively teaching the defect. It must now show a
    # self-loop, matching the rule 1 syntax example's convention.
    assert "[STATE: VERIFY_PATIENT → TERMINAL]" not in system_prompt.FORMAT_RULES
    assert "[STATE: VERIFY_PATIENT → VERIFY_PATIENT]" in system_prompt.FORMAT_RULES


def test_retry_rule_states_a_budget_and_fallback():
    assert "retry" in system_prompt.FORMAT_RULES.lower()
    assert "N " in system_prompt.FORMAT_RULES or "{n}" not in system_prompt.FORMAT_RULES
    # rule must not just say "attempt recovery before escalating" verbatim any more
    assert "attempt recovery before escalating" not in system_prompt.FORMAT_RULES


def test_only_exact_0_disables_the_rule(monkeypatch):
    monkeypatch.setenv("TASK_A_STAY_RULE", "0")
    import importlib
    reloaded = importlib.reload(system_prompt)
    assert "do NOT advance" not in reloaded.FORMAT_RULES and "does NOT advance" not in reloaded.FORMAT_RULES
    monkeypatch.delenv("TASK_A_STAY_RULE", raising=False)
    importlib.reload(system_prompt)  # restore default for subsequent tests


def test_disabled_prompt_is_byte_identical_to_v1_baseline(monkeypatch):
    monkeypatch.setenv("TASK_A_STAY_RULE", "0")
    import importlib
    reloaded = importlib.reload(system_prompt)
    # This golden string is the exact v1 FORMAT_RULES text every existing
    # checkpoint (ckpt-500, ckpt-1770) was trained against. Do not edit this
    # string when changing the default prompt -- it exists specifically to
    # prove the opt-out path is unchanged.
    from tests.unit._v1_format_rules_golden import V1_FORMAT_RULES
    assert reloaded.FORMAT_RULES == V1_FORMAT_RULES
    monkeypatch.delenv("TASK_A_STAY_RULE", raising=False)
    importlib.reload(system_prompt)
```

Also create the golden fixture (captured from the current, pre-edit file — run this *before* Step 3 below):

```bash
source .venv/bin/activate && python3 - <<'PY'
from llm_workflow_agents.data.system_prompt import FORMAT_RULES
with open("tests/unit/_v1_format_rules_golden.py", "w") as f:
    f.write("# Auto-captured from system_prompt.py before the 2026-07-31 stay-rule promotion.\n")
    f.write("# Do NOT edit -- this is the exact byte content every existing Cat A checkpoint\n")
    f.write("# (ckpt-500, ckpt-1770) was trained against.\n")
    f.write(f"V1_FORMAT_RULES = {FORMAT_RULES!r}\n")
PY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_stay_rule_flag.py -v`
Expected: FAIL — `test_rule_2_worked_example_is_a_self_loop` and `test_retry_rule_states_a_budget_and_fallback` fail against the current file; `test_only_exact_0_disables_the_rule` fails because the flag is currently opt-in, not opt-out.

- [ ] **Step 3: Edit `system_prompt.py`**

Replace the module's flag logic and `FORMAT_RULES` construction (currently lines 15–79) with:

```python
# Default-on since 2026-07-31 (task-a-sft-v2). Opt-out via TASK_A_STAY_RULE=0
# to reproduce the exact v1 prompt every existing checkpoint (ckpt-500,
# ckpt-1770) was trained against -- see
# tests/unit/_v1_format_rules_golden.py and
# docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md.
#
# FORMAT_RULES rule 1 gives the SYNTAX for a self-loop but never the POLICY.
# Meanwhile the workflow script said only "on success: proceed to [Y]",
# which a policy follows literally: it advances on the same turn it issues
# the tool call, before the result exists. Measured on ckpt-500 /
# task-a-sft-v1: gold expects a self-loop on 41.6% of turns, the policy
# emits one on 4.8%, and state accuracy on stay+tool-expected turns is
# 0.055. See docs/cat_a_state_annotation_convention_review.md §5.
_STAY_RULE_ENABLED = os.environ.get("TASK_A_STAY_RULE", "1") != "0"

STAY_RULE = """\
2. Tool-execution turns do NOT advance. When your turn issues a <tool_call>, you have not
   yet seen the result, so you MUST remain in the current state and write the same name on
   both sides:
       [STATE: SEARCH_OPTIONS → SEARCH_OPTIONS]
       <tool_call>{"name": "search_flights", "arguments": {...}}</tool_call>
   Advance only on a LATER turn, after the tool result has come back and you have used it.
   The workflow script's "- on success: proceed to [Y]" describes where to go once the
   result confirms success — it does NOT mean advance on the turn that issues the call.
   Never announce a state you do not yet have the evidence to be in."""

_RETRY_RULE = """\
{n}. If a tool returns an error, stay in the current state and you may retry the SAME
   call, annotating [STATE: X → X] on every retry — never advance while retrying. Retry
   at most {retry_budget} time(s) total (including the first attempt). If it still fails
   after that budget, stop retrying: follow the workflow's error path if the script names
   one, otherwise say plainly that this step cannot be completed right now and that you
   will hand off / follow up, while remaining in [STATE: X → X]. Do not invent a
   transition just to move past a failure."""

_STAY_TOOL_RULE_NUM = 2 if _STAY_RULE_ENABLED else None


def _format_rules(retry_budget: int = 1) -> str:
    lines = [
        "Rules:",
        "",
        """1. Turn template — EVERY assistant turn MUST start with a state annotation, including
   tool-only turns and the terminal turn. Use exactly this format on the first line:
       [STATE: CURRENT → NEXT]
   If the state does not change, write the same name on both sides:
       [STATE: QUALIFY_PROSPECT → QUALIFY_PROSPECT]
   Never omit this line. A turn that is "just a tool call" still needs the STATE line
   above the <tool_call> tag.""",
    ]
    next_num = 2
    if _STAY_RULE_ENABLED:
        lines.append(STAY_RULE)
        next_num = 3
    lines.append(f"""{next_num}. Tool-call format — when you call a tool, emit it on its own line(s) as:
       <tool_call>{{"name": "<tool_name>", "arguments": {{<arg_key>: <arg_value>, ...}}}}</tool_call>
   The two top-level keys are exactly "name" and "arguments". Do NOT flatten arguments
   into the top level. Worked example for a schema with required=[patient_id, specialty]
   and optional reason:
       [STATE: VERIFY_PATIENT → VERIFY_PATIENT]
       <tool_call>{{"name": "request_referral", "arguments": {{"patient_id": "P12345", "specialty": "cardiology"}}}}</tool_call>""")
    next_num += 1
    lines.append(f"""{next_num}. Tool authority — the "Tool schemas" section is the ONLY authoritative source for which
   tools exist and which parameters they accept. The "Workflow script" hints at conversation
   flow but its per-state tool listings are UNRELIABLE; if it conflicts with a tool schema,
   trust the schema. Note that the schema uses "parameters" (OpenAI tools format) while
   your <tool_call> emits "arguments" — these refer to the same thing, do not confuse them.""")
    next_num += 1
    lines.append(f"""{next_num}. Argument discipline (strict):
   a. Pass ONLY parameters listed in the schema's "required" array, plus any optional
      parameter for which the user has EXPLICITLY stated a value in the conversation.
   b. Do NOT invent values for optional parameters. If the user has not said anything
      about `reason`, `description`, `offer_details`, `notes`, etc., omit those fields.
   c. Use parameter values verbatim from the user. Do not paraphrase, expand abbreviations,
      or reformat (e.g. user says "premium" → pass exactly "premium", not "premium package";
      user says "competitor.com" → ask for the full URL before calling; do not fabricate one).""")
    next_num += 1
    lines.append(f"""{next_num}. Tool-call necessity — do not call a tool unless the workflow requires it. Greetings,
   acknowledgements, clarifying questions, and terminal closings are text-only turns; do
   not append a tool call just to "wrap up" the conversation.""")
    next_num += 1
    lines.append(f"""{next_num}. Multi-turn negotiation — if a required argument is missing from what the user has said
   so far, ask the user for it BEFORE calling the tool. Do not synthesize plausible values.""")
    next_num += 1
    lines.append(_RETRY_RULE.format(n=next_num, retry_budget=retry_budget))
    next_num += 1
    lines.append(f"{next_num}. Reach a terminal state to complete the workflow.")
    next_num += 1
    lines.append(f"{next_num}. Never skip states or make invalid transitions.")
    return "\n\n".join(lines)


FORMAT_RULES = _format_rules()
```

Then delete the old standalone `STAY_RULE`/`FORMAT_RULES` block that previously lived at lines 15–79, since the constants above replace it. Leave everything from `def build_enriched_system_prompt(...)` onward unchanged in this task (Task 8 wires `retry_budget` through the call chain into `_format_rules`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_stay_rule_flag.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full existing test suite for regressions this prompt change is expected to cause**

Run: `source .venv/bin/activate && pytest tests/unit/test_teacher_prompt_contract.py tests/unit/test_data_generation.py -v`

Expected: `test_teacher_prompt_contract.py` still PASSes (it only asserts on `_TEACHER_SYSTEM_PROMPT` in `generate_workflows.py`, untouched by this task). Any `test_data_generation.py` failures whose fixtures assert the literal old `FORMAT_RULES` text are expected — fix those fixtures now (update the expected rule count / rule-2 text to match), since encoding the old convention in a fixture is the bug this task exists to remove.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/system_prompt.py tests/unit/test_stay_rule_flag.py tests/unit/_v1_format_rules_golden.py
git commit -m "fix(data): promote stay-rule to default-on, fix wrong worked example, state retry budget"
```

---

## Task 6: `quality_profiler.py` + `data_validator.py` hard defect wiring

**Files:**
- Modify: `src/llm_workflow_agents/data/quality_profiler.py`
- Modify: `src/llm_workflow_agents/data/data_validator.py`
- Modify: `.claude/agents/dataset-verifier.md`
- Test: `tests/unit/test_quality_profiler.py` (or the existing profiler test file — check `tests/unit/` for its actual name before creating a duplicate)
- Test: `tests/unit/test_data_validator.py`

**Interfaces:**
- Consumes: `state_convention.find_tool_stay_violations` (Task 1).
- Produces: `profile_task_a` now includes `distributions["tool_turn_state"] = {"self_loop": int, "advancing": int, "pct_conformant": float}`; `ProfileReport.defects` includes `f"{cid}: {violation}"` entries from `find_tool_stay_violations`. No signature changes.

- [ ] **Step 1: Confirm the existing profiler test filename**

```bash
grep -rl "profile_task_a" tests/unit/
```

Use whatever file that returns (do not guess a name and create a duplicate).

- [ ] **Step 2: Write the failing tests**

Add to that file (or `tests/unit/test_data_validator.py` for the validator half):

```python
def test_profile_task_a_flags_advancing_tool_turn(tmp_path):
    from llm_workflow_agents.data.quality_profiler import profile_task_a

    rec = {
        "conversation_id": "X_001", "complexity_level": "L1", "domain": "d",
        "workflow_graph": {"states": ["A", "TERMINAL"],
                            "transitions": [{"from": "A", "to": "TERMINAL", "condition": "", "priority": 0}],
                            "initial": "A", "terminal": ["TERMINAL"]},
        "tool_schemas": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
        "messages": [
            {"role": "assistant", "annotations": None,
             "content": '[STATE: A → TERMINAL]\n<tool_call>{"name": "t", "arguments": {}}</tool_call>'},
        ],
        "ground_truth": {"state_sequence": [{"from": "A", "to": "TERMINAL"}], "tool_calls": [],
                          "tool_chain_dependencies": [], "terminal_state": "TERMINAL", "terminal_reached": True},
        "language": "en",
    }
    path = tmp_path / "l1_test.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    report = profile_task_a(path)
    assert any(d.startswith("X_001: assistant turn 1 issues a <tool_call>") for d in report.defects)
    assert report.distributions["tool_turn_state"]["advancing"] == 1
    assert report.distributions["tool_turn_state"]["self_loop"] == 0
```

(Add `import json` at the top of the test file if not already present.) Similarly for `data_validator.py`:

```python
def test_validate_workflow_sample_flags_advancing_tool_turn():
    from llm_workflow_agents.data.data_validator import validate_dataset
    # construct a single-line JSONL fixture identical in shape to the profiler
    # test above, using the same _record-style dict, and assert
    # result.valid is False and one of result.errors mentions "tool-execution
    # turn must annotate"
```

(Write the actual fixture-and-assert code following the pattern of existing tests in that file — read `tests/unit/test_data_validator.py`'s current fixtures first so the new test matches its established record-building helper rather than duplicating one.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest -k "advancing_tool_turn" -v`
Expected: FAIL (assertions about `tool_turn_state` / the new defect message don't hold yet)

- [ ] **Step 4: Wire the check into `quality_profiler.py`**

In `profile_task_a`'s per-sample loop, immediately after the existing state-sequence-equality block (search for the `msg_states != gt_seq` check):

```python
from llm_workflow_agents.data.state_convention import find_tool_stay_violations
# ... inside the per-sample loop, after the msg_states/gt_seq check:
stay_violations = find_tool_stay_violations(messages)
for v in stay_violations:
    rep.defects.append(f"{cid}: {v}")
n_stay_advancing += len(stay_violations)
n_stay_self_loop += sum(
    1 for m in messages
    if m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or "")
) - len(stay_violations)
```

And in the report-assembly section where `distributions` is built:

```python
_tool_total = n_stay_self_loop + n_stay_advancing
rep.distributions["tool_turn_state"] = {
    "self_loop": n_stay_self_loop,
    "advancing": n_stay_advancing,
    "pct_conformant": round(100 * n_stay_self_loop / _tool_total, 1) if _tool_total else 100.0,
}
```

(Initialize `n_stay_advancing = n_stay_self_loop = 0` alongside the loop's other counters, matching the existing code's style for `self_loops`/`total_transitions` a few lines above.)

- [ ] **Step 5: Wire the mirror check into `data_validator.py`**

In `_validate_workflow_sample`, next to the existing `find_continuity_violations` call:

```python
from llm_workflow_agents.data.state_convention import find_tool_stay_violations
# ...
errors.extend(find_tool_stay_violations(sample["messages"]))
```

(Match the surrounding code's pattern for whether violations go to `errors` or `warnings` — per the design, this must be a hard `errors` entry, consistent with `profile_task_a` treating it as a hard defect.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest -k "advancing_tool_turn" -v`
Expected: PASS

- [ ] **Step 7: Update `.claude/agents/dataset-verifier.md`**

Find the paragraph stating self-loops are "a legitimate generator convention... not defects... expected, not a finding" and replace it with:

```markdown
Tool-call turns MUST self-loop (`[STATE: X → X]`) — the advance to a new
state happens on the *next* turn, after the tool result has come back. This
is now a hard defect class (`find_tool_stay_violations`, wired into
`profile_task_a` and `data_validator` as of task-a-sft-v2). Check
`distributions.tool_turn_state.pct_conformant` — it must read 100.0. A
corpus-wide self-loop share materially below ~40% is itself a finding, not
merely "expected." See
docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md.
```

- [ ] **Step 8: Run the full data-generation test suite and fix fixtures that encode the old convention**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py tests/unit/test_data_validator.py -v`

Any failing fixture that has a tool-call turn annotated with an advancing transition needs its literal `[STATE: X → Y]` text changed to `[STATE: X → X]` (and its neighboring turn adjusted to carry the advance) — updating these fixtures *is* this task's deliverable, not an unrelated regression to work around.

- [ ] **Step 9: Commit**

```bash
git add src/llm_workflow_agents/data/quality_profiler.py src/llm_workflow_agents/data/data_validator.py \
        .claude/agents/dataset-verifier.md tests/unit/
git commit -m "feat(eval): make tool-call stay-convention a hard defect in the profiler and validator"
```

---

## Task 7: Generator repair-loop insertion

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py`

**Interfaces:**
- Consumes: `state_convention.find_tool_stay_violations` (Task 1).
- Produces: no new public names; `_find_violations` (or the inline `or`-chain at the call site — confirm the exact enclosing function name by reading the file before editing) now also rejects advancing tool turns.

- [ ] **Step 1: Locate the exact call site**

```bash
grep -n "find_tool_placement_violations(allowed" -A 6 src/llm_workflow_agents/data/generate_workflows.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_data_generation.py (append)
def test_repair_loop_rejects_advancing_tool_turn(monkeypatch):
    # Monkeypatch the teacher call to return a fixed, non-conforming
    # conversation once, then a conforming one, and assert the repair loop's
    # feedback on the first violation mentions "tool-execution turn must
    # annotate" (find_tool_stay_violations' message text) -- proving the new
    # check fires before the sample is accepted.
    ...
```

(Follow the existing monkeypatching pattern already used elsewhere in `test_data_generation.py` for repair-loop tests — search the file for `repair_feedback` or `_attempt_sample` to find the established fixture style and match it; do not invent a new mocking approach.)

- [ ] **Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -k advancing_tool_turn -v`
Expected: FAIL

- [ ] **Step 4: Edit the `or`-chain**

```python
from llm_workflow_agents.data.state_convention import find_tool_stay_violations
# ... at the violation-check call site:
violations = (
    find_tool_placement_violations(allowed, msgs, schema_names)
    or _find_transition_violations(valid_edge_pairs, msgs)
    or find_tool_stay_violations(msgs)
    or find_continuity_violations(msgs, initial_name, terminal_names)
    or find_shape_violations(msgs, initiator)
)
```

Position 3 of 5: after the two referential checks (a nonexistent tool/edge can't be fixed by relabelling), before continuity/shape (a stay violation's correct repair changes the continuity chain — feeding continuity feedback first would chase the symptom). v1 has 4,333 stay violations and 0 continuity defects, so this reordering is free.

- [ ] **Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -k advancing_tool_turn -v`
Expected: PASS

- [ ] **Step 6: Run the full generation test suite**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -v`
Expected: PASS. Fix any fixture whose mocked "teacher output" advances on a tool turn (same rationale as Task 6 Step 8).

- [ ] **Step 7: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): reject advancing tool-call turns in the teacher repair loop"
```

---

## Task 8: `_workflow_script.py::build_workflow_script` — `tool_turn_semantics` and `retry_budget`

**Files:**
- Modify: `src/llm_workflow_agents/data/_workflow_script.py`
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (the `_graph_to_script` call site)
- Modify: `src/llm_workflow_agents/data/system_prompt.py` (the `build_enriched_system_prompt` call site)
- Test: `tests/unit/test_playbook_render.py` (new, for the Task C byte-identity guard) and the existing `_workflow_script.py` test file

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_workflow_script(workflow_graph, tool_schemas=None, language="en", messages=None, *, tool_turn_semantics: bool = False, retry_budget: int = 1) -> str`. Both Task A call sites (`generate_workflows._graph_to_script`, `system_prompt.build_enriched_system_prompt`) must pass `tool_turn_semantics=True`; Task C's `_playbook_render.py` call site is left unchanged (defaults to `False`).

- [ ] **Step 1: Find the current signature and call sites**

```bash
grep -n "def build_workflow_script" -A 15 src/llm_workflow_agents/data/_workflow_script.py
grep -rn "build_workflow_script(" src/llm_workflow_agents/data/*.py
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_playbook_render.py (new)
from llm_workflow_agents.data._workflow_script import build_workflow_script


def _tool_state_graph():
    return {
        "states": ["A", "TERMINAL"],
        "state_details": [
            {"name": "A", "tools": ["t"], "entry_actions": [], "instruction": "do it"},
            {"name": "TERMINAL", "tools": [], "entry_actions": [], "instruction": ""},
        ],
        "transitions": [{"from": "A", "to": "TERMINAL", "condition": "", "priority": 0}],
        "initial": "A", "terminal": ["TERMINAL"],
    }


def test_state_script_unchanged_by_tool_turn_semantics_default():
    graph = _tool_state_graph()
    default_output = build_workflow_script(graph, tool_schemas=[{"type": "function", "function": {"name": "t"}}])
    explicit_off = build_workflow_script(graph, tool_schemas=[{"type": "function", "function": {"name": "t"}}],
                                          tool_turn_semantics=False)
    assert default_output == explicit_off


def test_tool_turn_semantics_rewrites_success_line_and_adds_stay_note():
    graph = _tool_state_graph()
    output = build_workflow_script(graph, tool_schemas=[{"type": "function", "function": {"name": "t"}}],
                                    tool_turn_semantics=True, retry_budget=2)
    assert "on a LATER turn" in output
    assert "stays in" in output.lower() or "stay in" in output.lower()
    assert "retry at most 2" in output.lower()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_playbook_render.py -v`
Expected: FAIL — `TypeError: build_workflow_script() got an unexpected keyword argument 'tool_turn_semantics'`

- [ ] **Step 4: Edit `build_workflow_script`**

Add the keyword-only parameters and, inside the per-state rendering loop, branch on `tool_turn_semantics` and whether `state.tools` is non-empty:

```python
def build_workflow_script(
    workflow_graph, tool_schemas=None, language="en", messages=None,
    *, tool_turn_semantics: bool = False, retry_budget: int = 1,
) -> str:
    templates = _SCRIPT_TEMPLATES[language]
    # ... existing setup unchanged ...
    for state in ...:
        # ... existing header/instruction/tools_intro rendering unchanged ...
        has_tools = bool(state_tools)  # however the existing code names this
        if has_tools and tool_turn_semantics:
            lines.append(_TOOL_TURN_NOTE[language].format(name=state.name))
            lines.append(_PRIMARY_BRANCH_TOOL[language].format(to=primary_to))
            lines.append(_RETRY_NOTE[language].format(name=state.name, n=retry_budget,
                                                        fallback=_FALLBACK_TEXT[language]))
        else:
            lines.append(templates["primary_branch"].format(to=primary_to))
        # ... existing alt_branch rendering unchanged ...
```

Add the new per-language template dicts near `_SCRIPT_TEMPLATES`:

```python
_TOOL_TURN_NOTE = {
    "en": "- The turn that issues the <tool_call> stays in [{name}]: [STATE: {name} → {name}]",
    "th": "- เทิร์นที่เรียกเครื่องมือให้คงอยู่ที่ [{name}]: [STATE: {name} → {name}]",
}
_PRIMARY_BRANCH_TOOL = {
    "en": "- On success: proceed to [{to}] — on a LATER turn, after the tool result has come back",
    "th": "- เมื่อสำเร็จ: ดำเนินการต่อที่ [{to}] — ในเทิร์นถัดไป หลังได้รับผลลัพธ์จากเครื่องมือแล้ว",
}
_RETRY_NOTE = {
    "en": "- On tool error: stay in [{name}] and retry at most {n} time(s); after that, {fallback}",
    "th": "- หากเครื่องมือผิดพลาด: คงอยู่ที่ [{name}] และลองใหม่ได้ไม่เกิน {n} ครั้ง; หลังจากนั้น {fallback}",
}
_FALLBACK_TEXT = {
    "en": "follow the workflow's error path, or state plainly the step cannot be completed and hand off",
    "th": "ดำเนินการตามเส้นทางข้อผิดพลาดของขั้นตอนงาน หรือแจ้งตรงๆ ว่าไม่สามารถทำขั้นตอนนี้ให้เสร็จได้และส่งต่อ",
}
```

Read the existing function body fully before editing (it was shown earlier in this plan's exploration at lines 274–349-ish) so the insertion point matches the real variable names (`state_tools`, `primary_to`, etc. above are placeholders for whatever the existing code actually calls them — do not introduce new names that shadow existing ones).

- [ ] **Step 5: Update the two Task A call sites to pass `tool_turn_semantics=True`**

```bash
grep -n "build_workflow_script(" src/llm_workflow_agents/data/generate_workflows.py src/llm_workflow_agents/data/system_prompt.py
```

Add `tool_turn_semantics=True` (and thread `retry_budget=spec.retry_budget` at the `generate_workflows.py` call site once Task 9 adds that field — for now, pass `retry_budget=1` as a placeholder value and revisit in Task 9's Step 5).

- [ ] **Step 6: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_playbook_render.py -v`
Expected: PASS

- [ ] **Step 7: Confirm Task C is untouched**

```bash
source .venv/bin/activate && pytest tests/unit -k "playbook_render or _playbook_render or task_c" -v
```
Expected: all PASS, no changes needed to any Task C test (this is the whole point of the default-`False` guard).

- [ ] **Step 8: Commit**

```bash
git add src/llm_workflow_agents/data/_workflow_script.py src/llm_workflow_agents/data/generate_workflows.py \
        src/llm_workflow_agents/data/system_prompt.py tests/unit/test_playbook_render.py
git commit -m "feat(data): teach the workflow script the tool-stay convention behind a default-off flag"
```

---

## Task 9: `ComplexitySpec` retry fields + `select_subgraph` resolution + placeholder generator retry arc

**Files:**
- Modify: `src/llm_workflow_agents/config/schema.py`
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (`select_subgraph`, `_generate_placeholder_conversation`, `_build_teacher_prompt`)
- Test: `tests/unit/test_data_generation.py`

**Interfaces:**
- Consumes: `ComplexitySpec` (existing dataclass).
- Produces: `ComplexitySpec.retry_budget: int` and `ComplexitySpec.retry_exhaustion: Literal["none","error_path","handoff_in_state"]`, both with defaults so **no existing caller breaks**. `select_subgraph` return value (`WorkflowGraph`) is unchanged in shape; the resolution of `"error_path"` → `"handoff_in_state"` happens as a new attribute the placeholder generator and prompt builder read — expose it as `WorkflowGraph`-adjacent metadata returned alongside the graph, e.g. a new `resolved_retry_exhaustion: str` local computed at the `select_subgraph` call site in `_build_one_sample`, not inside `WorkflowGraph` itself (keeps `WorkflowGraph`'s dataclass shape backward compatible for every other caller).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_data_generation.py (append)
from llm_workflow_agents.config.schema import COMPLEXITY_SPECS


def test_complexity_specs_have_retry_fields():
    assert COMPLEXITY_SPECS["L1"].retry_budget == 1
    assert COMPLEXITY_SPECS["L1"].retry_exhaustion == "none"
    assert COMPLEXITY_SPECS["L3"].retry_budget == 2
    assert COMPLEXITY_SPECS["L3"].retry_exhaustion == "error_path"
    assert COMPLEXITY_SPECS["L5"].retry_budget == 3


def test_retry_arc_is_deterministic_across_max_workers():
    from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        meta1 = generate_workflow_dataset(
            complexity_level="L3", num_samples=6, teacher_model=None,
            output_dir=Path(d1), seed=123, language="en", max_workers=1,
        )
        meta2 = generate_workflow_dataset(
            complexity_level="L3", num_samples=6, teacher_model=None,
            output_dir=Path(d2), seed=123, language="en", max_workers=4,
        )
        assert meta1.output_files[0].read_text() == meta2.output_files[0].read_text()
```

(The second test reuses the pattern of the existing `test_concurrent_output_matches_serial` test — check that test's exact fixture/args first and mirror them rather than guessing `generate_workflow_dataset`'s full parameter list.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -k "retry_fields or retry_arc" -v`
Expected: FAIL — `AttributeError: 'ComplexitySpec' object has no attribute 'retry_budget'`

- [ ] **Step 3: Add the fields to `ComplexitySpec`**

```python
# src/llm_workflow_agents/config/schema.py
@dataclass
class ComplexitySpec:
    level: str
    target_path_len: tuple[int, int]
    num_branches: tuple[int, int]
    num_loops: tuple[int, int]
    include_recovery: bool
    num_tools: int
    chain_depth: int
    retry_budget: int = 1
    retry_exhaustion: str = "none"   # "none" | "error_path" | "handoff_in_state"


COMPLEXITY_SPECS = {
    "L1": ComplexitySpec("L1", (3,4),   (0,0), (0,0), False, 1, 0, retry_budget=1, retry_exhaustion="none"),
    "L2": ComplexitySpec("L2", (5,7),   (1,1), (0,0), False, 2, 1, retry_budget=1, retry_exhaustion="none"),
    "L3": ComplexitySpec("L3", (8,12),  (2,3), (0,1), True,  4, 2, retry_budget=2, retry_exhaustion="error_path"),
    "L4": ComplexitySpec("L4", (12,16), (3,5), (1,1), True,  6, 3, retry_budget=2, retry_exhaustion="error_path"),
    "L5": ComplexitySpec("L5", (16,20), (0,99),(1,2), True,  7, 4, retry_budget=3, retry_exhaustion="error_path"),
}
```

- [ ] **Step 4: Resolve `"error_path"` to `"handoff_in_state"` at sample time**

In `_build_one_sample` (or wherever `select_subgraph(domain, spec, rng, ...)` is called), immediately after the call:

```python
resolved_retry_exhaustion = spec.retry_exhaustion
if spec.retry_exhaustion == "error_path":
    has_tool_error_edge = any(t.trigger == "tool_error" for t in workflow_graph_obj.transitions)
    if not has_tool_error_edge:
        resolved_retry_exhaustion = "handoff_in_state"
```

Thread `resolved_retry_exhaustion` and `spec.retry_budget` into both `_build_teacher_prompt` (as new interpolated values feeding the `_RETRY_RULE`-equivalent line already added to `FORMAT_RULES` in Task 5, and into the workflow-script `retry_budget` kwarg from Task 8) and into `_generate_placeholder_conversation`.

- [ ] **Step 5: Teach the placeholder generator the retry loop**

In `_generate_placeholder_conversation` (~line 990), where a single tool call is currently emitted per state visit, wrap it in a bounded retry loop using the per-sample `rng`:

```python
attempts = 0
success = False
while attempts < spec.retry_budget and not success:
    attempts += 1
    is_error = rng.random() < TOOL_ERROR_RATE and attempts < spec.retry_budget
    # emit [STATE: X -> X] + <tool_call> turn (existing code path)
    # emit the tool-role result: error payload if is_error else success payload
    success = not is_error
if not success and resolved_retry_exhaustion == "handoff_in_state":
    # emit an in-state [STATE: X -> X] prose turn stating the step could not
    # be completed and a hand-off will happen, THEN continue the spine
    ...
```

Every `rng.random()` call here must use the loop's existing per-sample `rng` (never a fresh `random.Random()` or the module-level `random`), matching the determinism contract. Also pass `retry_budget=spec.retry_budget` into the same-call `_workflow_script.build_workflow_script(..., tool_turn_semantics=True, retry_budget=spec.retry_budget)` from Task 8's Step 5 placeholder value.

- [ ] **Step 6: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -k "retry_fields or retry_arc" -v`
Expected: PASS

- [ ] **Step 7: Run the full generation suite**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/llm_workflow_agents/config/schema.py src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): add per-level retry budget and in-state handoff on retry exhaustion"
```

---

## Task 10: Generation-script flags and observability

**Files:**
- Modify: `scripts/generate_sft_until_target.py`
- Modify: `scripts/generate_sft_data.sh`
- Test: extend whatever existing test covers `generate_sft_until_target.py`'s CLI parsing (check `tests/unit/` for it first; if none exists, this task's tests are the first ones for this script — write focused argparse tests, not a full integration test, since actual generation needs API keys)

**Interfaces:**
- Consumes: `ComplexitySpec.retry_budget/retry_exhaustion` (Task 9).
- Produces: new CLI flags `--retry-budget`, `--retry-exhaustion`, `--require-tool-stay/--no-require-tool-stay` on both scripts; `generate_leg`'s stats dict gains a `stay_dropped: int` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_generate_sft_until_target_cli.py (new)
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_sft_until_target.py"


def test_dry_run_accepts_retry_flags(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run",
         "--levels", "L1", "--languages", "en",
         "--retry-budget", "2", "--retry-exhaustion", "error_path",
         "--no-require-tool-stay",
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_generate_sft_until_target_cli.py -v`
Expected: FAIL — `error: unrecognized arguments: --retry-budget 2 ...`

- [ ] **Step 3: Add the flags**

In `main()`'s `argparse` block:

```python
p.add_argument("--retry-budget", type=int, default=None,
               help="Override retry budget for all levels (else per-level from COMPLEXITY_SPECS).")
p.add_argument("--retry-exhaustion", choices=["auto", "error_path", "handoff_in_state", "none"],
               default="auto", help="Override retry-exhaustion policy (default: per-level 'auto').")
p.add_argument("--require-tool-stay", action="store_true", default=True,
               help="Reject batches with advancing tool-call turns (default: on).")
p.add_argument("--no-require-tool-stay", dest="require_tool_stay", action="store_false",
               help="Regenerate v1-comparable data by disabling the stay-convention gate.")
```

Thread `args.retry_budget`, `args.retry_exhaustion`, `args.require_tool_stay` into `generate_leg(...)`'s call in `main()`, and add matching parameters to `generate_leg`'s signature, passed through to `generate_workflow_dataset(...)`.

- [ ] **Step 4: Add observability to `generate_leg`**

Where `defect_dropped` is computed (from `defective_conversation_ids(profile_task_a(clean_file))`), add:

```python
stay_dropped = sum(
    1 for d in profile_task_a(clean_file).defects
    if "issues a <tool_call> but annotates an advancing transition" in d
)
```

Add `stay_dropped` to the per-iteration print line and to the returned stats dict, and add `"stay_dropped": stats["stay_dropped"]` to the leg summary written into `loop_stats_*.json`.

When `args.require_tool_stay` is `False`, wrap the profiler-based defect filtering so stay violations are excluded from `defective_conversation_ids`'s effective drop set for that run (e.g. filter `bad = {cid for cid in defective_conversation_ids(...) }` down by re-running `find_tool_stay_violations`-only-caused entries back in) — implement this as a small helper `_qualified_ids(report, require_tool_stay: bool) -> set[str]` next to `generate_leg` rather than duplicating the filter logic inline.

- [ ] **Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/unit/test_generate_sft_until_target_cli.py -v`
Expected: PASS

- [ ] **Step 6: Mirror the flags into `generate_sft_data.sh`**

Add `--retry-budget`/`--retry-exhaustion` to the `case` block (same pattern as the existing `--behavior-preset` handling) and pass them into all three inline `python3 -c` blocks as new `retry_budget=$RETRY_BUDGET, retry_exhaustion='$RETRY_EXHAUSTION'` kwargs to `generate_workflow_dataset(...)`. Update the header comment block to document the two new flags.

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_sft_until_target.py scripts/generate_sft_data.sh tests/unit/test_generate_sft_until_target_cli.py
git commit -m "feat(scripts): add retry-budget/retry-exhaustion/require-tool-stay flags to generation scripts"
```

---

## Task 11: `.claude/agents/corpus-remediator.md` + remediation playbook

**Files:**
- Create: `.claude/agents/corpus-remediator.md`
- Create: `docs/task_a_state_convention_remediation_playbook.md`

**Interfaces:**
- Consumes: nothing (this is documentation + an agent definition, no code).
- Produces: the `corpus-remediator` agent name, which Task 12's driver script invokes via `claude -p ... --agent corpus-remediator`. The exact output contract (`LEDGER: <path> <count>` reply, one `LedgerEntry` JSON per ledger line) documented here **must match** what Task 12's `validate_entry` parses — read Task 12 before finalizing wording here, or write both tasks' contract text from the same source block.

- [ ] **Step 1: Write `.claude/agents/corpus-remediator.md`**

```markdown
---
name: corpus-remediator
description: >-
  Authors the short missing assistant/user messages required to bring Task A
  SFT conversations onto the tool-call state convention (a tool-calling turn
  stays in its state; the advance moves to a later turn). Use ONLY when
  driven by scripts/build_remediation_ledger.py with a batch request file.
  Writes a decision ledger of proposed message contents — never edits corpus
  JSONL, never changes state annotations, never adds or removes tool calls.
tools: Bash, Read, Grep, Glob, Write
model: inherit
---

# Task A Corpus Remediator

You author short, missing conversation turns for a batch of Task A SFT
conversations that are structurally sound except for 1–2 messages the
deterministic repair pipeline could not safely synthesize on its own. You
are invoked headlessly (`claude -p`) by `scripts/build_remediation_ledger.py`
with a batch request file path in your prompt.

**You are not deciding structure.** The insert position, the required
`[STATE: X → Y]` marker (for assistant inserts), and the role of each
message are already decided by
`src/llm_workflow_agents/data/state_convention_repair.py` and are
non-negotiable. Your only job is the prose.

## Always activate the venv

Every Python invocation in this project is prefixed:

```bash
source .venv/bin/activate && python3 ...
```

Use `uv`, not `pip`, if you ever need to install something (you should not
need to for this task).

## The convention you are serving

1. A turn that emits `<tool_call>` annotates `[STATE: X → X]` (stay).
2. A `role: "tool"` message returns the result.
3. On success, the *next* assistant turn advances: `[STATE: X → Y]`.
4. On error, the next turn stays `[STATE: X → X]` and may retry.
5. After N failed attempts, stop retrying and take the fallback path.

Two insert kinds you will be asked for:
- **hand-off turn** (`role: "assistant"`): inserted between a `tool` result
  and a *second* bare tool-call turn, so the second call has somewhere
  legal to attribute its `from`-state. Report the first tool's result in
  one short sentence and say what you're checking/doing next — do NOT
  claim the second tool's work is already done.
- **closing pair** (`role: "user"` then `role: "assistant"`): appended at
  the end of a conversation whose last turn was a tool call. The `user`
  message is a short, plausible acknowledgement/follow-up; the `assistant`
  message is the terminal closing turn, carrying the required
  `[STATE: X → TERMINAL]`-shaped marker you were given.

## Procedure

1. Read the batch request file (path given in your prompt):

```bash
source .venv/bin/activate && python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), indent=2))" <batch_file>
```

2. For each request object `{insert_id, position_after_msg_index, role,
   required_marker, must_not_contain, context_window, language,
   conversation_id}`, read `context_window` (the surrounding messages,
   verbatim) to determine language, register, and what has already
   happened in the conversation.

3. Author `content`:
   - Match the row's `language` exactly, including `code_switch` register
     (mixed Thai/English mid-sentence, matching the surrounding turns'
     style — do not switch a code_switch conversation to pure English).
   - 1–2 sentences, 20–600 characters.
   - For `role: "assistant"`, the content MUST start with exactly
     `required_marker` verbatim (arrow glyph included) followed by a
     newline, then your prose. It must NOT contain `<tool_call>` or a
     second `[STATE:` marker.
   - For `role: "user"`, no marker — just a short natural follow-up.
   - Never mention states, the workflow graph, or "tools" as a concept.
     Write as the persona already established in the conversation.

4. If a request looks structurally wrong (context doesn't support any
   plausible message, or the required marker references a state that
   doesn't appear anywhere in `context_window`), do not guess — emit a
   refusal entry instead (see Output contract) and move on.

5. Append one JSON line per request to `<batch_file with .ledger.jsonl
   suffix>` (same directory, given in your prompt) as you go — do not hold
   everything in memory and write once at the end, so a mid-batch failure
   still yields partial output:

```bash
source .venv/bin/activate && python3 -c "
import json
entry = {...}
with open('<ledger_path>', 'a') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + chr(10))
"
```

6. Self-check before replying: grep the ledger file for `<tool_call>` and
   for a second `[STATE:` per entry — both must be absent.

```bash
grep -c '<tool_call>' <ledger_path>   # must print 0
```

## Output contract

Each ledger line is exactly:

```json
{"insert_id": "<from the request>", "conversation_id": "<from the request>",
 "role": "user|assistant", "content": "<authored text>",
 "rationale": "<<=200 chars, why this content>", "agent_model": "<your model id>",
 "schema_version": 1}
```

A refusal is the same shape with `"refuse": true` and no `content` key.

Your final reply (after the ledger file is fully written) is a single line:

```
LEDGER: <absolute path to the .ledger.jsonl file> <count of entries written>
```

followed by at most 3 sentences of summary. The driver locates your ledger
by the path it gave you, not by parsing this line's path — but a wrong
count here is a signal the driver logs, so get it right.

## Scope notes

You never edit any file under `data/output/`. You never call
`remediate_task_a_states.py` or any other script that mutates the corpus.
You never accept a request that isn't in the batch file you were given.
When in doubt, refuse — the driver treats a refusal exactly like a
deterministic-gate rejection (the row falls back to being dropped from the
corpus), which is always safe.
```

- [ ] **Step 2: Write `docs/task_a_state_convention_remediation_playbook.md`**

```markdown
# Task A Tool-Stay Convention: Remediation Playbook

Operator runbook for producing `task-a-sft-v2` from `task-a-sft-v1`. Design
rationale lives in
`docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md`;
this document is the sequence of commands and the decision points between
them.

## 1. The convention, and why

[Copy the "Target convention" and "Measured baseline" sections verbatim
from the design spec — do not re-derive, the numbers there are the
code-verified triage output from Task 4 Step 5, which supersedes the
exploratory estimates cited in
`docs/cat_a_state_annotation_convention_review.md`.]

## 2. Invariant table

[Copy the "Invariants any repair must preserve" table from the design spec.]

## 3. The queue algorithm, worked

Trace one real CASE_A row and one real tail-deficit row end to end (pull
two examples from `data/interim/task_a_state_triage/report.json` after
Task 4 Step 5 — one with `"move": "relabel"`, one with
`"move": "append_closing_pair"` — and show the before/after `messages` and
`ground_truth.state_sequence` side by side).

## 4. Move ladder and rejected moves

[Copy the move-ladder table and the "Rejected" paragraph from the design
spec; append the actual measured counts from Task 4 Step 5's triage run,
noting any deviation from the design spec's exploratory estimates.]

## 5. Drift

`RepairPlan.drift_turns` marks prose turns whose destination changed
because the trajectory was re-derived rather than relabelled in place.
Before running the full ledger pass (§8), sample 30 conversations with
`drift_turns` non-empty across all three language legs and read them by
hand: does the prose still make sense given the new destination? If more
than a handful read badly, tighten `plan_repair` (Task 2/3) rather than
proceeding — this is a quality gate, not a formality.

## 6. Ledger contract and the three gate layers

[Copy from the design spec's A4 section: entry / record / file layers, the
never-raises contract, resumability via `accepted.jsonl`.]

## 7. Acceptance criteria

[Copy the 9-point acceptance gate list from the design spec verbatim —
these are the numbers `remediate_task_a_states.py verify --strict` and the
manual checks in §9 of this playbook must satisfy before tagging v2.]

## 8. Runbook

```bash
source .venv/bin/activate

# 1. Triage (already run once in Task 4 Step 5; re-run if the input corpus changed)
python scripts/remediate_task_a_states.py triage \
  --input-dir data/output/sft/task_a \
  --report data/interim/task_a_state_triage/report.json

# 2. Deterministic apply (moves 1-2; everything needing an insert is dropped
#    at this stage since no --ledger-dir is given yet)
python scripts/remediate_task_a_states.py apply \
  --input-dir data/output/sft/task_a \
  --output-dir /tmp/task_a_deterministic_only \
  --on-unrepairable drop

# 3. Smoke the agent pass -- 20 conversations, read every accepted entry by hand
python scripts/build_remediation_ledger.py \
  --input-dir data/output/sft/task_a \
  --triage-report data/interim/task_a_state_triage/report.json \
  --ledger-dir data/interim/task_a_remediation_ledger \
  --limit 20
cat data/interim/task_a_remediation_ledger/accepted.jsonl | python3 -m json.tool

# 4. Full ledger run (costly: ~$8-13, 1.5-3h at 4 workers -- get explicit
#    go-ahead before running this against the full ~1,465-conversation queue;
#    see design spec's Task-2 note for why this grew from the original ~930)
python scripts/build_remediation_ledger.py \
  --input-dir data/output/sft/task_a \
  --triage-report data/interim/task_a_state_triage/report.json \
  --ledger-dir data/interim/task_a_remediation_ledger

# 5. Final apply, now with the ledger
python scripts/remediate_task_a_states.py apply \
  --input-dir data/output/sft/task_a \
  --output-dir data/output/sft/task_a_remediated \
  --ledger-dir data/interim/task_a_remediation_ledger \
  --rebuild-prompts \
  --on-unrepairable drop

# 6. Verify
python scripts/remediate_task_a_states.py verify \
  --input-dir data/output/sft/task_a_remediated --strict
python scripts/remediate_task_a_states.py diff \
  --before data/output/sft/task_a --after data/output/sft/task_a_remediated
```

## 9. If the drop rate exceeds budget

[Describe: check `by_move`/`drop_reasons` breakdown by level and language;
if any level lost >5% of its rows (acceptance gate 6), do not proceed to
DVC repro — either loosen a move (re-check §5's drift sample first) or
backfill the missing rows with the fixed generator (Task 9/10) at the
affected level/language before splitting.]

## 10. Lineage

`task-a-sft-v1` = `data/output/sft/task_a` as of commit `93e0cf7`.
`task-a-sft-v2` = `data/output/sft/task_a_remediated` + the D3 retry slice,
merged via `scripts/concat_task_a.py`. After `dvc repro` (Task 14), run
`dvc status` and compare the reproduced directory hash against `dvc.lock`
before `dvc push` and tagging — this exact silent-lineage-drift failure has
happened twice before in this project (see
`docs/cat_a_state_annotation_convention_review.md` §1, §6.6).
```

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/corpus-remediator.md docs/task_a_state_convention_remediation_playbook.md
git commit -m "docs: add corpus-remediator agent and the v2 remediation playbook"
```

---

## Task 12: `scripts/build_remediation_ledger.py` — the `claude -p` driver

**Files:**
- Create: `scripts/build_remediation_ledger.py`
- Test: `tests/unit/test_build_remediation_ledger.py`

**Interfaces:**
- Consumes: the `triage` report schema from Task 4 (`records[].inserts[].insert_id/position_after_msg_index/role/required_marker`), the `corpus-remediator` agent contract from Task 11.
- Produces: `<ledger-dir>/accepted.jsonl`, `<ledger-dir>/rejected.jsonl`, `<ledger-dir>/progress.json`. `accepted.jsonl`'s schema (`{insert_id, conversation_id, role, content, rationale, agent_model, schema_version}`) is what Task 4's `_load_ledger` (already implemented) reads — do not change field names without updating that function too.

- [ ] **Step 1: Write the failing tests (mocking `subprocess.run`, never calling the real `claude` CLI)**

```python
# tests/unit/test_build_remediation_ledger.py
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_remediation_ledger as brl  # noqa: E402


def _fake_run_factory(ledger_path: Path, entries: list[dict]):
    def _fake_run(cmd, cwd, capture_output, text, timeout):
        ledger_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"result": f"LEDGER: {ledger_path} {len(entries)}\nDone."}), stderr="",
        )
        return result
    return _fake_run


def test_validate_entry_rejects_wrong_marker():
    request = {"insert_id": "f:0:0", "role": "assistant", "required_marker": "[STATE: A → A]",
               "position_after_msg_index": 0, "context_window": []}
    entry = {"insert_id": "f:0:0", "role": "assistant",
             "content": "[STATE: A → B]\nAll set.", "rationale": "x",
             "agent_model": "test", "schema_version": 1}
    violations = brl.validate_entry(entry, request)
    assert any("marker" in v for v in violations)


def test_validate_entry_rejects_tool_call_in_content():
    request = {"insert_id": "f:0:0", "role": "assistant", "required_marker": "[STATE: A → A]",
               "position_after_msg_index": 0, "context_window": []}
    entry = {"insert_id": "f:0:0", "role": "assistant",
             "content": '[STATE: A → A]\n<tool_call>{"name":"t","arguments":{}}</tool_call>',
             "rationale": "x", "agent_model": "test", "schema_version": 1}
    assert any("tool_call" in v for v in brl.validate_entry(entry, request))


def test_validate_entry_accepts_conforming_entry():
    request = {"insert_id": "f:0:0", "role": "assistant", "required_marker": "[STATE: A → A]",
               "position_after_msg_index": 0, "context_window": []}
    entry = {"insert_id": "f:0:0", "role": "assistant",
             "content": "[STATE: A → A]\nLet me check on that for you.",
             "rationale": "x", "agent_model": "test", "schema_version": 1}
    assert brl.validate_entry(entry, request) == []


def test_refusal_entry_is_treated_as_rejected():
    request = {"insert_id": "f:0:0", "role": "assistant", "required_marker": "[STATE: A → A]",
               "position_after_msg_index": 0, "context_window": []}
    entry = {"insert_id": "f:0:0", "refuse": True, "rationale": "context insufficient",
              "agent_model": "test", "schema_version": 1}
    assert brl.validate_entry(entry, request) == ["agent refused this insert"]


def test_run_batch_writes_accepted_and_skips_on_missing_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(brl.shutil, "which", lambda _: None)
    result = brl.run_agent_batch(tmp_path / "batch_0.json", tmp_path, timeout=10)
    assert result["error"] == "claude CLI not found"


def test_resume_skips_accepted_insert_ids(tmp_path):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "accepted.jsonl").write_text(
        json.dumps({"insert_id": "f:0:0", "role": "assistant", "content": "x",
                    "rationale": "x", "agent_model": "t", "schema_version": 1}) + "\n"
    )
    seen = brl.load_accepted_ids(ledger_dir)
    assert seen == {"f:0:0"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_build_remediation_ledger.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""claude -p driver that authors the ~17% of Task A conversations needing
1-2 short inserted messages to conform to the tool-stay convention.

The agent NEVER writes corpus rows -- it writes a decision ledger (one JSON
line per requested insert) that scripts/remediate_task_a_states.py apply
replays deterministically. This keeps `dvc repro` free of LLM spend, makes
the ledger PR-reviewable, and makes a rejected entry degrade to `drop` in
isolation.

Mirrors scripts/generate_sft_until_target.py::verify_batch_with_agent's
subprocess pattern exactly (this repo's only other `claude -p` call site):
never raises, preflight-checks the CLI, unwraps the --output-format json
envelope.

Usage:
    python scripts/build_remediation_ledger.py \\
        --input-dir data/output/sft/task_a \\
        --triage-report data/interim/task_a_state_triage/report.json \\
        --ledger-dir data/interim/task_a_remediation_ledger \\
        [--batch-size 10] [--max-workers 4] [--timeout 900] [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1

_PROMPT_TEMPLATE = (
    "Author the missing conversation turns described in the batch request file "
    "at {batch_path}. Activate the project venv first "
    "(source .venv/bin/activate) before running any Python tools. Write your "
    "ledger entries, one JSON object per line, to {ledger_path} as you go. "
    "When you are done, reply with a single line exactly of the form "
    "'LEDGER: {ledger_path} <count>', then at most 3 sentences of summary. "
    "Never edit any file under data/output/."
)


def validate_entry(entry: dict, request: dict) -> list[str]:
    """Deterministic entry-level gate. Empty list == accept."""
    if entry.get("refuse"):
        return ["agent refused this insert"]
    violations = []
    for field in ("insert_id", "role", "content"):
        if field not in entry:
            violations.append(f"missing field '{field}'")
    if violations:
        return violations
    if entry["insert_id"] != request["insert_id"]:
        violations.append("insert_id does not match the outstanding request")
    if entry["role"] != request["role"]:
        violations.append(f"role '{entry['role']}' does not match requested role '{request['role']}'")
    content = entry["content"]
    if "<tool_call>" in content:
        violations.append("content must not contain <tool_call>")
    required_marker = request.get("required_marker", "")
    if required_marker:
        if not content.startswith(required_marker):
            violations.append(f"content does not start with required marker '{required_marker}'")
        remainder = content[len(required_marker):]
        if "[STATE:" in remainder:
            violations.append("content contains a second [STATE:] marker")
    elif "[STATE:" in content:
        violations.append("user-role insert must not contain a [STATE:] marker")
    if not (20 <= len(content) <= 600):
        violations.append(f"content length {len(content)} outside [20, 600]")
    return violations


def load_accepted_ids(ledger_dir: Path) -> set[str]:
    accepted = ledger_dir / "accepted.jsonl"
    if not accepted.exists():
        return set()
    return {
        json.loads(line)["insert_id"]
        for line in accepted.read_text().splitlines() if line.strip()
    }


def run_agent_batch(batch_path: Path, scratch_dir: Path, timeout: int) -> dict:
    """Returns {"error": str|None, "ledger_path": Path|None}. Never raises."""
    if shutil.which("claude") is None:
        return {"error": "claude CLI not found", "ledger_path": None}
    ledger_path = batch_path.with_suffix(".ledger.jsonl")
    prompt = _PROMPT_TEMPLATE.format(batch_path=batch_path, ledger_path=ledger_path)
    cmd = ["claude", "-p", prompt, "--agent", "corpus-remediator", "--output-format", "json"]
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s", "ledger_path": None}
    if proc.returncode != 0:
        return {"error": f"claude exited {proc.returncode}: {proc.stderr.strip()[:200]}", "ledger_path": None}
    return {"error": None, "ledger_path": ledger_path if ledger_path.exists() else None}


def process_batch(requests: list[dict], scratch_dir: Path, timeout: int) -> tuple[list[dict], list[dict]]:
    """Returns (accepted_entries, rejected_records)."""
    batch_path = scratch_dir / "batch.json"
    batch_path.write_text(json.dumps(requests, ensure_ascii=False, indent=2))
    outcome = run_agent_batch(batch_path, scratch_dir, timeout)
    by_id = {r["insert_id"]: r for r in requests}
    accepted, rejected = [], []
    if outcome["error"] or outcome["ledger_path"] is None:
        for r in requests:
            rejected.append({"insert_id": r["insert_id"], "reasons": [outcome["error"] or "no ledger produced"]})
        return accepted, rejected
    seen_ids = set()
    for line in outcome["ledger_path"].read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        insert_id = entry.get("insert_id")
        request = by_id.get(insert_id)
        if request is None:
            continue
        seen_ids.add(insert_id)
        violations = validate_entry(entry, request)
        if violations:
            rejected.append({"insert_id": insert_id, "reasons": violations})
        else:
            entry.setdefault("schema_version", SCHEMA_VERSION)
            accepted.append(entry)
    for insert_id in by_id.keys() - seen_ids:
        rejected.append({"insert_id": insert_id, "reasons": ["no ledger entry produced"]})
    return accepted, rejected


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--triage-report", required=True)
    p.add_argument("--ledger-dir", required=True)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ledger_dir = Path(args.ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.triage_report).read_text())

    all_requests = []
    for rec in report["records"]:
        if rec["move"] not in ("insert_handoff_turn", "append_closing_pair"):
            continue
        stem, line_index = rec["key"]
        for ins in rec["inserts"]:
            all_requests.append({**ins, "conversation_id": rec["conversation_id"],
                                  "language": rec["language"], "key": [stem, line_index]})

    already_done = load_accepted_ids(ledger_dir)
    pending = [r for r in all_requests if r["insert_id"] not in already_done]
    if args.limit:
        pending = pending[: args.limit]

    batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    print(f"Pending inserts: {len(pending)} in {len(batches)} batch(es)")

    if args.dry_run:
        if batches:
            print(json.dumps(batches[0], indent=2, ensure_ascii=False)[:2000])
        return 0

    accepted_path = ledger_dir / "accepted.jsonl"
    rejected_path = ledger_dir / "rejected.jsonl"
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool, \
         tempfile.TemporaryDirectory() as scratch_root:
        futures = {
            pool.submit(process_batch, batch, Path(tempfile.mkdtemp(dir=scratch_root)), args.timeout): i
            for i, batch in enumerate(batches)
        }
        done = 0
        for future in as_completed(futures):
            accepted, rejected = future.result()
            with open(accepted_path, "a") as fh:
                for entry in accepted:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            with open(rejected_path, "a") as fh:
                for rec in rejected:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            print(f"batch {done}/{len(batches)}: +{len(accepted)} accepted, +{len(rejected)} rejected")

    print(f"Done. See {accepted_path} and {rejected_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_build_remediation_ledger.py -v`
Expected: PASS (6 tests, no real `claude` CLI invoked)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_remediation_ledger.py tests/unit/test_build_remediation_ledger.py
git commit -m "feat(scripts): add claude -p ledger driver for the 930 authoring cases"
```

**Do not run this script against the real corpus yet** — that is Task 15, gated on explicit user go-ahead (it spends money and API tokens).

---

## Task 13: `dvc.yaml` — new remediation stage

**Files:**
- Modify: `dvc.yaml`

**Interfaces:**
- Consumes: `scripts/remediate_task_a_states.py` (Task 4), `state_convention.py`/`state_convention_repair.py` (Tasks 1–3).
- Produces: a new stage `task_a_sft_remediate`; `task_a_sft_clean`'s `--input-dir` and first `deps` entry repointed.

- [ ] **Step 1: Read the current `task_a_sft_generate` and `task_a_sft_clean` stage blocks**

```bash
sed -n '1,90p' dvc.yaml
```

- [ ] **Step 2: Insert the new stage between them**

```yaml
  task_a_sft_remediate:
    desc: >-
      Bring the raw Task A corpus onto the tool-call state convention (a turn
      that emits <tool_call> annotates [STATE: X -> X]; the advance moves to
      a later turn). Deterministic: relabels the trajectory (never moving a
      tool call to a different state), then replays the DVC-tracked
      authoring ledger produced
      out-of-band by scripts/build_remediation_ledger.py. Every repaired row
      is re-validated and dropped on any violation. No API keys, no LLM
      calls at repro time. Produces task-a-sft-v2. See
      docs/task_a_state_convention_remediation_playbook.md.
    cmd: >-
      python scripts/remediate_task_a_states.py apply
      --input-dir data/output/sft/task_a
      --output-dir data/output/sft/task_a_remediated
      --ledger-dir data/interim/task_a_remediation_ledger
      --rebuild-prompts
      --on-unrepairable drop
    deps:
      - data/output/sft/task_a
      - data/interim/task_a_remediation_ledger
      - scripts/remediate_task_a_states.py
      - src/llm_workflow_agents/data/state_convention.py
      - src/llm_workflow_agents/data/state_convention_repair.py
    outs:
      - data/output/sft/task_a_remediated
```

- [ ] **Step 3: Repoint `task_a_sft_clean`**

Change its `deps` entry `data/output/sft/task_a` to `data/output/sft/task_a_remediated`, and the `cmd`'s `--input-dir data/output/sft/task_a` to `--input-dir data/output/sft/task_a_remediated`. Update its `desc` if it references stale row counts (check the current text — the file's own comment already flags such counts as stale).

- [ ] **Step 4: Validate the YAML**

```bash
source .venv/bin/activate && python3 -c "import yaml; yaml.safe_load(open('dvc.yaml'))" && echo OK
.venv-train/bin/dvc dag task_a_sft_clean
```

Expected: `OK`, and the DAG output shows `task_a_sft_remediate` upstream of `task_a_sft_clean`.

- [ ] **Step 5: Commit**

```bash
git add dvc.yaml
git commit -m "chore(dvc): add task_a_sft_remediate stage between generate and clean"
```

**Do not run `dvc repro` yet** — the `task_a_sft_remediate` stage depends on `data/interim/task_a_remediation_ledger`, which does not exist until Task 15 runs the (costly) ledger pass.

---

## Task 14 (gated — requires explicit go-ahead before running): Full corpus remediation run

This task spends real money (`claude -p` API usage, estimated $8–13 for ~1,465 authoring cases — see the design spec's Task-2 note for why this grew from the original ~930/$5–8 estimate after the `split_fused_tool_turn` placement-safety fix) and 1.5–3 hours of wall-clock time. **Do not run Steps 3+ without the user's explicit go-ahead in that session** — Steps 1–2 (triage, deterministic-only apply) are free and safe to run any time.

- [ ] **Step 1:** Re-run `triage` if the corpus changed since Task 4 Step 5; otherwise reuse `data/interim/task_a_state_triage/report.json`.
- [ ] **Step 2:** Run `remediate_task_a_states.py apply` with no `--ledger-dir` to confirm the deterministic-only kept/dropped counts match Task 4 Step 5's expectations.
- [ ] **Step 3 (COSTLY — confirm first):** `python scripts/build_remediation_ledger.py ... --limit 20`; hand-read every accepted entry in `data/interim/task_a_remediation_ledger/accepted.jsonl`.
- [ ] **Step 4 (COSTLY — confirm first):** Full ledger run (drop `--limit`).
- [ ] **Step 5:** Final `apply --ledger-dir ... --rebuild-prompts`, then `verify --strict`, then `diff`, then check all 9 acceptance gates from the design spec / playbook §7 by hand.
- [ ] **Step 6:** `git add` the (small) ledger dir if keeping it in git, or `dvc add data/interim/task_a_remediation_ledger` per the playbook; commit.

---

## Task 15 (gated — requires explicit go-ahead, needs teacher API keys): D3 retry slice + DVC repro + tag

- [ ] **Step 1 (COSTLY — confirm first, needs `GEMINI_API_KEY`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`):** Generate the 500–800-conversation retry-exhaustion slice: `python scripts/generate_sft_until_target.py --levels L3,L4,L5 --retry-exhaustion auto --samples-per-leg <n>`.
- [ ] **Step 2:** Merge via `scripts/concat_task_a.py` into `data/output/sft/task_a_remediated`.
- [ ] **Step 3:** `.venv-train/bin/dvc repro task_a_sft_remediate task_a_sft_clean task_a_sft_splits task_a_grpo`.
- [ ] **Step 4:** `.venv-train/bin/dvc status` — confirm clean; recompute and compare the output directory hash against `dvc.lock` (per the playbook's §10 lineage warning — this exact drift has happened twice before).
- [ ] **Step 5 (outward-facing — confirm first):** `.venv-train/bin/dvc push`; tag `task-a-sft-v2` with the confirmed hash in the annotation.
- [ ] **Step 6:** Retrain Cat A SFT on v2 (separate GPU work, out of scope for this repo environment — hand off via `scripts/run_phase2_sft.sh` on a machine with a GPU) and re-run `scripts/analyze_selfloop_habit.py` / `scripts/analyze_composite_decomposition.py` against the pre-registered success criterion (self-loop emission ≥30%, composite ≥0.75).

---

## Self-review notes

- **Spec coverage:** Part A (Tasks 1–4, 11–14), Part B (Tasks 5–10), model routing followed implicitly by task ordering (library/agent work first, plumbing after), DVC (Task 13), gated costly steps (Tasks 14–15) all present.
- **Placeholder scan:** Task 6 Step 2 and Task 7 Step 2 leave one sub-step ("write the actual fixture...", "follow the existing pattern...") pointing at an existing test file to mirror rather than inventing a parallel convention — this is intentional (existing-codebase-pattern-following per the skill's own guidance), not a TBD; the code that must exist (the wiring, the CLI, the library) is fully specified everywhere else.
- **Type consistency:** `RepairPlan.move`, `TurnPlan.content_op`, and `InsertRequest` field names are introduced once in Task 2 and used identically in Tasks 3, 4, 12, and the playbook.
- **Determinism guard:** Task 9 explicitly calls out using the per-sample `rng`, matching the Global Constraints section.
