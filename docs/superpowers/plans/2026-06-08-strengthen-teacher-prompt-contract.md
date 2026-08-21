# Strengthen Teacher Prompt (Workflow Contract) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, authoritative "WORKFLOW CONTRACT" (legal transitions + per-state tool permissions) to the teacher generation prompt so teacher output stops tripping the repair loop and falling back to the placeholder generator.

**Architecture:** A pure helper derives the exact constraint sets the repair loop enforces (`_find_transition_violations` legal edges + `find_tool_placement_violations` per-state tools); a renderer turns them into a prompt block inserted into `_build_teacher_prompt`; two reinforcing rules go into `_TEACHER_SYSTEM_PROMPT`. A structural no-drift test pins the rendered contract to the repair-loop checks. All changes are confined to teacher generation — `FORMAT_RULES` and the training/eval enrichment path are untouched.

**Tech Stack:** Python 3, `random.Random`, pytest. Env: `source .venv/bin/activate && …` (use `uv`, not pip).

**Spec:** `docs/superpowers/specs/2026-06-08-strengthen-teacher-prompt-contract-design.md`

---

## Background the engineer needs

The teacher generator (`generate_workflow_dataset` in
`src/llm_workflow_agents/data/generate_workflows.py`) runs a repair loop that judges a teacher
conversation *incoherent* if either:
- `_find_transition_violations(valid_edges, messages)` (`generate_workflows.py:52`) finds an
  assistant `[STATE: X → Y]` with `X ≠ Y` that is not a legal edge, or
- `find_tool_placement_violations(allowed, messages, schema_names)` (`_workflow_script.py:113`)
  finds a tool called from a state whose curated tool list does not include it.

Inside `generate_workflow_dataset` the constraint sets are built as (verbatim):
```python
allowed = {s.name: set(s.tools) for s in workflow.states}
id_to_name = {s.id: s.name for s in workflow.states}
valid_edge_pairs = {
    (id_to_name.get(t.from_state, t.from_state), id_to_name.get(t.to_state, t.to_state))
    for t in workflow.transitions
} | {(id_to_name.get(sid, sid), id_to_name.get(sid, sid))
     for sid in {t.from_state for t in workflow.transitions}}
```
`_find_transition_violations` only flags `X ≠ Y` pairs, so the **legal directed edges** the loop
enforces are the `X ≠ Y` subset of `valid_edge_pairs`.

Relevant dataclasses (already defined in `generate_workflows.py`):
- `WorkflowState(id, name, tools: list[str], …)`
- `WorkflowTransition(from_state, to_state, …)` — `from_state`/`to_state` hold state **ids**.
- `WorkflowGraph(states, transitions, initial_state, terminal_states)`.

Subgraphs are built by `select_subgraph(domain_spec, spec, rng, intent_category)` (already exists).

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/llm_workflow_agents/data/generate_workflows.py` | Modify | `_workflow_contract` (constraint extraction), `_render_workflow_contract` (prompt block), insert block into `_build_teacher_prompt`, add 2 rules to `_TEACHER_SYSTEM_PROMPT` |
| `tests/unit/test_teacher_prompt_contract.py` | Create | Structural no-drift test (contract == repair-loop sets; prompt contains contract) |

---

## Task 1: `_workflow_contract` extractor + no-drift parity test

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (add `_workflow_contract` after `_find_transition_violations`, which ends at line 66)
- Create: `tests/unit/test_teacher_prompt_contract.py`

- [ ] **Step 1: Write the failing parity test**

Create `tests/unit/test_teacher_prompt_contract.py`:

```python
"""The teacher-prompt WORKFLOW CONTRACT must exactly match what the repair loop enforces."""

from __future__ import annotations

import random

from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
import llm_workflow_agents.data.generate_workflows as gw

_LEVELS = [ComplexityLevel.L1, ComplexityLevel.L2, ComplexityLevel.L3]


def _repair_loop_sets(workflow):
    """Recompute exactly what generate_workflow_dataset's repair loop accepts."""
    allowed = {s.name: set(s.tools) for s in workflow.states}
    id_to_name = {s.id: s.name for s in workflow.states}
    valid = {
        (id_to_name.get(t.from_state, t.from_state),
         id_to_name.get(t.to_state, t.to_state))
        for t in workflow.transitions
    } | {
        (id_to_name.get(sid, sid), id_to_name.get(sid, sid))
        for sid in {t.from_state for t in workflow.transitions}
    }
    legal_edges = {(a, b) for (a, b) in valid if a != b}
    return legal_edges, allowed


def test_contract_edges_match_repair_loop():
    for level in _LEVELS:
        spec = COMPLEXITY_SPECS[level]
        for key, dom in DOMAIN_REGISTRY.items():
            rng = random.Random(0)
            wf = gw.select_subgraph(dom, spec, rng, "service")
            c_edges, _c_tools = gw._workflow_contract(wf)
            legal, _allowed = _repair_loop_sets(wf)
            assert c_edges == legal, f"{key}/{level}: edge mismatch {c_edges ^ legal}"


def test_contract_tools_match_repair_loop():
    for level in _LEVELS:
        spec = COMPLEXITY_SPECS[level]
        for key, dom in DOMAIN_REGISTRY.items():
            rng = random.Random(0)
            wf = gw.select_subgraph(dom, spec, rng, "service")
            _c_edges, c_tools = gw._workflow_contract(wf)
            _legal, allowed = _repair_loop_sets(wf)
            as_sets = {k: set(v) for k, v in c_tools.items()}
            assert as_sets == allowed, f"{key}/{level}: tool map mismatch"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_teacher_prompt_contract.py -q`
Expected: FAIL — `AttributeError: module 'llm_workflow_agents.data.generate_workflows' has no attribute '_workflow_contract'`.

- [ ] **Step 3: Add `_workflow_contract`**

In `src/llm_workflow_agents/data/generate_workflows.py`, immediately after the
`_find_transition_violations` function (it ends at line 66 with `return violations`), add:

```python
def _workflow_contract(
    workflow: "WorkflowGraph",
) -> tuple[set[tuple[str, str]], dict[str, list[str]]]:
    """Return (legal_directed_edges, tools_by_state) keyed by state NAME.

    Mirrors exactly what the generator's repair loop enforces:
    - legal_directed_edges: the X!=Y subset of valid_edge_pairs, i.e. every
      (from_name, to_name) transition where the two states differ. Staying in a
      state (X->X) is always allowed and is intentionally not listed.
    - tools_by_state: {state name: sorted tool names} == the ``allowed`` map used
      by find_tool_placement_violations.
    """
    id_to_name = {s.id: s.name for s in workflow.states}
    edges: set[tuple[str, str]] = set()
    for t in workflow.transitions:
        src = id_to_name.get(t.from_state, t.from_state)
        dst = id_to_name.get(t.to_state, t.to_state)
        if src != dst:
            edges.add((src, dst))
    tools_by_state = {s.name: sorted(s.tools) for s in workflow.states}
    return edges, tools_by_state
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_teacher_prompt_contract.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_teacher_prompt_contract.py
git commit -m "feat(data): _workflow_contract extractor mirroring repair-loop constraints"
```

---

## Task 2: Render the contract into the teacher prompt

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (add `_render_workflow_contract`; wire into `_build_teacher_prompt`, lines 959-1010)
- Modify: `tests/unit/test_teacher_prompt_contract.py`

- [ ] **Step 1: Write the failing render test**

Append to `tests/unit/test_teacher_prompt_contract.py`:

```python
def test_rendered_prompt_contains_full_contract():
    spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
    dom = DOMAIN_REGISTRY["account_management"]
    rng = random.Random(0)
    wf = gw.select_subgraph(dom, spec, rng, "service")
    tools = [t for t in dom.tools]
    prompt = gw._build_teacher_prompt(
        wf, tools, "cooperative", spec, dom, "en", "service",
    )
    assert "WORKFLOW CONTRACT" in prompt
    edges, tools_by_state = gw._workflow_contract(wf)
    # Every legal edge appears as an "X → Y" line.
    for src, dst in edges:
        assert f"{src} → {dst}" in prompt, f"missing edge line {src} → {dst}"
    # Every state's tool permission appears verbatim.
    for s in wf.states:
        st = tools_by_state[s.name]
        if st:
            assert f"{s.name}: {', '.join(st)}" in prompt, f"missing tools for {s.name}"
        else:
            assert f"{s.name}: (text only" in prompt, f"missing text-only line for {s.name}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_teacher_prompt_contract.py::test_rendered_prompt_contains_full_contract -q`
Expected: FAIL — `assert "WORKFLOW CONTRACT" in prompt` is False (block not rendered yet).

- [ ] **Step 3: Add `_render_workflow_contract`**

In `generate_workflows.py`, directly below `_workflow_contract` (added in Task 1), add:

```python
def _render_workflow_contract(workflow: "WorkflowGraph") -> str:
    """Render the legality/placement contract block for the teacher prompt."""
    edges, tools_by_state = _workflow_contract(workflow)
    if edges:
        edge_lines = "\n".join(f"  {src} → {dst}" for src, dst in sorted(edges))
    else:
        edge_lines = "  (none — single-state workflow; stay in the initial state)"
    tool_lines = []
    for s in workflow.states:  # preserve graph order for readability
        tools = tools_by_state.get(s.name, [])
        if tools:
            tool_lines.append(f"  {s.name}: {', '.join(tools)}")
        else:
            tool_lines.append(f"  {s.name}: (text only — no tools)")
    tools_block = "\n".join(tool_lines)
    return (
        "WORKFLOW CONTRACT — these are hard constraints. Output that violates them "
        "is rejected and regenerated.\n\n"
        "ALLOWED TRANSITIONS (only these state changes are legal; staying in the "
        "same state is always allowed):\n"
        f"{edge_lines}\n"
        "TOOL PERMISSIONS PER STATE (a tool may ONLY be called from a state that "
        "lists it):\n"
        f"{tools_block}\n\n"
        "Authority: the tool SCHEMAS define which tools exist and what arguments "
        "they take; this CONTRACT defines where each tool may be called and which "
        "transitions are legal; the workflow script is only a flow hint. When they "
        "disagree, the schema wins for arguments and the CONTRACT wins for "
        "placement and transitions."
    )
```

- [ ] **Step 4: Wire the block into `_build_teacher_prompt`**

In `_build_teacher_prompt` (line 959), build the block just before the `return` (after the
`transition_key = (...)` assignment that ends at line 994). Add:

```python
    contract_block = _render_workflow_contract(workflow)
```

Then change the return expression (lines 995-1010) so the block is inserted after the
`lang_instruction` blank line and before the workflow script — i.e. replace:

```python
        f"{lang_instruction}\n\n"
        f"Workflow script (natural language — follow this for conversation flow):\n{script}\n\n"
```

with:

```python
        f"{lang_instruction}\n\n"
        f"{contract_block}\n\n"
        f"Workflow script (natural language — follow this for conversation flow):\n{script}\n\n"
```

- [ ] **Step 5: Run the render test (and the parity tests) to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_teacher_prompt_contract.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_teacher_prompt_contract.py
git commit -m "feat(data): insert WORKFLOW CONTRACT block into teacher prompt"
```

---

## Task 3: Reinforcing rules in `_TEACHER_SYSTEM_PROMPT`

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (`_TEACHER_SYSTEM_PROMPT` RULES list)
- Modify: `tests/unit/test_teacher_prompt_contract.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_teacher_prompt_contract.py`:

```python
def test_teacher_system_prompt_references_contract():
    sp = gw._TEACHER_SYSTEM_PROMPT
    assert "ALLOWED TRANSITIONS" in sp
    assert "TOOL PERMISSIONS PER STATE" in sp
    assert "never invent a transition" in sp.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_teacher_prompt_contract.py::test_teacher_system_prompt_references_contract -q`
Expected: FAIL — those phrases are not in `_TEACHER_SYSTEM_PROMPT` yet.

- [ ] **Step 3: Add the two rules**

In `generate_workflows.py`, locate the `_TEACHER_SYSTEM_PROMPT` string's RULES list. Find this line:

```python
- When invoking a tool include <tool_call>{"name": "...", "arguments": {...}}</tool_call>.
```

Insert these two bullet lines immediately after it:

```python
- Every [STATE: X → Y] you emit with X != Y MUST appear in the WORKFLOW CONTRACT's ALLOWED TRANSITIONS list (provided in the user message). Never invent a transition; if unsure how to proceed, stay in the current state ([STATE: X → X]).
- Only call a tool from a state that lists it under TOOL PERMISSIONS PER STATE. Never call a tool in a state marked "text only", and never call a tool absent from the tool schemas.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_teacher_prompt_contract.py::test_teacher_system_prompt_references_contract -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_teacher_prompt_contract.py
git commit -m "feat(data): teacher system prompt enforces WORKFLOW CONTRACT transitions + tool placement"
```

---

## Final Verification

- [ ] **Full contract test file green**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_teacher_prompt_contract.py -q`
Expected: PASS (4 tests).

- [ ] **No regression in the existing teacher-prompt tests**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py -q`
Expected: PASS (includes `TestTeacherPromptOutbound`, which still expects `OUTBOUND` in the prompt and no outbound block for inbound).

- [ ] **Full unit suite green**

Run: `source .venv/bin/activate && python -m pytest tests/unit -q`
Expected: PASS.

- [ ] **Eyeball one rendered prompt** (sanity, not a gate)

```bash
source .venv/bin/activate && python -c "
import random
from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
import llm_workflow_agents.data.generate_workflows as gw
spec = COMPLEXITY_SPECS[ComplexityLevel.L2]; dom = DOMAIN_REGISTRY['banking']
wf = gw.select_subgraph(dom, spec, random.Random(0), 'service')
print(gw._render_workflow_contract(wf))
"
```
Confirm the ALLOWED TRANSITIONS and TOOL PERMISSIONS lists read sensibly for the domain.

- [ ] **(Optional, separate — needs API key) Live fallback-rate check**

Run a teacher generation with a key and compare the `repair_fallbacks` stat before/after on the
same seed, e.g. `./scripts/generate_benchmark_data_teacher.sh --teacher gemini-3-5-flash --levels L1 --samples 50`.

---

## Self-Review

- **Spec coverage:** §1 contract helpers → Task 1 (`_workflow_contract`) + Task 2 (`_render_workflow_contract`); §2 CONTRACT block + placement → Task 2; §3 reinforcing rules → Task 3; §4 structural test (edges equal, tools equal, prompt contains) → Tasks 1-3 tests. `FORMAT_RULES` untouched (no task modifies it). ✓
- **Type consistency:** `_workflow_contract(workflow) -> (set[tuple[str,str]], dict[str,list[str]])` is defined in Task 1 and consumed unchanged in Task 2's renderer and all tests. `WorkflowTransition.from_state/to_state` (ids) + `WorkflowState.id/name/tools` used consistently. ✓
- **No placeholders:** every code/step is concrete; the only deferred item (live API check) is explicitly marked optional/out-of-session. ✓
