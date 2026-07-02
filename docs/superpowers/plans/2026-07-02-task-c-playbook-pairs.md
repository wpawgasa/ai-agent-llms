# Task C Playbook→Graph Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the data-generation pipeline specified in `docs/data_generation_recipes_task_c.md` — ~5,000 (NL playbook → WorkflowGraph JSON) training pairs via graph-first inverse rendering.

**Architecture:** Four new modules under `src/llm_workflow_agents/data/` (graph invention, playbook rendering, verification, orchestration), two stdlib-only scripts (clean, split), two shell runners, three DVC stages. Gold graphs come 70% from the existing registry subgraph sampler and 30% from teacher invention; teachers render each graph as 4–6 playbook registers; deterministic string checks + cross-family back-extraction gate every rendering.

**Tech Stack:** Python ≥3.11, existing `call_teacher_model` client, `jsonschema`, `eval/graph_extraction_eval.py` predicates, pytest with monkeypatched teachers.

**Context:** The recipe spec (`docs/data_generation_recipes_task_c.md`, committed 6645713) defines every field, threshold, and CLI. The repo already provides: `select_subgraph(domain, spec, rng, intent_category="service") -> WorkflowGraph` (`generate_workflows.py:244`), `_select_domain(rng, domain, spec)` (`:831`), `COMPLEXITY_SPECS`/`ComplexityLevel` (`config/schema.py`), `call_teacher_model(model, system_prompt, user_prompt) -> str` with retries (`_teacher_client.py:156`), `build_workflow_script(workflow_graph: dict, tool_schemas=None, language="en", messages=None) -> str` (`_workflow_script.py:274`), eval predicates `check_structural_validity`/`check_mermaid_renderability`/`parse_graph_json`/`evaluate_graph_extraction` (`eval/graph_extraction_eval.py`), and the repair-feedback pattern (`generate_workflows.py:1599-1623`, `## CORRECTIONS REQUIRED` appended to user prompt).

## Global Constraints

- Spec is normative: `docs/data_generation_recipes_task_c.md`. Field names, thresholds (node F1 ≥ 0.90 ∧ edge F1 ≥ 0.80 back-extraction gate; edge-ref coverage ≥ 0.90 with 100% of priority>0 edges; halt leg at >20% spot-check failures), seeds (SFT=142, benchmark=200), and CLIs must match it verbatim.
- Graph interchange format inside the pipeline = the name-keyed `WorkflowGraph.to_dict()` shape (`{"states","state_details","transitions"("from"/"to"),"initial","terminal"}`). The id-keyed eval shape (`graph_output_schema.json`: `nodes/edges(from_state/to_state)/initial_state/terminal_states`) appears only in: invention-teacher output, row assembly, back-extraction scoring.
- `scripts/*.py` are stdlib-only (argparse/json/random/pathlib), pure testable core functions, mirroring `clean_task_a_sft.py` / `split_task_a_sft.py`.
- Shell runners mirror `generate_sft_data.sh`: `set -euo pipefail`, `.env` sourcing via `set -a; source; set +a`, `run()` dry-run wrapper, provider-prefix→API-key `case` check, `python3 -c "from llm_workflow_agents..."` invocation (no venv/uv inside scripts).
- Tests live in `tests/unit/`, import the installed package; scripts imported via `sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))`. Teacher never called for real: monkeypatch `call_teacher_model` **in the module that imports it** (e.g. `_playbook_render.call_teacher_model`).
- Run tests with `source .venv/bin/activate && python -m pytest tests/unit/<file> -q` (user preference: always activate venv first; use uv—not pip—if any dependency work arises).
- All new modules start with `from __future__ import annotations`.
- Commit after each task (small conventional commits). Branch: create `feat/task-c-playbook-pairs` from main before Task 1.

---

### Task 1: `_graph_invention.py` — domain briefs, gold-graph validation, teacher invention

**Files:**
- Create: `src/llm_workflow_agents/data/_graph_invention.py`
- Test: `tests/unit/test_graph_invention.py`

**Interfaces:**
- Consumes: `call_teacher_model` (`_teacher_client.py:156`); `check_structural_validity`, `check_mermaid_renderability`, `WorkflowGraph` (eval dataclass) from `llm_workflow_agents.eval.graph_extraction_eval`; `jsonschema.validate` against `data/templates/graph_output_schema.json`.
- Produces:
  - `@dataclass(frozen=True) DomainBrief(slug: str, title: str, description: str, suggested_tools: tuple[str, ...])`
  - `DOMAIN_BRIEFS: tuple[DomainBrief, ...]` — 25 briefs including the 8 spec examples (hr_onboarding, it_incident_response, loan_underwriting, restaurant_reservations, warehouse_returns, clinical_trial_screening, freight_dispatch, campus_admissions) + 17 more non-registry domains.
  - `validate_gold_graph(graph_dict: dict) -> list[str]` — id-keyed eval shape in, itemized violation strings out (empty = valid).
  - `@dataclass(frozen=True) InventedGraph(brief_slug: str, graph: dict)` — `.graph` is **name-keyed** interchange.
  - `invent_novel_graphs(briefs: Sequence[DomainBrief], count: int, teacher_model: str, rng: random.Random, max_repair_retries: int = 2) -> list[InventedGraph]`
  - `_normalize_to_name_keyed(eval_dict: dict) -> dict` (private)

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_graph_invention.py
from __future__ import annotations

import json
import random

import llm_workflow_agents.data._graph_invention as gi
from llm_workflow_agents.data._graph_invention import (
    DOMAIN_BRIEFS, InventedGraph, invent_novel_graphs, validate_gold_graph,
)

WORKED_EXAMPLE = {  # spec §Worked Example gold JSON, verbatim
    "nodes": [
        {"id": "S1", "name": "VERIFY_POLICYHOLDER", "tools": ["verify_policy"], "entry_actions": []},
        {"id": "S2", "name": "CLAIM_INTAKE", "tools": ["file_claim"], "entry_actions": []},
        {"id": "S3", "name": "ASSESS_COVERAGE", "tools": [], "entry_actions": []},
        {"id": "S4", "name": "RESOLVE", "tools": [], "entry_actions": []},
        {"id": "S5", "name": "REQUEST_DOCUMENTATION", "tools": [], "entry_actions": []},
        {"id": "S6", "name": "TERMINAL", "tools": [], "entry_actions": []},
    ],
    "edges": [
        {"from_state": "S1", "to_state": "S2", "condition": "policyholder verified", "priority": 0},
        {"from_state": "S2", "to_state": "S3", "condition": "claim filed", "priority": 0},
        {"from_state": "S3", "to_state": "S4", "condition": "coverage assessed", "priority": 0},
        {"from_state": "S3", "to_state": "S5", "condition": "documentation missing", "priority": 1},
        {"from_state": "S5", "to_state": "S3", "condition": "documents received", "priority": 0},
        {"from_state": "S4", "to_state": "S6", "condition": "case closed", "priority": 0},
    ],
    "initial_state": "S1",
    "terminal_states": ["S6"],
}


def test_validate_gold_graph_accepts_worked_example():
    assert validate_gold_graph(WORKED_EXAMPLE) == []


def test_validate_gold_graph_itemizes_violations():
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    bad["edges"].append({"from_state": "S4", "to_state": "S4", "condition": "loop", "priority": 0})
    bad["nodes"][1]["name"] = "verify_policyholder"  # not SCREAMING_SNAKE
    violations = validate_gold_graph(bad)
    assert any("self-loop" in v for v in violations)
    assert any("SCREAMING_SNAKE" in v for v in violations)


def test_validate_gold_graph_rejects_sink_and_dup_priority():
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    bad["edges"] = [e for e in bad["edges"] if e["from_state"] != "S5"]  # S5 = non-terminal sink
    violations = validate_gold_graph(bad)
    assert any("no outgoing" in v for v in violations)


def test_invent_repairs_then_accepts(monkeypatch):
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    bad["edges"].append({"from_state": "S2", "to_state": "S2", "condition": "x", "priority": 0})
    responses = iter([json.dumps(bad), json.dumps(WORKED_EXAMPLE)])
    prompts: list[str] = []

    def fake_teacher(model, system_prompt, user_prompt):
        prompts.append(user_prompt)
        return next(responses)

    monkeypatch.setattr(gi, "call_teacher_model", fake_teacher)
    out = invent_novel_graphs(DOMAIN_BRIEFS[:1], count=1, teacher_model="gpt-x", rng=random.Random(1))
    assert len(out) == 1 and isinstance(out[0], InventedGraph)
    assert "CORRECTIONS REQUIRED" in prompts[1] and "self-loop" in prompts[1]
    assert out[0].graph["initial"] == "VERIFY_POLICYHOLDER"  # name-keyed, ids discarded


def test_invent_drops_after_exhausted_repairs(monkeypatch):
    bad = json.loads(json.dumps(WORKED_EXAMPLE))
    del bad["initial_state"]
    calls = []
    monkeypatch.setattr(gi, "call_teacher_model",
                        lambda m, s, u: (calls.append(1), json.dumps(bad))[1])
    out = invent_novel_graphs(DOMAIN_BRIEFS[:1], count=1, teacher_model="gpt-x", rng=random.Random(1))
    assert out == [] and len(calls) == 3  # initial + 2 repairs


def test_briefs_wellformed():
    slugs = [b.slug for b in DOMAIN_BRIEFS]
    assert len(DOMAIN_BRIEFS) >= 25 and len(set(slugs)) == len(slugs)
    from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
    assert not set(slugs) & set(DOMAIN_REGISTRY)  # disjoint from registry
```

- [ ] **Step 2: Run tests, verify they fail** — `source .venv/bin/activate && python -m pytest tests/unit/test_graph_invention.py -q` → ImportError (module missing).

- [ ] **Step 3: Implement `_graph_invention.py`**

Key pieces (full violation logic; schema loaded once at module level):

```python
_SCHEMA = json.loads((Path(__file__).parents[3] / "data/templates/graph_output_schema.json").read_text())
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

def validate_gold_graph(graph_dict: dict) -> list[str]:
    violations: list[str] = []
    try:
        jsonschema.validate(graph_dict, _SCHEMA)
    except jsonschema.ValidationError as exc:
        return [f"schema: {exc.message}"]
    names = [n["name"] for n in graph_dict["nodes"]]
    ids = {n["id"] for n in graph_dict["nodes"]}
    if len(set(names)) != len(names):
        violations.append("duplicate state name")
    for name in names:
        if not _NAME_RE.match(name):
            violations.append(f"state name not SCREAMING_SNAKE: {name}")
    outgoing: dict[str, list[int]] = {i: [] for i in ids}
    for e in graph_dict["edges"]:
        if e["from_state"] == e["to_state"]:
            violations.append(f"self-loop edge: {e['from_state']}")
        if e["from_state"] in outgoing:
            outgoing[e["from_state"]].append(e.get("priority", 0))
    terminals = set(graph_dict["terminal_states"])
    for node_id, prios in outgoing.items():
        if node_id not in terminals and not prios:
            violations.append(f"non-terminal state has no outgoing edge: {node_id}")
        if len(prios) != len(set(prios)):
            violations.append(f"duplicate edge priorities from: {node_id}")
    eval_graph = EvalWorkflowGraph(nodes=graph_dict["nodes"], edges=graph_dict["edges"],
                                   initial_state=graph_dict["initial_state"],
                                   terminal_states=graph_dict["terminal_states"])
    if not check_structural_validity(eval_graph):
        violations.append("structural validity failed (initial/terminal reachability or orphan nodes)")
    if not check_mermaid_renderability(eval_graph):
        violations.append("mermaid renderability failed (invalid ids or dangling edge endpoints)")
    return violations
```

`invent_novel_graphs`: for each requested graph, `rng.choice(briefs)`, call teacher with `_INVENTION_SYSTEM` (instructs: return ONLY the JSON object per schema; 4–14 states; SCREAMING_SNAKE names; ≥1 terminal; tools chosen from the brief's `suggested_tools`; S-ids in any order — they will be discarded) and a user prompt containing the brief + the JSON schema text. `json.loads` → `validate_gold_graph` → on violations append `## CORRECTIONS REQUIRED\n- ...` and retry (≤2), else `_normalize_to_name_keyed` and append. Teacher exception or exhausted repairs → skip (count continues to next attempt slot; function returns what succeeded).

`_normalize_to_name_keyed`: build id→name map, emit `{"states": [names], "state_details": [{"name","tools","entry_actions","instruction": ""}], "transitions": [{"from": name, "to": name, "condition", "priority"}], "initial": name, "terminal": [names]}`.

- [ ] **Step 4: Run tests, verify pass** — same command, expect all green.
- [ ] **Step 5: Commit** — `git add src/llm_workflow_agents/data/_graph_invention.py tests/unit/test_graph_invention.py && git commit -m "feat(data): add Task C gold-graph invention and validation"`

---

### Task 2: `generate_playbook_pairs.py` part 1 — pool assembly, split pre-assignment, rendering plans

**Files:**
- Create: `src/llm_workflow_agents/data/generate_playbook_pairs.py`
- Test: `tests/unit/test_playbook_pairs.py`

**Interfaces:**
- Consumes: `select_subgraph`, `_select_domain` (`generate_workflows.py:244,:831`); `COMPLEXITY_SPECS`, `ComplexityLevel` (`llm_workflow_agents.config.schema`); `DOMAIN_REGISTRY`; `invent_novel_graphs`, `DOMAIN_BRIEFS` (Task 1).
- Produces:
  - `TASK_C_LEVEL_WEIGHTS = {"L1": 0.10, "L2": 0.25, "L3": 0.35, "L4": 0.20, "L5": 0.10}`
  - `REGISTERS = ("sop_document", "prose_narrative", "bullet_quick_reference", "wiki_training_notes", "state_script", "manager_transcript")` (re-exported from Task 3's module once it exists; define here first, move in Task 3 — final home is `_playbook_render.Register`)
  - `@dataclass GraphPoolEntry(graph_id: str, source: str, domain: str, complexity_level: str, graph: dict, tool_schemas: list[dict])`
  - `build_graph_pool(num_graphs: int, invented_ratio: float, invention_teacher: str, rng: random.Random) -> list[GraphPoolEntry]` — graph_ids `G0001…` in draw order; registry entries: draw level by `TASK_C_LEVEL_WEIGHTS`, `_select_domain(rng, None, spec)`, `select_subgraph(...)`, `graph = workflow.to_dict()`, `tool_schemas = [t for t in domain_spec.tools if t["function"]["name"] in used]`; invented entries: from `invent_novel_graphs`, tool_schemas = minimal stubs `{"type":"function","function":{"name": tool, "description": tool.replace("_"," "), "parameters": {"type":"object","properties":{}}}}` for tools named in states.
  - `assign_splits(pool, heldout_domains=("utilities","surveys"), heldout_brief_fraction=0.20, ratios=(0.85, 0.10, 0.05), seed: int = 142) -> dict[str, str]`
  - `@dataclass RenderingPlan(pair_id: str, graph_id: str, register: str, language: str, distractor_count: int, paraphrase_density: str, condition_explicitness: str)`
  - `plan_renderings(pool, splits, language_mix, rng) -> list[RenderingPlan]`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_playbook_pairs.py  (pool/split/plan section)
def _fake_invention(monkeypatch, n):
    """Patch invent_novel_graphs to return n copies of a tiny valid name-keyed graph."""
    import llm_workflow_agents.data.generate_playbook_pairs as gpp
    tiny = {"states": ["START", "WORK", "TERMINAL"],
            "state_details": [{"name": "START", "tools": [], "entry_actions": [], "instruction": ""},
                              {"name": "WORK", "tools": ["do_thing"], "entry_actions": [], "instruction": ""},
                              {"name": "TERMINAL", "tools": [], "entry_actions": [], "instruction": ""}],
            "transitions": [{"from": "START", "to": "WORK", "condition": "begin", "priority": 0},
                            {"from": "WORK", "to": "TERMINAL", "condition": "done", "priority": 0}],
            "initial": "START", "terminal": ["TERMINAL"]}
    from llm_workflow_agents.data._graph_invention import InventedGraph, DOMAIN_BRIEFS
    fakes = [InventedGraph(DOMAIN_BRIEFS[i % len(DOMAIN_BRIEFS)].slug, tiny) for i in range(n)]
    monkeypatch.setattr(gpp, "invent_novel_graphs", lambda *a, **k: fakes[: k.get("count", a[1])])


def test_build_graph_pool_ratio_levels_determinism(monkeypatch):
    _fake_invention(monkeypatch, 30)
    pool1 = build_graph_pool(100, 0.30, "gpt-x", random.Random(142))
    pool2 = build_graph_pool(100, 0.30, "gpt-x", random.Random(142))
    assert [(e.graph_id, e.source, e.domain, e.complexity_level) for e in pool1] == \
           [(e.graph_id, e.source, e.domain, e.complexity_level) for e in pool2]
    assert pool1[0].graph_id == "G0001" and pool1[99].graph_id == "G0100"
    n_inv = sum(1 for e in pool1 if e.source == "invented")
    assert 25 <= n_inv <= 35
    l3 = sum(1 for e in pool1 if e.complexity_level == "L3")
    assert 20 <= l3 <= 50  # 35% ± tolerance on registry-only levels


def test_assign_splits_heldout_axes(monkeypatch):
    _fake_invention(monkeypatch, 30)
    pool = build_graph_pool(100, 0.30, "gpt-x", random.Random(142))
    splits = assign_splits(pool, seed=142)
    assert splits == assign_splits(pool, seed=142)  # deterministic
    for e in pool:
        if e.domain in ("utilities", "surveys"):
            assert splits[e.graph_id] == "test"
    counts = {s: list(splits.values()).count(s) for s in ("train", "validation", "test")}
    assert counts["train"] > counts["validation"] > 0 and counts["test"] > 0


def test_plan_renderings_constraints(monkeypatch):
    _fake_invention(monkeypatch, 30)
    pool = build_graph_pool(100, 0.30, "gpt-x", random.Random(142))
    splits = assign_splits(pool, seed=142)
    plans = plan_renderings(pool, splits,
                            {"en": 0.5, "th": 0.3, "code_switch": 0.2}, random.Random(142))
    by_graph: dict[str, list] = {}
    for p in plans:
        by_graph.setdefault(p.graph_id, []).append(p)
    for gid, ps in by_graph.items():
        regs = [p.register for p in ps]
        assert 4 <= len(regs) <= 6 and len(set(regs)) == len(regs)
        assert "state_script" in regs
        assert {"sop_document", "prose_narrative"} & set(regs)
        if splits[gid] == "train":
            assert "manager_transcript" not in regs
        assert all(p.pair_id == f"{gid}_r{i+1}" for i, p in enumerate(ps))
    langs = [p.language for p in plans]
    assert 0.35 < langs.count("en") / len(langs) < 0.65
    with_distractors = sum(1 for p in plans if p.distractor_count > 0)
    assert 0.15 < with_distractors / len(plans) < 0.45
```

- [ ] **Step 2: Run tests, verify fail** — ImportError.
- [ ] **Step 3: Implement.** `assign_splits` full logic:

```python
def assign_splits(pool, heldout_domains=("utilities", "surveys"),
                  heldout_brief_fraction=0.20, ratios=(0.85, 0.10, 0.05),
                  seed: int = 142) -> dict[str, str]:
    rng = random.Random(seed)
    by_id = {e.graph_id: e for e in pool}
    ids = sorted(by_id)
    invented_briefs = sorted({e.domain for e in pool if e.source == "invented"})
    n_held = max(1, round(len(invented_briefs) * heldout_brief_fraction)) if invented_briefs else 0
    held_briefs = set(rng.sample(invented_briefs, n_held)) if n_held else set()
    heldout = set(heldout_domains) | held_briefs
    test = {gid for gid in ids if by_id[gid].domain in heldout}
    remaining = [gid for gid in ids if gid not in test]
    rng.shuffle(remaining)
    n = len(ids)
    while len(test) < round(n * ratios[2]) and remaining:
        test.add(remaining.pop())
    n_val = round(n * ratios[1])
    val = set(remaining[:n_val])
    return {gid: "test" if gid in test else "validation" if gid in val else "train"
            for gid in ids}
```

`plan_renderings` per graph: `k = rng.randint(4, 6)`; candidate registers = all 6 minus `manager_transcript` if train; mandatory picks = `state_script` + `rng.choice(("sop_document","prose_narrative"))`; fill remaining slots by `rng.sample` from the rest; knobs drawn per spec (`distractor_count`: 0 with p=0.70 else `rng.randint(1,3)`; `paraphrase_density` weights 40/40/20; `condition_explicitness` weights 40/35/25); `language` weighted by `language_mix`.

- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(data): Task C graph pool, split pre-assignment, rendering plans"`

---

### Task 3: `_playbook_render.py` — registers, prompts, distractors, `render_playbook`

**Files:**
- Create: `src/llm_workflow_agents/data/_playbook_render.py`
- Modify: `src/llm_workflow_agents/data/generate_playbook_pairs.py` (import `Register` from here; drop the temporary `REGISTERS` tuple)
- Test: `tests/unit/test_playbook_render.py`

**Interfaces:**
- Consumes: `build_workflow_script` (`_workflow_script.py:274`), `call_teacher_model`.
- Produces:
  - `class Register(StrEnum)`: `SOP_DOCUMENT, PROSE_NARRATIVE, BULLET_QUICK_REFERENCE, WIKI_TRAINING_NOTES, STATE_SCRIPT, MANAGER_TRANSCRIPT` (values = spec keys)
  - `TEACHER_REGISTERS: frozenset[Register]` (all minus STATE_SCRIPT)
  - `DISTRACTOR_LIBRARY: dict[str, tuple[str, ...]]` — keys `"en"`, `"th"`; ~8 paragraphs each (compliance boilerplate, tone guidance, SLA text, data-privacy reminders); `code_switch` draws from both.
  - `PLAYBOOK_LANGUAGE_INSTRUCTIONS: dict[str, str]` — en/th/code_switch, modeled on `_LANGUAGE_INSTRUCTIONS` (`generate_workflows.py:1065`) but for documents; th/code_switch variants state: "State names (SCREAMING_SNAKE), tool names, and any JSON stay in English/ASCII."
  - `draw_distractors(count: int, language: str, rng: random.Random, forbidden_terms: Iterable[str]) -> list[str]`
  - `build_render_prompts(graph_dict, tool_schemas, register, language, knobs) -> tuple[str, str]`
  - `render_playbook(graph_dict, tool_schemas, register, language, knobs, teacher_model, rng, corrections: list[str] | None = None) -> str`

Where `knobs` is the `RenderingPlan` fields dict (`distractor_count`, `paraphrase_density`, `condition_explicitness`).

- [ ] **Step 1: Write failing tests**

```python
def test_state_script_no_teacher_call(monkeypatch):
    import llm_workflow_agents.data._playbook_render as pr
    monkeypatch.setattr(pr, "call_teacher_model",
                        lambda *a: (_ for _ in ()).throw(AssertionError("teacher called")))
    text = pr.render_playbook(TINY_GRAPH, [], pr.Register.STATE_SCRIPT, "en",
                              {"distractor_count": 0, "paraphrase_density": "low",
                               "condition_explicitness": "explicit"},
                              teacher_model="gpt-x", rng=random.Random(1))
    for name in TINY_GRAPH["states"]:
        assert name in text


def test_render_prompt_contains_contract():
    import llm_workflow_agents.data._playbook_render as pr
    for register in pr.TEACHER_REGISTERS:
        system, user = pr.build_render_prompts(TINY_GRAPH, TOOL_SCHEMAS, register, "en", KNOBS)
        for name in TINY_GRAPH["states"]:
            assert name in user
        assert "```json" in user                       # fenced gold graph (test-mock affordance)
        assert "verbatim at least once" in user        # anchor rule
        assert "before any other state" in user        # initial-first rule
        assert "explicitly signal" in user             # terminal signposting


def test_render_prompt_language_and_knobs():
    import llm_workflow_agents.data._playbook_render as pr
    _, user_th = pr.build_render_prompts(TINY_GRAPH, [], pr.Register.SOP_DOCUMENT, "th", KNOBS)
    assert "English/ASCII" in user_th
    _, user_lo = pr.build_render_prompts(TINY_GRAPH, [], pr.Register.BULLET_QUICK_REFERENCE, "en",
                                         dict(KNOBS, condition_explicitness="listing_order"))
    assert "order they are listed" in user_lo
    knobs2 = dict(KNOBS, distractor_count=2, _distractors=["ALPHA BOILERPLATE", "BETA BOILERPLATE"])
    _, user_d = pr.build_render_prompts(TINY_GRAPH, [], pr.Register.SOP_DOCUMENT, "en", knobs2)
    assert "ALPHA BOILERPLATE" in user_d and "BETA BOILERPLATE" in user_d


def test_render_corrections_appended(monkeypatch):
    import llm_workflow_agents.data._playbook_render as pr
    captured = {}
    monkeypatch.setattr(pr, "call_teacher_model",
                        lambda m, s, u: captured.update(user=u) or json.dumps({"playbook": "text"}))
    pr.render_playbook(TINY_GRAPH, [], pr.Register.SOP_DOCUMENT, "en", KNOBS,
                       "gpt-x", random.Random(1), corrections=["missing anchor: WORK"])
    assert "CORRECTIONS REQUIRED" in captured["user"] and "missing anchor: WORK" in captured["user"]


def test_distractor_library_globally_pure():
    import llm_workflow_agents.data._playbook_render as pr
    from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
    all_names = {s.name for d in DOMAIN_REGISTRY.values() for s in d.states}
    all_tools = {t["function"]["name"] for d in DOMAIN_REGISTRY.values() for t in d.tools}
    for lang, paras in pr.DISTRACTOR_LIBRARY.items():
        for para in paras:
            assert not any(re.search(rf"(?<![A-Z0-9_]){re.escape(n)}(?![A-Z0-9_])", para)
                           for n in all_names | all_tools)


def test_draw_distractors_deterministic_and_filtered():
    import llm_workflow_agents.data._playbook_render as pr
    a = pr.draw_distractors(2, "en", random.Random(7), forbidden_terms=[])
    b = pr.draw_distractors(2, "en", random.Random(7), forbidden_terms=[])
    assert a == b and len(a) == 2
```

- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement.** Teacher contract: response is `{"playbook": "<text>"}` (JSON mode already requested by `call_teacher_model` providers). `_RENDER_SYSTEM` states the role ("You author internal business playbooks from workflow graphs") and the JSON envelope. `build_render_prompts` assembles the user prompt from labeled blocks, in order:
  1. Register style guide (per-register paragraph — SOP: numbered sections/formal; prose: flowing paragraphs, no headers; bullets: terse fragments; wiki: informal, asides, emoji ok; transcript: spoken monologue, digressions).
  2. `## GOLD WORKFLOW GRAPH` + fenced ```json block of `{"graph": graph_dict, "tools": tool_schemas}`.
  3. `## HARD REQUIREMENTS` — the contract lines the tests assert: each state's canonical name must appear **verbatim at least once** (heading, parenthetical, or quoted label); the initial state must be introduced **before any other state**; the document must **explicitly signal** where the workflow ends (terminal states); every tool granted to a state must be named verbatim in that state's description; for every transition, mention the target state's name within or immediately after the source state's description; branch transitions (priority > 0) must always be described alongside their source state.
  4. Knob instructions: paraphrase density (low: reuse the canonical name throughout / medium: canonical once then paraphrase / high: canonical once, aggressive paraphrase + pronouns); condition explicitness (explicit: state each condition; narrative_order: imply the default path by sequencing; listing_order: "describe alternatives in the **order they are listed**, first = default"); distractors: "Insert the following paragraphs verbatim as standalone paragraphs at natural points: <texts>".
  5. Language instruction from `PLAYBOOK_LANGUAGE_INSTRUCTIONS`.
  6. Optional `## CORRECTIONS REQUIRED` block (same wording pattern as `generate_workflows.py:1273-1282`).

  `render_playbook` dispatch: STATE_SCRIPT → `build_workflow_script(graph_dict, tool_schemas, language="en" if language == "code_switch" else language)`, then if `distractor_count > 0` insert drawn distractors between `### [` sections deterministically via `rng`. Teacher registers → prompts, `call_teacher_model`, `json.loads(raw)["playbook"]`, raise `ValueError` on empty. Distractors are drawn by the **caller** (orchestrator) so the purity check can redraw without re-rendering; `knobs["_distractors"]` carries the texts into the prompt.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(data): Task C playbook registers and teacher rendering"`

---

### Task 4: `_playbook_verify.py` part 1 — deterministic checks

**Files:**
- Create: `src/llm_workflow_agents/data/_playbook_verify.py`
- Test: `tests/unit/test_playbook_verify.py`

**Interfaces:**
- Produces:
  - `find_anchor_occurrences(text: str, state_names: Sequence[str]) -> list[tuple[int, str]]`
  - `assign_state_ids(graph_dict: dict, playbook: str) -> dict[str, str] | None` — name→`S<n>` by mention order; `None` if any state unanchored
  - `graph_to_eval_shape(graph_dict: dict, state_ids: dict[str, str]) -> dict` — id-keyed eval schema
  - `@dataclass EdgeRefResult(coverage: float, failed_edges: list[tuple[str, str]], branch_ok: bool)`
  - `check_edge_references(playbook: str, graph_dict: dict) -> EdgeRefResult`
  - `check_distractor_purity(distractors: list[str], state_names: Iterable[str], tool_names: Iterable[str]) -> list[str]` — returns offending distractor texts
  - `@dataclass VerificationReport(accepted: bool, corrections: list[str], anchor_coverage: float, tool_coverage: float, edge_ref_coverage: float, initial_first: bool, state_ids: dict[str, str] | None, back_extraction: dict | None = None)`
  - `verify_rendering(playbook: str, graph_dict: dict, tool_names: Iterable[str]) -> VerificationReport`

- [ ] **Step 1: Write failing tests** (core algorithm cases; full test file also covers the spec worked example verbatim)

```python
def test_anchor_regex_prefix_safe():
    ids = assign_state_ids(
        {"states": ["RESOLVE", "RESOLVE_ESCALATION"], "transitions": [],
         "initial": "RESOLVE", "terminal": ["RESOLVE_ESCALATION"],
         "state_details": []},
        "Only RESOLVE_ESCALATION is mentioned here.")
    assert ids is None  # RESOLVE itself never anchored


def test_assign_state_ids_mention_order():
    text = "Start at ALPHA. Then BETA, but GAMMA may interrupt BETA."
    ids = assign_state_ids(GRAPH_ABG, text)  # states ALPHA/BETA/GAMMA
    assert ids == {"ALPHA": "S1", "BETA": "S2", "GAMMA": "S3"}


def test_edge_ref_window_worked_example():
    result = check_edge_references(SOP_RENDERING_A, INSURANCE_GRAPH_NAME_KEYED)
    assert result.coverage == 1.0 and result.branch_ok


def test_edge_ref_window_gates():
    # branch edge whose target is only mentioned 3 paragraphs later -> fails
    text = ("### ALPHA\ngo on. \n\n### BETA\nnothing here.\n\n"
            "### GAMMA\nfrom ALPHA you may also reach GAMMA.")
    res = check_edge_references(text, GRAPH_WITH_BRANCH_ALPHA_TO_GAMMA)
    assert ("ALPHA", "GAMMA") in res.failed_edges and not res.branch_ok


def test_state_script_render_passes_verification():
    from llm_workflow_agents.data._workflow_script import build_workflow_script
    from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
    from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
    from llm_workflow_agents.data.generate_workflows import select_subgraph
    wf = select_subgraph(DOMAIN_REGISTRY["banking"], COMPLEXITY_SPECS[ComplexityLevel.L2],
                         random.Random(142))
    gd = wf.to_dict()
    text = build_workflow_script(gd)
    tools = {t for sd in gd["state_details"] for t in sd["tools"]}
    report = verify_rendering(text, gd, tools)
    assert report.accepted, report.corrections


def test_graph_to_eval_shape_permutes_with_mention_order():
    ids_a = {"VERIFY_POLICYHOLDER": "S1", "CLAIM_INTAKE": "S2", "ASSESS_COVERAGE": "S3",
             "RESOLVE": "S4", "REQUEST_DOCUMENTATION": "S5", "TERMINAL": "S6"}
    ids_b = dict(ids_a, RESOLVE="S5", REQUEST_DOCUMENTATION="S4")
    shape_a = graph_to_eval_shape(INSURANCE_GRAPH_NAME_KEYED, ids_a)
    shape_b = graph_to_eval_shape(INSURANCE_GRAPH_NAME_KEYED, ids_b)
    from llm_workflow_agents.eval.graph_extraction_eval import WorkflowGraph, check_structural_validity
    assert check_structural_validity(WorkflowGraph(**shape_a))
    assert check_structural_validity(WorkflowGraph(**shape_b))
    name_of = {n["id"]: n["name"] for n in shape_b["nodes"]}
    assert name_of["S4"] == "REQUEST_DOCUMENTATION" and name_of["S5"] == "RESOLVE"
```

- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement.** Core algorithms in full:

```python
def _anchor_re(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])")

def find_anchor_occurrences(text, state_names):
    occ = [(m.start(), name) for name in state_names for m in _anchor_re(name).finditer(text)]
    return sorted(occ)

def assign_state_ids(graph_dict, playbook):
    names = list(graph_dict["states"])
    first: dict[str, int] = {}
    for pos, name in find_anchor_occurrences(playbook, names):
        first.setdefault(name, pos)
    if len(first) != len(names):
        return None
    ordered = sorted(names, key=lambda n: first[n])
    return {name: f"S{i + 1}" for i, name in enumerate(ordered)}

_HEADING_RE = re.compile(r"^\s*#{1,6}\s|^\s*\*\*|^\s*\d+(?:\.\d+)*[.)]?\s")
_LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+(?:\.\d+)*[.)])\s")

def _segments(text: str) -> list[str]:
    segs: list[list[str]] = [[]]
    for line in text.splitlines():
        if not line.strip():
            segs.append([])
        elif _HEADING_RE.match(line) or _LIST_RE.match(line):
            segs.append([line])
        else:
            segs[-1].append(line)
    return ["\n".join(s) for s in segs if s]

def check_edge_references(playbook, graph_dict):
    names = list(graph_dict["states"])
    anchored = []
    for seg in _segments(playbook):
        present = {n for n in names if _anchor_re(n).search(seg)}
        if present:
            anchored.append(present)
    failed: list[tuple[str, str]] = []
    branch_ok = True
    edges = graph_dict["transitions"]
    for e in edges:
        src, dst = e["from"], e["to"]
        ok = any(src in anchored[i]
                 and (dst in anchored[i] or (i + 1 < len(anchored) and dst in anchored[i + 1]))
                 for i in range(len(anchored)))
        if not ok:
            failed.append((src, dst))
            if e.get("priority", 0) > 0:
                branch_ok = False
    coverage = 1.0 if not edges else (len(edges) - len(failed)) / len(edges)
    return EdgeRefResult(coverage, failed, branch_ok)
```

`graph_to_eval_shape`: nodes from `state_details` (id = `state_ids[name]`, keep tools/entry_actions), edges from `transitions` translating names→ids and `from`/`to` → `from_state`/`to_state`, `initial_state = state_ids[graph_dict["initial"]]`, `terminal_states = [state_ids[t] for t in graph_dict["terminal"]]`. Nodes emitted sorted by numeric id.

`verify_rendering` gate order: anchors (`assign_state_ids`; missing names → corrections `"missing state anchor: NAME"`) → tools (each tool name via `_anchor_re`-style boundary search but case-sensitive exact; corrections `"missing tool mention: NAME"`) → initial-first (`state_ids[graph_dict["initial"]] == "S1"`; correction `"initial state must be introduced before any other state"`) → edge windows (corrections `"edge reference missing: SRC -> DST"` for each failed edge when coverage < 0.90 or a branch edge failed). `accepted = not corrections`.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(data): Task C rendering verification (anchors, mention-order ids, edge windows)"`

---

### Task 5: `_playbook_verify.py` part 2 — cross-family back-extraction

**Files:**
- Modify: `src/llm_workflow_agents/data/_playbook_verify.py`
- Test: `tests/unit/test_playbook_verify.py` (append)

**Interfaces:**
- Consumes: `call_teacher_model`; `parse_graph_json`, `evaluate_graph_extraction`, `WorkflowGraph` from eval module.
- Produces:
  - `EXTRACTION_SYSTEM_PROMPT: str` — the spec's fixed ~120-token extraction system prompt, verbatim from `docs/data_generation_recipes_task_c.md` §System prompt. Single source of truth: the orchestrator imports it for `messages[0]`.
  - `pick_verifier(render_model: str, verify_teachers: dict[str, str]) -> str` — prefix→family (`gpt*`→"gpt", `gemini*`→"gemini", else ValueError), return `verify_teachers[<other family>]`.
  - `back_extract_check(playbook: str, gold_eval_dict: dict, verifier_model: str) -> dict` — returns `{"node_f1": float, "edge_f1": float, "passed": bool}`; gate node_f1 ≥ 0.90 ∧ edge_f1 ≥ 0.80; unparseable verifier output → `{"node_f1": 0.0, "edge_f1": 0.0, "passed": False}`.

- [ ] **Step 1: Write failing tests**

```python
def test_back_extraction_pass_and_prompt(monkeypatch):
    import llm_workflow_agents.data._playbook_verify as pv
    captured = {}
    monkeypatch.setattr(pv, "call_teacher_model",
                        lambda m, s, u: captured.update(sys=s, user=u) or json.dumps(GOLD_EVAL))
    res = pv.back_extract_check("the playbook text", GOLD_EVAL, "gemini-3-flash")
    assert res == {"node_f1": 1.0, "edge_f1": 1.0, "passed": True}
    assert captured["sys"] == pv.EXTRACTION_SYSTEM_PROMPT and captured["user"] == "the playbook text"


def test_back_extraction_fail_thresholds(monkeypatch):
    import llm_workflow_agents.data._playbook_verify as pv
    partial = json.loads(json.dumps(GOLD_EVAL))
    partial["nodes"] = partial["nodes"][:-1]          # 5 of 6 nodes -> node F1 ~0.909... wait
    partial["edges"] = partial["edges"][:3]           # 3 of 6 edges -> edge F1 < 0.80
    monkeypatch.setattr(pv, "call_teacher_model", lambda m, s, u: json.dumps(partial))
    assert not pv.back_extract_check("x", GOLD_EVAL, "gemini-3-flash")["passed"]


def test_back_extraction_unparseable(monkeypatch):
    import llm_workflow_agents.data._playbook_verify as pv
    monkeypatch.setattr(pv, "call_teacher_model", lambda m, s, u: "sorry, I cannot")
    assert pv.back_extract_check("x", GOLD_EVAL, "gpt-5.4-nano-2026-03-17")["passed"] is False


def test_pick_verifier_cross_family():
    import llm_workflow_agents.data._playbook_verify as pv
    vt = {"gpt": "gemini-3-flash", "gemini": "gpt-5.4-nano-2026-03-17"}
    assert pv.pick_verifier("gpt-5.4-mini-2026-03-17", vt) == "gemini-3-flash"
    assert pv.pick_verifier("gemini-3-flash-preview", vt) == "gpt-5.4-nano-2026-03-17"
    with pytest.raises(ValueError):
        pv.pick_verifier("claude-x", vt)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** `back_extract_check`: `raw = call_teacher_model(verifier_model, EXTRACTION_SYSTEM_PROMPT, playbook)`; `parsed, ok = parse_graph_json(raw)`; if not ok/None → fail-zero dict. Else compute node/edge F1 via `compute_node_f1`/`compute_edge_f1` against `WorkflowGraph(**gold_eval_dict)` (import them; avoids the full `evaluate_graph_extraction` list API for a single pair). Note in the fail-thresholds test: dropping 1 of 6 nodes gives node F1 = 10/11 ≈ 0.909 which passes the node gate — the test relies on the **edge** gate (3/6 edges → F1 ≈ 0.67 < 0.80); keep the assertion on `passed` only.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(data): Task C cross-family back-extraction gate"`

---

### Task 6: `generate_playbook_pairs.py` part 2 — per-rendering pipeline + row assembly

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_playbook_pairs.py`
- Create: `tests/unit/_task_c_helpers.py` (shared `CompliantTeacher` + `_patch_all_teachers`)
- Test: `tests/unit/test_playbook_pairs.py` (append)

**Interfaces:**
- Consumes: `render_playbook`, `draw_distractors`, `Register` (Task 3); `verify_rendering`, `assign_state_ids`, `graph_to_eval_shape`, `check_distractor_purity`, `back_extract_check`, `pick_verifier`, `EXTRACTION_SYSTEM_PROMPT` (Tasks 4–5).
- Produces: `produce_rendering(entry: GraphPoolEntry, plan: RenderingPlan, split: str | None, render_teacher: str, verify_teachers: dict[str, str], do_back_extraction: bool, rng: random.Random, max_repair_retries: int = 2) -> tuple[dict | None, str]` — returns `(row, "accepted")` or `(None, drop_reason)` where drop_reason ∈ `{"teacher_error", "verification", "back_extraction"}`.

Row assembly (all 15 spec fields): `pair_id, graph_id, source, domain, complexity_level, register, language, num_states, num_edges, distractor_count, paraphrase_density, condition_explicitness, verification, graph, messages` + `split` when not None. `verification = {"anchor_coverage": 1.0, "edge_ref_coverage": report.edge_ref_coverage, "back_extraction": None | {"node_f1", "edge_f1"}}`. `messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}, {"role": "user", "content": playbook}, {"role": "assistant", "content": json.dumps(eval_shape, separators=(",", ":"), ensure_ascii=True)}]`. `row["graph"] = eval_shape`.

**`CompliantTeacher` test helper** — create in `tests/unit/_task_c_helpers.py` (importable from any test file via `import _task_c_helpers` since pytest prepends the test dir to `sys.path`; reused by Tasks 7 and 12 alongside a `_patch_all_teachers(monkeypatch, teacher, echo_gold_on_verify)` helper that patches `call_teacher_model` in `_playbook_render`, `_playbook_verify`, and `_graph_invention`): parses the fenced `## GOLD WORKFLOW GRAPH` JSON out of the user prompt and emits `{"playbook": ...}` where the playbook introduces the initial state first, then every other state in graph order, one paragraph per state separated by blank lines, each paragraph naming the state, its tools, and every outgoing transition's target name; appends any `Insert the following paragraphs verbatim` distractor texts as their own paragraphs. Records `calls` (int) and `prompts` (list of user prompts). Options: `drop_anchor_on_first_call=<STATE>`, `always_drop_anchor=<STATE>`.

- [ ] **Step 1: Write failing tests**

```python
def test_row_schema_complete(monkeypatch):
    row, reason = _produce_one(monkeypatch, CompliantTeacher())  # helper wires a registry L2 entry
    assert reason == "accepted"
    expected_keys = {"pair_id", "graph_id", "source", "domain", "complexity_level", "register",
                     "language", "num_states", "num_edges", "distractor_count",
                     "paraphrase_density", "condition_explicitness", "verification", "graph",
                     "messages", "split"}
    assert set(row) == expected_keys
    assert json.loads(row["messages"][2]["content"]) == row["graph"]
    assert row["messages"][2]["content"].isascii()
    assert set(row["verification"]) == {"anchor_coverage", "edge_ref_coverage", "back_extraction"}


def test_repair_then_accept(monkeypatch):
    teacher = CompliantTeacher(drop_anchor_on_first_call="WORK")
    row, reason = _produce_one(monkeypatch, teacher)
    assert reason == "accepted" and teacher.calls == 2
    assert "missing state anchor: WORK" in teacher.prompts[1]


def test_drop_after_exhausted_repairs(monkeypatch):
    teacher = CompliantTeacher(always_drop_anchor="WORK")
    row, reason = _produce_one(monkeypatch, teacher)
    assert row is None and reason == "verification" and teacher.calls == 3


def test_distractor_purity_redraw_no_teacher_recall(monkeypatch):
    import llm_workflow_agents.data._playbook_render as pr
    # Poison the library: first en entry names a tool, forcing one redraw cycle.
    poisoned = ("Remember to log every do_thing call in the ledger.",) + pr.DISTRACTOR_LIBRARY["en"]
    monkeypatch.setattr(pr, "DISTRACTOR_LIBRARY", {**pr.DISTRACTOR_LIBRARY, "en": poisoned})
    teacher = CompliantTeacher()
    row, reason = _produce_one(monkeypatch, teacher, distractor_count=1)
    assert reason == "accepted" and teacher.calls == 1  # redraw is free
    assert "do_thing call in the ledger" not in row["messages"][1]["content"]


def test_ids_rederived_per_rendering(monkeypatch):
    row_a = _produce_with_order(monkeypatch, ["RESOLVE", "REQUEST_DOCUMENTATION"])
    row_b = _produce_with_order(monkeypatch, ["REQUEST_DOCUMENTATION", "RESOLVE"])
    name_a = {n["id"]: n["name"] for n in row_a["graph"]["nodes"]}
    name_b = {n["id"]: n["name"] for n in row_b["graph"]["nodes"]}
    assert name_a["S4"] == "RESOLVE" and name_b["S4"] == "REQUEST_DOCUMENTATION"


def test_back_extraction_failure_drops(monkeypatch):
    # verify-module teacher mocked to return an empty graph -> gate fails
    row, reason = _produce_one(monkeypatch, CompliantTeacher(), back_extraction=True,
                               verifier_response=json.dumps({"nodes": [], "edges": [],
                                                             "initial_state": "", "terminal_states": []}))
    assert row is None and reason == "back_extraction"
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `produce_rendering`.** Flow: draw distractors (`draw_distractors(plan.distractor_count, plan.language, rng, forbidden_terms=state_names + tool_names)`; `check_distractor_purity` redraw loop ≤5, no teacher cost) → render/verify/repair loop:

```python
corrections: list[str] | None = None
for attempt in range(max_repair_retries + 1):
    try:
        playbook = render_playbook(entry.graph, entry.tool_schemas, plan.register,
                                   plan.language, knobs, render_teacher, rng,
                                   corrections=corrections)
    except Exception:
        return None, "teacher_error"
    report = verify_rendering(playbook, entry.graph, tool_names)
    if report.accepted:
        break
    corrections = report.corrections
else:
    return None, "verification"
```

then optional back-extraction (`pick_verifier(render_teacher, verify_teachers)`; for STATE_SCRIPT renderings use `verify_teachers["gpt"]`), then eval-shape + row assembly. STATE_SCRIPT never enters the teacher repair path (deterministic; if verification fails that is a bug — raise).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(data): Task C per-rendering pipeline and row assembly"`

---

### Task 7: `generate_playbook_pairs.py` part 3 — dataset orchestration, stats, halt-leg, merge, export

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_playbook_pairs.py`, `src/llm_workflow_agents/data/__init__.py`
- Test: `tests/unit/test_playbook_pairs.py` (append)

**Interfaces:**
- Produces:
  - `@dataclass DatasetStats(output_dir: Path, output_files: list[Path], stats_file: Path, graphs_registry: int, graphs_invented: int, invention_dropped: int, graphs_dropped_lt3: int, renderings_attempted: int, renderings_accepted: int, dropped_by_reason: dict[str, int], repairs_used: int, back_extraction: dict, halted_legs: list[str], rows_by_register: dict, rows_by_language: dict, rows_by_split: dict)`
  - `generate_playbook_dataset(num_graphs=850, renderings_per_graph=(4, 6), invented_ratio=0.30, language_mix={"en": 0.5, "th": 0.3, "code_switch": 0.2}, render_teachers={"en": "gpt-5.4-mini-2026-03-17", "th": "gemini-3-flash-preview", "code_switch": "gpt-5.4-nano-2026-03-17"}, verify_teachers={"gpt": "gemini-3-flash", "gemini": "gpt-5.4-nano-2026-03-17"}, back_extraction_rate=0.10, seed=142, output_dir=Path("data/output/sft/task_c"), invention_teacher="gpt-5.4-mini-2026-03-17", benchmark_mode=False) -> DatasetStats`
  - `merge_benchmark_runs(input_paths: list[Path], output_path: Path) -> int` — union by `pair_id`, first-listed run wins; returns merged count.

Orchestration semantics:
- Renderings processed grouped by language leg. Per-leg back-extraction tracking: spot-check selection = `rng.random() < back_extraction_rate`; once a leg has ≥10 completed checks and failure ratio > 0.20, mark leg halted — remaining renderings for that leg are skipped (counted under `dropped_by_reason["leg_halted"]`), already-accepted rows kept.
- Graphs with <3 accepted renderings dropped entirely (`graphs_dropped_lt3`), their rows removed before writing.
- Output files: rows bucketed by `(source, language)`; filename `pairs_{source}_{lang}_{model_sanitized}_{timestamp}.jsonl` (dots→dashes in model, `datetime.now().strftime("%Y%m%d_%H%M%S")`). Stats JSON written to `output_dir / f"stats_{timestamp}.json"`.
- `benchmark_mode=True`: single teacher (the `render_teachers` dict collapsed — pass the same model for all legs from the shell script), all 6 registers per graph, no `split` field, filename `playbook_pairs_{model_sanitized}_{timestamp}.jsonl`, `num_graphs=25`, seed comes from caller (200).

- [ ] **Step 1: Write failing tests** — `test_generate_dataset_end_counts` (8 graphs, `CompliantTeacher` patched into `_playbook_render` + `_playbook_verify` + `_graph_invention`; assert `renderings_attempted == renderings_accepted + sum(dropped_by_reason.values())`, every row's `split` matches `assign_splits` output, files exist and parse); `test_seed_determinism` (two runs same seed → identical sorted row bytes, ignoring filenames: `sorted(json.dumps(r, sort_keys=True) for r in all_rows)`); `test_graph_dropped_below_three_renderings` (teacher noncompliant for one graph_id → absent from rows, counter == 1); `test_halt_leg_on_back_extraction` (`back_extraction_rate=1.0`, verifier fails only when the playbook is Thai → `"th" in stats.halted_legs`, no th rows beyond the ≥10 checked, en rows complete); `test_benchmark_mode` (4 graphs → 6 registers each, no `split` key); `test_merge_benchmark_runs_dedup` (two files, 3 shared pair_ids → union count, first wins); `test_public_export` (`from llm_workflow_agents.data import generate_playbook_dataset`).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**, add to `data/__init__.py`: `from llm_workflow_agents.data.generate_playbook_pairs import generate_playbook_dataset`.
- [ ] **Step 4: Run the whole new-module suite** — `python -m pytest tests/unit/test_playbook_pairs.py tests/unit/test_graph_invention.py tests/unit/test_playbook_render.py tests/unit/test_playbook_verify.py -q` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(data): Task C dataset orchestration, stats, benchmark mode, merge"`

---

### Task 8: `scripts/clean_task_c_pairs.py`

**Files:**
- Create: `scripts/clean_task_c_pairs.py`
- Test: `tests/unit/test_clean_task_c_pairs.py`

**Interfaces (stdlib-only; mirrors `clean_task_a_sft.py` CLI and structure):**
- `clean_record(record: dict) -> tuple[dict | None, str | None]` — pure. Drop reasons (string): `missing_field:<name>`; `bad_messages_shape` (must be exactly [system, user, assistant]); `assistant_not_json`; `assistant_graph_mismatch` (parsed assistant ≠ `record["graph"]`); `structural:<detail>` from `_structurally_valid(graph) -> str | None` (mirrors eval predicate: unique `^S\d+$` ids, edge endpoints ∈ ids, initial ∈ ids, terminals non-empty ⊆ ids, terminals reachable from initial via BFS); `bad_verification_shape`.
- `main()` — args `--input-dir` (required), `--output-dir` (required), `--dry-run`, `-q/--quiet`; sorted glob `*.jsonl`; cross-file dedupe on `record["messages"][1]["content"]` keep-first; per-reason drop counts printed after `"=" * 60`.

- [ ] **Step 1: Write failing tests** — import via `sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))`; `_base_record()` helper builds a valid worked-example row. Tests: `test_clean_record_valid_passthrough`, `test_drops_assistant_graph_mismatch`, `test_drops_unparseable_assistant`, `test_drops_structural_invalid` (edge to unknown id; terminal not reachable), `test_structural_mirror_matches_eval_predicate` (parametrize ~6 graphs: `(_structurally_valid(g) is None) == check_structural_validity(WorkflowGraph(**g))` — the test imports the package, the script does not), `test_dedupe_keeps_first_across_files` (write two tmp jsonl files with one shared playbook text, run `main()` via monkeypatched `sys.argv`, assert kept row is from the alphabetically-first file), `test_idempotent` (cleaning the cleaned dir drops nothing, output byte-identical).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** (shebang, `from __future__ import annotations`, argparse block copied in style from `clean_task_a_sft.py:117-142`).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(scripts): Task C pair cleaning with dedupe and structural mirror"`

---

### Task 9: `scripts/split_task_c_pairs.py`

**Files:**
- Create: `scripts/split_task_c_pairs.py`
- Test: `tests/unit/test_split_task_c_pairs.py`

**Interfaces (stdlib-only; Q4 semantics — never re-derives splits):**
- `partition_rows(rows: list[dict], group_key: str = "graph_id") -> dict[str, list[dict]]` — buckets by each row's recorded `split` field.
- `validate_partition(rows: list[dict], group_key: str, heldout_domains: list[str], heldout_registers: list[str]) -> list[str]` — violations: `missing split field: <pair_id>`, `group spans splits: <graph_id>`, `held-out domain outside test: <pair_id>`, `held-out register in train: <pair_id>`.
- `main()` — args `--input-dir`, `--output-dir`, `--group-key` (default `graph_id`), `--heldout-domains` (comma list, default `utilities,surveys`), `--heldout-registers` (comma list, default `manager_transcript`), `--seed` (default 142; recorded in the printed summary only — the docstring explains splits are generator-assigned), `--dry-run`, `--force`. Writes `{train,validation,test}.jsonl` with rows sorted by `pair_id`. Any violation → print all violations, `sys.exit(1)`, write nothing.

- [ ] **Step 1: Write failing tests** — `test_partition_by_recorded_split`, `test_group_integrity_violation_blocks_writes` (one graph_id spanning train+test → exit code 1 via `pytest.raises(SystemExit)`, output dir empty), `test_heldout_domain_and_register_violations` (utilities row labeled train; manager_transcript row labeled train → both violation strings present), `test_missing_split_field_errors`, `test_output_deterministic_bytes` (run `main()` twice → byte-identical files).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** (mirror `split_task_a_sft.py` structure: module constants, `_load_rows` sorted-glob, `_write_split` with `ensure_ascii=False`, idempotency guard behind `--force`).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(scripts): Task C group-integrity split with held-out validation"`

---

### Task 10: Shell runners

**Files:**
- Create: `scripts/generate_playbook_sft_data.sh`, `scripts/generate_playbook_benchmark_data.sh`

**Interfaces:**
- SFT runner flags: `--output-dir` (default `$PROJECT_ROOT/data/output`), `--seed` (142), `--num-graphs` (850), `--smoke-test` (sets `--num-graphs 6`), `--dry-run`, `--render-teacher-en/-th/-cs`, `--invention-teacher`. Checks **both** `OPENAI_API_KEY` and `GEMINI_API_KEY` unless `--dry-run` (legs span providers). Single `run python3 -c` invocation of `generate_playbook_dataset(...)`; afterwards parses the stats JSON (`python3 -c` + glob newest `stats_*.json`) and exits 1 with a message if `halted_legs` is non-empty.
- Benchmark runner flags: `--teacher` (required for generate mode), `--seed` (200), `--output-dir` (default `$PROJECT_ROOT/data/output/benchmark/task_c`), `--merge` (mode switch: globs `playbook_pairs_*.jsonl` excluding `*_merged.jsonl`, calls `merge_benchmark_runs`, writes `playbook_pairs_gemini-3_merged.jsonl`), `--dry-run`. Generate mode: `benchmark_mode=True, num_graphs=25`, all legs = `--teacher` model. Header comment documents the union-by-pair_id/first-run-wins merge decision.

Both scripts copy the house pattern verbatim: `set -euo pipefail`, SCRIPT_DIR/PROJECT_ROOT resolution, `.env` sourcing with `set -a`, arg `while/case` loop, provider-prefix→key `case` check (`generate_sft_data.sh:92-103`), `run()` wrapper (`:105-111`).

- [ ] **Step 1: Write both scripts.**
- [ ] **Step 2: Verify syntax** — `bash -n scripts/generate_playbook_sft_data.sh && bash -n scripts/generate_playbook_benchmark_data.sh`; run `shellcheck` on both if available.
- [ ] **Step 3: Verify dry-run output** — `bash scripts/generate_playbook_sft_data.sh --dry-run --smoke-test` prints a `[DRY RUN] python3 -c ...` line containing `seed=142` and `num_graphs=6`; `bash scripts/generate_playbook_benchmark_data.sh --teacher gemini-3-flash --dry-run` prints `seed=200`; with `GEMINI_API_KEY` unset and no `--dry-run`, the benchmark script exits 1 naming `GEMINI_API_KEY`.
- [ ] **Step 4: Cross-check flag spellings against the spec's §Full Generation Order** command lines (`--input-dir/--output-dir/--group-key/--heldout-domains/--heldout-registers/--seed` for the Python scripts; `--teacher` for benchmark).
- [ ] **Step 5: Commit** — `git commit -m "feat(scripts): Task C SFT and benchmark shell runners"`

---

### Task 11: DVC stages + doc updates + deprecation note

**Files:**
- Modify: `dvc.yaml` (replace placeholder at lines 168-170), `docs/data_generation_recipes_task_c.md` (status block), `src/llm_workflow_agents/data/generate_graph_pairs.py` (docstring)

- [ ] **Step 1: Replace the dvc.yaml placeholder** with:

```yaml
  task_c_pairs_generate:
    desc: >-
      ~5,000 playbook -> WorkflowGraph pairs (Task C SFT corpus), seed=142.
      Requires OPENAI_API_KEY and GEMINI_API_KEY.
    cmd: ./scripts/generate_playbook_sft_data.sh
    deps:
      - scripts/generate_playbook_sft_data.sh
      - src/llm_workflow_agents/data/generate_playbook_pairs.py
      - src/llm_workflow_agents/data/_graph_invention.py
      - src/llm_workflow_agents/data/_playbook_render.py
      - src/llm_workflow_agents/data/_playbook_verify.py
      - src/llm_workflow_agents/data/_workflow_script.py
      - src/llm_workflow_agents/data/domain_registry.py
      - src/llm_workflow_agents/eval/graph_extraction_eval.py
      - data/templates/graph_output_schema.json
    outs:
      - data/output/sft/task_c
  task_c_pairs_clean:
    desc: >-
      Re-validate, dedupe, and drop malformed Task C pairs.
    cmd: >-
      python scripts/clean_task_c_pairs.py
        --input-dir data/output/sft/task_c
        --output-dir data/output/sft/task_c_cleaned
        --quiet
    deps:
      - scripts/clean_task_c_pairs.py
      - data/output/sft/task_c
    outs:
      - data/output/sft/task_c_cleaned
  task_c_pairs_split:
    desc: >-
      Group-wise train/validation/test partition of Task C pairs by
      generator-recorded split field, with held-out domain/register checks.
    cmd: >-
      python scripts/split_task_c_pairs.py
        --input-dir data/output/sft/task_c_cleaned
        --output-dir data/output/sft/task_c_splits
        --group-key graph_id
        --heldout-domains utilities,surveys
        --heldout-registers manager_transcript
        --seed 142
        --force
    deps:
      - scripts/split_task_c_pairs.py
      - data/output/sft/task_c_cleaned
    outs:
      - data/output/sft/task_c_splits
```

- [ ] **Step 2: Verify** — `python3 -c "import yaml; yaml.safe_load(open('dvc.yaml'))"` and, if dvc is installed, `dvc status` parses without stage errors.
- [ ] **Step 3: Update the recipe doc status block** — replace the `> **Status: recipe only.**` paragraph with an implemented-status note listing the landed entry points (`generate_playbook_dataset`, the two shell runners, clean/split scripts, the three DVC stages) and stating `generate_graph_pairs.py` is now formally superseded (removal tracked separately). Sweep the doc for "intended interfaces"/"none are implemented yet" phrasing and update.
- [ ] **Step 4: Add deprecation pointer** to `generate_graph_pairs.py` module docstring: "Superseded by generate_playbook_pairs.py (see docs/data_generation_recipes_task_c.md); kept for the legacy Task A-derived path."
- [ ] **Step 5: Commit** — `git commit -m "chore: Task C DVC stages, doc status, deprecation pointer"`

---

### Task 12: Integration smoke test

**Files:**
- Create: `tests/unit/test_task_c_integration.py`

**Interfaces:**
- Consumes everything; `CompliantTeacher` / `_patch_all_teachers` imported from `tests/unit/_task_c_helpers.py` (created in Task 6).

- [ ] **Step 1: Write the end-to-end test**

```python
# tests/unit/test_task_c_integration.py — header
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from clean_task_c_pairs import main as clean_main            # noqa: E402
from split_task_c_pairs import main as split_main            # noqa: E402
from llm_workflow_agents.data import generate_playbook_dataset  # noqa: E402
from _task_c_helpers import CompliantTeacher, _patch_all_teachers  # noqa: E402


def test_pipeline_end_to_end(monkeypatch, tmp_path):
    _patch_all_teachers(monkeypatch, CompliantTeacher(), echo_gold_on_verify=True)
    stats = generate_playbook_dataset(num_graphs=4, invented_ratio=0.5,
                                      back_extraction_rate=1.0, seed=142,
                                      output_dir=tmp_path / "raw")
    assert stats.renderings_accepted > 0 and not stats.halted_legs
    # clean
    sys.argv = ["clean_task_c_pairs.py", "--input-dir", str(tmp_path / "raw"),
                "--output-dir", str(tmp_path / "cleaned"), "--quiet"]
    clean_main()
    # split
    sys.argv = ["split_task_c_pairs.py", "--input-dir", str(tmp_path / "cleaned"),
                "--output-dir", str(tmp_path / "splits"), "--force"]
    split_main()
    for name in ("train", "validation", "test"):
        assert (tmp_path / "splits" / f"{name}.jsonl").exists()
    train_rows = _read_jsonl(tmp_path / "splits" / "train.jsonl")
    assert all(r["register"] != "manager_transcript" for r in train_rows)


def test_gold_scores_perfect_under_eval(monkeypatch, tmp_path):
    from llm_workflow_agents.eval.graph_extraction_eval import (
        WorkflowGraph, check_mermaid_renderability, check_structural_validity,
        compute_edge_f1, compute_node_f1, parse_graph_json,
    )
    _patch_all_teachers(monkeypatch, CompliantTeacher(), echo_gold_on_verify=True)
    generate_playbook_dataset(num_graphs=4, invented_ratio=0.5, back_extraction_rate=0.0,
                              seed=142, output_dir=tmp_path)
    rows = [json.loads(line) for f in sorted(tmp_path.glob("pairs_*.jsonl"))
            for line in f.read_text().splitlines()]
    assert rows
    for row in rows:
        parsed, ok = parse_graph_json(row["messages"][2]["content"])
        assert ok and parsed is not None
        gold = WorkflowGraph(**row["graph"])
        assert check_structural_validity(gold) and check_mermaid_renderability(gold)
        assert compute_node_f1(parsed, gold) == 1.0 and compute_edge_f1(parsed, gold) == 1.0


def test_pipeline_determinism(monkeypatch, tmp_path):
    outputs = []
    for run in ("a", "b"):
        _patch_all_teachers(monkeypatch, CompliantTeacher(), echo_gold_on_verify=True)
        raw = tmp_path / run / "raw"
        generate_playbook_dataset(num_graphs=4, invented_ratio=0.5, back_extraction_rate=1.0,
                                  seed=142, output_dir=raw)
        sys.argv = ["clean_task_c_pairs.py", "--input-dir", str(raw),
                    "--output-dir", str(tmp_path / run / "cleaned"), "--quiet"]
        clean_main()
        sys.argv = ["split_task_c_pairs.py", "--input-dir", str(tmp_path / run / "cleaned"),
                    "--output-dir", str(tmp_path / run / "splits"), "--force"]
        split_main()
        outputs.append({name: (tmp_path / run / "splits" / f"{name}.jsonl").read_bytes()
                        for name in ("train", "validation", "test")})
    assert outputs[0] == outputs[1]
```

- [ ] **Step 2: Run the integration file** — `python -m pytest tests/unit/test_task_c_integration.py -q` → pass.
- [ ] **Step 3: Run the full suite for regressions** — `python -m pytest tests/unit -q` → no failures anywhere (notably `test_graph_extraction.py`, `test_data_generation.py`, `test_clean_task_a_sft.py` untouched).
- [ ] **Step 4: Commit** — `git commit -m "test: Task C end-to-end integration smoke"`

---

## Verification (whole feature)

1. `source .venv/bin/activate && python -m pytest tests/unit -q` — full suite green.
2. `bash scripts/generate_playbook_sft_data.sh --dry-run --smoke-test` and `bash scripts/generate_playbook_benchmark_data.sh --teacher gemini-3-flash --dry-run` — correct printed invocations, correct missing-key errors.
3. `python3 -c "import yaml; yaml.safe_load(open('dvc.yaml'))"` — parses.
4. Optional live smoke (needs real API keys, user-triggered): `bash scripts/generate_playbook_sft_data.sh --smoke-test` (6 graphs ≈ 25 teacher calls), then clean + split on the output, then eyeball 2–3 rows against the spec's Output Data Format.
5. After approval + merge decision, save this plan to `docs/superpowers/plans/2026-07-02-task-c-playbook-pairs.md` (user convention) and commit it.

## Execution notes

- Branch `feat/task-c-playbook-pairs` off main first (verify with `git branch --show-current` before every commit — main is unprotected in practice).
- Tasks 1→2→{3,4}→5→6→7→{8,9}→10→11→12; 3∥4 and 8∥9 can be parallelized by subagents.
- The only file shared across tasks is `generate_playbook_pairs.py` (Tasks 2, 6, 7 — strictly sequential layers).
