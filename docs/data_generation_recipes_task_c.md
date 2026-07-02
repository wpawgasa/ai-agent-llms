# Data Generation Recipe — Task C: Playbook → Workflow Graph

This document is the sibling of [`data_generation_recipes.md`](data_generation_recipes.md) (Task A). It specifies the data generation recipe for training a small LLM (4B–12B) to convert a **playbook described in natural language** into a structured **WorkflowGraph JSON** — the schema in `data/templates/graph_output_schema.json`, scored by `src/llm_workflow_agents/eval/graph_extraction_eval.py`.

> **Status: recipe only.** The generator scripts named below are *intended interfaces* — none are implemented yet. The existing `src/llm_workflow_agents/data/generate_graph_pairs.py` is **superseded** by this recipe: its NL side is a reused Task A system message (weak extraction signal), its augmentation is trivial prefixing, and it has never been materialized to `data/output/`. It stays in-tree until `generate_playbook_pairs.py` lands.

| Split | Purpose | Generator (intended) | Approx. size |
|-------|---------|----------------------|--------------|
| Benchmark | Rank pre-trained 4B–12B candidates before fine-tuning | `generate_playbook_benchmark_data.sh` (Gemini teachers) | **~150 pairs** (25 graphs × 6 registers) |
| SFT | Fine-tuning; supplies train / validation / test splits | `generate_playbook_sft_data.sh` (GPT + Gemini legs) | **~5,000 pairs** (~850 graphs × 4–6 registers) |

As with Task A, there is no separate GRPO dataset: if Task C gets a GRPO stage, it reuses the SFT train split with the verifiable `reward_graph_extraction` reward (node F1 / edge F1 / GED are computable online).

## Task Definition

**Input:** a playbook — a natural-language document describing how an agent should handle a workflow (an SOP, a wiki page, training notes, a manager's verbal explanation…).
**Output:** a single JSON object conforming to `graph_output_schema.json`:

```json
{
  "nodes": [{"id": "S1", "name": "VERIFY_POLICYHOLDER", "tools": ["verify_policy"], "entry_actions": []}, ...],
  "edges": [{"from_state": "S1", "to_state": "S2", "condition": "policyholder verified", "priority": 0}, ...],
  "initial_state": "S1",
  "terminal_states": ["S6"]
}
```

Evaluation reuses `graph_extraction_eval.py` unchanged: node F1 ≥ 85% (matched by `id`), edge F1 ≥ 75% (matched by `(from_state, to_state)` ID pair), normalized GED ≤ 0.20 (node labels = `name`), JSON validity ≥ 95%, structural validity ≥ 90%, Mermaid renderability ≥ 90%.

---

## Shared Concepts

### Generation Direction: Graph-First Inverse Rendering

Every pair is generated **graph-first**: draw a validated gold graph, then have a teacher *render* it as a playbook in a given register and language. This is the proven direction from Task A's rich-prompt path (`_RICH_PROMPT_SYSTEM` in `generate_workflows.py`) and it makes ground truth exact by construction — the graph is never inferred from prose, so there is no label noise to audit. The prose is what gets verified (see [Faithfulness Verification](#faithfulness-verification)).

### Gold Graph Sourcing — Hybrid 70/30

**~70% registry-derived (~595 graphs).** Reuse Task A's `_select_domain` + `ComplexitySpec` L1–L5 subgraph sampler (`generate_workflows.py`) over the 18-domain registry. Level mix is shifted toward the middle relative to Task A — tiny L1 graphs saturate extraction quickly:

| Level | Share | States |
|-------|-------|--------|
| L1 | 10% | 3–4 |
| L2 | 25% | 5–7 |
| L3 | 35% | 8–12 |
| L4 | 20% | 12–16 |
| L5 | 10% | 16–20 |

These graphs are free (no teacher call), deterministic, and pre-validated by `validate_domain`.

**~30% teacher-invented (~255 graphs)** across ~25 short **domain briefs** deliberately *outside* the registry — e.g. HR onboarding, IT incident response, loan underwriting, restaurant reservations, warehouse returns, clinical-trial screening, freight dispatch, campus admissions. The teacher receives a brief plus size constraints (4–14 states) and returns graph JSON only, no prose. Without this slice the model would memorize 18 vocabularies of canonical state names instead of learning extraction.

**Novel-graph validation (reject → repair → drop):**
1. `jsonschema` validation against `data/templates/graph_output_schema.json`.
2. The eval module's own predicates, imported verbatim: `check_structural_validity` + `check_mermaid_renderability`. Validating gold with the exact eval predicates guarantees gold can never score 0 on structural metrics.
3. Registry-style invariants ported from `validate_domain`: unique state names, no edge self-loops, every non-terminal state has ≥1 outgoing edge, unique priorities per source state.

Violations are itemized and fed back for up to 2 repair attempts (the same `repair_feedback` pattern as Task A's teacher conversation loop); still-invalid graphs are dropped.

### Playbook Registers

Each gold graph is rendered 4–6 times across six registers. Multi-register coverage is the core value of this dataset over the old Task C generator — real businesses write playbooks in wildly different shapes.

| Key | Register | Style |
|-----|----------|-------|
| `sop_document` | Formal SOP | Numbered sections, "the representative shall…", compliance tone |
| `prose_narrative` | Prose paragraphs | "When a customer calls, first verify their identity. If they…" |
| `bullet_quick_reference` | Bullet quick reference | Terse bullets, arrows, fragments |
| `wiki_training_notes` | Informal wiki / training notes | Conversational, emoji, asides, team shorthand |
| `state_script` | Rich `### [state]` prompt | The existing Task A workflow-script style |
| `manager_transcript` | Manager-explaining transcript | Spoken register, digressions, "so basically what you do is…" |

`state_script` is rendered **programmatically** by `build_workflow_script` (`data/_workflow_script.py`) — zero teacher calls, and it anchors the dataset to the exact format Task A system prompts use. The other five are teacher-rendered.

**Register-per-graph assignment:** each graph draws 4–6 registers without replacement, always including at least one of {`sop_document`, `prose_narrative`} and `state_script`. `manager_transcript` is only drawn for graphs pre-assigned to validation/test (see [Splits](#trainvaltest-splits)).

### Naming & ID Contract

This is the scoreability keystone. `compute_node_f1` matches nodes by `id` and `compute_edge_f1` matches `(from_state, to_state)` ID pairs, so **both names and IDs must be deterministic from the playbook text alone**:

1. **Anchor rule.** Every rendering must contain each state's canonical `SCREAMING_SNAKE` name verbatim (ASCII) at least once — as a heading (`### [VERIFY_IDENTITY]`), an inline parenthetical ("check the customer's ID *(VERIFY_IDENTITY)*"), or a quoted label ("we call this step VERIFY_IDENTITY"). All surrounding prose paraphrases freely and inconsistently — mapping messy prose onto the anchored canonical names **is the skill being trained**.
2. **Output `name` = anchored canonical name**, never the paraphrase.
3. **ID assignment = order of first anchor mention.** `S1` is the first-mentioned state, and the rendering contract requires playbooks to introduce the starting step before any other, so `S1` is always the initial state. Gold JSON is re-derived **per rendering** from that rendering's mention order — two registers of the same graph may permute IDs, which is correct: each pair is (*this text*, *this graph*).
4. **Terminals must be explicitly signposted** ("the call ends here", "wrap up and close the ticket").

Unanchored/implicit states (behavior described without a nameable state) are **deferred to a v2 hard mode**: under ID/name-based matching an unnamed state has no unique scoreable answer, so including them would poison the metrics rather than strengthen the model.

### Difficulty Knobs

Recorded per sample as metadata so evaluation can be difficulty-stratified:

| Knob | Values | Distribution |
|------|--------|--------------|
| `distractor_count` | 0–3 irrelevant policy paragraphs (compliance boilerplate, tone guidance, SLA text) | 30% of renderings get ≥1 |
| `paraphrase_density` | `low` (anchor used throughout) / `medium` (anchor once, paraphrases after) / `high` (anchor once, aggressive paraphrase + pronoun reference) | ~40 / 40 / 20 |
| `condition_explicitness` | `explicit` ("if the ID check fails, go to ESCALATE") / `narrative_order` (priority-0 spine implied by sequencing) / `listing_order` ("first try…, otherwise…, as a last resort…" → model infers `priority` 0/1/2 from list position) | ~40 / 35 / 25 |

Distractor paragraphs are verified to contain **no state anchors or tool names** (purity check) so they never create ambiguity about gold.

### Languages

| Language | Share | Notes |
|----------|-------|-------|
| `en` | ~50% | |
| `th` | ~30% | Anchors and tool names stay ASCII (same convention as Task A annotations) |
| `code_switch` | ~20% | Thai structure with embedded English terms — realistic for Thai internal wiki/SOP documents |

The output graph JSON is **always English/ASCII** regardless of playbook language.

### Faithfulness Verification

Each teacher rendering passes a layered gate, cheapest checks first. Because anchors are verbatim ASCII strings, the first four checks are exact string matching — deterministic, free, and language-independent.

| Check | Method | On failure |
|-------|--------|------------|
| State anchor coverage | Every canonical state name appears verbatim ≥1× | Repair ≤2× (itemized missing anchors fed back) |
| Tool coverage | Every tool granted to a state appears verbatim | Repair ≤2× |
| Edge reference coverage | For each edge, the target state's anchor appears within the source state's section/paragraph window; ≥90% of edges must pass, **100% of branch (priority > 0) edges** | Repair ≤2× |
| Distractor purity | Injected distractor paragraphs contain no anchors / tool names | Re-draw distractor text (no teacher call) |
| Back-extraction spot check | **10% of accepted renderings**: a *different-family* teacher extracts a graph from the playbook, scored with `evaluate_graph_extraction`; gate node F1 ≥ 0.90 ∧ edge F1 ≥ 0.80 | **Drop** — a playbook a strong teacher can't decode is ambiguous, not repairable. If >20% of a leg's spot checks fail, halt the leg and inspect the rendering prompt |

Renderings that exhaust repairs are dropped; the graph keeps its remaining renderings (a graph is only discarded if it ends with <3 renderings).

### Seed Allocation

Disjoint from Task A's seeds (42 SFT / 100 benchmark) so no gold graph subgraph draw collides:

| Dataset | Seed |
|---------|------|
| Benchmark | 200 |
| SFT | 142 |

### Teacher Models

| Stage | Leg | Teacher | ~Calls |
|-------|-----|---------|--------|
| Graph invention (+ repairs) | en only (graphs are ASCII) | `gpt-5.4-mini-2026-03-17` | ~400 |
| Playbook rendering | en (~50%) | `gpt-5.4-mini-2026-03-17` | ~2,400 incl. repairs |
| Playbook rendering | th (~30%) | `gemini-3-flash-preview` | ~1,450 |
| Playbook rendering | code_switch (~20%) | `gpt-5.4-nano-2026-03-17` | ~950 |
| Back-extraction verification (10%) | en / code_switch | `gemini-3-flash` (cross-family vs GPT renderer) | ~350 |
| Back-extraction verification (10%) | th | `gpt-5.4-nano-2026-03-17` (cross-family vs Gemini renderer) | ~150 |
| Benchmark rendering | mixed | `gemini-3-flash` + `gemini-3.1-flash-lite` | ~180 |

The `state_script` register (~1/6 of renderings) costs zero teacher calls. Total ≈ 5,900 calls at ~2k input / ~1.5k output tokens each ≈ 20M tokens — mini/flash-class pricing puts the full corpus in the tens of dollars, an order of magnitude below the Task A SFT run. Cross-family back-extraction (Gemini checks GPT-rendered playbooks and vice versa) avoids correlated blind spots.

---

## Output Data Format

One JSONL row per (rendering, graph) pair:

| Field | Type | Description |
|-------|------|-------------|
| `pair_id` | str | `"{graph_id}_r{k}"`, e.g. `"G0142_r3"` |
| `graph_id` | str | Stable gold-graph id, e.g. `"G0142"` — the **group key for splitting** |
| `source` | str | `"registry"` or `"invented"` |
| `domain` | str | Registry key or invented brief slug (e.g. `"it_incident_response"`) |
| `complexity_level` | str | `"L1"`–`"L5"` for registry graphs; `"NA"` for invented |
| `register` | str | One of the 6 register keys |
| `language` | str | `"en"`, `"th"`, or `"code_switch"` |
| `num_states`, `num_edges` | int | Convenience counters |
| `distractor_count` | int | 0–3 |
| `paraphrase_density` | str | `low` / `medium` / `high` |
| `condition_explicitness` | str | `explicit` / `narrative_order` / `listing_order` |
| `verification` | dict | `{"anchor_coverage": 1.0, "edge_ref_coverage": 0.91, "back_extraction": null \| {"node_f1": …, "edge_f1": …}}` |
| `graph` | dict | The gold WorkflowGraph, parsed (duplicates the assistant content so eval/cleaning never re-parse the string — same convention as the old `generate_graph_pairs.py`) |
| `messages` | list | The training conversation — the only field SFT consumes |

### `messages`

```json
[
  {"role": "system",    "content": "<fixed extraction system prompt, below>"},
  {"role": "user",      "content": "<the playbook text>"},
  {"role": "assistant", "content": "<compact ASCII json.dumps(graph)>"}
]
```

### System prompt (fixed, English, ~120 tokens)

> You are a workflow-graph extraction engine. Read the playbook and output ONLY a JSON object with keys `nodes`, `edges`, `initial_state`, `terminal_states`. Nodes: `{"id": "S<n>", "name": "<SCREAMING_SNAKE state name as written in the playbook>", "tools": [...], "entry_actions": [...]}`. Number states in order of first mention; S1 is the starting state. Edges: `{"from_state", "to_state", "condition", "priority"}` — priority 0 for the default path, 1+ for alternatives in the order the playbook lists them. Ignore content that does not describe workflow states, transitions, or tools. Output English/ASCII JSON regardless of the playbook language. No markdown fences.

The prompt is deliberately schema-complete: at 4B–12B scale the model should never have to guess the output contract, and a fixed prompt keeps the benchmark comparable across candidates. Constrained decoding (Outlines/XGrammar, already integrated for Task C eval) remains available at inference but is **not** assumed during data generation or SFT.

---

## Recipes

### Benchmark (`generate_playbook_benchmark_data.sh`)

Ranks pre-trained 4B–12B candidates before any fine-tuning (the Task C analogue of Task A's Phase 1 split).

```
Graphs:       25 (disjoint from the SFT pool by seed; ~17 registry L1–L5 + ~8 invented)
Registers:    all 6 per graph
Total:        ~150 pairs
Language:     mixed ~50/25/25 en/th/code_switch
Teachers:     gemini-3-flash + gemini-3.1-flash-lite (two runs, merged)
Seed:         200
Output:       data/output/benchmark/task_c/playbook_pairs_gemini-3_merged.jsonl
```

Same rationale as the Task A benchmark: small volume, cheap, fast, reliable JSON emitters — and it keeps the GPT legs' stylistic fingerprint out of the pre-training ranking data.

```bash
GEMINI_API_KEY=... ./scripts/generate_playbook_benchmark_data.sh --teacher gemini-3-flash
GEMINI_API_KEY=... ./scripts/generate_playbook_benchmark_data.sh --teacher gemini-3.1-flash-lite
# then merge the two runs into playbook_pairs_gemini-3_merged.jsonl
```

### SFT (`generate_playbook_sft_data.sh`)

```
Graphs:       ~850 (≈595 registry / ≈255 invented)
Renderings:   4–6 per graph → ~5,000 pairs
Languages:    en ~50% / th ~30% / code_switch ~20%
Teachers:     three legs (see Teacher Models table)
Seed:         142
Output:       data/output/sft/task_c/          (raw)
              data/output/sft/task_c_cleaned/  (after clean_task_c_pairs.py)
```

Intended entry point:

```python
def generate_playbook_dataset(
    num_graphs: int = 850,
    renderings_per_graph: tuple[int, int] = (4, 6),
    invented_ratio: float = 0.30,
    language_mix: dict[str, float] = {"en": 0.5, "th": 0.3, "code_switch": 0.2},
    render_teachers: dict[str, str] = ...,   # language leg → teacher model
    verify_teachers: dict[str, str] = ...,   # renderer family → cross-family verifier
    back_extraction_rate: float = 0.10,
    seed: int = 142,
    output_dir: Path = Path("data/output/sft/task_c"),
) -> DatasetStats
```

in `src/llm_workflow_agents/data/generate_playbook_pairs.py`, with helpers `invent_novel_graphs(...)`, `render_playbook(graph, register, language, knobs, teacher) -> str`, `verify_rendering(playbook, graph) -> VerificationReport`, `assign_state_ids(graph, playbook) -> dict` (mention-order S-numbering), and `validate_gold_graph(graph)` (imports the eval-module predicates). The shell runner mirrors `generate_sft_data.sh`: three language legs, per-leg API-key checks, `--dry-run` support.

**Cleanup** (`scripts/clean_task_c_pairs.py`): re-validates every row (schema + structural predicates), dedupes identical playbook texts, drops rows whose assistant JSON fails to round-trip, and reports drop stats. Idempotent, mirrors `clean_task_a_sft.py`.

### Train/Val/Test Splits

Produced by `scripts/split_task_c_pairs.py` (85/10/5, seed 142) into `data/output/sft/task_c_splits/{train,validation,test}.jsonl`. Three hygiene rules, in priority order:

1. **Group split by `graph_id`.** All renderings of one graph land in the same split — never row-level. With 4–6 near-duplicate rows sharing identical labels, row-level splitting would leak essentially every test label into train.
2. **Held-out domain axis.** Two registry domains — `utilities` and `surveys` (mid-size, not needed for L4/L5 eligibility) — plus ~20% of invented briefs appear **only in test**. This gives a domain-OOD read-out at zero extra generation cost.
3. **Held-out register axis.** `manager_transcript` renderings exist **only for graphs pre-assigned to validation/test** (assignment is drawn deterministically from `graph_id` before rendering, so no train graph ever has a transcript rendering to leak). This gives a register-OOD read-out.

Split construction order: the test pool is assembled first (held-out domains + held-out briefs + a random in-domain slice up to ~5% of graphs), then validation (~10%), then the remainder is train. Expected volumes: ~4,250 train / ~500 validation / ~250 test pairs.

> **Important:** as with Task A, `test.jsonl` must not touch training or hyperparameter selection — final evaluation only. Domain-OOD and register-OOD subsets are reported separately from the in-distribution test score.

DVC stages: `task_c_pairs_generate`, `task_c_pairs_clean`, `task_c_pairs_split` (fills the "Task B and Task C stages will be added once recipes are defined" placeholder at `dvc.yaml:169`).

---

## Worked Example

**Gold graph** — registry `insurance` L2 subgraph, 6 states. Spine `VERIFY_POLICYHOLDER → CLAIM_INTAKE → ASSESS_COVERAGE → RESOLVE → TERMINAL`; branch `ASSESS_COVERAGE → REQUEST_DOCUMENTATION` (condition "documentation missing", priority 1) with loop-back `REQUEST_DOCUMENTATION → ASSESS_COVERAGE`; tools `verify_policy` on VERIFY_POLICYHOLDER, `file_claim` on CLAIM_INTAKE.

**Rendering A — `sop_document`, en, `condition_explicitness=explicit`:**

> **SOP-CL-07: Claims Intake Procedure.**
> 1.0 **VERIFY_POLICYHOLDER.** The representative shall confirm the caller's identity using the `verify_policy` tool before any disclosure is made. Upon successful verification, proceed to section 2.0.
> 2.0 **CLAIM_INTAKE.** Record the incident details and file the claim with `file_claim`. Continue to 3.0.
> 3.0 **ASSESS_COVERAGE.** Determine whether the incident falls within policy coverage. When coverage is confirmed, continue to 4.0 **RESOLVE**. If documentation is missing, transfer the case to 3.1 **REQUEST_DOCUMENTATION** and return to 3.0 upon receipt.
> …the case is then closed (**TERMINAL**).

**Rendering B — `wiki_training_notes`, code_switch, `condition_explicitness=listing_order`, 1 distractor:**

> ทีมเคลม notes 📋 — เริ่มจากเช็คตัวตนลูกค้าก่อนเสมอ (ขั้น VERIFY_POLICYHOLDER, ใช้ verify_policy) แล้วค่อยรับเรื่อง CLAIM_INTAKE ด้วย file_claim… พอถึง ASSESS_COVERAGE: ปกติก็ไป RESOLVE เลย แต่ถ้าเอกสารไม่ครบให้เด้งไป REQUEST_DOCUMENTATION ก่อนแล้วค่อยวนกลับมาเช็คใหม่ จบเคสที่ TERMINAL
> *(อย่าลืม: ห้ามรับปากเรื่องยอดเงินก่อน approve — policy ของฝ่าย legal)* ← distractor paragraph: no anchors, no tool names.

**Gold assistant JSON** (topology identical for both; IDs follow each rendering's mention order — identical here since both introduce RESOLVE before REQUEST_DOCUMENTATION; had Rendering A introduced them in the opposite order, its gold would swap S4/S5 while B's stayed as shown):

```json
{
  "nodes": [
    {"id": "S1", "name": "VERIFY_POLICYHOLDER", "tools": ["verify_policy"], "entry_actions": []},
    {"id": "S2", "name": "CLAIM_INTAKE",        "tools": ["file_claim"],    "entry_actions": []},
    {"id": "S3", "name": "ASSESS_COVERAGE",     "tools": [],                "entry_actions": []},
    {"id": "S4", "name": "RESOLVE",             "tools": [],                "entry_actions": []},
    {"id": "S5", "name": "REQUEST_DOCUMENTATION","tools": [],               "entry_actions": []},
    {"id": "S6", "name": "TERMINAL",            "tools": [],                "entry_actions": []}
  ],
  "edges": [
    {"from_state": "S1", "to_state": "S2", "condition": "policyholder verified",   "priority": 0},
    {"from_state": "S2", "to_state": "S3", "condition": "claim filed",             "priority": 0},
    {"from_state": "S3", "to_state": "S4", "condition": "coverage assessed",       "priority": 0},
    {"from_state": "S3", "to_state": "S5", "condition": "documentation missing",   "priority": 1},
    {"from_state": "S5", "to_state": "S3", "condition": "documents received",      "priority": 0},
    {"from_state": "S4", "to_state": "S6", "condition": "case closed",             "priority": 0}
  ],
  "initial_state": "S1",
  "terminal_states": ["S6"]
}
```

Note how rendering B never uses the phrase "documentation missing" — it says "ถ้าเอกสารไม่ครบ" — yet the gold edge condition is the canonical English string. That is fine because `condition` text is never string-matched by the eval: edge F1 compares only `(from_state, to_state)` ID pairs, and GED matches node labels. Scoring depends solely on anchored names and mention-order IDs, which the anchor rule makes deterministic.

---

## Full Generation Order

```bash
# 1. Benchmark — Gemini teachers (two runs, merged)
GEMINI_API_KEY=... ./scripts/generate_playbook_benchmark_data.sh --teacher gemini-3-flash
GEMINI_API_KEY=... ./scripts/generate_playbook_benchmark_data.sh --teacher gemini-3.1-flash-lite
# (merge into data/output/benchmark/task_c/playbook_pairs_gemini-3_merged.jsonl)

# 2. SFT pairs (graph sourcing + rendering + verification, three language legs)
OPENAI_API_KEY=sk-... GEMINI_API_KEY=... ./scripts/generate_playbook_sft_data.sh

# 3. Cleanup (idempotent)
python scripts/clean_task_c_pairs.py \
    --input-dir data/output/sft/task_c \
    --output-dir data/output/sft/task_c_cleaned

# 4. Group-wise 85/10/5 split with held-out axes (seed=142)
python scripts/split_task_c_pairs.py \
    --input-dir data/output/sft/task_c_cleaned \
    --output-dir data/output/sft/task_c_splits \
    --group-key graph_id \
    --heldout-domains utilities,surveys \
    --heldout-registers manager_transcript \
    --seed 142
```

**Total samples:** ~5,150 (benchmark ~150 + SFT ~5,000 before cleanup losses). Evaluation reuses `eval/graph_extraction_eval.py` via the existing `scripts/run_exp_c.sh` harness.

---

## Output File Naming

SFT legs write one JSONL per (source, language, teacher, timestamp):

```
pairs_{source}_{lang}_{model}_{timestamp}.jsonl
```

Examples:

```
pairs_registry_en_gpt-5-4-mini-2026-03-17_20260702_101500.jsonl
pairs_invented_th_gemini-3-flash-preview_20260702_102200.jsonl
pairs_registry_code_switch_gpt-5-4-nano-2026-03-17_20260702_103100.jsonl
```

Benchmark uses the compact merged-artifact name `playbook_pairs_gemini-3_merged.jsonl` (mirroring Task A's `l{level}_mixed_gemini-3_merged.jsonl` convention; no per-level files since register, not level, is the primary stratum here). Register, difficulty knobs, and split-relevant metadata live in each row, not in filenames.
