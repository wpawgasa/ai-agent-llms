# Fix teacher-model data generation: provenance, redraw budget, coherence checks

## Context

An audit of generated Task A benchmark JSONL files (L1/L2/L3, 50 samples each) found three defect classes in `src/llm_workflow_agents/data/generate_workflows.py`:

1. **Silent placeholder fallbacks** — at L3, 21/50 samples were deterministic placeholder skeletons (template text, `{"placeholder": "value"}` tool args), emitted after `max_sample_attempts=3` with no provenance marker and no persisted stats. They poison quality metrics (tool-argument exact-match is unattainable on them).
2. **Bug A — ground-truth corruption (L3_043)**: `_backfill_annotations` (generate_workflows.py:614) fills only *missing* annotation fields, preserving teacher-supplied `annotations` even when they contradict the inline `[STATE: X → Y]` marker in the same message. `_extract_ground_truth` (:659) prefers annotations, so ground truth becomes discontinuous/wrong.
3. **Bug B — incoherent teacher output passes repair (L2_029, L3_026)**: the repair loop's `_has_violations` (:1481) checks only tool placement and per-edge legality. It misses: mid-message `[STATE:` markers, discontinuous annotation sequences, first state ≠ initial, last state not terminal. Also (user-approved additions): inbound-labeled samples where the assistant speaks first, and prose→prose consecutive assistant turns (break strict-alternation chat templates).

Decisions made with user: implement all fixes including both shape checks; do **not** regenerate L2/L3 data in this change (user will regenerate later).

## Changes

### 1. Fix A — inline markers authoritative in `_backfill_annotations` (generate_workflows.py:614-656)

- `state_transition`: if `_STATE_ANNOTATION_RE.search(content)` matches, **overwrite** `ann["state_transition"]` from it; keep teacher annotation only when content has no marker.
- `tool_calls`: parse all `_TOOL_CALL_RE` blocks; if the parsed list is non-empty, **overwrite** `ann["tool_calls"]`; keep teacher list when content has no parseable blocks (don't wipe on JSON parse failure).
- Per-field policy (state may be overwritten while tool_calls kept). Update docstring to the new "content wins" contract.
- Leave `_extract_ground_truth` (:659) annotations-first — backfill runs at :1464, :1501, :1544, covering all paths before extraction at :1628. Add a docstring note that it assumes backfill has normalized annotations.

### 2. Fix B — shared coherence checkers in `_workflow_script.py`

Add next to `find_tool_placement_violations` (:113), reusing module `_STATE_RE` (:49); both shared by generator repair loop and `data_validator.py` (same lazy-import pattern as validator:153):

```python
def find_continuity_violations(messages, initial_state, terminal_states) -> list[str]
```
Checks (state NAMEs; callers map IDs→names): (a) >1 `[STATE:` marker per assistant message, or first marker not at start of `content.lstrip()`; (b) consecutive assistant annotations discontinuous (`prev.to != cur.from`); (c) first annotation `from` ≠ initial; (d) last annotation `to` not in terminals; (e) assistant message with no `annotations.state_transition` (post-backfill = no marker and no annotation).

```python
def find_shape_violations(messages, initiator) -> list[str]
```
Skip a leading `system` message if present, then: (f) `initiator == "user"` → first message must be `user` (and `"agent"` → `assistant`); (g) consecutive assistant messages where the second, after stripping a leading `[STATE: …]` marker, does **not** start with `<tool_call>` (prose→prose split; speak-then-tool-call splits stay allowed).

**Wiring — repair loop** (generate_workflows.py:1470-1501): in the setup block compute `initial_name` / `terminal_names` via the existing `id_to_name` map; extend `_has_violations` with `or find_continuity_violations(...) or find_shape_violations(msgs, initiator)`. Optional `logger.debug("teacher_coherence_violations", violations=...)` before retries.

**Wiring — data_validator.py** `_check_workflow_rationality` (:111): after the tool-placement check (~:160), run both checkers (continuity gated on `graph.get("initial")`/`graph.get("terminal")` present; shape using the sample's `conversation_initiator`). Append violations as `f"Sample {idx}: {violation}"`. Note: previously generated files with these defects will now fail `validate_dataset` — intended.

### 3. Fix C — provenance tag, global redraw budget, stats sidecar (generate_workflows.py)

**3a. `generation_source`** on `ConversationSample` (:732-769): new defaulted field + `to_dict()` key. Set in the per-sample loop (~:1615): `"placeholder"` (teacher_model=None), `"placeholder_fallback"` (teacher path, `result["fell_back"]`), else `"teacher"`. Tally `source_counts`. Downstream-safe: validator is required-fields-only; grpo/sft/converter/benchmark read keys via `.get()` (verified).

**3b. Global redraw budget** replacing accept-placeholder-after-3 (loop :1520-1537): new param `max_total_redraws: int | None = None`; effective `redraw_budget = max_total_redraws if not None else (max_sample_attempts - 1) * num_samples`. Loop redraws fresh samples while `result["fell_back"] and redraws_used < redraw_budget`; on exhaustion emit placeholder tagged `placeholder_fallback` with the existing `sample_fallback_placeholder_emitted` warning (+ `redraw_budget_exhausted=True`). Properties: hard samples can use many redraws; permanently-broken teacher bounded by `num_samples + redraw_budget` attempts; **aggregate stats identical to today in the broken-teacher case**, so `test_irreparable_teacher_falls_back_to_placeholder`'s exact assertion values still hold (only per-sample attempt distribution changes). Keep `max_sample_attempts` for back-compat (scripts/generate_data.sh passes kwargs); re-document as the per-sample average redraw allowance feeding the default budget.

**3c. Stats sidecar**: after JSONL write (:1634-1637), extend `stats` with `generation_source_counts` and `redraw_budget`; write `output_file.with_suffix(".stats.json")` with envelope (`complexity_level`, `num_samples`, `teacher_model`, `seed`, `output_file` name, timestamp, full `stats`), `json.dump(indent=2, ensure_ascii=False)`. Add `stats_file: Path | None = None` to `DatasetMetadata` (:165-173) — do **not** append to `output_files` (callers iterate it into `validate_dataset`).

### 4. Tests (tests/unit/test_data_generation.py, + validator tests)

**New fixture** in `TestPostGenerationRepair`: `_coherent_conversation(workflow)` — greedy legal-edge walk from initial to a terminal, one user+assistant pair per hop, markers at position 0, user-first. (Existing coherent stubs `s0→s0` / `s0→s1` violate the new terminal check.)

**Updates**: `test_irreparable_teacher_falls_back_to_placeholder` (:568) — values unchanged, add `generation_source == "placeholder_fallback"` + sidecar count assertions; `test_retry_succeeds_without_fallback` (:591) and `test_fresh_retry_recovers_without_placeholder` (:625) — swap recovery stub for `_coherent_conversation`; `test_teacher_system_message_is_stripped` (:752) — pass `repair_incoherent=False` (its stub ends non-terminal; preserves the test's intent); `test_placeholder_path_needs_no_repair` (:662) — add `generation_source == "placeholder"` assertion.

**New**:
- Bug A: backfill overwrites contradicting state annotation (L3_043 repro); overwrites contradicting tool_calls; keeps annotation when content has no marker / unparseable JSON; integration — `ground_truth.state_sequence` follows inline markers on disagreement (`repair_incoherent=False`).
- Bug B unit (per violation type on `find_continuity_violations`): mid-message marker (L3_026 repro), discontinuity, wrong initial, non-terminal ending (L2_029 repro), missing annotation, clean-walk → `[]` (with self-loops + leading newline tolerated).
- Shape unit (`find_shape_violations`): inbound assistant-first flagged; outbound assistant-first OK; prose→prose flagged; speak→tool-call split allowed.
- Integration: discontinuous-but-edge-legal teacher output triggers repair (`repair_retries >= 1`, no fallback).
- Fix C: redraw budget allows recovery beyond old 3-attempt cap (`max_repair_retries=0`, 5 bad calls then good, `max_total_redraws=10` → `generation_source == "teacher"`); budget exhaustion emits tagged fallback; sidecar written and `generation_source_counts` matches a re-read of the JSONL; `to_dict()` includes `generation_source`.
- Validator: rejects discontinuous / non-terminal / inbound-assistant-first hand-built samples; placeholder-generated dataset still passes full validation (regression guard for the stricter validator).

## Files

- `src/llm_workflow_agents/data/generate_workflows.py` — Fixes A, C; repair-loop wiring
- `src/llm_workflow_agents/data/_workflow_script.py` — `find_continuity_violations`, `find_shape_violations`
- `src/llm_workflow_agents/data/data_validator.py` — validator wiring
- `tests/unit/test_data_generation.py`, `tests/unit/test_data_validator.py` — tests
- Copy this plan to `docs/superpowers/plans/2026-06-11-teacher-pipeline-fallback-fixes.md` (user's standing preference for plan location)

## Risks / notes

- Stricter checks raise teacher retry pressure (L3 already fell back 42%); budget + provenance make it visible instead of silent. Future (out of scope): feed violation strings back into the retry prompt.
- Retroactive validator strictness: old files now fail validation — intended; user regenerates data separately.
- One pathological sample can drain the shared budget; worst case equals today's total cost. Documented in the param docstring.

## Verification

1. `source .venv/bin/activate && pytest tests/unit/test_data_generation.py tests/unit/test_data_validator.py -x` (fast, no API).
2. Full suite: `pytest tests/ -m "not gpu and not slow"`.
3. End-to-end without API: run `generate_workflow_dataset("L3", num_samples=5, teacher_model=None, output_dir=tmp)` → every sample `generation_source == "placeholder"`, sidecar exists, `validate_dataset` passes.
4. Re-run the audit's fingerprint script against a placeholder-mode output to confirm `generation_source` matches fingerprint detection.
5. Run `validate_dataset` on the existing audited L3 file → confirm it now reports the L3_026/L3_043-class violations (proves the validator catches what the audit found by hand).
