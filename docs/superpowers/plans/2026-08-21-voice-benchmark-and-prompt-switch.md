# Voice Mode Switch and Voice Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the system prompt select voice mode, teach the placeholder generator barge-in, add an additive voice benchmark stratum, and blend the two strata into one Phase 1 quality number without moving the existing ranking until voice data exists.

**Architecture:** One `VOICE_FORMAT_RULES` constant in `voice_convention.py` feeds both the teacher prompt and a new serving-prompt voice block. The existing benchmark corpus is frozen as the text stratum; a new teacher-generated voice corpus is added alongside. Scoring runs the existing per-stratum formula unchanged, then blends the two scores with a weight that defaults to 0.30.

**Tech Stack:** Python 3.12, pytest, DVC. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-21-voice-benchmark-and-prompt-switch-design.md`

## Global Constraints

- Use `uv`, never `pip`. Prefix every Python command with `source .venv/bin/activate &&`.
- **A text sample's enriched system prompt must render byte-identically to today.** This keeps the 0.7595 composite in risk R17 comparable.
- **Do NOT run `dvc repro task_a_benchmark`.** That stage's description is wrong (it claims 1,000 placeholder conversations; the artifact holds 258 teacher-generated ones). Running it destroys the data behind the current Cat A ranking.
- **Do NOT regenerate `data/output/benchmark/task_a`.** It is the frozen text stratum.
- **Do NOT add the voice weight to `configs/benchmark/selection_weights.yaml`.** No code reads that file.
- Chunk limits: chunk target 100 chars, violation above 160. Turn target 3 chunks, violation above 5.
- A state marker's arrow may be `→` or `->`.
- Do NOT use `git stash`. This repo holds an unrelated stash entry from another branch.
- Known pre-existing test failures: `trl`→vllm→torchvision collection errors, and bf16/no-GPU failures. Any OTHER failure is new breakage.
- No GPU and no `dvc` CLI in this environment.
- Write the test first. Run it. Watch it fail. Then implement.
- Branch: `feat/voice-benchmark-and-prompt-switch`. It exists and holds the spec commit.

---

### Task 1: One source for the voice rules

Move the V1–V6 prose into `voice_convention.py` so the teacher prompt and the new serving prompt render from one string.

**Files:**
- Modify: `src/llm_workflow_agents/data/voice_convention.py`
- Modify: `src/llm_workflow_agents/data/generate_workflows.py:1240-1264` (`_VOICE_RULES`), `:1295-1305` (`_teacher_system_prompt`)
- Test: `tests/unit/test_voice_prompt_contract.py`

**Interfaces:**
- Consumes: `CHUNK_TARGET_CHARS`, `CHUNK_MAX_CHARS`, `TURN_TARGET_CHUNKS`, `TURN_MAX_CHUNKS` (already in `voice_convention.py`).
- Produces: `render_voice_format_rules() -> str` in `voice_convention.py`, returning the V1–V6 block with the four limits already substituted.

**Scope note the brief must respect:** `_RICH_VOICE_OVERRIDE` is NOT a copy of the rules. It is a short instruction about authoring quoted dialogue lines in a system prompt. Leave it alone. Only `_VOICE_RULES` moves.

- [ ] **Step 1: Write the failing test**

Replace the body of `tests/unit/test_voice_prompt_contract.py` (keep its docstring, updating the claim that the copies are not generated from the module):

```python
def test_render_voice_format_rules_substitutes_every_limit():
    from llm_workflow_agents.data.voice_convention import render_voice_format_rules

    text = render_voice_format_rules()
    # No placeholder may survive. Do not assert "no braces at all": the worked
    # example contains real JSON, and .format() un-doubles those braces.
    for name in ("{chunk_target}", "{chunk_max}", "{turn_target}", "{turn_max}"):
        assert name not in text
    for value in (vc.CHUNK_TARGET_CHARS, vc.CHUNK_MAX_CHARS,
                  vc.TURN_TARGET_CHUNKS, vc.TURN_MAX_CHUNKS):
        assert str(value) in text


def test_teacher_voice_prompt_embeds_the_shared_rules_verbatim():
    """Identity, not keyword presence. The old test asserted only that certain
    strings appeared, so inverting a rule's meaning left it passing."""
    from llm_workflow_agents.data.generate_workflows import _teacher_system_prompt
    from llm_workflow_agents.data.voice_convention import render_voice_format_rules

    assert render_voice_format_rules() in _teacher_system_prompt("voice")


def test_text_teacher_prompt_holds_no_voice_rules():
    from llm_workflow_agents.data.generate_workflows import _teacher_system_prompt
    from llm_workflow_agents.data.voice_convention import render_voice_format_rules

    assert render_voice_format_rules() not in _teacher_system_prompt("text")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_prompt_contract.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_voice_format_rules'`

- [ ] **Step 3: Move the prose into voice_convention.py**

Add to `src/llm_workflow_agents/data/voice_convention.py`:

```python
#: The voice format contract, stated in prose for a language model.
#:
#: `find_voice_violations` is the enforced contract; this is the same contract
#: written for a reader that cannot run code. Both the teacher prompt and the
#: serving system prompt render from THIS string, so the two can never drift.
#: The braces in the worked example are doubled because the caller may pass the
#: result through str.format; render_voice_format_rules does the substitution.
_VOICE_FORMAT_RULES_TEMPLATE = """\

VOICE MODE — this conversation is spoken aloud through a text-to-speech engine.
The orchestrator reads your output as a stream, finds each <S>...</S> chunk, and
sends it to the engine in order. Six extra rules apply:

- V1. Put the [STATE: X → Y] marker on the first line, OUTSIDE every <S>. The
  agent never speaks it.
- V2. Put every <tool_call> block OUTSIDE every <S>. The agent never speaks it.
- V3. Put every spoken word INSIDE a chunk. No spoken text may sit outside
  <S>...</S>. A turn with no spoken text at all is legal and carries no chunk;
  a turn that only calls a tool is silent on the line. Never invent filler
  speech to give such a turn a chunk.
- V4. Split at natural pause points. Keep a chunk to {chunk_target} characters
  and never above {chunk_max}. Keep a turn to {turn_target} chunks and never
  above {turn_max}.
- V5. Keep replies short. A spoken reply is one or two sentences. Use no
  markdown, no bullet points, no numbered lists, no headers.
- V6. End a terminal turn with [END_CONVERSATION] after the last </S>, outside
  the chunks. Never put it on a turn that also calls a tool.

Worked example of one voice assistant turn:
    [STATE: VERIFY_PATIENT → VERIFY_PATIENT]
    <S>ได้เลยค่ะ</S><S>ขออนุญาตตรวจสอบข้อมูลสักครู่นะคะ</S>
    <tool_call>{{"name": "request_referral", "arguments": {{"patient_id": "P12345"}}}}</tool_call>
"""


def render_voice_format_rules() -> str:
    """Return the voice format contract with the four limits substituted."""
    return _VOICE_FORMAT_RULES_TEMPLATE.format(
        chunk_target=CHUNK_TARGET_CHARS,
        chunk_max=CHUNK_MAX_CHARS,
        turn_target=TURN_TARGET_CHUNKS,
        turn_max=TURN_MAX_CHUNKS,
    )
```

Note V3 gained the silent-turn sentence. That matches the current checker and `.claude/rules/02-data-generation.md`; the old `_VOICE_RULES` omitted it.

- [ ] **Step 4: Point the teacher prompt at it**

In `generate_workflows.py`, delete the `_VOICE_RULES` constant and change `_teacher_system_prompt`'s voice branch to:

```python
    from llm_workflow_agents.data.voice_convention import render_voice_format_rules

    return _TEACHER_SYSTEM_PROMPT + render_voice_format_rules()
```

Delete the now-unused `.format(...)` call and the four constant imports if nothing else in that function uses them.

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_prompt_contract.py tests/unit/test_data_generation.py -q`
Expected: PASS. `test_text_system_prompt_is_unchanged` in `test_data_generation.py` must still pass — the text branch's bytes are frozen.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/voice_convention.py src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_voice_prompt_contract.py
git commit -m "refactor(data): one source for the voice format rules

The teacher prompt held its own prose copy of V1-V6. The contract test
asserted only that certain strings appeared, so a reviewer inverted a
rule's meaning from 'outside' to 'inside' and every assertion still
passed. Both now render from render_voice_format_rules(), and the test
is an identity check.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The serving-prompt voice block

Make `build_enriched_system_prompt` select voice mode. This is the gap that blocks everything else.

**Files:**
- Modify: `src/llm_workflow_agents/data/system_prompt.py:295-400` (`build_enriched_system_prompt`)
- Test: `tests/unit/test_system_prompt_voice.py` (create)

**Interfaces:**
- Consumes: `render_voice_format_rules()` from Task 1.
- Produces: `build_enriched_system_prompt` appends the voice block when `sample.get("modality") == "voice"`. Signature unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_system_prompt_voice.py`:

```python
"""The enriched system prompt must select voice mode.

Four consumers build prompts through this one function: eval/agent_benchmark.py,
scripts/heldout_composite_audit.py, GRPO rollouts, and SFT training. Before this
change none of them told a model to chunk its speech, so a voice row measured
whether the model guessed an unstated convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
from llm_workflow_agents.data.voice_convention import render_voice_format_rules

CORPUS = Path("data/output/sft/task_a_splits/test.jsonl")


def _row(**overrides):
    sample = {
        "workflow_graph": {"initial": "A", "terminal": ["B"], "state_details": {}},
        "tool_schemas": [],
        "messages": [],
        "language": "en",
        "complexity_level": "L1",
    }
    sample.update(overrides)
    return sample


def test_voice_sample_gets_the_block():
    out = build_enriched_system_prompt(_row(modality="voice"), "You are an agent.")
    assert render_voice_format_rules() in out


def test_text_sample_does_not_get_the_block():
    out = build_enriched_system_prompt(_row(modality="text"), "You are an agent.")
    assert render_voice_format_rules() not in out


def test_absent_modality_is_treated_as_text():
    """Every pre-existing corpus row predates the field."""
    out = build_enriched_system_prompt(_row(), "You are an agent.")
    assert render_voice_format_rules() not in out


def test_force_rebuild_regenerates_the_block():
    sample = _row(modality="voice")
    once = build_enriched_system_prompt(sample, "You are an agent.")
    twice = build_enriched_system_prompt(sample, once, force_rebuild=True)
    assert twice.count(render_voice_format_rules()) == 1


def test_idempotent_without_force_rebuild():
    sample = _row(modality="voice")
    once = build_enriched_system_prompt(sample, "You are an agent.")
    assert build_enriched_system_prompt(sample, once) == once


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not materialized")
def test_real_text_rows_render_byte_identically(tmp_path):
    """The property that keeps R17's 0.7595 comparable.

    Baseline is captured from the CURRENT implementation before the voice block
    lands, written to a file, and compared after. Regenerate the baseline only
    when a change to the text prompt is intended.
    """
    baseline = Path("tests/fixtures/text_prompt_baseline.json")
    rows = [json.loads(x) for x in CORPUS.read_text().splitlines()[:20]]
    rendered = [
        build_enriched_system_prompt(r, r["messages"][0]["content"], force_rebuild=True)
        for r in rows
    ]
    assert baseline.exists(), "run the baseline capture step first"
    assert json.loads(baseline.read_text()) == rendered
```

- [ ] **Step 2: Capture the byte-identity baseline BEFORE implementing**

This step must run before any change to `system_prompt.py`, or the baseline records the new behaviour and the test proves nothing.

FIRST un-ignore the fixture. `.gitignore:62` is a blanket `*.json` guarding GCP
credentials, so `git add tests/fixtures/text_prompt_baseline.json` would SILENTLY
SKIP it, and the byte-identity test would then fail on any fresh checkout with
"run the baseline capture step first" — a test that cannot run. The repo already
uses this escape hatch one line below for `!deployments/**/config.json`, so add
the same kind of negation immediately after that line:

```
!tests/fixtures/*.json
```

Confirm it worked before continuing: `git check-ignore -v tests/fixtures/text_prompt_baseline.json` must print nothing.

```bash
source .venv/bin/activate && python -c "
import json, pathlib
from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt
rows = [json.loads(x) for x in pathlib.Path('data/output/sft/task_a_splits/test.jsonl').read_text().splitlines()[:20]]
out = [build_enriched_system_prompt(r, r['messages'][0]['content'], force_rebuild=True) for r in rows]
p = pathlib.Path('tests/fixtures/text_prompt_baseline.json'); p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(out))
print('captured', len(out), 'baseline prompts')
"
```
Expected: `captured 20 baseline prompts`

- [ ] **Step 3: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_system_prompt_voice.py -v`
Expected: the three voice tests FAIL; the byte-identity test PASSES (nothing has changed yet).

- [ ] **Step 4: Implement the block**

In `build_enriched_system_prompt`, immediately before the final `FORMAT_RULES` append (the line `parts.append(f"\n{_format_rules_cached(retry_budget, _STAY_RULE_ENABLED)}")`), insert:

```python
    # Voice mode. A row with no `modality` field is text: every conversation
    # generated before the field existed is a written one. The block sits after
    # the "Workflow script" marker so force_rebuild regenerates it from current
    # code rather than preserving a stale copy (the defect risk R13 records).
    if (sample.get("modality") or "text") == "voice":
        from llm_workflow_agents.data.voice_convention import render_voice_format_rules

        parts.append(render_voice_format_rules())
```

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_system_prompt_voice.py -v`
Expected: PASS, 6 tests. The byte-identity test must STILL pass — if it fails, a text prompt moved and that is a stop-and-report condition.

- [ ] **Step 6: Confirm nothing else regressed**

Run: `source .venv/bin/activate && pytest tests/unit -q --continue-on-collection-errors 2>&1 | tail -3`
Expected: same pre-existing failure set, no new failures.

- [ ] **Step 7: Commit**

```bash
git add src/llm_workflow_agents/data/system_prompt.py tests/unit/test_system_prompt_voice.py tests/fixtures/text_prompt_baseline.json
git commit -m "feat(data): the system prompt selects voice mode

build_enriched_system_prompt now appends the voice format rules for a
voice sample. Four consumers inherit it: agent_benchmark, the held-out
audit, GRPO rollouts and SFT training. Before this, a voice row carried
no voice instruction unless it happened to have a teacher-authored rich
prompt (~30% of samples).

A text sample's prompt is unchanged, byte for byte, against a captured
20-row baseline.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Barge-in from the placeholder generator

**Files:**
- Modify: `src/llm_workflow_agents/data/voice_convention.py` (acknowledgement lookup)
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (`_generate_placeholder_conversation`)
- Test: `tests/unit/test_data_generation.py`, `tests/unit/test_voice_convention.py`

**Interfaces:**
- Consumes: `ACKNOWLEDGEMENTS`, `iter_chunks`, `apply_barge_in_loss_flag` (all present).
- Produces: `acknowledgement_for(language: str) -> tuple[str, ...]` in `voice_convention.py`. The placeholder honours `barge_in` for voice samples.

- [ ] **Step 1: Write the failing tests for the language lookup**

Append to `tests/unit/test_voice_convention.py`:

```python
def test_acknowledgement_for_code_switch_returns_thai():
    """Code-switched conversations are Thai-primary; English openers read wrong."""
    from llm_workflow_agents.data.voice_convention import ACKNOWLEDGEMENTS, acknowledgement_for

    assert acknowledgement_for("code_switch") == ACKNOWLEDGEMENTS["th"]


def test_acknowledgement_for_known_languages():
    from llm_workflow_agents.data.voice_convention import ACKNOWLEDGEMENTS, acknowledgement_for

    assert acknowledgement_for("th") == ACKNOWLEDGEMENTS["th"]
    assert acknowledgement_for("en") == ACKNOWLEDGEMENTS["en"]


def test_acknowledgement_for_unknown_language_falls_back_to_english():
    from llm_workflow_agents.data.voice_convention import ACKNOWLEDGEMENTS, acknowledgement_for

    assert acknowledgement_for("de") == ACKNOWLEDGEMENTS["en"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v -k acknowledgement_for`
Expected: FAIL with `ImportError: cannot import name 'acknowledgement_for'`

- [ ] **Step 3: Implement the lookup**

Add to `voice_convention.py`:

```python
def acknowledgement_for(language: str) -> tuple[str, ...]:
    """Return the acknowledgement openers for one language.

    `code_switch` maps to Thai: a code-switched conversation is Thai-primary,
    so an English opener reads wrong even though English words appear in it.
    """
    if language == "code_switch":
        return ACKNOWLEDGEMENTS["th"]
    return ACKNOWLEDGEMENTS.get(language, ACKNOWLEDGEMENTS["en"])
```

Then replace every existing `ACKNOWLEDGEMENTS.get(language, ...)` call site in `generate_workflows.py` with `acknowledgement_for(language)`. Find them with:

```bash
grep -n "ACKNOWLEDGEMENTS" src/llm_workflow_agents/data/generate_workflows.py
```

- [ ] **Step 4: Run the lookup tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v -k acknowledgement_for`
Expected: PASS, 3 tests.

- [ ] **Step 5: Write the failing tests for placeholder barge-in**

Append to `tests/unit/test_data_generation.py`:

```python
class TestPlaceholderBargeIn:
    """The placeholder must be able to produce an interruption.

    The teacher writes barge-ins for teacher runs. The placeholder is the
    offline, reproducible path, and it produced none.
    """

    def _voice_rows(self, tmp_path, seed=17, rate=1.0, n=6):
        meta = generate_workflow_dataset(
            "L3", num_samples=n, output_dir=tmp_path, seed=seed,
            modality_preset="voice_only", barge_in_rate=rate,
        )
        return [json.loads(x) for x in meta.output_files[0].read_text().splitlines()]

    def test_placeholder_emits_barge_in_at_rate_one(self, tmp_path):
        rows = self._voice_rows(tmp_path)
        assert any(r["barge_in"] for r in rows)

    def test_barge_in_rows_pass_the_checker(self, tmp_path):
        from llm_workflow_agents.data.voice_convention import find_voice_violations

        for r in self._voice_rows(tmp_path):
            assert find_voice_violations(r["messages"], "voice") == []

    def test_marker_bearing_turn_carries_loss_false(self, tmp_path):
        for r in self._voice_rows(tmp_path):
            if not r["barge_in"]:
                continue
            marked = [m for m in r["messages"] if "<unspoken>" in (m.get("content") or "")]
            assert len(marked) == 1
            assert marked[0]["loss"] is False

    def test_rate_zero_emits_none(self, tmp_path):
        rows = self._voice_rows(tmp_path, rate=0.0)
        assert not any(r["barge_in"] for r in rows)

    def test_reproducible_at_a_fixed_seed(self, tmp_path):
        a = self._voice_rows(tmp_path / "a")
        b = self._voice_rows(tmp_path / "b")
        assert [r["messages"] for r in a] == [r["messages"] for r in b]

    def test_text_samples_never_get_a_marker(self, tmp_path):
        meta = generate_workflow_dataset(
            "L3", num_samples=6, output_dir=tmp_path, seed=17, barge_in_rate=1.0,
        )
        rows = [json.loads(x) for x in meta.output_files[0].read_text().splitlines()]
        assert not any(r["barge_in"] for r in rows)
        assert not any("<unspoken>" in (m.get("content") or "")
                       for r in rows for m in r["messages"])
```

- [ ] **Step 6: Run to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py::TestPlaceholderBargeIn -v`
Expected: FAIL — the placeholder emits no marker, so `barge_in` is False everywhere.

- [ ] **Step 7: Implement placeholder barge-in**

In `_generate_placeholder_conversation`, after the message list is complete and before it is returned, apply the interruption when the sample is voice and its `barge_in` draw is true. The function must receive that draw; thread it in the same way `modality` already is.

```python
def _insert_placeholder_barge_in(
    messages: list[dict[str, Any]],
    language: str,
    rng: random.Random,
) -> None:
    """Insert one interruption in place. No-op when no turn qualifies.

    Three coordinated edits, because a barge-in is not one insertion: the
    interrupted turn is cut, an interrupting user turn follows it, and a
    recovery assistant turn follows that. The recovery annotates the SAME
    state — an interruption completes nothing, so the workflow must not
    advance, and find_barge_in_violations enforces exactly that.
    """
    from llm_workflow_agents.data.voice_convention import (
        acknowledgement_for,
        iter_chunks,
    )

    candidates = [
        i for i, m in enumerate(messages)
        if m.get("role") == "assistant"
        and i not in (0, len(messages) - 1)
        and len(iter_chunks(m.get("content") or "")) >= 2
        and "[END_CONVERSATION]" not in (m.get("content") or "")
    ]
    if not candidates:
        return

    idx = candidates[rng.randrange(len(candidates))]
    content = messages[idx]["content"]
    chunks = iter_chunks(content)
    victim = chunks[-1]
    words = victim.split()
    if len(words) < 2:
        return
    cut = len(" ".join(words[: max(1, len(words) // 2)]))
    interrupted = content.replace(
        f"<S>{victim}</S>", f"<S>{victim[:cut]}<unspoken>{victim[cut:]}</S>", 1
    )
    messages[idx]["content"] = interrupted

    state = _STATE_ANNOTATION_RE.search(content)
    from_state = state.group(1) if state else ""
    opener = acknowledgement_for(language)[0]
    messages.insert(idx + 1, {"role": "user", "content": "ขอโทษนะคะ ขอถามหน่อย"})
    messages.insert(idx + 2, {
        "role": "assistant",
        "content": (
            f"[STATE: {from_state} → {from_state}]\n"
            f"<S>{opener}</S><S>{victim[cut:].strip() or victim}</S>"
        ),
    })
```

Call it from `_generate_placeholder_conversation` before the return, guarded on `modality == "voice" and barge_in`.

- [ ] **Step 8: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py::TestPlaceholderBargeIn tests/unit/test_voice_convention.py -v`
Expected: PASS.

- [ ] **Step 9: Confirm seed determinism still holds**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -v -k "determinism or does_not_shift"`
Expected: PASS, 3 tests.

- [ ] **Step 10: Commit**

```bash
git add src/llm_workflow_agents/data/voice_convention.py src/llm_workflow_agents/data/generate_workflows.py tests/unit/
git commit -m "feat(data): the placeholder generator can produce a barge-in

Three coordinated edits, not one insertion: the interrupted turn is cut
at a word boundary, an interrupting user turn follows, and a recovery
turn acknowledges and re-delivers while holding the same state.

Also routes code_switch to the Thai acknowledgements; a code-switched
conversation is Thai-primary and the English openers read wrong.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Blended scoring

**Files:**
- Modify: `src/llm_workflow_agents/eval/composite_score.py`
- Modify: `src/llm_workflow_agents/eval/agent_benchmark.py` (aggregation + CLI)
- Test: `tests/unit/test_benchmark_scoring.py` (create)

**Interfaces:**
- Consumes: `compute_weighted_workflow_score(state, tool) -> float` (existing, unchanged).
- Produces: `DEFAULT_VOICE_WEIGHT = 0.30` and
  `blend_modality_scores(score_text: float | None, score_voice: float | None, voice_weight: float = DEFAULT_VOICE_WEIGHT) -> float`
  in `composite_score.py`.

**Do not touch** `configs/benchmark/selection_weights.yaml`. Nothing reads it; a weight added there would do nothing.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_benchmark_scoring.py`:

```python
"""Blending the two modality strata into one Phase 1 quality number.

The blend must be a weighted mean of per-stratum means, never a mean over
pooled rows: a pooled mean takes its weight from row counts, so the weighting
drifts silently whenever anyone regenerates the data.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.composite_score import (
    DEFAULT_VOICE_WEIGHT,
    blend_modality_scores,
)


def test_default_weight_is_030():
    assert DEFAULT_VOICE_WEIGHT == 0.30


def test_blend_at_weight_zero_is_the_text_score():
    assert blend_modality_scores(0.8, 0.2, voice_weight=0.0) == 0.8


def test_blend_at_weight_one_is_the_voice_score():
    assert blend_modality_scores(0.8, 0.2, voice_weight=1.0) == 0.2


def test_blend_at_default_weight():
    assert blend_modality_scores(0.8, 0.2) == pytest.approx(0.7 * 0.8 + 0.3 * 0.2)


def test_no_voice_stratum_returns_the_text_score_exactly():
    """Float identity, not approximate.

    This is what makes the change safe to merge: results move when a person
    adds the voice corpus, not because someone merged a branch.
    """
    assert blend_modality_scores(0.7595, None) == 0.7595


def test_no_text_stratum_returns_the_voice_score_exactly():
    assert blend_modality_scores(None, 0.4242) == 0.4242


def test_both_absent_is_zero():
    assert blend_modality_scores(None, None) == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_weight_outside_zero_to_one_is_rejected(bad):
    with pytest.raises(ValueError, match="voice_weight"):
        blend_modality_scores(0.5, 0.5, voice_weight=bad)
```

- [ ] **Step 2: Run to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_benchmark_scoring.py -v`
Expected: FAIL with `ImportError: cannot import name 'DEFAULT_VOICE_WEIGHT'`

- [ ] **Step 3: Implement the blend**

Add to `src/llm_workflow_agents/eval/composite_score.py`:

```python
#: Share of Phase 1 quality carried by the voice stratum.
#:
#: NOT in configs/benchmark/selection_weights.yaml on purpose: no code reads
#: that file, so a weight placed there would be a knob that does nothing —
#: the defect shape risks R16 and R18c both record.
DEFAULT_VOICE_WEIGHT = 0.30


def blend_modality_scores(
    score_text: float | None,
    score_voice: float | None,
    voice_weight: float = DEFAULT_VOICE_WEIGHT,
) -> float:
    """Blend the two modality strata into one quality number.

    A weighted mean of two per-stratum scores, NOT a mean over pooled rows.
    A pooled mean takes its effective weight from however many rows each
    stratum happens to hold, so it drifts every time the data is regenerated
    and nobody notices, because the result still looks like a number.

    With one stratum absent the other is returned unchanged, by identity and
    not by arithmetic. That keeps this change inert until a voice corpus
    exists: the ranking moves when a person adds voice data, not when someone
    merges a branch.
    """
    if not 0.0 <= voice_weight <= 1.0:
        raise ValueError(f"voice_weight must be within 0.0 and 1.0, got {voice_weight}")
    if score_voice is None:
        return score_text if score_text is not None else 0.0
    if score_text is None:
        return score_voice
    return voice_weight * score_voice + (1.0 - voice_weight) * score_text
```

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_benchmark_scoring.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Group the benchmark's rows by modality**

In `src/llm_workflow_agents/eval/agent_benchmark.py`, locate the site that aggregates per-conversation results into the run summary. Find it with:

```bash
grep -n "state_metrics\|tool_metrics\|def .*summar\|def run_benchmark" src/llm_workflow_agents/eval/agent_benchmark.py
```

Partition the scored conversations by `(conv.get("modality") or "text")`, compute the score once per stratum, and report the results.

**Use `agent_benchmark.py`'s OWN `compute_weighted_score(state, tool, completion)`,
defined at about line 62 of that file.** Do NOT use
`composite_score.compute_weighted_workflow_score(state, tool)`. They are
different functions: the benchmark's takes `completion` explicitly and uses the
BETTER of per-turn and conversation-level state accuracy, and `agent_benchmark.py`
does not import `composite_score` at all. Blending a different formula than the
one this benchmark has always used would silently change what its headline number
means. `blend_modality_scores` is unaffected — it blends two floats and does not
care which scorer produced them.

Report:

- `quality_text`, or `null` when the stratum is empty
- `quality_voice`, or `null` when the stratum is empty
- `quality` — `blend_modality_scores(quality_text, quality_voice, args.voice_weight)`
- `n_text`, `n_voice`
- `voice_weight` used

Print the two per-stratum numbers beside the blend. Without them nobody can tell whether a winner is better at state machines or better at emitting `<S>`.

- [ ] **Step 6: Add the CLI flag**

Beside the existing arguments (around line 624):

```python
    parser.add_argument(
        "--voice-weight",
        type=float,
        default=DEFAULT_VOICE_WEIGHT,
        help=(
            "Share of quality carried by the voice stratum (default: "
            f"{DEFAULT_VOICE_WEIGHT}). Ignored when the run holds no voice "
            "conversations, in which case quality equals the text score exactly."
        ),
    )
```

- [ ] **Step 7: Prove the change is inert on the existing benchmark**

Run the benchmark's summary path against the frozen text corpus and confirm `quality` equals `quality_text` exactly and `quality_voice` is null. If your environment cannot reach a serving endpoint, construct the summary from stored per-conversation metrics instead and state in your report which you did.

- [ ] **Step 8: Commit**

```bash
git add src/llm_workflow_agents/eval/composite_score.py src/llm_workflow_agents/eval/agent_benchmark.py tests/unit/test_benchmark_scoring.py
git commit -m "feat(eval): blend the modality strata into one quality number

A weighted mean of two per-stratum scores, not a mean over pooled rows:
a pooled mean takes its weight from row counts and drifts whenever the
data is regenerated.

With no voice stratum the blend returns the text score by identity, so
merging this cannot move the existing Cat A ranking.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Chunk diagnostics

**Files:**
- Create: `src/llm_workflow_agents/eval/chunk_diagnostics.py`
- Modify: `src/llm_workflow_agents/eval/agent_benchmark.py` (report them)
- Test: `tests/unit/test_chunk_diagnostics.py` (create)

**Interfaces:**
- Consumes: `iter_chunks` from `voice_convention.py`.
- Produces: `chunk_diagnostics(completions: list[str], language: str = "en") -> dict[str, float | int]` returning keys `first_chunk_p50`, `first_chunk_p90`, `chunk_len_p50`, `chunk_len_p90`, `chunks_per_turn_p50`, `boundary_quality`, `n_turns_with_chunks`.

These are **guardrails**. None enters the composite.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_chunk_diagnostics.py`:

```python
"""Reference-free chunk diagnostics.

Not measured against gold: the benchmark scores free generation, so the model's
words differ from the gold words, and once the words differ the boundaries
cannot be aligned. A "chunk F1 against gold" would compare boundaries in two
different sentences and return noise.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.eval.chunk_diagnostics import chunk_diagnostics


def test_first_chunk_length_is_the_first_chunk_not_the_shortest():
    out = chunk_diagnostics(["<S>" + "a" * 40 + "</S><S>bb</S>"])
    assert out["first_chunk_p50"] == 40


def test_chunk_length_percentiles():
    """Three chunks of 10 / 50 / 90 characters.

    _pct uses nearest-rank: index int(q * n), clamped. So p50 of three values
    is the middle one and p90 is the largest. Two values would make p50 the
    UPPER of the pair, which is why this fixture uses three.
    """
    turn = "<S>" + "a" * 10 + "</S><S>" + "b" * 50 + "</S><S>" + "c" * 90 + "</S>"
    out = chunk_diagnostics([turn])
    assert out["chunk_len_p50"] == 50
    assert out["chunk_len_p90"] == 90


def test_chunks_per_turn():
    out = chunk_diagnostics(["<S>a</S><S>b</S><S>c</S>"])
    assert out["chunks_per_turn_p50"] == 3


def test_boundary_quality_english_terminal_punctuation():
    good = chunk_diagnostics(["<S>Hello there.</S><S>How can I help?</S>"], "en")
    bad = chunk_diagnostics(["<S>Hello there and</S><S>how can I</S>"], "en")
    assert good["boundary_quality"] == 1.0
    assert bad["boundary_quality"] == 0.0


def test_boundary_quality_thai_final_particles():
    out = chunk_diagnostics(["<S>สวัสดีค่ะ</S><S>ยินดีให้บริการค่ะ</S>"], "th")
    assert out["boundary_quality"] == 1.0


def test_code_switch_accepts_either_convention():
    out = chunk_diagnostics(["<S>Hello there.</S><S>ยินดีให้บริการค่ะ</S>"], "code_switch")
    assert out["boundary_quality"] == 1.0


def test_turn_with_no_chunks_is_excluded_not_counted_as_bad():
    """A silent tool-call turn is legal and carries no chunk."""
    out = chunk_diagnostics(["[STATE: A → A]\n<tool_call>{}</tool_call>"])
    assert out["n_turns_with_chunks"] == 0
    assert out["boundary_quality"] == 0.0


def test_empty_input_does_not_raise():
    out = chunk_diagnostics([])
    assert out["n_turns_with_chunks"] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_chunk_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/llm_workflow_agents/eval/chunk_diagnostics.py`:

```python
"""Reference-free diagnostics for voice chunking.

Guardrails, never composite terms. Chunk formatting is cheap to install
through fine-tuning, so letting it move the Phase 1 winner would select on the
wrong capability. These metrics diagnose a candidate; they do not rank one.
"""

from __future__ import annotations

import statistics
from typing import Any

from llm_workflow_agents.data.voice_convention import iter_chunks

#: Thai sentence-final particles. They mark a pause point explicitly, which
#: makes boundary quality unusually tractable in Thai.
_THAI_FINALS = ("ค่ะ", "คะ", "ครับ", "นะคะ", "นะครับ", "ค่า")
_EN_FINALS = (".", "?", "!", "…")


def _ends_well(chunk: str, language: str) -> bool:
    text = chunk.strip()
    if not text:
        return False
    if language == "th":
        return text.endswith(_THAI_FINALS)
    if language == "code_switch":
        return text.endswith(_THAI_FINALS) or text.endswith(_EN_FINALS)
    return text.endswith(_EN_FINALS)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])


def chunk_diagnostics(
    completions: list[str], language: str = "en"
) -> dict[str, Any]:
    """Return chunk-shape diagnostics over a list of generated turns."""
    first_lengths: list[float] = []
    all_lengths: list[float] = []
    counts: list[float] = []
    well_ended = 0
    total_chunks = 0

    for text in completions:
        chunks = iter_chunks(text or "")
        if not chunks:
            continue
        counts.append(len(chunks))
        first_lengths.append(len(chunks[0]))
        for c in chunks:
            all_lengths.append(len(c))
            total_chunks += 1
            if _ends_well(c, language):
                well_ended += 1

    return {
        "first_chunk_p50": _pct(first_lengths, 0.5),
        "first_chunk_p90": _pct(first_lengths, 0.9),
        "chunk_len_p50": _pct(all_lengths, 0.5),
        "chunk_len_p90": _pct(all_lengths, 0.9),
        "chunks_per_turn_p50": statistics.median(counts) if counts else 0.0,
        "boundary_quality": (well_ended / total_chunks) if total_chunks else 0.0,
        "n_turns_with_chunks": len(counts),
    }
```

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_chunk_diagnostics.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Report them from the benchmark**

In `agent_benchmark.py`'s summary, add a `chunk_diagnostics` block computed over the VOICE stratum's completions only. Add nothing to the composite. Add a test asserting that `quality` is unchanged when the diagnostics block is present.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/eval/chunk_diagnostics.py src/llm_workflow_agents/eval/agent_benchmark.py tests/unit/test_chunk_diagnostics.py
git commit -m "feat(eval): reference-free chunk diagnostics

First-chunk length is the latency metric: the orchestrator starts TTS on
chunk 1 while the model still writes chunk 2, so a 150-character opener
delays audio about ten times as long as a 15-character one.

Guardrails only. Chunk formatting is cheap for SFT to install, so it
must not move the Phase 1 winner.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The voice benchmark stratum

**Files:**
- Create: `scripts/generate_benchmark_voice_data.sh`
- Modify: `dvc.yaml`
- Test: `tests/unit/test_generate_benchmark_voice_data_sh.py` (create)

**Interfaces:**
- Consumes: `generate_workflow_dataset(..., modality_preset=..., barge_in_rate=...)`.
- Produces: `data/output/benchmark/task_a_voice`, 250 conversations, 50 per level.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_generate_benchmark_voice_data_sh.py`:

```python
"""The voice benchmark runner must match the text stratum on everything but modality."""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_benchmark_voice_data.sh"


def _dry_run() -> str:
    return subprocess.run(["bash", str(SCRIPT), "--dry-run"],
                          capture_output=True, text=True, check=True).stdout


def _blocks(out: str) -> list[str]:
    return re.findall(r"(meta = generate_workflow_dataset\(.*?\n\))", out, re.S)


def test_script_exists():
    assert SCRIPT.exists()


def test_five_levels_of_fifty():
    blocks = _blocks(_dry_run())
    assert len(blocks) == 5
    counts = [int(re.search(r"num_samples=(\d+)", b).group(1)) for b in blocks]
    assert counts == [50, 50, 50, 50, 50]
    assert sum(counts) == 250


def test_every_call_is_voice_only():
    assert all('modality_preset="voice_only"' in b for b in _blocks(_dry_run()))


def test_no_language_argument_so_it_mixes_like_the_text_stratum():
    """The text stratum passes no language and draws en/th evenly. Matching it
    keeps modality the only difference between the strata."""
    assert all("language=" not in b for b in _blocks(_dry_run()))


def test_one_teacher_model_for_every_level():
    models = {re.search(r"teacher_model='([^']+)'", b).group(1) for b in _blocks(_dry_run())}
    assert len(models) == 1


def test_kwargs_bind_against_the_real_signature():
    from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset

    for block in _blocks(_dry_run()):
        call = ast.parse(block).body[-1].value
        kwargs = {kw.arg: ast.literal_eval(kw.value)
                  for kw in call.keywords if kw.arg != "output_dir"}
        kwargs["output_dir"] = "."
        inspect.signature(generate_workflow_dataset).bind(**kwargs)
```

- [ ] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_generate_benchmark_voice_data_sh.py -v`
Expected: FAIL on `test_script_exists`.

- [ ] **Step 3: Write the runner**

Create `scripts/generate_benchmark_voice_data.sh`. Copy the argument parsing, `.env` loading, `run` helper and `--dry-run` behaviour from `scripts/generate_voice_data.sh` (lines 57-180). Header comment must state:

- 250 conversations, 50 per level, L1 to L5.
- `modality_preset="voice_only"`, `barge_in_rate` default 0.25.
- Writes `data/output/benchmark/task_a_voice`.
- ONE teacher model for every level, named in the header. The text stratum used two (`gemini-3-flash-preview` for L1-L3, `gemini-3-5-flash` for L4-L5); that is a defect and must not be copied.
- No `language` argument — the text stratum passes none and draws English or Thai evenly per sample. Matching it keeps modality the only difference between the strata.
- Seed distinct from both the text benchmark (100) and the SFT voice batch (4242).

`chmod +x` it and confirm `git ls-files -s` shows mode 100755.

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_generate_benchmark_voice_data_sh.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Smoke the offline path**

```bash
source .venv/bin/activate && ./scripts/generate_benchmark_voice_data.sh --smoke-test --output-dir /tmp/bench_voice_smoke
source .venv/bin/activate && python -c "
import json, glob
from llm_workflow_agents.data.voice_convention import find_voice_violations
rows = [json.loads(l) for f in glob.glob('/tmp/bench_voice_smoke/*.jsonl') for l in open(f)]
bad = [r['conversation_id'] for r in rows if find_voice_violations(r['messages'], r['modality'])]
print(f'{len(rows)} rows, {len(bad)} with violations')
assert not bad
"
```
Expected: zero violations.

- [ ] **Step 6: Add the DVC stage**

In `dvc.yaml`, after `task_a_benchmark`:

```yaml
  task_a_benchmark_voice:
    desc: >-
      Voice stratum for Phase 1, 250 conversations at 50 per level. Additive:
      the text stratum at data/output/benchmark/task_a is FROZEN and must not
      be regenerated. Matched to it on size, language mix and generation
      source so modality is the only difference between the strata. Requires
      a teacher API key.
    cmd: ./scripts/generate_benchmark_voice_data.sh
    deps:
      - scripts/generate_benchmark_voice_data.sh
      - src/llm_workflow_agents/data/generate_workflows.py
      - src/llm_workflow_agents/data/voice_convention.py
      - data/templates/tool_schemas_L1_to_L5.json
    outs:
      - data/output/benchmark/task_a_voice
```

Do NOT modify the `task_a_benchmark` stage.

- [ ] **Step 7: Check the pipeline parses**

Run: `source .venv/bin/activate && python -c "import yaml; d=yaml.safe_load(open('dvc.yaml')); assert 'task_a_benchmark_voice' in d['stages']; print(sorted(d['stages']))"`
Expected: the stage list prints and includes the new stage.

- [ ] **Step 8: Commit**

```bash
git add scripts/generate_benchmark_voice_data.sh dvc.yaml tests/unit/test_generate_benchmark_voice_data_sh.py
git commit -m "feat(scripts): voice benchmark stratum

250 conversations, 50 per level, additive to the frozen text stratum and
matched to it on size, language mix and generation source so modality is
the only difference between them.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Documentation and full verification

**Files:**
- Modify: `CLAUDE.md`, `.claude/rules/05-eval.md`, `docs/data_generation_recipes.md`

- [ ] **Step 1: Run the whole unit suite**

Run: `source .venv/bin/activate && python -m pytest tests/unit -q --continue-on-collection-errors 2>&1 | tail -3`
Report the real numbers. Every failure must belong to the two known families (trl→vllm→torchvision; bf16/no-GPU). Any other failure is a regression from this branch — stop and report it.

- [ ] **Step 2: Prove the text prompt did not move**

Run: `source .venv/bin/activate && pytest tests/unit/test_system_prompt_voice.py::test_real_text_rows_render_byte_identically -v`
Expected: PASS.

- [ ] **Step 3: Prove the pinned held-out gate still holds**

```bash
source .venv/bin/activate && python scripts/materialize_dvc_lineage.py --dir-hash 6bb5eb6f7c48356ca05078c537ae68b1 --out /tmp/v1_splits
source .venv/bin/activate && python scripts/build_heldout_clean_set.py \
    --candidate-split data/output/sft/task_a_splits/test.jsonl \
    --exclusion-split /tmp/v1_splits/train.jsonl \
    --exclusion-split /tmp/v1_splits/validation.jsonl \
    --out-dir /tmp/heldout_task7 --expect-clean 206 \
    --verify-against runs/audit/heldout_ckpt1767_v2corpus.json
```
Expected: `278 candidates`, `206 clean`, `72 excluded`, `[verify] OK — 206/206`.

- [ ] **Step 4: Write the documentation**

`CLAUDE.md` — extend risk R20 (do not add a new number) with: the system prompt now selects voice mode via `build_enriched_system_prompt`; the rules live once in `render_voice_format_rules()`; Phase 1 quality blends the two strata with `DEFAULT_VOICE_WEIGHT = 0.30` and reduces exactly to the text score when no voice corpus exists; and the benchmark text stratum is FROZEN.

Add a new risk entry recording the pre-existing provenance defect: `dvc.yaml`'s `task_a_benchmark` stage claims 1,000 placeholder conversations while the artifact holds 258 teacher-generated ones from two models, so running that stage destroys the data behind the current Cat A ranking. Also record that `configs/benchmark/selection_weights.yaml` has no code consumer.

`.claude/rules/05-eval.md` — under the composite section, document `blend_modality_scores`, the three chunk diagnostics, and the rule that guardrails never enter the composite.

`docs/data_generation_recipes.md` — add the voice benchmark stratum to the generation order.

Put no test counts in any of these files; they go stale immediately.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md .claude/rules/05-eval.md docs/data_generation_recipes.md
git commit -m "docs: record the voice mode switch, blended scoring and the benchmark provenance defect

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## What this plan does NOT do

- **It does not fix `dvc.yaml`'s `task_a_benchmark` stage description.** That defect is pre-existing and documented in Task 7. Fixing it means deciding whether the stage should describe the teacher-generated reality or the artifact should be regenerated — a decision with a 258-conversation blast radius, and not this plan's to make.
- **It does not wire `configs/benchmark/selection_weights.yaml` into anything.** No code reads it today. Adding the voice weight there would create a knob that does nothing.
- **It does not run the 250-conversation teacher generation.** That costs API budget. Run it after this merges, then check the batch before use.
- **It does not re-run Phase 1.** Blending changed what the composite measures, so a comparable ranking needs all 15 candidates re-scored. That is a separate, deliberate spend.
