# Voice Conversation Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Task A teacher data generator produce voice conversations — assistant turns split into `<S>…</S>` chunks, short enough for text-to-speech — alongside the written conversations it already produces.

**Architecture:** A `modality` field ("text" | "voice") on each generated sample, drawn from a preset table exactly like the three preset tables already in `generate_workflows.py`. A new `voice_convention.py` module holds the format rules and plugs into the existing repair loop. A per-message `loss` flag keeps the orchestrator-written `<unspoken>` marker out of the training targets.

**Tech Stack:** Python 3.12, pytest, DVC. No new dependencies. Run every command through `source .venv/bin/activate &&`.

**Spec:** `docs/superpowers/specs/2026-08-20-voice-conversation-generation-design.md`

## Global Constraints

- Use `uv`, never `pip`, for any dependency work.
- Prefix every Python command with `source .venv/bin/activate &&`.
- The `default` modality preset must consume no randomness. A shifted random stream invalidates every existing config.
- The existing 5,549 text conversations must not change.
- Chunk limit: target 100 characters, violation above 160.
- Turn limit: target 3 chunks, violation above 5.
- The arrow in a state marker may be `→` or `->`. Handle both, as `state_convention.py` does.
- Write the test first. Run it. Watch it fail. Then implement.
- Branch: `feat/voice-conversation-generation`. It exists and holds the spec commit.

---

### Task 1: The voice format checker

Build the module that decides whether a turn obeys the format. Everything else depends on it.

**Files:**
- Create: `src/llm_workflow_agents/data/voice_convention.py`
- Test: `tests/unit/test_voice_convention.py`

**Interfaces:**
- Consumes: `_STATE_RE` from `llm_workflow_agents.data._workflow_script`.
- Produces:
  - `CHUNK_TARGET_CHARS = 100`, `CHUNK_MAX_CHARS = 160`, `TURN_TARGET_CHUNKS = 3`, `TURN_MAX_CHUNKS = 5`
  - `iter_chunks(text: str) -> list[str]`
  - `strip_voice_markup(text: str) -> str`
  - `find_voice_violations(messages: list[dict[str, Any]], modality: str) -> list[str]`
  - `ACKNOWLEDGEMENTS: dict[str, tuple[str, ...]]`

- [ ] **Step 1: Write the failing tests for chunk parsing and stripping**

Create `tests/unit/test_voice_convention.py`:

```python
"""Tests for the voice conversation format convention."""

from __future__ import annotations

from llm_workflow_agents.data.voice_convention import (
    ACKNOWLEDGEMENTS,
    CHUNK_MAX_CHARS,
    TURN_MAX_CHUNKS,
    find_voice_violations,
    iter_chunks,
    strip_voice_markup,
)


def test_iter_chunks_returns_each_chunk_in_order():
    assert iter_chunks("<S>one</S><S>two</S>") == ["one", "two"]


def test_iter_chunks_returns_empty_list_when_no_chunks():
    assert iter_chunks("plain text") == []


def test_strip_voice_markup_deletes_chunk_tags():
    assert strip_voice_markup("<S>one</S><S>two</S>") == "onetwo"


def test_strip_voice_markup_deletes_end_conversation():
    assert strip_voice_markup("<S>bye</S>[END_CONVERSATION]") == "bye"


def test_strip_voice_markup_deletes_unspoken_marker():
    assert strip_voice_markup("<S>hel<unspoken>lo</S>") == "hello"


def test_strip_voice_markup_leaves_plain_text_alone():
    assert strip_voice_markup("plain text") == "plain text"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_workflow_agents.data.voice_convention'`

- [ ] **Step 3: Write the module with the parsing helpers only**

Create `src/llm_workflow_agents/data/voice_convention.py`:

```python
"""Single source of truth for the voice conversation format.

Convention: an assistant turn in a voice conversation splits its spoken text
into <S>...</S> chunks. The [STATE: X -> Y] marker and every <tool_call> block
sit OUTSIDE the chunks, because the agent never speaks them. A terminal turn
ends with [END_CONVERSATION], also outside the chunks.

See docs/superpowers/specs/2026-08-20-voice-conversation-generation-design.md.

Pure stdlib, no heavy imports, so generate_workflows, data_validator, and the
eval harness can all import this without import cycles. Mirrors the posture of
state_convention.py.
"""

from __future__ import annotations

import re
from typing import Any

from llm_workflow_agents.data._workflow_script import _STATE_RE

#: Target chunk length. A longer chunk delays audio playback.
CHUNK_TARGET_CHARS = 100
#: A chunk longer than this is a violation. The two reference prompts peak at 117.
CHUNK_MAX_CHARS = 160
#: Target chunk count for one assistant turn.
TURN_TARGET_CHUNKS = 3
#: A turn with more chunks than this is a violation. The reference prompts peak at 5.
TURN_MAX_CHUNKS = 5

_CHUNK_RE = re.compile(r"<S>(.*?)</S>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_END_MARKER = "[END_CONVERSATION]"
_UNSPOKEN_MARKER = "<unspoken>"

#: Openers a recovery turn may use after an interruption. The Thai entries come
#: from 4_barge_in_block in data/templates/monomax_prompt_parts.json and
#: data/templates/oceanlife_prompt_parts.json. The teacher prompt states the
#: same list, so the checker and the teacher model never disagree.
ACKNOWLEDGEMENTS: dict[str, tuple[str, ...]] = {
    "th": (
        "ขอโทษที่พูดแทรก",
        "รับทราบ",
        "โทษทีนะ",
        "อ๋อ เข้าใจแล้ว",
        "ได้เลย",
    ),
    "en": (
        "Sorry to interrupt",
        "Understood",
        "Got it",
        "Of course",
        "Right",
    ),
}


def iter_chunks(text: str) -> list[str]:
    """Return the spoken chunks of one turn, in order."""
    return _CHUNK_RE.findall(text)


def strip_voice_markup(text: str) -> str:
    """Delete every voice marker, leaving the spoken words.

    Used by the held-out audit and the reward functions. Both compare a voice
    completion against a text-convention ground truth. The markup must not
    count as a difference.
    """
    out = text.replace("<S>", "").replace("</S>", "")
    out = out.replace(_END_MARKER, "").replace(_UNSPOKEN_MARKER, "")
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Write the failing tests for the five format rules**

Append to `tests/unit/test_voice_convention.py`:

```python
def _voice_turn(content: str) -> list[dict]:
    return [
        {"role": "user", "content": "สวัสดีค่ะ"},
        {"role": "assistant", "content": content},
    ]


def test_conforming_voice_turn_has_no_violations():
    msgs = _voice_turn("[STATE: GREET → GREET]\n<S>สวัสดีค่ะ</S><S>ยินดีให้บริการค่ะ</S>")
    assert find_voice_violations(msgs, "voice") == []


def test_rule_1_state_marker_inside_a_chunk_is_a_violation():
    msgs = _voice_turn("<S>[STATE: GREET → GREET]</S><S>สวัสดีค่ะ</S>")
    assert any("state marker" in v for v in find_voice_violations(msgs, "voice"))


def test_rule_2_tool_call_inside_a_chunk_is_a_violation():
    msgs = _voice_turn(
        '[STATE: GREET → GREET]\n'
        '<S>สักครู่นะคะ<tool_call>{"name": "f", "arguments": {}}</tool_call></S>'
    )
    assert any("tool call" in v for v in find_voice_violations(msgs, "voice"))


def test_rule_3_spoken_text_outside_a_chunk_is_a_violation():
    msgs = _voice_turn("[STATE: GREET → GREET]\nสวัสดีค่ะ<S>ยินดีค่ะ</S>")
    assert any("outside" in v for v in find_voice_violations(msgs, "voice"))


def test_rule_4_unbalanced_chunk_tags_are_a_violation():
    msgs = _voice_turn("[STATE: GREET → GREET]\n<S>สวัสดีค่ะ<S>ยินดีค่ะ</S>")
    assert any("balanced" in v for v in find_voice_violations(msgs, "voice"))


def test_rule_5_end_conversation_with_a_tool_call_is_a_violation():
    msgs = _voice_turn(
        '[STATE: GREET → GREET]\n<S>บายค่ะ</S>\n'
        '<tool_call>{"name": "f", "arguments": {}}</tool_call>[END_CONVERSATION]'
    )
    assert any("END_CONVERSATION" in v for v in find_voice_violations(msgs, "voice"))


def test_assistant_turn_with_no_chunks_at_all_is_a_violation():
    msgs = _voice_turn("[STATE: GREET → GREET]\nสวัสดีค่ะ")
    assert find_voice_violations(msgs, "voice") != []
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v -k "rule or conforming or no_chunks"`
Expected: FAIL with `ImportError: cannot import name 'find_voice_violations'`

- [ ] **Step 7: Implement the five format rules**

Append to `src/llm_workflow_agents/data/voice_convention.py`:

```python
def _check_voice_turn(content: str, turn_index: int) -> list[str]:
    """Return every format violation in one voice assistant turn."""
    violations: list[str] = []
    where = f"assistant turn {turn_index}"

    # Rule 4 first: the other rules read chunk boundaries, so unbalanced tags
    # would make their messages meaningless.
    if content.count("<S>") != content.count("</S>"):
        violations.append(
            f"{where}: chunk tags are not balanced "
            f"({content.count('<S>')} <S> vs {content.count('</S>')} </S>); "
            f"every <S> needs one </S>"
        )
        return violations

    chunks = iter_chunks(content)
    if not chunks:
        violations.append(
            f"{where}: holds no <S>...</S> chunk; every spoken word in a voice "
            f"conversation must sit inside a chunk"
        )
        return violations

    joined = "".join(chunks)

    # Rule 1.
    if _STATE_RE.search(joined):
        violations.append(
            f"{where}: the state marker sits inside a chunk; it is never "
            f"spoken, so put it on the first line outside every <S>"
        )
    # Rule 2.
    if _TOOL_CALL_RE.search(joined):
        violations.append(
            f"{where}: a tool call sits inside a chunk; it is never spoken, "
            f"so put it outside every <S>"
        )
    # Rule 5.
    if _END_MARKER in content and _TOOL_CALL_RE.search(content):
        violations.append(
            f"{where}: holds both [END_CONVERSATION] and a tool call; a turn "
            f"that calls a tool must not end the conversation"
        )
    if _END_MARKER in joined:
        violations.append(
            f"{where}: [END_CONVERSATION] sits inside a chunk; it is never "
            f"spoken, so put it after the last </S>"
        )

    # Rule 3: delete the markers that are allowed outside a chunk, then delete
    # the chunks themselves. Anything left is unspoken text the agent would
    # never say aloud.
    remainder = _STATE_RE.sub("", content)
    remainder = _TOOL_CALL_RE.sub("", remainder)
    remainder = remainder.replace(_END_MARKER, "").replace(_UNSPOKEN_MARKER, "")
    remainder = _CHUNK_RE.sub("", remainder)
    if remainder.strip():
        violations.append(
            f"{where}: text sits outside every chunk: "
            f"{remainder.strip()[:60]!r}; all spoken text goes inside <S>...</S>"
        )

    return violations


def find_voice_violations(
    messages: list[dict[str, Any]], modality: str
) -> list[str]:
    """Return every format violation in one conversation.

    For a voice conversation this checks the five format rules, both length
    limits, and the barge-in recovery. For a text conversation it checks that
    no voice marker appears at all. Both directions matter. Without the second
    the modality field is advisory.
    """
    if modality == "text":
        return _find_text_violations(messages)
    violations: list[str] = []
    turn_index = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        turn_index += 1
        violations.extend(_check_voice_turn(msg.get("content") or "", turn_index))
    return violations


def _find_text_violations(messages: list[dict[str, Any]]) -> list[str]:
    """Return a violation for every voice marker found in a text conversation."""
    violations: list[str] = []
    for index, msg in enumerate(messages):
        content = msg.get("content") or ""
        if not isinstance(content, str):
            continue
        for marker in ("<S>", _END_MARKER, _UNSPOKEN_MARKER):
            if marker in content:
                violations.append(
                    f"message {index} of a text conversation holds the voice "
                    f"marker {marker}; a text conversation carries no voice markers"
                )
    return violations
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 9: Write the failing tests for the length limits and the text direction**

Append to `tests/unit/test_voice_convention.py`:

```python
def test_chunk_at_the_limit_is_accepted():
    body = "ก" * CHUNK_MAX_CHARS
    msgs = _voice_turn(f"[STATE: A → A]\n<S>{body}</S>")
    assert find_voice_violations(msgs, "voice") == []


def test_chunk_one_over_the_limit_is_a_violation():
    body = "ก" * (CHUNK_MAX_CHARS + 1)
    msgs = _voice_turn(f"[STATE: A → A]\n<S>{body}</S>")
    assert any("too long" in v for v in find_voice_violations(msgs, "voice"))


def test_turn_at_the_chunk_limit_is_accepted():
    body = "<S>ok</S>" * TURN_MAX_CHUNKS
    msgs = _voice_turn(f"[STATE: A → A]\n{body}")
    assert find_voice_violations(msgs, "voice") == []


def test_turn_one_over_the_chunk_limit_is_a_violation():
    body = "<S>ok</S>" * (TURN_MAX_CHUNKS + 1)
    msgs = _voice_turn(f"[STATE: A → A]\n{body}")
    assert any("too many chunks" in v for v in find_voice_violations(msgs, "voice"))


def test_text_conversation_with_a_chunk_tag_is_a_violation():
    msgs = _voice_turn("[STATE: A → A]\n<S>hello</S>")
    assert find_voice_violations(msgs, "text") != []


def test_text_conversation_with_end_conversation_is_a_violation():
    msgs = _voice_turn("[STATE: A → A]\nhello[END_CONVERSATION]")
    assert find_voice_violations(msgs, "text") != []


def test_plain_text_conversation_has_no_violations():
    msgs = _voice_turn("[STATE: A → A]\nHello, how can I help you today?")
    assert find_voice_violations(msgs, "text") == []


def test_acknowledgements_cover_both_languages():
    assert set(ACKNOWLEDGEMENTS) == {"th", "en"}
    assert all(ACKNOWLEDGEMENTS[lang] for lang in ACKNOWLEDGEMENTS)
```

- [ ] **Step 10: Run the tests to verify the four length tests fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v -k "limit"`
Expected: FAIL. The two "one over" tests fail because no length check exists yet.

- [ ] **Step 11: Add the length checks**

In `_check_voice_turn`, insert this block immediately after `joined = "".join(chunks)`:

```python
    if len(chunks) > TURN_MAX_CHUNKS:
        violations.append(
            f"{where}: holds {len(chunks)} chunks, too many chunks for one "
            f"spoken reply; keep a turn to {TURN_TARGET_CHUNKS} chunks and "
            f"never more than {TURN_MAX_CHUNKS}"
        )
    for position, chunk in enumerate(chunks, start=1):
        if len(chunk) > CHUNK_MAX_CHARS:
            violations.append(
                f"{where}, chunk {position}: {len(chunk)} characters is too "
                f"long for one spoken chunk; keep a chunk to "
                f"{CHUNK_TARGET_CHARS} characters and never more than "
                f"{CHUNK_MAX_CHARS}"
            )
```

- [ ] **Step 12: Run the whole test file**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v`
Expected: PASS, 21 tests.

- [ ] **Step 13: Commit**

```bash
git add src/llm_workflow_agents/data/voice_convention.py tests/unit/test_voice_convention.py
git commit -m "feat(data): voice format convention module

Five format rules plus both length limits, checked in both directions.
Limits measured from the two production reference prompts: chunk p90 85
chars, max 117; turn median 2 chunks, max 5.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The barge-in recovery check

Add the fourth part of `find_voice_violations`. Split from Task 1 because a reviewer can accept the format rules and reject the recovery rules.

**Files:**
- Modify: `src/llm_workflow_agents/data/voice_convention.py`
- Test: `tests/unit/test_voice_convention.py`

**Interfaces:**
- Consumes: `ACKNOWLEDGEMENTS`, `find_voice_violations` from Task 1.
- Produces: `find_voice_violations` now also checks barge-in. Its signature does not change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_voice_convention.py`:

```python
def _barge_in_conversation(recovery: str) -> list[dict]:
    return [
        {"role": "user", "content": "สวัสดีค่ะ"},
        {
            "role": "assistant",
            "content": "[STATE: A → A]\n<S>ราคาแพ็คเกจ<unspoken>อยู่ที่ 5999 บาทค่ะ</S>",
        },
        {"role": "user", "content": "เดี๋ยวก่อนค่ะ"},
        {"role": "assistant", "content": recovery},
    ]


def test_valid_barge_in_recovery_has_no_violations():
    msgs = _barge_in_conversation(
        "[STATE: A → A]\n<S>ขอโทษที่พูดแทรกนะคะ</S><S>ราคาอยู่ที่ 5999 บาทค่ะ</S>"
    )
    assert find_voice_violations(msgs, "voice") == []


def test_recovery_without_an_acknowledgement_is_a_violation():
    msgs = _barge_in_conversation("[STATE: A → A]\n<S>ราคาอยู่ที่ 5999 บาทค่ะ</S>")
    assert any("acknowledgement" in v for v in find_voice_violations(msgs, "voice"))


def test_recovery_that_advances_the_state_is_a_violation():
    msgs = _barge_in_conversation(
        "[STATE: A → B]\n<S>ขอโทษที่พูดแทรกนะคะ</S><S>ราคาอยู่ที่ 5999 บาทค่ะ</S>"
    )
    assert any("advances" in v for v in find_voice_violations(msgs, "voice"))


def test_two_unspoken_markers_are_a_violation():
    msgs = _barge_in_conversation(
        "[STATE: A → A]\n<S>ขอโทษที่พูดแทรกนะคะ<unspoken>ค่ะ</S>"
    )
    assert any("exactly once" in v for v in find_voice_violations(msgs, "voice"))


def test_unspoken_marker_in_the_last_turn_is_a_violation():
    msgs = [
        {"role": "user", "content": "สวัสดีค่ะ"},
        {"role": "assistant", "content": "[STATE: A → A]\n<S>ราคา<unspoken>ค่ะ</S>"},
    ]
    assert any("last" in v for v in find_voice_violations(msgs, "voice"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v -k "barge or recovery or unspoken"`
Expected: FAIL. Four of the five fail because no barge-in check exists.

- [ ] **Step 3: Implement the barge-in check**

Append to `src/llm_workflow_agents/data/voice_convention.py`:

```python
def _all_acknowledgements() -> tuple[str, ...]:
    out: list[str] = []
    for openers in ACKNOWLEDGEMENTS.values():
        out.extend(openers)
    return tuple(out)


def find_barge_in_violations(messages: list[dict[str, Any]]) -> list[str]:
    """Check the four facts that make a barge-in recovery well formed.

    1. The marker appears exactly once in the conversation.
    2. The marker sits in an assistant turn that is not the last turn.
    3. The next assistant turn starts with a known acknowledgement.
    4. The next assistant turn annotates the same state as the interrupted turn.

    Fact 4 has a reason. A barge-in completes nothing, so the workflow does not
    advance. A conversation with no marker is valid and returns an empty list.
    """
    marked = [
        index
        for index, msg in enumerate(messages)
        if msg.get("role") == "assistant"
        and _UNSPOKEN_MARKER in (msg.get("content") or "")
    ]
    total = sum(
        (msg.get("content") or "").count(_UNSPOKEN_MARKER)
        for msg in messages
        if msg.get("role") == "assistant"
    )
    if total == 0:
        return []
    if total > 1:
        return [
            f"the {_UNSPOKEN_MARKER} marker appears {total} times; it must "
            f"appear exactly once in a conversation"
        ]

    marker_index = marked[0]
    later = [
        index
        for index, msg in enumerate(messages)
        if index > marker_index and msg.get("role") == "assistant"
    ]
    if not later:
        return [
            f"the {_UNSPOKEN_MARKER} marker sits in the last assistant turn; "
            f"the caller interrupted, so a recovery turn must follow"
        ]

    violations: list[str] = []
    interrupted = _STATE_RE.search(messages[marker_index].get("content") or "")
    recovery_content = messages[later[0]].get("content") or ""
    recovery = _STATE_RE.search(recovery_content)

    spoken = "".join(iter_chunks(recovery_content)).lstrip()
    if not spoken.startswith(_all_acknowledgements()):
        violations.append(
            f"the recovery turn opens with {spoken[:40]!r} and not with an "
            f"acknowledgement; open it with one of {_all_acknowledgements()}"
        )
    if interrupted and recovery and recovery.group(2) != interrupted.group(1):
        violations.append(
            f"the recovery turn advances the state from "
            f"{interrupted.group(1)} to {recovery.group(2)}; a barge-in "
            f"completes nothing, so annotate "
            f"[{interrupted.group(1)} -> {interrupted.group(1)}]"
        )
    return violations
```

Then, in `find_voice_violations`, add one line immediately before `return violations`:

```python
    violations.extend(find_barge_in_violations(messages))
```

- [ ] **Step 4: Run the whole test file**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py -v`
Expected: PASS, 26 tests.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/voice_convention.py tests/unit/test_voice_convention.py
git commit -m "feat(data): barge-in recovery checks

Four mechanical facts, no semantics: one marker, not in the last turn,
recovery opens with a known acknowledgement, recovery does not advance
the state.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The modality preset and the sample field

Add the modality axis to the generator without changing any generated bytes for existing configs.

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py:137-176` (preset tables), `:758-800` (`ConversationSample`), `:803-833` (selector helpers), `:1418-1440` (signature), `:1654-1660` (draw site), `:1933` (sample construction)
- Test: `tests/unit/test_data_generation.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `MODALITY_PRESETS: dict[str, dict[str, float]]`
  - `_select_modality(rng: random.Random, distribution: dict[str, float]) -> str`
  - `ConversationSample.modality: str = "text"`
  - `generate_workflow_dataset(..., modality_preset: str = "default", barge_in_rate: float = 0.25)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data_generation.py`:

```python
class TestModality:
    """The modality axis: text or voice, drawn from a preset."""

    def test_default_preset_is_text_only(self):
        from llm_workflow_agents.data.generate_workflows import MODALITY_PRESETS

        assert MODALITY_PRESETS["default"] == {"text": 1.00, "voice": 0.00}

    def test_every_preset_sums_to_one(self):
        from llm_workflow_agents.data.generate_workflows import MODALITY_PRESETS

        for name, dist in MODALITY_PRESETS.items():
            assert abs(sum(dist.values()) - 1.0) < 1e-9, name

    def test_sample_defaults_to_text(self):
        sample = ConversationSample(
            conversation_id="x", complexity_level="L1", domain="banking",
            num_states=3, num_tools=1, chain_depth=0, workflow_graph={},
            workflow_script="", tool_schemas=[], messages=[], user_behavior="cooperative",
        )
        assert sample.modality == "text"
        assert sample.to_dict()["modality"] == "text"

    def test_unknown_preset_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="modality_preset"):
            generate_workflow_dataset(
                "L1", num_samples=1, output_dir=tmp_path, modality_preset="nope",
            )

    def test_negative_barge_in_rate_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="barge_in_rate"):
            generate_workflow_dataset(
                "L1", num_samples=1, output_dir=tmp_path, barge_in_rate=-0.1,
            )

    def test_default_preset_does_not_shift_the_random_stream(self, tmp_path):
        """The core reproducibility guard.

        Drawing a modality consumes randomness. If the default path drew one,
        every existing config would produce different output from the same
        seed. So the default path must draw nothing.
        """
        a = generate_workflow_dataset(
            "L1", num_samples=4, output_dir=tmp_path / "a", seed=42,
        )
        b = generate_workflow_dataset(
            "L1", num_samples=4, output_dir=tmp_path / "b", seed=42,
            modality_preset="default",
        )
        rows_a = [json.loads(x) for x in a.output_files[0].read_text().splitlines()]
        rows_b = [json.loads(x) for x in b.output_files[0].read_text().splitlines()]
        assert [r["messages"] for r in rows_a] == [r["messages"] for r in rows_b]
        assert all(r["modality"] == "text" for r in rows_a)

    def test_voice_only_preset_marks_every_sample_voice(self, tmp_path):
        meta = generate_workflow_dataset(
            "L1", num_samples=4, output_dir=tmp_path, seed=42,
            modality_preset="voice_only",
        )
        rows = [json.loads(x) for x in meta.output_files[0].read_text().splitlines()]
        assert all(r["modality"] == "voice" for r in rows)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py::TestModality -v`
Expected: FAIL with `ImportError: cannot import name 'MODALITY_PRESETS'`

- [ ] **Step 3: Add the preset table**

In `src/llm_workflow_agents/data/generate_workflows.py`, immediately after `INITIATION_PRESETS` (line 176):

```python
MODALITY_PRESETS: dict[str, dict[str, float]] = {
    # Text-only by default. A modality draw consumes randomness, so any other
    # default would shift every existing config's random stream and change the
    # data it reproduces from the same seed.
    "default":     {"text": 1.00, "voice": 0.00},
    "voice_mix":   {"text": 0.70, "voice": 0.30},
    "voice_heavy": {"text": 0.40, "voice": 0.60},
    "voice_only":  {"text": 0.00, "voice": 1.00},
}
```

- [ ] **Step 4: Add the selector helper**

Immediately after `_select_initiator` (line 833):

```python
def _select_modality(
    rng: random.Random,
    distribution: dict[str, float],
) -> str:
    """Select the conversation modality ('text' written | 'voice' spoken)."""
    cats = list(distribution.keys())
    weights = list(distribution.values())
    return rng.choices(cats, weights=weights, k=1)[0]
```

- [ ] **Step 5: Add the sample field**

In `ConversationSample`, after `outbound_reason: str | None = None`:

```python
    # "text" (written conversation) or "voice" (spoken, <S>-chunked). See
    # data/voice_convention.py for the voice format.
    modality: str = "text"
    # True when this conversation carries one <unspoken> barge-in. Always
    # False for a text conversation.
    barge_in: bool = False
```

And in `to_dict()`, after the `outbound_reason` entry:

```python
            "modality": self.modality,
            "barge_in": self.barge_in,
```

- [ ] **Step 6: Add the parameters and their validation**

In the `generate_workflow_dataset` signature, after `require_tool_stay: bool = True`:

```python
    modality_preset: str = "default",
    barge_in_rate: float = 0.25,
```

In the docstring's `Args:` block, after the `require_tool_stay` entry:

```
        modality_preset: Share of spoken (voice) conversations. ``"default"``
            is text-only and is the only preset that consumes no randomness,
            so it reproduces every pre-existing seed exactly. ``"voice_mix"``
            targets 30% voice, ``"voice_heavy"`` 60%, ``"voice_only"`` 100%.
            A voice conversation splits each assistant turn into <S>...</S>
            chunks; see data/voice_convention.py.
        barge_in_rate: Share of VOICE conversations that carry one <unspoken>
            barge-in and its recovery turn. Ignored for text conversations,
            which never draw it. Must be within 0.0 and 1.0.
```

With the other validators, after the `initiation_preset` check:

```python
    if modality_preset not in MODALITY_PRESETS:
        raise ValueError(
            f"Unknown modality_preset {modality_preset!r}. "
            f"Valid options: {list(MODALITY_PRESETS)}"
        )
    if not 0.0 <= barge_in_rate <= 1.0:
        raise ValueError(
            f"barge_in_rate must be within 0.0 and 1.0, got {barge_in_rate}"
        )
```

Beside the other active distributions:

```python
    active_modality_dist = MODALITY_PRESETS[modality_preset]
```

- [ ] **Step 7: Add the guarded draw**

At the draw site, immediately after the `behavior = _select_user_behavior(...)` line:

```python
        # Draw the modality ONLY for a non-default preset. The default path must
        # consume no randomness, or every existing seed reproduces different
        # data. tests/unit/test_data_generation.py guards this.
        if modality_preset == "default":
            modality = "text"
            barge_in = False
        else:
            modality = _select_modality(rng, active_modality_dist)
            barge_in = modality == "voice" and rng.random() < barge_in_rate
```

Add both to the bundle the inner function returns, beside `"behavior"`:

```python
            "modality": modality,
            "barge_in": barge_in,
```

Read them back where the other bundle values are unpacked, beside `behavior = result["behavior"]`:

```python
        modality = result["modality"]
        barge_in = result["barge_in"]
```

And pass them to `ConversationSample(...)`, after `outbound_reason=...`:

```python
            modality=modality,
            barge_in=barge_in,
```

- [ ] **Step 8: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -v`
Expected: PASS, including the pre-existing seed-determinism test at line 116.

- [ ] **Step 9: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): modality axis on generated conversations

MODALITY_PRESETS beside the three existing preset tables. The default
preset draws nothing, so every pre-existing seed reproduces byte for
byte; a test guards that.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Teach the teacher model the voice format

Give the teacher model the rules, and reject its output when it breaks them.

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py:1138-1176` (`_TEACHER_SYSTEM_PROMPT`), `:1196-1252` (`_build_teacher_prompt`), `:1254-1291` (`_RICH_PROMPT_SYSTEM`), `:1357-1416` (`_generate_teacher_conversation`), `:1755-1772` (`_find_violations`)
- Test: `tests/unit/test_data_generation.py`

**Interfaces:**
- Consumes: `find_voice_violations` (Task 1), `modality` and `barge_in` (Task 3).
- Produces:
  - `_teacher_system_prompt(modality: str) -> str`
  - `_rich_prompt_system(modality: str) -> str`
  - `_build_teacher_prompt(..., modality: str = "text", barge_in: bool = False)`
  - `_generate_teacher_conversation(..., modality: str = "text", barge_in: bool = False)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data_generation.py`:

```python
class TestVoiceTeacherPrompt:
    """The teacher model must be told the voice rules, and only for voice."""

    def test_text_system_prompt_is_unchanged(self):
        from llm_workflow_agents.data.generate_workflows import (
            _TEACHER_SYSTEM_PROMPT,
            _teacher_system_prompt,
        )

        assert _teacher_system_prompt("text") == _TEACHER_SYSTEM_PROMPT

    def test_voice_system_prompt_states_the_chunk_rule(self):
        from llm_workflow_agents.data.generate_workflows import _teacher_system_prompt

        prompt = _teacher_system_prompt("voice")
        assert "<S>" in prompt
        assert "</S>" in prompt

    def test_voice_system_prompt_keeps_markers_outside_chunks(self):
        from llm_workflow_agents.data.generate_workflows import _teacher_system_prompt

        prompt = _teacher_system_prompt("voice")
        assert "outside" in prompt.lower()

    def test_text_rich_prompt_still_forbids_voice_markers(self):
        from llm_workflow_agents.data.generate_workflows import _rich_prompt_system

        assert "Do NOT include" in _rich_prompt_system("text")

    def test_voice_rich_prompt_requires_chunked_dialogue(self):
        from llm_workflow_agents.data.generate_workflows import _rich_prompt_system

        assert "<S>" in _rich_prompt_system("voice")

    def test_voice_teacher_prompt_states_the_length_limits(self):
        from llm_workflow_agents.data.generate_workflows import _build_teacher_prompt
        from llm_workflow_agents.data.voice_convention import CHUNK_MAX_CHARS

        graph = WorkflowGraph(
            states=[WorkflowState(id="s0", name="A"), WorkflowState(id="s1", name="B")],
            transitions=[WorkflowTransition(from_state="s0", to_state="s1")],
            initial_state="s0", terminal_states=["s1"],
        )
        prompt = _build_teacher_prompt(
            graph, [], "cooperative", COMPLEXITY_SPECS[ComplexityLevel.L1], None,
            modality="voice",
        )
        assert str(CHUNK_MAX_CHARS) in prompt

    def test_text_teacher_prompt_mentions_no_voice_marker(self):
        from llm_workflow_agents.data.generate_workflows import _build_teacher_prompt

        graph = WorkflowGraph(
            states=[WorkflowState(id="s0", name="A"), WorkflowState(id="s1", name="B")],
            transitions=[WorkflowTransition(from_state="s0", to_state="s1")],
            initial_state="s0", terminal_states=["s1"],
        )
        prompt = _build_teacher_prompt(
            graph, [], "cooperative", COMPLEXITY_SPECS[ComplexityLevel.L1], None,
        )
        assert "<S>" not in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py::TestVoiceTeacherPrompt -v`
Expected: FAIL with `ImportError: cannot import name '_teacher_system_prompt'`

- [ ] **Step 3: Add the voice rules block and the system-prompt selector**

In `generate_workflows.py`, immediately after the `_TEACHER_SYSTEM_PROMPT` constant:

```python
_VOICE_RULES = """\

VOICE MODE — this conversation is spoken aloud through a text-to-speech engine.
The orchestrator reads your output as a stream, finds each <S>...</S> chunk, and
sends it to the engine in order. Six extra rules apply:

- V1. Put the [STATE: X → Y] marker on the first line, OUTSIDE every <S>. The
  agent never speaks it.
- V2. Put every <tool_call> block OUTSIDE every <S>. The agent never speaks it.
- V3. Put every spoken word INSIDE a chunk. No spoken text may sit outside
  <S>...</S>.
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


_BARGE_IN_RULES = """\

BARGE-IN — this conversation contains exactly one interruption. Write all three
turns of it, somewhere in the middle of the call:

- B1. One assistant turn is cut off. Put the marker <unspoken> at the exact word
  where the voice stopped. Keep the rest of the turn after the marker. The
  caller never heard it.
- B2. The next user turn is the caller cutting in.
- B3. The next assistant turn opens with a short acknowledgement, then repeats
  what the caller never heard. Use one of these openers: {openers}.
- B4. That recovery turn annotates the SAME state as the interrupted turn. An
  interruption completes nothing, so the workflow does not advance.
- B5. Use the <unspoken> marker exactly once in the whole conversation, and
  never in the last assistant turn.
"""


def _teacher_system_prompt(modality: str) -> str:
    """Return the teacher system prompt for one modality.

    The text branch returns the frozen constant byte for byte. Rendering the
    voice rules here rather than appending them to the user prompt matters: the
    constant's OUTPUT FORMAT example shows unchunked content, so an appended
    block would contradict an example the teacher model can see.
    """
    if modality != "voice":
        return _TEACHER_SYSTEM_PROMPT
    from llm_workflow_agents.data.voice_convention import (
        CHUNK_MAX_CHARS,
        CHUNK_TARGET_CHARS,
        TURN_MAX_CHUNKS,
        TURN_TARGET_CHUNKS,
    )

    return _TEACHER_SYSTEM_PROMPT + _VOICE_RULES.format(
        chunk_target=CHUNK_TARGET_CHARS,
        chunk_max=CHUNK_MAX_CHARS,
        turn_target=TURN_TARGET_CHUNKS,
        turn_max=TURN_MAX_CHUNKS,
    )
```

- [ ] **Step 4: Split the rich prompt system**

Leave the `_RICH_PROMPT_SYSTEM` constant exactly as it is. Its "Do NOT include
TTS or serving markers" rule is correct for a text sample and its bytes stay
frozen. Add an override block and a selector below the constant:

```python
_RICH_VOICE_OVERRIDE = """\

VOICE MODE — this prompt drives a spoken agent. Override rule 4 above: every
quoted dialogue line MUST be split into <S>...</S> chunks at natural pause
points, exactly as a text-to-speech engine needs. Example:
"<S>สวัสดีค่ะ</S><S>ไม่ทราบว่าสะดวกสนทนาสักครู่ไหมคะ</S>"
Do include [END_CONVERSATION] after the final chunk of a closing section.
Still do NOT include <F> or [TRANSFER].
"""


def _rich_prompt_system(modality: str) -> str:
    """Return the rich-prompt authoring instructions for one modality."""
    if modality != "voice":
        return _RICH_PROMPT_SYSTEM
    return _RICH_PROMPT_SYSTEM + _RICH_VOICE_OVERRIDE
```

- [ ] **Step 5: Thread the modality through the two prompt builders**

Add `modality: str = "text"` and `barge_in: bool = False` to the end of the `_build_teacher_prompt` signature. Build the extra block before the return:

```python
    voice_line = ""
    if modality == "voice":
        from llm_workflow_agents.data.voice_convention import (
            ACKNOWLEDGEMENTS,
            CHUNK_MAX_CHARS,
            CHUNK_TARGET_CHARS,
            TURN_MAX_CHUNKS,
            TURN_TARGET_CHUNKS,
        )

        voice_line = (
            f"Conversation modality: VOICE. A text-to-speech engine speaks every "
            f"assistant turn. Split each turn into <S>...</S> chunks: at most "
            f"{TURN_MAX_CHUNKS} chunks per turn (aim for {TURN_TARGET_CHUNKS}), "
            f"at most {CHUNK_MAX_CHARS} characters per chunk (aim for "
            f"{CHUNK_TARGET_CHARS}). Keep every reply to one or two sentences.\n"
        )
        if barge_in:
            openers = ", ".join(
                f'"{o}"' for o in ACKNOWLEDGEMENTS.get(language, ACKNOWLEDGEMENTS["en"])
            )
            voice_line += _BARGE_IN_RULES.format(openers=openers)
```

Insert `f"{voice_line}"` into the returned string immediately after `f"{outbound_line}"`.

Add the same two parameters to `_generate_teacher_conversation`, pass them into `_build_teacher_prompt`, and change its `call_teacher_model` line to use the per-modality system prompt:

```python
        raw = call_teacher_model(
            teacher_model, _teacher_system_prompt(modality), user_prompt
        )
```

- [ ] **Step 6: Pass the modality at the three call sites**

`_generate_teacher_conversation` is called twice (the first attempt and the repair retry). Add `modality=modality, barge_in=barge_in` to both. `_generate_rich_system_prompt` takes a `modality` parameter and passes `_rich_prompt_system(modality)` to `call_teacher_model`; add `modality=modality` at its call site.

- [ ] **Step 7: Add the voice check to the repair loop**

In `_find_violations`, add one clause to the chain, after the `find_tool_stay_violations` clause:

```python
                        or find_voice_violations(msgs, modality)
```

Add the import at the top of the file, beside the `state_convention` import:

```python
from llm_workflow_agents.data.voice_convention import find_voice_violations
```

- [ ] **Step 8: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): teach the teacher model the voice format

Per-modality system prompts, so the voice rules never contradict the
frozen OUTPUT FORMAT example. Reverses the rich-prompt marker ban for
voice only. Voice violations ride the existing repair loop.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Voice output from the placeholder generator

The placeholder is the fallback when the teacher model fails. A voice sample that falls back must still be a voice sample.

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py:916-1137` (`_generate_placeholder_conversation`)
- Modify: `src/llm_workflow_agents/data/voice_convention.py` (adds `chunk_spoken_text`)
- Test: `tests/unit/test_data_generation.py`, `tests/unit/test_voice_convention.py`

**Interfaces:**
- Consumes: `find_voice_violations` (Task 1), `modality` (Task 3).
- Produces: `_generate_placeholder_conversation(..., modality: str = "text")`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_data_generation.py`:

```python
class TestPlaceholderVoice:
    def test_placeholder_voice_output_passes_the_voice_checker(self, tmp_path):
        """A fallback keeps its sample's modality, so it must obey the format."""
        from llm_workflow_agents.data.voice_convention import find_voice_violations

        meta = generate_workflow_dataset(
            "L3", num_samples=6, output_dir=tmp_path, seed=42,
            modality_preset="voice_only",
        )
        rows = [json.loads(x) for x in meta.output_files[0].read_text().splitlines()]
        assert rows
        for row in rows:
            assert row["modality"] == "voice"
            assert find_voice_violations(row["messages"], "voice") == []

    def test_placeholder_text_output_holds_no_voice_marker(self, tmp_path):
        from llm_workflow_agents.data.voice_convention import find_voice_violations

        meta = generate_workflow_dataset("L3", num_samples=6, output_dir=tmp_path, seed=42)
        rows = [json.loads(x) for x in meta.output_files[0].read_text().splitlines()]
        for row in rows:
            assert find_voice_violations(row["messages"], "text") == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_data_generation.py::TestPlaceholderVoice -v`
Expected: FAIL. The voice test fails because the placeholder emits unchunked prose.

- [ ] **Step 3: Add a chunking helper**

In `voice_convention.py`:

```python
def chunk_spoken_text(text: str) -> str:
    """Split plain spoken text into <S>...</S> chunks.

    Splits on sentence-ending punctuation first. A piece still above
    CHUNK_MAX_CHARS is split again on the last space before the limit, and
    failing that at the limit itself, so the result always conforms.

    Deterministic. The placeholder generator needs it, and the placeholder
    generator must stay reproducible.
    """
    pieces = [p.strip() for p in re.split(r"(?<=[.!?。ฯ])\s+", text.strip()) if p.strip()]
    if not pieces:
        pieces = [text.strip() or "..."]

    out: list[str] = []
    for piece in pieces:
        while len(piece) > CHUNK_MAX_CHARS:
            cut = piece.rfind(" ", 0, CHUNK_MAX_CHARS)
            if cut <= 0:
                cut = CHUNK_MAX_CHARS
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)

    # A turn above TURN_MAX_CHUNKS is a violation, so merge the tail into the
    # last legal chunk rather than emit an illegal turn.
    if len(out) > TURN_MAX_CHUNKS:
        head = out[: TURN_MAX_CHUNKS - 1]
        tail = " ".join(out[TURN_MAX_CHUNKS - 1 :])
        out = head + [tail[:CHUNK_MAX_CHARS]]

    return "".join(f"<S>{c}</S>" for c in out)
```

Add its test to `tests/unit/test_voice_convention.py`:

```python
def test_chunk_spoken_text_splits_on_sentences():
    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    assert chunk_spoken_text("One. Two.") == "<S>One.</S><S>Two.</S>"


def test_chunk_spoken_text_never_exceeds_the_limits():
    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    long_text = ". ".join("word " * 40 for _ in range(12))
    result = chunk_spoken_text(long_text)
    chunks = iter_chunks(result)
    assert len(chunks) <= TURN_MAX_CHUNKS
    assert all(len(c) <= CHUNK_MAX_CHARS for c in chunks)


def test_chunk_spoken_text_handles_empty_input():
    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    assert iter_chunks(chunk_spoken_text("")) == ["..."]
```

- [ ] **Step 4: Apply the helper in the placeholder generator**

Add `modality: str = "text"` to the `_generate_placeholder_conversation` signature and pass it at the two `_placeholder()` call sites.

Inside, wrap the assistant content build. The function composes each assistant turn as a state marker, then prose, then an optional tool call. Change the composition so that when `modality == "voice"` the prose passes through `chunk_spoken_text` before it is joined:

```python
    from llm_workflow_agents.data.voice_convention import chunk_spoken_text

    def _speak(prose: str) -> str:
        """Render assistant prose for this sample's modality."""
        return chunk_spoken_text(prose) if modality == "voice" else prose
```

Find every place the function builds an assistant turn:

```bash
source .venv/bin/activate && grep -n '"role": "assistant"' src/llm_workflow_agents/data/generate_workflows.py
```

At each site the content is a state marker, then prose, then an optional tool
call. Wrap only the prose in `_speak(...)`. Leave the state marker and the tool
call outside the chunks — that is rules V1 and V2, and the tests in Task 1 fail
if either ends up inside one.

Finally, append `[END_CONVERSATION]` to the terminal assistant turn when `modality == "voice"` and that turn holds no tool call.

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_voice_convention.py tests/unit/test_data_generation.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/voice_convention.py src/llm_workflow_agents/data/generate_workflows.py tests/unit/
git commit -m "feat(data): voice output from the placeholder generator

A fallback keeps its sample's modality, so a voice fallback must obey
the voice format. Deterministic chunking, so the placeholder stays
reproducible.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The per-message loss flag

Keep the orchestrator-written `<unspoken>` marker out of the training targets.

**Files:**
- Modify: `src/llm_workflow_agents/training/sft.py:119-137`, `src/llm_workflow_agents/training/grpo.py:285-291`
- Test: `tests/unit/test_response_only_loss_flag.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `render_response_only_sample` and `_load_grpo_jsonl` both honour `msg["loss"] is False`. Neither signature changes.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_response_only_loss_flag.py`:

```python
"""A message marked loss:false must never become a training target.

The orchestrator writes the <unspoken> barge-in marker into the model's own
past turn. The model does not write it. Training on that turn would teach the
model to emit the marker. See risk R15 for what a uniform edit teaches.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.training.sft import render_response_only_sample

@pytest.fixture(scope="module")
def tokenizer():
    # Reuse the repo's existing helper: it skips cleanly on an offline machine
    # instead of erroring. Qwen/Qwen2.5-0.5B-Instruct is in the local HF cache.
    from tests.unit.test_training import _load_tokenizer_or_skip

    return _load_tokenizer_or_skip("Qwen/Qwen2.5-0.5B-Instruct")


def _messages(loss_flag):
    first = {"role": "assistant", "content": "Interrupted reply"}
    if loss_flag is not None:
        first["loss"] = loss_flag
    return [
        {"role": "user", "content": "Hello"},
        first,
        {"role": "user", "content": "Wait"},
        {"role": "assistant", "content": "Sorry to interrupt. Here it is."},
    ]


def test_absent_loss_key_keeps_the_turn_as_a_target(tokenizer):
    out = render_response_only_sample(_messages(None), tokenizer, 512)
    assert any(label != -100 for label in out["labels"])


def test_loss_true_keeps_the_turn_as_a_target(tokenizer):
    absent = render_response_only_sample(_messages(None), tokenizer, 512)
    explicit = render_response_only_sample(_messages(True), tokenizer, 512)
    assert absent["labels"] == explicit["labels"]


def test_loss_false_masks_that_turn_but_not_the_next(tokenizer):
    masked = render_response_only_sample(_messages(False), tokenizer, 512)
    unmasked = render_response_only_sample(_messages(None), tokenizer, 512)
    n_masked = sum(1 for label in masked["labels"] if label != -100)
    n_unmasked = sum(1 for label in unmasked["labels"] if label != -100)
    assert 0 < n_masked < n_unmasked
```

- [ ] **Step 2: Run the tests to verify one fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_response_only_loss_flag.py -v`
Expected: FAIL on `test_loss_false_masks_that_turn_but_not_the_next`. The other two pass, because the flag is ignored today.

- [ ] **Step 3: Honour the flag in the SFT renderer**

In `src/llm_workflow_agents/training/sft.py`, replace the role test at line 133:

```python
        # A message marked loss:false stays in the prompt prefix but is never a
        # target. The orchestrator writes the <unspoken> barge-in marker into
        # the model's own past turn, so training on it would teach the model to
        # emit the marker. The key is absent from every pre-existing row, and
        # its default of True keeps those rows unchanged.
        if msg.get("role") == "assistant" and msg.get("loss", True) is not False:
            labels.extend(new)
        else:
            labels.extend([-100] * len(new))
```

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_response_only_loss_flag.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Honour the flag in the GRPO row loader**

In `src/llm_workflow_agents/training/grpo.py`, change the `valid_pairs` comprehension:

```python
            valid_pairs = [
                i for i in asst_indices
                if i > 0
                and raw_msgs[i - 1].get("role") in ("user", "system")
                # A loss:false turn stays in the prompt prefix but never
                # becomes a training row. See sft.py for why.
                and raw_msgs[i].get("loss", True) is not False
            ]
```

- [ ] **Step 6: Add the warning for the all_tokens recipe**

The `all_tokens` recipe cannot express a per-message opt-out. Where `sft.py` selects the loss mask, add:

```python
    if loss_mask == "all_tokens" and any(
        m.get("loss") is False
        for row in dataset
        for m in (row.get("messages") or [])
    ):
        logger.warning(
            "all_tokens_ignores_loss_flag",
            detail=(
                "The corpus holds messages marked loss:false, but the "
                "all_tokens recipe cannot mask one message. Those turns WILL "
                "become training targets. On voice data this teaches the model "
                "to emit the <unspoken> barge-in marker. Use response_only."
            ),
        )
```

- [ ] **Step 7: Prove the preference and mining scripts inherit the fix**

Both `scripts/build_preference_pairs.py:97` and `scripts/mine_model_negatives.py:65`
build their rows by calling `_load_grpo_jsonl`. Step 5 therefore already stops a
`loss:false` turn from becoming a chosen turn or a gold turn. Prove it rather
than assume it.

Append to `tests/unit/test_response_only_loss_flag.py`:

```python
def test_grpo_loader_skips_a_loss_false_turn(tmp_path):
    """build_preference_pairs.py and mine_model_negatives.py both read their
    rows through _load_grpo_jsonl, so this one guard covers all three."""
    import json

    from llm_workflow_agents.training.grpo import _load_grpo_jsonl

    conv = {
        "workflow_graph": {"transitions": []},
        "ground_truth": {"terminal_state": "END", "terminal_reached": True},
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "[STATE: A → A]\nkept"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "[STATE: A → A]\nskipped", "loss": False},
        ],
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(conv) + "\n")

    ds = _load_grpo_jsonl(tmp_path, split="train")
    assert len(ds) == 1
```

Run: `source .venv/bin/activate && pytest tests/unit/test_response_only_loss_flag.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 8: Run the training tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_training.py tests/unit/test_response_only_loss_flag.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/llm_workflow_agents/training/sft.py src/llm_workflow_agents/training/grpo.py tests/unit/test_response_only_loss_flag.py
git commit -m "feat(training): honour a per-message loss flag

An interrupted voice turn carries loss:false. It stays in the prompt
prefix and never becomes a target, so the model never learns to emit
the orchestrator-written <unspoken> marker. Defaults to True, so all
5,549 existing rows are unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: The flag survives the pipeline

Guard the flag against the class of bug that risk R12 records.

**Files:**
- Modify: `src/llm_workflow_agents/data/heldout_clean_set.py:34-42`
- Test: `tests/unit/test_clean_task_a_sft.py`, `tests/unit/test_heldout_clean_set.py`

**Interfaces:**
- Consumes: `strip_voice_markup` (Task 1).
- Produces: `user_turn_fingerprint` and `user_turn_prefix_fingerprints` now hash stripped text. Neither signature changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_clean_task_a_sft.py`:

```python
def test_cleaner_preserves_the_loss_flag():
    """Risk R12 made the cleaner delete an unknown role. It must not start
    deleting an unknown field."""
    from scripts.clean_task_a_sft import clean_record

    record = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "loss": False},
        ],
    }
    cleaned, reason = clean_record(record)
    assert reason is None
    assert cleaned["messages"][-1]["loss"] is False


def test_cleaner_preserves_the_modality_field():
    from scripts.clean_task_a_sft import clean_record

    record = {
        "modality": "voice",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
    }
    cleaned, _ = clean_record(record)
    assert cleaned["modality"] == "voice"
```

Append to `tests/unit/test_heldout_clean_set.py`:

```python
def test_fingerprint_ignores_voice_markup():
    """A no-op today, since only an assistant turn carries markup. It keeps the
    fingerprint correct if a user turn ever carries a marker."""
    from llm_workflow_agents.data.heldout_clean_set import user_turn_fingerprint

    plain = {"messages": [{"role": "user", "content": "hello"}]}
    marked = {"messages": [{"role": "user", "content": "<S>hello</S>"}]}
    assert user_turn_fingerprint(plain) == user_turn_fingerprint(marked)
```

- [ ] **Step 2: Run the tests to verify the fingerprint test fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_clean_task_a_sft.py tests/unit/test_heldout_clean_set.py -v -k "loss or modality or markup"`
Expected: The two cleaner tests PASS already. `test_fingerprint_ignores_voice_markup` FAILS.

- [ ] **Step 3: Strip markup before fingerprinting**

In `src/llm_workflow_agents/data/heldout_clean_set.py`, add the import and apply it in both fingerprint functions:

```python
from llm_workflow_agents.data.voice_convention import strip_voice_markup
```

In `user_turn_fingerprint` and `user_turn_prefix_fingerprints`, change the list comprehension body:

```python
        strip_voice_markup(str(m.get("content", "") or ""))
```

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_clean_task_a_sft.py tests/unit/test_heldout_clean_set.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/heldout_clean_set.py tests/unit/
git commit -m "test(data): guard the loss flag and modality through the pipeline

Both survive the cleaner today; these tests keep it that way. The
fingerprint now strips voice markup, a no-op today and correct if a
user turn ever carries a marker.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Multiple input directories in the two corpus scripts

The voice batch skips the remediation stage, so it joins the chain at cleaning. Both scripts must read two directories.

**Files:**
- Modify: `scripts/clean_task_a_sft.py:136-167`, `scripts/split_task_a_sft.py:44-47,75,116-132`
- Test: `tests/unit/test_clean_task_a_sft.py`, `tests/unit/test_split_task_a_sft.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: both scripts accept `--input-dir` more than once. One use keeps today's behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_split_task_a_sft.py`:

```python
def test_load_rows_reads_two_directories(tmp_path):
    from scripts.split_task_a_sft import _load_rows

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.jsonl").write_text('{"conversation_id": "a1"}\n')
    (b / "two.jsonl").write_text('{"conversation_id": "b1"}\n')

    rows = _load_rows([a, b])
    assert {r["conversation_id"] for r in rows} == {"a1", "b1"}


def test_load_rows_reads_one_directory(tmp_path):
    from scripts.split_task_a_sft import _load_rows

    a = tmp_path / "a"
    a.mkdir()
    (a / "one.jsonl").write_text('{"conversation_id": "a1"}\n')

    assert len(_load_rows([a])) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_split_task_a_sft.py -v -k "directories or one_directory"`
Expected: FAIL with `AttributeError: 'list' object has no attribute 'glob'`

- [ ] **Step 3: Change both scripts**

In `scripts/split_task_a_sft.py`:

```python
def _load_rows(input_dirs: list[Path]) -> list[dict]:
    files: list[Path] = []
    for input_dir in input_dirs:
        files.extend(sorted(input_dir.glob("*.jsonl")))
    if not files:
        sys.exit(f"Error: no *.jsonl files found in {input_dirs}")
```

Keep the rest of the function body as it is. Change the argument definition:

```python
        "--input-dir",
        type=Path,
        action="append",
        default=None,   # NOT default=DEFAULT_INPUT: with action="append" argparse
                        # returns the default PLUS every appended value.
        help=(
            "Directory of *.jsonl conversations. Repeat the flag to read more "
            "than one directory, for example the text corpus and the voice "
            "corpus."
        ),
```

Change `input_dir: Path = args.input_dir` to
`input_dirs: list[Path] = sorted(args.input_dir or [DEFAULT_INPUT])`. The
`or [DEFAULT_INPUT]` fallback preserves the zero-argument call form that
`scripts/run_phase2_sft.sh:89` and `README.md:519` both use. The `sorted()`
makes the result independent of flag order, and cannot change single-directory
behaviour. Then change the `is_dir()` guard to a loop over `input_dirs`, and
pass the list to `_load_rows`.

Note `clean_task_a_sft.py` was always `required=True` and has no default to
preserve; only `split_task_a_sft.py` needs the fallback.

Apply the same three changes to `scripts/clean_task_a_sft.py`: `action="append"`, a loop for the `is_dir()` guard, and a loop that extends `src_files` across the directories.

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_split_task_a_sft.py tests/unit/test_clean_task_a_sft.py -v`
Expected: PASS.

- [ ] **Step 5: Check the existing DVC commands still parse**

Run: `source .venv/bin/activate && python scripts/split_task_a_sft.py --help && python scripts/clean_task_a_sft.py --help`
Expected: Both print usage and exit 0. Confirm `--input-dir` reads as repeatable.

- [ ] **Step 6: Commit**

```bash
git add scripts/clean_task_a_sft.py scripts/split_task_a_sft.py tests/unit/
git commit -m "feat(scripts): read more than one input directory

The voice batch skips task_a_sft_remediate, whose ledger names specific
existing conversations, so it joins the chain at cleaning. One use of
--input-dir keeps today's behaviour.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: The voice generation runner and its DVC stage

Wire the generator to a runner and to the pipeline.

**Files:**
- Create: `scripts/generate_voice_data.sh`
- Modify: `scripts/generate_sft_data.sh`, `scripts/generate_sft_until_target.py:273`, `dvc.yaml`
- Test: `tests/unit/test_generate_sft_data_sh.py`, `tests/unit/test_generate_voice_data_sh.py` (create)

**Interfaces:**
- Consumes: `generate_workflow_dataset(..., modality_preset=..., barge_in_rate=...)` (Task 3).
- Produces: `data/output/sft/task_a_voice`, holding 2,400 conversations at 480 en / 1,200 th / 720 code_switch.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_generate_voice_data_sh.py`:

```python
"""The voice runner must call the generator with the right kwargs.

Mirrors tests/unit/test_generate_sft_data_sh.py, which guards the text runner
against signature drift the same way.
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_voice_data.sh"


def _dry_run() -> str:
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True, text=True, check=True,
    ).stdout


def test_script_exists_and_is_executable():
    assert SCRIPT.exists()


def test_language_legs_are_20_50_30():
    out = _dry_run()
    counts = {
        lang: sum(int(n) for n in re.findall(rf'language="{lang}".*?num_samples=(\d+)', out, re.S))
        for lang in ("en", "th", "code_switch")
    }
    total = sum(counts.values())
    assert total == 2400
    assert counts["en"] == 480
    assert counts["th"] == 1200
    assert counts["code_switch"] == 720


def test_every_call_uses_the_voice_only_preset():
    out = _dry_run()
    calls = re.findall(r"generate_workflow_dataset\((.*?)\n\)", out, re.S)
    assert calls
    assert all('modality_preset="voice_only"' in c for c in calls)


def test_kwargs_bind_against_the_real_signature():
    from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset

    out = _dry_run()
    for block in re.findall(r"(meta = generate_workflow_dataset\(.*?\n\))", out, re.S):
        call = ast.parse(block).body[-1].value
        # Exclude output_dir: it is emitted as Path('...'), a call node that
        # ast.literal_eval cannot evaluate. tests/unit/test_generate_sft_data_sh.py
        # does exactly this, then re-adds it before binding.
        kwargs = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in call.keywords
            if kw.arg != "output_dir"
        }
        kwargs["output_dir"] = "."
        inspect.signature(generate_workflow_dataset).bind(**kwargs)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_generate_voice_data_sh.py -v`
Expected: FAIL on `test_script_exists_and_is_executable`.

- [ ] **Step 3: Add the pass-through parameter to the text runner**

In `scripts/generate_sft_data.sh`, add a `--modality-preset <preset>` option defaulting to `default`, document it in the header comment beside `--initiation`, and add `modality_preset="$MODALITY_PRESET",` to each of the three `generate_workflow_dataset(` blocks at lines 258, 280, and 301.

In `scripts/generate_sft_until_target.py`, add a matching `--modality-preset` argument and pass `modality_preset=args.modality_preset` at the `generate_workflow_dataset` call on line 273.

- [ ] **Step 4: Write the voice runner**

Create `scripts/generate_voice_data.sh`, modelled on `scripts/generate_sft_data.sh`:

```bash
#!/usr/bin/env bash
# Generate voice (spoken) Task A conversations.
#
# A voice conversation splits every assistant turn into <S>...</S> chunks for a
# text-to-speech engine. See data/voice_convention.py and
# docs/superpowers/specs/2026-08-20-voice-conversation-generation-design.md.
#
# Total 2400 conversations, weighted toward the Thai voicebot deployment:
#   en 480 (20%) / th 1200 (50%) / code_switch 720 (30%)
# Split across L1-L5 with the same curriculum weights as the text corpus.
#
# Usage:
#   ./scripts/generate_voice_data.sh [OPTIONS]
#
# Options:
#   --output-dir <path>      Output directory (default: data/output/sft/task_a_voice)
#   --seed <n>               Random seed (default: 4242; differs from the text
#                            corpus seed of 42 so the two batches draw different
#                            domains and workflows)
#   --total <n>              Total conversations (default: 2400)
#   --teacher-model <name>   Teacher model (default: gemini-3.5-flash)
#   --barge-in-rate <f>      Share of voice conversations with one interruption
#                            (default: 0.25)
#   --smoke-test             Shorthand for --total 15
#   --dry-run                Print the commands without running them

set -euo pipefail
```

Copy the argument parsing, the `.env` loading, and the `run` helper from
`scripts/generate_sft_data.sh` lines 57-180. Then emit one
`generate_workflow_dataset(` block per (level, language) leg.

Fifteen legs: five levels times three languages. Use the same curriculum weights
as the text corpus, so each level takes its share of the 2,400 and each language
takes 20 / 50 / 30 of that level's share:

| Level | Weight | en | th | code_switch |
|---|---|---|---|---|
| L1 | 0.24 | 115 | 288 | 173 |
| L2 | 0.24 | 115 | 288 | 173 |
| L3 | 0.20 | 96 | 240 | 144 |
| L4 | 0.16 | 77 | 192 | 115 |
| L5 | 0.16 | 77 | 192 | 115 |

Give the last leg the rounding remainder, so the fifteen legs sum to exactly
2,400. The test in step 1 checks that sum.

Every block passes `modality_preset="voice_only"`, `barge_in_rate=$BARGE_IN_RATE`,
`require_tool_stay=True`, and `teacher_model="$TEACHER_MODEL"`.

Make it executable:

```bash
chmod +x scripts/generate_voice_data.sh
```

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_generate_voice_data_sh.py tests/unit/test_generate_sft_data_sh.py -v`
Expected: PASS.

- [ ] **Step 6: Run a smoke generation with no teacher model**

Run:
```bash
source .venv/bin/activate && ./scripts/generate_voice_data.sh --smoke-test --output-dir /tmp/voice_smoke
```
Expected: JSONL files appear. Then check the format holds:

```bash
source .venv/bin/activate && python -c "
import json, glob
from llm_workflow_agents.data.voice_convention import find_voice_violations
rows = [json.loads(l) for f in glob.glob('/tmp/voice_smoke/*.jsonl') for l in open(f)]
bad = [(r['conversation_id'], find_voice_violations(r['messages'], r['modality'])) for r in rows]
bad = [(c, v) for c, v in bad if v]
print(f'{len(rows)} rows, {len(bad)} with violations')
for c, v in bad[:5]: print(c, v[:2])
assert not bad
"
```
Expected: `15 rows, 0 with violations`.

- [ ] **Step 7: Add the DVC stage**

In `dvc.yaml`, after `task_a_sft_generate`:

```yaml
  task_a_sft_generate_voice:
    desc: >-
      Voice (spoken) Task A conversations, 2400 at 20/50/30 en/th/code_switch.
      Every assistant turn splits into <S>...</S> chunks for a text-to-speech
      engine. Skips task_a_sft_remediate: that stage replays an authoring
      ledger keyed to specific existing conversations, and this batch needs no
      repair because require_tool_stay is on at generation time. Joins the
      chain at task_a_sft_clean. Requires a teacher API key.
    cmd: ./scripts/generate_voice_data.sh
    deps:
      - scripts/generate_voice_data.sh
      - src/llm_workflow_agents/data/generate_workflows.py
      - src/llm_workflow_agents/data/voice_convention.py
      - data/templates/workflow_prompt_template.txt
      - data/templates/tool_schemas_L1_to_L5.json
    outs:
      - data/output/sft/task_a_voice
```

Then add `--input-dir data/output/sft/task_a_voice` to the `task_a_sft_clean` command, and `data/output/sft/task_a_voice` to its `deps`.

- [ ] **Step 8: Check the pipeline graph parses**

Run: `source .venv/bin/activate && python -c "import yaml; d = yaml.safe_load(open('dvc.yaml')); print(list(d['stages']))"`
Expected: The stage list prints and holds `task_a_sft_generate_voice`.

- [ ] **Step 9: Commit**

```bash
git add scripts/generate_voice_data.sh scripts/generate_sft_data.sh scripts/generate_sft_until_target.py dvc.yaml tests/unit/test_generate_voice_data_sh.py
git commit -m "feat(scripts): voice generation runner and DVC stage

2400 conversations at 20/50/30 en/th/code_switch, seeded apart from the
text corpus. The stage skips remediation and joins at cleaning.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Split the held-out set by modality

Keep the two held-out numbers separate, so cell C2's 0.7595 stays meaningful.

**Files:**
- Modify: `scripts/build_heldout_clean_set.py:120-129`, `scripts/heldout_composite_audit.py`
- Test: `tests/unit/test_heldout_clean_set.py`

**Interfaces:**
- Consumes: `modality` on each row (Task 3), `find_voice_violations` (Task 1).
- Produces: `--modality {text,voice,all}` on `build_heldout_clean_set.py`, default `all`. A `voice_format_compliance` field in the audit output.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_heldout_clean_set.py`:

```python
def test_modality_filter_keeps_only_the_named_modality():
    from llm_workflow_agents.data.heldout_clean_set import filter_by_modality

    rows = [
        {"conversation_id": "a", "modality": "text"},
        {"conversation_id": "b", "modality": "voice"},
        {"conversation_id": "c"},  # no field: a pre-existing text row
    ]
    assert [r["conversation_id"] for r in filter_by_modality(rows, "voice")] == ["b"]
    assert [r["conversation_id"] for r in filter_by_modality(rows, "text")] == ["a", "c"]
    assert len(filter_by_modality(rows, "all")) == 3


def test_missing_modality_counts_as_text():
    """Every one of the 5,549 pre-existing rows predates the field."""
    from llm_workflow_agents.data.heldout_clean_set import filter_by_modality

    assert filter_by_modality([{"conversation_id": "x"}], "text") != []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/test_heldout_clean_set.py -v -k modality`
Expected: FAIL with `ImportError: cannot import name 'filter_by_modality'`

- [ ] **Step 3: Add the filter**

In `src/llm_workflow_agents/data/heldout_clean_set.py`:

```python
def filter_by_modality(
    rows: list[dict[str, Any]], modality: str
) -> list[dict[str, Any]]:
    """Keep only the rows of one modality.

    A row with no ``modality`` field counts as text. Every one of the 5,549
    conversations that predate the field is a written conversation.
    """
    if modality == "all":
        return list(rows)
    return [r for r in rows if (r.get("modality") or "text") == modality]
```

- [ ] **Step 4: Add the flag**

In `scripts/build_heldout_clean_set.py`, beside the other arguments:

```python
    p.add_argument(
        "--modality",
        choices=("text", "voice", "all"),
        default="all",
        help=(
            "Keep only conversations of this modality. The default of 'all' "
            "keeps today's behaviour, so the pinned 206-row set of risk R17 "
            "rebuilds unchanged."
        ),
    )
```

Apply `filter_by_modality(candidates, args.modality)` immediately after the candidate rows are loaded and before any sampling.

- [ ] **Step 5: Check the pinned set still rebuilds**

This is the gate on the whole task. Run the rebuild from
`docs/cat_a_c2_heldout_result.md:241`, unchanged:

This machine has no `.venv-train`, and `build_heldout_clean_set.py` runs fine in
`.venv`. Run it there:

```bash
source .venv/bin/activate && python scripts/build_heldout_clean_set.py \
    --candidate-split data/output/sft/task_a_splits/test.jsonl \
    --exclusion-split /tmp/v1_splits/train.jsonl \
    --exclusion-split /tmp/v1_splits/validation.jsonl \
    --out-dir data/output/heldout/cat_a_v2_test_not_in_v1 \
    --expect-clean 206 \
    --verify-against runs/audit/heldout_ckpt1767_v2corpus.json
```

Step 1 of that document materializes `/tmp/v1_splits` first. Run it if the
directory is absent.

Expected: `278 candidates, 63 in v1 train + 9 in v1 val excluded, 206 clean`,
then `[verify] OK — 206/206 rows match`.

If the count is not 206, stop and revert. The `--modality` default of `all` has
failed to preserve the old path, and that breaks the only link to cell C2's
0.7595.

- [ ] **Step 6: Add the compliance guardrail to the audit**

In `scripts/heldout_composite_audit.py`, compute one extra number and report it beside the composite. Do not add it to the composite.

Two facts about that file, verified: the per-row list is named `rows`, not
`results`, and it is built at line 135 as
`rows.append({"row_index": i, "completion": comp, **comps})`. That dict does not
carry the modality. Add it there first, reading it from the source conversation
and defaulting to `"text"`:

```python
        rows.append({
            "row_index": i,
            "completion": comp,
            "modality": (conv.get("modality") or "text"),
            **comps,
        })
```

Then compute the guardrail:

```python
    # A guardrail, never a composite term. Adding a term would change what
    # cell C2's 0.7595 means.
    voice_rows = [r for r in rows if r.get("modality") == "voice"]
    if voice_rows:
        clean = sum(
            1 for r in voice_rows
            if not find_voice_violations(
                [{"role": "assistant", "content": r.get("completion", "")}], "voice"
            )
        )
        summary["voice_format_compliance"] = clean / len(voice_rows)
        summary["voice_rows"] = len(voice_rows)
```

- [ ] **Step 7: Run the tests**

Run: `source .venv/bin/activate && pytest tests/unit/test_heldout_clean_set.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/llm_workflow_agents/data/heldout_clean_set.py scripts/build_heldout_clean_set.py scripts/heldout_composite_audit.py tests/unit/test_heldout_clean_set.py
git commit -m "feat(eval): split the held-out set by modality

Two numbers, never blended: text on the pinned 206 rows, voice on the
voice slice. Voice format compliance reports as a guardrail, not as a
composite term, so 0.7595 keeps its meaning.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Full suite, documentation, and the pull request

**Files:**
- Modify: `CLAUDE.md`, `.claude/rules/02-data-generation.md`, `.claude/rules/03-training.md`

- [ ] **Step 1: Run the whole unit suite**

Run: `source .venv/bin/activate && pytest tests/unit/ -q`
Expected: PASS. The baseline before this work was 1433 passed, 0 failed. Report the new count. Any new failure is a regression from this branch.

- [ ] **Step 2: Confirm the text corpus is untouched**

Run:
```bash
source .venv/bin/activate && python -c "
import json, glob
from llm_workflow_agents.data.voice_convention import find_voice_violations
files = glob.glob('data/output/sft/task_a/*.jsonl')
rows = [json.loads(l) for f in files for l in open(f)]
bad = [r['conversation_id'] for r in rows if find_voice_violations(r['messages'], 'text')]
print(f'{len(rows)} text rows, {len(bad)} holding a voice marker')
assert not bad
"
```
Expected: `5549 text rows, 0 holding a voice marker`.

- [ ] **Step 3: Record the work in CLAUDE.md**

Add a numbered risk entry after R19, stating: the voice modality exists; the format lives in `data/voice_convention.py`; the `all_tokens` recipe cannot honour the loss flag, so `response_only` is required for voice data; the held-out audit reports text and voice separately and never blended.

Add to `.claude/rules/02-data-generation.md`, under the Task A section, a short subsection naming `modality_preset`, `barge_in_rate`, and the five format rules.

Add to `.claude/rules/03-training.md`, under the SFT section, two sentences on the `loss` flag and why `all_tokens` cannot honour it.

- [ ] **Step 4: Commit the documentation**

```bash
git add CLAUDE.md .claude/rules/02-data-generation.md .claude/rules/03-training.md
git commit -m "docs: record the voice modality

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Open the pull request**

```bash
git push -u origin feat/voice-conversation-generation
gh pr create --fill --base main
gh pr merge --auto --squash
```

- [ ] **Step 6: Watch the pull request until it merges**

Run: `gh pr checks --watch` then `gh pr view --json state`
Expected: `MERGED`. Do not report the work complete before this.

---

## What this plan does NOT do

Generating the 2,400 voice conversations costs real teacher API calls. That run is not a step here. Run it after this branch merges, then read `repair_fallbacks` and `generation_source` in the stats sidecar before merging the batch into the corpus. A high fallback rate means the teacher model failed the format, and the batch is then 2,400 deterministic placeholder rows under a voice label. Risk R15 records what a structurally uniform corpus teaches a model.

Retraining is also out of scope. So is `dvc commit` and `dvc push`, because this environment holds no `dvc` command. That is the same manual step risk R12 already carries.
