# Mining Yield Investigation + Contingent Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostic script that decomposes the 38.0% (held-out audit, TEST) vs 12.8% (first mining run, TRAIN) error-rate gap into a split effect and a classifier effect, and ship a contingent, tested-but-not-activated fix (mine negatives from the GRPO `validation` split instead of `train`) that reserves a fingerprint-partitioned slice of validation so it never overlaps with `dpo.py`'s R5 held-out reward-hacking guardrail.

**Architecture:** One new pure partitioning function (`reserve_guardrail_slice`) shared by two consumers — `scripts/mine_model_negatives.py` (excludes the reserved slice when mining from validation) and `src/llm_workflow_agents/training/dpo.py` (restricts its guardrail to the reserved slice). One new script (`scripts/investigate_mining_yield.py`) that reuses `mine_model_negatives.py`'s existing `_select_prompts`/`_classify` functions unchanged, adding two probes on top. No GPU-dependent code path in this plan is exercised at implementation time — every step is tested with a mock or a synthetic fixture standing in for the model/checkpoint.

**Tech Stack:** Python 3.12, pytest, no new third-party dependencies.

## Global Constraints

- Never generate fresh completions against `--split test` anywhere in this plan — that is the true 206-conversation held-out set, the only contamination-free measuring stick for this lineage (CLAUDE.md R18, `docs/cat_a_c2_heldout_result.md`).
- `dvc.yaml`'s `task_a_preference_model_negatives` stage `cmd` keeps `--split train` at the end of this plan. The validation-mining path must be fully built and tested but not activated by default (spec §3.3, §6).
- All fingerprint-based exclusion logic must operate on **prefix** fingerprints (`user_turn_prefix_fingerprints`), never whole-conversation fingerprints, when matched against a per-turn row's own fingerprint — a per-turn row's prompt is only a *prefix* of its conversation's user turns, so a whole-conversation fingerprint can never equal it. Getting this backwards was the exact bug fixed in commit `165f2cc` (`build_preference_pairs.py`'s first contamination-guard attempt). See Task 1.
- New code follows the existing per-script import convention: a script in `scripts/` that needs another script's functions does `sys.path.insert(0, str(Path(__file__).resolve().parent))` then a plain `from other_script import name` (see `scripts/mine_model_negatives.py:49,163` and `scripts/build_preference_pairs.py`). Tests import a `scripts/` module the same way test files already do it (see `tests/unit/test_clean_task_a_sft.py:10-11`).

---

## Task 1: `reserve_guardrail_slice()` — the shared partitioning primitive

**Files:**
- Modify: `src/llm_workflow_agents/data/heldout_clean_set.py`
- Test: `tests/unit/test_heldout_clean_set_guardrail.py` (new file)

**Interfaces:**
- Consumes: `_read_jsonl(path) -> list[dict]`, `user_turn_fingerprint(conversation) -> str`, `user_turn_prefix_fingerprints(conversation) -> set[str]` — all already defined in `heldout_clean_set.py`.
- Produces: `reserve_guardrail_slice(data_dir: Path, split: str = "validation", reserved_fraction: float = 0.2, seed: int = 42) -> set[str]`. Returns **prefix fingerprints** (not whole-conversation fingerprints) of the reserved conversations — directly comparable via `in`/`not in` against `user_turn_fingerprint({"messages": <a prompt's message list>})`, which is itself always a prefix. Task 3 and Task 6 both consume this exact return shape.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_heldout_clean_set_guardrail.py`:

```python
"""Tests for reserve_guardrail_slice — the fingerprint partition that keeps
scripts/mine_model_negatives.py's validation-split mining and dpo.py's R5
held-out guardrail from ever drawing on the same conversations.

Returns PREFIX fingerprints, not whole-conversation fingerprints, because a
per-turn row's prompt (what both consumers actually fingerprint at call time)
is only a prefix of its conversation's user turns — matching it against a
whole-conversation fingerprint would silently match nothing, exactly the bug
fixed in commit 165f2cc for the held-out contamination guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.data.heldout_clean_set import (
    reserve_guardrail_slice,
    user_turn_fingerprint,
    user_turn_prefix_fingerprints,
)


def _conv(user_turns: list[str], cid: str) -> dict:
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for u in user_turns:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": "ok"})
    return {"conversation_id": cid, "messages": msgs}


def _write_split(tmp_path: Path, conversations: list[dict], split: str = "validation") -> Path:
    path = tmp_path / f"{split}.jsonl"
    with open(path, "w") as fh:
        for c in conversations:
            fh.write(json.dumps(c) + "\n")
    return tmp_path


def test_returns_prefix_fingerprints_not_whole_conversation_ones(tmp_path):
    conv = _conv(["turn one", "turn two"], "c1")
    _write_split(tmp_path, [conv] * 10)  # all identical -> all reserved regardless of fraction
    reserved = reserve_guardrail_slice(tmp_path, reserved_fraction=0.5, seed=1)
    # A per-turn row's prompt after the FIRST user turn only ("turn one") must
    # match — that is a prefix, and prefix fingerprints are what both real
    # consumers compute against a prompt's own message list.
    prefix_fp = user_turn_fingerprint({"messages": [{"role": "user", "content": "turn one"}]})
    assert prefix_fp in reserved
    # And it must be a SUBSET of the full prefix-fingerprint expansion for
    # every reserved conversation, matching user_turn_prefix_fingerprints exactly.
    assert reserved == user_turn_prefix_fingerprints(conv)


def test_reserved_and_minable_conversations_are_disjoint(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(50)]
    _write_split(tmp_path, conversations)
    reserved = reserve_guardrail_slice(tmp_path, reserved_fraction=0.2, seed=7)

    reserved_convs = [c for c in conversations if user_turn_fingerprint(c) in
                       {fp for c2 in conversations if user_turn_fingerprint(c2) in reserved
                        for fp in [user_turn_fingerprint(c2)]}]
    # Simpler, direct check: every conversation is either fully inside
    # `reserved` (all its prefixes present) or fully outside (none present).
    for c in conversations:
        prefixes = user_turn_prefix_fingerprints(c)
        overlap = prefixes & reserved
        assert overlap == prefixes or overlap == set()


def test_reserved_fraction_is_honored_within_rounding(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(100)]
    _write_split(tmp_path, conversations)
    reserved = reserve_guardrail_slice(tmp_path, reserved_fraction=0.2, seed=3)
    reserved_conv_count = sum(
        1 for c in conversations if user_turn_fingerprint(c) in reserved
    )
    assert reserved_conv_count == 20


def test_deterministic_across_repeated_calls_with_same_seed(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(30)]
    _write_split(tmp_path, conversations)
    a = reserve_guardrail_slice(tmp_path, reserved_fraction=0.3, seed=11)
    b = reserve_guardrail_slice(tmp_path, reserved_fraction=0.3, seed=11)
    assert a == b


def test_different_seeds_give_different_partitions(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(30)]
    _write_split(tmp_path, conversations)
    a = reserve_guardrail_slice(tmp_path, reserved_fraction=0.3, seed=11)
    b = reserve_guardrail_slice(tmp_path, reserved_fraction=0.3, seed=99)
    assert a != b


def test_result_is_stable_if_the_file_is_rewritten_in_a_different_row_order(tmp_path):
    conversations = [_conv([f"q{i}"], f"c{i}") for i in range(20)]
    _write_split(tmp_path, conversations)
    a = reserve_guardrail_slice(tmp_path, reserved_fraction=0.25, seed=5)

    _write_split(tmp_path, list(reversed(conversations)))
    b = reserve_guardrail_slice(tmp_path, reserved_fraction=0.25, seed=5)
    assert a == b


def test_rejects_fraction_out_of_range(tmp_path):
    _write_split(tmp_path, [_conv(["q"], "c1")])
    with pytest.raises(ValueError, match="reserved_fraction"):
        reserve_guardrail_slice(tmp_path, reserved_fraction=0.0, seed=1)
    with pytest.raises(ValueError, match="reserved_fraction"):
        reserve_guardrail_slice(tmp_path, reserved_fraction=1.0, seed=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_heldout_clean_set_guardrail.py -v`
Expected: FAIL with `ImportError: cannot import name 'reserve_guardrail_slice'`

- [ ] **Step 3: Implement `reserve_guardrail_slice`**

In `src/llm_workflow_agents/data/heldout_clean_set.py`, add `import random` to the imports (currently `hashlib`, `json`, `Path`, `Any`), then add this function after `user_turn_prefix_fingerprints` (which it calls) and before `load_prefix_fingerprints`:

```python
def reserve_guardrail_slice(
    data_dir: Path,
    split: str = "validation",
    reserved_fraction: float = 0.2,
    seed: int = 42,
) -> set[str]:
    """Deterministically partition one split's conversations for a guardrail.

    A downstream mining process and a downstream training guardrail can both
    read from the same split (typically the GRPO ``validation`` split) without
    risking overlap: this reserves ``reserved_fraction`` of the split's
    conversations, by a stable hash of each conversation's own content (not
    file row order, which can change across a corpus regeneration), and
    returns every reserved conversation's full set of PREFIX fingerprints
    (:func:`user_turn_prefix_fingerprints`) — not whole-conversation
    fingerprints.

    Prefix fingerprints are the right return shape because every real
    consumer of this set fingerprints a per-turn row's *prompt*, which is only
    a prefix of its conversation's user turns — a per-turn row's prompt after
    turn 2 of a 5-turn conversation can never equal that conversation's whole
    fingerprint. Returning whole-conversation fingerprints here would silently
    match nothing, the exact bug commit 165f2cc fixed for the held-out
    contamination guard.

    Deterministic and order-independent: sorts fingerprints before shuffling,
    so a corpus regeneration that reorders ``<split>.jsonl`` rows without
    changing their content produces the identical partition.
    """
    if not 0.0 < reserved_fraction < 1.0:
        raise ValueError(
            f"reserved_fraction must be in (0, 1), got {reserved_fraction}"
        )
    path = Path(data_dir) / f"{split}.jsonl"
    conversations = _read_jsonl(path)

    whole_fingerprints = sorted({user_turn_fingerprint(c) for c in conversations})
    rng = random.Random(seed)
    rng.shuffle(whole_fingerprints)
    n_reserved = max(1, round(len(whole_fingerprints) * reserved_fraction))
    reserved_whole = set(whole_fingerprints[:n_reserved])

    out: set[str] = set()
    for conv in conversations:
        if user_turn_fingerprint(conv) in reserved_whole:
            out |= user_turn_prefix_fingerprints(conv)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_heldout_clean_set_guardrail.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full existing suite for this file to check nothing broke**

Run: `pytest tests/unit/test_build_heldout_clean_set.py tests/unit/test_dpo_dataset.py -v`
Expected: all pass (these files import other names from `heldout_clean_set.py`; confirms the new `import random` and function addition didn't disturb them)

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/heldout_clean_set.py tests/unit/test_heldout_clean_set_guardrail.py
git commit -m "feat(data): reserve_guardrail_slice — shared fingerprint partition for validation-split mining"
```

---

## Task 2: Regression coverage for `_classify()` and `_select_prompts()`

**Why this task exists:** `scripts/mine_model_negatives.py` has no test file today. Task 3 modifies `main()` right next to these two functions; Task 4/5 (`investigate_mining_yield.py`) import and depend on both directly. Locking in their current behavior first means a future edit that accidentally changes `_classify`'s fault-type taxonomy or `_select_prompts`'s dedup/tool-share logic fails loudly here, not silently inside a downstream script.

**Files:**
- Test: `tests/unit/test_mine_model_negatives.py` (new file)
- Read only: `scripts/mine_model_negatives.py` (no changes in this task)

**Interfaces:**
- Consumes: `_classify(completion: str, gt: dict) -> str | None` and `_select_prompts(data_dir, split, n, tool_share, seed) -> list[dict]`, both already defined in `scripts/mine_model_negatives.py`.
- Produces: nothing new — this task adds tests only.

- [ ] **Step 1: Write the tests**

Create `tests/unit/test_mine_model_negatives.py`:

```python
"""Regression coverage for scripts/mine_model_negatives.py's pure logic.

_classify() and _select_prompts() had no test file before this — both are
reused directly by scripts/investigate_mining_yield.py (Task 4/5), so their
current behavior needs to be locked in before anything downstream depends on
it, and before Task 3 edits main() next to them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from mine_model_negatives import _classify, _select_prompts  # noqa: E402


class TestClassify:
    def test_perfect_match_returns_none(self):
        gt = {
            "tool_calls": [{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            "state_sequence": [{"from": "S1", "to": "S1"}],
        }
        completion = (
            "[STATE: S1 -> S1] "
            '<tool_call>{"name": "book_flight", "arguments": {"dest": "SFO"}}</tool_call>'
        )
        assert _classify(completion, gt) is None

    def test_no_tool_call_when_one_is_expected(self):
        gt = {"tool_calls": [{"name": "book_flight", "arguments": {}}], "state_sequence": []}
        assert _classify("[STATE: S1 -> S1] Sure, one moment.", gt) == "model_no_tool_call"

    def test_wrong_tool_name(self):
        gt = {"tool_calls": [{"name": "book_flight", "arguments": {}}], "state_sequence": []}
        completion = '<tool_call>{"name": "cancel_flight", "arguments": {}}</tool_call>'
        assert _classify(completion, gt) == "model_wrong_tool"

    def test_wrong_arguments_same_tool_name(self):
        gt = {
            "tool_calls": [{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            "state_sequence": [],
        }
        completion = '<tool_call>{"name": "book_flight", "arguments": {"dest": "LAX"}}</tool_call>'
        assert _classify(completion, gt) == "model_wrong_args"

    def test_spurious_call_on_a_turn_needing_none(self):
        gt = {"tool_calls": [], "state_sequence": []}
        completion = '<tool_call>{"name": "book_flight", "arguments": {}}</tool_call>'
        assert _classify(completion, gt) == "model_wrong_tool"

    def test_state_error_when_tool_call_is_correct(self):
        gt = {
            "tool_calls": [{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            "state_sequence": [{"from": "S1", "to": "S2"}],
        }
        completion = (
            "[STATE: S1 -> S1] "
            '<tool_call>{"name": "book_flight", "arguments": {"dest": "SFO"}}</tool_call>'
        )
        assert _classify(completion, gt) == "model_state_error"

    def test_missing_state_annotation_when_one_is_expected(self):
        gt = {"tool_calls": [], "state_sequence": [{"from": "S1", "to": "S2"}]}
        assert _classify("Sure, one moment.", gt) == "model_other"

    def test_no_gt_tools_no_gt_states_and_silent_completion_is_acceptable(self):
        gt = {"tool_calls": [], "state_sequence": []}
        assert _classify("Sure, one moment.", gt) is None


class TestSelectPrompts:
    def _write_grpo_split(self, tmp_path: Path, n_tool: int, n_other: int) -> Path:
        rows = []
        for i in range(n_tool):
            rows.append(self._conv(f"tool-conv-{i}", tool_calls=[{"name": "x", "arguments": {}}]))
        for i in range(n_other):
            rows.append(self._conv(f"other-conv-{i}", tool_calls=[]))
        path = tmp_path / "train.jsonl"
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return tmp_path

    @staticmethod
    def _conv(cid: str, tool_calls: list[dict]) -> dict:
        return {
            "conversation_id": cid,
            "complexity_level": "L2",
            "domain": "banking",
            "workflow_graph": {},
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"hello from {cid}"},
                {"role": "assistant", "content": "[STATE: S1 -> S1] ok"},
            ],
            "ground_truth": {
                "state_sequence": [{"from": "S1", "to": "S1"}],
                "terminal_state": "S1",
                "tool_calls": tool_calls,
            },
        }

    def test_respects_tool_share(self, tmp_path):
        data_dir = self._write_grpo_split(tmp_path, n_tool=20, n_other=20)
        picked = _select_prompts(data_dir, "train", n=10, tool_share=0.75, seed=1)
        n_tool_picked = sum(1 for p in picked if json.loads(p["ground_truth"]).get("tool_calls"))
        assert len(picked) == 10
        assert n_tool_picked == 8  # round(10 * 0.75)

    def test_deterministic_given_a_seed(self, tmp_path):
        data_dir = self._write_grpo_split(tmp_path, n_tool=20, n_other=20)
        a = _select_prompts(data_dir, "train", n=10, tool_share=0.5, seed=42)
        b = _select_prompts(data_dir, "train", n=10, tool_share=0.5, seed=42)
        assert a == b

    def test_clamps_to_available_rows_when_n_exceeds_population(self, tmp_path):
        data_dir = self._write_grpo_split(tmp_path, n_tool=2, n_other=2)
        picked = _select_prompts(data_dir, "train", n=100, tool_share=0.75, seed=1)
        assert len(picked) == 4
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/unit/test_mine_model_negatives.py -v`
Expected: 11 passed. (These characterize *existing* behavior, so they should pass immediately — if any fails, it means this plan's understanding of `_classify`/`_select_prompts` is wrong and must be corrected before continuing, not worked around.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_mine_model_negatives.py
git commit -m "test(scripts): regression coverage for mine_model_negatives._classify/_select_prompts"
```

---

## Task 3: Wire the reserved slice into `mine_model_negatives.py --split validation`

**Files:**
- Modify: `scripts/mine_model_negatives.py`
- Test: `tests/unit/test_mine_model_negatives.py` (extend from Task 2)

**Interfaces:**
- Consumes: `reserve_guardrail_slice` from Task 1 (`llm_workflow_agents.data.heldout_clean_set`).
- Produces: `_excluded_fingerprints(data_dir: Path, split: str, heldout: list[Path], guardrail_reserved_fraction: float, guardrail_reserved_seed: int) -> set[str]` — a small new pure function factoring the exclusion-set computation out of `main()` so it is testable without argparse or a real checkpoint. `main()` calls it and two new CLI flags (`--guardrail-reserved-fraction`, `--guardrail-reserved-seed`) feed it.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_mine_model_negatives.py`:

```python
from llm_workflow_agents.data.heldout_clean_set import user_turn_fingerprint


class TestExcludedFingerprints:
    def _write_validation_split(self, tmp_path: Path, n: int) -> Path:
        rows = [TestSelectPrompts._conv(f"c{i}", tool_calls=[]) for i in range(n)]
        path = tmp_path / "validation.jsonl"
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return tmp_path

    def test_train_split_has_no_guardrail_exclusion(self, tmp_path):
        from mine_model_negatives import _excluded_fingerprints

        data_dir = self._write_validation_split(tmp_path, n=20)
        excluded = _excluded_fingerprints(
            data_dir, split="train", heldout=[],
            guardrail_reserved_fraction=0.2, guardrail_reserved_seed=42,
        )
        assert excluded == set()

    def test_validation_split_excludes_the_reserved_slice(self, tmp_path):
        from mine_model_negatives import _excluded_fingerprints

        data_dir = self._write_validation_split(tmp_path, n=20)
        excluded = _excluded_fingerprints(
            data_dir, split="validation", heldout=[],
            guardrail_reserved_fraction=0.2, guardrail_reserved_seed=42,
        )
        assert len(excluded) > 0

        from llm_workflow_agents.data.heldout_clean_set import reserve_guardrail_slice
        assert excluded == reserve_guardrail_slice(
            data_dir, split="validation", reserved_fraction=0.2, seed=42
        )

    def test_validation_split_combines_heldout_and_guardrail_exclusions(self, tmp_path):
        from mine_model_negatives import _excluded_fingerprints

        data_dir = self._write_validation_split(tmp_path, n=20)
        heldout_path = tmp_path / "heldout_test.jsonl"
        heldout_conv = TestSelectPrompts._conv("heldout-c", tool_calls=[])
        with open(heldout_path, "w") as fh:
            fh.write(json.dumps(heldout_conv) + "\n")

        excluded = _excluded_fingerprints(
            data_dir, split="validation", heldout=[heldout_path],
            guardrail_reserved_fraction=0.2, guardrail_reserved_seed=42,
        )
        heldout_fp = user_turn_fingerprint(
            {"messages": [{"role": "user", "content": "hello from heldout-c"}]}
        )
        assert heldout_fp in excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mine_model_negatives.py::TestExcludedFingerprints -v`
Expected: FAIL with `ImportError: cannot import name '_excluded_fingerprints'`

- [ ] **Step 3: Implement `_excluded_fingerprints` and wire it into `main()`**

In `scripts/mine_model_negatives.py`, change the import block (currently lines 52-55):

```python
from llm_workflow_agents.data.heldout_clean_set import (  # noqa: E402
    load_prefix_fingerprints,
    user_turn_fingerprint,
)
```

to:

```python
from llm_workflow_agents.data.heldout_clean_set import (  # noqa: E402
    load_prefix_fingerprints,
    reserve_guardrail_slice,
    user_turn_fingerprint,
)
```

Add this function after `_select_prompts` and before `_classify`:

```python
def _excluded_fingerprints(
    data_dir: Path,
    split: str,
    heldout: list[Path],
    guardrail_reserved_fraction: float,
    guardrail_reserved_seed: int,
) -> set[str]:
    """Union of the held-out contamination guard and, for --split validation
    only, the guardrail-reserved slice dpo.py's R5 callback depends on.

    Gated to validation only: mining from --split train has no overlap risk
    with the guardrail, which only ever reads from validation (dpo.py
    ::_build_heldout_callback, Task 6) — computing the reservation for train
    would just waste a file read and could never exclude anything relevant.
    """
    excluded = load_prefix_fingerprints(heldout or [])
    if split == "validation":
        excluded = excluded | reserve_guardrail_slice(
            data_dir,
            split="validation",
            reserved_fraction=guardrail_reserved_fraction,
            seed=guardrail_reserved_seed,
        )
    return excluded
```

In `main()`, add two new arguments after the existing `--heldout` argument (currently line 142):

```python
    ap.add_argument(
        "--guardrail-reserved-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of --split validation reserved for dpo.py's R5 held-out "
            "guardrail and excluded from mining. Ignored for --split train."
        ),
    )
    ap.add_argument("--guardrail-reserved-seed", type=int, default=42)
```

Replace the two lines that currently build `contaminated` (currently lines 165 and 169-173):

```python
    contaminated = load_prefix_fingerprints(args.heldout or [])
    rows = _select_prompts(
        args.data_dir, args.split, args.n_prompts, args.tool_share, args.seed
    )
    rows = [
        r
        for r in rows
        if user_turn_fingerprint({"messages": r["prompt_messages"]}) not in contaminated
    ]
```

with:

```python
    contaminated = _excluded_fingerprints(
        args.data_dir,
        args.split,
        args.heldout or [],
        args.guardrail_reserved_fraction,
        args.guardrail_reserved_seed,
    )
    rows = _select_prompts(
        args.data_dir, args.split, args.n_prompts, args.tool_share, args.seed
    )
    rows = [
        r
        for r in rows
        if user_turn_fingerprint({"messages": r["prompt_messages"]}) not in contaminated
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_mine_model_negatives.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/mine_model_negatives.py tests/unit/test_mine_model_negatives.py
git commit -m "feat(scripts): mine_model_negatives --split validation excludes the R5 guardrail's reserved slice"
```

---

## Task 4: `investigate_mining_yield.py` — Probe 1 (classifier effect, no GPU)

**Files:**
- Create: `scripts/investigate_mining_yield.py`
- Test: `tests/unit/test_investigate_mining_yield.py` (new file)

**Interfaces:**
- Consumes: `_classify` from `mine_model_negatives.py` (Task 2's coverage already locks its behavior in).
- Produces: `_gt_from_audit_row(row: dict) -> dict`, `classify_rate_from_audit_json(path: Path, tool_bearing_only: bool = True) -> dict[str, Any]`, `print_decomposition_table(train_rate, validation_probe, heldout_classify_probe, heldout_composite_rate) -> None`. Task 5 adds `classify_rate_from_split` and `main()` to this same file.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_investigate_mining_yield.py`:

```python
"""Tests for scripts/investigate_mining_yield.py.

Probe 1 (classifier effect) needs no GPU and no checkpoint: it reclassifies
completions a prior scripts/heldout_composite_audit.py run already generated,
using the exact row schema that script writes (see its `_components` /
`main()` — row_index, completion, n_gt_tools, gt_tool_calls, gt_state_sequence,
gt_terminal, composite, ...).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from investigate_mining_yield import (  # noqa: E402
    _gt_from_audit_row,
    classify_rate_from_audit_json,
    print_decomposition_table,
)


def _audit_row(
    row_index: int,
    completion: str,
    gt_tool_calls: list[dict],
    gt_state_sequence: list[dict],
    n_gt_tools: int | None = None,
) -> dict:
    return {
        "row_index": row_index,
        "completion": completion,
        "composite": 1.0,
        "state_acc": 1.0,
        "tool_f1": 1.0,
        "task": 1.0,
        "n_gt_trans": len(gt_state_sequence),
        "n_pred_trans": 0,
        "n_gt_tools": n_gt_tools if n_gt_tools is not None else len(gt_tool_calls),
        "n_pred_tools": 0,
        "gt_state_sequence": gt_state_sequence,
        "gt_tool_calls": gt_tool_calls,
        "gt_terminal": "S_END",
        "pred_trans": [],
        "pred_tools": [],
    }


def _write_audit_json(path: Path, rows: list[dict]) -> None:
    payload = {"summary": {"n_rows": len(rows)}, "rows": rows}
    path.write_text(json.dumps(payload))


class TestGtFromAuditRow:
    def test_reconstructs_the_shape_classify_expects(self):
        row = _audit_row(
            0, "completion text",
            gt_tool_calls=[{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            gt_state_sequence=[{"from": "S1", "to": "S2"}],
        )
        gt = _gt_from_audit_row(row)
        assert gt == {
            "tool_calls": [{"name": "book_flight", "arguments": {"dest": "SFO"}}],
            "state_sequence": [{"from": "S1", "to": "S2"}],
        }

    def test_handles_missing_fields_as_empty(self):
        row = {"completion": "x"}
        assert _gt_from_audit_row(row) == {"tool_calls": [], "state_sequence": []}


class TestClassifyRateFromAuditJson:
    def test_counts_wrong_rows_using_classify(self, tmp_path):
        perfect = _audit_row(
            0,
            '[STATE: S1 -> S1] <tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>',
            gt_tool_calls=[{"name": "x", "arguments": {"a": 1}}],
            gt_state_sequence=[{"from": "S1", "to": "S1"}],
        )
        wrong = _audit_row(
            1, "[STATE: S1 -> S1] Sure, one moment.",
            gt_tool_calls=[{"name": "x", "arguments": {"a": 1}}],
            gt_state_sequence=[{"from": "S1", "to": "S1"}],
        )
        path = tmp_path / "audit.json"
        _write_audit_json(path, [perfect, wrong])

        result = classify_rate_from_audit_json(path)
        assert result["n_rows"] == 2
        assert result["n_wrong"] == 1
        assert result["wrong_rate"] == 0.5
        assert result["by_kind"] == {"model_no_tool_call": 1}

    def test_filters_to_tool_bearing_rows_by_default(self, tmp_path):
        tool_bearing_wrong = _audit_row(
            0, "no call here",
            gt_tool_calls=[{"name": "x", "arguments": {}}],
            gt_state_sequence=[],
        )
        no_tool_row = _audit_row(
            1, "Sure, one moment.", gt_tool_calls=[], gt_state_sequence=[],
        )
        path = tmp_path / "audit.json"
        _write_audit_json(path, [tool_bearing_wrong, no_tool_row])

        result = classify_rate_from_audit_json(path, tool_bearing_only=True)
        assert result["n_rows"] == 1
        assert result["n_wrong"] == 1

        result_all = classify_rate_from_audit_json(path, tool_bearing_only=False)
        assert result_all["n_rows"] == 2
        assert result_all["n_wrong"] == 1

    def test_empty_rows_gives_zero_rate_not_a_crash(self, tmp_path):
        path = tmp_path / "audit.json"
        _write_audit_json(path, [])
        result = classify_rate_from_audit_json(path)
        assert result["n_rows"] == 0
        assert result["wrong_rate"] == 0.0


class TestPrintDecompositionTable:
    def test_prints_all_four_known_and_new_rows(self, capsys):
        print_decomposition_table(
            train_rate=0.128,
            validation_probe={"wrong_rate": 0.25},
            heldout_classify_probe={"wrong_rate": 0.30},
            heldout_composite_rate=0.380,
        )
        out = capsys.readouterr().out
        assert "12.8%" in out
        assert "25.0%" in out
        assert "30.0%" in out
        assert "38.0%" in out

    def test_prints_not_run_for_missing_probes(self, capsys):
        print_decomposition_table(
            train_rate=0.128,
            validation_probe=None,
            heldout_classify_probe=None,
            heldout_composite_rate=0.380,
        )
        out = capsys.readouterr().out
        assert out.count("not run") == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_investigate_mining_yield.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'investigate_mining_yield'`

- [ ] **Step 3: Create `scripts/investigate_mining_yield.py`**

```python
#!/usr/bin/env python3
"""Decompose the mining-yield gap into a split effect and a classifier effect.

`scripts/mine_model_negatives.py`'s first run found C2 wrong on 51 of 399
TRAIN prompts (12.8%). `docs/cat_a_c2_heldout_result.md`'s held-out audit
found C2 wrong on 27 of 71 tool-bearing TEST-derived rows (38.0%) — a
contamination-free set the checkpoint never trained on. CLAUDE.md's `dc84adb`
commit names two untested explanations for the gap:

  1. SPLIT EFFECT   — TRAIN rows were seen during SFT; TEST rows were not.
  2. CLASSIFIER EFFECT — mine_model_negatives.py's `_classify()` may be more
     lenient than the held-out audit's strict composite scorer.

This script measures each while holding the other fixed:

  PROBE 1 (classifier effect, no GPU) --from-audit-json PATH
    Reclassifies a stored scripts/heldout_composite_audit.py output's
    completions with `_classify()` — the SAME rows that produced 38.0%, so
    only the scoring function varies.

  PROBE 2 (split effect, needs GPU) --checkpoint PATH --split validation
    Runs mine_model_negatives.py's own `_select_prompts`/`_classify` code
    path, unchanged except for --split, directly comparable to the known
    12.8% TRAIN figure — only the split varies.

Never point --split at "test" here — see mine_model_negatives.py's identical
guard and CLAUDE.md R18. Probe 1 does not generate anything; it only
reclassifies completions a prior audit already produced, and this script
never writes a chosen/rejected-shaped pairs file, so its output can never be
mistaken for DPO training data.

Usage:
    # Cheap, no GPU:
    python scripts/investigate_mining_yield.py \\
        --from-audit-json runs/audit/heldout_c2_ckpt1767_v2corpus.json \\
        --output runs/audit/mining_yield_investigation.json

    # Needs .venv-train + GPU:
    .venv-train/bin/python scripts/investigate_mining_yield.py \\
        --from-audit-json runs/audit/heldout_c2_ckpt1767_v2corpus.json \\
        --checkpoint checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767 \\
        --split validation --output runs/audit/mining_yield_investigation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mine_model_negatives import _classify, _select_prompts  # noqa: E402

#: Known figures this script's new measurements get compared against.
_KNOWN_TRAIN_WRONG_RATE = 0.128  # dc84adb: 51/399
_KNOWN_HELDOUT_COMPOSITE_WRONG_RATE = 0.380  # docs/cat_a_c2_heldout_result.md: 27/71


def _gt_from_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the {tool_calls, state_sequence} shape _classify() expects
    from one row of a scripts/heldout_composite_audit.py output JSON.

    The audit stores GT as separate `gt_tool_calls` / `gt_state_sequence`
    fields (see that script's `_components()`), not a single `ground_truth`
    dict — this bridges the two shapes.
    """
    return {
        "tool_calls": row.get("gt_tool_calls") or [],
        "state_sequence": row.get("gt_state_sequence") or [],
    }


def classify_rate_from_audit_json(
    path: Path, tool_bearing_only: bool = True
) -> dict[str, Any]:
    """Probe 1: reclassify a stored audit's completions with _classify().

    Holds the split fixed (these are the exact rows that produced the
    composite "wrong" rate) and varies only the scoring function, isolating
    the classifier effect.
    """
    payload = json.loads(Path(path).read_text())
    rows = payload["rows"]
    if tool_bearing_only:
        rows = [r for r in rows if r.get("n_gt_tools", 0) > 0]

    wrong = 0
    by_kind: dict[str, int] = {}
    for row in rows:
        gt = _gt_from_audit_row(row)
        kind = _classify(row.get("completion", ""), gt)
        if kind is not None:
            wrong += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1

    n = len(rows)
    return {
        "probe": "classifier_effect",
        "source": str(path),
        "tool_bearing_only": tool_bearing_only,
        "n_rows": n,
        "n_wrong": wrong,
        "wrong_rate": wrong / n if n else 0.0,
        "by_kind": by_kind,
    }


def classify_rate_from_split(
    checkpoint: str,
    data_dir: Path,
    split: str,
    n_prompts: int,
    tool_share: float,
    seed: int,
    max_new_tokens: int,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Probe 2: mine_model_negatives.py's own selection + classify code path
    against `split`, unchanged except for which split it reads.

    Directly comparable to the known 12.8% TRAIN figure, which used the
    identical function with split="train" — only the split varies, isolating
    the split effect.
    """
    if split == "test":
        raise SystemExit(
            "refusing to generate against the test split — it feeds the "
            "held-out audit set, the only contamination-free measuring "
            "stick for this lineage"
        )

    from preflight_entropy_diag import _generate_for_checkpoint

    rows = _select_prompts(data_dir, split, n_prompts, tool_share, seed)
    gens = _generate_for_checkpoint(
        checkpoint=checkpoint,
        prompts=rows,
        n_completions=1,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        seed=seed,
        do_sample=False,
    )

    wrong = 0
    by_kind: dict[str, int] = {}
    n_scored = 0
    for row, comp_list in zip(rows, gens):
        completion = (comp_list[0] if comp_list else "") or ""
        gt = json.loads(row["ground_truth"])
        gold = (gt.get("messages") or [{}])[0].get("content") or ""
        if not completion.strip() or not gold.strip():
            continue
        if completion.strip() == gold.strip():
            continue
        n_scored += 1
        kind = _classify(completion, gt)
        if kind is not None:
            wrong += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1

    return {
        "probe": "split_effect",
        "split": split,
        "n_prompts_sampled": len(rows),
        "n_scored": n_scored,
        "n_wrong": wrong,
        "wrong_rate": wrong / n_scored if n_scored else 0.0,
        "by_kind": by_kind,
    }


def print_decomposition_table(
    train_rate: float,
    validation_probe: dict[str, Any] | None,
    heldout_classify_probe: dict[str, Any] | None,
    heldout_composite_rate: float,
) -> None:
    """Print the 4-row comparison table. Pure formatting: no I/O, no model."""
    rows = [
        ("train      (classify, known)", f"{train_rate:.1%}"),
        (
            "validation (classify, new)",
            f"{validation_probe['wrong_rate']:.1%}" if validation_probe else "not run",
        ),
        (
            "held-out   (classify, new)",
            f"{heldout_classify_probe['wrong_rate']:.1%}" if heldout_classify_probe else "not run",
        ),
        ("held-out   (composite, known)", f"{heldout_composite_rate:.1%}"),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-audit-json", type=Path, default=None,
        help="Probe 1: reclassify a stored heldout_composite_audit.py output.",
    )
    ap.add_argument(
        "--include-non-tool-rows", action="store_true",
        help="Probe 1: score all rows, not just the tool-bearing subset.",
    )
    ap.add_argument(
        "--checkpoint", default=None,
        help="Probe 2: run a fresh generation pass against --split.",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/output/grpo/task_a"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-prompts", type=int, default=400)
    ap.add_argument("--tool-share", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-wrong-rate", type=float, default=_KNOWN_TRAIN_WRONG_RATE)
    ap.add_argument(
        "--heldout-composite-wrong-rate", type=float,
        default=_KNOWN_HELDOUT_COMPOSITE_WRONG_RATE,
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.from_audit_json and not args.checkpoint:
        print(
            "Nothing to do: pass --from-audit-json and/or --checkpoint.",
            file=sys.stderr,
        )
        return 1

    result: dict[str, Any] = {
        "train_wrong_rate": args.train_wrong_rate,
        "heldout_composite_wrong_rate": args.heldout_composite_wrong_rate,
    }

    heldout_classify_probe = None
    if args.from_audit_json:
        heldout_classify_probe = classify_rate_from_audit_json(
            args.from_audit_json, tool_bearing_only=not args.include_non_tool_rows
        )
        result["classifier_effect_probe"] = heldout_classify_probe
        print(
            f"[probe1] classifier effect: {heldout_classify_probe['wrong_rate']:.1%} "
            f"on {heldout_classify_probe['n_rows']} rows "
            f"(composite said {args.heldout_composite_wrong_rate:.1%})"
        )

    validation_probe = None
    if args.checkpoint:
        t0 = time.time()
        validation_probe = classify_rate_from_split(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            split=args.split,
            n_prompts=args.n_prompts,
            tool_share=args.tool_share,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
        )
        result["split_effect_probe"] = validation_probe
        print(
            f"[probe2] split effect on --split {args.split}: "
            f"{validation_probe['wrong_rate']:.1%} "
            f"({time.time() - t0:.0f}s, {validation_probe['n_scored']} rows scored)"
        )

    print()
    print_decomposition_table(
        args.train_wrong_rate,
        validation_probe,
        heldout_classify_probe,
        args.heldout_composite_wrong_rate,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_investigate_mining_yield.py -v`
Expected: 7 passed

- [ ] **Step 5: Manually exercise Probe 1 end to end with a synthetic audit file**

```bash
python3 -c "
import json
rows = [
    {'row_index': 0, 'completion': '[STATE: S1 -> S1] <tool_call>{\"name\": \"x\", \"arguments\": {\"a\": 1}}</tool_call>',
     'n_gt_tools': 1, 'gt_tool_calls': [{'name': 'x', 'arguments': {'a': 1}}],
     'gt_state_sequence': [{'from': 'S1', 'to': 'S1'}]},
    {'row_index': 1, 'completion': '[STATE: S1 -> S1] Sure.',
     'n_gt_tools': 1, 'gt_tool_calls': [{'name': 'x', 'arguments': {'a': 1}}],
     'gt_state_sequence': [{'from': 'S1', 'to': 'S1'}]},
]
json.dump({'summary': {}, 'rows': rows}, open('/tmp/fake_audit.json', 'w'))
"
python3 scripts/investigate_mining_yield.py --from-audit-json /tmp/fake_audit.json
rm /tmp/fake_audit.json
```

Expected: prints `[probe1] classifier effect: 50.0% on 2 rows (composite said 38.0%)` followed by the 4-row table with `validation` and `held-out (classify, new)` both showing the Probe 1 result at 50.0% for the "new" row, and `not run`... — actually only the `held-out (classify, new)` row is populated (Probe 1); `validation (classify, new)` prints `not run` since no `--checkpoint` was given. Confirm this by reading the printed output.

- [ ] **Step 6: Commit**

```bash
git add scripts/investigate_mining_yield.py tests/unit/test_investigate_mining_yield.py
git commit -m "feat(scripts): investigate_mining_yield Probe 1 — classifier-effect reclassification, no GPU"
```

---

## Task 5: `investigate_mining_yield.py` — Probe 2 (split effect, mocked generation)

**Note:** Task 4 already wrote the full file including `classify_rate_from_split` and `main()` (they were simplest to author together with Probe 1's plumbing). This task adds test coverage for the parts Task 4 could not exercise without mocking — `classify_rate_from_split`'s generation call and skip semantics.

**Files:**
- Test: `tests/unit/test_investigate_mining_yield.py` (extend from Task 4)
- Read only: `scripts/investigate_mining_yield.py` (no changes in this task)

**Interfaces:**
- Consumes: `classify_rate_from_split` (Task 4), `_select_prompts` (mocked via `monkeypatch.setattr`).
- Produces: nothing new — test coverage only.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_investigate_mining_yield.py`:

```python
import investigate_mining_yield  # noqa: E402


class TestClassifyRateFromSplit:
    def test_refuses_test_split(self):
        import pytest

        with pytest.raises(SystemExit, match="test split"):
            investigate_mining_yield.classify_rate_from_split(
                checkpoint="fake", data_dir=Path("unused"), split="test",
                n_prompts=1, tool_share=0.75, seed=1,
                max_new_tokens=8, max_seq_length=64, batch_size=1,
            )

    def test_scores_generations_with_classify_and_skips_empty_or_identical(
        self, monkeypatch
    ):
        fake_rows = [
            {
                "prompt_messages": [{"role": "user", "content": "q1"}],
                "ground_truth": json.dumps(
                    {
                        "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
                        "state_sequence": [],
                        "messages": [{"content": "gold answer one"}],
                    }
                ),
            },
            {
                "prompt_messages": [{"role": "user", "content": "q2"}],
                "ground_truth": json.dumps(
                    {
                        "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
                        "state_sequence": [],
                        "messages": [{"content": "gold answer two"}],
                    }
                ),
            },
            {
                "prompt_messages": [{"role": "user", "content": "q3"}],
                "ground_truth": json.dumps(
                    {
                        "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
                        "state_sequence": [],
                        "messages": [{"content": "gold answer three"}],
                    }
                ),
            },
        ]

        def fake_select_prompts(data_dir, split, n, tool_share, seed):
            return fake_rows

        def fake_generate_for_checkpoint(**kwargs):
            # row 1: wrong (no tool call) -> counted
            # row 2: identical to gold -> skipped, not scored
            # row 3: empty -> skipped, not scored
            return [["not a tool call"], ["gold answer two"], [""]]

        monkeypatch.setattr(investigate_mining_yield, "_select_prompts", fake_select_prompts)
        import types

        fake_module = types.SimpleNamespace(
            _generate_for_checkpoint=fake_generate_for_checkpoint
        )
        monkeypatch.setitem(sys.modules, "preflight_entropy_diag", fake_module)

        result = investigate_mining_yield.classify_rate_from_split(
            checkpoint="fake-checkpoint",
            data_dir=Path("unused"),
            split="validation",
            n_prompts=3,
            tool_share=0.75,
            seed=1,
            max_new_tokens=8,
            max_seq_length=64,
            batch_size=1,
        )
        assert result["n_prompts_sampled"] == 3
        assert result["n_scored"] == 1
        assert result["n_wrong"] == 1
        assert result["wrong_rate"] == 1.0
        assert result["by_kind"] == {"model_no_tool_call": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_investigate_mining_yield.py::TestClassifyRateFromSplit -v`
Expected: FAIL — `test_refuses_test_split` should already PASS (the guard exists from Task 4), but `test_scores_generations_with_classify_and_skips_empty_or_identical` fails if the mock wiring doesn't match; run it first to confirm the mocks actually intercept correctly before treating this as done.

- [ ] **Step 3: Fix the test if needed, not the implementation**

If `test_scores_generations_with_classify_and_skips_empty_or_identical` fails because `from preflight_entropy_diag import _generate_for_checkpoint` inside `classify_rate_from_split` doesn't pick up the monkeypatched module, this is a test-wiring issue, not a code bug — `monkeypatch.setitem(sys.modules, "preflight_entropy_diag", fake_module)` must run **before** `classify_rate_from_split` executes its local import. Confirm the fixture order in the test above already satisfies this (the `monkeypatch.setitem` call precedes the `classify_rate_from_split(...)` call in the same test function) — it does, as written above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_investigate_mining_yield.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_investigate_mining_yield.py
git commit -m "test(scripts): investigate_mining_yield Probe 2 — mocked generation coverage"
```

---

## Task 6: Wire the reserved slice into `dpo.py`'s R5 guardrail

**Files:**
- Modify: `src/llm_workflow_agents/training/dpo.py`
- Test: `tests/unit/test_dpo_heldout_guardrail.py` (new file)

**Interfaces:**
- Consumes: `reserve_guardrail_slice` (Task 1).
- Produces: `_build_heldout_callback`'s held-out row selection changes from "first N rows of validation" to "N rows whose prompt fingerprint is in the guardrail-reserved slice."

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_dpo_heldout_guardrail.py`:

```python
"""dpo.py's R5 held-out guardrail must draw only from the fingerprint slice
scripts/mine_model_negatives.py reserves (Task 1/3) — never from rows that
could also be mined as DPO training negatives, or the guardrail is no longer
independent of what the policy was just trained to prefer.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_workflow_agents.data.heldout_clean_set import user_turn_fingerprint
from llm_workflow_agents.training.dpo import _build_heldout_callback


def _row(cid: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"hello from {cid}"}],
        "ground_truth": json.dumps({"tool_calls": [], "state_sequence": []}),
    }


class _FakeDataset(list):
    pass


def test_heldout_rows_are_restricted_to_the_reserved_fingerprints(tmp_path):
    all_rows = [_row(f"c{i}") for i in range(10)]
    fake_ds = _FakeDataset(all_rows)

    # Reserve only c0, c1, c2's prefix fingerprints.
    reserved_fps = {
        user_turn_fingerprint({"messages": r["prompt"]}) for r in all_rows[:3]
    }

    with (
        patch(
            "llm_workflow_agents.training.grpo._load_grpo_jsonl",
            return_value=fake_ds,
        ),
        patch(
            "llm_workflow_agents.training.dpo.reserve_guardrail_slice",
            return_value=reserved_fps,
        ),
    ):
        callback = _build_heldout_callback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            data_cfg={"heldout_data_source": "data/output/grpo/task_a"},
            monitoring_cfg={"eval_held_out_num_prompts": 50},
            max_new_tokens=64,
        )

    selected_fps = {
        user_turn_fingerprint({"messages": r["prompt"]}) for r in callback.held_out_rows
    }
    assert selected_fps == reserved_fps
    assert len(callback.held_out_rows) == 3


def test_respects_eval_held_out_num_prompts_cap(tmp_path):
    all_rows = [_row(f"c{i}") for i in range(10)]
    fake_ds = _FakeDataset(all_rows)
    reserved_fps = {
        user_turn_fingerprint({"messages": r["prompt"]}) for r in all_rows
    }  # all 10 reserved

    with (
        patch(
            "llm_workflow_agents.training.grpo._load_grpo_jsonl",
            return_value=fake_ds,
        ),
        patch(
            "llm_workflow_agents.training.dpo.reserve_guardrail_slice",
            return_value=reserved_fps,
        ),
    ):
        callback = _build_heldout_callback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            data_cfg={"heldout_data_source": "data/output/grpo/task_a"},
            monitoring_cfg={"eval_held_out_num_prompts": 4},
            max_new_tokens=64,
        )

    assert len(callback.held_out_rows) == 4


def test_reserved_fraction_config_key_is_passed_through(tmp_path):
    all_rows = [_row("c0")]
    fake_ds = _FakeDataset(all_rows)

    with (
        patch(
            "llm_workflow_agents.training.grpo._load_grpo_jsonl",
            return_value=fake_ds,
        ),
        patch(
            "llm_workflow_agents.training.dpo.reserve_guardrail_slice",
            return_value=set(),
        ) as mock_reserve,
    ):
        _build_heldout_callback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            data_cfg={
                "heldout_data_source": "data/output/grpo/task_a",
                "guardrail_reserved_fraction": 0.35,
                "guardrail_reserved_seed": 7,
            },
            monitoring_cfg={"eval_held_out_num_prompts": 50},
            max_new_tokens=64,
        )

    mock_reserve.assert_called_once_with(
        Path("data/output/grpo/task_a"),
        split="validation",
        reserved_fraction=0.35,
        seed=7,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_dpo_heldout_guardrail.py -v`
Expected: FAIL — `callback.held_out_rows` doesn't exist yet (currently `held_out_rows` is a closure variable, not an attribute on the callback instance), and `reserve_guardrail_slice` isn't imported/called in `dpo.py` yet.

- [ ] **Step 3: Implement the change**

In `src/llm_workflow_agents/training/dpo.py`, change the import block near the top (currently):

```python
from llm_workflow_agents.training._utils import (
    unwrap_unsloth_gemma4_kv_zero_proxy,
)
from llm_workflow_agents.training.reward_utils import (
    heldout_composite_score,
    is_reward_hacking,
)
```

to also import `reserve_guardrail_slice` and `user_turn_fingerprint`:

```python
from llm_workflow_agents.data.heldout_clean_set import (
    reserve_guardrail_slice,
    user_turn_fingerprint,
)
from llm_workflow_agents.training._utils import (
    unwrap_unsloth_gemma4_kv_zero_proxy,
)
from llm_workflow_agents.training.reward_utils import (
    heldout_composite_score,
    is_reward_hacking,
)
```

In `_build_heldout_callback`, replace the row-selection block (currently lines 531-547):

```python
    heldout_data_source = monitoring_cfg.get(
        "heldout_data_source", data_cfg.get("heldout_data_source", "data/output/grpo/task_a")
    )
    eval_held_out_every = int(monitoring_cfg.get("eval_held_out_every", 50))
    n_held_out = int(monitoring_cfg.get("eval_held_out_num_prompts", 50))

    held_out_rows: list[dict[str, Any]] = []
    try:
        val_ds = _load_grpo_jsonl(Path(heldout_data_source), split="validation")
        held_out_rows = [val_ds[i] for i in range(min(n_held_out, len(val_ds)))]
        logger.info("dpo_heldout_loaded", n_prompts=len(held_out_rows))
    except FileNotFoundError:
        logger.warning(
            "dpo_heldout_split_missing",
            note="validation split not found; held-out guardrail disabled",
            data_source=heldout_data_source,
        )
```

with:

```python
    heldout_data_source = monitoring_cfg.get(
        "heldout_data_source", data_cfg.get("heldout_data_source", "data/output/grpo/task_a")
    )
    eval_held_out_every = int(monitoring_cfg.get("eval_held_out_every", 50))
    n_held_out = int(monitoring_cfg.get("eval_held_out_num_prompts", 50))
    guardrail_reserved_fraction = float(data_cfg.get("guardrail_reserved_fraction", 0.2))
    guardrail_reserved_seed = int(data_cfg.get("guardrail_reserved_seed", 42))

    held_out_rows: list[dict[str, Any]] = []
    try:
        reserved_fps = reserve_guardrail_slice(
            Path(heldout_data_source),
            split="validation",
            reserved_fraction=guardrail_reserved_fraction,
            seed=guardrail_reserved_seed,
        )
        val_ds = _load_grpo_jsonl(Path(heldout_data_source), split="validation")
        # Restricted to the reserved slice, NOT "first N rows of validation" —
        # this is what keeps the guardrail independent of anything mined as a
        # DPO negative from the rest of validation (see the reserved-slice
        # design in docs/superpowers/specs/2026-08-17-mining-yield-investigation-design.md).
        held_out_rows = [
            row
            for row in val_ds
            if user_turn_fingerprint({"messages": row["prompt"]}) in reserved_fps
        ][:n_held_out]
        logger.info(
            "dpo_heldout_loaded",
            n_prompts=len(held_out_rows),
            reserved_fraction=guardrail_reserved_fraction,
        )
    except FileNotFoundError:
        logger.warning(
            "dpo_heldout_split_missing",
            note="validation split not found; held-out guardrail disabled",
            data_source=heldout_data_source,
        )
```

Then add `self.held_out_rows = held_out_rows` inside `_DPOHeldOutEvalCallback.__init__`. Find:

```python
    class _DPOHeldOutEvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self.metric_history: list[float] = []
            self.held_out_history: list[float] = []
```

and change to:

```python
    class _DPOHeldOutEvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self.metric_history: list[float] = []
            self.held_out_history: list[float] = []
            self.held_out_rows = held_out_rows
```

(`held_out_rows` in `_evaluate` and elsewhere already refers to the closure variable; exposing it as `self.held_out_rows` too is what Task 6's tests assert on, and costs nothing at runtime since it is the same list object.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_dpo_heldout_guardrail.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full existing dpo.py test suite to confirm nothing broke**

Run: `pytest tests/unit/test_dpo_output_dir.py tests/unit/test_dpo_config.py tests/unit/test_dpo_dataset.py -v`
Expected: 36 passed (unchanged from before this task — none of these exercise `_build_heldout_callback`)

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/training/dpo.py tests/unit/test_dpo_heldout_guardrail.py
git commit -m "feat(training): dpo.py R5 guardrail restricted to the reserved validation slice"
```

---

## Task 7: Config and DVC documentation

**Files:**
- Modify: `configs/training/dpo_cat_a.yaml`
- Modify: `dvc.yaml`

**Interfaces:**
- Consumes: nothing (documentation/config only).
- Produces: nothing new — no code, no tests. Verified by a YAML parse check and a diff read, not pytest.

- [ ] **Step 1: Document the reserved-fraction knob in `dpo_cat_a.yaml`**

In `configs/training/dpo_cat_a.yaml`, in the `data:` section, after the existing `model_negative_share: 0.2` line and its comment block, add:

```yaml
  # Fraction of the GRPO validation split reserved for the R5 held-out
  # guardrail (dpo.py::_build_heldout_callback) and excluded from mining
  # (scripts/mine_model_negatives.py --split validation). Only meaningful if
  # mining is ever switched to --split validation — see
  # docs/superpowers/specs/2026-08-17-mining-yield-investigation-design.md.
  # Defaults to 0.2 if omitted; present here for discoverability.
  guardrail_reserved_fraction: 0.2
  guardrail_reserved_seed: 42
```

- [ ] **Step 2: Verify the YAML still parses and the DPO config tests still pass**

Run: `python3 -c "import yaml; yaml.safe_load(open('configs/training/dpo_cat_a.yaml'))" && echo OK`
Expected: `OK`

Run: `pytest tests/unit/test_dpo_config.py tests/unit/test_dpo_dataset.py -v`
Expected: all pass unchanged

- [ ] **Step 3: Add the contingency note to `dvc.yaml`'s `task_a_preference_model_negatives` stage**

In `dvc.yaml`, find the `task_a_preference_model_negatives` stage's `desc` field (the block ending `...and the current bottleneck.` followed by `cmd: >-`). Append this paragraph to the end of that `desc` block, before the `cmd:` line:

```yaml
      Mining source is --split train, deliberately: the split-effect
      hypothesis for why this yields far fewer negatives than the held-out
      audit's 38.0% error rate (see docs/superpowers/specs/
      2026-08-17-mining-yield-investigation-design.md and
      scripts/investigate_mining_yield.py) has not been confirmed with a
      real measurement. If it is confirmed, switch --split to "validation"
      here — mine_model_negatives.py already supports it and excludes the
      R5 guardrail's reserved slice automatically — and re-run this stage.
```

- [ ] **Step 4: Verify `dvc.yaml` still parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('dvc.yaml'))" && echo OK`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add configs/training/dpo_cat_a.yaml dvc.yaml
git commit -m "docs(config): document guardrail_reserved_fraction and the contingent validation-split mining path"
```

---

## Final verification

- [ ] **Run the full unit suite, excluding the two files that need a real `trl` import (pre-existing environment constraint, see `tests/unit/test_grpo_config_kwargs.py` / `tests/unit/test_dpo_config_kwargs.py`):**

```bash
pytest tests/unit -q --ignore=tests/unit/test_grpo_config_kwargs.py --ignore=tests/unit/test_dpo_config_kwargs.py
```

Expected: all pass, with a count of roughly 1484 + 33 new tests from Tasks 1-6 (7 + 11 + 3 + 7 + 2 + 3), i.e. around 1517 passed, plus the same 2 pre-existing GPU-only failures in `test_training.py` that predate this plan.

- [ ] **Confirm `dvc.yaml`'s actual mining command is unchanged:**

```bash
grep -A2 "^  task_a_preference_model_negatives:" dvc.yaml | grep -A5 "cmd:"
```

Expected: still shows `--split train` — the validation path is available but not activated, per spec §3.3 and the Global Constraints above.
