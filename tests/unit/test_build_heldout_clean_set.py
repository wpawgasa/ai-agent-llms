"""Tests for the contamination-free held-out set builder.

The audit set behind docs/cat_a_corpus_v2_heldout_regression.md was built
ad-hoc and lost, leaving §7's repro pointing at a `<clean-set-dir>`
placeholder. Any number scored on a differently-built set is not comparable
to the stored v3 (0.5709) and v4 (0.5120) composites, so the selection rule
needs to be pinned down in code and tested.

The rule: a v2 test conversation is usable only if it appears nowhere in v1's
train or validation splits. Conversations are matched across corpora by a
fingerprint of their USER turns — remediation rewrote assistant annotations
and tool calls but never touched what the user said, and `conversation_id`
is not unique (ids collide across splits).
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_workflow_agents.data.heldout_clean_set import (
    build_clean_set,
    load_fingerprints,
    select_clean,
    user_turn_fingerprint,
)


def _conv(user_turns: list[str], assistant: str = "ok", cid: str = "c1") -> dict:
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for u in user_turns:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": assistant})
    return {"conversation_id": cid, "messages": msgs}


class TestUserTurnFingerprint:
    def test_survives_assistant_remediation(self) -> None:
        # The whole point: v1 and v2 differ in assistant turns (state
        # annotations moved onto self-loops), so a fingerprint that changed
        # with them would match nothing and declare the whole split clean.
        v1 = _conv(["book a flight", "yes please"], assistant="[STATE: A -> B]")
        v2 = _conv(["book a flight", "yes please"], assistant="[STATE: A -> A]")

        assert user_turn_fingerprint(v1) == user_turn_fingerprint(v2)

    def test_differs_on_different_user_turns(self) -> None:
        assert user_turn_fingerprint(_conv(["a"])) != user_turn_fingerprint(_conv(["b"]))

    def test_differs_on_turn_order(self) -> None:
        assert user_turn_fingerprint(_conv(["a", "b"])) != user_turn_fingerprint(
            _conv(["b", "a"])
        )

    def test_ignores_conversation_id(self) -> None:
        # conversation_id is NOT unique across splits, so it must not
        # participate in the key.
        assert user_turn_fingerprint(_conv(["a"], cid="x")) == user_turn_fingerprint(
            _conv(["a"], cid="y")
        )

    def test_conversation_with_no_user_turns_is_stable(self) -> None:
        empty = {"messages": [{"role": "system", "content": "sys"}]}

        assert user_turn_fingerprint(empty) == user_turn_fingerprint(empty)


class TestSelectClean:
    def test_drops_contaminated_keeps_order(self) -> None:
        a, b, c = _conv(["a"]), _conv(["b"]), _conv(["c"])
        contaminated = {user_turn_fingerprint(b)}

        assert select_clean([a, b, c], contaminated) == [a, c]

    def test_empty_exclusion_keeps_everything(self) -> None:
        convs = [_conv(["a"]), _conv(["b"])]

        assert select_clean(convs, set()) == convs

    def test_all_contaminated_yields_empty(self) -> None:
        convs = [_conv(["a"]), _conv(["b"])]
        fps = {user_turn_fingerprint(c) for c in convs}

        assert select_clean(convs, fps) == []


class TestBuildCleanSet:
    def _write(self, path: Path, convs: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(c) for c in convs) + "\n")

    def test_excludes_candidates_present_in_any_exclusion_split(
        self, tmp_path: Path
    ) -> None:
        keep, in_train, in_val = _conv(["keep"]), _conv(["t"]), _conv(["v"])
        self._write(tmp_path / "cand" / "test.jsonl", [keep, in_train, in_val])
        self._write(tmp_path / "excl" / "train.jsonl", [in_train])
        self._write(tmp_path / "excl" / "validation.jsonl", [in_val])

        out = tmp_path / "clean"
        stats = build_clean_set(
            candidate_split=tmp_path / "cand" / "test.jsonl",
            exclusion_splits=[
                tmp_path / "excl" / "train.jsonl",
                tmp_path / "excl" / "validation.jsonl",
            ],
            out_dir=out,
            split_name="test",
        )

        written = [json.loads(x) for x in (out / "test.jsonl").read_text().splitlines()]
        assert written == [keep]
        assert stats["n_candidates"] == 3
        assert stats["n_clean"] == 1
        assert stats["n_excluded"] == 2

    def test_writes_under_the_requested_split_name(self, tmp_path: Path) -> None:
        self._write(tmp_path / "cand" / "test.jsonl", [_conv(["a"])])

        out = tmp_path / "clean"
        build_clean_set(
            candidate_split=tmp_path / "cand" / "test.jsonl",
            exclusion_splits=[],
            out_dir=out,
            split_name="test",
        )

        assert (out / "test.jsonl").exists()

    def test_load_fingerprints_spans_multiple_splits(self, tmp_path: Path) -> None:
        a, b = _conv(["a"]), _conv(["b"])
        self._write(tmp_path / "train.jsonl", [a])
        self._write(tmp_path / "validation.jsonl", [b])

        fps = load_fingerprints([tmp_path / "train.jsonl", tmp_path / "validation.jsonl"])

        assert fps == {user_turn_fingerprint(a), user_turn_fingerprint(b)}
