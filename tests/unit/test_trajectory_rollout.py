"""Tests for the pure trajectory-rollout logic and in-process replay rollout.

The GPU generation path is exercised with a monkeypatched ``model.generate`` and
a cached offline tokenizer (Qwen ChatML); no ``trl``/``unsloth`` is required —
``trajectory_rollout`` imports them lazily, so this whole file runs in ``.venv``.
"""

from __future__ import annotations

import pytest

from llm_workflow_agents.training.trajectory_rollout import (
    GoldScript,
    TrajectoryRolloutConfig,
    assert_trajectory_rollout_support,
    build_gold_script,
    classify_turn,
    make_replay_rollout_func,
    prompt_key,
    run_replay_rollout,
)


def _load_tokenizer_or_skip(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load a cached ChatML tokenizer for rollout tests, skipping if offline."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {model_id!r} unavailable offline: {e}")


class _ScriptedGenerate:
    """Mock model whose .generate appends a queued turn (same for every row).

    Each call pops the next scripted turn text, tokenizes it + a turn-end token,
    and returns ``cat([input_ids, gen])`` (HF convention). Ignores the input —
    the point is deterministic control over what each turn "generates".
    """

    def __init__(self, tokenizer, turns: list[str], turn_end_id: int) -> None:
        self.tok = tokenizer
        self.turns = list(turns)
        self.turn_end_id = turn_end_id
        self.calls = 0
        self.training = False

    def eval(self):  # noqa: ANN201
        return self

    def generate(self, input_ids=None, attention_mask=None, **kw):  # noqa: ANN001,ANN201
        import torch

        text = self.turns[self.calls] if self.calls < len(self.turns) else ""
        self.calls += 1
        gen = self.tok(text, add_special_tokens=False)["input_ids"] + [self.turn_end_id]
        b = input_ids.shape[0]
        gen_t = torch.tensor([gen] * b, dtype=input_ids.dtype)
        return torch.cat([input_ids, gen_t], dim=1)


class TestEnvGate:
    def test_importable_and_callable(self) -> None:
        # The module must import without trl/unsloth installed, and the gate
        # must be callable. The assertion body only runs meaningfully on the
        # train box (it imports trl), so we don't invoke it here.
        assert callable(assert_trajectory_rollout_support)


GOLD = [("A", "B"), ("B", "C"), ("C", "D")]


class TestClassifyTurn:
    def test_advance_single_step(self) -> None:
        assert classify_turn([("A", "B")], 0, GOLD) == ("advance", 1)

    def test_advance_consecutive_compression(self) -> None:
        assert classify_turn([("A", "B"), ("B", "C")], 0, GOLD) == ("advance", 2)

    def test_stall_on_self_loop(self) -> None:
        # A self-loop (X, X) is a neutral "stay" marker, not a transition.
        assert classify_turn([("B", "B")], 1, GOLD) == ("stall", 1)

    def test_stall_on_empty(self) -> None:
        assert classify_turn([], 1, GOLD) == ("stall", 1)

    def test_diverged_wrong_target(self) -> None:
        assert classify_turn([("B", "Z")], 1, GOLD) == ("diverged", 1)

    def test_diverged_non_consecutive(self) -> None:
        # (A,B) matches but (C,D) is not the immediately-next gold edge.
        assert classify_turn([("A", "B"), ("C", "D")], 0, GOLD) == ("diverged", 0)

    def test_cursor_at_end_any_transition_diverges(self) -> None:
        assert classify_turn([("D", "E")], 3, GOLD) == ("diverged", 3)

    def test_advance_from_mid_cursor(self) -> None:
        assert classify_turn([("B", "C")], 1, GOLD) == ("advance", 2)

    def test_self_loop_mixed_with_real_advance(self) -> None:
        # Self-loop is dropped, the real transition still advances.
        assert classify_turn([("A", "A"), ("A", "B")], 0, GOLD) == ("advance", 1)


# A synthetic Task A conversation: 3 gold assistant turns, spine A->B->C->TERMINAL.
RAW_ROW = {
    "conversation_id": "T_001",
    "workflow_graph": {
        "states": ["A", "B", "C", "TERMINAL"],
        "transitions": [
            {"from": "A", "to": "B", "condition": "c1", "priority": 0},
            {"from": "B", "to": "C", "condition": "c2", "priority": 0},
            {"from": "C", "to": "TERMINAL", "condition": "c3", "priority": 0},
        ],
        "initial": "A",
        "terminal": ["TERMINAL"],
    },
    "tool_schemas": [],
    "messages": [
        {"role": "system", "content": "ORIGINAL SYSTEM"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": '[STATE: A → B] ok <tool_call>{"name":"t1","arguments":{}}</tool_call>',
        },
        {"role": "tool", "content": '{"ok":true}'},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "[STATE: B → C] proceeding"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "[STATE: C → TERMINAL] done"},
    ],
    "ground_truth": {
        "state_sequence": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "TERMINAL"},
        ],
        "tool_calls": [{"name": "t1", "arguments": {}}],
        "terminal_state": "TERMINAL",
        "terminal_reached": True,
    },
}


class TestBuildGoldScript:
    def _script(self) -> GoldScript:
        return build_gold_script(RAW_ROW, enriched_system="ENRICHED SYSTEM")

    def test_prompt_messages_stop_before_first_assistant(self) -> None:
        s = self._script()
        assert [m["role"] for m in s.prompt_messages] == ["system", "user"]
        # The system content is swapped for the enriched prompt.
        assert s.prompt_messages[0]["content"] == "ENRICHED SYSTEM"
        assert s.prompt_messages[1]["content"] == "hi"

    def test_gold_transitions_and_invariant(self) -> None:
        s = self._script()
        assert s.gold_transitions == [("A", "B"), ("B", "C"), ("C", "TERMINAL")]
        # One transition per gold assistant turn.
        assert len(s.gold_transitions) == len(s.segments) == 3

    def test_segments_between_assistant_turns(self) -> None:
        s = self._script()
        # segment[0]: tool + user between asst#0 and asst#1
        assert [m["role"] for m in s.segments[0]] == ["tool", "user"]
        assert s.segments[0][1]["content"] == "next"
        # segment[1]: single user between asst#1 and asst#2
        assert [m["role"] for m in s.segments[1]] == ["user"]
        # segment[2]: trailing (conversation ends on assistant) -> empty
        assert s.segments[2] == []

    def test_terminal_and_valid_transitions(self) -> None:
        s = self._script()
        assert s.terminal_state == "TERMINAL"
        assert s.terminal_reached is True
        assert s.valid_transitions == [["A", "B"], ["B", "C"], ["C", "TERMINAL"]]
        assert s.gold_tool_calls == [{"name": "t1", "arguments": {}}]
        assert s.conversation_id == "T_001"

    def test_mismatched_invariant_raises(self) -> None:
        import copy

        bad = copy.deepcopy(RAW_ROW)
        # 2 transitions but 3 gold assistant turns -> invariant violation.
        bad["ground_truth"]["state_sequence"] = [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
        ]
        import pytest

        with pytest.raises(ValueError, match="gold_transitions"):
            build_gold_script(bad, enriched_system="X")


class TestPromptKey:
    def test_stable_across_key_order(self) -> None:
        a = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        b = [{"content": "s", "role": "system"}, {"content": "u", "role": "user"}]
        assert prompt_key(a) == prompt_key(b)

    def test_distinguishes_content(self) -> None:
        a = [{"role": "user", "content": "u1"}]
        b = [{"role": "user", "content": "u2"}]
        assert prompt_key(a) != prompt_key(b)


def _script(segments, gold, terminal="D") -> GoldScript:
    return GoldScript(
        conversation_id="t",
        prompt_messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ],
        segments=segments,
        gold_transitions=gold,
        gold_tool_calls=[],
        terminal_state=terminal,
        terminal_reached=True,
        valid_transitions=[list(g) for g in gold],
    )


def _turn_end_id(tok) -> int:
    from llm_workflow_agents.training.trajectory_rollout import _derive_turn_end_id

    return _derive_turn_end_id(tok)


class TestRunReplayRollout:
    def test_advance_then_diverge_masks_injected_gold(self) -> None:
        tok = _load_tokenizer_or_skip()
        end = _turn_end_id(tok)
        script = _script(
            segments=[
                [{"role": "user", "content": "SEG0USER"}],
                [{"role": "user", "content": "SEG1USER"}],
                [],
            ],
            gold=[("A", "B"), ("B", "C"), ("C", "D")],
        )
        model = _ScriptedGenerate(
            tok,
            ["[STATE: A → B] ok", "[STATE: B → C] ok", "[STATE: C → Z] oops"],
            end,
        )
        cfg = TrajectoryRolloutConfig(do_sample=False)
        [s] = run_replay_rollout(model, tok, [script], cfg)

        assert s.meta["stop_reason"] == "diverged"
        assert s.meta["cursor"] == 2
        assert s.meta["n_model_turns"] == 3
        assert s.meta["n_stall_turns"] == 0
        assert len(s.turn_texts) == 3

        # Mask alignment invariants (R1).
        assert len(s.env_mask) == len(s.completion_ids)
        assert s.env_mask[0] == 1
        assert sum(s.env_mask) > 0
        model_ids = [i for i, m in zip(s.completion_ids, s.env_mask) if m == 1]
        ext_ids = [i for i, m in zip(s.completion_ids, s.env_mask) if m == 0]
        model_text = tok.decode(model_ids, skip_special_tokens=True)
        ext_text = tok.decode(ext_ids, skip_special_tokens=True)
        assert "[STATE: A → B]" in model_text and "[STATE: C → Z]" in model_text
        # Injected gold user turns live ONLY under mask 0.
        assert "SEG0USER" in ext_text and "SEG1USER" in ext_text
        assert "SEG0USER" not in model_text and "SEG1USER" not in model_text
        # Trajectory ends on eos (immunizes against mask_truncated_completions).
        assert s.completion_ids[-1] == tok.eos_token_id

    def test_gold_complete_stop(self) -> None:
        tok = _load_tokenizer_or_skip()
        end = _turn_end_id(tok)
        script = _script(
            segments=[[{"role": "user", "content": "u1"}], [{"role": "user", "content": "u2"}], []],
            gold=[("A", "B"), ("B", "C"), ("C", "TERMINAL")],
            terminal="TERMINAL",
        )
        model = _ScriptedGenerate(
            tok,
            ["[STATE: A → B]", "[STATE: B → C]", "[STATE: C → TERMINAL]"],
            end,
        )
        [s] = run_replay_rollout(model, tok, [script], TrajectoryRolloutConfig(do_sample=False))
        assert s.meta["stop_reason"] == "gold_complete"
        assert s.meta["cursor"] == 3

    def test_budget_truncation_forces_masked_eos(self) -> None:
        tok = _load_tokenizer_or_skip()
        end = _turn_end_id(tok)
        script = _script(
            segments=[[{"role": "user", "content": "u0"}], [], []],
            gold=[("A", "B"), ("B", "C"), ("C", "D")],
        )
        model = _ScriptedGenerate(tok, ["[STATE: A → B]", "[STATE: B → C]"], end)
        # Tiny budget so the check trips right after turn 0 + its injected segment.
        cfg = TrajectoryRolloutConfig(
            do_sample=False, max_completion_tokens=40, per_turn_max_new_tokens=256
        )
        [s] = run_replay_rollout(model, tok, [script], cfg)
        assert s.meta["stop_reason"] == "budget"
        # Final token is a forced eos, masked out of the loss.
        assert s.completion_ids[-1] == tok.eos_token_id
        assert s.env_mask[-1] == 0
        assert len(s.env_mask) == len(s.completion_ids)

    def test_stall_counts(self) -> None:
        tok = _load_tokenizer_or_skip()
        end = _turn_end_id(tok)
        script = _script(
            segments=[[{"role": "user", "content": "u0"}], [{"role": "user", "content": "u1"}], []],
            gold=[("A", "B"), ("B", "C"), ("C", "D")],
        )
        # Turn 2 emits no transition -> stall; stall_turn_limit default 2 not hit.
        model = _ScriptedGenerate(
            tok, ["[STATE: A → B]", "just chatting, no state", "[STATE: B → C]"], end
        )
        [s] = run_replay_rollout(model, tok, [script], TrajectoryRolloutConfig(do_sample=False))
        assert s.meta["n_stall_turns"] == 1
        assert s.meta["cursor"] == 2  # advanced on turns 0 and 2


class TestMakeReplayRolloutFunc:
    def _stub_gate(self, monkeypatch) -> None:
        # The env gate imports trl, which can't load off the train box; stub it.
        import llm_workflow_agents.training.trajectory_rollout as tr

        monkeypatch.setattr(tr, "assert_trajectory_rollout_support", lambda: None)
        monkeypatch.setattr(tr, "_SUPPORT_CHECKED", False)

    def test_returns_trl_contract_and_looks_up_script(self, monkeypatch) -> None:
        self._stub_gate(monkeypatch)
        tok = _load_tokenizer_or_skip()
        end = _turn_end_id(tok)
        script = _script(
            segments=[[{"role": "user", "content": "u0"}], []],
            gold=[("A", "B"), ("B", "C")],
            terminal="C",
        )
        index = {prompt_key(script.prompt_messages): script}

        class _Args:
            temperature = 0.8
            top_p = 0.95

        class _Trainer:
            def __init__(self, model, tokenizer) -> None:
                self.model = model
                self.processing_class = tokenizer
                self.args = _Args()

        model = _ScriptedGenerate(tok, ["[STATE: A → B]", "[STATE: B → C]"], end)
        rollout = make_replay_rollout_func(index, TrajectoryRolloutConfig(do_sample=False))
        out = rollout([script.prompt_messages], _Trainer(model, tok))

        assert set(out) >= {
            "prompt_ids",
            "completion_ids",
            "logprobs",
            "env_mask",
            "trajectory",
            "rollout_meta",
        }
        assert out["logprobs"] is None
        assert len(out["env_mask"][0]) == len(out["completion_ids"][0])
        import json

        assert isinstance(json.loads(out["trajectory"][0]), list)

    def test_missing_script_is_hard_error(self, monkeypatch) -> None:
        self._stub_gate(monkeypatch)
        tok = _load_tokenizer_or_skip()

        class _Trainer:
            model = None
            processing_class = tok
            args = None

        rollout = make_replay_rollout_func({}, TrajectoryRolloutConfig())
        with pytest.raises(KeyError):
            rollout([[{"role": "user", "content": "unknown"}]], _Trainer())
