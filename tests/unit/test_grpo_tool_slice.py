"""Turn-level tool-bearing mix for the GRPO prompt set.

Why this exists: the 2026-08-16 50-step diagnostic on the C2 SFT base was
optimizer-stable (grad_norm <= 0.39, KL <= 1.94, both orders of magnitude under
the kill lines) but produced `reward 1.0, reward_std 0, frac_reward_zero_std 1`
at every step — no learning signal at all.

The reward function was not at fault; scored directly it returns 1.0 for a
GT-perfect completion, 0.0 for a bogus one and 0.44 for a missing annotation.
The prompt distribution was: 63.2% of GRPO turns carry no ground-truth tool
call, `tool_call_f1([], []) == 1.0` hands those rows 0.40 of the reward for
free, and C2's state accuracy is 0.9369 — so roughly 59% of prompts score
exactly 1.0 and every generation in the group ties. GRPO derives advantages
from within-group reward variance, so a tied group moves nothing.

This module rebalances toward the turns where C2 actually has headroom
(tool_f1 0.636 on tool-bearing rows, 12.7% emitting no call, 18/71 with wrong
arguments).

It rebalances rather than filters, deliberately. R15 showed that making a
behaviour structurally unconditional in the corpus teaches it as an
unconditional habit — the v2 remediation put every tool call on a self-loop
turn and the model learned "emit self-loops" rather than "emit self-loops when
calling a tool", driving spurious self-loops on advancing rows from 3.7% to
26.9%. A pure tool-only slice invites the same failure in a new direction:
"always call a tool", regressing the 1.5% spurious-call rate C2 currently
holds. Hence a ratio, and hence non-tool rows are never eliminated by default.
"""

import pytest

from llm_workflow_agents.training.grpo import _tool_bearing_mix_indices


def test_none_ratio_is_a_passthrough():
    """Absent config must not change existing runs' data."""
    has_tool = [True, False, False, True]
    assert _tool_bearing_mix_indices(has_tool, None) == [0, 1, 2, 3]


def test_balances_to_fifty_fifty():
    # 2 tool rows, 6 non-tool. At ratio 0.5 keep both tool rows + 2 non-tool.
    has_tool = [True, False, False, False, True, False, False, False]
    keep = _tool_bearing_mix_indices(has_tool, 0.5, seed=42)
    kept_tool = sum(has_tool[i] for i in keep)
    assert kept_tool == 2
    assert len(keep) == 4


def test_all_tool_rows_are_always_retained():
    """Tool rows are the scarce, informative ones — never dropped."""
    has_tool = [True, False, False, False, False, False]
    keep = _tool_bearing_mix_indices(has_tool, 0.5, seed=42)
    assert 0 in keep


def test_indices_are_sorted_and_unique():
    has_tool = [i % 3 == 0 for i in range(60)]
    keep = _tool_bearing_mix_indices(has_tool, 0.5, seed=7)
    assert keep == sorted(keep)
    assert len(keep) == len(set(keep))


def test_deterministic_for_a_given_seed():
    has_tool = [i % 4 == 0 for i in range(200)]
    a = _tool_bearing_mix_indices(has_tool, 0.5, seed=123)
    b = _tool_bearing_mix_indices(has_tool, 0.5, seed=123)
    assert a == b


def test_different_seeds_select_different_non_tool_rows():
    has_tool = [i % 4 == 0 for i in range(200)]
    a = _tool_bearing_mix_indices(has_tool, 0.5, seed=1)
    b = _tool_bearing_mix_indices(has_tool, 0.5, seed=2)
    assert a != b


def test_ratio_below_natural_is_a_passthrough():
    """Asking for less enrichment than the data already has must not downsample.

    The point is to raise the tool share, never to throw away data to hit a
    number the corpus already beats.
    """
    has_tool = [True] * 8 + [False] * 2  # already 80% tool-bearing
    assert _tool_bearing_mix_indices(has_tool, 0.5) == list(range(10))


def test_ratio_one_keeps_only_tool_rows():
    """Supported, but it is the R15-shaped setting — callers must opt in."""
    has_tool = [True, False, True, False]
    assert _tool_bearing_mix_indices(has_tool, 1.0) == [0, 2]


def test_no_tool_rows_at_all_is_a_passthrough():
    """Degenerate input must not produce an empty training set."""
    has_tool = [False, False, False]
    assert _tool_bearing_mix_indices(has_tool, 0.5) == [0, 1, 2]


def test_empty_input():
    assert _tool_bearing_mix_indices([], 0.5) == []


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_out_of_range_ratio_is_rejected(bad):
    with pytest.raises(ValueError):
        _tool_bearing_mix_indices([True, False], bad)


def test_realistic_corpus_proportions_reach_the_target():
    """36.8% tool-bearing (the measured Task A GRPO rate) -> 50%."""
    n = 27056
    has_tool = [i % 1000 < 368 for i in range(n)]
    keep = _tool_bearing_mix_indices(has_tool, 0.5, seed=42)
    share = sum(has_tool[i] for i in keep) / len(keep)
    assert 0.49 <= share <= 0.51
