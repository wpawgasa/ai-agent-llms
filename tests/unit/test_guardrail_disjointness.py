"""End-to-end proof that mining's exclusion set IS the guardrail's reservation.

The whole reserved-slice design rests on one property: when
``scripts/mine_model_negatives.py`` runs with ``--split validation`` and
``dpo.py::_build_heldout_callback`` runs the R5 guardrail, the rows one may use
and the rows the other may use are disjoint. Every other test in this area
exercises one side at a time. This file runs BOTH real, unmocked code paths
against ONE real synthetic ``validation.jsonl`` and compares their sets
directly.

It also proves the failure mode, not just the safe case: with a differing
``reserved_fraction`` or ``seed``, the two sides reserve *different* sets, which
is exactly the silent drift the "these values MUST match" warnings in
``mine_model_negatives.py``'s argparse help, ``configs/training/dpo_cat_a.yaml``
and ``dvc.yaml``'s ``task_a_preference_model_negatives`` stage exist to prevent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from llm_workflow_agents.data.heldout_clean_set import reserve_guardrail_slice
from mine_model_negatives import _excluded_fingerprints

_FRACTION = 0.2
_SEED = 42


def _conv(cid: str, n_user_turns: int = 2) -> dict:
    """A GRPO-corpus-shaped conversation, multi-turn so prefix fingerprints
    (what both consumers actually match against a per-turn prompt) are
    non-trivial rather than a single whole-conversation fingerprint."""
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for t in range(n_user_turns):
        msgs.append({"role": "user", "content": f"{cid} user turn {t}"})
        msgs.append(
            {
                "role": "assistant",
                "content": "[STATE: S1 -> S1] ok",
                "annotations": {
                    "state_transition": {"from": "S1", "to": "S1"},
                    "tool_calls": [],
                },
            }
        )
    return {
        "conversation_id": cid,
        "complexity_level": "L2",
        "domain": "banking",
        "workflow_graph": {},
        "messages": msgs,
        "ground_truth": {
            "state_sequence": [{"from": "S1", "to": "S1"}],
            "terminal_state": "S1",
            "tool_calls": [],
            "terminal_reached": True,
        },
    }


def _write_validation_split(tmp_path: Path, n: int = 40) -> Path:
    with open(tmp_path / "validation.jsonl", "w") as fh:
        for i in range(n):
            fh.write(json.dumps(_conv(f"c{i}")) + "\n")
    return tmp_path


class TestBothConsumersReferenceTheSameSet:
    """Both call sites, real code, same data, same parameters."""

    def test_mining_exclusion_equals_guardrail_reservation(self, tmp_path):
        data_dir = _write_validation_split(tmp_path)

        # dpo.py::_build_heldout_callback's side: the rows the R5 guardrail may
        # use are exactly those whose fingerprint is IN this set.
        reserved = reserve_guardrail_slice(
            data_dir,
            split="validation",
            reserved_fraction=_FRACTION,
            seed=_SEED,
        )
        # mine_model_negatives.py's side: the rows mining may use are exactly
        # those whose fingerprint is NOT IN this set.
        excluded = _excluded_fingerprints(
            data_dir,
            split="validation",
            heldout=[],
            guardrail_reserved_fraction=_FRACTION,
            guardrail_reserved_seed=_SEED,
        )

        assert reserved, "reservation must be non-empty or the property is vacuous"
        # Mining excludes precisely what the guardrail reserves => the minable
        # pool and the guardrail pool are disjoint by construction.
        assert excluded == reserved

    def test_every_conversation_lands_wholly_in_one_pool_and_both_are_populated(
        self, tmp_path
    ):
        """The consequence spelled out at conversation granularity: mining and
        the guardrail split the split into two non-empty halves, and no single
        conversation straddles them (a straddling conversation would let a
        later turn be mined while an earlier turn scores the guardrail)."""
        from llm_workflow_agents.data.heldout_clean_set import (
            user_turn_prefix_fingerprints,
        )

        convs = [_conv(f"c{i}") for i in range(40)]
        data_dir = _write_validation_split(tmp_path, n=40)
        excluded = _excluded_fingerprints(
            data_dir,
            split="validation",
            heldout=[],
            guardrail_reserved_fraction=_FRACTION,
            guardrail_reserved_seed=_SEED,
        )

        n_guardrail = 0
        n_minable = 0
        for conv in convs:
            prefixes = user_turn_prefix_fingerprints(conv)
            overlap = prefixes & excluded
            assert overlap == prefixes or overlap == set(), (
                f"{conv['conversation_id']} straddles the partition: "
                f"{len(overlap)} of {len(prefixes)} prefixes excluded"
            )
            if overlap:
                n_guardrail += 1
            else:
                n_minable += 1

        assert n_guardrail == 8  # round(40 * 0.2)
        assert n_minable == 32


class TestParameterDriftBreaksTheGuarantee:
    """Prove the danger named in the "these values MUST match" warnings."""

    def test_differing_fraction_gives_different_sets(self, tmp_path):
        data_dir = _write_validation_split(tmp_path, n=60)
        reserved = reserve_guardrail_slice(
            data_dir, split="validation", reserved_fraction=0.2, seed=_SEED
        )
        excluded = _excluded_fingerprints(
            data_dir,
            split="validation",
            heldout=[],
            guardrail_reserved_fraction=0.5,  # drifted
            guardrail_reserved_seed=_SEED,
        )
        assert excluded != reserved
        # And the drift is not benign: rows the guardrail reserved are now
        # missing from mining's exclusion set, i.e. minable AND guardrail rows.
        leaked = reserved - excluded
        assert leaked or (excluded - reserved), (
            "a fraction mismatch must change the partition in at least one "
            "direction, otherwise this test proves nothing"
        )

    def test_differing_seed_gives_different_sets(self, tmp_path):
        data_dir = _write_validation_split(tmp_path, n=60)
        reserved = reserve_guardrail_slice(
            data_dir, split="validation", reserved_fraction=_FRACTION, seed=42
        )
        excluded = _excluded_fingerprints(
            data_dir,
            split="validation",
            heldout=[],
            guardrail_reserved_fraction=_FRACTION,
            guardrail_reserved_seed=99,  # drifted
        )
        assert excluded != reserved
        # A seed mismatch keeps the SIZE the same while moving the membership —
        # the most silent form of the drift, hence worth asserting explicitly.
        assert len(excluded) == len(reserved)
        assert reserved - excluded, "seed drift must leak reserved rows into mining"


class TestCallSiteSignaturesAgree:
    """dpo.py's row selection is not a standalone unit, so instead assert the
    signature-level property its own (mock-based) tests rely on: both consumers
    call ``reserve_guardrail_slice`` with the same keyword names, on the same
    split, so identical config values really do mean identical sets."""

    def test_both_call_sites_pass_the_same_keywords(self):
        import inspect

        from llm_workflow_agents.training import dpo

        mine_src = inspect.getsource(_excluded_fingerprints)
        dpo_src = inspect.getsource(dpo._build_heldout_callback)
        for src, where in ((mine_src, "mine_model_negatives"), (dpo_src, "dpo")):
            assert "reserve_guardrail_slice(" in src, where
            assert 'split="validation"' in src, where
            assert "reserved_fraction=" in src, where
            assert "seed=" in src, where
