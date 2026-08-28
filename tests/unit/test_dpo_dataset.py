"""Preference-pair dataset loading and source mixing for training/dpo.py.

Covers the three pure-data steps between `scripts/build_preference_pairs.py` /
`scripts/mine_model_negatives.py` output and the `Dataset` handed to
DPOTrainer/ORPOTrainer:

  1. `_read_preference_jsonl` — load + validate one JSONL file of TRL
     conversational-preference rows (`{"prompt", "chosen", "rejected"}`).
  2. `_mix_preference_sources` — combine the SYNTHETIC pairs
     (`scripts/build_preference_pairs.py`) with the on-distribution MINED
     negatives (`scripts/mine_model_negatives.py`), optionally oversampling the
     scarce mined rows toward a target share (R18: 51 mined rows next to
     29,256 synthetic ones would otherwise be ~0.17% of the training signal).
  3. `_load_dpo_dataset` — the end-to-end assembly used by `train_dpo`.

No torch/transformers/trl import needed — `datasets.Dataset` is pure Python +
Arrow and safe to import in a CPU-only environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_workflow_agents.training.dpo import (
    _load_dpo_dataset,
    _mix_preference_sources,
    _read_preference_jsonl,
    _validate_pair_row,
)

GOOD_ROW = {
    "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    "chosen": [{"role": "assistant", "content": "the gold turn"}],
    "rejected": [{"role": "assistant", "content": "a wrong turn"}],
    "rejected_type": "drop_tool_calls",
    "source": "synthetic",
    "prompt_fingerprint": "abc123",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# --- _validate_pair_row ---


def test_validate_pair_row_accepts_the_real_shape():
    _validate_pair_row(GOOD_ROW)  # must not raise


@pytest.mark.parametrize("missing", ["prompt", "chosen", "rejected"])
def test_validate_pair_row_rejects_missing_key(missing):
    row = {k: v for k, v in GOOD_ROW.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        _validate_pair_row(row)


def test_validate_pair_row_rejects_empty_prompt():
    row = {**GOOD_ROW, "prompt": []}
    with pytest.raises(ValueError, match="prompt"):
        _validate_pair_row(row)


def test_validate_pair_row_rejects_chosen_not_starting_with_assistant():
    row = {**GOOD_ROW, "chosen": [{"role": "user", "content": "oops"}]}
    with pytest.raises(ValueError, match="chosen"):
        _validate_pair_row(row)


def test_validate_pair_row_rejects_rejected_not_starting_with_assistant():
    row = {**GOOD_ROW, "rejected": [{"role": "user", "content": "oops"}]}
    with pytest.raises(ValueError, match="rejected"):
        _validate_pair_row(row)


# --- _read_preference_jsonl ---


def test_read_preference_jsonl_round_trips(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [GOOD_ROW, GOOD_ROW])
    rows = _read_preference_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["chosen"][0]["content"] == "the gold turn"


def test_read_preference_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(GOOD_ROW) + "\n\n" + json.dumps(GOOD_ROW) + "\n")
    assert len(_read_preference_jsonl(path)) == 2


def test_read_preference_jsonl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _read_preference_jsonl(tmp_path / "does_not_exist.jsonl")


def test_read_preference_jsonl_empty_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _read_preference_jsonl(path)


def test_read_preference_jsonl_propagates_row_validation(tmp_path):
    bad_row = {k: v for k, v in GOOD_ROW.items() if k != "rejected"}
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [bad_row])
    with pytest.raises(ValueError, match="rejected"):
        _read_preference_jsonl(path)


# --- _mix_preference_sources ---


def _rows(n, source, tag):
    return [{**GOOD_ROW, "source": source, "prompt_fingerprint": f"{tag}{i}"} for i in range(n)]


def test_mix_passthrough_when_no_model_rows():
    synthetic = _rows(10, "synthetic", "s")
    assert _mix_preference_sources(synthetic, [], model_share=0.5) == synthetic


def test_mix_is_plain_concat_when_share_is_none():
    synthetic = _rows(10, "synthetic", "s")
    model = _rows(2, "model", "m")
    merged = _mix_preference_sources(synthetic, model, model_share=None)
    assert len(merged) == 12
    assert merged == synthetic + model


def test_mix_is_passthrough_at_or_below_natural_rate():
    """Never downsample either source to hit a number the data already beats."""
    synthetic = _rows(8, "synthetic", "s")
    model = _rows(2, "model", "m")  # natural share = 2/10 = 0.2
    merged = _mix_preference_sources(synthetic, model, model_share=0.2)
    assert len(merged) == 10
    assert sum(1 for r in merged if r["source"] == "model") == 2


def test_mix_oversamples_model_rows_toward_target_share():
    synthetic = _rows(1000, "synthetic", "s")
    model = _rows(10, "model", "m")  # natural share ~ 0.98%
    merged = _mix_preference_sources(synthetic, model, model_share=0.2, seed=1)
    model_count = sum(1 for r in merged if r["source"] == "model")
    total = len(merged)
    # n_model solves n_model / (1000 + n_model) == 0.2 -> n_model == 250
    assert model_count == 250
    assert total == 1250
    assert model_count / total == pytest.approx(0.2)


def test_mix_oversampling_is_deterministic_given_a_seed():
    synthetic = _rows(100, "synthetic", "s")
    model = _rows(5, "model", "m")
    a = _mix_preference_sources(synthetic, model, model_share=0.3, seed=7)
    b = _mix_preference_sources(synthetic, model, model_share=0.3, seed=7)
    assert a == b


def test_mix_rejects_share_out_of_range():
    synthetic = _rows(10, "synthetic", "s")
    model = _rows(1, "model", "m")
    with pytest.raises(ValueError, match="model_negative_share"):
        _mix_preference_sources(synthetic, model, model_share=1.0)
    with pytest.raises(ValueError, match="model_negative_share"):
        _mix_preference_sources(synthetic, model, model_share=-0.1)


# --- _load_dpo_dataset ---


def test_load_dpo_dataset_merges_sources_and_strips_bookkeeping_fields(tmp_path):
    train_path = tmp_path / "train.jsonl"
    neg_path = tmp_path / "model_negatives.jsonl"
    val_path = tmp_path / "validation.jsonl"
    _write_jsonl(train_path, _rows(4, "synthetic", "s"))
    _write_jsonl(neg_path, _rows(1, "model", "m"))
    _write_jsonl(val_path, _rows(2, "synthetic", "v"))

    data_cfg = {
        "train_sources": [str(train_path), str(neg_path)],
        "validation_source": str(val_path),
    }
    train_ds, eval_ds = _load_dpo_dataset(data_cfg)

    assert len(train_ds) == 5
    assert len(eval_ds) == 2
    assert set(train_ds.column_names) == {"prompt", "chosen", "rejected"}
    assert train_ds[0]["chosen"][0]["role"] == "assistant"


def test_load_dpo_dataset_requires_train_sources(tmp_path):
    with pytest.raises(ValueError, match="train_sources"):
        _load_dpo_dataset({"validation_source": str(tmp_path / "v.jsonl")})


def test_load_dpo_dataset_requires_validation_source(tmp_path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, _rows(2, "synthetic", "s"))
    with pytest.raises(ValueError, match="validation_source"):
        _load_dpo_dataset({"train_sources": [str(train_path)]})


def test_load_dpo_dataset_applies_model_negative_share(tmp_path):
    train_path = tmp_path / "train.jsonl"
    neg_path = tmp_path / "model_negatives.jsonl"
    val_path = tmp_path / "validation.jsonl"
    _write_jsonl(train_path, _rows(100, "synthetic", "s"))
    _write_jsonl(neg_path, _rows(1, "model", "m"))
    _write_jsonl(val_path, _rows(1, "synthetic", "v"))

    data_cfg = {
        "train_sources": [str(train_path), str(neg_path)],
        "validation_source": str(val_path),
        "model_negative_share": 0.1,
    }
    train_ds, _ = _load_dpo_dataset(data_cfg, seed=3)
    # n_model solves n_model / (100 + n_model) == 0.1 -> n_model == 11 (rounded)
    assert len(train_ds) == 111


# --------------------------------------------------------------------------- #
# data.max_train_rows — cap the merged train set to what the run consumes
# --------------------------------------------------------------------------- #
#
# WHY: `precompute_ref_log_probs` walks the WHOLE train split before step 1,
# and `run_phase2_dpo.sh --chunk-steps` repeats that once per chunk. The
# merged Cat A set is ~36,570 rows while a 500-step run at effective batch 8
# reads 4,000, so an uncapped 5-chunk run spends ~32 hours computing reference
# logprobs for rows it never reads. Measured 2026-08-28 at 1.6 rows/s.


def _cap_fixture(tmp_path: Path, n_synthetic: int, n_model: int) -> dict:
    syn = [dict(GOOD_ROW, prompt_fingerprint=f"s{i}") for i in range(n_synthetic)]
    mod = [
        dict(GOOD_ROW, source="model", prompt_fingerprint=f"m{i}")
        for i in range(n_model)
    ]
    _write_jsonl(tmp_path / "train.jsonl", syn)
    _write_jsonl(tmp_path / "model_negatives.jsonl", mod)
    _write_jsonl(tmp_path / "validation.jsonl", [dict(GOOD_ROW)])
    return {
        "train_sources": [
            str(tmp_path / "train.jsonl"),
            str(tmp_path / "model_negatives.jsonl"),
        ],
        "validation_source": str(tmp_path / "validation.jsonl"),
    }


def test_max_train_rows_truncates_the_merged_train_set(tmp_path):
    cfg = _cap_fixture(tmp_path, n_synthetic=50, n_model=10)
    cfg["max_train_rows"] = 20
    train, _ = _load_dpo_dataset(cfg, seed=42)
    assert len(train) == 20


def test_max_train_rows_absent_keeps_every_row(tmp_path):
    cfg = _cap_fixture(tmp_path, n_synthetic=50, n_model=10)
    train, _ = _load_dpo_dataset(cfg, seed=42)
    assert len(train) == 60


def test_max_train_rows_above_the_set_size_is_a_passthrough(tmp_path):
    cfg = _cap_fixture(tmp_path, n_synthetic=50, n_model=10)
    cfg["max_train_rows"] = 5000
    train, _ = _load_dpo_dataset(cfg, seed=42)
    assert len(train) == 60


def test_cap_does_not_discard_the_mined_negatives(tmp_path):
    """The trap: with no model_negative_share the mixer returns synthetic THEN
    mined, unshuffled. A head-slice would drop every mined row — the scarce,
    on-distribution ones R18 says carry the most value."""
    cfg = _cap_fixture(tmp_path, n_synthetic=50, n_model=10)
    cfg["max_train_rows"] = 30
    train, _ = _load_dpo_dataset(cfg, seed=42)
    kept = {r["chosen"][0]["content"] for r in train}
    assert len(train) == 30
    # Every row shares chosen/rejected text here, so assert via the mixer
    # instead: capping must sample across the whole merged set, not its head.
    from llm_workflow_agents.training.dpo import _cap_train_rows

    merged = [{"i": i, "src": "syn"} for i in range(50)]
    merged += [{"i": i, "src": "mod"} for i in range(10)]
    capped = _cap_train_rows(merged, 30, seed=42)
    assert len(capped) == 30
    assert any(r["src"] == "mod" for r in capped)
    assert kept  # loader path produced rows


def test_cap_is_deterministic_for_a_given_seed(tmp_path):
    from llm_workflow_agents.training.dpo import _cap_train_rows

    merged = [{"i": i} for i in range(100)]
    assert _cap_train_rows(merged, 10, seed=7) == _cap_train_rows(merged, 10, seed=7)
    assert _cap_train_rows(merged, 10, seed=7) != _cap_train_rows(merged, 10, seed=8)
