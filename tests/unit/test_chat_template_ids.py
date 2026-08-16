"""Normalization of `apply_chat_template(tokenize=True)` return values.

transformers 5.x returns a `BatchEncoding` from `apply_chat_template(...,
tokenize=True)` where 4.x returned a plain `list[int]`. That single change has
now broken two separate call sites in this repo, in two different ways:

  * `sft.py::render_response_only_sample` — `list(mapping)` yields the mapping's
    KEYS (`['input_ids', 'attention_mask']`), so every per-turn delta came out
    empty and each sample rendered as 2 tokens with 0 unmasked labels. Silent:
    no crash, just a total loss of training signal. Fixed in bac1d98.
  * `trajectory_rollout.py::_derive_turn_end_id` — `int(ids[-1])` on the mapping
    raises `TypeError: int() argument must be ... not 'tokenizers.Encoding'`,
    which is what disabled the trajectory rollout path entirely.

The second happened because the first was fixed in place rather than shared.
This helper is the single implementation both now use, so the next transformers
change is one fix, not N.

Ordering matters and is asserted below: the mapping must be unwrapped BEFORE
any list coercion, because `list(BatchEncoding)` succeeds and returns nonsense
rather than raising.
"""

import pytest

from llm_workflow_agents.training._utils import normalize_chat_template_ids


class FakeBatchEncoding(dict):
    """Stands in for transformers' BatchEncoding: a Mapping of str -> list."""


class FakeTensor:
    """Stands in for a torch tensor: exposes .tolist()."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


def test_plain_list_passes_through():
    assert normalize_chat_template_ids([1, 2, 3]) == [1, 2, 3]


def test_mapping_is_unwrapped_to_input_ids():
    out = FakeBatchEncoding(input_ids=[4, 5, 6], attention_mask=[1, 1, 1])
    assert normalize_chat_template_ids(out) == [4, 5, 6]


def test_mapping_is_unwrapped_before_list_coercion():
    """The bac1d98 regression: list(mapping) yields keys, not ids."""
    out = FakeBatchEncoding(input_ids=[7, 8], attention_mask=[1, 1])
    result = normalize_chat_template_ids(out)
    assert result == [7, 8]
    assert "input_ids" not in result
    assert "attention_mask" not in result


def test_tensor_is_converted_via_tolist():
    assert normalize_chat_template_ids(FakeTensor([9, 10])) == [9, 10]


def test_batched_nested_list_is_flattened_to_first_row():
    assert normalize_chat_template_ids([[11, 12, 13]]) == [11, 12, 13]


def test_mapping_holding_a_batched_tensor():
    """The realistic transformers 5.x shape: mapping -> tensor -> batch dim."""
    out = FakeBatchEncoding(input_ids=FakeTensor([[14, 15]]))
    assert normalize_chat_template_ids(out) == [14, 15]


def test_empty_inputs():
    assert normalize_chat_template_ids([]) == []
    assert normalize_chat_template_ids(FakeBatchEncoding(input_ids=[])) == []


def test_last_element_is_an_int_not_a_container():
    """_derive_turn_end_id does int(ids[-1]); guard the exact failing op."""
    out = FakeBatchEncoding(input_ids=FakeTensor([[16, 17, 18]]))
    ids = normalize_chat_template_ids(out)
    assert int(ids[-1]) == 18


def test_mapping_without_input_ids_is_an_error():
    """Fail loudly rather than silently returning keys, as the old code did."""
    with pytest.raises(KeyError):
        normalize_chat_template_ids(FakeBatchEncoding(attention_mask=[1]))
