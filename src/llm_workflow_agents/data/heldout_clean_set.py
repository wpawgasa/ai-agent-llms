"""Build a contamination-free held-out set for cross-lineage checkpoint audits.

Comparing a checkpoint trained on corpus v2 against one trained on v1 needs an
evaluation set neither model saw in training. Scoring on v2's full test split
leaks: 63 of its 278 conversations sit in v1's *training* set and 9 in its
validation set, which would hand the v1-trained model an unearned advantage on
23% of rows.

Conversations are matched across corpora by a fingerprint of their **user
turns**. Two properties make that the right key:

- Remediation rewrote assistant state annotations and tool calls but never
  touched what the user said, so the same conversation keeps its fingerprint
  across lineages — a key built from assistant content would match nothing and
  silently declare the whole split clean.
- ``conversation_id`` is **not** unique; ids collide across splits, so it
  cannot be used for this.

Used by ``scripts/build_heldout_clean_set.py``. The resulting directory is
consumed by ``scripts/heldout_composite_audit.py --data-dir <dir> --split test``
(``grpo._load_grpo_jsonl`` reads ``<dir>/<split>.jsonl`` and expects whole
conversations in corpus format, which is what this writes).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from llm_workflow_agents.data.voice_convention import strip_voice_markup


def user_turn_fingerprint(conversation: dict[str, Any]) -> str:
    """Stable cross-corpus key for one conversation, from its user turns only."""
    users = [
        strip_voice_markup(str(m.get("content", "") or ""))
        for m in (conversation.get("messages") or [])
        if m.get("role") == "user"
    ]
    joined = "\x00".join(users).encode("utf-8", errors="replace")
    return hashlib.blake2b(joined, digest_size=16).hexdigest()


def user_turn_prefix_fingerprints(conversation: dict[str, Any]) -> set[str]:
    """Fingerprints of every user-turn *prefix* of one conversation.

    :func:`user_turn_fingerprint` hashes a conversation's complete user-turn
    list. Per-turn training rows carry only a prefix of it (messages up to the
    turn being predicted), so their fingerprint can never equal the whole
    conversation's — a contamination guard comparing the two is inert and will
    silently pass everything.

    Expanding each excluded conversation into all of its prefixes makes the
    comparison meaningful: any turn row derived from that conversation matches
    one of them.
    """
    users = [
        strip_voice_markup(str(m.get("content", "") or ""))
        for m in (conversation.get("messages") or [])
        if m.get("role") == "user"
    ]
    out: set[str] = set()
    for k in range(1, len(users) + 1):
        joined = "\x00".join(users[:k]).encode("utf-8", errors="replace")
        out.add(hashlib.blake2b(joined, digest_size=16).hexdigest())
    return out


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


def load_prefix_fingerprints(splits: list[Path]) -> set[str]:
    """Union of :func:`user_turn_prefix_fingerprints` over every conversation."""
    out: set[str] = set()
    for split in splits:
        for conv in _read_jsonl(Path(split)):
            out |= user_turn_prefix_fingerprints(conv)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_fingerprints(splits: list[Path]) -> set[str]:
    """Collect user-turn fingerprints across every given split file."""
    fingerprints: set[str] = set()
    for split in splits:
        for conv in _read_jsonl(Path(split)):
            fingerprints.add(user_turn_fingerprint(conv))
    return fingerprints


def select_clean(
    candidates: list[dict[str, Any]], contaminated: set[str]
) -> list[dict[str, Any]]:
    """Candidates whose fingerprint appears in none of the exclusion splits.

    Order is preserved: the audit's sampler shuffles with a fixed seed over
    the emitted row order, so a reordered file yields a different sample.
    """
    return [c for c in candidates if user_turn_fingerprint(c) not in contaminated]


def build_clean_set(
    *,
    candidate_split: Path,
    exclusion_splits: list[Path],
    out_dir: Path,
    split_name: str = "test",
) -> dict[str, Any]:
    """Write the clean subset of ``candidate_split`` to ``out_dir/<split_name>.jsonl``."""
    candidates = _read_jsonl(Path(candidate_split))
    contaminated = load_fingerprints(exclusion_splits)
    clean = select_clean(candidates, contaminated)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{split_name}.jsonl"
    with open(target, "w") as fh:
        for conv in clean:
            fh.write(json.dumps(conv, ensure_ascii=False) + "\n")

    return {
        "candidate_split": str(candidate_split),
        "exclusion_splits": [str(p) for p in exclusion_splits],
        "output": str(target),
        "n_candidates": len(candidates),
        "n_clean": len(clean),
        "n_excluded": len(candidates) - len(clean),
    }
