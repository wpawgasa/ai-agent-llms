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
from pathlib import Path
from typing import Any


def user_turn_fingerprint(conversation: dict[str, Any]) -> str:
    """Stable cross-corpus key for one conversation, from its user turns only."""
    users = [
        str(m.get("content", "") or "")
        for m in (conversation.get("messages") or [])
        if m.get("role") == "user"
    ]
    joined = "\x00".join(users).encode("utf-8", errors="replace")
    return hashlib.blake2b(joined, digest_size=16).hexdigest()


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
