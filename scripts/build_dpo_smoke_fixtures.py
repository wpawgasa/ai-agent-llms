#!/usr/bin/env python3
"""Build small, length-filtered fixture files for the Cat A DPO smoke ladder.

The smoke runs under ``checkpoints/dpo_cat_a_smoke*/`` were originally driven by
ad-hoc JSONL subsets written into a per-session scratchpad directory. That path
does not survive a container rebuild, so every stored smoke config now points at
files that no longer exist and no smoke result can be reproduced. This script
makes the fixtures a real, repeatable artifact.

Selection is deterministic (sorted by fingerprint, no RNG) so two invocations on
the same source data produce byte-identical files.

Rows are filtered on TOKENIZED length so that ``prompt + chosen`` and
``prompt + rejected`` both fit under the cap the smoke run will use. Nothing is
truncated, which keeps per-step memory bounded and comparable across runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_SRC = Path("data/output/preference/task_a")
DEFAULT_OUT = DEFAULT_SRC / "smoke"
DEFAULT_TOKENIZER = "checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _pair_len(tok, row: dict, side: str) -> int:
    """Token length of prompt + one completion side, as the collator sees it."""
    text = tok.apply_chat_template(
        list(row["prompt"]) + list(row[side]), tokenize=False
    )
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _select(tok, rows: list[dict], cap: int, n: int, label: str) -> list[dict]:
    # Deterministic order — never rely on source file ordering or an RNG.
    rows = sorted(rows, key=lambda r: r.get("prompt_fingerprint", ""))
    kept: list[dict] = []
    scanned = 0
    for row in rows:
        if len(kept) >= n:
            break
        scanned += 1
        longest = max(_pair_len(tok, row, "chosen"), _pair_len(tok, row, "rejected"))
        if longest <= cap:
            kept.append(row)
    print(f"  {label}: kept {len(kept)} of {scanned} scanned (cap {cap})")
    if len(kept) < n:
        raise SystemExit(
            f"{label}: only {len(kept)} of the requested {n} rows fit under "
            f"cap {cap}; lower --n-* or raise --cap"
        )
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument(
        "--cap",
        type=int,
        default=5120,
        help="max_seq_length the smoke run will use; rows longer than this are "
        "dropped so nothing truncates (default: %(default)s)",
    )
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-validation", type=int, default=16)
    ap.add_argument("--n-model-negatives", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Building DPO smoke fixtures -> {args.out_dir}")
    for name, n in (
        ("train", args.n_train),
        ("validation", args.n_validation),
        ("model_negatives", args.n_model_negatives),
    ):
        src = args.src_dir / f"{name}.jsonl"
        rows = _read_jsonl(src)
        kept = _select(tok, rows, args.cap, n, name)
        out = args.out_dir / f"{name}.jsonl"
        _write_jsonl(out, kept)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
