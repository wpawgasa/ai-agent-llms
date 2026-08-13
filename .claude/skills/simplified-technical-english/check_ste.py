#!/usr/bin/env python3
"""Measure a document against the Simplified Technical English contract.

Usage:
    python check_ste.py FILE [FILE ...] [--procedural]

Reports sentence length distribution and passive voice, then prints every
sentence that breaks the contract. Exit code 1 if the document fails the gate,
so it can run in CI.

Gate (descriptive default):
    no sentence over 25 words, and under 10% of sentences over 20 words.
With --procedural, the hard cap drops to 20 words.

Markdown that is not prose is excluded: fenced code blocks, table rows,
headings, and block quotes. Inline code and links collapse to a single token so
a long path does not read as a long sentence.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

PASSIVE = re.compile(r"\b(is|are|was|were|be|been|being)\s+\w+(ed|en)\b", re.I)
HEDGE = re.compile(r"\b(should|might|could|may|typically|generally|usually|perhaps)\b", re.I)
SOFT_CAP = 20
HARD_CAP = 25


def prose_sentences(text: str) -> list[str]:
    """Extract prose sentences, dropping non-prose markdown.

    Splits line by line, never across lines. A bullet list rarely ends its
    items with a full stop, so splitting the whole document on [.!?] would
    concatenate every item into one enormous false sentence. Each list item is
    its own unit and carries the same word limit.
    """
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # YAML frontmatter
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # fenced code

    blocks: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("|", "#", ">", "---", "===")):
            blocks.append([])  # table, heading, quote, rule — ends the block
            continue
        if not stripped:
            blocks.append([])  # blank line ends the block
            continue
        if re.match(r"(?:[-*+]|\d+\.)\s+", stripped):
            blocks.append([])  # a list item starts its own block
        if not blocks:
            blocks.append([])
        blocks[-1].append(stripped)

    sentences: list[str] = []
    for block in blocks:
        if not block:
            continue
        # Rejoin hard-wrapped lines before splitting, or one long sentence
        # spread over three wrapped lines reads as three short ones.
        joined = " ".join(block)
        joined = re.sub(r"`[^`]*`", "CODE", joined)
        joined = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", joined)
        joined = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", joined)
        joined = re.sub(r"[*_]", "", joined)
        for part in re.split(r"(?<=[.!?])\s+", joined):
            part = " ".join(part.split())
            if len(part.split()) >= 4 and not part.startswith("CODE"):
                sentences.append(part)
    return sentences


def check(path: pathlib.Path, hard_cap: int) -> bool:
    sentences = prose_sentences(path.read_text(encoding="utf-8"))
    if not sentences:
        print(f"{path}: no prose found")
        return True

    lengths = [len(s.split()) for s in sentences]
    over_soft = [s for s in sentences if len(s.split()) > SOFT_CAP]
    over_hard = [s for s in sentences if len(s.split()) > hard_cap]
    passive = [s for s in sentences if PASSIVE.search(s)]
    hedged = [s for s in sentences if HEDGE.search(s)]

    share_soft = len(over_soft) / len(sentences)
    print(f"\n{path}")
    print(f"  sentences        {len(sentences)}")
    print(f"  mean length      {sum(lengths) / len(lengths):.1f} words")
    print(f"  longest          {max(lengths)} words")
    print(f"  over {SOFT_CAP} words    {len(over_soft)} ({share_soft:.0%})")
    print(f"  over {hard_cap} words    {len(over_hard)}")
    print(f"  passive voice    {len(passive)} ({len(passive) / len(sentences):.0%})")
    print(f"  hedged           {len(hedged)}")

    if over_hard:
        print("\n  FIX THESE FIRST — over the hard cap:")
        for s in sorted(over_hard, key=lambda x: -len(x.split())):
            print(f"    [{len(s.split())}w] {s[:110]}")

    passed = not over_hard and share_soft < 0.10
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--procedural",
        action="store_true",
        help=f"apply the procedural hard cap of {SOFT_CAP} words instead of {HARD_CAP}",
    )
    args = parser.parse_args()

    hard_cap = SOFT_CAP if args.procedural else HARD_CAP
    results = [check(p, hard_cap) for p in args.files]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
