#!/usr/bin/env python3
"""Materialize a DVC-tracked directory from the local cache to an arbitrary path.

Why this exists: the SFT checkpoint lineages ``sft-gemma4-v2`` and
``sft-gemma4-v3`` occupy the SAME DVC path
(``checkpoints/sft_cat_a/gemma-4-26B-A4B-it``) and both contain directories
named ``checkpoint-500`` / ``-1000`` / ``-1500`` with DIFFERENT weights.
A plain ``dvc checkout`` therefore silently replaces one with the other, and
scoring "checkpoint-1000" gives a plausible but wrong number depending on
which lineage happens to be on disk.

This script side-steps that by writing a chosen lineage to a path you name,
leaving the workspace copy untouched. It reads only the local DVC cache, so
it needs no network when the cache is already populated (verify first with
``dvc fetch`` / ``dvc pull`` on a fresh machine).

Usage:
    python scripts/materialize_dvc_lineage.py \\
        --dir-hash f89238076f5b09ca0d76e5b9ab98e4ec \\
        --out /tmp/sft_v2

    # or by tag, resolving the hash from that revision's dvc.lock
    python scripts/materialize_dvc_lineage.py \\
        --rev sft-gemma4-v2 \\
        --dvc-path checkpoints/sft_cat_a/gemma-4-26B-A4B-it \\
        --out /tmp/sft_v2

Known lineage hashes (see ``git tag -n20 -l 'sft-gemma4-*'``):
    sft-gemma4-v2  f89238076f5b09ca0d76e5b9ab98e4ec.dir  980 MB, 79 files
    sft-gemma4-v3  d5438dced5a25f54af0d73e8569c6483.dir  560 MB, 47 files
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / ".dvc" / "cache" / "files" / "md5"


def _cache_path(md5: str) -> Path:
    """Map an md5 to its content-addressed location in the DVC cache."""
    return CACHE / md5[:2] / md5[2:]


def resolve_hash_from_rev(rev: str, dvc_path: str) -> str:
    """Read ``dvc.lock`` at *rev* and return the .dir hash recorded for *dvc_path*.

    Scans every stage's ``outs``; the first match wins. Raises if the path is
    not an output at that revision.
    """
    import yaml

    blob = subprocess.run(
        ["git", "show", f"{rev}:dvc.lock"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lock = yaml.safe_load(blob) or {}
    for stage_name, stage in (lock.get("stages") or {}).items():
        for out in stage.get("outs") or []:
            if out.get("path") == dvc_path:
                return str(out["md5"])
    raise SystemExit(
        f"'{dvc_path}' is not a declared out in dvc.lock at rev '{rev}'"
    )


def materialize(dir_hash: str, out: Path, force: bool = False) -> int:
    """Copy every member of the ``.dir`` listing *dir_hash* into *out*.

    Returns the number of files written. Fails loudly and writes nothing if
    any member is absent from the cache — a partial checkpoint is worse than
    no checkpoint, since it looks loadable right up until it isn't.
    """
    dir_hash = dir_hash.removesuffix(".dir")
    listing = _cache_path(dir_hash + ".dir")
    if not listing.exists():
        raise SystemExit(
            f"cache miss for {dir_hash}.dir\n"
            f"  expected: {listing}\n"
            f"  run `dvc fetch --rev <tag> <path>` (or `dvc pull`) first"
        )

    entries = json.loads(listing.read_text())
    missing = [e for e in entries if not _cache_path(e["md5"]).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(entries)} member files missing from cache, "
            f"e.g. {[m['relpath'] for m in missing[:3]]}\n"
            "  refusing to write a partial checkpoint; fetch from the remote first"
        )

    if out.exists():
        if not force:
            raise SystemExit(f"{out} already exists (pass --force to overwrite)")
        shutil.rmtree(out)

    for e in entries:
        dest = out / e["relpath"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_cache_path(e["md5"]), dest)
    return len(entries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir-hash", help="the .dir md5 to materialize")
    src.add_argument("--rev", help="git rev/tag; resolves the hash via that rev's dvc.lock")
    ap.add_argument(
        "--dvc-path",
        default="checkpoints/sft_cat_a/gemma-4-26B-A4B-it",
        help="DVC out path to resolve when using --rev (default: %(default)s)",
    )
    ap.add_argument("--out", required=True, type=Path, help="destination directory")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    dir_hash = args.dir_hash or resolve_hash_from_rev(args.rev, args.dvc_path)
    n = materialize(dir_hash, args.out, force=args.force)
    print(f"materialized {n} files from {dir_hash[:16]}… -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
