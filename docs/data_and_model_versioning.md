# Data & Model Versioning Procedure

How corpora and checkpoints are versioned, recovered, and handed between machines.
Complements the [DVC Data Pipeline](../README.md#dvc-data-pipeline) section of the README,
which covers day-to-day `pull` / `repro` / `status`. This document covers **versioning** —
keeping more than one lineage of the same artifact recoverable over time.

Written 2026-07-25 after an audit found that one checkpoint lineage (`ckpt-1000`, the
baseline every §12 number is measured against) was reachable only by knowing to read
`dvc.lock` at a specific unrelated commit. Every claim below was verified against the live
repo and the GCS remote at that date.

---

## 1. The mental model: three independent layers

Most confusion here comes from treating DVC as if it stored history. It does not. Three
separate things cooperate, and only one of them is a history:

| Layer | What it holds | Overwrite behaviour |
|---|---|---|
| **Git tags / history** | The *hash* of each lineage, and the story of why it exists | Append-only. This is the version history. |
| **`dvc.lock`** | Exactly **one hash per (stage, path)** — whatever that stage produced on its last run | Overwritten in place on every run |
| **`.dvc/cache` + GCS remote** | The bytes, content-addressed by md5 | Never overwritten. Blobs coexist; only `dvc gc` deletes. |

The practical consequence: **a new run does not destroy the previous lineage's data.** It
replaces the workspace files and the `dvc.lock` pointer, but the old blobs remain in cache
and on the remote, addressed by their own hash. What you lose without a tag is not the
data — it is the *ability to find the hash*, which is just as fatal in practice.

`dvc.lock` is closer to `package-lock.json` than to a changelog. Reading it tells you what
the pipeline produced most recently; it cannot tell you what it produced before that.

### 1.1 Why `dvc.lock` can name two lineages at once

Each stage independently freezes the hashes it observed during *its own* last run. Two
stages that last ran at different times will disagree about the same path. Currently:

| Stage | Role | Lineage | Last ran |
|---|---|---|---|
| `task_a_sft_gemma4_26b_a4b` | `outs` | v3 (560 MB) | 2026-07-22 |
| `task_a_grpo_gemma4_26b_a4b` | `deps` | v2 (980 MB) | 2026-07-06 |

This is not corruption — it is the format working as designed. But it *is* a signal that
the GRPO stage is stale with respect to its dependency, and `dvc status` reports it as
`changed deps`. Treat a same-path disagreement between two stages as a prompt to check
which one is out of date before running anything (this same signature is what exposed the
R12 stale-corpus bug; see CLAUDE.md).

---

## 2. Current inventory

```bash
git tag -n1 -l 'sft-gemma4-*' 'task_a-corpus-*'    # one-line summaries
git tag -n30 sft-gemma4-v2                          # full annotation, incl. hash
```

| Tag | Commit | Artifact | Hash | Size |
|---|---|---|---|---|
| `task_a-corpus-v1-2026-07-22` | `8b0ac40` | Task A corpus, registered via `dvc commit` on a dry-run check | `3d6a4d3e…` / `6bb5eb6f…` | ~110 MB |
| `task_a-corpus-v2-2026-07-22` | `666fe86` | Same bytes, first actually executed via `dvc repro` | `3d6a4d3e…` / `6bb5eb6f…` | ~110 MB |
| `sft-gemma4-v2` | `b0d53f9` | ckpt-1000 baseline lineage (pre-R12 corpus) | `f89238076f…` | 980 MB, 79 files |
| `sft-gemma4-v3` | `480ffd0` | C0 control cell (clean corpus) | `d5438dced5…` | 560 MB, 47 files |

`sft-gemma4-v2` was created retroactively on 2026-07-25 — it did not exist when v3 was
registered, even though v3's own tag message referred to it.

**Untagged as of 2026-07-25:** the GRPO Cat A checkpoint lineage
(`checkpoints/grpo_cat_a/gemma-4-26B-A4B-it`, `e9b711c1f7…`, 487 MB, 35 files,
`checkpoint-50/-100/-150`), registered by `48028c5`. Its bytes are fully present in both the
local cache and the GCS remote, so it is recoverable *today* — but only via `dvc.lock`'s
current entry. The moment the `task_a_grpo_gemma4_26b_a4b` stage reruns, that entry is
overwritten and the lineage drops to git-archaeology-only (§5); `dvc gc -w` would delete it
outright (§6). It should be tagged before any further GRPO work.

---

## 3. Registering a new lineage

Run these **in order**. Steps 2 and 3 are what make the lineage recoverable; skipping
either leaves it reachable only by luck.

```bash
# 1. Register the artifact with DVC (or let `dvc repro` do it as part of the run)
dvc commit checkpoints/sft_cat_a/gemma-4-26B-A4B-it
git add dvc.lock && git commit -m "chore(dvc): register <artifact> <version>"

# 2. Push the bytes BEFORE the workspace copy is replaced by the next run.
#    A blob that only ever existed locally dies with the cache.
dvc push
dvc status --cloud          # verify: no "missing" entries for this path

# 3. Tag the commit from step 1 — AFTER it exists, never before.
git tag -a sft-gemma4-v4 -m "..."   # see §3.2 for what belongs in the message
git push origin main sft-gemma4-v4
```

### 3.1 Naming convention

- **Models:** `sft-<family>-v<N>` / `grpo-<family>-v<N>` — monotonic, no dates. The
  lineage number is the identity; the date lives in the annotation.
- **Corpora:** `task_<x>-corpus-v<N>-<YYYY-MM-DD>` — dated, because corpus regenerations
  are the thing most often correlated against a calendar.

### 3.2 What the tag annotation must contain

The annotation is the only durable record. At minimum:

- The **DVC `.dir` hash**, file count, and size — this is what makes recovery a one-liner.
- Which **corpus tag** it was trained on, and the row counts from `train.log` (they are the
  independent check on the corpus claim — see CLAUDE.md R13, where a frozen config lied and
  `train.log` was the thing that settled it).
- Key hyperparameters and the headline metric.
- What it supersedes, and whether the predecessor is still recoverable.
- Any **path collisions** with sibling lineages (§4).

---

## 4. Hazard: lineages sharing one DVC path

**All SFT lineages currently write to the same path**,
`checkpoints/sft_cat_a/gemma-4-26B-A4B-it`, and their inner checkpoint directories collide
by name. Verified between v2 and v3:

- 35 paths exist in both lineages; **28 of them differ by content hash**.
- Both contain `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500` — same names,
  different weights. (v2 additionally has `-2000/-2500/-3000/-3426`; v3 has `-1770`.)
- Even `chat_template.jinja` differs between them.

So `checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000` means **whichever lineage is
currently checked out**. Scoring it while expecting the baseline yields a wrong number that
looks entirely plausible — the same class of provenance failure as R13.

**Never `dvc checkout` an older lineage in place to inspect it.** Materialize it to a
separate path instead:

```bash
python3 scripts/materialize_dvc_lineage.py --rev sft-gemma4-v2 --out /tmp/sft_v2
```

That script reads only the local cache, refuses to write a partial checkpoint if any member
blob is missing, and leaves the workspace copy untouched.

The durable fix is to give each lineage its own `output_dir` so one path stops meaning two
things — `training/sft.py::_resolve_output_dir()` already supports an explicit `output_dir`
config key for exactly this (CLAUDE.md R13). Until every cell uses it, §4's caution applies.

---

## 5. Recovery

Recovering any tagged lineage, on any machine:

```bash
# Warm cache (same machine) — no network needed
python3 scripts/materialize_dvc_lineage.py --rev sft-gemma4-v3 --out /tmp/sft_v3

# Cold cache (fresh machine) — fetch the blobs first
dvc fetch --rev sft-gemma4-v3 checkpoints/sft_cat_a/gemma-4-26B-A4B-it
python3 scripts/materialize_dvc_lineage.py --rev sft-gemma4-v3 --out /tmp/sft_v3
```

To restore the lineage that `dvc.lock` currently points at, back into its normal workspace
location, the plain DVC command is correct and sufficient:

```bash
dvc checkout checkpoints/sft_cat_a/gemma-4-26B-A4B-it
```

Verified 2026-07-25: recovering v3 from its tag reproduced the workspace copy
byte-for-byte (`diff -rq`, 47 files, no differences).

If a lineage was never tagged, it is still recoverable *if* you can find a commit whose
`dvc.lock` recorded it:

```bash
git log --format=%h -- dvc.lock | while read c; do
  echo "$c $(git show "$c":dvc.lock \
    | grep -A3 'gemma-4-26B-A4B-it' \
    | grep -oE '[a-f0-9]{32}\.dir' | sort -u | tr '\n' ' ')"
done
```

Note the substring matches **both** `checkpoints/sft_cat_a/…` and `checkpoints/grpo_cat_a/…`,
so expect more than one hash per line; cross-reference against §8's stage-by-stage listing to
tell them apart. Once found, tag it retroactively so nobody has to repeat the archaeology —
that is exactly how `sft-gemma4-v2` came to exist.

---

## 6. Garbage collection — the only thing that truly deletes

`dvc repro` and new runs are safe. `dvc gc` is not:

```bash
dvc gc -w                # ⚠️ keeps ONLY what the current workspace references.
                         #    Deletes every older lineage's blobs.
dvc gc -w --cloud        # ⚠️ same, on GCS as well — unrecoverable.
```

Always preserve tagged lineages:

```bash
dvc gc --all-tags        # keeps everything any tag points at — the default choice
dvc gc --all-commits     # maximally conservative
```

This is the single strongest reason the tagging discipline in §3 matters: a tag is not just
documentation, it is what makes a lineage survive garbage collection.

---

## 7. Machine-to-machine handoff

Before moving to another machine:

1. `dvc push` — confirm with `dvc status --cloud` that nothing is missing.
2. `git push origin main --tags` — the hashes travel in the tags, not the data.
3. **Copy the GCS service-account key out-of-band.** `.dvc/config` points at
   `looloo-ocr-9e0b69945c03.json` in the **project root**, and it is gitignored
   (`.gitignore:51:*.json`) — correctly, since it is a credential, but it therefore will
   *not* be on the target machine. Without it DVC cannot reach the remote at all.

Verify from the target machine before relying on it:

```bash
dvc status --cloud                              # auth + reachability in one shot
git tag -n1 -l 'sft-gemma4-*'                   # tags arrived
```

---

## 8. Verification recipes

```bash
# Which lineage is on disk right now?
python3 -c "
import yaml; l=yaml.safe_load(open('dvc.lock'))
p='checkpoints/sft_cat_a/gemma-4-26B-A4B-it'
for s,b in l['stages'].items():
    for k in ('deps','outs'):
        for i in b.get(k) or []:
            if i.get('path','').startswith(p): print(s,k,i['md5'][:16])"

# Is a lineage fully present on the remote? (object-by-object, not just the .dir)
# See git log for the gcsfs snippet used in the 2026-07-25 audit.
dvc status --cloud
```

---

## 9. Known open items

- **Untagged GRPO lineage** (§2) — `e9b711c1f7…` is protected only by `dvc.lock`'s current
  entry. Tag it before the GRPO stage reruns.
- **Shared checkpoint path** (§4) — every SFT cell writes to one directory. Use the explicit
  `output_dir` config key per cell; until then, materialize to separate paths for comparison.
- **`dvc.lock` stage disagreement** (§1.1) — the GRPO stage's dep is pinned to v2 while the
  SFT stage's out is v3. Benign until the GRPO stage reruns, at which point it would consume
  v3 weights while its config names `checkpoint-500` — a different model in each lineage.
- **`scripts/run_phase2_grpo.sh` fixed-path patched config** — the R13 bug fixed in
  `run_phase2_sft.sh` is still present here (`PATCHED_CFG="$PATCHED_DIR/${GRPO_STEM}.yaml"`),
  so GRPO run provenance is still clobberable by a later invocation.
- **`task_a_grpo` `dvc.yaml` `cmd:` indentation** — the block-folding bug fixed for
  `task_a_sft_clean` / `task_a_sft_splits` in `666fe86` was not fixed for this stage.
