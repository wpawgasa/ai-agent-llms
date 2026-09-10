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
git tag -n1 -l 'corpus/*' 'model/*'    # one-line summaries
git tag -n40 model/sft-gemma4-c2-on-task-a-v2   # full annotation, incl. hashes
```

### Corpora

| Tag | Commit | Contents | Hash (`task_a_splits`) |
|---|---|---|---|
| `corpus/task-a-v1` | `6a50272` | 5,549 convs, all text; 4,716 / 554 / 279 | `6bb5eb6f…` |
| `corpus/task-a-v2` | `64e98e5` | 5,543 convs, tool-call stay convention; 4,711 / 554 / 278 | `21e33e25…` |
| `corpus/task-a-v3` | `ba7b827` | 9,932 convs (7,043 text + 2,889 voice); 8,441 / 992 / 499 | see annotation |

### Models

| Tag | Commit | Cell / recipe | Hash | Held-out composite |
|---|---|---|---|---|
| `model/sft-gemma4-v2-on-pre-r12` | `b0d53f9` | ckpt-1000 baseline, untagged corpus | `f89238076f…` | 0.7271 (different set) |
| `model/sft-gemma4-v3-on-task-a-v1` | `480ffd0` | C0, `all_tokens` @ 4096 | `d5438dced5…` | 0.5709 |
| `model/sft-gemma4-v4-on-task-a-v2` | `602de60` | C0, `all_tokens` @ 4096 | `57e40028fe…` | 0.5120 |
| `model/sft-gemma4-c2-step500-on-task-a-v2` | `8ec1929` | C2 snapshot, **incomplete** | `110bb1bf2e…` | — |
| **`model/sft-gemma4-c2-on-task-a-v2`** | `7e758da` | **C2, `response_only` @ 8192 — best** | `50ed6597b5…` | **0.7595** |

The three held-out composites above are all scored on the same 206-row set and are
comparable to each other. None is comparable to a score on the `corpus/task-a-v3` held-out
sets — see §3.3.

`model/sft-gemma4-v2-on-pre-r12` was created retroactively on 2026-07-25 — it did not exist
when the v3 lineage was registered, even though v3's own tag message referred to it.

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
git tag -a model/sft-gemma4-v5-on-task-a-v3 -F msg.txt   # see §3.2 for the message
git push origin main model/sft-gemma4-v5-on-task-a-v3
```

### 3.1 Naming convention

Three namespaces, separated by a `/` prefix so a corpus and a model can never collide:

```
corpus/<task>-v<N>                             immutable corpus bytes
model/<stage>-<family>-<lineage>-on-<corpus>   a trained artifact
derived/<task>-<kind>-v<N>                     derived sets (grpo, preference, heldout)
```

- **`<lineage>`** is `v<N>` for the monotonic line, or the factorial cell name where the
  cell is the identity (`c2`). Numbers never rewind and never name two different artifacts.
- **`-on-<corpus>`** is **mandatory** on every model tag, and names a `corpus/` tag without
  its prefix (`-on-task-a-v2`). If the training corpus has no tag, name the era instead
  (`-on-pre-r12`) and say in the annotation why it is untagged and whether it is still
  recoverable.
- **No dates in names.** Git stores `creatordate` already, and the dated scheme this
  replaced put `2026-07-22` on two artifacts registered on different days. Dates belong in
  the annotation body, where they can be qualified.
- **The third namespace is `derived/`, not `data/`.** A `data/…` ref would be ambiguous with
  the `data/` directory in every `git checkout`, `git log`, and grep — git would need a `--`
  to tell a ref from a path. No prefix may shadow a top-level directory name.

Why `-on-<corpus>` is mandatory: the previous rule made model tags corpus-agnostic and
recorded the corpus in the annotation only. That failed twice in six weeks — `sft-gemma4-c2`
gave no way to tell which corpus it trained on, and the fix, `sft-cat-a-c2-corpus-v2`, put a
corpus version where readers expected a lineage number, so its own annotation had to open by
warning against the misreading. A name that carries the corpus makes the question
unaskable.

### 3.2 What the tag annotation must contain

The annotation is the only durable record. At minimum:

- The **DVC `.dir` hash**, file count, and size — this is what makes recovery a one-liner.
- Which **corpus tag** it was trained on, and the row counts from `train.log` (they are the
  independent check on the corpus claim — see CLAUDE.md R13, where a frozen config lied and
  `train.log` was the thing that settled it).
- Key hyperparameters and the headline metric.
- What it supersedes, and whether the predecessor is still recoverable.
- Any **path collisions** with sibling lineages (§4).

### 3.3 Migration from the pre-2026-09 names

Eleven tags were created under two conflicting schemes. As of 2026-09-10 each artifact also
carries a conforming name. **The old tags are kept, not deleted** — they are referenced from
`CLAUDE.md`, the docs, and any clone that already fetched them, and a deleted tag is exactly
what `dvc gc` reads as "this lineage is garbage" (§6). Use the new names for new work.

| Use this | Deprecated |
|---|---|
| `corpus/task-a-v1` | `task-a-sft-v1`, `task_a-corpus-v1-2026-07-22`, `task_a-corpus-v2-2026-07-22` |
| `corpus/task-a-v2` | `task-a-sft-v2` |
| `corpus/task-a-v3` | `task-a-sft-v3` |
| `model/sft-gemma4-v2-on-pre-r12` | `sft-gemma4-v2` |
| `model/sft-gemma4-v3-on-task-a-v1` | `sft-gemma4-v3` |
| `model/sft-gemma4-v4-on-task-a-v2` | `sft-gemma4-v4` |
| `model/sft-gemma4-c2-step500-on-task-a-v2` | `sft-gemma4-c2-step500` |
| `model/sft-gemma4-c2-on-task-a-v2` | `sft-gemma4-c2`, `sft-cat-a-c2-corpus-v2` |

Every new tag was verified to carry the DVC `.dir` hashes of the tag(s) it supersedes.

#### The one number that changed meaning

**Old `task_a-corpus-v2-2026-07-22` is new `corpus/task-a-v1`.**

The dated scheme numbered two *registrations* of identical bytes as v1 and v2:
`task_a-corpus-v1` was registered via `dvc commit` trusting a `--dry-run` equivalence check,
and `task_a-corpus-v2` marked the first real `dvc repro` of the same data. Both name
`3d6a4d3e…` / `6bb5eb6f…`, and so does `task-a-sft-v1`. Three tags, one corpus.

The new numbering follows the `task-a-sft-v1/v2/v3` line, where each number is a genuinely
different corpus. So under the new scheme there is exactly one v1, and the registration
history that the old v1/v2 pair recorded lives in `corpus/task-a-v1`'s annotation instead.

The visible consequence: `sft-gemma4-v3`'s old annotation says it trained on
"task_a-corpus-v2-2026-07-22", which reads as a version mismatch against its new name
`model/sft-gemma4-v3-on-task-a-v1`. It is not a mismatch — it is the same bytes under the
corrected number.

#### Not renamed: `CLAUDE.md`

`CLAUDE.md`'s risk register (R15, R17, R22 and others) refers to tags as part of a
historical account of what was known when. Rewriting those names would misrepresent the
record. This table is the bridge; read a tag name in `CLAUDE.md` as the name it had at the
time of writing.
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
python3 scripts/materialize_dvc_lineage.py --rev model/sft-gemma4-v2-on-pre-r12 --out /tmp/sft_v2
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
python3 scripts/materialize_dvc_lineage.py --rev model/sft-gemma4-v3-on-task-a-v1 --out /tmp/sft_v3

# Cold cache (fresh machine) — fetch the blobs first.
# NOTE: `dvc fetch --rev` is NOT supported by the DVC version pinned here (CLAUDE.md R22).
# Use -T/--all-tags, or check the tag's lock file out first:
dvc fetch -T checkpoints/sft_cat_a/gemma-4-26B-A4B-it
#   or: git checkout model/sft-gemma4-v3-on-task-a-v1 -- dvc.lock && dvc fetch
python3 scripts/materialize_dvc_lineage.py --rev model/sft-gemma4-v3-on-task-a-v1 --out /tmp/sft_v3
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
that is exactly how `model/sft-gemma4-v2-on-pre-r12` came to exist.

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
- **Shared checkpoint path** (§4) — the historical lineages (`v2`/`v3`/`v4`) all sit in
  `checkpoints/sft_cat_a/gemma-4-26B-A4B-it`, so materialize to separate paths to compare
  them. Fixed going forward: `_resolve_output_dir()` in `training/{sft,grpo,dpo}.py` honours
  an explicit `output_dir` config key, and each new cell must set one.
- **`dvc.lock` stage disagreement** (§1.1) — the GRPO stage's dep is pinned to v2 while the
  SFT stage's out is v3. Benign until the GRPO stage reruns, at which point it would consume
  v3 weights while its config names `checkpoint-500` — a different model in each lineage.
- **Untagged derived sets** — `data/output/grpo/task_a` and `data/output/preference/task_a`
  have never been tagged and there is no `derived/` namespace tag yet (§3.1). Both are also
  still derived from `corpus/task-a-v2` while the working corpus is v3.
- **`corpus/task-a-v0` candidate** — the pre-R12 corpus that
  `model/sft-gemma4-v2-on-pre-r12` trained on (`8ef8681808…`, 131 files, ~228 MB) has never
  been tagged, and two annotations disagree about whether its bytes survive. Settle it with
  the DVC CLI; if they are present, tag them and re-point that model tag's `-on-` suffix.
- **`task_a_grpo` `dvc.yaml` `cmd:` indentation** — the block-folding bug fixed for
  `task_a_sft_clean` / `task_a_sft_splits` in `666fe86` was not fixed for this stage.
