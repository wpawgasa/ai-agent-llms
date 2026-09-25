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
stages that last ran at different times will disagree about the same path. As of 2026-09-11,
verified directly from `dvc.lock`:

| Stage | Role | Path | Hash | Lineage |
|---|---|---|---|---|
| `task_a_sft_gemma4_26b_a4b` | `outs` | `checkpoints/sft_cat_a/gemma-4-26B-A4B-it` | `57e40028fe…` | `model/sft-gemma4-v4-on-task-a-v2` |
| `task_a_grpo_gemma4_26b_a4b` | `deps` | `checkpoints/sft_cat_a/gemma-4-26B-A4B-it` | `f89238076f…` | `model/sft-gemma4-v2-on-pre-r12` |
| `task_a_grpo_gemma4_26b_a4b` | `deps` | `data/output/grpo/task_a` | `60831c6695…` | untagged; matches no known `derived/` tag |

Three-way, not two: the GRPO **training** stage's checkpoint dependency is pinned to the
*oldest* SFT lineage while the SFT stage's own output has since moved through v3 and to v4,
and its GRPO-prompt dependency is a third, still older hash that predates both
`derived/task-a-grpo-v3` and the pre-v3 set it replaced (§9) — evidence this stage has not
rerun since very early in the project.

This is not corruption — it is the format working as designed. But it *is* a signal that
the GRPO training stage is stale with respect to both its dependencies, and `dvc status`
reports it as `changed deps`. Treat a same-path disagreement between two stages as a prompt
to check which one is out of date before running anything (this same signature is what
exposed the R12 stale-corpus bug; see CLAUDE.md). **Do not confuse this with the `task_a_grpo`
*data* stage** (singular — the L3–L5 prompt filter, distinct from `task_a_grpo_gemma4_26b_a4b`
the training run): that one's dependency was found stale and fixed on 2026-09-10 (§9).

---

## 2. Current inventory

```bash
git tag -n1 -l 'corpus/*' 'model/*'    # one-line summaries
git tag -n40 model/sft-gemma4-c2-on-task-a-v2   # full annotation, incl. hashes
```

### Benchmark corpora (Phase 1 evaluation)

A different pipeline from the SFT corpus below. Every stage carries `frozen: true`.
Score a model on ONE benchmark version; the versions are separate scales (R28, R29).

| Tag | Restores | Contents |
|---|---|---|
| `corpus/task-a-benchmark-v1` | `data/output/benchmark/task_a` | 258 convs, text. Predates the tool-call stay rule (R25). |
| `corpus/task-a-benchmark-voice-v1` | `data/output/benchmark/task_a_voice` | 250 convs, voice, 50/level. |
| `corpus/task-a-benchmark-v2` | `data/output/benchmark/task_a_v2` | 258 convs, text, on the stay rule; 143 authored inserts. |
| `corpus/task-a-benchmark-v3` | `data/output/benchmark/task_a_v3` + `task_a_voice_v2` | 258 + 250. Traceability repair (R28). Superseded. |
| **`corpus/task-a-benchmark-v4`** | `data/output/benchmark/task_a_v4` + `task_a_voice_v3` | **259 + 250 — the current benchmark.** The 19 two-tool convs replaced. |

### Corpora (SFT training)

| Tag | Restores | Contents |
|---|---|---|
| `corpus/task-a-v1` | `data/output/sft/task_a_splits` | 5,549 convs, all text; 4,716 / 554 / 279. |
| `corpus/task-a-v2` | `data/output/sft/task_a_splits` | 5,543 convs, stay convention; 4,711 / 554 / 278. |
| **`corpus/task-a-v3`** | `data/output/sft/task_a_splits` | **9,932 convs** (7,043 text + 2,889 voice); 8,441 / 992 / 499. |
| *(untagged)* | `data/output/sft/task_a_splits_v4` | 9,891 convs — the R30 stage-1 repair. Its own path, so no tag is needed to tell it apart. |

The first three share ONE path. Only the tag tells them apart — see §4.

### Derived sets

| Tag | Restores |
|---|---|
| `derived/task-a-heldout-v3` | `data/output/heldout/cat_a_v3_test_not_in_v2`, `cat_a_v3_test_voice` |
| `derived/task-a-grpo-v3` | `data/output/grpo/task_a` |

### Models

| Tag | Restores | Recipe / result |
|---|---|---|
| `model/sft-gemma4-v2-on-pre-r12` | `checkpoints/sft_cat_a/gemma-4-26B-A4B-it` | ckpt-1000 baseline, untagged corpus. |
| `model/sft-gemma4-v3-on-task-a-v1` | same path | C0, `all_tokens` @ 4096 — held-out 0.5709. |
| `model/sft-gemma4-v4-on-task-a-v2` | same path | C0 on the stay corpus — held-out 0.5120. |
| `model/sft-gemma4-c2-step500-on-task-a-v2` | `checkpoints/sft_cat_a_c2_step500` | C2 mid-run snapshot, incomplete. |
| **`model/sft-gemma4-c2-on-task-a-v2`** | `checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it` | **C2 — held-out 0.7595, best 26B.** |
| `model/sft-gemma4-e4b-c2-on-task-a-v3` | `checkpoints/sft_cat_a_e4b/gemma-4-E4B-it` | E4B, C2 recipe on corpus v3. |
| `model/sft-gemma4-12b-c2-on-task-a-v3` | `checkpoints/sft_cat_a_12b/gemma-4-12B-it` | 12B, same recipe — the size arm. |
| **`model/sft-gemma4-e4b-toolresults-on-task-a-v3`** | `checkpoints/sft_cat_a_e4b_textturns/gemma-4-E4B-it` | **Best on v4: 0.8332 text / 0.8046 native.** |
| `model/sft-gemma4-12b-toolresults-on-task-a-v3` | `checkpoints/sft_cat_a_12b_textturns/gemma-4-12B-it` | 12B twin: 0.8084 text / 0.7864 native. |
| `model/sft-gemma4-e4b-on-repaired-corpus-v4` | `checkpoints/sft_cat_a_e4b_corpus_v4/gemma-4-E4B-it` | R30 null result: 0.8308 / 0.8189. |

The v3/v4/C2 held-out composites share one 206-row set and are comparable to each other.
The v4-benchmark numbers are a different scale again — never mix the two.

**Untagged:** the GRPO Cat A lineage (`checkpoints/grpo_cat_a/gemma-4-26B-A4B-it`) and the
DPO one (`checkpoints/dpo_cat_a/gemma-4-26B-A4B-it`). Both are reachable only through
`dvc.lock`'s current entry, and `dvc gc -w` would delete them (§6).

---

## 2b. Checking out a specific version

Two different jobs, and mixing them up is where the time goes:

- **Replace the workspace copy** — normal, for "give me version X to work with".
- **Materialize out-of-place** — for comparing two versions, or when the version you want
  shares a path with one you must not disturb (§4). Never touches the workspace.

Every command below was run on 2026-09-25; the counts are what it printed.

### Out-of-place (safe, preferred for comparisons)

```bash
# Benchmark: any version, into /tmp, workspace untouched
python scripts/materialize_dvc_lineage.py \
    --rev corpus/task-a-benchmark-v3 \
    --dvc-path data/output/benchmark/task_a_v3 \
    --out /tmp/bench_v3                      # -> 6 files, 258 conversations

# Checkpoint: --dvc-path is the stage's OUT, not the tag name
python scripts/materialize_dvc_lineage.py \
    --rev model/sft-gemma4-e4b-on-repaired-corpus-v4 \
    --dvc-path checkpoints/sft_cat_a_e4b_corpus_v4/gemma-4-E4B-it \
    --out /tmp/e4b_v4                        # -> 80 files

# Training corpus: v1/v2/v3 SHARE one path, so the tag is the only difference
python scripts/materialize_dvc_lineage.py \
    --rev corpus/task-a-v2 \
    --dvc-path data/output/sft/task_a_splits \
    --out /tmp/corpus_v2                     # -> 3 files, 5,543 conversations
```

### When the blobs are not in the local cache

`materialize_dvc_lineage.py` reads only the local cache, so on a fresh machine — or after a
cache cleanup — it exits with "not in cache". **`dvc fetch --rev` does not exist in the
pinned DVC (3.67.1); only `-T/--all-tags` and `-A/--all-commits` do** (the script's own error
message still suggests `--rev`; ignore it). Fetch by checking the tag's lock file out first,
then put yours back:

```bash
cp dvc.lock /tmp/lock.bak
git checkout corpus/task-a-v2 -- dvc.lock      # NOTE: this also STAGES dvc.lock
dvc fetch data/output/sft/task_a_splits        # -> 4 files fetched
cp /tmp/lock.bak dvc.lock
git restore --staged --worktree dvc.lock       # undo the staging, or your next commit carries it
python scripts/materialize_dvc_lineage.py --rev corpus/task-a-v2 \
    --dvc-path data/output/sft/task_a_splits --out /tmp/corpus_v2
```

`dvc fetch -T <path>` also works and needs no lock-file juggling, but it pulls that path for
EVERY tag — expensive on a checkpoint path carrying several lineages.

### Into the workspace

```bash
# The version dvc.lock currently names (after a normal clone or git checkout)
dvc pull data/output/benchmark/task_a_v4 checkpoints/sft_cat_a_e4b_textturns/gemma-4-E4B-it

# A different version of a SHARED path: check out the tag's lock file, then checkout
git checkout corpus/task-a-v2 -- dvc.lock
dvc checkout data/output/sft/task_a_splits     # workspace now holds v2
# ... and to go back:
git restore --staged --worktree dvc.lock && dvc checkout data/output/sft/task_a_splits
```

Checking out the whole tag (`git checkout <tag>` then `dvc checkout`) works too and is
simplest when you want that commit's code as well — but it detaches HEAD, so
`git switch main` afterwards.

### Which version is on disk right now?

```bash
# What each tag says the shared path should hash to
for t in corpus/task-a-v1 corpus/task-a-v2 corpus/task-a-v3; do
  echo "$t $(git show $t:dvc.lock | grep -A4 'path: data/output/sft/task_a_splits$' \
        | grep -oE '[a-f0-9]{32}\.dir' | head -1)"
done
```

A cheaper tell for the SFT corpus: `cat data/output/sft/task_a_splits/*.jsonl | wc -l`
gives 5,549 (v1), 5,543 (v2) or 9,932 (v3). For a benchmark directory, the result JSONs
record `data_sha256` per `--data` path (R29), so a stored result names exactly the bytes
it scored.

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
- **`dvc.lock` stage disagreement (§1.1) — STILL OPEN, and worse than previously described.**
  `task_a_grpo_gemma4_26b_a4b` (the GRPO **training** run) depends on
  `checkpoints/sft_cat_a/gemma-4-26B-A4B-it` at `f89238076f…` — `model/sft-gemma4-v2-on-pre-r12`,
  the oldest tagged lineage — while the SFT stage's own output has since moved through v3 to
  v4. Its dependency on `data/output/grpo/task_a` (`60831c6695…`) is a third, still older hash
  matching no known `derived/` tag. Benign until this stage reruns, at which point it would
  train on stale prompts against a config that may still name an old checkpoint. **Correction:**
  an earlier revision of this document marked this resolved by conflating it with the
  `task_a_grpo` *data* stage fix below — a different stage, on a different dependency. This one
  is untouched.
- **Untagged derived sets — PARTIALLY RESOLVED.** The `task_a_grpo` *data* stage (the L3–L5
  prompt filter — not the training stage above) was regenerated from `corpus/task-a-v3` on
  2026-09-10 and tagged `derived/task-a-grpo-v3`; its dependency on the SFT splits now matches
  current. `data/output/preference/task_a` remains untagged because it does not exist yet —
  mining on-distribution negatives needs a GPU pass over a checkpoint, and no model has been
  trained on `corpus/task-a-v3`.
- **`corpus/task-a-v0` candidate** — the pre-R12 corpus that
  `model/sft-gemma4-v2-on-pre-r12` trained on (`8ef8681808…`, 131 files, ~228 MB) has never
  been tagged, and two annotations disagree about whether its bytes survive. Settle it with
  the DVC CLI; if they are present, tag them and re-point that model tag's `-on-` suffix.
- **`task_a_grpo` `dvc.yaml` `cmd:` indentation** — the block-folding bug fixed for
  `task_a_sft_clean` / `task_a_sft_splits` in `666fe86` was not fixed for this stage.
- **Benchmark corpora — RESOLVED 2026-09-11.** `task_a_benchmark`'s stale lock (11 of 17
  files recorded) and `task_a_benchmark_voice`'s complete absence from `dvc.lock` (CLAUDE.md
  R21) are both fixed: re-committed, pushed, frozen, and tagged as
  `corpus/task-a-benchmark-v1` / `corpus/task-a-benchmark-voice-v1`.
