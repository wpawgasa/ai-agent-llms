# Generation runbook: +3,000 voice, +1,500 harder (L4/L5)

**For a machine with teacher-model API access.** Generation needs no GPU and no
existing corpus. It writes new files only; nothing is overwritten.

Target after the merge: about 10,000 conversations, roughly 45% L4/L5 and 30%
voice, up from 5,543 conversations at 32.5% L4/L5 and no voice.

The 30% voice share matches `DEFAULT_VOICE_WEIGHT = 0.30`, the weight
`blend_modality_scores` already uses for the Phase 1 voice stratum (R20). Voice
conversations carry complexity levels too, so the 3,000 voice rows contribute
about 1,200 of the L4/L5 total.

---

## 0. Before you start

**Rotate the API keys.** The Gemini, OpenAI and Anthropic keys in `.env` were
printed into a session transcript on 2026-08-31. Replace all three before use.

Set **one** key, matching the teacher you pick:

| teacher prefix | env var |
|---|---|
| `gemini-*` | `GEMINI_API_KEY` |
| `gpt-*` | `OPENAI_API_KEY` |
| `claude-*` | `ANTHROPIC_API_KEY` |

Both scripts load `.env` automatically.

```bash
git clone <repo> && cd ai-agent-llms && git checkout main
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

No `dvc pull` is needed. Generation does not read the existing corpus.

---

## 1. Voice smoke test — do this first, it is a real gate

R20: this is the only pre-flight signal that the teacher can produce the voice
format at all. Placeholder rows are format-perfect and deterministic, so a total
teacher failure otherwise looks exactly like success.

```bash
./scripts/generate_voice_data.sh \
    --smoke-test \
    --teacher-model gemini-3.5-flash \
    --output-dir /tmp/voice_smoke
```

`--teacher-model` passed explicitly is what makes this a **live** smoke rather
than an offline placeholder run.

**Read the output.** `scripts/check_voice_batch.py` runs at the end. Proceed
only if the placeholder share is low and `<S>` chunking looks right in the
generated rows. If the teacher cannot hold the format on 15 conversations, it
will not hold it on 1,500.

---

## 2. Voice batch — 3,000 conversations

```bash
./scripts/generate_voice_data.sh \
    --total 3000 \
    --teacher-model gemini-3.5-flash \
    --barge-in-rate 0.25 \
    --max-placeholder-share 0.10
```

- writes to `data/output/sft/task_a_voice/`
- seed defaults to **4242**, deliberately different from the text corpus seed of
  42 so the two batches draw different domains and workflows
- `--total 3000` overrides the per-leg table with a uniform count: 15 legs
  (5 levels x 3 languages), so 200 per leg
- the quality gate **fails the run** above a 10% placeholder share. Do not pass
  `--skip-gate` on a paid run.

---

## 3. Harder batch — 1,500 conversations at L4/L5 only

Use `generate_sft_until_target.py`, not `generate_sft_data.sh`. The shell script
always loops all five levels and has no level filter. This one takes `--levels`,
verifies each batch, drops unqualified rows and keeps going until the
**qualified** count reaches target.

```bash
.venv/bin/python scripts/generate_sft_until_target.py \
    --levels L4,L5 \
    --samples-per-leg 250 \
    --seed 8484 \
    --teacher-model gemini-3.5-flash \
    --output-dir data/output/sft/task_a \
    --require-tool-stay \
    --max-workers 8
```

**The seed matters.** Generation is deterministic per seed, and the existing
corpus used **42**. Reusing 42 would regenerate samples the corpus already has.
8484 is distinct from both 42 (text) and 4242 (voice).

- 2 levels x 3 language legs = 6 legs, 250 each = 1,500 qualified conversations
- `--require-tool-stay` is the default and must stay on: it enforces the v2
  convention that a turn issuing a `<tool_call>` stays in its state. Cell C2 was
  trained on that convention (R15/R17) and mixing conventions would corrupt the
  corpus.
- output files are timestamped, so this **adds** to
  `data/output/sft/task_a/` and overwrites nothing
- L4/L5 conversations are the long ones, so this is the slow leg. Raise
  `--max-workers` if the provider's rate limit allows.

---

## 4. Check what you got

```bash
# level mix of the new hard batch
python3 - <<'PY'
import json, glob, collections
c = collections.Counter()
for f in glob.glob("data/output/sft/task_a/*.jsonl"):
    for line in open(f):
        c[json.loads(line).get("complexity_level", "?")] += 1
print(dict(sorted(c.items())), "total", sum(c.values()))
PY

# voice batch: placeholder share and realised barge-ins
.venv/bin/python scripts/check_voice_batch.py \
    --input-dir data/output/sft/task_a_voice
```

`--input-dir` is required — the directory is not accepted as a positional
argument. Use the venv interpreter, not bare `python3`: the script's imports are
stdlib-only at the top, but it imports `find_voice_violations` from the package
inside the function that runs the format check, so a bare `python3` gets as far
as printing the placeholder share and then dies on `ModuleNotFoundError`.

Expect roughly 1,500 new L4/L5 rows and 3,000 voice rows. `check_voice_batch.py`
warns where the teacher was asked for a barge-in and did not deliver one; a
modest gap is normal, a total absence is not.

It scores **every** `*.stats.json` in the directory, with no date filter, so
point it at a directory holding only the batch you mean to judge. A smoke run
reusing an output directory from an earlier run scores both together and can
fail a batch that is fine on its own.

---

## 5. Send the data back

Two directories carry everything:

```
data/output/sft/task_a/         # existing rows + the new L4/L5 files
data/output/sft/task_a_voice/   # the voice batch
```

Preferred, if the machine has the GCS credentials:

```bash
dvc add data/output/sft/task_a data/output/sft/task_a_voice
dvc push
git add -A && git commit -m "data: +3000 voice, +1500 L4/L5" && git push
```

Otherwise archive both directories and transfer them.

---

## 6. What happens on the training machine — not your job, listed so the
##    hand-off is clear

1. `clean_task_a_sft.py` with a repeatable `--input-dir` for each source
   directory, then `split_task_a_sft.py`.
2. The split **must** run with `--assert-unmoved data/output/sft/task_a_splits`
   against the prior splits. R20: shuffling is per modality so an additive batch
   is additive, but adding rows to an existing modality group still reshuffles
   that group. The guard matches by `user_turn_fingerprint` and exits non-zero
   on any row that changed split.
3. Rebuild the pinned 206-row held-out set and confirm it still verifies against
   the stored C2 audit. That set is the only link back to C2's 0.7595, and it is
   text-only by construction — voice must never be blended into it.
4. Training must use the `response_only` loss recipe. `all_tokens` cannot honour
   the per-message `loss: false` flag that a voice barge-in turn carries, and
   would train the model to emit the `<unspoken>` marker.

---

## Things that will bite

**Do not pass `--skip-gate`** on the paid voice run. It exists for debugging.

**Do not reuse seed 42** for the hard batch.

**Do not use `generate_sft_data.sh`** for the hard batch. It regenerates all
five levels and there is no level filter.

**Do not delete or regenerate `data/output/benchmark/task_a`.** That stage is
frozen (R21) and holds the 258 teacher-generated conversations behind the
current Cat A ranking.
