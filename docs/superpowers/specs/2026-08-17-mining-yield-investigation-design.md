# Spec: Mining Yield Discrepancy — Investigation + Contingent Fix

**Status:** design approved (2026-08-17), pending implementation
**Prior art:** `dc84adb` (mining run: 51/399 = 12.8% error rate), `docs/cat_a_c2_heldout_result.md` (held-out audit: 38.0% wrong on the 71 tool-bearing rows of 206), CLAUDE.md R18 (why preference learning replaces GRPO on this task).

---

## 1. Problem

`scripts/mine_model_negatives.py` mines on-distribution DPO negatives by generating greedily from checkpoint C2 on TRAIN prompts and keeping the generations that are wrong. The first run sampled 399 TRAIN prompts (`--n-prompts 400 --tool-share 0.75`) and found only 51 wrong (12.8%). The held-out audit that produced C2's headline numbers found the checkpoint wrong on 38.0% of tool-bearing rows (27 of 71) — drawn from a contamination-free TEST-derived set the checkpoint never trained on. `dc84adb`'s commit message names two untested explanations and does not decide between them:

1. **Split effect** — TRAIN rows were seen during SFT; TEST rows were not.
2. **Classifier effect** — `_classify()` in `mine_model_negatives.py` may flag fewer rows as wrong than the held-out audit's strict composite scorer does.

A quick read of `docs/cat_a_c2_heldout_result.md`'s residual-failure taxonomy (44 perfect / 18 wrong-args / 9 silent, of 71) shows nearly the same failure axes `_classify()` checks (no call / wrong tool / wrong args), which weakens explanation 2 as the dominant factor — but this is an inference from documentation, not a measurement. The gap matters because it determines the practical mining yield: at true generalization difficulty, mining is far cheaper per negative than the 12.8% figure suggests.

## 2. Objective & acceptance criteria

Decompose the gap into the split effect and the classifier effect, each measured while holding the other variable fixed, and — if the split effect is confirmed dominant — ship (but do not activate by default) a fix that mines from the GRPO `validation` split instead of `train`, without compromising `dpo.py`'s independent R5 held-out reward-hacking guardrail.

**Acceptance criteria:**
1. `scripts/investigate_mining_yield.py` exists, runs its classifier-effect probe against a stored audit JSON with **zero new GPU generation**, and its split-effect probe reuses `mine_model_negatives.py`'s own `_select_prompts`/`_classify` code path unchanged except for the `--split` argument.
2. `reserve_guardrail_slice()` exists, is unit-tested, and both `mine_model_negatives.py` (when run with `--split validation`) and `dpo.py`'s `_build_heldout_callback` consult it, so the two can never draw from an overlapping fingerprint set.
3. All new logic is unit-testable without a GPU. The one thing that is genuinely out of scope for this implementation pass is *running* the split-effect probe for real — that needs `.venv-train` and either a GPU or a pre-existing cached audit JSON, neither available in this environment.
4. `dvc.yaml`'s `task_a_preference_model_negatives` stage keeps `--split train` as its actual `cmd` — the validation-mining path is documented as available, contingent on the probe's real-world result, not switched on.

## 3. Architecture

### 3.1 Diagnostic script — `scripts/investigate_mining_yield.py`

```
Probe 1 (classifier effect, cheap):
  --from-audit-json runs/audit/heldout_c2_ckpt1767_v2corpus.json
    └─ load stored {completion, ground_truth} per row (206 rows, 71 tool-bearing)
    └─ _classify() from mine_model_negatives.py, reused via import
    └─ report: classify-based wrong-rate on the SAME 71 rows that produced 38.0%

Probe 2 (split effect, needs GPU):
  --checkpoint <ckpt> --data-dir data/output/grpo/task_a --split validation
    --n-prompts 400 --tool-share 0.75 --seed 42   (defaults match the TRAIN run exactly)
    └─ _select_prompts() + _generate_for_checkpoint() + _classify(), all reused
       from mine_model_negatives.py, only the split differs from the known 12.8% run
    └─ report: classify-based wrong-rate on VALIDATION

Output: runs/audit/mining_yield_investigation_<ts>.json + a printed 4-row table:
    train      (classify, known)   12.8%
    validation (classify, new)     ??
    held-out   (classify, new, Probe 1, same 71 rows as below)   ??
    held-out   (composite, known)  38.0%
```

Reading the table: (validation − train) isolates the split effect with the classifier held constant. (held-out composite − held-out classify) isolates the classifier effect with the split held constant. If both deltas are small, the explanation is something else entirely and this decomposition will say so rather than force-fit one of the two hypotheses.

**Safety boundary, explicit in the script's docstring:** Probe 2 hard-refuses `--split test`, mirroring `mine_model_negatives.py`'s existing guard — this script must never generate a fresh pass over the true held-out set. Probe 1 only *reclassifies* completions a prior audit already generated; it writes no `chosen`/`rejected`-shaped file, so its output can never be mistaken for a training artifact even if someone later globs `runs/audit/*.json` into a data pipeline by accident.

### 3.2 The fix — mine from validation, reserve a guardrail slice

`reserve_guardrail_slice(data_dir, split, reserved_fraction=0.2, seed=42) -> set[str]` lands in `src/llm_workflow_agents/data/heldout_clean_set.py`, next to `user_turn_fingerprint` (which it reuses — this is the same partitioning primitive already used for the held-out contamination guard, not a new mechanism). It deterministically hashes each conversation's fingerprint into "minable" or "guardrail-reserved" by `reserved_fraction`, so two independent call sites agree on the same partition without a second physical file:

```
mine_model_negatives.py --split validation
    └─ excludes guardrail-reserved fingerprints, same shape as the existing
       --heldout exclusion (a second exclusion set, same mechanism)

dpo.py::_build_heldout_callback
    └─ selects held_out_rows ONLY from the guardrail-reserved fingerprints,
       instead of "first N rows of validation" — zero overlap by construction
```

`dpo_cat_a.yaml` gains `data.guardrail_reserved_fraction: 0.2` (documented, not required — defaults to 0.2 if absent).

**Why this and not a second physical file:** a fingerprint partition needs no data regeneration when `reserved_fraction` changes, stays correct if `validation.jsonl` is regenerated from a newer corpus lineage (fingerprints are content-derived, not row-index-derived), and mirrors the exact pattern `heldout_clean_set.py` already uses for the held-out/train-test guard — one mental model for "how do we keep two consumers of the same file from overlapping" instead of two.

### 3.3 What does NOT change in this pass

- `dvc.yaml`'s `task_a_preference_model_negatives` `cmd` keeps `--split train`. A commented note is added pointing at this spec and `investigate_mining_yield.py`, but the pipeline's actual behavior is unchanged until someone acts on the probe's real result.
- `mine_model_negatives.py`'s default `--split` stays `"train"` — only the (already-supported) `"validation"` value gains the new exclusion behavior.

## 4. Risks

| Risk | Mitigation |
|---|---|
| No cached audit JSON exists on the machine that eventually runs this | Probe 1 fails loudly with a clear message naming the expected path; Probe 2 (split effect) is independently useful without it. |
| `reserved_fraction` too small leaves too few guardrail rows for a stable R5 signal | Default 0.2 of validation (validation is currently 289 conversations per R17 — ~58 reserved, comparable to the 50 the guardrail already samples from unrestricted validation today). Configurable if that proves too few. |
| Someone flips `dvc.yaml` to `--split validation` without reading this spec | The stage's `desc` field gets an explicit warning naming the guardrail dependency; `mine_model_negatives.py` itself refuses to mine reserved-fingerprint rows regardless of who runs it, so the worst case is a silently smaller minable pool, not a silent guardrail leak. |
| **The reserved slice protects the R5 guardrail ONLY — not DPO's own eval split.** `configs/training/dpo_cat_a.yaml`'s `data.validation_source` (`data/output/preference/task_a/validation.jsonl`) is built by `dvc.yaml`'s `task_a_preference_pairs` stage (`build_preference_pairs.py --split validation`) from the SAME GRPO validation conversations `reserve_guardrail_slice` partitions. Flipping mining to `--split validation` would draw negatives from the ~80% non-reserved population that also feeds DPO's eval pairs, contaminating `eval_loss` and checkpoint selection with rows that are simultaneously training-mix negatives. | **Not mitigated by what this branch builds — documented only.** The two-way partition here (minable / R5-guardrail) is insufficient; flipping the mining source additionally requires a **three-way** partition (minable / DPO-eval / R5-guardrail) and re-deriving `data/output/preference/task_a/validation.jsonl` from a slice disjoint from the minable pool. That is deliberately out of scope for this pass, since the mining source is not being flipped. The caveat is stated in the `task_a_preference_model_negatives` stage `desc`, beside `data.validation_source` in `dpo_cat_a.yaml`, and here. |
| The two `reserve_guardrail_slice` call sites (`mine_model_negatives.py`'s `--guardrail-reserved-fraction`/`--guardrail-reserved-seed` and `dpo_cat_a.yaml`'s `data.guardrail_reserved_fraction`/`guardrail_reserved_seed`) drift apart, so mining and the guardrail reserve *different* sets and overlap again — silently. | Documented as a hard "these MUST match" requirement in three places a human sees before running the stage: the argparse `help=` text, the config comment, and the DVC stage `desc`. A mismatch in either value **or** in the corpus path (`--data-dir` vs `heldout_data_source`) breaks the guarantee. `tests/unit/test_guardrail_disjointness.py` proves both sides of the property: identical parameters ⇒ identical sets; differing parameters ⇒ different sets. A runtime fingerprint-disjointness assertion inside `dpo.py` would be strictly better and is deferred. |

## 5. Testing plan (all pure, no GPU)

- `reserve_guardrail_slice`: disjointness from its own complement; stability across repeated calls with the same seed; `reserved_fraction` honored within rounding; composes correctly with `load_prefix_fingerprints` (a fingerprint can be in the held-out-contamination set *and* the guardrail-reserved set — they guard different things and are allowed to overlap).
- The table/decomposition function in `investigate_mining_yield.py`: given four synthetic rates, produces the correct deltas and labels — no model or checkpoint needed.
- Probe 1's `_classify()` reuse: synthetic `{completion, ground_truth}` rows through the same code path `mine_model_negatives.py`'s own tests (if any) already cover — extend rather than duplicate.
- `dpo.py::_build_heldout_callback`'s row-selection change: unit test that it selects only from a given reserved-fingerprint set, with a mocked `_load_grpo_jsonl` return.

## 6. Rollout order

1. Build `reserve_guardrail_slice` + tests.
2. Build `investigate_mining_yield.py` + tests.
3. Wire `dpo.py`'s guardrail to the reserved slice (safe regardless of mining source — it only narrows which validation rows the guardrail can see, a strict subset of its current behavior).
4. Update `dvc.yaml`'s stage `desc` with the contingency note. Do not change its `cmd`.
5. **Out of scope for this implementation pass:** running `investigate_mining_yield.py` for real, reading its result, and — only if it confirms the split effect — flipping `dvc.yaml`'s `cmd` to `--split validation` and re-running the mining stage.
