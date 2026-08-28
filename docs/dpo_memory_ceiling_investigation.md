# Cat A DPO — the memory ceiling, and which eval path causes it

**Status as of 2026-08-28.** Preference learning (R18) is implemented and the
training loop works, but no Cat A DPO run has yet completed with the R5
held-out guardrail enabled. Fifteen smoke runs isolated the *trigger*; a
sixteenth found the *cause*, and it is not the one §5 predicted.

**The short version, if you read nothing else:** `load_in_4bit: true` quantizes
only 0.77 GiB of this model. The MoE experts — 42.5 GiB, 94.5% of the weight
mass — are fused 3-D `nn.Parameter` tensors that bitsandbytes cannot see, so
they stay bf16. The run carries **45.8 GiB of weights instead of ~13 GiB**.
That ~32 GiB overhead, not a leak, is why headroom was ever ~1 GiB. §8 has the
measurement. The `use_cache` hypothesis in §5 is **refuted** (§5, §8).

All runs start from `checkpoints/sft_cat_a_c2/gemma-4-26B-A4B-it/checkpoint-1767`
— the C2 checkpoint (R17), unchanged throughout.

---

## 1. Headline

**The R5 held-out guardrail is what breaks the run. HF Trainer's own periodic
evaluation is not.**

A two-arm bisect at `max_seq_length: 5120`, same fixture, same allocator
regime, 6 training steps each:

| Run | Guardrail | Trainer eval | Result |
|-----|-----------|--------------|--------|
| smoke14 | **on** (step 3) | off | fires at step 3 → **OOM at step 4** |
| smoke15 | off | **on** (steps 3, 6) | fires twice → **6/6 steps, 160.6 s, checkpoint-6 written** |
| smoke16 | **on** (step 3) | off | fires at step 3 → **OOM at step 4**, *with* the §5 fix and on a 93 GiB card |

The failure is always the *training step immediately following a guardrail
eval*, never the eval itself. `fed2ea5` added `torch.cuda.empty_cache()` to the
callback's `finally` block; that recovered headroom but did not clear the wall.

Something in `dpo.py::_build_heldout_callback` still changes the memory state
seen by the next optimizer step. Section 5 gives the leading hypothesis.

---

## 2. The smoke ladder

| Run | Cap | Notes | Outcome |
|-----|-----|-------|---------|
| smoke1 | 8192 | both evals on | OOM |
| smoke2 | 5120 | both evals on | OOM |
| smoke3 | 8192 | `precompute_ref_log_probs` on | OOM |
| smoke4 | 6144 | precompute on | OOM in backward (wanted 5.78 GiB, 0.49 free) |
| smoke5 | 8192 | `use_liger_kernel: true` | `NotImplementedError` — Liger DPO not implemented for PEFT |
| smoke6 | 8192 | liger on, precompute off | same `NotImplementedError` |
| smoke7 | 5120 | both evals on | **step-4** OOM, 52 MiB short, 525 MiB reserved-but-unallocated |
| smoke8 | 5120 | + `empty_cache()` fix | **step-4** OOM, 276 MiB short |
| smoke9 | 4096 | both evals on | **step-4** OOM, 276 MiB short |
| smoke10 | 5120 | **both evals off** | **6/6 steps**, 26.83 s/it, eval_loss 0.6546 |
| smoke11 | 6144 | cap raised — void experiment | OOM at step 0; never reached the bisect |
| smoke12 | 5120 | discarded, see §3 | killed before completion |
| smoke13 | 5120 | both evals on, standby unset | **step-4** OOM, 88 MiB short |
| smoke14 | 5120 | **guardrail on, trainer eval off** | **step-4** OOM, 350 MiB short |
| smoke15 | 5120 | **guardrail off, trainer eval on** | **6/6 steps**, 160.6 s |
| smoke16 | 5120 | smoke14 repeat, **`use_cache` fix in**, H100 NVL 93.09 GiB | **step-4** OOM, 350 MiB short — see §8 |

Settled along the way:

- **`precompute_ref_log_probs: true` is required.** Without it Unsloth's compiled
  DPO trainer materializes four `[2, S, 262144]` fp32 logits tensors per step.
- **`use_liger_kernel` is unusable.** It would be the correct fix — a fused
  chunked loss that never builds those tensors — but `_compute_loss_liger`
  raises `NotImplementedError: Liger DPO loss is not implemented for PEFT
  models`, and every Cat A DPO run is a LoRA adapter on C2.
- **The length cap is not the wall.** smoke8 (5120) and smoke9 (4096) fail
  identically. Lowering it does not buy a completed run.
- **Gradient checkpointing is already on.** Unsloth's `from_pretrained` defaults
  `use_gradient_checkpointing="unsloth"` and applies it at load. `dpo.py` skips
  `get_peft_model` (the checkpoint already carries the adapter), which makes it
  *look* like the knob was never set — it is, just not by our code. Not an
  available lever.
- **Headroom is ~1 GiB.** Steady state sits at ~78 GiB of 79.18 GiB. Any extra
  allocation tips it, which is why the failure is so sensitive.

Do not read much into the *size* of each shortfall (52 / 88 / 276 / 350 MiB).
It reports whichever allocation happened to fail first, not the magnitude of
the retention.

---

## 3. A container confound worth knowing about

The rebuilt container (2026-08-18) sets in `/etc/environment`:

```
UNSLOTH_VLLM_STANDBY="1"
UNSLOTH_DOCKER="1"
```

`scripts/run_phase2_dpo.sh:86` exports
`PYTORCH_ALLOC_CONF=expandable_segments:True` to fight allocator fragmentation.
Unsloth then **removes it at import**:

```
Unsloth: `UNSLOTH_VLLM_STANDBY` is on, but requires `expandable_segments`
to be off. We will remove `expandable_segments`.
```

So the runner exports a mitigation that a later import discards. DPO performs no
vLLM rollouts at all — that is the whole point of preference learning versus
GRPO (R18) — so standby buys nothing on this path. Launch with
`env -u UNSLOTH_VLLM_STANDBY` to keep the setting.

Two consequences:

1. Turning it off is a real but partial win: it narrowed one OOM from a 276 MiB
   shortfall to 88 MiB. It does not clear the wall.
2. **Runs from before the rebuild are not comparable with runs after it** unless
   the variable is unset. smoke12 was launched before this was noticed and was
   discarded rather than reported, because it would have compared a
   no-expandable-segments run against smoke10's expandable-segments baseline.
   smoke13–15 all ran with it unset.

---

## 4. What this does and does not establish

**Does:**
- The guardrail, not the trainer's evaluation, is responsible (§1).
- The trainer's periodic eval is safe to leave on — smoke15 ran it twice with no
  ill effect.
- The failure mode is reproducible and cheap to test: 6 steps, ~8 minutes
  end to end including the 26B load.

**Does not:**
- Identify the retention *mechanism*. §5 is a hypothesis, untested.
- Say anything about the real run's configuration. Every smoke used
  `eval_held_out_num_prompts: 2` at cap 5120;
  `configs/training/dpo_cat_a.yaml` specifies **50 prompts** at cap **8192**.
  Both are far heavier than anything tested here. Even after the guardrail is
  fixed, the real config is not proven to fit.
- Establish that 6 steps generalizes to 500. Nothing here rules out slow growth
  across a longer run.

---

## 5. Leading hypothesis for the residual retention

`_build_heldout_callback._evaluate` calls `model.generate()`
(`dpo.py:686-695`). Nothing in `training/` saves or restores `use_cache`:

```
$ grep -n use_cache src/llm_workflow_agents/training/*.py
(none)
```

HF Trainer sets `model.config.use_cache = False` at train start when gradient
checkpointing is on, precisely because a KV cache is dead weight during
training. `generate()` needs the cache and turns it back on. If it is not
restored, **every training step after the first guardrail eval allocates a KV
cache it did not allocate before** — which matches the observed signature
exactly:

- the failure is in the step *after* the eval, not the eval;
- `empty_cache()` does not help, because this is a fresh allocation each step
  rather than a parked block;
- HF Trainer's own eval does a forward pass, never a `generate()`, so it never
  flips the flag — consistent with smoke15 passing.

This is a hypothesis, not a finding. It is cheap to test: save
`model.config.use_cache`, force it `False` for training, restore in the
`finally` block alongside the existing `empty_cache()`, and re-run smoke14's
config. If it completes 6/6, that is the fix.

**REFUTED on GPU 2026-08-28 (smoke16).** The fix below is in the running code
and unit-tested, and the failure reproduced exactly: OOM at step 4, the step
after the step-3 guardrail eval, 350 MiB short — the same shortfall smoke14
reported. It also reproduced on a card with **14 GiB more headroom** than the
whole ladder had (H100 NVL, 93.09 GiB vs 79.18 GiB). A fixed retention would
have failed later, or not at all.

Keep the save/restore anyway: it is correct on its own terms — leaving
`use_cache` on after a `generate()` really does allocate a KV cache gradient
checkpointing had disabled — it is simply not what was tipping the run.

The reason the margin was ever thin enough for this to look plausible is in
§8, and it is arithmetic rather than a leak.

**Code side implemented 2026-08-19, GPU side closed 2026-08-28.** `_evaluate` now
snapshots `model.config.use_cache` right before `model.eval()` and restores
that snapshot in the `finally` block, after `model.train()` and before
`empty_cache()`. `tests/unit/test_dpo_heldout_guardrail.py::test_evaluate_restores_use_cache_after_generate_flips_it`
covers it with a fake model whose `generate()` sets `use_cache = True` as a
side effect (mirroring real HF behavior) — the test fails against the
pre-fix code and passes against the fix. No GPU was available in that
session, so §6 step 1 (re-run smoke14's config) is still outstanding; the
hypothesis is implemented, not confirmed.

---

## 6. Next state

**The leak hunt is over.** §8 shows the ceiling is structural: 45.8 GiB of
weights where the config intends ~13 GiB. There is ~32 GiB of overhead that no
callback fix can recover, so the remaining levers are all about *peak*, not
*retention*.

In order:

1. ~~**Test the `use_cache` hypothesis**~~ — done, **refuted** (§5, §8). The
   code fix stays because it is independently correct; it is not the answer.
2. ~~**Take the guardrail off the training GPU's peak.**~~ **Built and verified
   2026-08-28.** `scripts/run_phase2_dpo.sh --chunk-steps N` trains in chunks.
   Each chunk is its own process, so the GPU empties between training and
   scoring and only one model copy is ever resident. Scoring runs as a separate
   `scripts/heldout_composite_audit.py` process; `scripts/dpo_guardrail_decide.py`
   reads `trainer_state.json` and the audit JSONs and reuses the existing
   `is_reward_hacking` as the stop rule.

   A concurrent subprocess was considered and rejected: it would need its own
   ~46 GiB copy while training still holds ~46 GiB, and this machine has one
   GPU.

   Verified end to end on the 6-step / 3-step-chunk smoke
   (`configs/training/dpo_cat_a_smoke_chunked.yaml`): chunk 1 trained fresh to
   step 3, chunk 2 resumed from `checkpoint-3` and reached step 6, both
   checkpoints scored, GPU measured at 1 MiB between the two phases. **Step 4
   completed** — the step that failed in smoke7-9, 13, 14 and 16.

   Two limits to know. The live run exercised only the *continue* path; the
   *stop* path is covered by unit tests, not by a real run. And the audit
   script samples the whole validation split with a fixed seed rather than the
   reserved guardrail slice the in-process callback used — harmless while
   `mine_model_negatives.py` runs on `--split train`, but it must be revisited
   if mining ever moves to validation.
3. **Decide `load_in_4bit` deliberately** (§8). It currently buys ~0.8 GiB
   while imposing a 4-bit base under a bf16-trained adapter — the mismatch
   R17 already flags for the audit. Turning it off costs ~2 GiB and removes
   the mismatch. Someone should choose; today it is an inherited default.
4. **Only if 2 is not enough:** quantize the fused expert tensors. No
   bitsandbytes NF4 build of this model exists (§8), so this means custom
   wrapping or a different quant backend — real work, uncertain payoff.
5. **Then start the real 500-step run.** Note §4's caveat still stands
   untouched: the real config is cap **8192** with **50** guardrail prompts
   against the smokes' 5120 and 2. Walk both up one at a time.

Housekeeping, independent of the above:

- ~~`scripts/build_dpo_smoke_fixtures.py` is new and **untracked**~~ — committed
  in `8cf2b93`. The smoke *config* had the same problem and is now tracked as
  `configs/training/dpo_cat_a_smoke.yaml`.
- **`.venv-train` can be built without `trl`.** A rebuild that does not run
  `scripts/install_train.sh` to completion leaves a venv that shadows the base
  image's transformers with 5.x but ships no `trl` of its own, so `import trl`
  falls through to the image's 0.23.1 and DPO dies at trainer init 35 minutes
  in. `training/dpo.py::_assert_dpo_row_processing_support` now catches this
  before the 26B load. See §9.
- Have the runners unset `UNSLOTH_VLLM_STANDBY` (or fail loudly) rather than
  exporting an allocator setting that a later import silently discards (§3).
- `checkpoints/dpo_cat_a_smoke*/` is scratch. smoke10, 14, 15 and 16 carry the
  load-bearing results; the rest can go.

## 7. Reproducing

```bash
# 0a. Check the env FIRST. A .venv-train built without install_train.sh has no
#     trl of its own and silently uses the base image's 0.23.1, which dies at
#     trainer init after the 26B load (§9). Must print 1.0.0 from .venv-train:
.venv-train/bin/python -c "import trl; print(trl.__version__, trl.__file__)"
#     If not:  VIRTUAL_ENV=.venv-train uv pip install --no-deps "trl==1.0.0"
#     and then: rm -rf unsloth_compiled_cache/   # it inlines TRL's source

# 0b. Pre-fetch the base weights if the HF cache is cold. Note the repo:
#     Unsloth's mapper redirects google/gemma-4-26B-A4B-it to its OWN mirror
#     (unsloth/models/mapper.py:37), and the HF cache is per-repo, so fetching
#     the google copy downloads 51.6 GB that the run will not use. The weights
#     are byte-identical; only config.json and tokenizer_config.json differ.
huggingface-cli download unsloth/gemma-4-26b-a4b-it

# 0c. Rebuild the smoke fixtures (the originals did not survive the rebuild).
.venv-train/bin/python scripts/build_dpo_smoke_fixtures.py
# -> 64 train / 16 validation / 8 model-negative rows, all under cap 5120

# 1. Bisect arm A — guardrail on, trainer eval off. Expect OOM at step 4.
#    configs/training/dpo_cat_a_smoke.yaml IS this arm, tracked in git.
setsid env -u UNSLOTH_VLLM_STANDBY \
  bash scripts/run_phase2_dpo.sh \
    --dpo-config configs/training/dpo_cat_a_smoke.yaml --skip-pairs --no-wandb

# 2. Bisect arm B — guardrail off, trainer eval on. Expect 6/6 steps.
setsid env -u UNSLOTH_VLLM_STANDBY \
  bash scripts/run_phase2_dpo.sh --dpo-config <smoke15.yaml> --skip-pairs

# 3. Read the outcome of either.
L=checkpoints/dpo_cat_a_smoke15/gemma-4-26B-A4B-it/train.log
grep -E "dpo_heldout_eval" $L                  # guardrail firings
grep -o "'epoch': '[0-9.]*'" $L                # steps completed
grep -oE "train_runtime[^,]*|OutOfMemoryError.*" $L | tail -2
```

`env -u UNSLOTH_VLLM_STANDBY` is load-bearing (§3) — without it the run uses a
different allocator regime and the results are not comparable.

The two smoke configs differ from `configs/training/dpo_cat_a.yaml` only in
`output_dir`, `training_steps: 6`, `max_seq_length: 5120`, the fixture paths,
`eval_steps`, `eval_held_out_every`, `eval_held_out_num_prompts: 2` and
`reward_hacking_detector`. Frozen copies are written beside each run's
`train.log`.

---

## 8. The real ceiling: `load_in_4bit` does not reach the MoE experts

**Measured 2026-08-28**, by loading the C2 checkpoint exactly as
`dpo.py::train_dpo` does (`FastLanguageModel.from_pretrained(model_name=<C2>,
max_seq_length=5120, dtype=None, load_in_4bit=True)`, preceded by
`unwrap_unsloth_gemma4_kv_zero_proxy()`) and inspecting the result.

| parameter group | size | dtype |
|---|---|---|
| `language_model.layers.N.experts.gate_up_proj` | **28.36 GiB** | bf16 |
| `language_model.layers.N.experts.down_proj` | **14.18 GiB** | bf16 |
| `embed_tokens` | 1.38 GiB | bf16 |
| `vision_tower.*` (all) | 1.03 GiB | bf16 |
| router, misc | 0.04 GiB | bf16/fp32 |
| **everything bitsandbytes actually quantized** | **0.77 GiB** | uint8 |
| | **45.83 GiB** | |

`torch.cuda.memory_allocated()` right after the load: **45.89 GiB**.
Module census: **411 `Linear4bit`**, against **631 `Linear`** and **189
`Gemma4ClippableLinear`**. `model.config.quantization_config` is present and
says `load_in_4bit: True, quant_method: bitsandbytes, nf4` — the request is
honoured, it just does almost nothing.

**Why.** bitsandbytes quantizes by *swapping `nn.Linear` modules* for
`Linear4bit`. Gemma-4-26B-A4B packs its 128 experts per layer into fused 3-D
parameters — `experts.gate_up_proj` has shape `(128, 1408, 2816)` — which are
`nn.Parameter` tensors, not `nn.Linear` modules. There is nothing to swap, so
the swap skips them, silently. Only the attention projections are reachable,
and they are 0.77 GiB of a 26B model.

**Consequences, in order of how much they change the story:**

1. **The ~1 GiB headroom was never a leak.** The run carries ~32 GiB more
   weight than `load_in_4bit: true` implies. Steady state at ~78 of 79.18 GiB
   is what 45.8 GiB of weights plus DPO's logits tensors plus activations
   *should* look like on an 80 GB card. Nothing was retaining memory; there
   was never room.
2. **It explains the ladder's most confusing result** — that smoke8 (5120) and
   smoke9 (4096) "fail identically" (§2). Halving the cap moves the
   `[2, S, 262144]` logits tensor by roughly 4 GiB, against a 32 GiB
   overshoot. The cap was never going to be the lever.
3. **The guardrail is the trigger, not the cause.** §1's bisect is still
   correct and still reproducible: smoke14/16 fail, smoke15 passes. What §1
   could not see is that a modest `generate()` allocation is only fatal
   because of the 32 GiB that should not be there.
4. **smoke16's footprint grew to fill a bigger card** — 91.47 GiB allocated on
   93.09 GiB, against ~78 GiB on 79.18 GiB — while still failing at the *same
   step*. Not explained here. It is consistent with allocator behaviour rather
   than with a per-step leak, but it has not been measured.

**No drop-in fix.** There is no bitsandbytes NF4 build of this model on the
Hub. What exists is GGUF (llama.cpp), AWQ/GPTQ (inference-only), MLX (Apple)
and FP8 — none of them trainable through Unsloth + PEFT. This is also why
Unsloth's mapper sends `google/gemma-4-26B-A4B-it` to the **bf16** mirror
rather than to a `-bnb-4bit` repo: there isn't one.

**Therefore `load_in_4bit=True` at `dpo.py`'s load site is close to a no-op**
that still costs something: it puts a 4-bit base under an adapter trained in
bf16 under TRL (`checkpoints/sft_cat_a_c2/.../train.log`:
`framework=trl … precision=bf16`), which is the same handicap R17 records for
the held-out audit. Turning it off would cost ~2 GiB and remove the mismatch.
That is a decision, not a cleanup — it is listed in §6 rather than applied.

---

## 9. The environment defect that cost the first attempt

Worth recording because it is invisible until it costs 35 minutes, and because
it will recur on the next container rebuild.

`.venv-train` can be rebuilt *without* `scripts/install_train.sh` running to
completion. The result has torch, transformers 5.12.1, peft and datasets — but
**no `trl` and no `unsloth`**. Because `pyvenv.cfg` sets
`include-system-site-packages = true` and a `_opt_venv.pth` appends the base
image's site-packages, imports then resolve inconsistently:

- `transformers` → **5.12.1** from `.venv-train`
- `trl` → **0.23.1** from `/opt/venv`
- `unsloth` → 2026.5.9 from `/opt/venv`

TRL 0.23.1's `DPOTrainer` picks its row-processing path from
`model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` — a
property of the *model*, not of the `processing_class` it was handed — and then
dereferences `processing_class.tokenizer` unconditionally
(`trl/trainer/dpo_trainer.py:739`). Gemma-4 is a SigLIP+Gemma4 stack (R9), so
it takes that vision path while Unsloth supplies a plain tokenizer. On
transformers 5.x that is a `GemmaTokenizer`/`TokenizersBackend`, which has
`_tokenizer` and not `tokenizer`:

```
AttributeError: TokenizersBackend has no attribute tokenizer. Did you mean: '_tokenizer'?
```

TRL 1.0.0 — which `install_train.sh:62` pins for exactly this venv — branches
on `isinstance(processing_class, ProcessorMixin)` instead and takes the text
path, which is what Cat A preference pairs need: they carry no images.

**Do not "fix" this by downgrading to `/opt/venv` alone.** transformers 4.57.6
cannot load Gemma-4 at all (`AutoProcessor` → `ValueError: Unrecognized
processing class`; `AutoTokenizer` → `AttributeError: 'list' object has no
attribute 'keys'`). transformers 5.x is required and correct; the missing piece
is `trl`.

**Guarded** by `training/dpo.py::_assert_dpo_row_processing_support`, called
from `_resolve_trl_classes` *before* the 26B load, following
`trajectory_rollout.assert_trajectory_rollout_support()`: inspect the
installed, possibly Unsloth-patched source rather than trust a version string.
It scans the trainer's whole MRO and treats unreadable source as "cannot tell"
rather than blocking a run on a failed introspection. Cover:
`tests/unit/test_dpo_method_availability.py`.

Note `unsloth_compiled_cache/` inlines TRL's source into
`UnslothDPOTrainer.py`, so it must be deleted whenever TRL changes underneath
it — otherwise the old trainer keeps running from cache.
