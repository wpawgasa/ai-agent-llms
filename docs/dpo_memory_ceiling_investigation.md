# Cat A DPO — the memory ceiling, and which eval path causes it

**Status as of 2026-08-18.** Preference learning (R18) is implemented and the
training loop works, but no Cat A DPO run has yet completed with the R5
held-out guardrail enabled. Fifteen smoke runs isolated the cause. This
document records what is settled, what is not, and what to do next.

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

---

## 6. Next state

In order:

1. **Test the `use_cache` hypothesis** (§5). Smallest possible change to the
   callback's `finally` block, then re-run the smoke14 config — guardrail on,
   trainer eval off, cap 5120. Pass = 6/6 steps. ~8 min.
2. **If it passes, re-run smoke13's config** (both evals on, cap 5120) to
   confirm the two paths compose.
3. **Walk the cap back up** toward the real config's 8192, and raise
   `eval_held_out_num_prompts` from 2 toward 50, one at a time. §4 flags both as
   untested and materially heavier than anything the ladder covered.
4. **If the hypothesis fails**, instrument the callback with
   `torch.cuda.memory_allocated()` / `memory_reserved()` before, inside and
   after, and diff across the eval boundary. Given ~1 GiB of headroom, also
   consider attacking peak memory rather than the leak — the guardrail could run
   under a separate short-lived process, or on CPU, or be moved off the training
   GPU entirely.
5. **Only then start the real 500-step run.**

Housekeeping, independent of the above:

- `scripts/build_dpo_smoke_fixtures.py` is new and **untracked**. The original
  fixtures lived in a per-session scratchpad directory that the container
  rebuild deleted, so every stored smoke config points at files that no longer
  exist and no smoke result was reproducible. Commit it, and consider a DVC
  stage for `data/output/preference/task_a/smoke/`.
- Have the runners unset `UNSLOTH_VLLM_STANDBY` (or fail loudly) rather than
  exporting an allocator setting that a later import silently discards (§3).
- `checkpoints/dpo_cat_a_smoke*/` is 15 directories of scratch. smoke10, 14 and
  15 carry the load-bearing results; the rest can go.

---

## 7. Reproducing

```bash
# 0. Rebuild the smoke fixtures (the originals did not survive the rebuild).
.venv-train/bin/python scripts/build_dpo_smoke_fixtures.py
# -> 64 train / 16 validation / 8 model-negative rows, all under cap 5120

# 1. Bisect arm A — guardrail on, trainer eval off. Expect OOM at step 4.
setsid env -u UNSLOTH_VLLM_STANDBY \
  bash scripts/run_phase2_dpo.sh --dpo-config <smoke14.yaml> --skip-pairs

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
