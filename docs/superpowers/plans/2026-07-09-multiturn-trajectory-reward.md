# Multi-Turn Trajectory Reward for GRPO — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give GRPO a whole-conversation trajectory reward (replay-scripted rollout, free-run + truncate-on-divergence, coverage+outcome blend) so within-group reward variance stops collapsing.

**Architecture:** A TRL 1.0.0 `rollout_func` runs an in-process replay: the model free-runs its assistant turns while gold user/tool turns are injected (masked out of the loss via `env_mask`); a trajectory-level reward returns one scalar per rollout. Pure logic (alignment, reward, gate) is unit-tested on CPU; the GPU rollout is validated by a cheap micro-probe before any training step.

**Tech Stack:** Python, PyTorch, TRL 1.0.0 (GRPOTrainer `rollout_func`/`env_mask`), Unsloth, HF `model.generate`, pytest, `uv`.

**Full design reference:** `docs/superpowers/specs/2026-07-09-multiturn-trajectory-reward-design.md` — every task below cites the spec section carrying the algorithm detail and verified TRL line anchors. Read the cited section before implementing the task.

## Global Constraints

- **TRL must be exactly `1.0.0`** — the `rollout_func`/`env_mask` pathway does not exist in 0.24.0. Pin verbatim: `trl==1.0.0`.
- **Never touch the stabilizers** inherited from `configs/training/grpo_cat_a_diagnostic.yaml`: `scale_rewards: "none"`, `loss_type: "dr_grpo"`, `max_grad_norm: 0.2`, `learning_rate: 1.0e-6`, `beta: 0.05`.
- **Reward weights are fixed** at `0.40 / 0.40 / 0.10 / 0.10` (coverage / tool-F1 / terminal / legality).
- **The legacy single-turn path must stay byte-identical in behavior** — `_load_grpo_jsonl`, `reward_business_logic`, `_make_reward_adapter`, `_HeldOutEvalCallback` are not modified; new code is added alongside and gated on `grpo.rollout.mode == "trajectory"`.
- **Use `uv` / activated `.venv`**; prefix python with `source .venv/bin/activate &&`. GPU steps run on the H100 box via `.venv-train`.
- **Branch off fresh `main`** (verify `git branch --show-current`); ship via auto-merged squash PR; commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context)` line. Markdown docs (spec + this plan) ship in the same PR.
- **`[IMPLEMENTER DECIDES]` items** (only three, all in the spec): Gemma-4 turn-end/eos reconciliation (Task 4); LR 1e-6→2e-6 only after 50-step evidence (Task 7); truncation-relaxation knobs if the probe returns `NO_GO_TRUNCATION` (Task 5).

---

## File Structure

- **Create** `src/llm_workflow_agents/training/trajectory_rollout.py` — all pure trajectory logic + the in-process rollout: `assert_trajectory_rollout_support`, `GoldScript`, `build_gold_script`, `prompt_key`, `classify_turn`, `TrajectoryRolloutConfig`, `RolloutSample`, `run_replay_rollout`, `make_replay_rollout_func`.
- **Modify** `src/llm_workflow_agents/training/rewards/reward_business_logic.py` — append `reward_business_logic_trajectory`.
- **Modify** `src/llm_workflow_agents/training/grpo.py` — `_load_grpo_trajectory_jsonl`, `_make_trajectory_reward_adapter`, `_TrajectoryHeldOutEvalCallback`, registry entry, `train_grpo` trajectory branch.
- **Create** `configs/training/grpo_cat_a_trajectory.yaml`.
- **Create** `scripts/trajectory_variance_probe.py`.
- **Modify** `pyproject.toml`, `requirements-train.txt` (pin).
- **Create** tests: `tests/unit/test_trajectory_rollout.py`, `tests/unit/test_reward_trajectory.py`, `tests/unit/test_trajectory_probe.py`.

---

### Task 1: Version pin + runtime env gate

Spec §2.3, §3e. Smallest shippable unit: pins corrected and the guard importable.

**Files:**
- Modify: `pyproject.toml` (the `trl==0.24.0` line), `requirements-train.txt` (the `trl==` line)
- Create: `src/llm_workflow_agents/training/trajectory_rollout.py`
- Test: `tests/unit/test_trajectory_rollout.py`

**Interfaces:**
- Produces: `assert_trajectory_rollout_support() -> None` (raises `RuntimeError` if the installed TRL `GRPOTrainer._generate` source lacks both `"rollout_func"` and `"env_mask"`).

- [ ] **Step 1: Fix the pins.** In `pyproject.toml` and `requirements-train.txt`, change `trl==0.24.0` to `trl==1.0.0`. Verify: `grep -n "trl==" pyproject.toml requirements-train.txt` shows `trl==1.0.0` in both.

- [ ] **Step 2: Write the failing test** (`tests/unit/test_trajectory_rollout.py`):

```python
from llm_workflow_agents.training.trajectory_rollout import assert_trajectory_rollout_support

def test_env_gate_importable_and_callable():
    # Import must succeed even without TRL installed in this venv;
    # the call is only meaningful on the train box, so we only assert it's callable.
    assert callable(assert_trajectory_rollout_support)
```

- [ ] **Step 3: Run it, expect fail.** `source .venv/bin/activate && pytest tests/unit/test_trajectory_rollout.py::test_env_gate_importable_and_callable -q` → FAIL (module missing).

- [ ] **Step 4: Implement** `assert_trajectory_rollout_support` in `trajectory_rollout.py` exactly as spec §2.3 (inspect `GRPOTrainer._generate` source for `"rollout_func"` and `"env_mask"`; import inside the function so the module imports without TRL).

- [ ] **Step 5: Run it, expect pass.** Same command → PASS.

- [ ] **Step 6: Commit.** `git add pyproject.toml requirements-train.txt src/llm_workflow_agents/training/trajectory_rollout.py tests/unit/test_trajectory_rollout.py && git commit` — `feat(grpo): pin trl==1.0.0 + trajectory rollout env gate`.

---

### Task 2: Pure core — GoldScript, build_gold_script, prompt_key, classify_turn

Spec §3a, §3b step 4 (the `classify_turn` predicate). This is the load-bearing alignment logic; TDD it fully.

**Files:**
- Modify: `src/llm_workflow_agents/training/trajectory_rollout.py`
- Test: `tests/unit/test_trajectory_rollout.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) GoldScript` (fields per spec §3a).
  - `build_gold_script(raw_row: dict, enriched_system: str) -> GoldScript`
  - `prompt_key(prompt_messages: list[dict]) -> str` — `sha256(json.dumps(..., sort_keys=True, ensure_ascii=False))`
  - `classify_turn(pred_transitions: list[tuple[str,str]], cursor: int, gold_transitions: list[tuple[str,str]]) -> tuple[str, int]` returning `("advance"|"stall"|"diverged", new_cursor)`.

- [ ] **Step 1: Write failing tests for `classify_turn`** (the exact predicate from spec §3b.4):

```python
from llm_workflow_agents.training.trajectory_rollout import classify_turn

GOLD = [("A", "B"), ("B", "C"), ("C", "D")]

def test_advance_single_step():
    assert classify_turn([("A", "B")], 0, GOLD) == ("advance", 1)

def test_advance_consecutive_compression():
    assert classify_turn([("A", "B"), ("B", "C")], 0, GOLD) == ("advance", 2)

def test_stall_on_no_effective_transition():
    assert classify_turn([("B", "B")], 1, GOLD) == ("stall", 1)   # self-loop is neutral

def test_stall_on_empty():
    assert classify_turn([], 1, GOLD) == ("stall", 1)

def test_diverged_wrong_target():
    assert classify_turn([("B", "Z")], 1, GOLD) == ("diverged", 1)

def test_diverged_non_consecutive():
    assert classify_turn([("A", "B"), ("C", "D")], 0, GOLD) == ("diverged", 0)

def test_cursor_at_end_any_transition_diverges():
    assert classify_turn([("D", "E")], 3, GOLD) == ("diverged", 3)
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/unit/test_trajectory_rollout.py -q -k classify` → FAIL.

- [ ] **Step 3: Implement `classify_turn`** per spec §3b.4: drop self-loops (`t[0]==t[1]`) to get `effective`; empty → `("stall", cursor)`; if `effective == gold[cursor:cursor+len(effective)]` and non-empty → `("advance", cursor+len(effective))`; else `("diverged", cursor)`.

- [ ] **Step 4: Run, expect pass.** Same command → PASS.

- [ ] **Step 5: Write failing tests for `GoldScript` / `build_gold_script` / `prompt_key`** using a synthetic conversation row (system, user, assistant with `[STATE: A → B]`, tool, user, assistant …) mirroring the Task A schema (spec §3a). Assert: `len(gold_transitions) == n_gold_assistant_turns`; `segments[t]` holds the non-assistant messages between assistant turns; `prompt_messages` ends before the first assistant; `prompt_key` is stable across dict key order.

- [ ] **Step 6: Run, expect fail; implement `build_gold_script`/`prompt_key`/`GoldScript` per spec §3a; run, expect pass.**

- [ ] **Step 7: Commit.** `feat(grpo): trajectory gold-script + turn-alignment core`.

---

### Task 3: Trajectory reward + adapter

Spec §3c. TDD the reward math (known-answer, order-sensitivity, renorm, stall_factor, coverage cap).

**Files:**
- Modify: `src/llm_workflow_agents/training/rewards/reward_business_logic.py` (append `reward_business_logic_trajectory`)
- Modify: `src/llm_workflow_agents/training/grpo.py` (`_make_trajectory_reward_adapter`; add `"reward_business_logic_trajectory"` to `_REWARD_REGISTRY`)
- Test: `tests/unit/test_reward_trajectory.py`

**Interfaces:**
- Consumes: `classify_turn` (Task 2); `graded_tool_call_f1`, `transition_legality_score`, `extract_state_annotations`, `extract_tool_calls` (existing `reward_utils.py`).
- Produces: `reward_business_logic_trajectory(prompts, trajectories, metas, ground_truths) -> list[float]` (signature per spec §3c); `_make_trajectory_reward_adapter(reward_fn)` reading the `trajectory`/`rollout_meta` extra-field kwargs.

- [ ] **Step 1: Write failing known-answer tests.** Hand-compute expected scores. E.g. gold path of 4 transitions, model traverses 2 then diverges, all components active, no tool calls → coverage 0.5, tool-F1 by the graded scorer, terminal 0 (not gold_complete), legality·stall_factor; assert the renormalized blend matches the hand value. Add: order-sensitivity (a permuted-gold trajectory scores strictly lower); coverage never exceeds 1.0; `stall_factor` reduces legality when stalls present; renormalization when `terminal_reached` is False drops the terminal weight.

```python
from llm_workflow_agents.training.rewards.reward_business_logic import reward_business_logic_trajectory

def test_partial_coverage_no_tools_known_answer():
    gt = {"state_sequence": [{"from":"A","to":"B"},{"from":"B","to":"C"},
                             {"from":"C","to":"D"},{"from":"D","to":"E"}],
          "tool_calls": [], "terminal_state": "E", "terminal_reached": True,
          "valid_transitions": [["A","B"],["B","C"],["C","D"],["D","E"]]}
    traj = ["[STATE: A → B]", "[STATE: B → C]", "[STATE: C → Z]"]  # diverges on turn 3
    meta = {"cursor": 2, "stop_reason": "diverged", "n_model_turns": 3,
            "n_stall_turns": 0, "gold_len": 4}
    (score,) = reward_business_logic_trajectory([None], [traj], [meta], [gt])
    # coverage 2/4=0.5, tool F1 has no gold+no pred → component inactive/renorm per spec,
    # terminal 0, legality 1.0*stall_factor(1.0). Assert against the hand-computed value.
    assert 0.0 <= score <= 1.0
    assert abs(score - EXPECTED) < 1e-6   # EXPECTED computed by hand per spec §3c.3
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/unit/test_reward_trajectory.py -q` → FAIL.

- [ ] **Step 3: Implement `reward_business_logic_trajectory`** per spec §3c (recompute alignment via `classify_turn`; four components; active-component renormalization following `reward_business_logic.py:181-184`; clip [0,1]).

- [ ] **Step 4: Run, expect pass.** Same command → PASS.

- [ ] **Step 5: Implement `_make_trajectory_reward_adapter` + registry entry** per spec §3c (read `kwargs["trajectory"]`/`kwargs["rollout_meta"]`, JSON-decode; **do not** collapse `completions`; hard error on missing `trajectory`; keep the `_LATEST_INSTRUMENTATION` stash). Add a small adapter test with a mocked TRL-style kwargs call.

- [ ] **Step 6: Run full reward test file, commit.** `feat(grpo): trajectory coverage+outcome reward + adapter`.

---

### Task 4: In-process replay rollout + mask assembly

Spec §3b (full turn loop), §2.2 (mask semantics). No real GPU in tests — monkeypatch `model.generate` with canned token scripts.

**Files:**
- Modify: `src/llm_workflow_agents/training/trajectory_rollout.py` (`TrajectoryRolloutConfig`, `RolloutSample`, `run_replay_rollout`, `make_replay_rollout_func`)
- Test: `tests/unit/test_trajectory_rollout.py`

**Interfaces:**
- Consumes: `GoldScript`, `classify_turn`, `assert_trajectory_rollout_support` (Tasks 1–2).
- Produces: `run_replay_rollout(model, tokenizer, scripts, cfg) -> list[RolloutSample]`; `make_replay_rollout_func(script_index, cfg) -> Callable[[list, trainer], dict]` returning the exact keys of spec §3b.4 (`prompt_ids`/`completion_ids`/`logprobs=None`/`env_mask`/`trajectory`/`rollout_meta`).

- [ ] **Step 1: `[IMPLEMENTER DECIDES]` — inspect the Gemma-4 SFT tokenizer** (`checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000`) to reconcile the assistant turn-end id with `tokenizer.eos_token_id` (spec §3b.1). Record the choice in a module comment; the unit test below locks the invariant regardless.

- [ ] **Step 2: Write failing mask-invariant test** with a monkeypatched `generate` + a small real ChatML tokenizer fixture (or minimal jinja stub) that exercises the `_get_tool_suffix_ids`-style prefix diff:

```python
def test_env_mask_zeros_over_injected_segments(tiny_tokenizer, scripted_generate):
    # scripted_generate emits: turn1 → "[STATE: A → B]<end_of_turn>", then a gold user segment
    # is injected, then turn2 → diverges → stop.
    script = make_two_turn_script()          # helper building a GoldScript
    cfg = TrajectoryRolloutConfig(max_completion_tokens=512)
    [sample] = run_replay_rollout(model=scripted_generate, tokenizer=tiny_tokenizer,
                                  scripts=[script], cfg=cfg)
    assert len(sample.env_mask) == len(sample.completion_ids)
    assert sample.env_mask[0] == 1                       # first token is model
    assert sum(sample.env_mask) > 0
    # decoding mask==1 positions reproduces the model turn texts:
    model_ids = [i for i, m in zip(sample.completion_ids, sample.env_mask) if m == 1]
    assert "STATE: A" in tiny_tokenizer.decode(model_ids)
    # injected gold user text lives only under mask==0:
    ext_ids = [i for i, m in zip(sample.completion_ids, sample.env_mask) if m == 0]
    assert "GOLD_USER" in tiny_tokenizer.decode(ext_ids)
    assert sample.completion_ids[-1] == tiny_tokenizer.eos_token_id   # final-token guarantee
```

- [ ] **Step 3: Run, expect fail.** `pytest tests/unit/test_trajectory_rollout.py -q -k mask` → FAIL.

- [ ] **Step 4: Implement `run_replay_rollout` + `make_replay_rollout_func`** exactly per spec §3b (batched active-set turn loop; per-turn generate under `no_grad`/`eval`; append model ids mask=1; `classify_turn` + stop conditions §3b.5; gold-segment injection via dummy-diff §3b.6 mask=0; final-token guarantee §3b.7; per-sample invariants §3b.8). `make_replay_rollout_func`: order-preserving `prompt_key` lookup (hard error on miss), `assert_trajectory_rollout_support()` on first call.

- [ ] **Step 5: Run, expect pass.** Same command → PASS. Add a truncation-path test (budget stop → forced eos, mask 0) and a `gold_complete` test.

- [ ] **Step 6: Commit.** `feat(grpo): in-process replay rollout with env_mask assembly`.

---

### Task 5: Cheap-first micro-probe (build; run on H100)

Spec §4. TDD the pure `summarize_trajectory_probe`/`classify_gate`; the GPU driver is exercised manually.

**Files:**
- Create: `scripts/trajectory_variance_probe.py`
- Test: `tests/unit/test_trajectory_probe.py`

**Interfaces:**
- Consumes: `run_replay_rollout`, `reward_business_logic_trajectory` (Tasks 3–4); checkpoint-loading idiom from `scripts/preflight_entropy_diag.py`.
- Produces: `summarize_trajectory_probe(group_rewards, group_coverages, group_metas) -> dict`; `classify_gate(summary) -> tuple[str, str]` (verdict ∈ `GO_TRAJECTORY`/`NO_GO_VARIANCE`/`NO_GO_TRUNCATION`/`MARGINAL`).

- [ ] **Step 1: Write failing gate tests** (thresholds from spec §4):

```python
from trajectory_variance_probe import classify_gate   # scripts/ added to sys.path in test

def _s(std, collapsed, turns, cov):
    return {"median_reward_std": std, "frac_collapsed_groups": collapsed,
            "mean_model_turns": turns, "mean_coverage": cov}

def test_go_trajectory():
    assert classify_gate(_s(0.08, 0.30, 5.0, 0.5))[0] == "GO_TRAJECTORY"
def test_no_go_variance():
    assert classify_gate(_s(0.01, 0.90, 6.0, 0.9))[0] == "NO_GO_VARIANCE"
def test_no_go_truncation():
    assert classify_gate(_s(0.06, 0.30, 1.5, 0.2))[0] == "NO_GO_TRUNCATION"
def test_marginal():
    assert classify_gate(_s(0.03, 0.60, 4.0, 0.5))[0] == "MARGINAL"
```

- [ ] **Step 2: Run, expect fail; implement `summarize_trajectory_probe` + `classify_gate` + the `main()` GPU driver** (mirror `rft_headroom_probe.py`: load checkpoint, sample N conversations, duplicate ×N-completions, `run_replay_rollout`, score, summarize, write JSON, print verdict + histograms + mask audit). Run, expect pass on the pure tests: `pytest tests/unit/test_trajectory_probe.py -q`.

- [ ] **Step 3: Commit.** `feat(grpo): trajectory variance micro-probe + gate`.

- [ ] **Step 4 (GPU, manual — the go/no-go gate):**

```bash
.venv-train/bin/python scripts/trajectory_variance_probe.py \
  --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000 \
  --data-dir data/output/grpo/task_a --split validation \
  --n-conversations 50 --n-completions 8 --temperature 0.8 --top-p 0.95 \
  --output runs/preflight/trajectory_variance_probe.json
```
Expected on GO: `median_reward_std ≥ 0.05`, `frac_collapsed_groups < 0.50`, `mean_model_turns ≥ 3.0`, mask audit clean. **STOP the plan here if the verdict is `NO_GO_*`** — see spec §7 (falsification → RFT/SFT-only). `[IMPLEMENTER DECIDES]`: on `NO_GO_TRUNCATION`, bump `stall_turn_limit` 2→3 (then a one-legal-edge tolerance) and re-probe before abandoning.

---

### Task 6: Trainer wiring (gated on `rollout.mode == "trajectory"`)

Spec §3a (loader), §3d (callback), §3e (config + `train_grpo` branch). Legacy path untouched.

**Files:**
- Modify: `src/llm_workflow_agents/training/grpo.py` (`_load_grpo_trajectory_jsonl`; `_TrajectoryHeldOutEvalCallback`; `train_grpo` branch; rollout-stats instrumentation)
- Create: `configs/training/grpo_cat_a_trajectory.yaml`
- Test: `tests/unit/test_trajectory_rollout.py` (loader on a synthetic 2-conversation JSONL)

**Interfaces:**
- Consumes: `make_replay_rollout_func`, `GoldScript`, `_make_trajectory_reward_adapter`, `reward_business_logic_trajectory` (Tasks 2–4).
- Produces: `_load_grpo_trajectory_jsonl(data_dir, split="train") -> tuple[Dataset, dict[str, GoldScript]]`; trajectory branch in `train_grpo`.

- [ ] **Step 1: Write a failing loader test** against a tiny fixture JSONL (2 conversations): assert one dataset row per conversation, the `script_index` has both `prompt_key`s, the per-row invariant holds, and a deliberately-broken row (mismatched `state_sequence` length) is dropped with the counter incremented.

- [ ] **Step 2: Run, expect fail; implement `_load_grpo_trajectory_jsonl`** per spec §3a (re-enrich system prompt, first-assistant split, segmentation, GT JSON-string column, `script_index`, collision + overlong + invariant guards). Run, expect pass.

- [ ] **Step 3: Implement `_TrajectoryHeldOutEvalCallback`** per spec §3d (greedy `run_replay_rollout` on validation scripts; strict independent composite `0.4·state_sequence_match + 0.4·tool_call_f1 + 0.2·terminal`; log `eval/held_out_composite`; keep `_is_reward_hacking` auto-stop). Reuse the existing `_HeldOutEvalCallback` skeleton without modifying it.

- [ ] **Step 4: Add the `train_grpo` trajectory branch** per spec §3e: when `grpo.rollout.mode == "trajectory"` → `assert_trajectory_rollout_support()`; `_load_grpo_trajectory_jsonl`; `rollout_func=make_replay_rollout_func(script_index, cfg)` passed to `GRPOTrainer(...)`; `_make_trajectory_reward_adapter`; trajectory callback; force `use_vllm=False`; assert `mask_truncated_completions` is False; stash rollout stats into `_LATEST_INSTRUMENTATION`. Any other `mode` → existing single-turn path, unchanged.

- [ ] **Step 5: Create `configs/training/grpo_cat_a_trajectory.yaml`** — copy of `grpo_cat_a_diagnostic.yaml` with the additions listed in spec §3e (keep every stabilizer).

- [ ] **Step 6: Run the FULL unit suite** to prove the legacy path is untouched: `source .venv/bin/activate && pytest tests/unit -q`. Expected: all green, including `test_grpo_outbound.py` and `test_reward_functions.py`.

- [ ] **Step 7: Commit.** `feat(grpo): trajectory loader, held-out callback, train_grpo wiring + config`.

---

### Task 7: 50-step diagnostic (GPU, manual — the training gate)

Spec §1.4, §6.7. Not automatable here; the gate that authorizes a 1000-step run.

**Files:** none (uses Task 6 artifacts).

- [ ] **Step 1: Launch the diagnostic** (only after Task 5 returned `GO_TRAJECTORY`):

```bash
./scripts/run_phase2_grpo.sh \
  --grpo-config configs/training/grpo_cat_a_trajectory.yaml \
  --sft-checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000
```

- [ ] **Step 2: Watch W&B** (`llm-workflow-agents-grpo-trajectory`). Kill immediately if `grad_norm > 50` for 3 consecutive steps or `KL > 10` (inherited criteria). Confirm the step-0 mask audit logged clean.

- [ ] **Step 3: Read the success gate** (spec §1.4): `frac_reward_zero_std < 0.30`, `train/reward_std` materially above 0.02, held-out composite non-degrading. `[IMPLEMENTER DECIDES]`: if reward_std is healthy but reward mean is immobile, bump LR 1e-6→2e-6 once (dr_grpo denominator shift, R6) and re-run — never with any other change.

- [ ] **Step 4: Promote to 1000 steps** only on a passing gate — flip `training_steps: 50 → 1000` in the config (a one-line change). If the gate fails per spec §7, stop and fall back to RFT/SFT-only.

---

## Self-Review

**Spec coverage:** §2.2 masking → Task 4 (env_mask assembly + invariant test); §2.3 env gate → Task 1; §3a loader → Tasks 2 (GoldScript) + 6 (loader); §3b rollout → Task 4; §3c reward → Task 3; §3d callback → Task 6; §3e config/pin → Tasks 1 + 6; §4 probe → Task 5; §6 order → Tasks 1-7 map 1:1; §7 falsification → Task 5 Step 4 + Task 7 Step 4. No uncovered section.

**Placeholder scan:** the only "decide later" items are the three explicit `[IMPLEMENTER DECIDES]` from the spec (Gemma-4 eos in Task 4; LR bump in Task 7; truncation relaxation in Task 5) — each names its exact knob and trigger. No generic "add error handling"/"write tests" placeholders; pure-logic tasks carry real test + implementation code, GPU tasks carry exact commands + expected gates.

**Type consistency:** `classify_turn` signature identical in Task 2 (definition), Task 3 (reward consumer), Task 4 (rollout consumer). `run_replay_rollout`/`make_replay_rollout_func` return keys identical in Task 4 (producer) and Task 5/6 (consumers). `reward_business_logic_trajectory(prompts, trajectories, metas, ground_truths)` identical in Task 3 and the adapter. `GoldScript` fields consistent across Tasks 2, 4, 6.
