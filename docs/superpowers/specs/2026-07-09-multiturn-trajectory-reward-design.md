# Spec & Implementation Plan: Multi-Turn Trajectory Reward for GRPO — Cat A (Gemma-4-26B-A4B-it)

**Status:** design complete, verified against installed code (2026-07-09)
**Author:** Fable 5 → **Target implementer:** claude-opus-4-8 (multi-file, correctness-critical: a masking bug produces silently-garbage gradients on a multi-hour H100 run)
**Verification note:** the load-bearing token-masking claim (§2.2) was independently re-verified against `.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py` after authoring — `env_mask` pathway confirmed at lines 1646-1649, loss-mask product at 2149/2265, `tools`-falsy branch at 1635/1646.
**Locked decisions (do not revisit):** (1) replay-scripted environment; (2) free-run + truncate-on-divergence; (3) coverage+outcome blend reward at 0.40/0.40/0.10/0.10; (4) this document is the Opus handover.
**Prior art:** `docs/grpo_diagnosis_gemma4_26b.md` (4 killed runs; bqbxnqxw grad-norm 50k-class `scale_rewards` blowup; df4dot2d grad-norm 1126/KL 40), `docs/grpo_viability_investigation.md` (single-turn GRPO verdict: inadmissible; multi-turn = escalation path #5).

---

## 1. Objective & acceptance criteria

### Objective
Replace the per-turn single-completion GRPO rollout with a **whole-conversation replay rollout**: the model free-runs its own assistant turns against the gold conversation's scripted user turns and tool results, truncating when it leaves the gold path, and receives **one scalar trajectory reward** = 0.40·coverage + 0.40·graded tool-F1 + 0.10·terminal + 0.10·legality. Aggregating over a median-13-turn gold path turns the single-turn discrete lattice ({0, 0.3, 0.5, 1.0} per turn) into a near-continuous distribution (coverage alone has T+1 rungs, T = gold length), manufacturing the within-group reward variance GRPO's advantage needs.

### Acceptance criteria (measurable, in order of gating)
1. **Unit tests green** (pure CPU, no GPU): alignment walk, reward math, mask assembly with mocked `generate` — commands in §6.
2. **Micro-probe GO** (§4) on ~50 validation conversations × N=8 at T=0.8, run BEFORE any trainer step:
   - primary: **median within-group `reward_std` ≥ 0.05** and **`frac_collapsed_groups` (std < 0.01) < 0.50** — the exact GRPO admissibility bar from `docs/grpo_viability_investigation.md` §1 that every single-turn measurement failed (~0.77 collapsed at bqbxnqxw geometry);
   - anti-truncation: **mean model turns before stop ≥ 3.0** and mean coverage in (0.05, 0.95);
   - environment: probe asserts `trl.__version__ == "1.0.0"` and that the `env_mask` pathway exists in the *patched* trainer source (see §2.3).
3. **Mask audit passes** at rollout time: for every sample, `len(env_mask) == len(completion_ids)`; decoding the mask==1 positions reproduces exactly the concatenated model-turn texts (modulo special tokens); decoding mask==0 positions contains only injected gold user/tool text + turn scaffolding. Logged once at probe time and at trainer step 0.
4. **50-step diagnostic run** (inheriting all stabilizers from `configs/training/grpo_cat_a_diagnostic.yaml`): completes without kill-criteria breach (grad_norm < 50 sustained, KL < 10), `frac_reward_zero_std < 0.30` (vs 0.70–1.0 historically), `train/reward_std` materially above 0.02, and the trajectory held-out composite non-degrading.
5. Done = criteria 1–4 met and the 1000-step config is a one-line change (`training_steps`).

---

## 2. Architecture

### 2.1 Data flow

```
train.jsonl (1 row = 1 conversation)
   └─ _load_grpo_trajectory_jsonl ──► HF Dataset rows: {prompt, ground_truth}   (TRL-visible)
                                  └─► script_index: {prompt_key → GoldScript}   (closure-held, TRL-invisible)
GRPOTrainer._generate (trl 1.0.0, grpo_trainer.py:1589)
   └─ rollout_func(prompts, trainer)          ← prompts = generation batch, each unique prompt
        │                                        repeated num_generations× (RepeatSampler)
        │  per prompt: look up GoldScript by prompt_key; batched turn loop:
        │    sample assistant turn (mask=1) → parse [STATE:] → align vs gold →
        │    stop OR inject gold tool/user segment tokens + next gen header (mask=0) → repeat
        └─ returns {prompt_ids, completion_ids, logprobs=None,
                    env_mask, trajectory, rollout_meta}
   env_mask popped at :1649 → tool_mask ──► loss mask product (:2265) → only model tokens trained
   trajectory/rollout_meta merged into reward inputs (:1949-1956)
_calculate_rewards (:1961, contract :1196-1198)
   └─ _make_trajectory_reward_adapter → reward_business_logic_trajectory → one scalar/trajectory
advantage math (:1962-2007) — turn-count-agnostic, unchanged
_compute_loss (:2258-2454) — per-token loss × (completion_mask · tool_mask); dr_grpo normalizer :2396
```

### 2.2 The token-masking crux — RESOLVED

**Mechanism chosen: `rollout_func` + the `env_mask` extra field.** Verified line-by-line against `.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py` (TRL 1.0.0):

- `rollout_func` contract (`:114-117`, `:1596-1613`): receives `(prompts, trainer)` with structured (conversational) prompts, must return `prompt_ids`, `completion_ids`, `logprobs`; **all other returned keys become per-completion extra fields**.
- **`env_mask` is a first-class supported channel**: at `:1646-1649`, when `self.tools` is falsy (ours is), `tool_mask = extra_fields.pop("env_mask", None)` — the comment reads *"Support custom env_mask from rollout_func (e.g., for environment feedback masking). Internally treated as tool_mask - marks model tokens (1) vs external tokens (0)."* This is exactly our need.
- Downstream, `tool_mask` is padded with 1 (`:1788-1794`), and:
  - **loss**: `_compute_loss:2265` → `mask = completion_mask * inputs["tool_mask"]`; every loss type multiplies by `mask` (`grpo`/`bnpo`/`dr_grpo`/`dapo` at `:2389-2407`), so injected tokens contribute **zero** policy-gradient and zero KL (KL enters `per_token_loss` before the mask product, `:2384-2385` then `:2389+`);
  - **attention**: `completion_mask` stays all-ones over injected tokens (`:1768`), so the model attends to gold user/tool context during logp computation — correct;
  - **metrics**: entropy/KL logging uses `masked_batch_mean` over the same product mask (`:2412-2425`); `num_items_in_batch` counts only mask==1 tokens (`:1653-1656`) — DAPO-style normalizers stay honest;
  - **liger path** honors it too (`:2148-2149`) — though we run the non-liger path.
- `logprobs: None` is safe: only consumed for vLLM importance-sampling correction (`:1885`, `:2381`) and OPSM (`:2303`), all gated off under `use_vllm=False` / default `off_policy_mask_threshold=None`; `None` skips padding at `:1778-1787`. Old-policy logps for the PPO ratio are **recomputed by the trainer itself** via a no-grad forward over our returned ids (`:1860-1882`, triggered because `gradient_accumulation_steps(1) % generate_every(4) != 0`) — valid for interleaved sequences since it operates on token ids.
- `env_mask` is **popped** from extra fields, so it does not leak into reward kwargs; our other extra fields (`trajectory`, `rollout_meta`) are merged per-completion into reward inputs at `:1949-1956` and forwarded to the reward at `:1156-1157` + `:1196-1198`. `GRPOConfig.remove_unused_columns` defaults to `False` (grpo_config.py:375-377), so the `ground_truth` dataset column also reaches the reward.

**Mechanisms considered and rejected:**
- `tools` + `_tool_call_loop` (`:1410-1587`): executes real Python tools and regenerates keyed on parsed structured `tool_calls`; it cannot inject **user** turns, and structured tool-call parsing depends on `parse_response`/`response_schema` the Gemma-4 tokenizer may not ship. Rejected.
- `environment_factory` (`:1712-1717`, `:446-474`): `reset()` only mutates the last prompt message before generation — no per-turn hook. Rejected.
- `GRPOTrainer` subclass overriding `_generate`: viable **fallback** if the Unsloth patch layer on the training box breaks the rollout_func path (§5 R2), but unnecessary otherwise.

**Two verified hazards with mandated mitigations:**
1. `mask_truncated_completions` (default `False`, grpo_config.py:744): if ever enabled, `:1796-1804` zeroes the *entire* completion whose last token isn't EOS/pad — i.e., precisely our divergence-truncated trajectories. Mitigation: wiring asserts it is False **and** the rollout guarantees every trajectory's final token is `tokenizer.eos_token_id` (force-appended with mask=0 when a turn hits its token budget; trainer's `self.eos_token_id = tokenizer.eos_token_id`, `:327`). This also keeps the `completions/clipped_ratio` metric meaningful.
2. `dr_grpo` normalizes by `max_completion_length` (`:2396-2397`). Raising it 512→4096 shrinks per-token loss ~8×, partially offset by ~6× more unmasked tokens per sequence. Keep `lr: 1.0e-6` for the first run and monitor `grad_norm`; raising to 2e-6 if updates are vanishing is **[IMPLEMENTER DECIDES]** after inspecting the 50-step diagnostic.

### 2.3 Environment/version gate (pin drift is real)

Verified on this box: `.venv` has **TRL 1.0.0** (`trl-1.0.0.dist-info`), while `pyproject.toml:64` and `requirements-train.txt:22` pin `trl==0.24.0`, and grpo.py comments cite 0.23.1. Moreover **`.venv-train` — the venv `scripts/run_phase2_grpo.sh:82` actually sources — does not exist here**, and unsloth is not importable in `.venv`; the training box's env is unauditable from this session. Unsloth also monkey-patches TRL trainers at import. Therefore:

- Fix the pins to `trl==1.0.0` (§3e).
- Add a **runtime feature assertion**, executed after `import unsloth` (so it sees the patched class), in both the micro-probe and the `grpo.py` trajectory path:

```python
def assert_trajectory_rollout_support() -> None:
    import inspect, trl
    from trl.trainer.grpo_trainer import GRPOTrainer
    src = inspect.getsource(GRPOTrainer._generate)
    if "rollout_func" not in src or "env_mask" not in src:
        raise RuntimeError(
            f"TRL {trl.__version__}: GRPOTrainer._generate lacks the rollout_func/env_mask "
            "pathway (required by trajectory GRPO; present in trl==1.0.0). "
            "Check .venv-train's trl pin and whether Unsloth's RL patch replaced _generate."
        )
```

---

## 3. Component specs

All new run-time code lives in `src/llm_workflow_agents/training/trajectory_rollout.py` (new module) plus surgical additions to `grpo.py` and `rewards/reward_business_logic.py`. Everything below the model call is pure and unit-testable.

### 3a. Trajectory loader — one row per conversation

**File:** `src/llm_workflow_agents/training/grpo.py` (beside `_load_grpo_jsonl`, :161-302) with the pure pieces in `trajectory_rollout.py`.

```python
# trajectory_rollout.py
@dataclass(frozen=True)
class GoldScript:
    conversation_id: str
    prompt_messages: list[dict]           # [enriched system (+ first user)] — up to first assistant
    segments: list[list[dict]]            # segments[t] = gold NON-assistant messages between gold
                                          #   assistant turn t and t+1, stripped to {role, content};
                                          #   len == n_gold_assistant_turns (last entry may be [])
    gold_transitions: list[tuple[str, str]]  # ground_truth.state_sequence as tuples, order-preserving
    gold_tool_calls: list[dict]           # flat ground_truth.tool_calls (whole conversation)
    terminal_state: str
    terminal_reached: bool
    valid_transitions: list[list[str]]    # workflow_graph.transitions → [[from, to], ...]

def build_gold_script(raw_row: dict, enriched_system: str) -> GoldScript
def prompt_key(prompt_messages: list[dict]) -> str
    # sha256(json.dumps(prompt_messages, sort_keys=True, ensure_ascii=False))
```

```python
# grpo.py
def _load_grpo_trajectory_jsonl(
    data_dir: Path, split: str = "train"
) -> tuple["Dataset", dict[str, GoldScript]]:
```

**Algorithm** (mirrors `_load_grpo_jsonl`'s enrichment, drops its flattening):
1. Per JSONL line: re-enrich `messages[0]` via `build_enriched_system_prompt(raw, raw_msgs[0]["content"], force_rebuild=True)` (`data/system_prompt.py:60`), exactly as :211-224 does today.
2. `prompt_messages` = messages up to (excluding) the **first** assistant turn, `_slim_content`-stripped. Verified shapes: 495/501 sampled rows have first assistant at index 2 (`[system, user]` prompt), 6/501 at index 1 (opener rows → `[system]` prompt — precedent: the single-turn loader already emits system-terminated prompts, :248-251).
3. Segmentation: for gold assistant turn index `t` (0-based over assistant turns), `segments[t]` = the gold messages strictly between assistant turn `t` and assistant turn `t+1`, roles ∈ {tool, user} (verified adjacency: only `(assistant→tool→user)` and `(assistant→user)` occur). `segments[-1]` = trailing non-assistant messages (usually `[]` — conversations end on assistant).
4. `gold_transitions` from `ground_truth.state_sequence` (`[{from,to}]` → tuples). **Invariant to assert per row:** `len(gold_transitions) == n_gold_assistant_turns` (verified on data: every assistant turn carries exactly one `annotations.state_transition`; L4 sample: 20 == 20). Rows violating it: log + skip.
5. Emit dataset row `{"prompt": prompt_messages, "ground_truth": json.dumps({...})}` where the GT dict carries: `state_sequence` (full, ordered), `tool_calls` (flat), `terminal_state`, `terminal_reached`, `valid_transitions` — same JSON-string-column trick as :287-290 (pyarrow schema bypass).
6. Build `script_index[prompt_key(prompt_messages)] = GoldScript(...)`. **Collision check:** if a key repeats with a different `conversation_id`, drop the later row and log `trajectory_prompt_key_collision` (system prompts embed per-conversation graphs; collisions should be ~zero, but silence here would mis-script rollouts).
7. Overlong guard: rows whose templated prompt exceeds `max_prompt_length` tokens are dropped with a counter (TRL does not truncate prompts on the rollout_func path; a silent overrun would blow the 8192 budget).

Yields ~2,502 train / 290 validation rows (verified counts).

### 3b. In-process replay rollout

**File:** `src/llm_workflow_agents/training/trajectory_rollout.py`

```python
@dataclass(frozen=True)
class TrajectoryRolloutConfig:
    max_turns: int = 24                  # p90 gold length is 21
    per_turn_max_new_tokens: int = 256
    max_completion_tokens: int = 4096    # must equal GRPOConfig.max_completion_length
    stall_turn_limit: int = 2            # consecutive no-transition turns before stop
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

@dataclass
class RolloutSample:
    prompt_ids: list[int]
    completion_ids: list[int]
    env_mask: list[int]                  # len == len(completion_ids); 1=model, 0=injected/forced
    turn_texts: list[str]                # decoded model turns, in order
    meta: dict                           # {"cursor": int, "stop_reason": str, "n_model_turns": int,
                                         #  "n_stall_turns": int, "gold_len": int, "conversation_id": str}

def run_replay_rollout(
    model, tokenizer,
    scripts: list[GoldScript],           # one per requested completion (duplicates for a group)
    cfg: TrajectoryRolloutConfig,
) -> list[RolloutSample]

def make_replay_rollout_func(
    script_index: dict[str, GoldScript],
    cfg: TrajectoryRolloutConfig,
) -> "RolloutFunc"                       # closure: (prompts, trainer) -> dict
```

**`make_replay_rollout_func` returned callable** (the TRL-facing adapter):
1. `assert_trajectory_rollout_support()` on first call.
2. For each prompt in `prompts` (order-preserving — group advantage at `:1967` depends on it): `scripts.append(script_index[prompt_key(prompt)])`; a missing key is a hard error (never silently degrade to single-turn).
3. `samples = run_replay_rollout(trainer.model, trainer.processing_class, scripts, cfg)` — temperature/top_p read from `trainer.args`.
4. Return:
```python
{
  "prompt_ids":     [s.prompt_ids for s in samples],
  "completion_ids": [s.completion_ids for s in samples],
  "logprobs":       None,
  "env_mask":       [s.env_mask for s in samples],
  "trajectory":     [json.dumps(s.turn_texts) for s in samples],   # per-completion extra field
  "rollout_meta":   [json.dumps(s.meta) for s in samples],
}
```

**`run_replay_rollout` turn loop** (batched, active-set):
1. `prompt_ids` = `tokenizer.apply_chat_template(script.prompt_messages, add_generation_prompt=True, tokenize=True)`. Derive the assistant **turn-end token id(s)** once via a dummy render (`[user:"dummy", assistant:"dummy"]`, take trailing special token(s)); assert it includes `tokenizer.eos_token_id` — for Gemma chat tokenizers eos is `<end_of_turn>`; if they differ, generation must stop on the turn-end id and the force-append token must be `tokenizer.eos_token_id`-compatible (**[IMPLEMENTER DECIDES]** the exact reconciliation after inspecting the actual Gemma-4 SFT tokenizer; unit-test locks the invariant "trajectory ends on `tokenizer.eos_token_id`").
2. While any sample active: left-pad the batch of `prompt_ids + completion_ids`, call `model.generate(..., max_new_tokens=cfg.per_turn_max_new_tokens, do_sample=cfg.do_sample, temperature, top_p, eos_token_id=turn_end_ids)` under `torch.no_grad()` with `model.eval()` (restore mode after — mirror `_HeldOutEvalCallback._evaluate`, grpo.py:732-779). Per sequence, take generated ids up to and **including** the first turn-end token → append to `completion_ids` with `env_mask += [1]*len`. If no turn-end within budget: append the sampled ids, then **force-append the turn-end id with mask 0** (not sampled → not trained).
3. Decode the turn (`skip_special_tokens=True`) → `turn_texts.append`; parse transitions via `extract_state_annotations` (reward_utils.py:17-22 → `parse_state_transitions`, eval/state_accuracy.py:64-79).
4. **Alignment & truncation predicate** — `classify_turn(pred_transitions, cursor, gold_transitions)` (pure, shared with the reward):
   - `effective = [t for t in pred_transitions if t[0] != t[1]]` (self-loops `(X,X)` are "stay" markers, legal per `transition_legality_score` semantics — neutral for alignment).
   - `effective == []` → `("stall", cursor)`.
   - `effective == gold_transitions[cursor : cursor+len(effective)]` (exact, order-sensitive, consecutive; ≥1 element) → `("advance", cursor+len(effective))`. Multi-step compression is allowed **only** as an exact consecutive gold run.
   - anything else → `("diverged", cursor)` — this *is* the "leaves the gold-reachable set" rule, keyed off gold `state_sequence` order, never a re-simulated transition function (the serialized graph has no trigger semantics — verified: `transitions[].condition` is an opaque label).
5. **Stop conditions**, checked after the turn (the turn's tokens always stay in the trajectory and are scored):
   - `diverged` → stop, `stop_reason="diverged"`;
   - `cursor == len(gold_transitions)` → stop, `"gold_complete"`;
   - consecutive stalls ≥ `stall_turn_limit` → stop, `"stall"`;
   - turn index `t+1 >= min(len(segments), cfg.max_turns)` → stop, `"script_exhausted"` / `"turn_cap"`;
   - `len(completion_ids) + per_turn_max_new_tokens + len(next_segment_ids) + 8 > cfg.max_completion_tokens` → stop, `"budget"`.
6. **Gold injection** (if continuing): compute segment ids with the **dummy-conversation diff** — the exact in-library technique of `_get_tool_suffix_ids` (grpo_trainer.py:1381-1408): render `[{user:"dummy"},{assistant:"dummy"}]` without generation prompt, render the same + `segments[t]` with `add_generation_prompt=True`, trim the prefix at its last eos, assert prefix property, take the suffix. Append suffix ids with `env_mask += [0]*len` (covers tool text, user text, and the next `<start_of_turn>model` header — mirroring `_tool_call_loop`'s mask update at :1551-1559). Tool results are injected **unconditionally** from gold, whether or not the model emitted the matching call (replay-scripted; same semantics as `_replay_conversation`'s "GT tool responses kept as-is", agent_benchmark.py:441-445). Cache segment-id computation per (script, t).
7. **Final-token guarantee:** if the loop ends with a mask-0 injected suffix pending or a non-eos tail, force-append `tokenizer.eos_token_id` with mask 0 (keeps `is_truncated` at :1674 False and immunizes against `mask_truncated_completions`).
8. **Mask invariants** (assert per sample, cheap): `len(env_mask)==len(completion_ids)`; `sum(env_mask) > 0`; first token has mask 1.

Cost model: 32-sample generation batch × median ~8–13 rounds × ≤256 tokens ≈ 3–6× the single-turn ~31 s/step HF-rollout cost → ~2–4 min/generation cycle; 50-step diagnostic ≈ 2–3 H100-h. Trajectory v1 is **HF-generate only**: if the YAML requests `generation_backend: vllm` with trajectory mode, force HF with a warning (Gemma-4 can't use Unsloth vLLM anyway — Risk R9, grpo.py:526-543).

### 3c. Trajectory reward

**File:** `src/llm_workflow_agents/training/rewards/reward_business_logic.py` (append; register in `grpo.py:_REWARD_REGISTRY` as `"reward_business_logic_trajectory"`).

```python
def reward_business_logic_trajectory(
    prompts: list[Any],
    trajectories: list[list[str]],       # per-completion model turn texts
    metas: list[dict[str, Any]],         # rollout_meta (trusted: produced by our rollout, not the model)
    ground_truths: list[dict[str, Any]], # decoded ground_truth column
) -> list[float]:
```

Per completion, with `gold = [(f,t)...]` from `gt["state_sequence"]`, `T = len(gold)` (≥1 by loader invariant):

1. **Recompute alignment** by replaying `classify_turn` over `extract_state_annotations(turn)` for each turn (same pure function the rollout used — single source of truth; warn-log if the recomputed cursor ≠ `meta["cursor"]`). Yields `cursor`, `n_stall_turns`, `pred_transitions_all` (flat, ordered), `pred_tools_all` (flat, ordered, via `extract_tool_calls` per turn).
2. Components:
   - `r_coverage = cursor / T` — **order-sensitive by construction** (cursor only advances on the exact next gold transition or an exact consecutive gold run) and **capped at 1.0 structurally** (cursor ≤ T).
   - `r_tool = graded_tool_call_f1(pred_tools_all, _strip_placeholder_args(gt["tool_calls"]))` (reward_utils.py:50-60 → `compute_argument_graded_f1`, eval/tool_call_f1.py:258; placeholder-stub handling :116-131). Gold side = the **whole conversation's** tool calls (locked decision 3): recall scales with progress, precision punishes spam.
   - `r_terminal = 1.0` iff `meta["stop_reason"] == "gold_complete"` **and** `pred_transitions_all` is non-empty **and** `pred_transitions_all[-1][1] == gt["terminal_state"]`; component **active only when** `gt["terminal_reached"]` (else dropped + weights renormalized — existing pattern, reward_business_logic.py:181-184).
   - `r_legality = transition_legality_score(pred_transitions_all, gt["valid_transitions"]) * stall_factor`, where `stall_factor = 1 - n_stall_turns / max(n_model_turns, 1)`; component active only when `valid_transitions` non-empty (:173-176 pattern).
3. `score = Σ wᵢsᵢ / Σ wᵢ` over active components with `W = (0.40, 0.40, 0.10, 0.10)`; clip to [0,1].

**Anti-hacking guards** (locked decision 3, made concrete): order-sensitivity + coverage cap via the cursor; stall farming punished twice (stall_factor + the rollout's consecutive-stall stop); transition spam ends the rollout (divergence) *and* depresses legality; tool spam depresses F1 precision; and the independent strict held-out composite (§3d) + `_is_reward_hacking` auto-stop (grpo.py:451-468) remains the backstop.

**Adapter** — `grpo.py::_make_trajectory_reward_adapter(reward_fn)`: same keyword contract as `_make_reward_adapter` (:334-340) but reads `kwargs["trajectory"]` and `kwargs["rollout_meta"]` (per-completion extra fields, JSON-decoded) **instead of collapsing `completions` to the last message** (:363-370 — that collapse would hand the reward TRL's `batch_decode` of the whole interleaved stream, :1629-1630, which must not be scored). Missing `trajectory` key → hard error. Keep the `_LATEST_INSTRUMENTATION` unique-completions stash (:376-387), keyed on prompt, using `"\n".join(turn_texts)` as the completion identity.

### 3d. Trajectory held-out eval callback

**File:** `grpo.py` — new `_TrajectoryHeldOutEvalCallback` replacing `_HeldOutEvalCallback` (:712-814) when trajectory mode is on. Same `on_log`/`on_step_end`/auto-stop skeleton and `_is_reward_hacking` wiring; `_evaluate` changes to:
- run `run_replay_rollout(model, tokenizer, heldout_scripts, cfg_greedy)` with `do_sample=False` on `eval_held_out_num_prompts` **validation conversations** (scripts built by `_load_grpo_trajectory_jsonl(..., split="validation")`);
- score with a **strict** composite, numerically independent of the graded training reward (same design rationale as `_heldout_composite_score`, :394-448): `0.4 · state_sequence_match(pred_transitions_all, gold)` (strict positional accuracy over the FULL gold sequence — punishes truncation differently than prefix-fraction coverage; reward_utils.py:32-40) `+ 0.4 · tool_call_f1(pred_tools_all, gt_tools)` (strict AST, :43-47) `+ 0.2 · terminal` (strict, `reached_terminal` on the last turn);
- log as `eval/held_out_composite` (same key, comparable across runs).
Budget: 12 conversations × ~13 greedy turns ≈ 2–4 min per eval at `eval_held_out_every: 10`.

### 3e. Config + pin fix

**New file `configs/training/grpo_cat_a_trajectory.yaml`** — copy of `grpo_cat_a_diagnostic.yaml` (keep ALL stabilizers: `scale_rewards: "none"`, `loss_type: "dr_grpo"`, `max_grad_norm: 0.2`, `learning_rate: 1.0e-6`, `beta: 0.05`, `num_generations: 16`, `generation_batch_size: 32`, `per_device_train_batch_size: 8`, `temperature: 0.8`) with:

```yaml
grpo:
  max_completion_length: 4096      # whole interleaved trajectory (p90 gold content ≈1.7K tok + scaffolding)
  max_prompt_length: 4096          # enriched system + first user only (no longer the whole conversation)
  training_steps: 50               # diagnostic first; 1000 after gate
  rollout:
    mode: "trajectory"             # anything else → legacy single-turn path, fully preserved
    max_turns: 24
    per_turn_max_new_tokens: 256
    stall_turn_limit: 2
reward:
  function: "reward_business_logic_trajectory"
monitoring:
  eval_held_out_every: 10
  eval_held_out_num_prompts: 12
logging:
  wandb_project: "llm-workflow-agents-grpo-trajectory"
```

**Wiring in `train_grpo`** (grpo.py:471-844): when `grpo.rollout.mode == "trajectory"` → call `assert_trajectory_rollout_support()`; use `_load_grpo_trajectory_jsonl`; build `rollout_func=make_replay_rollout_func(script_index, cfg)` and pass it to `GRPOTrainer(...)` (param verified at grpo_trainer.py:277; expect the one-time experimental warning, :421-429); use `_make_trajectory_reward_adapter`; swap in the trajectory callback; force `use_vllm=False` for trajectory mode; assert `mask_truncated_completions` unset/False. Also add a `rollout_trajectory_stats` metric hook: stash mean turns / diverged-frac / coverage from `rollout_meta` into `_LATEST_INSTRUMENTATION` and surface via `_UniqueCompletionsCallback` (:690-707).

**Pin fix:** `pyproject.toml:64` and `requirements-train.txt:22`: `trl==0.24.0` → `trl==1.0.0` (the hooks require it). Opportunistically correct the stale "TRL 0.23.1" comments in `grpo.py` (:165, :316-321, :661-664) — non-load-bearing but misleading. Note in the PR that **`.venv-train` on the training box must be re-synced** and the runtime assertion is the enforcement.

---

## 4. Cheap-first micro-probe (BUILD AND RUN FIRST)

**File:** `scripts/trajectory_variance_probe.py` — mirrors `scripts/rft_headroom_probe.py`'s structure (pure `summarize`/`classify_gate` + Unsloth checkpoint loader from `preflight_entropy_diag.py::_generate_for_checkpoint`'s loading idiom). No TRL trainer involved — it drives `run_replay_rollout` + `reward_business_logic_trajectory` directly, which is precisely why it derisks the design before any training step.

Per conversation (default 50, from `validation.jsonl`, seeded sample): duplicate the script N=8×, run the batched replay rollout at T=0.8/top_p 0.95 (matching the training config so numbers transfer), score, then report:
- `median_reward_std`, `mean_reward_std`, `frac_collapsed_groups` (std < 0.01) — direct comparison against the single-turn preflight numbers;
- coverage stats: mean/p10/p90 coverage, mean within-group coverage spread (max−min);
- `mean_model_turns`, stop-reason histogram (`diverged`/`gold_complete`/`stall`/`turn_cap`/`budget`/`script_exhausted`);
- reward-rung histogram (distinct rewards per group, mirroring `rung_histogram`);
- mask audit result (§1 criterion 3) on the first batch;
- env assertions: `trl.__version__`, env_mask pathway present.

**Gate (pure `classify_gate`, unit-tested):**
- `GO_TRAJECTORY`: `median_reward_std ≥ 0.05` AND `frac_collapsed_groups < 0.50` AND `mean_model_turns ≥ 3.0` AND `0.05 < mean_coverage < 0.95`.
- `NO_GO_VARIANCE`: `median_reward_std < 0.02` — the trajectory lattice still collapses; abandon (§7).
- `NO_GO_TRUNCATION`: `mean_model_turns < 2.0` — over-truncation re-collapses variance; either relax the predicate (**[IMPLEMENTER DECIDES]**: first knob is `stall_turn_limit` 2→3; second is tolerating one non-gold-but-legal edge before stopping) and re-probe, or abandon.
- `MARGINAL`: otherwise → inspect histograms, re-probe with 150 conversations.

Runtime estimate: 400 trajectories, batch 32, ~6–13 turns → **~30–60 min on the H100**. Command in §6.

**Result (2026-07-09, ckpt-1000, 48 validation conversations [2 skipped: gold-transition/turn mismatch], N=8, T=0.8 / top_p 0.95, ~29 min wall):** verdict **MARGINAL** — 3 of 4 GO checks pass, only median std misses.
- `median_reward_std = 0.0294` (GO ≥ 0.05 — miss; but clears the 0.02 NO_GO_VARIANCE floor). `mean_reward_std = 0.0503` → right-skewed: a minority of high-variance groups sit at/above the bar while the median is below.
- `frac_collapsed_groups = 0.271` (GO ✓) — **down from 0.716 on the single-turn headroom probe (same checkpoint)**; the collapse that killed four single-turn GRPO runs is largely resolved. Rung histogram `{1:9, 2:18, 3:12, 4:7, 5:2}` → 81% of groups occupy ≥2 rungs (vs. 29% single-turn).
- `mean_model_turns = 4.60` (GO ✓, no over-truncation — R3 clear); `mean_coverage = 0.292`, p10→p90 0.07→0.67, within-group spread 0.25 (GO ✓).
- Stop reasons `{diverged: 334, gold_complete: 20, script_exhausted: 26, stall: 4}` — divergence (the variance source) dominates as expected; only 4/384 stalls.
- **Mask audit PASS** (R1): `len_match=True`, model_frac=0.826 (in the ~0.35–0.7… note: above band, worth a glance), ends_on_eos=True. Env-token masking wired correctly.
- Env (R2): `trl==1.0.0` re-synced into `.venv-train` via `install_train.sh:62`; `assert_trajectory_rollout_support()` passes.

The premise holds — trajectory aggregation *does* manufacture within-group variance the single-turn reward could not. But median std lands just under the GO line on n=48. Per the MARGINAL branch, **re-probe with ~150 conversations** to tighten the estimate before building trainer wiring; the structural signal (collapse frac, turn count, coverage, mask) is already strongly favorable. Artifact: `runs/preflight/trajectory_variance_probe.json` (gitignored).

**Re-probe (2026-07-10, ckpt-1000, 145 validation conversations [150 requested, 5 skipped: gold-transition/turn mismatch], N=8, T=0.8 / top_p 0.95, ~72 min wall):** verdict **MARGINAL again** — same 3-of-4 pattern, median std still the only miss, but the tighter estimate resolves the n=48 uncertainty *against* GO rather than for it.
- `median_reward_std = 0.0219` — **down** from 0.0294 at n=48, now barely above the 0.02 `NO_GO_VARIANCE` floor. The n=48 median was optimistic sampling noise, not a right-skew the median would clear: this was a CI problem and the CI has now resolved **below** the 0.05 GO line, not toward it. `mean_reward_std = 0.0377` (down from 0.0503) — the right skew persists but is milder; a shrinking minority of high-variance groups no longer drags the median up.
- `frac_collapsed_groups = 0.366` (GO ✓, still well under 0.50; up from 0.271 at n=48 but nowhere near the single-turn 0.716). Rung histogram `{1:38, 2:49, 3:33, 4:12, 5:12, 6:1}` → 74% of groups occupy ≥2 rungs.
- `mean_model_turns = 4.48` (GO ✓, no over-truncation); `mean_coverage = 0.279`, p10→p90 0.0625→0.60, within-group spread 0.19 (GO ✓). Stop reasons `{diverged: 1017, gold_complete: 44, script_exhausted: 88, stall: 11}` — divergence still dominates as designed.
- **Mask audit PASS** (R1): `len_match=True`, `ends_on_eos=True`, `model_frac=0.923` — **above** the expected ~0.35–0.7 band (n=48 was 0.826, already flagged); worth a glance when trainer wiring lands, but not a probe blocker.

**Routing — do NOT build trainer wiring.** Not GO (median 0.0219 < 0.05), and n=150 shows this is a genuine variance ceiling at ckpt-1000, not a sample-size artifact. Per §4 MARGINAL / §7, try exactly one cheap knob before abandoning the trajectory track: re-probe at **T→1.0** (widen sampling diversity — the same lever the single-turn memo suggested) **or** on a **later checkpoint (1500/2000)** (a less gold-deterministic policy). One, not both at once. If that also lands MARGINAL/NO_GO, fall back to single-turn RFT on its concentrated-but-real headroom (`docs/grpo_viability_investigation.md` Path #1) or ship SFT-only for Phase 2 Cat A. Artifact: `runs/preflight/trajectory_variance_probe_n150.json` (gitignored).

**The one cheap knob — T→1.0 (2026-07-13, ckpt-1000, 145 validation conversations [150 requested, 5 skipped], N=8, T=1.0 / top_p 0.95, ~81 min wall):** verdict **MARGINAL again** — the knob did not clear GO. Temperature moved the collapse metrics but *not* the decision metric.
- `median_reward_std = 0.0245` — up only +0.0026 from the T=0.8 baseline (0.0219), still **less than half** the 0.05 GO line and only just above the 0.02 `NO_GO_VARIANCE` floor. `mean_reward_std = 0.0444` (up from 0.0377) — the right skew persists; higher T fattens the high-variance tail but does not lift the median group off the coarse rung lattice.
- `frac_collapsed_groups = 0.269` (GO ✓, **improved** from 0.366) and rung histogram `{1:28, 2:53, 3:34, 4:16, 5:10, 6:3, 7:1}` → 81% of groups occupy ≥2 rungs (up from 74%). So T→1.0 *does* help occupancy — but the extra rungs land close together, so the within-group *std* barely rises. This is the tell that the ceiling is the reward lattice's rung *spacing* at this policy, not sampling diversity.
- `mean_model_turns = 4.48` (GO ✓, unchanged); `mean_coverage = 0.282`, p10→p90 0.059→0.60, within-group spread 0.226 (GO ✓). Stop reasons `{diverged: 1025, gold_complete: 43, script_exhausted: 85, stall: 7}` — divergence still dominates as designed.
- **Mask audit PASS** (R1): `len_match=True`, `ends_on_eos=True`, `model_frac=0.835` (down from 0.923 at T=0.8 — closer to the expected ~0.35–0.7 band top, still slightly above; not a blocker). Artifact: `runs/preflight/trajectory_variance_probe_t1.json` (gitignored).

**Routing — trajectory track's cheap knobs are exhausted; fall back.** The one permitted knob (T→1.0) landed MARGINAL, confirming a real variance ceiling at ckpt-1000 that temperature cannot lift (it raises rung *occupancy* but not rung *spread*). Per the n=150 routing above ("if that also lands MARGINAL/NO_GO, fall back"), do **not** now try the later-checkpoint knob (that would be "both") and do **not** build trainer wiring. Fall back to one of: **(a)** single-turn RFT on its concentrated-but-real headroom (`docs/grpo_viability_investigation.md` Path #1 — the headroom probe already qualified this as MARGINAL-but-actionable at 13% frontier), or **(b)** ship SFT-only for Phase 2 Cat A. This fork is a strategy decision for the maintainer.

**Probe-runnability note:** the first live run surfaced three Gemma-4 26B-A4B GPU-path bugs in the probe/rollout (the GPU path was never executed when this spec was authored — `.venv-train` absent, R2). Fixed under PR #45: (1) use `_patch_unsloth_gemma4_proxy_iter` not grpo.py's `_unwrap_...` for the KV-zero proxy (validator hits `__getattr__`, proxy must stay live); (2) return the inner `.tokenizer` not the multimodal `Gemma4Processor` (nested-ids); (3) move `_left_pad` tensors to the model device before `generate`.

## 5. Risk register

| # | Failure mode | Detection signal | Mitigation |
|---|---|---|---|
| R1 | **Token mask wrong → silent garbage gradients** (injected gold tokens trained, or model tokens masked) | Step-0 mask audit (decode mask-1 vs mask-0 spans, log to W&B); per-sample invariants in rollout; `train/env_mask_frac` (expect ~0.35–0.7, drift = bug); unit test with mocked `generate` | The audit is a hard assert at probe time; `env_mask` semantics pinned by tests against grpo_trainer.py:1649/:2265 |
| R2 | **TRL/Unsloth pin drift** — training box `.venv-train` has trl≠1.0.0, or Unsloth's RL patcher replaces `_generate` and drops rollout_func/env_mask | `assert_trajectory_rollout_support()` hard-fails at startup and in the probe (source-inspection of the *patched* class) | Pin fix (§3e); fallback: minimal `GRPOTrainer` subclass overriding `_generate` (§2.2) |
| R3 | **Truncation too aggressive** → 1–2-turn trajectories → coverage ∈ {0, 1/T} → variance re-collapse | Probe: `mean_model_turns`, stop-reason histogram (`diverged` ≫ others at turn 1); in-run `train/rollout_mean_turns`, `train/rollout_diverged_frac` | Stall tolerance (limit 2), self-loops neutral, exact-consecutive compression allowed; relaxation knobs gated on re-probe (§4) |
| R4 | **Reward hacking**: stall/self-loop farming, transition spam to inflate legality, tool spam, or coverage-seeking degenerate prose incoherent with injected gold user turns | Strict trajectory held-out composite diverging from training reward (`_is_reward_hacking` auto-stop); `unique_completions_per_group`; manual transcript audit of 20 probe trajectories | stall_factor × legality, divergence-stop on spam, F1 precision term; held-out metric numerically independent (§3d) |
| R5 | **Off-policy/incoherence confound**: model stays state-aligned but semantically diverges, so scripted gold user replies become incoherent context — training on plausible-looking but mismatched conversations | Probe-stage human read of `gold_complete` transcripts; held-out composite flat-or-down while reward rises; `completions/mean_length` drift | Accepted for v1 (locked decision 1/2 — no user simulator available); divergence predicate keys on state only; document as known ceiling |
| R6 | **dr_grpo loss-scale shift** (`max_completion_length` 512→4096 denominator, :2396) → vanishing updates masquerading as "stable but flat" | `grad_norm` persistently ≪ historical stable range; reward_std healthy but reward mean immobile | LR bump 1e-6→2e-6 [IMPLEMENTER DECIDES] only after 50-step evidence; never together with other changes |

## 6. Implementation plan (handover to Opus)

**Target model:** claude-opus-4-8 — multi-file, correctness-critical token accounting; the spec resolves all design questions except the three explicitly marked [IMPLEMENTER DECIDES].
**Context the implementer lacks:** everything above; assume zero shared memory. Repo: `/workspaces/ai-agent-llms`. Use `uv`/activated `.venv` per project convention; GPU steps run on the H100 box via `.venv-train`. Branch first (main is protected but bypassable — verify `git branch --show-current`); ship via auto-merged squash PR per project convention.

Ordered, independently verifiable steps:

1. **Pins + env gate** — edit `pyproject.toml:64`, `requirements-train.txt:22` → `trl==1.0.0`; add `assert_trajectory_rollout_support()` to `trajectory_rollout.py` (new file).
   *Verify:* `grep -n "trl==" pyproject.toml requirements-train.txt`; `source .venv/bin/activate && python -c "from llm_workflow_agents.training.trajectory_rollout import assert_trajectory_rollout_support"` (assertion itself needs the train box).
2. **Pure core** — `trajectory_rollout.py`: `GoldScript`, `build_gold_script`, `prompt_key`, `classify_turn`, `walk_trajectory_alignment`; tests `tests/unit/test_trajectory_rollout.py` (segmentation on a synthetic conversation; advance/stall/diverge/compression/self-loop/cursor-cap cases; key stability; loader invariant `len(gold)==n_asst`).
   *Verify:* `source .venv/bin/activate && pytest tests/unit/test_trajectory_rollout.py -q`
3. **Reward** — `reward_business_logic_trajectory` + registry entry + `_make_trajectory_reward_adapter`; tests `tests/unit/test_reward_trajectory.py` (known-answer trajectories with hand-computed scores; renormalization when terminal/legality inactive; stall_factor; coverage cap; order sensitivity — permuted gold path scores lower).
   *Verify:* `pytest tests/unit/test_reward_trajectory.py -q`
4. **Rollout loop** — `run_replay_rollout` + `make_replay_rollout_func` + dummy-diff segment tokenizer + mask assembly. Tests: monkeypatch `model.generate` with canned token scripts and a small real tokenizer fixture (**[IMPLEMENTER DECIDES]** fixture: a cached tiny ChatML tokenizer or a minimal jinja stub — must exercise the prefix-diff assertion) → assert env_mask zeros exactly over injected segments/forced eos, ones over sampled spans, final token == eos.
   *Verify:* `pytest tests/unit/test_trajectory_rollout.py -q -k mask`
5. **Micro-probe** — `scripts/trajectory_variance_probe.py` (§4; pure `summarize_trajectory_probe`/`classify_gate` unit-tested in `tests/unit/test_trajectory_probe.py`). **Run it on the H100 before writing any trainer wiring beyond what the probe needs:**
   `.venv-train/bin/python scripts/trajectory_variance_probe.py --checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000 --data-dir data/output/grpo/task_a --split validation --n-conversations 50 --n-completions 8 --temperature 0.8 --top-p 0.95 --output runs/preflight/trajectory_variance_probe.json`
   (ckpt-1000 per the viability memo's loss-elbow seeding; ckpt-500 fallback.) **STOP at NO_GO** (§7).
6. **Trainer wiring** — `_load_grpo_trajectory_jsonl`, trajectory branch in `train_grpo`, `_TrajectoryHeldOutEvalCallback`, rollout-stats instrumentation; `configs/training/grpo_cat_a_trajectory.yaml`. Legacy single-turn path must remain byte-identical in behavior (existing tests `test_grpo_outbound.py`, `test_reward_functions.py` stay green).
   *Verify:* `pytest tests/unit -q` (full unit suite).
7. **50-step diagnostic** —
   `./scripts/run_phase2_grpo.sh --grpo-config configs/training/grpo_cat_a_trajectory.yaml --sft-checkpoint checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000`
   Watch W&B (`llm-workflow-agents-grpo-trajectory`): kill criteria grad_norm > 50 ×3 steps or KL > 10 (inherited); success per §1.4. Only then flip `training_steps: 1000`.

**Constraints & non-goals:** do not modify the single-turn reward, loader, or adapter behavior; no user simulator or tool executor (locked); no vLLM rollouts in trajectory v1; no TRL fork; keep the 0.40/0.40/0.10/0.10 weights; don't touch `scale_rewards`/`loss_type`/`max_grad_norm`/`beta` stabilizers.

## 7. What would falsify this design (abandon → RFT/SFT-only)

1. **Variance doesn't materialize:** probe `median_reward_std < 0.02` while `mean_model_turns ≥ 3` — trajectories are long but groups still land on one rung (e.g., the SFT policy walks the gold path near-deterministically at T=0.8, coverage ≈ 1.0 everywhere). The premise "aggregation manufactures variance" is wrong for this policy/data; the task is trajectory-saturated → ship SFT-only or RFT.
2. **Over-truncation floor is structural:** `mean_model_turns < 2` even after stall-limit 3 and one-legal-edge tolerance — the policy exits the gold-reachable set immediately, so on-policy multi-turn conditioning has no support. Free-run + gold-script replay (locked decision 2) cannot work; fall back to RFT on gold-context turns.
3. **Stable but flat, again:** 50-step diagnostic with healthy `reward_std` (> 0.05) and sane optimization (grad_norm, KL nominal) yet reward mean and held-out composite immobile after the LR check (R6) — variance was necessary but not sufficient; step economics at lr ~1e-6 can't move a 26B policy on this signal. Kill per the same step-economics argument that killed single-turn (`docs/grpo_viability_investigation.md` §1).
4. **Reward-quality divergence:** training reward climbs while the strict trajectory held-out composite falls across ≥2 consecutive evals (auto-stop fires) and transcript audit confirms degenerate coverage-farming — the blend is hackable in practice despite the guards; retire the online optimizer for good.
5. **Mechanism unsound on the real stack:** the chat-template prefix-diff assertion (`_get_tool_suffix_ids`-style) cannot be satisfied for the Gemma-4 tokenizer (template not prefix-stable), or the training box's Unsloth-patched trainer loses `env_mask` with no subclass workaround — the masking correctness guarantee is gone; do not train with an unverifiable mask.

---
**Verification provenance (all checked 2026-07-09):** TRL 1.0.0 at `.venv/lib/python3.12/site-packages/trl` — line anchors re-confirmed this session: RolloutFunc contract :1606-1613, init param :277, dispatch/env_mask :1646-1649, completion-length count :1653-1656, extra-field merge :1949-1956, loss-mask product :2149/:2265, `_get_tool_suffix_ids` :1381-1408, `_tool_call_loop` mask update :1551-1559, `mask_truncated_completions` :1796-1804. Repo: `grpo.py` :161-302/:315-391/:363-370/:394-448/:526-543/:649-669/:712-814; `reward_utils.py` :17-135; `reward_business_logic.py` :82-113/:134-194; `agent_benchmark.py` :433-597; `configs/training/grpo_cat_a_diagnostic.yaml`; `scripts/rft_headroom_probe.py`. Data: 2,502/290 conversations; per-turn annotation invariant and adjacency patterns verified on live samples; `.venv-train` absent on this box (pin-drift risk R2 is live, not hypothetical).
