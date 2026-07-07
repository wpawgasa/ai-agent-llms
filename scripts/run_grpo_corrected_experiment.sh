#!/usr/bin/env bash
# Corrected GRPO experiment runner for Gemma-4-26B-A4B Cat A.
#
# Executes the gated protocol from docs/grpo_diagnosis_gemma4_26b.md addendum
# "New-Dataset Retry (2026-07-07)":
#
#   0. provenance  — hash the on-disk data so it can be checked vs the W&B
#                    artifacts of the SFT/GRPO runs (splits were rewritten
#                    after the SFT checkpoints; do not trust mtimes).
#   1. preflight   — within-group reward-variance check on the SFT base, NO
#                    optimizer. Gate: frac_collapsed_groups < 0.5 AND
#                    mean_unique_per_group > 3. Auto-falls back to the
#                    higher-entropy checkpoint-500 if checkpoint-1000 fails.
#   2. diagnostic  — stabilized 50-step GRPO (scale_rewards=none, dr_grpo,
#                    max_grad_norm=0.2, ...) with the live held-out R5 guardrail.
#   3. analyze     — read the diagnostic's trainer_state.json and print the
#                    go/no-go verdict for the full 1000-step run.
#
# Usage:
#   scripts/run_grpo_corrected_experiment.sh [all|provenance|preflight|diagnostic|analyze]
#
# Env overrides:
#   SFT_CKPT           default checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000
#   SFT_CKPT_FALLBACK  default checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-500
#   GRPO_DATA_DIR      default data/output/grpo/task_a
#   DIAG_CONFIG        default configs/training/grpo_cat_a_diagnostic.yaml
#   PREFLIGHT_N        default 20   (prompts)
#   PREFLIGHT_K        default 8    (rollouts/prompt)
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# ---- config ----------------------------------------------------------------
SFT_CKPT="${SFT_CKPT:-checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-1000}"
SFT_CKPT_FALLBACK="${SFT_CKPT_FALLBACK:-checkpoints/sft_cat_a/gemma-4-26B-A4B-it/checkpoint-500}"
GRPO_DATA_DIR="${GRPO_DATA_DIR:-data/output/grpo/task_a}"
DIAG_CONFIG="${DIAG_CONFIG:-configs/training/grpo_cat_a_diagnostic.yaml}"
PREFLIGHT_N="${PREFLIGHT_N:-20}"
PREFLIGHT_K="${PREFLIGHT_K:-8}"

# Gate thresholds (docs/grpo_diagnosis_gemma4_26b.md)
GATE_FRAC_COLLAPSED_MAX="0.5"
GATE_UNIQUE_MIN="3"
# Diagnostic stability kill/gate thresholds
KILL_GRAD_NORM="50"
KILL_KL="10"
GATE_REWARD_STD_MIN="0.02"

# The preflight must run under .venv-train (transformers 5.9.0 knows gemma4);
# the default /opt/venv fails at config load. See doc L754.
PY_TRAIN=".venv-train/bin/python"

RUNS_DIR="runs/preflight"
STAMP="$(date +%Y%m%d_%H%M%S)"

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die()    { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

require_venv_train() {
  [ -x "$PY_TRAIN" ] || die "$PY_TRAIN not found. The preflight needs the transformers-5.9.0 venv (gemma4). Create/activate .venv-train first."
}

# ---- 0. provenance ---------------------------------------------------------
provenance() {
  banner "STAGE 0 — data provenance (verify vs W&B artifacts)"
  mkdir -p "$RUNS_DIR"
  local out="$RUNS_DIR/provenance_${STAMP}.txt"
  {
    echo "# GRPO corrected-experiment data provenance — $STAMP"
    echo "# Compare these sha256 against the dataset artifact hashes logged by"
    echo "#   SFT  run: https://wandb.ai/wpawgasa/huggingface/runs/uklfswk5"
    echo "#   GRPO run: https://wandb.ai/wpawgasa/huggingface/runs/bqbxnqxw"
    echo
    for f in \
      "$GRPO_DATA_DIR/train.jsonl" \
      "$GRPO_DATA_DIR/validation.jsonl" \
      "data/output/sft/task_a_splits/train.jsonl" \
      "data/output/sft/task_a_splits/validation.jsonl" \
      "data/output/sft/task_a_splits/test.jsonl"
    do
      if [ -f "$f" ]; then
        printf '%s  %s  (mtime %s, %s lines)\n' \
          "$(sha256sum "$f" | cut -d' ' -f1)" "$f" \
          "$(date -r "$f" +%Y-%m-%dT%H:%M:%S)" "$(wc -l < "$f")"
      else
        printf 'MISSING  %s\n' "$f"
      fi
    done
  } | tee "$out"
  echo
  echo "Saved: $out"
  echo "ACTION: confirm the GRPO train/validation hashes match the artifact"
  echo "        the SFT run trained on. If they differ, repin before training."
}

# ---- 1. preflight ----------------------------------------------------------
# Runs preflight_entropy_diag on one checkpoint, then evaluates the gate from
# the emitted JSON. Echoes PASS/FAIL. Returns 0 on PASS, 1 on FAIL.
run_preflight_one() {
  local ckpt="$1"
  local tag; tag="$(basename "$ckpt")"
  local out="$RUNS_DIR/newdata_${PREFLIGHT_N}x${PREFLIGHT_K}_${tag}.json"
  [ -d "$ckpt" ] || die "SFT checkpoint not found: $ckpt"

  banner "STAGE 1 — preflight entropy diag on $tag (N=$PREFLIGHT_N K=$PREFLIGHT_K)"
  "$PY_TRAIN" scripts/preflight_entropy_diag.py \
    --checkpoints "$ckpt" \
    --data-dir "$GRPO_DATA_DIR" \
    --split validation \
    --n-prompts "$PREFLIGHT_N" \
    --n-completions "$PREFLIGHT_K" \
    --output "$out"

  "$PY_TRAIN" - "$out" "$GATE_FRAC_COLLAPSED_MAX" "$GATE_UNIQUE_MIN" <<'PY'
import json, sys
out, frac_max, uniq_min = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
d = json.load(open(out))
# aggregate metrics may sit at top level or under a per-checkpoint summary
def find(keys):
    stack = [d]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k in keys:
                if k in cur and isinstance(cur[k], (int, float)):
                    return cur[k]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None
frac = find(["frac_collapsed_groups"])
uniq = find(["mean_unique_per_group"])
print(f"\n  frac_collapsed_groups = {frac}  (gate: < {frac_max})")
print(f"  mean_unique_per_group = {uniq}  (gate: > {uniq_min})")
ok = (frac is not None and uniq is not None and frac < frac_max and uniq > uniq_min)
print("  RESULT:", "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m")
sys.exit(0 if ok else 1)
PY
}

preflight() {
  require_venv_train
  mkdir -p "$RUNS_DIR"
  if run_preflight_one "$SFT_CKPT"; then
    SELECTED_CKPT="$SFT_CKPT"
    echo "Preflight PASSED on $(basename "$SFT_CKPT") — use it as the GRPO seed."
    return 0
  fi
  echo
  echo "Preflight FAILED on $(basename "$SFT_CKPT"); retrying on higher-entropy fallback."
  if run_preflight_one "$SFT_CKPT_FALLBACK"; then
    SELECTED_CKPT="$SFT_CKPT_FALLBACK"
    echo "Preflight PASSED on $(basename "$SFT_CKPT_FALLBACK") — use it as the GRPO seed."
    return 0
  fi
  die "Preflight FAILED on both checkpoints. The SFT policy under-produces within-group variance. Do NOT run GRPO — re-SFT with num_epochs=1 + early stop (see doc 'Going Forward' section), then re-preflight."
}

# ---- 2. diagnostic ---------------------------------------------------------
diagnostic() {
  local seed="${SELECTED_CKPT:-$SFT_CKPT}"
  [ -d "$seed" ] || die "GRPO seed checkpoint not found: $seed"
  banner "STAGE 2 — stabilized 50-step GRPO diagnostic from $(basename "$seed")"
  cat <<EOF
Config : $DIAG_CONFIG  (scale_rewards=none, loss_type=dr_grpo, max_grad_norm=0.2)
Watch  : train/grad_norm, train/kl, train/frac_reward_zero_std,
         train/unique_completions_per_group, eval/held_out_composite
KILL   : grad_norm > $KILL_GRAD_NORM for 3 consecutive steps, OR kl > $KILL_KL at any step
EOF
  ./scripts/run_phase2_grpo.sh \
    --grpo-config "$DIAG_CONFIG" \
    --sft-checkpoint "$seed" \
    --skip-filter
}

# ---- 3. analyze ------------------------------------------------------------
# Post-hoc go/no-go read of the diagnostic's trainer_state.json.
analyze() {
  banner "STAGE 3 — diagnostic go/no-go verdict"
  local stem; stem="$(basename "${DIAG_CONFIG%.yaml}")"
  local ckpt_root="checkpoints/${stem}/gemma-4-26B-A4B-it"
  "${PY_TRAIN:-python3}" - "$ckpt_root" "$KILL_GRAD_NORM" "$KILL_KL" "$GATE_REWARD_STD_MIN" <<'PY'
import json, sys, glob, os, statistics as st
root, kill_gn, kill_kl, rstd_min = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
cks = sorted(glob.glob(os.path.join(root, "checkpoint-*")),
             key=lambda p: int(p.rsplit("-", 1)[-1]))
if not cks:
    print(f"No diagnostic checkpoints under {root}. Run the diagnostic stage first."); sys.exit(2)
h = [e for e in json.load(open(os.path.join(cks[-1], "trainer_state.json")))["log_history"] if "reward" in e]
gn = [e.get("grad_norm") for e in h if e.get("grad_norm") is not None]
kl = [e.get("kl") for e in h if e.get("kl") is not None]
rs = [e.get("reward_std") for e in h if e.get("reward_std") is not None]
fz = [e.get("frac_reward_zero_std") for e in h if e.get("frac_reward_zero_std") is not None]
# 3-consecutive grad-norm > kill
run = mx = 0
for x in gn:
    run = run + 1 if x > kill_gn else 0
    mx = max(mx, run)
kl_brex = sum(1 for x in kl if x > kill_kl)
rstd_tail = st.mean(rs[-10:]) if rs else 0.0
fz_first = st.mean(fz[:10]) if fz else None
fz_last = st.mean(fz[-10:]) if fz else None
print(f"  steps logged            : {len(h)}")
print(f"  grad_norm max            : {max(gn):.1f}   (>{kill_gn} x3-consec = {mx} max run)")
print(f"  kl max                   : {max(kl):.1f}   (steps >{kill_kl}: {kl_brex})")
print(f"  reward_std (last 10 mean): {rstd_tail:.4f}   (gate > {rstd_min})")
if fz_first is not None:
    print(f"  frac_reward_zero_std     : {fz_first:.2f} -> {fz_last:.2f} (want trending down)")
unstable = mx >= 3 or kl_brex > 0
print()
if unstable:
    print("  VERDICT: \033[31mUNSTABLE\033[0m — optimizer still exploding. Halve LR / tighten max_grad_norm before any long run.")
elif rstd_tail > rstd_min:
    print("  VERDICT: \033[32mGO\033[0m — stable + reward_std healthy. Promote optimizer keys into grpo_cat_a.yaml")
    print("           (training_steps 1000, save_steps 50) and launch the full run.")
else:
    print("  VERDICT: \033[33mSTABLE-BUT-FLAT\033[0m — optimizer fixed but reward under-resolves.")
    print("           Next ticket is reward-resolution / tool-emission data slice, NOT a 1000-step run.")
PY
}

# ---- dispatch --------------------------------------------------------------
case "${1:-all}" in
  provenance) provenance ;;
  preflight)  preflight ;;
  diagnostic) diagnostic ;;
  analyze)    analyze ;;
  all)        provenance; preflight; diagnostic; analyze ;;
  *) die "unknown stage '$1' (want: all|provenance|preflight|diagnostic|analyze)" ;;
esac
