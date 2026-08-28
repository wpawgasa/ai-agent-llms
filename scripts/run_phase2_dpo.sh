#!/usr/bin/env bash
# Run Unsloth DPO/ORPO (Phase 2 preference learning) for Task A.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
DPO_CONFIG="configs/training/dpo_cat_a.yaml"
SFT_CHECKPOINT=""
METHOD=""
PAIRS_DATA_DIR="data/output/grpo/task_a"
HELDOUT="data/output/heldout/cat_a_v2_test_not_in_v1/test.jsonl"
TRAIN_PAIRS_OUT="data/output/preference/task_a/train.jsonl"
VAL_PAIRS_OUT="data/output/preference/task_a/validation.jsonl"
DRY_RUN=0
NO_WANDB=0
SKIP_PAIRS=0
CHUNK_STEPS=0

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Phase 2 DPO/ORPO runner for Task A (Gemma4-26B-A4B C2 checkpoint default).

Preference learning replaces GRPO on this task (CLAUDE.md R18): the GRPO
reward has no resolution, so it needs a checkpoint plus a (chosen, rejected)
pair set — no online generation, no reward function.

Options:
  --dpo-config PATH       DPO/ORPO training YAML (default: $DPO_CONFIG)
  --sft-checkpoint PATH   SFT checkpoint to start preference learning from.
                          If omitted, the config's own model.sft_checkpoint
                          is used unchanged.
  --method dpo|orpo       Override dpo.method in the config.
  --pairs-data-dir PATH   GRPO-format corpus the pair builder and the
                          held-out guardrail both read from
                          (default: $PAIRS_DATA_DIR)
  --heldout PATH          Contamination-guard file — pairs whose prompt
                          matches a held-out conversation are refused
                          (default: $HELDOUT)
  --skip-pairs            Skip scripts/build_preference_pairs.py (assume the
                          synthetic train/validation pair files already
                          exist)
  --chunk-steps N         Train in chunks of N steps, scoring the R5 held-out
                          guardrail BETWEEN chunks in a separate process
                          (default: 0 = one straight-through run).

                          The in-process guardrail cannot run on this model:
                          load_in_4bit reaches only 0.77 GiB of
                          Gemma-4-26B-A4B (the MoE experts are fused 3-D
                          tensors bitsandbytes cannot swap), so training holds
                          ~46 GiB and a second model copy does not fit on one
                          GPU. See CLAUDE.md R19 and
                          docs/dpo_memory_ceiling_investigation.md section 8.

                          Each chunk is its own process, so the GPU empties
                          between train and score. N must equal dpo.save_steps
                          or no checkpoint exists at the boundary. This mode
                          forces monitoring.reward_hacking_detector off.

                          Two caveats. Scoring uses
                          scripts/heldout_composite_audit.py, which samples the
                          whole validation split with a fixed seed rather than
                          the reserved guardrail slice the in-process callback
                          used; harmless while mining runs on --split train.
                          And dpo.save_total_limit still prunes old
                          checkpoints, so a stop at step K may have deleted the
                          best earlier checkpoint.
  --dry-run               Prepare pairs + patched config, exit before
                          training
  --no-wandb              Disable W&B logging (overrides YAML)
  -h, --help              Show this help

Mined negatives (scripts/mine_model_negatives.py) are NOT run by this
launcher — mining takes ~35 min on an H100 and is not idempotent across
checkpoints, so it stays a separate, deliberate step. If the config's
data.train_sources lists a model_negatives.jsonl that does not exist yet,
training fails fast with a pointer to that script.

Resume: training/dpo.py auto-resumes from the highest checkpoint-N under
this run's output_dir, the same as grpo.py. Re-launching with the same
--dpo-config (and therefore the same output_dir) continues automatically;
no --resume flag is needed.
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dpo-config)      DPO_CONFIG="$2";      shift 2 ;;
    --sft-checkpoint)  SFT_CHECKPOINT="$2";  shift 2 ;;
    --method)          METHOD="$2";          shift 2 ;;
    --pairs-data-dir)  PAIRS_DATA_DIR="$2";  shift 2 ;;
    --heldout)         HELDOUT="$2";         shift 2 ;;
    --chunk-steps)     CHUNK_STEPS="$2";     shift 2 ;;
    --skip-pairs)       SKIP_PAIRS=1;        shift   ;;
    --dry-run)          DRY_RUN=1;           shift   ;;
    --no-wandb)         NO_WANDB=1;          shift   ;;
    -h|--help)          usage                ;;
    *) echo "Error: unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Environment ───────────────────────────────────────────────────────────────
# Same rationale as run_phase2_sft.sh (CLAUDE.md R16): the fp32 cross-entropy
# logits tensor over Gemma-4's 262,144-token vocab is the largest single
# allocation, and DPO/ORPO forward BOTH chosen and rejected sequences (plus an
# implicit reference pass for DPO), so the OOM risk is at least as high here.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
# Activate .venv-train if it exists; otherwise assume the current environment
# already has Unsloth installed (e.g. .devcontainer/Dockerfile.unsloth image).
if [[ -f .venv-train/bin/activate ]]; then
  source .venv-train/bin/activate
elif ! python3 -c "import unsloth" &>/dev/null; then
  echo "Error: .venv-train/ not found and 'unsloth' is not importable in the current environment." >&2
  echo "       Run ./scripts/install_train.sh, or activate the venv that has Unsloth installed." >&2
  exit 1
fi

# ── Validate inputs ───────────────────────────────────────────────────────────
[[ -f "$DPO_CONFIG" ]] || { echo "Error: DPO config not found: $DPO_CONFIG" >&2; exit 1; }
[[ "${HF_TOKEN:-}" != "" ]] || echo "Warning: HF_TOKEN not set — gated models will fail to download." >&2

if [[ -n "$METHOD" && "$METHOD" != "dpo" && "$METHOD" != "orpo" ]]; then
  echo "Error: --method must be 'dpo' or 'orpo', got: $METHOD" >&2
  exit 1
fi

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
fi

# The contamination guard fails loudly rather than silently if this is
# missing (CLAUDE.md R18 / scripts/build_preference_pairs.py), so check for
# it here too and point at the two-tag rebuild recipe instead of a bare
# Python traceback.
if [[ $SKIP_PAIRS -eq 0 && ! -f "$HELDOUT" ]]; then
  echo "Error: held-out contamination file not found: $HELDOUT" >&2
  echo "       Rebuild it first — see scripts/build_heldout_clean_set.py's" >&2
  echo "       module docstring for the two-tag recipe." >&2
  exit 1
fi

# ── Prepare preference pairs ──────────────────────────────────────────────────
# SYNTHETIC pairs only (scripts/build_preference_pairs.py) — deterministic and
# CPU-only, so safe to run on every launch, matching split_task_a_sft.py's
# role in run_phase2_sft.sh. Mined negatives are a separate, GPU-bound step;
# see the --skip-pairs / mined-negatives note in `usage` above.
if [[ $SKIP_PAIRS -eq 0 ]]; then
  mkdir -p "$(dirname "$TRAIN_PAIRS_OUT")"
  python3 scripts/build_preference_pairs.py --split train \
    --data-dir "$PAIRS_DATA_DIR" --heldout "$HELDOUT" --out "$TRAIN_PAIRS_OUT"
  python3 scripts/build_preference_pairs.py --split validation \
    --data-dir "$PAIRS_DATA_DIR" --heldout "$HELDOUT" --out "$VAL_PAIRS_OUT"
fi

# ── Patch DPO config ──────────────────────────────────────────────────────────
# RUN_TS makes the patched config run-specific: a fixed filename here would be
# silently overwritten by the next invocation (even --dry-run), leaving no
# reliable record of what config actually produced a given checkpoint. The
# checkpoint path is held stable independently via the explicit output_dir key
# below, so the timestamp cannot leak into it. See CLAUDE.md R13 (ported from
# run_phase2_sft.sh / run_phase2_grpo.sh).
DPO_STEM=$(basename "${DPO_CONFIG%.*}")
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PATCHED_DIR="$PROJECT_ROOT/.runs/$DPO_STEM"
mkdir -p "$PATCHED_DIR"
PATCHED_CFG="$PATCHED_DIR/${DPO_STEM}_${RUN_TS}.yaml"

python3 -c "
from pathlib import Path
import yaml

cfg = yaml.safe_load(open('${DPO_CONFIG}'))
if '${SFT_CHECKPOINT}':
    cfg.setdefault('model', {})['sft_checkpoint'] = str(Path('${SFT_CHECKPOINT}').resolve())
if '${METHOD}':
    cfg.setdefault('dpo', {})['method'] = '${METHOD}'
# Pin the checkpoint directory to the *base* config stem, not the timestamped
# patched-config stem. dpo.py would otherwise derive it from the patched
# filename and write outside the DVC-tracked path. Set output_dir in the base
# config to give a run its own directory (e.g. one per method/cell).
cfg.setdefault('output_dir', Path('${DPO_CONFIG}').stem)
if ${NO_WANDB}:
    cfg.setdefault('logging', {}).pop('wandb_project', None)

with open('${PATCHED_CFG}', 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
"

RESOLVED_CHECKPOINT=$(python3 -c "
import yaml
print(yaml.safe_load(open('${PATCHED_CFG}'))['model']['sft_checkpoint'])
")
[[ -n "$RESOLVED_CHECKPOINT" && "$RESOLVED_CHECKPOINT" != "None" ]] || {
  echo "Error: model.sft_checkpoint not set — pass --sft-checkpoint PATH or set it in $DPO_CONFIG" >&2
  exit 1
}
[[ -d "$RESOLVED_CHECKPOINT" ]] || { echo "Error: SFT checkpoint not found: $RESOLVED_CHECKPOINT" >&2; exit 1; }

RESOLVED_METHOD=$(python3 -c "
import yaml
print(yaml.safe_load(open('${PATCHED_CFG}')).get('dpo', {}).get('method', 'dpo'))
")

# checkpoints/<model-basename>/checkpoint-N -> model-basename is the parent
# dir name of the checkpoint, matching training/dpo.py's own resolution
# (Path(sft_checkpoint).parent.name) exactly — no separate model-config
# lookup needed, unlike run_phase2_grpo.sh.
MODEL_BASENAME=$(basename "$(dirname "$RESOLVED_CHECKPOINT")")

# Read back the resolved run name from the patched config so CKPT_DIR below is
# exactly what dpo.py will use — the SFT/GRPO pair drifted apart once already.
RUN_NAME=$(python3 -c "
import yaml
print(yaml.safe_load(open('${PATCHED_CFG}'))['output_dir'])
")

# ── Banner ────────────────────────────────────────────────────────────────────
# ── Chunked mode: validate the plan before anything expensive ───────────────
if [[ "$CHUNK_STEPS" -gt 0 ]]; then
  read -r TOTAL_STEPS SAVE_STEPS N_PROMPTS MAX_NEW MAX_SEQ HELDOUT_DIR <<<"$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${PATCHED_CFG}'))
d, m, dat = cfg.get('dpo', {}), cfg.get('monitoring', {}), cfg.get('data', {})
print(d.get('training_steps', 500), d.get('save_steps', 100),
      m.get('eval_held_out_num_prompts', 50), d.get('max_completion_length', 512),
      d.get('max_seq_length', 8192),
      dat.get('heldout_data_source', 'data/output/grpo/task_a'))
")"

  if [[ "$CHUNK_STEPS" -ne "$SAVE_STEPS" ]]; then
    echo "Error: --chunk-steps ($CHUNK_STEPS) must equal dpo.save_steps ($SAVE_STEPS)." >&2
    echo "       Otherwise no checkpoint exists at the chunk boundary to score." >&2
    exit 1
  fi

fi

echo "=== Task A DPO/ORPO — Phase 2 ==="
echo "  DPO config     : $DPO_CONFIG"
echo "  Method         : $RESOLVED_METHOD"
echo "  Patched cfg    : $PATCHED_CFG"
echo "  SFT checkpoint : $RESOLVED_CHECKPOINT"
echo "  Pairs data dir : $PAIRS_DATA_DIR"
echo "  Checkpoint     : $PROJECT_ROOT/checkpoints/$RUN_NAME/$MODEL_BASENAME/"
echo "  W&B            : $([ "$NO_WANDB" -eq 1 ] && echo disabled || echo enabled)"
if [[ "$CHUNK_STEPS" -gt 0 ]]; then
  echo "  Chunked        : $TOTAL_STEPS steps in chunks of $CHUNK_STEPS"
  echo "  Guardrail      : between chunks, separate process, $N_PROMPTS prompts"
fi
echo "=================================="

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry-run: pairs prepared and config patched. Exiting without training."
  exit 0
fi

# ── Train ─────────────────────────────────────────────────────────────────────
CKPT_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME/$MODEL_BASENAME"
mkdir -p "$CKPT_DIR"
LOG_FILE="$CKPT_DIR/train.log"
echo "Logs: $LOG_FILE"

# Co-locate the frozen config with train.log so per-run provenance doesn't
# require knowing about .runs/ at all. Timestamped for the same reason as
# PATCHED_CFG above.
cp "$PATCHED_CFG" "$CKPT_DIR/frozen_dpo_config_${RUN_TS}.yaml"

# One training process. Called once for a straight-through run, or once per
# chunk in --chunk-steps mode. Each call is a *separate* process, which is the
# whole point in chunked mode: the GPU empties when it exits.
train_once() {
  python3 -c "
import sys
from pathlib import Path
from llm_workflow_agents.training.dpo import train_dpo

result = train_dpo(Path('$1'))
if result.error:
    print(f'ERROR: {result.error}', file=sys.stderr)
    sys.exit(1)
print(f'Method           : {result.method}')
print(f'Best eval loss   : {result.best_eval_loss}')
print(f'Total steps      : {result.total_steps}')
print(f'Early stopped    : {result.early_stopped}')
print(f'Held-out samples : {len(result.held_out_scores)}')
print(f'Checkpoint       : {result.checkpoint_path}')
" 2>&1 | tee -a "$LOG_FILE"
}

if [[ "$CHUNK_STEPS" -le 0 ]]; then
  train_once "$PATCHED_CFG"
  echo "Done. Checkpoint: $CKPT_DIR"
  exit 0
fi

AUDIT_DIR="$PROJECT_ROOT/runs/audit/${RUN_NAME}_guardrail"
mkdir -p "$AUDIT_DIR"
echo "Chunked mode: $TOTAL_STEPS steps in chunks of $CHUNK_STEPS; audits -> $AUDIT_DIR"

DONE_STEPS=0
while [[ $DONE_STEPS -lt $TOTAL_STEPS ]]; do
  TARGET=$(( DONE_STEPS + CHUNK_STEPS ))
  [[ $TARGET -gt $TOTAL_STEPS ]] && TARGET=$TOTAL_STEPS

  # Per-chunk config: cumulative max_steps, and the in-process guardrail off —
  # it is the thing that runs out of memory (R19).
  CHUNK_CFG="$PATCHED_DIR/${DPO_STEM}_${RUN_TS}_to${TARGET}.yaml"
  python3 -c "
import yaml
cfg = yaml.safe_load(open('${PATCHED_CFG}'))
cfg.setdefault('dpo', {})['training_steps'] = ${TARGET}
cfg.setdefault('monitoring', {})['reward_hacking_detector'] = False
with open('${CHUNK_CFG}', 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
"
  cp "$CHUNK_CFG" "$CKPT_DIR/frozen_dpo_config_${RUN_TS}_to${TARGET}.yaml"

  echo "=== chunk: training to step $TARGET of $TOTAL_STEPS ===" | tee -a "$LOG_FILE"
  train_once "$CHUNK_CFG"

  CKPT="$CKPT_DIR/checkpoint-${TARGET}"
  [[ -d "$CKPT" ]] || {
    echo "Error: expected checkpoint not found: $CKPT" >&2
    echo "       --chunk-steps must equal dpo.save_steps." >&2
    exit 1
  }

  echo "=== chunk: scoring $CKPT ===" | tee -a "$LOG_FILE"
  python3 scripts/heldout_composite_audit.py \
    --checkpoint "$CKPT" \
    --data-dir "$HELDOUT_DIR" \
    --split validation \
    --n-prompts "$N_PROMPTS" \
    --max-new-tokens "$MAX_NEW" \
    --max-seq-length "$MAX_SEQ" \
    --seed 42 \
    --output "$AUDIT_DIR/step-${TARGET}.json" 2>&1 | tee -a "$LOG_FILE"

  set +e
  python3 scripts/dpo_guardrail_decide.py \
    --trainer-state "$CKPT/trainer_state.json" \
    --audit-dir "$AUDIT_DIR" 2>&1 | tee -a "$LOG_FILE"
  DECISION=${PIPESTATUS[0]}
  set -e

  if [[ $DECISION -eq 10 ]]; then
    echo "Guardrail STOPPED the run at step $TARGET (reward hacking)." | tee -a "$LOG_FILE"
    break
  elif [[ $DECISION -ne 0 ]]; then
    echo "Error: guardrail decision failed with exit $DECISION" >&2
    exit 1
  fi

  DONE_STEPS=$TARGET
done

echo "Done. Checkpoint: $CKPT_DIR"
