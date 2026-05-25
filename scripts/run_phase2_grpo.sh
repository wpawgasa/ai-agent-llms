#!/usr/bin/env bash
# Run Unsloth GRPO RL (Phase 2) for Task A.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_CONFIG="configs/models_exp_a/gemma4_26b_a4b.yaml"
GRPO_CONFIG="configs/training/grpo_cat_a.yaml"
SFT_CHECKPOINT=""
GRPO_DATA_DIR="data/output/grpo/task_a"
GRPO_LEVELS=("L3" "L4" "L5")
DRY_RUN=0
NO_WANDB=0
SKIP_FILTER=0

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Phase 2 GRPO RL runner for Task A (Gemma4-26B-A4B default).

GRPO consumes an SFT checkpoint (Phase 2 SFT output) plus a filtered prompt
set (default: L3-L5 from the cleaned SFT corpus). Rewards are recomputed
online from policy generations — no new ground-truth labels are needed.

Options:
  --model-config PATH     Cat A model YAML (banner only — base model is
                          resolved from the SFT checkpoint)
                          (default: $MODEL_CONFIG)
  --grpo-config PATH      GRPO training YAML (default: $GRPO_CONFIG)
  --sft-checkpoint PATH   SFT checkpoint to seed RL from. If omitted, the
                          latest checkpoint-* under
                          checkpoints/sft_cat_a/<model-basename>/ is used.
  --data-dir PATH         GRPO prompt set directory
                          (default: $GRPO_DATA_DIR)
  --levels L1 L2 ...      Complexity levels for filter_grpo_data.py
                          (default: ${GRPO_LEVELS[*]})
  --skip-filter           Skip filter_grpo_data.py (assume --data-dir is
                          already populated)
  --dry-run               Prepare prompts + patched config, exit before
                          training
  --no-wandb              Disable W&B logging (overrides YAML)
  -h, --help              Show this help

Notes:
  - GRPO has no resume hook in training/grpo.py today, so this runner does
    not expose --resume. Re-launching starts a fresh trainer; intermediate
    checkpoints under checkpoints/<grpo-config-stem>/ are not auto-loaded.
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-config)    MODEL_CONFIG="$2";    shift 2 ;;
    --grpo-config)     GRPO_CONFIG="$2";     shift 2 ;;
    --sft-checkpoint)  SFT_CHECKPOINT="$2";  shift 2 ;;
    --data-dir)        GRPO_DATA_DIR="$2";   shift 2 ;;
    --levels)
      shift
      GRPO_LEVELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        GRPO_LEVELS+=("$1"); shift
      done
      ;;
    --skip-filter)     SKIP_FILTER=1;        shift   ;;
    --dry-run)         DRY_RUN=1;            shift   ;;
    --no-wandb)        NO_WANDB=1;           shift   ;;
    -h|--help)         usage                 ;;
    *) echo "Error: unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Environment ───────────────────────────────────────────────────────────────
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
[[ -f "$MODEL_CONFIG" ]] || { echo "Error: model config not found: $MODEL_CONFIG" >&2; exit 1; }
[[ -f "$GRPO_CONFIG"  ]] || { echo "Error: GRPO config not found: $GRPO_CONFIG"   >&2; exit 1; }
[[ "${HF_TOKEN:-}" != "" ]] || echo "Warning: HF_TOKEN not set — gated models will fail to download." >&2

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
fi

# ── Resolve SFT checkpoint ────────────────────────────────────────────────────
# Mirrors the path layout that sft.py writes:
#   checkpoints/sft_cat_a/<HF-model-basename>/checkpoint-N
MODEL_BASENAME=$(python3 -c "
import yaml, pathlib
print(pathlib.PurePosixPath(yaml.safe_load(open('${MODEL_CONFIG}'))['model']['name']).name)
")

if [[ -z "$SFT_CHECKPOINT" ]]; then
  SFT_BASE_DIR="$PROJECT_ROOT/checkpoints/sft_cat_a/$MODEL_BASENAME"
  if [[ ! -d "$SFT_BASE_DIR" ]]; then
    echo "Error: no SFT checkpoint dir at $SFT_BASE_DIR" >&2
    echo "       Run ./scripts/run_phase2_sft.sh first, or pass --sft-checkpoint PATH." >&2
    exit 1
  fi
  SFT_CHECKPOINT=$(ls -d "$SFT_BASE_DIR"/checkpoint-* 2>/dev/null \
    | sort -t- -k2 -n | tail -1)
  if [[ -z "$SFT_CHECKPOINT" || ! -d "$SFT_CHECKPOINT" ]]; then
    echo "Error: no checkpoint-* directories found under $SFT_BASE_DIR" >&2
    exit 1
  fi
fi
[[ -d "$SFT_CHECKPOINT" ]] || { echo "Error: SFT checkpoint not found: $SFT_CHECKPOINT" >&2; exit 1; }

# ── Prepare GRPO prompt set ───────────────────────────────────────────────────
# filter_grpo_data.py reads the deterministic SFT splits and writes a filtered
# train/validation pair. It's idempotent — safe to call every run. The 'test'
# split is intentionally reserved for final eval (not produced here).
if [[ $SKIP_FILTER -eq 0 ]]; then
  python3 scripts/filter_grpo_data.py \
    --output-dir "$GRPO_DATA_DIR" \
    --levels "${GRPO_LEVELS[@]}"
fi

[[ -d "$GRPO_DATA_DIR" ]] || { echo "Error: GRPO data dir missing: $GRPO_DATA_DIR" >&2; exit 1; }

# ── Patch GRPO config ─────────────────────────────────────────────────────────
# Inject the resolved SFT checkpoint + data dir so grpo.py picks them up.
# Preserve the original config's stem in the patched file path — grpo.py derives
# the checkpoint dir from Path(config_path).stem, so renaming here would route
# every run into checkpoints/grpo_cat_a/ regardless of --grpo-config.
GRPO_STEM=$(basename "${GRPO_CONFIG%.*}")
PATCHED_DIR="$PROJECT_ROOT/.runs/$GRPO_STEM"
mkdir -p "$PATCHED_DIR"
PATCHED_CFG="$PATCHED_DIR/${GRPO_STEM}.yaml"

python3 -c "
from pathlib import Path
import yaml

cfg = yaml.safe_load(open('${GRPO_CONFIG}'))
cfg.setdefault('model', {})['sft_checkpoint'] = str(Path('${SFT_CHECKPOINT}').resolve())
cfg.setdefault('model', {})['config_path']    = str(Path('${MODEL_CONFIG}').resolve())
cfg.setdefault('data', {})['source']          = str(Path('${GRPO_DATA_DIR}').resolve())
if ${NO_WANDB}:
    cfg.setdefault('logging', {}).pop('wandb_project', None)

with open('${PATCHED_CFG}', 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "=== Task A GRPO — Phase 2 ==="
echo "  Model config   : $MODEL_CONFIG"
echo "  GRPO config    : $GRPO_CONFIG"
echo "  Patched cfg    : $PATCHED_CFG"
echo "  SFT checkpoint : $SFT_CHECKPOINT"
echo "  Data dir       : $GRPO_DATA_DIR (levels: ${GRPO_LEVELS[*]})"
echo "  Checkpoint     : $PROJECT_ROOT/checkpoints/$GRPO_STEM/$MODEL_BASENAME/"
echo "  W&B            : $([ "$NO_WANDB" -eq 1 ] && echo disabled || echo enabled)"
echo "============================="

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry-run: prompts prepared and config patched. Exiting without training."
  exit 0
fi

# ── Train ─────────────────────────────────────────────────────────────────────
CKPT_DIR="$PROJECT_ROOT/checkpoints/$GRPO_STEM/$MODEL_BASENAME"
mkdir -p "$CKPT_DIR"
LOG_FILE="$CKPT_DIR/train.log"
echo "Logs: $LOG_FILE"

python3 -c "
import sys
from pathlib import Path
from llm_workflow_agents.training.grpo import train_grpo

result = train_grpo(Path('${PATCHED_CFG}'))
if result.error:
    print(f'ERROR: {result.error}', file=sys.stderr)
    sys.exit(1)
print(f'Total steps      : {result.total_steps}')
print(f'Early stopped    : {result.early_stopped}')
print(f'Reward samples   : {len(result.reward_curves)}')
print(f'Held-out samples : {len(result.held_out_scores)}')
print(f'Checkpoint       : {result.checkpoint_path}')
" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Checkpoint: $CKPT_DIR"
