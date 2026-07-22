#!/usr/bin/env bash
# Run Unsloth SFT (Phase 2) for Task A.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_CONFIG="configs/models_exp_a/gemma4_26b_a4b.yaml"
SFT_CONFIG="configs/training/sft_cat_a.yaml"
DRY_RUN=0
NO_WANDB=0
RESUME=0
RESUME_FROM=""

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Phase 2 SFT runner for Task A (Gemma4-26B-A4B FP8 default).

Options:
  --model-config PATH   Cat A model YAML  (default: $MODEL_CONFIG)
  --sft-config PATH     SFT training YAML (default: $SFT_CONFIG)
  --dry-run             Prepare splits + patched config, exit before training
  --no-wandb            Disable W&B logging (overrides YAML)
  --resume              Resume from the latest checkpoint under
                        checkpoints/<sft-config-stem>/checkpoint-*
  --resume-from PATH    Resume from a specific checkpoint directory
  -h, --help            Show this help

Pause/resume:
  - Stop a running job with Ctrl+C; checkpoints saved every save_steps remain
    on disk under checkpoints/<sft-config-stem>/checkpoint-N.
  - Re-launch with --resume (auto-pick latest) or --resume-from PATH to
    continue from where it stopped. Optimizer state, scheduler, RNG, and
    epoch counter are restored.
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --sft-config)   SFT_CONFIG="$2";   shift 2 ;;
    --dry-run)      DRY_RUN=1;         shift   ;;
    --no-wandb)     NO_WANDB=1;        shift   ;;
    --resume)       RESUME=1;          shift   ;;
    --resume-from)  RESUME_FROM="$2";  shift 2 ;;
    -h|--help)      usage              ;;
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
[[ -f "$SFT_CONFIG"   ]] || { echo "Error: SFT config not found: $SFT_CONFIG"     >&2; exit 1; }
[[ "${HF_TOKEN:-}" != "" ]] || echo "Warning: HF_TOKEN not set — gated models will fail to download." >&2

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
fi

# ── Prepare data splits ───────────────────────────────────────────────────────
# Splits are produced by scripts/split_task_a_sft.py (DVC stage: task_a_sft_splits).
# Calling it here keeps the SFT runner self-contained for users who run outside DVC;
# the splitter is idempotent and a no-op if the splits already exist.
python3 scripts/split_task_a_sft.py

# ── Patch SFT config ──────────────────────────────────────────────────────────
# The SFT config's data.source already points at the splits dir
# (data/output/sft/task_a_splits/). The only patch here is wiring the model
# config path through so sft.py can resolve the base model.
#
# RUN_TS makes the patched config run-specific: a fixed filename here would be
# silently overwritten by the next invocation (even --dry-run), leaving no
# reliable record of what config actually produced a given checkpoint. See
# CLAUDE.md R13.
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PATCHED_DIR="$PROJECT_ROOT/.runs/sft_cat_a"
mkdir -p "$PATCHED_DIR"
PATCHED_CFG="$PATCHED_DIR/sft_cat_a_${RUN_TS}.yaml"

python3 -c "
from pathlib import Path
import yaml

cfg = yaml.safe_load(open('${SFT_CONFIG}'))
cfg.setdefault('model', {})['config_path'] = str(Path('${MODEL_CONFIG}').resolve())
cfg['data']['source'] = str(Path(cfg['data']['source']).resolve())
# Pin the checkpoint directory to the *base* config stem, not the timestamped
# patched-config stem. sft.py would otherwise derive it from the patched
# filename and write outside the DVC-tracked path. Set output_dir in the base
# config to give a run its own directory (e.g. one per factorial cell).
cfg.setdefault('output_dir', Path('${SFT_CONFIG}').stem)
if ${NO_WANDB}:
    cfg.setdefault('logging', {}).pop('wandb_project', None)

with open('${PATCHED_CFG}', 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
"

# Resolve the HF model basename from the model YAML (matches the
# `Path(model_name).name` segment that sft.py appends to output_dir).
MODEL_BASENAME=$(python3 -c "
import yaml, pathlib
print(pathlib.PurePosixPath(yaml.safe_load(open('${MODEL_CONFIG}'))['model']['name']).name)
")

# Read back the resolved run name from the patched config so CKPT_DIR below is
# exactly what sft.py will use — these two drifted apart once already.
RUN_NAME=$(python3 -c "
import yaml
print(yaml.safe_load(open('${PATCHED_CFG}'))['output_dir'])
")

# ── Banner ────────────────────────────────────────────────────────────────────
echo "=== Task A SFT — Phase 2 ==="
echo "  Model config : $MODEL_CONFIG"
echo "  SFT config   : $SFT_CONFIG"
echo "  Patched cfg  : $PATCHED_CFG"
echo "  Checkpoint   : $PROJECT_ROOT/checkpoints/$RUN_NAME/$MODEL_BASENAME/"
echo "  W&B          : $([ "$NO_WANDB" -eq 1 ] && echo disabled || echo enabled)"
if [[ -n "$RESUME_FROM" ]]; then
  echo "  Resume       : $RESUME_FROM"
elif [[ "$RESUME" -eq 1 ]]; then
  echo "  Resume       : auto (latest checkpoint)"
fi
echo "==========================="

# Validate resume target now so we fail fast.
if [[ -n "$RESUME_FROM" && ! -d "$RESUME_FROM" ]]; then
  echo "Error: --resume-from path does not exist: $RESUME_FROM" >&2
  exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry-run: splits prepared and config patched. Exiting without training."
  exit 0
fi

# ── Train ─────────────────────────────────────────────────────────────────────
CKPT_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME/$MODEL_BASENAME"
mkdir -p "$CKPT_DIR"
LOG_FILE="$CKPT_DIR/train.log"
echo "Logs: $LOG_FILE"

# Co-locate a copy of the exact resolved config with the checkpoint dir itself
# (alongside train.log), so provenance for this run doesn't require knowing
# about .runs/ at all. Timestamped for the same reason as PATCHED_CFG above.
cp "$PATCHED_CFG" "$CKPT_DIR/frozen_sft_config_${RUN_TS}.yaml"

RESUME_FROM="$RESUME_FROM" RESUME="$RESUME" python3 -c "
import os
import sys
from pathlib import Path
from llm_workflow_agents.training.sft import train_sft

resume_from = os.environ.get('RESUME_FROM') or None
resume_flag = os.environ.get('RESUME', '0') == '1'
resume_arg = resume_from if resume_from else (True if resume_flag else None)

result = train_sft(Path('${PATCHED_CFG}'), resume_from_checkpoint=resume_arg)
if result.error:
    print(f'ERROR: {result.error}', file=sys.stderr)
    sys.exit(1)
print(f'Best eval loss : {result.best_eval_loss}')
print(f'Total steps    : {result.total_steps}')
print(f'Checkpoint     : {result.checkpoint_path}')
" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Checkpoint: $CKPT_DIR"
