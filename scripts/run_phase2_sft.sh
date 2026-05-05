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
  -h, --help            Show this help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --sft-config)   SFT_CONFIG="$2";   shift 2 ;;
    --dry-run)      DRY_RUN=1;         shift   ;;
    --no-wandb)     NO_WANDB=1;        shift   ;;
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
python3 -c "
import json, random, sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(open('${SFT_CONFIG}'))
data_dir = Path(cfg['data']['source'])
splits_dir = data_dir / 'splits'

if (splits_dir / 'train.jsonl').exists():
    counts = {s: sum(1 for _ in open(splits_dir / f'{s}.jsonl')) for s in ('train', 'validation', 'test')}
    print(f'Splits already exist: {counts}')
    sys.exit(0)

files = sorted(data_dir.glob('*.jsonl'))
rows = []
for f in files:
    with open(f) as fh:
        rows.extend(json.loads(line) for line in fh if line.strip())

random.Random(42).shuffle(rows)
n = len(rows)
ratios = cfg['data']['splits']
n_train = int(n * ratios['train'])
n_val   = int(n * ratios.get('val', ratios.get('validation', 0.10)))
n_test  = n - n_train - n_val

splits_dir.mkdir(parents=True, exist_ok=True)
for name, chunk in [('train',      rows[:n_train]),
                    ('validation', rows[n_train:n_train + n_val]),
                    ('test',       rows[n_train + n_val:])]:
    with open(splits_dir / f'{name}.jsonl', 'w') as fh:
        for r in chunk:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'Splits written to {splits_dir}')
print(f'  train={n_train}  validation={n_val}  test={n_test}  total={n}')
"

# ── Patch SFT config ──────────────────────────────────────────────────────────
PATCHED_DIR="$PROJECT_ROOT/.runs/sft_cat_a"
mkdir -p "$PATCHED_DIR"
PATCHED_CFG="$PATCHED_DIR/sft_cat_a.yaml"

python3 -c "
from pathlib import Path
import yaml

cfg = yaml.safe_load(open('${SFT_CONFIG}'))
cfg.setdefault('model', {})['config_path'] = str(Path('${MODEL_CONFIG}').resolve())
cfg['data']['source'] = str((Path(cfg['data']['source']) / 'splits').resolve())
if ${NO_WANDB}:
    cfg.setdefault('logging', {}).pop('wandb_project', None)

with open('${PATCHED_CFG}', 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "=== Task A SFT — Phase 2 ==="
echo "  Model config : $MODEL_CONFIG"
echo "  SFT config   : $SFT_CONFIG"
echo "  Patched cfg  : $PATCHED_CFG"
echo "  Checkpoint   : $PROJECT_ROOT/checkpoints/sft_cat_a/"
echo "  W&B          : $([ "$NO_WANDB" -eq 1 ] && echo disabled || echo enabled)"
echo "==========================="

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry-run: splits prepared and config patched. Exiting without training."
  exit 0
fi

# ── Train ─────────────────────────────────────────────────────────────────────
CKPT_DIR="$PROJECT_ROOT/checkpoints/sft_cat_a"
mkdir -p "$CKPT_DIR"
LOG_FILE="$CKPT_DIR/train.log"
echo "Logs: $LOG_FILE"

python3 -c "
import sys
from pathlib import Path
from llm_workflow_agents.training.sft import train_sft

result = train_sft(Path('${PATCHED_CFG}'))
if result.error:
    print(f'ERROR: {result.error}', file=sys.stderr)
    sys.exit(1)
print(f'Best eval loss : {result.best_eval_loss}')
print(f'Total steps    : {result.total_steps}')
print(f'Checkpoint     : {result.checkpoint_path}')
" 2>&1 | tee "$LOG_FILE"

echo "Done. Checkpoint: $CKPT_DIR"
