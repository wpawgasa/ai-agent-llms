#!/usr/bin/env bash
# Generate validation and held-out test datasets.
#
# Both splits: 500 samples total, 100 per complexity level (L1-L5).
# Mixed language, gpt-5.4-nano-2026-03-17 teacher, default behavior distribution.
# Different seeds ensure no overlap with SFT (42), benchmark (100), or GRPO (200).
#
#   Validation: seed=300  ->  data/output/val/task_a/
#   Test:       seed=400  ->  data/output/test/task_a/
#
# The test split must NOT be used during training or hyperparameter tuning.
#
# Required env vars: OPENAI_API_KEY
#
# Usage:
#   ./scripts/generate_eval_data.sh [OPTIONS]
#
# Options:
#   --split <val|test|both>   Which split to generate (default: both)
#   --output-dir <path>       Base output directory (default: data/output)
#   --dry-run                 Print commands without executing
#
# Examples:
#   OPENAI_API_KEY=sk-... ./scripts/generate_eval_data.sh
#   ./scripts/generate_eval_data.sh --split val --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output"
SPLIT="both"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --split)      SPLIT="$2";      shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$DRY_RUN" = false ]]; then
    [[ -z "${OPENAI_API_KEY:-}" ]] && { echo "Error: OPENAI_API_KEY is not set" >&2; exit 1; }
fi

run() {
    if [[ "$DRY_RUN" = true ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

generate_split() {
    local NAME="$1"
    local SEED="$2"
    local DEST="$OUTPUT_DIR/$NAME/task_a"

    echo ""
    echo "  --- $NAME split (seed=$SEED, 500 samples) ---"
    echo "  Output: $DEST"

    for LEVEL in L1 L2 L3 L4 L5; do
        echo "  [$NAME/$LEVEL] gpt-5.4-nano-2026-03-17 / mixed / 100 samples..."
        run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=100,
    teacher_model='gpt-5.4-nano-2026-03-17',
    output_dir=Path('$DEST'),
    seed=$SEED,
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"
    done
    echo "  $NAME complete (500 samples)."
}

echo "=== Eval Data Generation ==="
echo "Output base:  $OUTPUT_DIR"
echo "Split(s):     $SPLIT"
echo "Language:     mixed (en/th)"
echo "Model:        gpt-5.4-nano-2026-03-17"
echo "Distribution: default"
echo "Seeds:        val=300, test=400 (no overlap with sft=42, benchmark=100, grpo=200)"
echo "============================"

case "$SPLIT" in
    val)  generate_split "val"  300 ;;
    test) generate_split "test" 400 ;;
    both) generate_split "val"  300; generate_split "test" 400 ;;
    *)
        echo "Unknown split: $SPLIT  (expected val, test, or both)" >&2
        exit 1
        ;;
esac

echo ""
echo "=== Done. Eval data in $OUTPUT_DIR/{val,test}/task_a/ ==="
