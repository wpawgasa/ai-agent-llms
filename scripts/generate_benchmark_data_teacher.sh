#!/usr/bin/env bash
# Generate benchmark evaluation data using a teacher model (GPT-4o / Gemini / Claude).
#
# Produces 200 samples per complexity level (1000 total) with mixed language.
# Requires the relevant API key in .env (auto-loaded via python-dotenv).
#
# Usage:
#   ./scripts/generate_benchmark_data_teacher.sh [OPTIONS]
#
# Options:
#   --teacher <model>     Teacher model name (default: gpt-4o)
#                         Supported prefixes: gpt-*, gemini-*, claude-*
#   --output-dir <path>   Base output directory (default: data/output)
#   --seed <n>            Random seed (default: 100)
#   --samples <n>         Samples per level (default: 200)
#   --levels <L1,L2,...>  Comma-separated levels (default: L1,L2,L3,L4,L5)
#   --language <lang>     Language: en, th, code_switch, or mixed (default: mixed)
#   --behavior <preset>   Behavior preset: default, adversarial, balanced (default: default)
#   --dry-run             Print commands without executing
#
# Examples:
#   ./scripts/generate_benchmark_data_teacher.sh
#   ./scripts/generate_benchmark_data_teacher.sh --teacher gemini-2.0-flash --samples 50
#   ./scripts/generate_benchmark_data_teacher.sh --teacher gpt-4o --levels L1,L2 --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEACHER="gpt-4o"
OUTPUT_DIR="$PROJECT_ROOT/data/output"
SEED=100
SAMPLES=200
LEVELS="L1,L2,L3,L4,L5"
LANGUAGE=""  # empty = mixed (en/th 50/50)
BEHAVIOR="default"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher)    TEACHER="$2";    shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --samples)    SAMPLES="$2";    shift 2 ;;
        --levels)     LEVELS="$2";     shift 2 ;;
        --language)   LANGUAGE="$2";   shift 2 ;;
        --behavior)   BEHAVIOR="$2";   shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

run() {
    if [[ "$DRY_RUN" = true ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

DEST="$OUTPUT_DIR/benchmark/task_a"
LANG_DISPLAY="${LANGUAGE:-mixed (en/th)}"
TOTAL_LEVELS=$(echo "$LEVELS" | tr ',' '\n' | wc -l | tr -d ' ')
TOTAL_SAMPLES=$((SAMPLES * TOTAL_LEVELS))

echo "=== Benchmark Data Generation (Teacher Model) ==="
echo "Output dir:   $DEST"
echo "Teacher:      $TEACHER"
echo "Seed:         $SEED"
echo "Levels:       $LEVELS ($SAMPLES samples each, $TOTAL_SAMPLES total)"
echo "Language:     $LANG_DISPLAY"
echo "Behavior:     $BEHAVIOR"
echo "=================================================="

# Build language arg for Python (None for mixed)
if [[ -z "$LANGUAGE" ]]; then
    LANG_ARG="None"
else
    LANG_ARG="'$LANGUAGE'"
fi

IFS=',' read -ra LEVEL_ARRAY <<< "$LEVELS"
for LEVEL in "${LEVEL_ARRAY[@]}"; do
    echo ""
    echo "  Generating $LEVEL ($SAMPLES samples via $TEACHER)..."
    run python3 -c "
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$SAMPLES,
    teacher_model='$TEACHER',
    output_dir=Path('$DEST'),
    seed=$SEED,
    language=$LANG_ARG,
    behavior_preset='$BEHAVIOR',
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"
done

echo ""
echo "=== Done. Benchmark data in $DEST ==="
