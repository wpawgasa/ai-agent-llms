#!/usr/bin/env bash
# Generate benchmark evaluation data for Phase 1 pre-trained model ranking.
#
# Produces 200 samples per complexity level (1000 total) with mixed language
# using the placeholder generator. No teacher model or API key required.
#
# Usage:
#   ./scripts/generate_benchmark_data.sh [OPTIONS]
#
# Options:
#   --output-dir <path>   Base output directory (default: data/output)
#   --seed <n>            Random seed (default: 100)
#   --dry-run             Print commands without executing
#
# Examples:
#   ./scripts/generate_benchmark_data.sh
#   ./scripts/generate_benchmark_data.sh --output-dir /mnt/data/output --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output"
SEED=100
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
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

echo "=== Benchmark Data Generation ==="
echo "Output dir:   $DEST"
echo "Seed:         $SEED"
echo "Levels:       L1-L5, 200 samples each (1000 total)"
echo "Language:     mixed (en/th)"
echo "Model:        placeholder (no API key needed)"
echo "Distribution: default"
echo "================================="

for LEVEL in L1 L2 L3 L4 L5; do
    echo ""
    echo "  Generating $LEVEL (200 samples)..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=200,
    output_dir=Path('$DEST'),
    seed=$SEED,
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"
done

echo ""
echo "=== Done. Benchmark data in $DEST ==="
