#!/usr/bin/env bash
# Generate synthetic training data for Tasks A, B, and C.
#
# Usage:
#   ./scripts/generate_data.sh [OPTIONS]
#
# Options:
#   --task <a|b|c|all>       Which task(s) to generate (default: all)
#   --levels <L1,L2,...>     Comma-separated complexity levels for Task A (default: L1,L2,L3,L4,L5)
#   --samples <n>            Samples per complexity level for Task A (default: 200)
#   --synthetic-size <n>     Synthetic samples for Task B (default: 15000)
#   --teacher-model <name>   Teacher model for live API generation, e.g. gemini-2.0-flash,
#                            gpt-4o, claude-sonnet-4-6. Omit to use placeholder generator.
#   --output-dir <path>      Base output directory (default: data/output)
#   --seed <n>               Random seed (default: 42)
#   --language <en|th>       Conversation language for Task A (default: mixed 50/50)
#   --dry-run                Print commands without executing
#
# Examples:
#   # Placeholder generation (no API key needed)
#   ./scripts/generate_data.sh
#
#   # Task A only, L1+L2, 50 samples each
#   ./scripts/generate_data.sh --task a --levels L1,L2 --samples 50
#
#   # All tasks with Gemini teacher model, Thai only
#   ./scripts/generate_data.sh --teacher-model gemini-2.0-flash --language th

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
TASK="all"
LEVELS="L1,L2,L3,L4,L5"
SAMPLES=200
SYNTHETIC_SIZE=15000
TEACHER_MODEL=""
OUTPUT_DIR="$PROJECT_ROOT/data/output"
SEED=42
LANGUAGE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)           TASK="$2";           shift 2 ;;
        --levels)         LEVELS="$2";         shift 2 ;;
        --samples)        SAMPLES="$2";        shift 2 ;;
        --synthetic-size) SYNTHETIC_SIZE="$2"; shift 2 ;;
        --teacher-model)  TEACHER_MODEL="$2";  shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --language)       LANGUAGE="$2";       shift 2 ;;
        --dry-run)        DRY_RUN=true;        shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

TEACHER_ARG=""
if [[ -n "$TEACHER_MODEL" ]]; then
    TEACHER_ARG="teacher_model='$TEACHER_MODEL',"
fi

LANGUAGE_ARG=""
if [[ -n "$LANGUAGE" ]]; then
    LANGUAGE_ARG="language='$LANGUAGE',"
fi

run() {
    if [[ "$DRY_RUN" = true ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

echo "=== Data Generation ==="
echo "Task(s):       $TASK"
echo "Output dir:    $OUTPUT_DIR"
echo "Teacher model: ${TEACHER_MODEL:-placeholder}"
echo "Language:      ${LANGUAGE:-mixed (en/th)}"
echo "Seed:          $SEED"
echo "======================="

# ── Task A ──────────────────────────────────────────────────────────────────
generate_task_a() {
    echo ""
    echo "--- Task A: Workflow Conversations ---"
    IFS=',' read -ra LEVEL_LIST <<< "$LEVELS"
    for LEVEL in "${LEVEL_LIST[@]}"; do
        echo "  Generating $LEVEL ($SAMPLES samples)..."
        run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$SAMPLES,
    ${TEACHER_ARG}
    output_dir=Path('$OUTPUT_DIR/task_a'),
    seed=$SEED,
    ${LANGUAGE_ARG}
)
print(f'  -> {meta.output_files[0]}  ({meta.num_samples} samples)')
"
    done
    echo "  Task A complete."
}

# ── Task B ──────────────────────────────────────────────────────────────────
generate_task_b() {
    echo ""
    echo "--- Task B: Tool-Call Dataset ---"
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_tool_call_data import generate_tool_call_dataset
splits = generate_tool_call_dataset(
    custom_synthetic_size=$SYNTHETIC_SIZE,
    ${TEACHER_ARG}
    output_dir=Path('$OUTPUT_DIR/task_b'),
    seed=$SEED,
)
total = splits.train_size + splits.val_size + splits.test_size
print(f'  -> train={splits.train_size}  val={splits.val_size}  test={splits.test_size}  total={total}')
"
    echo "  Task B complete."
}

# ── Task C ──────────────────────────────────────────────────────────────────
generate_task_c() {
    echo ""
    echo "--- Task C: Graph Pairs ---"
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_graph_pairs import generate_graph_pairs
splits = generate_graph_pairs(
    workflow_prompts_dir=Path('$OUTPUT_DIR/task_a'),
    ${TEACHER_ARG}
    output_dir=Path('$OUTPUT_DIR/task_c'),
    seed=$SEED,
)
total = splits.train_size + splits.val_size + splits.test_size
print(f'  -> train={splits.train_size}  val={splits.val_size}  test={splits.test_size}  total={total}')
"
    echo "  Task C complete."
}

case "$TASK" in
    a)   generate_task_a ;;
    b)   generate_task_b ;;
    c)   generate_task_a; generate_task_c ;;  # C depends on A output
    all) generate_task_a; generate_task_b; generate_task_c ;;
    *)
        echo "Unknown task: $TASK  (expected a, b, c, or all)" >&2
        exit 1
        ;;
esac

echo ""
echo "=== Done. Output in $OUTPUT_DIR ==="
