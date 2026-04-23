#!/usr/bin/env bash
# Generate supervised fine-tuning (SFT) training data.
#
# Curriculum-weighted: L1=3000, L2=3000, L3=2502, L4=2001, L5=2001 (~12504 total).
# Each level is split evenly across three teacher/language runs:
#   - 1/3: gpt-5.4-mini-2026-03-17           / English        / adversarial preset
#   - 1/3: gemini-3-flash-preview / Thai           / adversarial preset
#   - 1/3: gpt-5.4-nano-2026-03-17           / code-switching  / adversarial preset
#
# Adversarial preset: cooperative=0.45, adversarial_probing=0.25,
#                     digressing=0.15, invalid_tool_inputs=0.15
#
# Required env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY
#
# Usage:
#   ./scripts/generate_sft_data.sh [OPTIONS]
#
# Options:
#   --output-dir <path>   Base output directory (default: data/output)
#   --seed <n>            Random seed (default: 42)
#   --samples-per-leg <n> Override per-leg sample count for all levels
#   --smoke-test          Shorthand for --samples-per-leg 3 (quick pipeline check)
#   --dry-run             Print commands without executing
#
# Examples:
#   OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-... ./scripts/generate_sft_data.sh
#   ./scripts/generate_sft_data.sh --smoke-test --dry-run
#   ./scripts/generate_sft_data.sh --samples-per-leg 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output"
SEED=42
DRY_RUN=false
SAMPLES_PER_LEG=""  # empty = use curriculum defaults

# Load .env if present (mirrors python-dotenv behaviour in _teacher_client.py)
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --seed)            SEED="$2";             shift 2 ;;
        --samples-per-leg) SAMPLES_PER_LEG="$2"; shift 2 ;;
        --smoke-test)      SAMPLES_PER_LEG=3;    shift ;;
        --dry-run)         DRY_RUN=true;          shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$DRY_RUN" = false ]]; then
    [[ -z "${OPENAI_API_KEY:-}" ]]     && { echo "Error: OPENAI_API_KEY is not set"     >&2; exit 1; }
    [[ -z "${ANTHROPIC_API_KEY:-}" ]] && { echo "Error: ANTHROPIC_API_KEY is not set" >&2; exit 1; }
fi

run() {
    if [[ "$DRY_RUN" = true ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

DEST="$OUTPUT_DIR/sft/task_a"

# Samples per level per leg (1/3 of total, rounded to clean integers)
declare -A CURRICULUM=([L1]=1000 [L2]=1000 [L3]=834 [L4]=667 [L5]=667)

# Apply --samples-per-leg / --smoke-test override
declare -A THIRD
for LEVEL in L1 L2 L3 L4 L5; do
    THIRD[$LEVEL]="${SAMPLES_PER_LEG:-${CURRICULUM[$LEVEL]}}"
done

if [[ -n "$SAMPLES_PER_LEG" ]]; then
    TOTAL_ALL=$(( SAMPLES_PER_LEG * 3 * 5 ))
    TOTALS_MSG="all levels: ${SAMPLES_PER_LEG} per leg × 3 legs × 5 levels = ${TOTAL_ALL} total (override)"
else
    TOTALS_MSG="L1=3000, L2=3000, L3=2502, L4=2001, L5=2001  (~12504 total)"
fi

echo "=== SFT Data Generation ==="
echo "Output dir:   $DEST"
echo "Seed:         $SEED"
echo "Distribution: adversarial (cooperative=0.45, adversarial_probing=0.25, digressing=0.15, invalid_tool_inputs=0.15)"
echo "Totals:       $TOTALS_MSG"
echo "Split:        1/3 gpt-5.4-mini-2026-03-17/en  +  1/3 gemini-3-flash-preview/th  +  1/3 gpt-5.4-nano-2026-03-17/code_switch per level"
echo "==========================="

for LEVEL in L1 L2 L3 L4 L5; do
    T="${THIRD[$LEVEL]}"
    TOTAL=$(( T * 3 ))
    echo ""
    echo "  --- $LEVEL ($TOTAL samples) ---"

    echo "  [$LEVEL] gpt-5.4-mini-2026-03-17 / English / $T samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$T,
    teacher_model='gpt-5.4-mini-2026-03-17',
    output_dir=Path('$DEST'),
    seed=$SEED,
    language='en',
    behavior_preset='adversarial',
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"

    echo "  [$LEVEL] gemini-3-flash-preview / Thai / $T samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$T,
    teacher_model='gemini-3-flash-preview',
    output_dir=Path('$DEST'),
    seed=$SEED,
    language='th',
    behavior_preset='adversarial',
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"

    echo "  [$LEVEL] gpt-5.4-nano-2026-03-17 / code-switching / $T samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$T,
    teacher_model='gpt-5.4-nano-2026-03-17',
    output_dir=Path('$DEST'),
    seed=$SEED,
    language='code_switch',
    behavior_preset='adversarial',
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"
done

echo ""
echo "=== Done. SFT data in $DEST ($TOTALS_MSG) ==="
