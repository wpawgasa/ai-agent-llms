#!/usr/bin/env bash
# Generate GRPO (Group Relative Policy Optimization) training prompts.
#
# Covers L3-L5 only (750 samples each = 2250 total). These are the harder
# complexity levels where reward-based RL delivers the most improvement over SFT.
# Three teacher/language runs per level (250 each): gpt-5.4-nano-2026-03-17/mixed, claude/mixed,
# gpt-5.4-nano-2026-03-17/code_switch. Balanced behavior distribution ensures the policy is
# challenged across all behavior types and language registers.
#
# Balanced preset: all 4 user behaviors at 25% each.
#
# Required env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY
#
# Usage:
#   ./scripts/generate_grpo_data.sh [OPTIONS]
#
# Options:
#   --output-dir <path>   Base output directory (default: data/output)
#   --seed <n>            Random seed (default: 200)
#   --dry-run             Print commands without executing
#
# Examples:
#   OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-... ./scripts/generate_grpo_data.sh
#   ./scripts/generate_grpo_data.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output"
SEED=200
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

DEST="$OUTPUT_DIR/grpo/task_a"

echo "=== GRPO Data Generation ==="
echo "Output dir:   $DEST"
echo "Seed:         $SEED"
echo "Levels:       L3, L4, L5 — 750 samples each (2250 total)"
echo "Distribution: balanced (all 4 behaviors at 25% each)"
echo "Teachers:     gpt-5.4-nano-2026-03-17/mixed (250) + gemini-3-flash-preview/mixed (250) + gpt-5.4-nano-2026-03-17/code_switch (250) per level"
echo "============================"

for LEVEL in L3 L4 L5; do
    echo ""
    echo "  --- $LEVEL (750 samples) ---"

    echo "  [$LEVEL] gpt-5.4-nano-2026-03-17 / mixed / 250 samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=250,
    teacher_model='gpt-5.4-nano-2026-03-17',
    output_dir=Path('$DEST'),
    seed=$SEED,
    behavior_preset='balanced',
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"

    echo "  [$LEVEL] gemini-3-flash-preview / mixed / 250 samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=250,
    teacher_model='gemini-3-flash-preview',
    output_dir=Path('$DEST'),
    seed=$SEED,
    behavior_preset='balanced',
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"

    echo "  [$LEVEL] gpt-5.4-nano-2026-03-17 / code-switching / 250 samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=250,
    teacher_model='gpt-5.4-nano-2026-03-17',
    output_dir=Path('$DEST'),
    seed=$SEED,
    language='code_switch',
    behavior_preset='balanced',
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"
done

echo ""
echo "=== Done. GRPO data in $DEST (2250 samples) ==="
