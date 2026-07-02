#!/usr/bin/env bash
# Generate the Task C SFT dataset: natural-language playbooks -> WorkflowGraph JSON.
#
# One run produces ~5,000 (playbook, graph) pairs across three language legs
# (en/th/code_switch) and six registers per graph. See
# docs/data_generation_recipes_task_c.md for the recipe.
#
# Required env vars: the provider keys for the configured teachers. With the
# defaults (gpt-* en/code_switch, gemini-* th + invention) both OPENAI_API_KEY
# and GEMINI_API_KEY are required.
#
# Usage:
#   OPENAI_API_KEY=... GEMINI_API_KEY=... ./scripts/generate_playbook_sft_data.sh
#   ./scripts/generate_playbook_sft_data.sh --smoke-test --dry-run
#
# Options:
#   --output-dir <path>          Base output directory (default: data/output)
#   --seed <n>                   Random seed (default: 142)
#   --num-graphs <n>             Number of gold graphs (default: 850)
#   --smoke-test                 Shorthand for --num-graphs 6
#   --render-teacher-en <name>   English-leg render teacher
#   --render-teacher-th <name>   Thai-leg render teacher
#   --render-teacher-cs <name>   Code-switch-leg render teacher
#   --invention-teacher <name>   Novel-graph invention teacher
#   --dry-run                    Print the invocation without executing
#
# Exits nonzero if any language leg is halted by back-extraction failures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output"
SEED=142
NUM_GRAPHS=850
DRY_RUN=false
T_EN="gpt-5.4-mini-2026-03-17"
T_TH="gemini-3-flash-preview"
T_CS="gpt-5.4-nano-2026-03-17"
T_INV="gpt-5.4-mini-2026-03-17"

# Load .env if present (mirrors python-dotenv behaviour in _teacher_client.py)
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)         OUTPUT_DIR="$2";  shift 2 ;;
        --seed)               SEED="$2";        shift 2 ;;
        --num-graphs)         NUM_GRAPHS="$2";  shift 2 ;;
        --smoke-test)         NUM_GRAPHS=6;     shift ;;
        --render-teacher-en)  T_EN="$2";        shift 2 ;;
        --render-teacher-th)  T_TH="$2";        shift 2 ;;
        --render-teacher-cs)  T_CS="$2";        shift 2 ;;
        --invention-teacher)  T_INV="$2";       shift 2 ;;
        --dry-run)            DRY_RUN=true;     shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Map a teacher model to its required env-var key (mirrors _teacher_client.py routing).
required_key_for() {
    case "$1" in
        gemini*) echo "GEMINI_API_KEY" ;;
        gpt*)    echo "OPENAI_API_KEY" ;;
        claude*) echo "ANTHROPIC_API_KEY" ;;
        *) echo "UNSUPPORTED:$1" ;;
    esac
}

if [[ "$DRY_RUN" = false ]]; then
    declare -A NEEDED=()
    for model in "$T_EN" "$T_TH" "$T_CS" "$T_INV"; do
        key="$(required_key_for "$model")"
        [[ "$key" == UNSUPPORTED:* ]] && { echo "Unsupported teacher model: $model" >&2; exit 1; }
        NEEDED["$key"]=1
    done
    for key in "${!NEEDED[@]}"; do
        [[ -z "${!key:-}" ]] && { echo "Error: $key is not set (required for the configured teachers)" >&2; exit 1; }
    done
fi

run() {
    if [[ "$DRY_RUN" = true ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

DEST="$OUTPUT_DIR/sft/task_c"

run python3 -c "
import sys
from pathlib import Path
from llm_workflow_agents.data.generate_playbook_pairs import generate_playbook_dataset
stats = generate_playbook_dataset(
    num_graphs=$NUM_GRAPHS,
    seed=$SEED,
    output_dir=Path('$DEST'),
    render_teachers={'en': '$T_EN', 'th': '$T_TH', 'code_switch': '$T_CS'},
    invention_teacher='$T_INV',
)
print(f'  -> accepted={stats.renderings_accepted} files={len(stats.output_files)} halted={stats.halted_legs}')
if stats.halted_legs:
    print(f'ERROR: legs halted on back-extraction failures: {stats.halted_legs}', file=sys.stderr)
    sys.exit(1)
"
