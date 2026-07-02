#!/usr/bin/env bash
# Generate (or merge) the Task C benchmark split: ~150 playbook->graph pairs for
# ranking pre-trained 4B-12B candidates. See docs/data_generation_recipes_task_c.md.
#
# Two Gemini teacher runs are produced then merged. Merge policy: union by
# pair_id, first-listed run wins (the second teacher fills renderings the first
# dropped).
#
# Usage:
#   GEMINI_API_KEY=... ./scripts/generate_playbook_benchmark_data.sh --teacher gemini-3-flash
#   GEMINI_API_KEY=... ./scripts/generate_playbook_benchmark_data.sh --teacher gemini-3.1-flash-lite
#   ./scripts/generate_playbook_benchmark_data.sh --merge
#
# Options:
#   --teacher <name>     Teacher model for all legs (required in generate mode)
#   --seed <n>           Random seed (default: 200)
#   --num-graphs <n>     Number of gold graphs (default: 25)
#   --output-dir <path>  Output directory (default: data/output/benchmark/task_c)
#   --merge              Merge existing run files into playbook_pairs_gemini-3_merged.jsonl
#   --dry-run            Print the invocation without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output/benchmark/task_c"
SEED=200
NUM_GRAPHS=25
TEACHER=""
MERGE=false
DRY_RUN=false

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher)     TEACHER="$2";    shift 2 ;;
        --seed)        SEED="$2";       shift 2 ;;
        --num-graphs)  NUM_GRAPHS="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --merge)       MERGE=true;      shift ;;
        --dry-run)     DRY_RUN=true;    shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

run() {
    if [[ "$DRY_RUN" = true ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

if [[ "$MERGE" = true ]]; then
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_playbook_pairs import merge_benchmark_runs
out_dir = Path('$OUTPUT_DIR')
runs = sorted(p for p in out_dir.glob('playbook_pairs_*.jsonl') if not p.name.endswith('_merged.jsonl'))
if not runs:
    raise SystemExit('No playbook_pairs_*.jsonl run files found to merge in $OUTPUT_DIR')
count = merge_benchmark_runs(runs, out_dir / 'playbook_pairs_gemini-3_merged.jsonl')
print(f'  -> merged {count} pairs from {len(runs)} run(s)')
"
    exit 0
fi

if [[ -z "$TEACHER" ]]; then
    echo "Error: --teacher is required in generate mode" >&2
    exit 1
fi

case "$TEACHER" in
    gemini*) REQUIRED_KEY="GEMINI_API_KEY" ;;
    gpt*)    REQUIRED_KEY="OPENAI_API_KEY" ;;
    claude*) REQUIRED_KEY="ANTHROPIC_API_KEY" ;;
    *) echo "Unsupported --teacher: $TEACHER (expected prefix gemini-*, gpt-*, or claude-*)" >&2; exit 1 ;;
esac

if [[ "$DRY_RUN" = false ]]; then
    [[ -z "${!REQUIRED_KEY:-}" ]] && { echo "Error: $REQUIRED_KEY is not set (required for teacher $TEACHER)" >&2; exit 1; }
fi

run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_playbook_pairs import generate_playbook_dataset
stats = generate_playbook_dataset(
    num_graphs=$NUM_GRAPHS,
    seed=$SEED,
    output_dir=Path('$OUTPUT_DIR'),
    render_teachers={'en': '$TEACHER', 'th': '$TEACHER', 'code_switch': '$TEACHER'},
    invention_teacher='$TEACHER',
    benchmark_mode=True,
)
print(f'  -> accepted={stats.renderings_accepted} files={len(stats.output_files)}')
"
