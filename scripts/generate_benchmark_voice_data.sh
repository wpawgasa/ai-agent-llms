#!/usr/bin/env bash
# Generate the voice (spoken) stratum for the Phase 1 benchmark corpus.
#
# This is ADDITIVE to the existing text stratum at data/output/benchmark/task_a,
# which is FROZEN (258 conversations, 250 teacher-generated) -- this script
# never touches it. It writes a second, separate stratum so the two can be
# scored independently and blended (see DEFAULT_VOICE_WEIGHT).
#
#   - 250 conversations, 50 per level, L1 to L5.
#   - modality_preset="voice_only", barge_in_rate default 0.25.
#   - Writes data/output/benchmark/task_a_voice.
#   - ONE teacher model for every level: gemini-3.5-flash. The text stratum
#     used two (gemini-3-flash-preview for L1-L3, gemini-3-5-flash for
#     L4-L5); that split is a pre-existing defect in the text artifact, not
#     a pattern to copy here.
#   - No `language` argument is passed -- like the text stratum, this draws
#     English or Thai per sample at even odds. This deliberately does NOT
#     use the 20/50/30 Thai-weighted mix generate_voice_data.sh uses for the
#     training corpus: that weighting is right for training and wrong here,
#     since a Thai-weighted voice stratum could make a voice-vs-text
#     difference indistinguishable from a Thai-vs-English one.
#   - Seed 777: distinct from both the text benchmark's seed (100) and the
#     SFT voice batch's seed (4242), so this corpus draws its own domains
#     and workflows rather than shadowing either.
#
# Required env var: matches the chosen --teacher-model provider prefix
#   gemini-*  → GEMINI_API_KEY    gpt-*  → OPENAI_API_KEY    claude-*  → ANTHROPIC_API_KEY
#
# Usage:
#   ./scripts/generate_benchmark_voice_data.sh [OPTIONS]
#
# Options:
#   --output-dir <path>      Output directory (default: data/output/benchmark/task_a_voice)
#   --seed <n>                Random seed (default: 777)
#   --teacher-model <name>    Teacher model (default: gemini-3.5-flash). Pass ""
#                             (or "placeholder"/"none") for offline placeholder
#                             generation -- no API, no key required. Passing
#                             this explicitly always wins over --smoke-test's
#                             own default, regardless of argument order.
#   --barge-in-rate <f>       Share of voice conversations with one interruption
#                             (default: 0.25)
#   --max-placeholder-share <f>
#                             Quality gate. After generation, scripts/check_voice_batch.py
#                             aggregates the per-level .stats.json sidecars and FAILS
#                             (non-zero exit) when more than this share of the batch
#                             came from the offline placeholder generator rather than
#                             the teacher model. Default: 0.10 (10%).
#                             A run invoked with no teacher model is exempt (it asked
#                             for placeholders).
#   --skip-gate               Generate without running the quality gate. For
#                             debugging only; never for the paid 250-row run.
#   --smoke-test               Shorthand for --total 15 (3 per level). Also defaults
#                             the teacher to the offline placeholder (fast, no API
#                             key needed) UNLESS --teacher-model is passed
#                             explicitly (in either order), in which case the
#                             explicit model wins and a real, small, live-teacher
#                             smoke run is performed instead.
#   --dry-run                  Print the commands without running them
#
# Examples:
#   GEMINI_API_KEY=... ./scripts/generate_benchmark_voice_data.sh
#   ./scripts/generate_benchmark_voice_data.sh --smoke-test --dry-run
#   ./scripts/generate_benchmark_voice_data.sh --smoke-test --output-dir /tmp/bench_voice_smoke
#   ./scripts/generate_benchmark_voice_data.sh --smoke-test --teacher-model gemini-3.5-flash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output/benchmark/task_a_voice"
SEED=777
DRY_RUN=false
TOTAL=250
TEACHER_MODEL="gemini-3.5-flash"
TEACHER_MODEL_EXPLICIT=false
SMOKE_TEST=false
BARGE_IN_RATE=0.25
MAX_PLACEHOLDER_SHARE=0.10
SKIP_GATE=false

# Load .env if present (mirrors python-dotenv behaviour in _teacher_client.py)
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --total)          TOTAL="$2";          shift 2 ;;
        --teacher-model)  TEACHER_MODEL="$2";  TEACHER_MODEL_EXPLICIT=true; shift 2 ;;
        --barge-in-rate)  BARGE_IN_RATE="$2";  shift 2 ;;
        --max-placeholder-share) MAX_PLACEHOLDER_SHARE="$2"; shift 2 ;;
        --skip-gate)      SKIP_GATE=true;      shift ;;
        --smoke-test)     TOTAL=15; SMOKE_TEST=true; shift ;;
        --dry-run)        DRY_RUN=true;        shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --smoke-test defaults the teacher to the offline placeholder, but an
# explicit --teacher-model always wins, regardless of which flag came first
# on the command line -- this check runs only after the whole command line
# has been parsed, so order cannot matter.
if [[ "$SMOKE_TEST" == true && "$TEACHER_MODEL_EXPLICIT" == false ]]; then
    TEACHER_MODEL=""
fi

if ! [[ "$TOTAL" =~ ^[0-9]+$ ]] || (( TOTAL < 1 )); then
    echo "Invalid --total: $TOTAL (expected an integer >= 1)" >&2; exit 1
fi

# "" / "placeholder" / "none" (case-insensitive) mean offline placeholder
# generation (mirrors PLACEHOLDER_ALIASES in generate_sft_until_target.py):
# no API, no key required. Normalize to "" here so the emitted Python's
# teacher_model=None takes the same placeholder branch generate_workflow_dataset
# uses for teacher_model=None.
case "${TEACHER_MODEL,,}" in
    ""|placeholder|none) TEACHER_MODEL=""; REQUIRED_KEY="" ;;
    gemini*) REQUIRED_KEY="GEMINI_API_KEY" ;;
    gpt*)    REQUIRED_KEY="OPENAI_API_KEY" ;;
    claude*) REQUIRED_KEY="ANTHROPIC_API_KEY" ;;
    *) echo "Unsupported --teacher-model: $TEACHER_MODEL (expected prefix gemini-*, gpt-*, or claude-*, or \"\"/placeholder/none for offline)" >&2; exit 1 ;;
esac

if [[ "$DRY_RUN" = false && -n "$REQUIRED_KEY" ]]; then
    [[ -z "${!REQUIRED_KEY:-}" ]] && { echo "Error: $REQUIRED_KEY is not set (required for teacher model $TEACHER_MODEL)" >&2; exit 1; }
fi

# Python kwarg spelling: empty (placeholder) means "no teacher model", which
# the library spells as None.
if [[ -z "$TEACHER_MODEL" ]]; then
    TEACHER_MODEL_PY="None"
else
    TEACHER_MODEL_PY="'$TEACHER_MODEL'"
fi

run() {
    if [[ "$DRY_RUN" = true ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

DEST="$OUTPUT_DIR"

LEVELS=(L1 L2 L3 L4 L5)

declare -A COUNT=( [L1]=50 [L2]=50 [L3]=50 [L4]=50 [L5]=50 )

if [[ "$TOTAL" -ne 250 ]]; then
    # Override: uniform per-level count across all five levels, with the
    # rounding remainder given to the last level (L5) so the total is
    # exactly $TOTAL.
    LEVEL_BASE=$(( TOTAL / 5 ))
    LEVEL_REM=$(( TOTAL - LEVEL_BASE * 5 ))
    for LEVEL in "${LEVELS[@]}"; do
        COUNT[$LEVEL]=$LEVEL_BASE
    done
    COUNT[L5]=$(( LEVEL_BASE + LEVEL_REM ))
fi

TOTAL_ALL=0
for LEVEL in "${LEVELS[@]}"; do
    TOTAL_ALL=$(( TOTAL_ALL + COUNT[$LEVEL] ))
done

echo "=== Benchmark Voice Data Generation ==="
echo "Output dir:    $DEST"
echo "Seed:          $SEED"
echo "Teacher model: ${TEACHER_MODEL:-placeholder (offline)}"
echo "Barge-in rate: $BARGE_IN_RATE"
echo "Gate:          max placeholder share $MAX_PLACEHOLDER_SHARE$( [[ "$SKIP_GATE" == true ]] && echo " (SKIPPED)" )"
echo "Modality:      voice_only"
echo "Language:      mixed (en/th, no language= argument passed)"
echo "Totals:        L1=${COUNT[L1]} L2=${COUNT[L2]} L3=${COUNT[L3]} L4=${COUNT[L4]} L5=${COUNT[L5]}  (${TOTAL_ALL} total)"
echo "==========================="

for LEVEL in "${LEVELS[@]}"; do
    N="${COUNT[$LEVEL]}"
    echo ""
    echo "  --- $LEVEL ($N samples) ---"
    echo "  [$LEVEL] ${TEACHER_MODEL:-placeholder} / $N samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$N,
    teacher_model=$TEACHER_MODEL_PY,
    output_dir=Path('$DEST'),
    seed=$SEED,
    modality_preset=\"voice_only\",
    barge_in_rate=$BARGE_IN_RATE,
    require_tool_stay=True,
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"
done

echo ""
echo "=== Generated. Benchmark voice data in $DEST (${TOTAL_ALL} total) ==="

# Quality gate (spec risk 2). Placeholder output is format-perfect by
# construction, so a teacher model that failed on every sample yields a batch
# with zero format violations and five success lines above -- success and
# total failure are indistinguishable without reading the sidecars. This reads
# them. Non-zero exit here means DO NOT MERGE.
if [[ "$SKIP_GATE" == true ]]; then
    echo "[skip-gate] quality gate not run (--skip-gate)"
else
    echo ""
    run python3 "$PROJECT_ROOT/scripts/check_voice_batch.py" \
        --input-dir "$DEST" \
        --max-placeholder-share "$MAX_PLACEHOLDER_SHARE"
fi

echo ""
echo "=== Done. ==="
