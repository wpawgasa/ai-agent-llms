#!/usr/bin/env bash
# Generate voice (spoken) Task A conversations.
#
# A voice conversation splits every assistant turn into <S>...</S> chunks for a
# text-to-speech engine. See data/voice_convention.py and
# docs/superpowers/specs/2026-08-20-voice-conversation-generation-design.md.
#
# Total 2400 conversations, weighted toward the Thai voicebot deployment:
#   en 480 (20%) / th 1200 (50%) / code_switch 720 (30%)
# Split across L1-L5 with the same curriculum weights as the text corpus.
#
# Fifteen legs (5 levels x 3 languages), fixed per-level/per-language sizes
# (the last leg of each level already absorbs the 20/50/30 rounding remainder):
#
#   Level   Weight   en    th    code_switch   Level total
#   L1      0.24     115   288   173           576
#   L2      0.24     115   288   173           576
#   L3      0.20     96    240   144           480
#   L4      0.16     77    192   115           384
#   L5      0.16     77    192   115           384
#
# Required env var: matches the chosen --teacher-model provider prefix
#   gemini-*  → GEMINI_API_KEY    gpt-*  → OPENAI_API_KEY    claude-*  → ANTHROPIC_API_KEY
#
# Usage:
#   ./scripts/generate_voice_data.sh [OPTIONS]
#
# Options:
#   --output-dir <path>      Output directory (default: data/output/sft/task_a_voice)
#   --seed <n>               Random seed (default: 4242; differs from the text
#                            corpus seed of 42 so the two batches draw different
#                            domains and workflows)
#   --total <n>              Total conversations (default: 2400). Overrides the
#                            fixed per-leg table with a uniform per-leg count
#                            (n / 15, remainder on the last leg).
#   --teacher-model <name>   Teacher model (default: gemini-3.5-flash). Pass ""
#                            (or "placeholder"/"none") for offline placeholder
#                            generation -- no API, no key required. Passing
#                            this explicitly always wins over --smoke-test's
#                            own default, regardless of argument order.
#   --barge-in-rate <f>      Share of voice conversations with one interruption
#                            (default: 0.25)
#   --max-placeholder-share <f>
#                            Quality gate. After generation, scripts/check_voice_batch.py
#                            aggregates the per-leg .stats.json sidecars and FAILS
#                            (non-zero exit) when more than this share of the batch
#                            came from the offline placeholder generator rather than
#                            the teacher model. Default: 0.10 (10%).
#                            Placeholder rows are format-perfect and deterministic,
#                            so a total teacher failure otherwise looks exactly like
#                            success -- see spec risk 2. A run invoked with no teacher
#                            model is exempt (it asked for placeholders).
#   --skip-gate              Generate without running the quality gate. For
#                            debugging only; never for the paid 2400-row run.
#   --smoke-test             Shorthand for --total 15. Also defaults the
#                            teacher to the offline placeholder (fast, no API
#                            key needed) UNLESS --teacher-model is passed
#                            explicitly (in either order), in which case the
#                            explicit model wins and a real, small, live-teacher
#                            smoke run is performed instead -- useful for
#                            cheaply checking <S> format compliance before
#                            committing to the full 2400-conversation batch.
#   --dry-run                Print the commands without running them
#
# Examples:
#   GEMINI_API_KEY=... ./scripts/generate_voice_data.sh
#   ./scripts/generate_voice_data.sh --smoke-test --dry-run
#   ./scripts/generate_voice_data.sh --smoke-test --output-dir /tmp/voice_smoke
#   ./scripts/generate_voice_data.sh --smoke-test --teacher-model gemini-3.5-flash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output/sft/task_a_voice"
SEED=4242
DRY_RUN=false
TOTAL=2400
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
LANGS=(en th code_switch)

# Fixed per-level/per-language leg sizes: each level's share of 2400
# (0.24/0.24/0.20/0.16/0.16), split 20/50/30 en/th/code_switch within the
# level. The fifteen numbers already sum to exactly 2400 with the rounding
# remainder on each level's last (code_switch) leg.
declare -A EN=( [L1]=115 [L2]=115 [L3]=96 [L4]=77 [L5]=77 )
declare -A TH=( [L1]=288 [L2]=288 [L3]=240 [L4]=192 [L5]=192 )
declare -A CS=( [L1]=173 [L2]=173 [L3]=144 [L4]=115 [L5]=115 )

if [[ "$TOTAL" -ne 2400 ]]; then
    # Override: uniform per-leg count across all fifteen (level, language)
    # legs, with the rounding remainder given to the very last leg
    # (L5 / code_switch) so the total is exactly $TOTAL.
    LEG_BASE=$(( TOTAL / 15 ))
    LEG_REM=$(( TOTAL - LEG_BASE * 15 ))
    for LEVEL in "${LEVELS[@]}"; do
        EN[$LEVEL]=$LEG_BASE
        TH[$LEVEL]=$LEG_BASE
        CS[$LEVEL]=$LEG_BASE
    done
    CS[L5]=$(( LEG_BASE + LEG_REM ))
fi

TOTAL_ALL=0
for LEVEL in "${LEVELS[@]}"; do
    TOTAL_ALL=$(( TOTAL_ALL + EN[$LEVEL] + TH[$LEVEL] + CS[$LEVEL] ))
done

echo "=== Voice Data Generation ==="
echo "Output dir:    $DEST"
echo "Seed:          $SEED"
echo "Teacher model: ${TEACHER_MODEL:-placeholder (offline)}"
echo "Barge-in rate: $BARGE_IN_RATE"
echo "Gate:          max placeholder share $MAX_PLACEHOLDER_SHARE$( [[ "$SKIP_GATE" == true ]] && echo " (SKIPPED)" )"
echo "Modality:      voice_only"
echo "Totals:        en=$(( EN[L1]+EN[L2]+EN[L3]+EN[L4]+EN[L5] )) / th=$(( TH[L1]+TH[L2]+TH[L3]+TH[L4]+TH[L5] )) / code_switch=$(( CS[L1]+CS[L2]+CS[L3]+CS[L4]+CS[L5] ))  (~${TOTAL_ALL} total)"
echo "==========================="

for LEVEL in "${LEVELS[@]}"; do
    LEVEL_TOTAL=$(( EN[$LEVEL] + TH[$LEVEL] + CS[$LEVEL] ))
    echo ""
    echo "  --- $LEVEL ($LEVEL_TOTAL samples) ---"

    for LANG in "${LANGS[@]}"; do
        case "$LANG" in
            en) N="${EN[$LEVEL]}" ;;
            th) N="${TH[$LEVEL]}" ;;
            code_switch) N="${CS[$LEVEL]}" ;;
        esac

        echo "  [$LEVEL] ${TEACHER_MODEL:-placeholder} / $LANG / $N samples..."
        run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    language=\"$LANG\",
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
done

echo ""
echo "=== Generated. Voice data in $DEST (~${TOTAL_ALL} total) ==="

# Quality gate (spec risk 2). Placeholder output is format-perfect by
# construction, so a teacher model that failed on every sample yields a batch
# with zero format violations and fifteen success lines above -- success and
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
