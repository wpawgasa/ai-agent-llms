#!/usr/bin/env bash
# Generate supervised fine-tuning (SFT) training data.
#
# Curriculum-weighted: L1=3000, L2=3000, L3=2502, L4=2001, L5=2001 (~12504 total).
# Each level is split evenly across three language runs (all use --teacher-model):
#   - 1/3: <teacher-model>  / English        / <behavior-preset>
#   - 1/3: <teacher-model>  / Thai            / <behavior-preset>
#   - 1/3: <teacher-model>  / code-switching  / <behavior-preset>
#
# Behavior presets (--behavior-preset):
#   default          cooperative=0.60, adversarial_probing=0.15, digressing=0.10, invalid_tool_inputs=0.15
#   adversarial      cooperative=0.45, adversarial_probing=0.25, digressing=0.15, invalid_tool_inputs=0.15
#   balanced         cooperative=0.25, adversarial_probing=0.25, digressing=0.25, invalid_tool_inputs=0.25
#   cooperative_only cooperative=1.00, adversarial_probing=0.00, digressing=0.00, invalid_tool_inputs=0.00
#
# Required env var: matches the chosen --teacher-model provider prefix
#   gemini-*  → GEMINI_API_KEY    gpt-*  → OPENAI_API_KEY    claude-*  → ANTHROPIC_API_KEY
#
# Usage:
#   ./scripts/generate_sft_data.sh [OPTIONS]
#
# Options:
#   --output-dir <path>        Base output directory (default: data/output)
#   --seed <n>                 Random seed (default: 42)
#   --samples-per-leg <n>      Override per-leg sample count for all levels
#   --smoke-test               Shorthand for --samples-per-leg 3 (quick pipeline check)
#   --teacher-model <name>     Teacher model for all three language legs
#                              (default: gemini-3.5-flash; routed by prefix gemini-*/gpt-*/claude-*)
#   --behavior-preset <preset> User behavior distribution (default: adversarial)
#   --intent-category <preset> Intent mix: default (70/30 service/upsell),
#                              service_only, upsell_heavy (default: default)
#   --initiation <preset>      Inbound/outbound mix: default (100% inbound),
#                              balanced (70/30 user/agent), outbound_heavy (40/60) (default: default)
#   --retry-budget <n>         Override the tool-error retry budget for ALL levels: TOTAL
#                              attempts at a failing call, counting the first (so 1 = no
#                              retry). Must be >= 1. Default: per-level from
#                              COMPLEXITY_SPECS (L1-L4: 2, L5: 3)
#   --retry-exhaustion <p>     What the agent does once the budget is spent: auto (default,
#                              keep the per-level policy), error_path, handoff_in_state, none.
#                              error_path still degrades per-sample to handoff_in_state on
#                              subgraphs with no tool_error arc
#   --require-tool-stay        Enforce the stay convention: a turn issuing a <tool_call>
#                              must annotate [STATE: X → X] (default: on)
#   --no-require-tool-stay     Disable the stay-convention gate to regenerate v1-comparable
#                              data (advancing tool-call turns are not repaired)
#   --dry-run                  Print commands without executing
#
# Examples:
#   GEMINI_API_KEY=... ./scripts/generate_sft_data.sh
#   ./scripts/generate_sft_data.sh --smoke-test --dry-run
#   ./scripts/generate_sft_data.sh --samples-per-leg 270 --behavior-preset cooperative_only
#   ./scripts/generate_sft_data.sh --teacher-model gpt-5.4-mini-2026-03-17
#   # v1-comparable corpus: no stay gate, single attempt at every level
#   ./scripts/generate_sft_data.sh --no-require-tool-stay --retry-budget 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/data/output"
SEED=42
DRY_RUN=false
SAMPLES_PER_LEG=""  # empty = use curriculum defaults
TEACHER_MODEL="gemini-3.5-flash"
BEHAVIOR_PRESET="adversarial"
INTENT_CATEGORY="default"
INITIATION="default"
RETRY_BUDGET=""            # empty = per-level from COMPLEXITY_SPECS
RETRY_EXHAUSTION="auto"    # auto = per-level from COMPLEXITY_SPECS
REQUIRE_TOOL_STAY=true

# Load .env if present (mirrors python-dotenv behaviour in _teacher_client.py)
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)       OUTPUT_DIR="$2";       shift 2 ;;
        --seed)             SEED="$2";             shift 2 ;;
        --samples-per-leg)  SAMPLES_PER_LEG="$2"; shift 2 ;;
        --smoke-test)       SAMPLES_PER_LEG=3;    shift ;;
        --teacher-model)    TEACHER_MODEL="$2";   shift 2 ;;
        --behavior-preset)  BEHAVIOR_PRESET="$2"; shift 2 ;;
        --intent-category)  INTENT_CATEGORY="$2"; shift 2 ;;
        --initiation)       INITIATION="$2";      shift 2 ;;
        --retry-budget)     RETRY_BUDGET="$2";    shift 2 ;;
        --retry-exhaustion) RETRY_EXHAUSTION="$2"; shift 2 ;;
        --require-tool-stay)    REQUIRE_TOOL_STAY=true;  shift ;;
        --no-require-tool-stay) REQUIRE_TOOL_STAY=false; shift ;;
        --dry-run)          DRY_RUN=true;          shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

case "$INTENT_CATEGORY" in
    default|service_only|upsell_heavy) ;;
    *) echo "Unknown --intent-category: $INTENT_CATEGORY (expected default, service_only, upsell_heavy)" >&2; exit 1 ;;
esac

case "$INITIATION" in
    default|balanced|outbound_heavy) ;;
    *) echo "Unknown --initiation: $INITIATION (expected default, balanced, outbound_heavy)" >&2; exit 1 ;;
esac

case "$RETRY_EXHAUSTION" in
    auto|error_path|handoff_in_state|none) ;;
    *) echo "Unknown --retry-exhaustion: $RETRY_EXHAUSTION (expected auto, error_path, handoff_in_state, none)" >&2; exit 1 ;;
esac

# The budget counts the first attempt, so anything below 1 is meaningless.
# Validated here rather than only in Python so a typo fails before the first
# (billable) teacher call rather than after it.
if [[ -n "$RETRY_BUDGET" ]]; then
    if ! [[ "$RETRY_BUDGET" =~ ^[0-9]+$ ]] || (( RETRY_BUDGET < 1 )); then
        echo "Invalid --retry-budget: $RETRY_BUDGET (expected an integer >= 1)" >&2; exit 1
    fi
fi

# Python kwarg spellings: empty/auto mean "no override", which the library
# spells as None.
RETRY_BUDGET_PY="${RETRY_BUDGET:-None}"
if [[ "$RETRY_EXHAUSTION" == "auto" ]]; then
    RETRY_EXHAUSTION_PY="None"
else
    RETRY_EXHAUSTION_PY="'$RETRY_EXHAUSTION'"
fi
if [[ "$REQUIRE_TOOL_STAY" == true ]]; then
    REQUIRE_TOOL_STAY_PY="True"
else
    REQUIRE_TOOL_STAY_PY="False"
fi

# Required API key is determined by the teacher model's provider prefix
# (mirrors call_teacher_model routing in data/_teacher_client.py).
case "$TEACHER_MODEL" in
    gemini*) REQUIRED_KEY="GEMINI_API_KEY" ;;
    gpt*)    REQUIRED_KEY="OPENAI_API_KEY" ;;
    claude*) REQUIRED_KEY="ANTHROPIC_API_KEY" ;;
    *) echo "Unsupported --teacher-model: $TEACHER_MODEL (expected prefix gemini-*, gpt-*, or claude-*)" >&2; exit 1 ;;
esac

if [[ "$DRY_RUN" = false ]]; then
    [[ -z "${!REQUIRED_KEY:-}" ]] && { echo "Error: $REQUIRED_KEY is not set (required for teacher model $TEACHER_MODEL)" >&2; exit 1; }
fi

run() {
    if [[ "$DRY_RUN" = true ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

DEST="$OUTPUT_DIR/sft/task_a"

# Samples per level per leg (1/3 of total, rounded to clean integers).
#
# Single-sourced from generate_sft_until_target.py's CURRICULUM so the two
# generation entry points can never drift into producing differently-sized
# corpora. Read with `ast` rather than by importing the module: importing it
# pulls in llm_workflow_agents.data.generate_workflows (and transitively the
# whole data stack), which would make --dry-run — today a pure, dependency-free
# preview — fail in any environment without the package installed. `ast` needs
# nothing but stdlib, evaluates no code from the file, and exits non-zero (which
# `set -e` turns into an abort) if the constant is ever renamed or removed,
# rather than silently falling back to a stale copy of the numbers.
#
# Read into a variable (not a process substitution) so a parse failure is a
# failed command substitution we can act on: `while ... done < <(cmd)` reports
# the *loop's* status, so a dead `cmd` would leave the array empty and sail past
# `set -e`.
CURRICULUM_RAW="$(python3 - "$SCRIPT_DIR/generate_sft_until_target.py" <<'PYEOF'
import ast, sys

src = open(sys.argv[1]).read()
for node in ast.parse(src).body:
    target = None
    if isinstance(node, ast.AnnAssign):
        target = node.target
    elif isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    if isinstance(target, ast.Name) and target.id == "CURRICULUM":
        for k, v in ast.literal_eval(node.value).items():
            print(k, v)
        break
else:
    sys.exit(f"CURRICULUM not found in {sys.argv[1]} (single-source-of-truth broken)")
PYEOF
)" || { echo "Failed to read CURRICULUM from generate_sft_until_target.py" >&2; exit 1; }

declare -A CURRICULUM
while read -r _lvl _n; do
    [[ -n "$_lvl" ]] && CURRICULUM["$_lvl"]="$_n"
done <<< "$CURRICULUM_RAW"

CURRICULUM_COUNT=${#CURRICULUM[@]}
(( CURRICULUM_COUNT == 5 )) || {
    echo "Expected 5 curriculum levels from generate_sft_until_target.py, parsed $CURRICULUM_COUNT" >&2
    exit 1
}

# Apply --samples-per-leg / --smoke-test override
declare -A THIRD
for LEVEL in L1 L2 L3 L4 L5; do
    [[ -n "${CURRICULUM[$LEVEL]:-}" ]] || {
        echo "CURRICULUM from generate_sft_until_target.py has no entry for $LEVEL" >&2; exit 1
    }
    THIRD[$LEVEL]="${SAMPLES_PER_LEG:-${CURRICULUM[$LEVEL]}}"
done

if [[ -n "$SAMPLES_PER_LEG" ]]; then
    TOTAL_ALL=$(( SAMPLES_PER_LEG * 3 * 5 ))
    TOTALS_MSG="all levels: ${SAMPLES_PER_LEG} per leg × 3 legs × 5 levels = ${TOTAL_ALL} total (override)"
else
    # Derived, not hardcoded: these numbers ARE the curriculum × 3 legs, so they
    # cannot go stale when the single source of truth changes.
    TOTAL_ALL=0
    TOTALS_PARTS=()
    for LEVEL in L1 L2 L3 L4 L5; do
        LEVEL_TOTAL=$(( ${CURRICULUM[$LEVEL]} * 3 ))
        TOTALS_PARTS+=("$LEVEL=$LEVEL_TOTAL")
        TOTAL_ALL=$(( TOTAL_ALL + LEVEL_TOTAL ))
    done
    TOTALS_MSG="$(printf '%s, ' "${TOTALS_PARTS[@]}")"
    TOTALS_MSG="${TOTALS_MSG%, }  (~${TOTAL_ALL} total)"
fi

echo "=== SFT Data Generation ==="
echo "Output dir:    $DEST"
echo "Seed:          $SEED"
echo "Teacher model: $TEACHER_MODEL"
echo "Behavior:      $BEHAVIOR_PRESET"
echo "Intent mix:    $INTENT_CATEGORY"
echo "Initiation:    $INITIATION"
echo "Retry budget:  ${RETRY_BUDGET:-per-level}   Exhaustion: $RETRY_EXHAUSTION"
echo "Tool-stay:     $([[ "$REQUIRE_TOOL_STAY" == true ]] && echo "on" || echo "OFF (v1-comparable)")"
echo "Totals:        $TOTALS_MSG"
echo "Split:         1/3 en  +  1/3 th  +  1/3 code_switch per level (teacher: $TEACHER_MODEL)"
echo "==========================="

for LEVEL in L1 L2 L3 L4 L5; do
    T="${THIRD[$LEVEL]}"
    TOTAL=$(( T * 3 ))
    echo ""
    echo "  --- $LEVEL ($TOTAL samples) ---"

    echo "  [$LEVEL] $TEACHER_MODEL / English / $T samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$T,
    teacher_model='$TEACHER_MODEL',
    output_dir=Path('$DEST'),
    seed=$SEED,
    language='en',
    behavior_preset='$BEHAVIOR_PRESET',
    intent_category_preset='$INTENT_CATEGORY',
    initiation_preset='$INITIATION',
    retry_budget=$RETRY_BUDGET_PY,
    retry_exhaustion=$RETRY_EXHAUSTION_PY,
    require_tool_stay=$REQUIRE_TOOL_STAY_PY,
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"

    echo "  [$LEVEL] $TEACHER_MODEL / Thai / $T samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$T,
    teacher_model='$TEACHER_MODEL',
    output_dir=Path('$DEST'),
    seed=$SEED,
    language='th',
    behavior_preset='$BEHAVIOR_PRESET',
    intent_category_preset='$INTENT_CATEGORY',
    initiation_preset='$INITIATION',
    retry_budget=$RETRY_BUDGET_PY,
    retry_exhaustion=$RETRY_EXHAUSTION_PY,
    require_tool_stay=$REQUIRE_TOOL_STAY_PY,
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"

    echo "  [$LEVEL] $TEACHER_MODEL / code-switching / $T samples..."
    run python3 -c "
from pathlib import Path
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
meta = generate_workflow_dataset(
    complexity_level='$LEVEL',
    num_samples=$T,
    teacher_model='$TEACHER_MODEL',
    output_dir=Path('$DEST'),
    seed=$SEED,
    language='code_switch',
    behavior_preset='$BEHAVIOR_PRESET',
    intent_category_preset='$INTENT_CATEGORY',
    initiation_preset='$INITIATION',
    retry_budget=$RETRY_BUDGET_PY,
    retry_exhaustion=$RETRY_EXHAUSTION_PY,
    require_tool_stay=$REQUIRE_TOOL_STAY_PY,
)
print(f'  -> {meta.output_files[0].name}  ({meta.num_samples} samples)')
"
done

echo ""
echo "=== Done. SFT data in $DEST ($TOTALS_MSG) ==="
