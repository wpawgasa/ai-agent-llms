#!/usr/bin/env bash
# Run scripts/run_exp_a_single.sh once per level (L1..L5), writing both result
# JSON and log file per level.
#
# Usage:
#   bash scripts/run_exp_a_per_level.sh --frontier-model <provider/model> [extra args...]
#   bash scripts/run_exp_a_per_level.sh <vllm-config>                     [extra args...]
#
# Each per-level invocation forwards extra args verbatim to run_exp_a_single.sh.
# Output files:
#   results/exp_a/<MODEL>_frontier_l<N>.json   (result JSON)
#   results/exp_a/<MODEL>_frontier_l<N>.log    (run log)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp_a"
mkdir -p "$RESULTS_DIR"

# Identify model tag for log filename: prefer --frontier-model, else basename of positional config.
MODEL_TAG=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--frontier-model" && $((i+1)) -lt ${#ARGS[@]} ]]; then
        MODEL_TAG="${ARGS[i+1]//\//_}"
        break
    fi
done
if [[ -z "$MODEL_TAG" && ${#ARGS[@]} -gt 0 && "${ARGS[0]}" != --* ]]; then
    MODEL_TAG=$(basename "${ARGS[0]}" .yaml)
fi
if [[ -z "$MODEL_TAG" ]]; then
    MODEL_TAG="run"
fi

echo "=== Per-level Experiment A — model tag: $MODEL_TAG ==="

for L in L1 L2 L3 L4 L5; do
    LEVEL_LC=$(echo "$L" | tr '[:upper:]' '[:lower:]')
    LOG_FILE="$RESULTS_DIR/${MODEL_TAG}_frontier_${LEVEL_LC}.log"
    echo
    echo ">>> $L ->  log: $LOG_FILE"
    bash "$PROJECT_ROOT/scripts/run_exp_a_single.sh" --level "$L" "$@" 2>&1 | tee "$LOG_FILE"
done

echo
echo "=== Per-level results ==="
for L in l1 l2 l3 l4 l5; do
    F="$RESULTS_DIR/${MODEL_TAG}_frontier_${L}.json"
    if [[ -f "$F" ]]; then
        SCORE=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
m = d.get('metrics', d)
print(f\"{m.get('weighted_workflow_score', 0):.3f}\")
" "$F")
        echo "  $L: weighted_workflow_score = $SCORE   ($F)"
    else
        echo "  $L: (missing) $F"
    fi
done
