#!/bin/bash
# Train and evaluate one experiment for one ADM1 variant.
#
#   bash scripts/run_experiment.sh <experiment> <model> [run-name]
#   bash scripts/run_experiment.sh fold_fast_narrow_tan r4
#
# <experiment> names a file in configs/experiments/, e.g. fold_fast_narrow_tan.
# The run lands in $ADM1_RUN_DIR/<model>/<run-name> (default ./runs), named
# <experiment>_<timestamp> unless a name is given. Run from the repository
# root, with the Python environment active.
#
# Environment:
#   EPOCHS            epochs per fold (default 100)
#   FOLD_WORKERS      folds trained in parallel, one core each (default 5)
#   HYPERPARAMETERS   a tuned best_hyperparameters.json (default: each
#                     variant's default values)
#   FOLD_HYPERPARAMETERS
#                     every fold's own selection instead, as a path with {fold}
#                     and {model} in it, e.g. a nested search's
#                     runs/{model}/tuning_<name>_outer{fold}/best_hyperparameters.json
#   SEED              training seed, train_cv.py --seed (default: its own, 42)

set -euo pipefail

if [[ $# -lt 2 ]]; then
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}" >&2
    exit 1
fi

EXPERIMENT="$1"
MODEL="$2"
RUN_NAME="${3:-${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_FILE="configs/experiments/${EXPERIMENT}.json"
RUN_DIR="${ADM1_RUN_DIR:-$PWD/runs}"
RUN="$RUN_DIR/$MODEL/$RUN_NAME"

[[ -f "$EXPERIMENT_FILE" ]] || { echo "ERROR: $EXPERIMENT_FILE not found" >&2; exit 1; }
case "$MODEL" in r2|r3|r4) ;; *) echo "ERROR: model must be r2, r3 or r4" >&2; exit 1 ;; esac

# One process per core and one thread per process: JAX and BLAS otherwise each
# claim every core and the fold workers oversubscribe.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export ADM1_RUN_DIR="$RUN_DIR"

echo "== $(date '+%F %T')  $MODEL / $EXPERIMENT -> $RUN"
python scripts/train_cv.py \
    --model "$MODEL" \
    --experiment "$EXPERIMENT_FILE" \
    --epochs "${EPOCHS:-100}" \
    --fold-workers "${FOLD_WORKERS:-5}" \
    --run-name "$RUN_NAME" \
    ${HYPERPARAMETERS:+--hyperparameters "$HYPERPARAMETERS"} \
    ${FOLD_HYPERPARAMETERS:+--fold-hyperparameters "$FOLD_HYPERPARAMETERS"} \
    ${SEED:+--seed "$SEED"}

echo "== $(date '+%F %T')  evaluating $RUN"
python scripts/evaluate.py --model "$MODEL" --run "$RUN"

echo "== $(date '+%F %T')  done: $RUN/results"
