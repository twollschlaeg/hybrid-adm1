#!/bin/bash
# Run the planned experiments on a server without SLURM.
#
#   bash scripts/run_experiments.sh --seeds "45 46 47"
#   bash scripts/run_experiments.sh --experiments "fold_fast_narrow_tan fold_fast_narrow_tan_linear" \
#       --models r4 --parallel 3
#
# Options:
#   --experiments "..."  files in configs/experiments/ (default: "fold_fast_narrow_tan")
#   --models "..."       variants (default: "r2 r3 r4")
#   --parallel N         experiment/variant pairs run at once (default 1). Each pair
#                        trains its folds in parallel on FOLD_WORKERS cores (default 5),
#                        so N pairs need about 5 N cores.
#   --tag NAME           names the runs <experiment>_<tag> (default: a timestamp)
#   --seeds "..."        training seeds: every pair runs once per seed, named
#                        <experiment>_seed<N>_<tag> (default: one run with the
#                        default seed, named <experiment>_<tag>)
#   --fold-hyperparameters PATTERN
#                        every fold's own selection, a path with {model} and {fold},
#                        e.g. a nested search's
#                        "runs/{model}/tuning_<name>_outer{fold}/best_hyperparameters.json"
#
# EPOCHS, FOLD_WORKERS and HYPERPARAMETERS are passed on to run_experiment.sh.
# Each pair logs to logs/<experiment>_<model>_<tag>.log (with _seed<N> before the tag
# for a seed). When all pairs are done, the variants of every experiment are merged
# into evaluations/<run name>/, one per seed.
#
# Start it so that it survives logging out, e.g.
#   mkdir -p logs; nohup bash scripts/run_experiments.sh --parallel 3 > logs/experiments.log 2>&1 &

set -euo pipefail

EXPERIMENTS="fold_fast_narrow_tan"
MODELS="r2 r3 r4"
PARALLEL=1
TAG="$(date +%Y%m%d_%H%M%S)"
SEEDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --models)      MODELS="$2"; shift 2 ;;
        --parallel)    PARALLEL="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --seeds)       SEEDS="$2"; shift 2 ;;
        --fold-hyperparameters) export FOLD_HYPERPARAMETERS="$2"; shift 2 ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
                       "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

for seed in $SEEDS; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "ERROR: seed '$seed' is not a whole number" >&2; exit 1; }
done
# One run per seed; without seeds, one run with train_cv.py's default seed.
read -ra SEED_LIST <<< "$SEEDS"
(( ${#SEED_LIST[@]} )) || SEED_LIST=("")

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN_DIR="${ADM1_RUN_DIR:-$PWD/runs}"
DATA_DIR="${ADM1_DATA_DIR:-$PWD/data}"
for campaign in LaRA-temp LaRA-substrate rice-bran; do
    [[ -d "$DATA_DIR/$campaign" ]] || { echo "ERROR: $DATA_DIR/$campaign is missing" >&2; exit 1; }
done
for experiment in $EXPERIMENTS; do
    [[ -f "configs/experiments/$experiment.json" ]] || {
        echo "ERROR: configs/experiments/$experiment.json not found" >&2; exit 1; }
done
python -c "import jax, equinox, diffrax" || { echo "ERROR: activate the Python environment first" >&2; exit 1; }
if [[ -n "${FOLD_HYPERPARAMETERS:-}" ]]; then
    [[ -z "${HYPERPARAMETERS:-}" ]] || { echo "ERROR: HYPERPARAMETERS and --fold-hyperparameters exclude each other" >&2; exit 1; }
    [[ "$FOLD_HYPERPARAMETERS" == *"{fold}"* ]] || { echo "ERROR: --fold-hyperparameters names no {fold}" >&2; exit 1; }
    FOLDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["k_folds"])' configs/split.json)"
    for model in $MODELS; do
        for fold in $(seq 1 "$FOLDS"); do
            selection="${FOLD_HYPERPARAMETERS//\{model\}/$model}"
            selection="${selection//\{fold\}/$fold}"
            [[ -f "$selection" ]] || { echo "ERROR: $selection not found (--fold-hyperparameters)" >&2; exit 1; }
        done
    done
fi
# Where an experiment with per-fold calibrations reads them (calibrate_folds.py).
export ADM1_BENCHMARKS_DIR="${ADM1_BENCHMARKS_DIR:-$RUN_DIR/benchmarks}"
# e.g. fold_fast_narrow_tan_no_ph fits no pH, which R4 does not produce; every
# experiment needs every fold's calibration under ADM1_BENCHMARKS_DIR.
python -c "import sys; from adm1_package.experiments.training import check_experiments; check_experiments(sys.argv[1].split(), sys.argv[2].split())" \
    "$EXPERIMENTS" "$MODELS" || { echo "ERROR: an experiment does not apply to every model listed, or its per-fold calibrations are missing" >&2; exit 1; }
mkdir -p logs

echo "experiments: $EXPERIMENTS | models: $MODELS | seeds: ${SEEDS:-default} | $PARALLEL at a time | runs: $RUN_DIR/<model>/<experiment>_$TAG"

failed=()
running=0
for experiment in $EXPERIMENTS; do
    for model in $MODELS; do
        for seed in "${SEED_LIST[@]}"; do
            name="${experiment}${seed:+_seed$seed}_${TAG}"
            log="logs/${experiment}_${model}${seed:+_seed$seed}_${TAG}.log"
            echo "start  $experiment / $model${seed:+ / seed $seed}  (log: $log)"
            ( SEED="$seed" bash scripts/run_experiment.sh "$experiment" "$model" "$name" > "$log" 2>&1 \
                  || echo "$experiment/$model${seed:+ seed $seed}" >> "logs/failed_${TAG}.txt" ) &
            running=$(( running + 1 ))
            if (( running >= PARALLEL )); then
                wait -n || true
                running=$(( running - 1 ))
            fi
        done
    done
done
wait

if [[ -f "logs/failed_${TAG}.txt" ]]; then
    mapfile -t failed < "logs/failed_${TAG}.txt"
    echo "FAILED: ${failed[*]} -- see logs/<experiment>_<model>_${TAG}.log"
fi

for experiment in $EXPERIMENTS; do
    for seed in "${SEED_LIST[@]}"; do
        name="${experiment}${seed:+_seed$seed}_${TAG}"
        results=()
        for model in $MODELS; do
            directory="$RUN_DIR/$model/$name/results"
            [[ -f "$directory/cv_fold_metrics.csv" ]] && results+=("$directory")
        done
        if (( ${#results[@]} )); then
            python scripts/combine_results.py --results "${results[@]}" \
                --output "evaluations/$name"
            echo "merged $experiment${seed:+ seed $seed} -> evaluations/$name"
        fi
    done
done

(( ${#failed[@]} == 0 ))
