#!/bin/bash
# Submit the planned experiments: one array task per experiment, variant and seed,
# then a job that merges the variants of every experiment once the array has ended.
#
#   bash scripts/hpc/submit_experiments.sh --dry-run      # check and print the plan
#   bash scripts/hpc/submit_experiments.sh                # fold_fast_narrow_tan x r2, r3, r4
#   bash scripts/hpc/submit_experiments.sh --experiments "fold_fast_narrow_tan_linear" --models "r4"
#
# Options:
#   --experiments "..."  files in configs/experiments/ (default: "fold_fast_narrow_tan")
#   --models "..."       variants (default: "r2 r3 r4")
#   --epochs N           epochs per fold (default 100)
#   --seeds "..."        training seeds: one task per seed, runs named
#                        <experiment>_seed<N>_<job> (default: one task with the
#                        default seed, named <experiment>_<job>)
#   --hyperparameters F  a tuned best_hyperparameters.json, used for every model
#                        listed (default: each variant's default values)
#   --fold-hyperparameters PATTERN
#                        every fold's own selection instead, a path with {model} and
#                        {fold}, e.g. what submit_nested.sh prints:
#                        "runs/{model}/tuning_<name>_outer{fold}/best_hyperparameters.json"
#   --time HH:MM:SS      wall clock per task (default 12:00:00)
#   --partition NAME     partition (default: the cluster's default)
#   --cpus N             cores per task; one fold per core (default 5)
#   --mem-per-cpu M      memory per core (default 4G)
#   --dry-run            print the plan and the sbatch commands, submit nothing
#
# Needs ADM1_VENV (see scripts/hpc/README.md). Runs go to $ADM1_RUN_DIR, default
# ${SCRATCH:-$PWD}/runs; the results and the merged tables end up in ./runs and
# ./evaluations. Every experiment is built on per-fold calibrations and reads them
# from $ADM1_BENCHMARKS_DIR, default ./runs/benchmarks: copy that folder from the
# machine that ran scripts/calibrate/calibrate_folds.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENTS="fold_fast_narrow_tan"
MODELS="r2 r3 r4"
EPOCHS=100
SEEDS=""
HYPERPARAMETERS=""
FOLD_HYPERPARAMETERS=""
WALLTIME="12:00:00"
PARTITION=""
CPUS=5
MEM_PER_CPU=4G
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiments)     EXPERIMENTS="$2"; shift 2 ;;
        --models)          MODELS="$2"; shift 2 ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --seeds)           SEEDS="$2"; shift 2 ;;
        --hyperparameters) HYPERPARAMETERS="$2"; shift 2 ;;
        --fold-hyperparameters) FOLD_HYPERPARAMETERS="$2"; shift 2 ;;
        --time)            WALLTIME="$2"; shift 2 ;;
        --partition)       PARTITION="$2"; shift 2 ;;
        --cpus)            CPUS="$2"; shift 2 ;;
        --mem-per-cpu)     MEM_PER_CPU="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
                       "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

for seed in $SEEDS; do
    [[ "$seed" =~ ^[0-9]+$ ]] || fail "seed '$seed' is not a whole number"
done

# ---------------------------------------------------------------------------
# Checks that otherwise only fail after the queue wait
# ---------------------------------------------------------------------------
command -v sbatch >/dev/null 2>&1 || fail "sbatch not found; is this a login node?
     Without SLURM, use: bash scripts/run_experiments.sh"
[[ -n "${ADM1_VENV:-}" ]] || fail "ADM1_VENV is not set; build a venv with bash scripts/hpc/setup_env.sh"
[[ -d "$ADM1_VENV" ]] || fail "ADM1_VENV=$ADM1_VENV does not exist"

echo "checking activation as a batch job would see it..."
ADM1_VENV="$ADM1_VENV" bash -c '
    source scripts/hpc/activate_env.sh || exit 1
    python -c "
import jax, equinox, diffrax, orbax.checkpoint
jax.config.update(\"jax_enable_x64\", True)
import jax.numpy as jnp
assert jnp.zeros(1).dtype == jnp.float64, \"float64 unavailable\"
"' || fail "the environment does not come up in a non-interactive shell (see above)"

[[ -f configs/split.json ]] || fail "configs/split.json is missing. Run: python scripts/make_splits.py"
DATA_DIR="${ADM1_DATA_DIR:-$PWD/data}"
for campaign in LaRA-temp LaRA-substrate rice-bran; do
    [[ -d "$DATA_DIR/$campaign" ]] || fail \
        "$DATA_DIR/$campaign is missing. Copy data/ (it is not in git) or set ADM1_DATA_DIR."
done
for experiment in $EXPERIMENTS; do
    [[ -f "configs/experiments/$experiment.json" ]] || fail "configs/experiments/$experiment.json not found"
done
for model in $MODELS; do
    case "$model" in r2|r3|r4) ;; *) fail "unknown model '$model'; use r2, r3 or r4" ;; esac
done
# Where the experiments read the per-fold calibrations; exported, so the jobs
# (training, evaluation) read the same folder.
export ADM1_BENCHMARKS_DIR="${ADM1_BENCHMARKS_DIR:-$PWD/runs/benchmarks}"
# e.g. fold_fast_narrow_tan_no_ph fits no pH, which R4 does not produce; every
# experiment needs ADM1_BENCHMARKS_DIR to hold every fold's calibration.
ADM1_VENV="$ADM1_VENV" bash -c '
    source scripts/hpc/activate_env.sh || exit 1
    python -c "import sys; from adm1_package.experiments.training import check_experiments; check_experiments(sys.argv[1].split(), sys.argv[2].split())" "$1" "$2"
' _ "$EXPERIMENTS" "$MODELS" || fail "an experiment does not apply to every model listed, or its per-fold calibrations are missing (see above)"
if [[ -n "$HYPERPARAMETERS" ]]; then
    [[ -f "$HYPERPARAMETERS" ]] || fail "$HYPERPARAMETERS not found"
    HYPERPARAMETERS="$(cd "$(dirname "$HYPERPARAMETERS")" && pwd)/$(basename "$HYPERPARAMETERS")"
fi
FOLDS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["k_folds"])' configs/split.json 2>/dev/null || echo 5)"
if [[ -n "$FOLD_HYPERPARAMETERS" ]]; then
    [[ -z "$HYPERPARAMETERS" ]] || fail "--hyperparameters and --fold-hyperparameters exclude each other"
    [[ "$FOLD_HYPERPARAMETERS" == *"{fold}"* ]] || fail "--fold-hyperparameters names no {fold}"
    # Relative to the repository, where the jobs run.
    [[ "$FOLD_HYPERPARAMETERS" == /* ]] || FOLD_HYPERPARAMETERS="$PWD/$FOLD_HYPERPARAMETERS"
    for model in $MODELS; do
        for fold in $(seq 1 "$FOLDS"); do
            selection="${FOLD_HYPERPARAMETERS//\{model\}/$model}"
            selection="${selection//\{fold\}/$fold}"
            [[ -f "$selection" ]] || fail "$selection not found: no selection for $model fold $fold"
        done
    done
fi
# SLURM opens the output file before the script runs; without the directory the
# job's whole output is lost.
mkdir -p logs

# A request above the partition's limit is accepted and then held forever with
# reason PartitionTimeLimit, which looks like an ordinary queue wait.
_seconds() {
    local spec="$1" days=0 rest h=0 m=0 s=0
    case "${spec,,}" in infinite|unlimited|"") echo 999999999; return ;; esac
    [[ "$spec" == *-* ]] && { days="${spec%%-*}"; rest="${spec#*-}"; } || rest="$spec"
    local -a parts; IFS=':' read -ra parts <<< "$rest"
    case "${#parts[@]}" in
        3) h="${parts[0]}"; m="${parts[1]}"; s="${parts[2]}" ;;
        2) if [[ "$spec" == *-* ]]; then h="${parts[0]}"; m="${parts[1]}"; else m="${parts[0]}"; s="${parts[1]}"; fi ;;
        1) if [[ "$spec" == *-* ]]; then h="${parts[0]}"; else m="${parts[0]}"; fi ;;
    esac
    echo $(( 10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
if command -v sinfo >/dev/null 2>&1; then
    [[ -n "$PARTITION" ]] || PARTITION="$(sinfo -h -o '%P' 2>/dev/null | grep '\*' | tr -d '*' | head -1 || true)"
    if [[ -n "$PARTITION" ]]; then
        MAXTIME="$(sinfo -h -p "$PARTITION" -o '%l' 2>/dev/null | head -1 | tr -d ' ' || true)"
        if [[ -n "$MAXTIME" ]] && (( $(_seconds "$WALLTIME") > $(_seconds "$MAXTIME") )); then
            fail "--time $WALLTIME exceeds partition '$PARTITION' (limit $MAXTIME)"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
read -ra EXPERIMENT_LIST <<< "$EXPERIMENTS"
read -ra MODEL_LIST <<< "$MODELS"
read -ra SEED_LIST <<< "$SEEDS"
SEED_COUNT=$(( ${#SEED_LIST[@]} > 0 ? ${#SEED_LIST[@]} : 1 ))
TASKS=$(( ${#EXPERIMENT_LIST[@]} * ${#MODEL_LIST[@]} * SEED_COUNT ))
WORKERS=$(( CPUS < FOLDS ? CPUS : FOLDS ))

# Workstation timings per fold and epoch: R4 ~30 s, R3 ~40 s, R2 ~55 s. Folds run
# in parallel, so a task takes about epochs x seconds x (folds / workers), plus
# evaluation.
estimate() { echo $(( EPOCHS * $1 * ( (FOLDS + WORKERS - 1) / WORKERS ) / 3600 + 1 )); }

cat <<INFO
================================================================
experiments : ${EXPERIMENTS}
models      : ${MODELS}
seeds       : ${SEEDS:-the default of train_cv.py}
array       : ${TASKS} tasks (0-$(( TASKS - 1 )))
epochs      : ${EPOCHS} per fold, ${FOLDS} folds, ${WORKERS} at a time
resources   : ${CPUS} cores x ${MEM_PER_CPU} per task, ${WALLTIME}, partition ${PARTITION:-(default)}${MAXTIME:+ (max $MAXTIME)}
recipe      : ${FOLD_HYPERPARAMETERS:+per fold: }${FOLD_HYPERPARAMETERS:-${HYPERPARAMETERS:-default hyperparameters per variant}}
env         : ${ADM1_VENV}
data        : ${DATA_DIR}
benchmarks  : ${ADM1_BENCHMARKS_DIR} (per-fold calibrations, where an experiment uses them)
runs        : ${ADM1_RUN_DIR:-${SCRATCH:-$PWD}/runs}/<model>/<experiment>${SEEDS:+_seed<N>}_<job id>
rough time per task: r2 ~$(estimate 55) h, r3 ~$(estimate 40) h, r4 ~$(estimate 30) h
================================================================
INFO

# ALL carries the environment, which is how the lists (they contain spaces, which
# --export does not take reliably) and ADM1_DATA_DIR / ADM1_RUN_DIR reach the
# jobs. The rest is listed as well, as in submit_tune.sh, so a cluster default
# of --export=NONE cannot drop it.
export ADM1_EXPERIMENTS="$EXPERIMENTS" ADM1_MODELS="$MODELS" ADM1_SEEDS="$SEEDS"
if [[ -n "$HYPERPARAMETERS" ]]; then export HYPERPARAMETERS; else unset HYPERPARAMETERS; fi
if [[ -n "$FOLD_HYPERPARAMETERS" ]]; then export FOLD_HYPERPARAMETERS; else unset FOLD_HYPERPARAMETERS; fi
EXPORTS="ALL,EPOCHS=$EPOCHS,ADM1_VENV=$ADM1_VENV,ADM1_REPO=$REPO_ROOT,ADM1_BENCHMARKS_DIR=$ADM1_BENCHMARKS_DIR"
ARRAY_ARGS=(--parsable --array="0-$(( TASKS - 1 ))" --time="$WALLTIME" --cpus-per-task="$CPUS"
            --mem-per-cpu="$MEM_PER_CPU" ${PARTITION:+--partition="$PARTITION"} --export="$EXPORTS")

if (( DRY_RUN )); then
    echo "sbatch ${ARRAY_ARGS[*]} scripts/hpc/slurm_experiments.sh"
    echo "sbatch --parsable --dependency=afterany:<array job> ${PARTITION:+--partition=$PARTITION} \\"
    echo "    --export=$EXPORTS,ADM1_ARRAY_JOB=<array job> scripts/hpc/slurm_combine.sh"
    exit 0
fi

ARRAY_JOB="$(sbatch "${ARRAY_ARGS[@]}" scripts/hpc/slurm_experiments.sh)"
ARRAY_JOB="${ARRAY_JOB%%;*}"
MERGE_JOB="$(sbatch --parsable --dependency="afterany:$ARRAY_JOB" ${PARTITION:+--partition="$PARTITION"} \
    --export="$EXPORTS,ADM1_ARRAY_JOB=$ARRAY_JOB" scripts/hpc/slurm_combine.sh)"
MERGE_JOB="${MERGE_JOB%%;*}"

cat <<DONE
submitted : array ${ARRAY_JOB} (${TASKS} tasks), merge ${MERGE_JOB} after it
watch     : squeue -u \$USER;  tail -f logs/adm1-exp-${ARRAY_JOB}_0.out
results   : runs/<model>/<experiment>_${ARRAY_JOB}/  and  evaluations/<experiment>_${ARRAY_JOB}/
DONE
