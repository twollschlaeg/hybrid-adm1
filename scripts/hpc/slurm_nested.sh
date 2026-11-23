#!/bin/bash
# One task of a nested hyperparameter search: trials on one outer fold of one variant.
#
# Submit with scripts/hpc/submit_nested.sh, which prepares the inner splits, checks
# them and sizes the array. With the outer folds in ADM1_OUTER_FOLDS (n of them),
# task i searches outer fold ADM1_OUTER_FOLDS[i % n] as worker i / n of that fold's
# study. All tasks given one outer fold share its study through the journal file in
#   $ADM1_RUN_DIR/<model>/tuning_<name>_outer<k>/
# and every trial trains on that outer fold's inner folds only (--split and
# --benchmarks from scripts/make_nested.py), so the reactors the outer fold is scored
# on never enter its search. Worker 0 of each study starts from the default values.
#
#SBATCH --job-name=adm1-nested
#SBATCH --output=logs/adm1-nested-%A_%a.out
#SBATCH --error=logs/adm1-nested-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
# Per CPU, as in slurm_tune.sh: one worker process per core, each importing jax.
#SBATCH --mem-per-cpu=4G
#
# Uncomment and set for your cluster:
##SBATCH --partition=standard
##SBATCH --account=your-project

set -euo pipefail

# sbatch runs a copy of this script from its spool directory, so anchor on the
# submit directory (the repository root), as slurm_tune.sh does.
ADM1_REPO="${ADM1_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ADM1_REPO"
# Here-strings and Python need temporary files; the node's /tmp may be full.
source scripts/hpc/job_tmpdir.sh
source scripts/hpc/activate_env.sh

: "${ADM1_MODEL:?set ADM1_MODEL to r2, r3 or r4}"
: "${ADM1_EXPERIMENT:?set ADM1_EXPERIMENT to the experiment file the search trains}"
: "${ADM1_NESTED_DIR:?set ADM1_NESTED_DIR to the directory scripts/make_nested.py wrote}"
read -ra OUTER_FOLDS <<< "${ADM1_OUTER_FOLDS:-1 2 3 4 5}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUTER="${OUTER_FOLDS[$(( TASK_ID % ${#OUTER_FOLDS[@]} ))]}"
# Worker 0 of every outer fold's study enqueues the default values, and the worker
# number offsets that study's sampler seed, so it counts the tasks of one outer fold.
WORKER=$(( TASK_ID / ${#OUTER_FOLDS[@]} ))
# The same for every task of the array, so the tasks of one outer fold share a study.
NAME="${ADM1_NESTED_NAME:-nested-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"
TUNING_NAME="${NAME}_outer${OUTER}"
SPLIT="$ADM1_NESTED_DIR/outer_$OUTER/split.json"
BENCHMARKS="$ADM1_NESTED_DIR/outer_$OUTER/benchmarks"

export ADM1_RUN_DIR="${ADM1_RUN_DIR:-${SCRATCH:-$PWD}/runs}"
export ADM1_DATA_DIR="${ADM1_DATA_DIR:-$PWD/data}"
export ADM1_PLATFORM=cpu
# One thread per process, one process per core (see slurm_tune.sh).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

for path in "$SPLIT" "$BENCHMARKS"; do
    [[ -e "$path" ]] || { echo "ERROR: $path is missing; run scripts/make_nested.py" >&2; exit 1; }
done

# A trial trains every inner fold at once, one process each, and the task runs as
# many trials at once as its cores hold whole.
CORES="${SLURM_CPUS_PER_TASK:-4}"
FOLDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["k_folds"])' "$SPLIT")"
PARALLEL_TRIALS=$(( CORES / FOLDS ))
(( PARALLEL_TRIALS >= 1 )) || PARALLEL_TRIALS=1
FOLD_WORKERS=$(( PARALLEL_TRIALS * FOLDS ))
(( FOLD_WORKERS <= CORES )) || FOLD_WORKERS="$CORES"
SEARCH="$ADM1_RUN_DIR/$ADM1_MODEL/tuning_$TUNING_NAME"

cat <<INFO
================================================================
task       : ${TASK_ID} of job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}, worker ${WORKER} of outer fold ${OUTER}
node       : $(hostname)
model      : ${ADM1_MODEL}, ${ADM1_EXPERIMENT}
inner split: ${SPLIT}
anchors    : ${BENCHMARKS}
search     : ${SEARCH}
trials     : ${N_TRIALS:-12} x ${EPOCHS:-200} epochs, ${PARALLEL_TRIALS} at once on ${CORES} cores (${FOLDS} inner folds)
python     : $(command -v python)
================================================================
INFO

# Plain python, not srun: since SLURM 22.05 srun no longer inherits --cpus-per-task
# and would bind the fold workers to one core.
python scripts/tune_hyperparameters.py \
    --model "$ADM1_MODEL" \
    --experiment "$ADM1_EXPERIMENT" \
    --split "$SPLIT" \
    --benchmarks "$BENCHMARKS" \
    --n-trials "${N_TRIALS:-12}" \
    --epochs "${EPOCHS:-200}" \
    --fold-workers "$FOLD_WORKERS" \
    --parallel-trials "$PARALLEL_TRIALS" \
    --worker-id "$WORKER" \
    --storage journal \
    --tuning-name "$TUNING_NAME" \
    --study-name "nested_${ADM1_MODEL}_outer${OUTER}" \
    --start-from-defaults

# The selection so far, off scratch; the collection job after the array writes the
# final one from every task's trials.
KEEP="$PWD/runs/$ADM1_MODEL/tuning_$TUNING_NAME"
mkdir -p "$KEEP"
for file in best_hyperparameters.json trials.csv; do
    if [[ -f "$SEARCH/$file" && ! "$SEARCH/$file" -ef "$KEEP/$file" ]]; then
        cp "$SEARCH/$file" "$KEEP/$file"
    fi
done
echo "selection so far: $KEEP/best_hyperparameters.json"
