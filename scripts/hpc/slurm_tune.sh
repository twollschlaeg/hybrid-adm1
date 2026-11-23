#!/bin/bash
# Cross-validated hyperparameter search on a SLURM cluster.
#
# Submit with scripts/hpc/submit_tune.sh, which creates the log directory and
# checks the environment first. To submit by hand:
#
#   mkdir -p logs
#   sbatch --array=0-2  scripts/hpc/slurm_tune.sh     # one task per variant
#   sbatch --array=0-11 scripts/hpc/slurm_tune.sh     # four tasks per variant
#
# Tasks map to variants round-robin, so tasks 0/3/6/9 all work on R2 and
# contribute trials to one shared study, which Optuna coordinates through the
# storage backend. Each task offsets the sampler seed by its array index, so
# they do not all open with the same configurations.
#
#SBATCH --job-name=adm1-tune
#SBATCH --output=logs/adm1-tune-%A_%a.out
#SBATCH --error=logs/adm1-tune-%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
# Per CPU, NOT per node: this script runs one worker process per reserved core
# and each one imports jax and torch independently (~400 MB before it computes
# anything), so the requirement scales with --cpus-per-task. A fixed --mem here
# would keep the 16 GB that suits the 8-core default and leave a larger run
# OUT_OF_MEMORY within seconds. submit_tune.sh overrides this.
#SBATCH --mem-per-cpu=4G
#
# Uncomment and set for your cluster:
##SBATCH --partition=standard
##SBATCH --account=your-project
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.org

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# SLURM exports the submitting shell's environment, but rather than trust that
# the PATH came across intact, activate the venv explicitly from ADM1_VENV.
# activate_env.sh sources bin/activate and then checks that `python` really is
# the venv's, comparing device and inode rather than strings.
#
# Do NOT use "$(dirname "${BASH_SOURCE[0]}")" to find it. sbatch COPIES this
# script into the spool directory and runs the copy, so inside the job
# BASH_SOURCE[0] is /var/spool/slurm/d/job<id>/slurm_script and its directory
# holds nothing else -- sourcing a sibling path there fails with "No such file
# or directory" after the queue wait. Anchor on the submit directory instead:
# sbatch starts the job with cwd set to wherever it was submitted from, and
# submit_tune.sh always submits from the repository root. ADM1_REPO overrides
# that, and is what submit_tune.sh passes explicitly.
ADM1_REPO="${ADM1_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ADM1_REPO"

if [[ ! -f scripts/hpc/activate_env.sh ]]; then
    echo "ERROR: $ADM1_REPO does not look like the repository root --" >&2
    echo "       scripts/hpc/activate_env.sh is not there." >&2
    echo "       Submit via scripts/hpc/submit_tune.sh, or set ADM1_REPO." >&2
    exit 1
fi
# Here-strings and Python need temporary files; the node's /tmp may be full.
source scripts/hpc/job_tmpdir.sh
source scripts/hpc/activate_env.sh

# ---------------------------------------------------------------------------
# Which variant this task works on
# ---------------------------------------------------------------------------
MODELS=(r2 r3 r4)
# Default to 0 so the script also runs as a plain (non-array) job; under
# `set -u` an unset SLURM_ARRAY_TASK_ID in an arithmetic expression aborts.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${ADM1_MODEL:-${MODELS[$((TASK_ID % ${#MODELS[@]}))]}}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Scratch is much faster than home for checkpoint writes, and usually the only
# filesystem with enough quota. Runs are recoverable, so scratch purging is an
# acceptable risk -- copy best_hyperparameters.json to home when the job ends.
export ADM1_RUN_DIR="${ADM1_RUN_DIR:-${SCRATCH:-$PWD}/runs}"
export ADM1_DATA_DIR="${ADM1_DATA_DIR:-$PWD/data}"
export ADM1_PLATFORM=cpu

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
# JAX and the BLAS libraries each try to claim every visible core. With several
# fold workers in one job step they then oversubscribe and run slower than a
# single worker would. One thread per process, one process per reserved core.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

# Optuna runs `--parallel-trials` trials at once in threads, and EVERY one of
# them submits its folds to the same ProcessPoolExecutor sized by
# `--fold-workers`. The pool, not the thread count, is what actually consumes
# cores -- so the two have to be sized together. Capping the pool at FOLDS
# would run exactly one trial's worth of work at a time no matter how large the
# allocation: a 30-core task would keep 5 cores busy and idle 25.
#
# One process per core and one thread per process (see above), so:
#     parallel trials = cores / folds
#     pool size       = parallel trials * folds
CORES="${SLURM_CPUS_PER_TASK:-4}"
FOLDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["k_folds"])' configs/split.json)"

PARALLEL_TRIALS="${PARALLEL_TRIALS:-$(( CORES / FOLDS ))}"
(( PARALLEL_TRIALS >= 1 )) || PARALLEL_TRIALS=1
FOLD_WORKERS="${FOLD_WORKERS:-$(( PARALLEL_TRIALS * FOLDS ))}"
# Never start more worker processes than there are cores to run them on.
(( FOLD_WORKERS <= CORES )) || FOLD_WORKERS="$CORES"

# ---------------------------------------------------------------------------
# Optuna storage
# ---------------------------------------------------------------------------
# "journal" selects Optuna's append-only file backend. SQLite on a network
# filesystem relies on POSIX locks that NFS and Lustre implement loosely, so
# concurrent array tasks hit "database is locked" or corrupt the study. Set
# ADM1_OPTUNA_STORAGE to a postgresql:// URL if the cluster offers one.
STORAGE="${ADM1_OPTUNA_STORAGE:-journal}"

# ---------------------------------------------------------------------------
# Where this search writes
# ---------------------------------------------------------------------------
# One directory per submission, so a new search does not overwrite the previous
# one's best_hyperparameters.json and trials.csv.
#
# SLURM_ARRAY_JOB_ID is the same for every task of an array and comes first for
# that reason: the tasks share one Optuna study through a file in this
# directory, so a per-task name would split the study into private copies and
# each task would explore alone. SLURM_JOB_ID differs per task and is only the
# fallback for a job submitted without an array.
#
# Passed to the CLI rather than letting it derive its own, so the name the shell
# copies from below cannot drift from the name Python wrote to.
TUNING_NAME="tuning_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
TUNING_DIR="$ADM1_RUN_DIR/$MODEL/$TUNING_NAME"
mkdir -p "$TUNING_DIR"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
# Fail in the first seconds rather than after the queue wait plus a partial run.
for campaign in LaRA-temp LaRA-substrate rice-bran; do
    if [[ ! -d "$ADM1_DATA_DIR/$campaign" ]]; then
        echo "ERROR: $ADM1_DATA_DIR/$campaign is missing (see data/README.md)" >&2
        exit 1
    fi
done

if [[ ! -f configs/split.json ]]; then
    echo "ERROR: configs/split.json is missing. Run scripts/make_splits.py and" >&2
    echo "       commit the result before submitting -- deriving the split here" >&2
    echo "       would give each task a different fold assignment." >&2
    exit 1
fi

cat <<INFO
================================================================
task     : ${TASK_ID} of job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
node     : $(hostname)
model    : ${MODEL}
env      : ${ADM1_VENV}
activated: ${ADM1_ACTIVATION}
python   : $(command -v python)
run dir  : ${ADM1_RUN_DIR}
data dir : ${ADM1_DATA_DIR}
storage  : ${STORAGE}
search   : ${TUNING_DIR}
cores    : ${CORES}, parallel trials: ${PARALLEL_TRIALS}, fold workers: ${FOLD_WORKERS}, folds: ${FOLDS}
trials   : ${N_TRIALS:-20} x ${EPOCHS:-40} epochs
================================================================
INFO

# Plain `python`, not `srun`: since SLURM 22.05 srun no longer inherits
# --cpus-per-task from sbatch, so it would bind this task to one core and the
# fold workers would then fight over it. This is a single-node process pool,
# not an MPI job, so srun buys nothing here.
# Model flags (see scripts/tune_hyperparameters.py --help), e.g.
#   ADM1_UDE_ARGS="--experiment configs/experiments/fold_fast_narrow_tan.json"
#   ADM1_UDE_ARGS="--physics arrhenius nh3_inhibition"
# Every task of one array gets the same flags, and each submission gets its
# own study directory, so different models never share a study.
read -ra UDE_ARGS <<< "${ADM1_UDE_ARGS:-}"

python scripts/tune_hyperparameters.py \
    --model "$MODEL" \
    --n-trials "${N_TRIALS:-20}" \
    --epochs "${EPOCHS:-40}" \
    --fold-workers "$FOLD_WORKERS" \
    --parallel-trials "$PARALLEL_TRIALS" \
    --worker-id "$TASK_ID" \
    --storage "$STORAGE" \
    --tuning-name "$TUNING_NAME" \
    --study-name "adm1_${MODEL}_cv" \
    ${UDE_ARGS[@]+"${UDE_ARGS[@]}"}

# Keep the selected values off scratch, which gets purged.
BEST="$TUNING_DIR/best_hyperparameters.json"
if [[ -f "$BEST" ]]; then
    mkdir -p "$PWD/runs/$MODEL/$TUNING_NAME"
    cp "$BEST" "$PWD/runs/$MODEL/$TUNING_NAME/best_hyperparameters.json"
    echo "copied $BEST -> $PWD/runs/$MODEL/$TUNING_NAME/"
fi

# Next, on the workstation or as a follow-up job:
#
#   python scripts/train_cv.py --model $MODEL --epochs 100 \
#       --hyperparameters runs/$MODEL/$TUNING_NAME/best_hyperparameters.json
#   python scripts/evaluate.py --model $MODEL --run runs/$MODEL/<run-name>
