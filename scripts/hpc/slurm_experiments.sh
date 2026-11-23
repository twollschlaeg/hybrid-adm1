#!/bin/bash
# One experiment for one variant and seed per array task: train the folds, evaluate.
#
# Submit with scripts/hpc/submit_experiments.sh, which checks the environment,
# sizes the array and adds the job that merges the results. With n_seeds seeds in
# ADM1_SEEDS (none: one task with the default seed), task i runs seed
# ADM1_SEEDS[i % n_seeds] of experiment ADM1_EXPERIMENTS[j / n_models] on model
# ADM1_MODELS[j % n_models], where j = i / n_seeds.
#
#SBATCH --job-name=adm1-exp
#SBATCH --output=logs/adm1-exp-%A_%a.out
#SBATCH --error=logs/adm1-exp-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=5
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

read -ra EXPERIMENTS <<< "${ADM1_EXPERIMENTS:-fold_fast_narrow_tan}"
read -ra MODELS <<< "${ADM1_MODELS:-r2 r3 r4}"
read -ra SEEDS <<< "${ADM1_SEEDS:-}"
(( ${#SEEDS[@]} )) || SEEDS=("")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$(( TASK_ID % ${#SEEDS[@]} ))]}"
PAIR=$(( TASK_ID / ${#SEEDS[@]} ))
EXPERIMENT="${EXPERIMENTS[$(( PAIR / ${#MODELS[@]} ))]}"
MODEL="${MODELS[$(( PAIR % ${#MODELS[@]} ))]}"

# Scratch is faster and roomier for checkpoints; the merge job copies the results
# back to the repository.
export ADM1_RUN_DIR="${ADM1_RUN_DIR:-${SCRATCH:-$PWD}/runs}"
export ADM1_DATA_DIR="${ADM1_DATA_DIR:-$PWD/data}"
export ADM1_PLATFORM=cpu

# One fold per core, never more workers than folds.
FOLDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["k_folds"])' configs/split.json)"
CORES="${SLURM_CPUS_PER_TASK:-5}"
export FOLD_WORKERS="$(( CORES < FOLDS ? CORES : FOLDS ))"

RUN_NAME="${EXPERIMENT}${SEED:+_seed$SEED}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"

cat <<INFO
================================================================
task       : ${TASK_ID} of job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
node       : $(hostname)
experiment : ${EXPERIMENT} (configs/experiments/${EXPERIMENT}.json)
model      : ${MODEL}
seed       : ${SEED:-default}
run        : ${ADM1_RUN_DIR}/${MODEL}/${RUN_NAME}
benchmarks : ${ADM1_BENCHMARKS_DIR:-${ADM1_RUN_DIR}/benchmarks} (read by per-fold experiments)
epochs     : ${EPOCHS:-100}, ${FOLD_WORKERS} fold workers on ${CORES} cores
python     : $(command -v python)
================================================================
INFO

# Plain bash, not srun: since SLURM 22.05 srun no longer inherits --cpus-per-task
# and would bind the fold workers to one core.
SEED="$SEED" bash scripts/run_experiment.sh "$EXPERIMENT" "$MODEL" "$RUN_NAME"
