#!/bin/bash
# Merge the variants of every experiment (and seed) of one array job, once all its
# tasks ended.
#
# submit_experiments.sh queues this behind the array with afterany, so an
# experiment is merged from whichever variants finished. The results (not the
# checkpoints) are copied from the run directory to ./runs, and the
# merged tables go to ./evaluations/<experiment>_<job>/.
#
#SBATCH --job-name=adm1-merge
#SBATCH --output=logs/adm1-merge-%j.out
#SBATCH --error=logs/adm1-merge-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

ADM1_REPO="${ADM1_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ADM1_REPO"
# Here-strings and Python need temporary files; the node's /tmp may be full.
source scripts/hpc/job_tmpdir.sh
source scripts/hpc/activate_env.sh

: "${ADM1_ARRAY_JOB:?set ADM1_ARRAY_JOB to the job id of the experiment array}"
read -ra EXPERIMENTS <<< "${ADM1_EXPERIMENTS:-fold_fast_narrow_tan}"
read -ra MODELS <<< "${ADM1_MODELS:-r2 r3 r4}"
read -ra SEEDS <<< "${ADM1_SEEDS:-}"
(( ${#SEEDS[@]} )) || SEEDS=("")
RUN_DIR="${ADM1_RUN_DIR:-${SCRATCH:-$PWD}/runs}"

for experiment in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        name="${experiment}${seed:+_seed$seed}_${ADM1_ARRAY_JOB}"
        results=()
        for model in "${MODELS[@]}"; do
            run="$RUN_DIR/$model/$name"
            if [[ ! -f "$run/results/cv_fold_metrics.csv" ]]; then
                echo "missing: $run/results -- that task failed or did not finish"
                continue
            fi
            results+=("$run/results")
            keep="$PWD/runs/$model/$name"
            if [[ "$(cd "$run" && pwd -P)" != "$(mkdir -p "$keep" && cd "$keep" && pwd -P)" ]]; then
                # A run trained with a selection per fold has fold_hyperparameters.json
                # instead of hyperparameters.json; cp copies whichever exists.
                cp -r "$run/results" "$run/hyperparameters.json" \
                    "$run/fold_hyperparameters.json" "$run/metadata.json" "$keep/" 2>/dev/null || true
            fi
        done
        if (( ${#results[@]} )); then
            python scripts/combine_results.py --results "${results[@]}" \
                --output "evaluations/$name"
            echo "merged ${name}: ${#results[@]} variant(s) -> evaluations/$name"
        fi
    done
done
