#!/bin/bash
# Collect a nested search once every task of its array has ended, killed or not.
#
# submit_nested.sh queues this behind the array with afterany. For every outer fold
# it writes the selection from all trials in that fold's study (tune_hyperparameters.py
# --n-trials 0 trains nothing), copies it and the trial table off scratch into
#   ./runs/<model>/tuning_<name>_outer<k>/
# and tabulates the selections in ./runs/<model>/<name>_selections.csv.
#
#SBATCH --job-name=adm1-nested-collect
#SBATCH --output=logs/adm1-nested-collect-%j.out
#SBATCH --error=logs/adm1-nested-collect-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

ADM1_REPO="${ADM1_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ADM1_REPO"
# Here-strings and Python need temporary files; the node's /tmp may be full.
source scripts/hpc/job_tmpdir.sh
source scripts/hpc/activate_env.sh

: "${ADM1_MODEL:?set ADM1_MODEL to r2, r3 or r4}"
: "${ADM1_EXPERIMENT:?set ADM1_EXPERIMENT to the experiment file the search trained}"
: "${ADM1_NESTED_DIR:?set ADM1_NESTED_DIR to the directory scripts/make_nested.py wrote}"
# The array's own job id is not this job's, so the name has to be given.
: "${ADM1_NESTED_NAME:?set ADM1_NESTED_NAME to the name of the searches, e.g. nested-<array job id>}"
read -ra OUTER_FOLDS <<< "${ADM1_OUTER_FOLDS:-1 2 3 4 5}"
export ADM1_RUN_DIR="${ADM1_RUN_DIR:-${SCRATCH:-$PWD}/runs}"
export ADM1_PLATFORM=cpu

missing=()
for outer in "${OUTER_FOLDS[@]}"; do
    name="${ADM1_NESTED_NAME}_outer${outer}"
    # The search's epochs and seed: a selection is written with the epochs it was
    # searched with, and a study refuses settings other than its own.
    if ! python scripts/tune_hyperparameters.py \
            --model "$ADM1_MODEL" \
            --experiment "$ADM1_EXPERIMENT" \
            --split "$ADM1_NESTED_DIR/outer_$outer/split.json" \
            --benchmarks "$ADM1_NESTED_DIR/outer_$outer/benchmarks" \
            --n-trials 0 \
            --epochs "${EPOCHS:-200}" \
            --storage journal \
            --tuning-name "$name" \
            --study-name "nested_${ADM1_MODEL}_outer${outer}"; then
        echo "outer fold $outer: no completed trial to select from"
        missing+=("$outer")
        continue
    fi
    search="$ADM1_RUN_DIR/$ADM1_MODEL/tuning_$name"
    keep="$PWD/runs/$ADM1_MODEL/tuning_$name"
    mkdir -p "$keep"
    for file in best_hyperparameters.json trials.csv; do
        if [[ -f "$search/$file" && ! "$search/$file" -ef "$keep/$file" ]]; then
            cp "$search/$file" "$keep/$file"
        fi
    done
    echo "outer fold $outer: $keep/best_hyperparameters.json"
done

python scripts/nested_summary.py --models "$ADM1_MODEL" \
    --pattern "runs/{model}/tuning_${ADM1_NESTED_NAME}_outer{fold}/best_hyperparameters.json" \
    --output "runs/$ADM1_MODEL/${ADM1_NESTED_NAME}_selections.csv" || true

if (( ${#missing[@]} )); then
    echo "outer folds without a selection: ${missing[*]} -- resubmit them with --name $ADM1_NESTED_NAME"
    exit 1
fi
