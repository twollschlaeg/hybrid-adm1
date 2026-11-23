#!/bin/bash
# Submit a nested hyperparameter search for one variant: an array that searches every
# outer fold on inner folds of its training reactors, one study per outer fold, then a
# job that collects the selections once the array has ended.
#
#   bash scripts/hpc/submit_nested.sh --model r4 --name nested-fast --dry-run
#   bash scripts/hpc/submit_nested.sh --model r4 --name nested-fast
#   bash scripts/hpc/submit_nested.sh --model r2 --name nested-fast --tasks-per-fold 3
#
# Options:
#   --model r2|r3|r4      the variant (required)
#   --experiment NAME     configs/experiments/NAME.json, the model the search trains
#                         (default: the final models', fold_fast_narrow_tan for r4 and
#                         fold_fast_narrow_tan_no_ph for r2 and r3)
#   --benchmarks DIR      the outer folds' per-fold calibrations
#                         (default: runs/benchmarks-smoothstep-fast)
#   --nested DIR          where the inner splits and their calibrations go
#                         (default: runs/nested-<DIR of --benchmarks without "benchmarks-">)
#   --name NAME           the searches are tuning_<NAME>_outer<k> (default: nested-<array
#                         job id>). The same name for every variant lets one pattern find
#                         all selections; an earlier name adds trials to its searches.
#   --outer-folds "..."   outer folds to search (default: every fold of the split)
#   --inner-folds N       inner folds per outer fold (default 4)
#   --tasks-per-fold N    array tasks sharing each outer fold's study (default 2)
#   --trials N            trials each task runs (default 12)
#   --epochs N            epochs per trial and inner fold (default 200, as the final models)
#   --time HH:MM:SS       wall clock per task (default 24:00:00)
#   --partition NAME      partition (default: the cluster's default)
#   --cpus N              cores per task (default 16); a trial takes one per inner fold
#   --mem-per-cpu M       memory per core (default 4G)
#   --dry-run             prepare and check, print the plan, submit nothing
#
# Needs ADM1_VENV (see scripts/hpc/README.md) and the outer calibrations under
# --benchmarks, copied from the machine that ran scripts/calibrate/calibrate_folds.py.
# The inner splits and calibrations are prepared here, before anything is queued
# (scripts/make_nested.py); they hold reactor data and stay under runs/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL=""
EXPERIMENT=""
BENCHMARKS="runs/benchmarks-smoothstep-fast"
NESTED=""
NAME=""
OUTER_FOLDS=""
INNER_FOLDS=4
TASKS_PER_FOLD=2
TRIALS=12
EPOCHS=200
WALLTIME="24:00:00"
PARTITION=""
CPUS=16
MEM_PER_CPU=4G
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2 ;;
        --experiment)     EXPERIMENT="$2"; shift 2 ;;
        --benchmarks)     BENCHMARKS="$2"; shift 2 ;;
        --nested)         NESTED="$2"; shift 2 ;;
        --name)           NAME="$2"; shift 2 ;;
        --outer-folds)    OUTER_FOLDS="$2"; shift 2 ;;
        --inner-folds)    INNER_FOLDS="$2"; shift 2 ;;
        --tasks-per-fold) TASKS_PER_FOLD="$2"; shift 2 ;;
        --trials)         TRIALS="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --time)           WALLTIME="$2"; shift 2 ;;
        --partition)      PARTITION="$2"; shift 2 ;;
        --cpus)           CPUS="$2"; shift 2 ;;
        --mem-per-cpu)    MEM_PER_CPU="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
                       "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
case "$MODEL" in
    r2|r3|r4) ;;
    "") fail "--model is required: r2, r3 or r4" ;;
    *) fail "unknown model '$MODEL'; use r2, r3 or r4" ;;
esac
if [[ -z "$EXPERIMENT" ]]; then
    if [[ "$MODEL" == r4 ]]; then EXPERIMENT=fold_fast_narrow_tan; else EXPERIMENT=fold_fast_narrow_tan_no_ph; fi
fi
EXPERIMENT_FILE="configs/experiments/$EXPERIMENT.json"
[[ -f "$EXPERIMENT_FILE" ]] || fail "$EXPERIMENT_FILE not found"
for number in "$INNER_FOLDS" "$TASKS_PER_FOLD" "$TRIALS" "$EPOCHS" "$CPUS"; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || fail "'$number' is not a positive whole number"
done
# It becomes part of directory and study names on the cluster.
[[ -z "$NAME" || "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--name takes letters, digits, '.', '_' and '-' only"

# ---------------------------------------------------------------------------
# Checks that otherwise only fail after the queue wait
# ---------------------------------------------------------------------------
command -v sbatch >/dev/null 2>&1 || fail "sbatch not found; is this a login node?"
[[ -n "${ADM1_VENV:-}" ]] || fail "ADM1_VENV is not set; build a venv with bash scripts/hpc/setup_env.sh"
[[ -d "$ADM1_VENV" ]] || fail "ADM1_VENV=$ADM1_VENV does not exist"
[[ -f configs/split.json ]] || fail "configs/split.json is missing. Run: python scripts/make_splits.py"
[[ -f "$BENCHMARKS/split.json" ]] || fail "$BENCHMARKS holds no per-fold calibrations (no split.json);
     copy them from the machine that ran scripts/calibrate/calibrate_folds.py"

# A command as the batch job runs it: activated in a non-interactive shell.
in_env() {
    ADM1_VENV="$ADM1_VENV" bash -c 'source scripts/hpc/activate_env.sh || exit 1; "$@"' _ "$@"
}

echo "checking activation as a batch job would see it..."
in_env python -c "
import jax, optuna, equinox, diffrax, orbax.checkpoint
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
assert jnp.zeros(1).dtype == jnp.float64, 'float64 unavailable'
" || fail "the environment does not come up in a non-interactive shell (see above)"

BENCHMARKS="$(cd "$BENCHMARKS" && pwd)"
if [[ -z "$NESTED" ]]; then
    base="$(basename "$BENCHMARKS")"
    NESTED="runs/nested-${base#benchmarks-}"
fi
echo "preparing the inner splits and their calibrations in $NESTED..."
in_env python scripts/make_nested.py --benchmarks "$BENCHMARKS" --output "$NESTED" \
    --inner-folds "$INNER_FOLDS" || fail "could not prepare the inner splits (see above)"
NESTED="$(cd "$NESTED" && pwd)"

FOLD_COUNT="$(in_env python -c 'import json,sys; print(json.load(open(sys.argv[1]))["k_folds"])' configs/split.json)"
[[ -n "$OUTER_FOLDS" ]] || OUTER_FOLDS="$(seq -s ' ' 1 "$FOLD_COUNT")"
read -ra OUTER_LIST <<< "$OUTER_FOLDS"
(( ${#OUTER_LIST[@]} )) || fail "--outer-folds lists no fold"
for outer in "${OUTER_LIST[@]}"; do
    [[ "$outer" =~ ^[0-9]+$ ]] && (( outer >= 1 && outer <= FOLD_COUNT )) \
        || fail "outer fold '$outer' does not exist; the split has $FOLD_COUNT"
    # The experiment applies to the variant, and every inner fold finds a calibration
    # made with the experiment's physics, bounds and forcing.
    in_env python -c "
import sys
from pathlib import Path
from adm1_package.experiments.splits import load_split
from adm1_package.experiments.training import check_experiments
where = Path(sys.argv[3])
check_experiments([sys.argv[1]], [sys.argv[2]], split=load_split(where / 'split.json'),
                  benchmarks=where / 'benchmarks')
" "$EXPERIMENT" "$MODEL" "$NESTED/outer_$outer" \
        || fail "outer fold $outer: $EXPERIMENT does not apply to $MODEL, or the inner folds' calibrations do not fit it (see above)"
done

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
MAXTIME=""
if command -v sinfo >/dev/null 2>&1; then
    [[ -n "$PARTITION" ]] || PARTITION="$(sinfo -h -o '%P' 2>/dev/null | grep '\*' | tr -d '*' | head -1 || true)"
    if [[ -n "$PARTITION" ]]; then
        MAXTIME="$(sinfo -h -p "$PARTITION" -o '%l' 2>/dev/null | head -1 | tr -d ' ' || true)"
        if [[ -n "$MAXTIME" ]] && (( $(_seconds "$WALLTIME") > $(_seconds "$MAXTIME") )); then
            fail "--time $WALLTIME exceeds partition '$PARTITION' (limit $MAXTIME). A task killed
     at the wall clock loses only its trials in flight; ask for less and more tasks"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
TASKS=$(( ${#OUTER_LIST[@]} * TASKS_PER_FOLD ))
PARALLEL=$(( CPUS / INNER_FOLDS ))
(( PARALLEL >= 1 )) || PARALLEL=1
ROUNDS=$(( (TRIALS + PARALLEL - 1) / PARALLEL ))
# Seconds per epoch of one fold on its 12 training reactors, averaged over 200 epochs
# of the length schedule: 65 s for R4 in the paper's final runs on a multi-core server,
# which ran many folds at once. R3 takes about 1.4 and R2 1.85 times as long (workstation
# timings). An inner fold trains on (inner folds - 1) / inner folds of the
# reactors. Cores differ: an order of magnitude, not a promise.
case "$MODEL" in r4) PER_EPOCH=65 ;; r3) PER_EPOCH=91 ;; r2) PER_EPOCH=120 ;; esac
TRIAL_HOURS="$(awk -v e="$EPOCHS" -v s="$PER_EPOCH" -v f="$INNER_FOLDS" 'BEGIN { printf "%.1f", e * s * (f - 1) / f / 3600 }')"
TASK_HOURS="$(awk -v r="$ROUNDS" -v t="$TRIAL_HOURS" 'BEGIN { printf "%.1f", r * t }')"
CORE_HOURS="$(awk -v n="$TASKS" -v t="$TRIALS" -v f="$INNER_FOLDS" -v h="$TRIAL_HOURS" 'BEGIN { printf "%.0f", n * t * f * h }')"
WALL_HOURS=$(( $(_seconds "$WALLTIME") / 3600 ))

cat <<INFO
================================================================
model       : ${MODEL}, ${EXPERIMENT_FILE}
outer folds : ${OUTER_FOLDS}, ${INNER_FOLDS} inner folds each (${NESTED})
anchors     : each outer fold's calibration in ${BENCHMARKS}
searches    : runs/${MODEL}/tuning_${NAME:-nested-<array job id>}_outer<k>, started from the default values
array       : ${TASKS} tasks (0-$(( TASKS - 1 ))), ${TASKS_PER_FOLD} per outer fold
trials      : ${TRIALS} per task, $(( TRIALS * TASKS_PER_FOLD )) per outer fold, ${EPOCHS} epochs each
resources   : ${CPUS} cores x ${MEM_PER_CPU} per task, ${PARALLEL} trials at once, ${WALLTIME}, partition ${PARTITION:-(default)}${MAXTIME:+ (max $MAXTIME)}
env         : ${ADM1_VENV}
runs        : ${ADM1_RUN_DIR:-${SCRATCH:-$PWD}/runs}
rough time  : ~${TRIAL_HOURS} h per trial, ~${TASK_HOURS} h per task, ~${CORE_HOURS} core-hours in all
================================================================
INFO
if awk -v t="$TASK_HOURS" -v w="$WALL_HOURS" 'BEGIN { exit !(t > w) }'; then
    cat <<WARN
WARNING: a task may need ~${TASK_HOURS} h but asks for ${WALLTIME}. A task killed at the wall
         clock loses only its trials in flight and the collection still selects from the
         rest, but prefer more tasks with fewer trials each: --tasks-per-fold, --trials.
WARN
fi

# ALL carries the environment, which is how the lists (they contain spaces, which
# --export does not take reliably) and ADM1_RUN_DIR reach the jobs; the rest is
# listed as well, so a cluster default of --export=NONE cannot drop it.
export ADM1_MODEL="$MODEL" ADM1_EXPERIMENT="$EXPERIMENT_FILE" ADM1_NESTED_DIR="$NESTED" \
       ADM1_OUTER_FOLDS="$OUTER_FOLDS"
if [[ -n "$NAME" ]]; then export ADM1_NESTED_NAME="$NAME"; else unset ADM1_NESTED_NAME; fi
EXPORTS="ALL,N_TRIALS=$TRIALS,EPOCHS=$EPOCHS,ADM1_VENV=$ADM1_VENV,ADM1_REPO=$REPO_ROOT"
ARRAY_ARGS=(--parsable --array="0-$(( TASKS - 1 ))" --time="$WALLTIME" --cpus-per-task="$CPUS"
            --mem-per-cpu="$MEM_PER_CPU" ${PARTITION:+--partition="$PARTITION"} --export="$EXPORTS")

next_steps() {
    cat <<NEXT
Then train every outer fold with its selection, with the seeds of the paper's final runs:
  ADM1_BENCHMARKS_DIR=${BENCHMARKS} bash scripts/hpc/submit_experiments.sh \\
      --experiments ${EXPERIMENT} --models ${MODEL} --seeds "45 46 47" --epochs ${EPOCHS} \\
      --fold-hyperparameters "runs/{model}/tuning_$1_outer{fold}/best_hyperparameters.json"
NEXT
}

if (( DRY_RUN )); then
    echo "sbatch ${ARRAY_ARGS[*]} scripts/hpc/slurm_nested.sh"
    echo "sbatch --parsable --dependency=afterany:<array job> ${PARTITION:+--partition=$PARTITION} \\"
    echo "    --export=$EXPORTS,ADM1_NESTED_NAME=${NAME:-nested-<array job>} scripts/hpc/slurm_nested_collect.sh"
    echo
    next_steps "${NAME:-nested-<array job>}"
    exit 0
fi

ARRAY_JOB="$(sbatch "${ARRAY_ARGS[@]}" scripts/hpc/slurm_nested.sh)"
ARRAY_JOB="${ARRAY_JOB%%;*}"
NAME="${NAME:-nested-$ARRAY_JOB}"
COLLECT_JOB="$(sbatch --parsable --dependency="afterany:$ARRAY_JOB" ${PARTITION:+--partition="$PARTITION"} \
    --export="$EXPORTS,ADM1_NESTED_NAME=$NAME" scripts/hpc/slurm_nested_collect.sh)"
COLLECT_JOB="${COLLECT_JOB%%;*}"

cat <<DONE
submitted  : array ${ARRAY_JOB} (${TASKS} tasks), collection ${COLLECT_JOB} after it
watch      : squeue -u \$USER;  tail -f logs/adm1-nested-${ARRAY_JOB}_0.out
selections : runs/${MODEL}/tuning_${NAME}_outer<k>/best_hyperparameters.json,
             tabulated in runs/${MODEL}/${NAME}_selections.csv
DONE
next_steps "$NAME"
