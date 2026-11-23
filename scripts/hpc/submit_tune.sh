#!/bin/bash
# Submit the hyperparameter search, after checking the things that otherwise
# only fail once the job has already waited in the queue.
#
#   bash scripts/hpc/submit_tune.sh                        # all three variants
#   bash scripts/hpc/submit_tune.sh --model r4 --tasks 4   # four tasks on R4
#   bash scripts/hpc/submit_tune.sh --dry-run
#
# Options:
#   --model r2|r3|r4   only this variant (default: all three, round-robin)
#   --tasks N          array tasks per variant (default 1)
#   --trials N         trials each task contributes (default 20)
#   --epochs N         epochs per fold during the search (default 50)
#   --time HH:MM:SS    wall-clock limit (default 72:00:00)
#   --partition NAME   partition to submit to (default: the cluster's default)
#   --cpus N           cores per task (default 8)
#   --mem-per-cpu M    memory per core (default 4G); scales with --cpus
#   --ude-args "..."   flags passed on to tune_hyperparameters.py, e.g.
#                      "--experiment configs/experiments/fold_fast_narrow_tan.json" or
#                      "--physics arrhenius nh3_inhibition" (default: none, i.e.
#                      the default model)
#   --dry-run          print the sbatch command and the estimate, submit nothing
#
# A job asking for more wall clock than the partition allows is not rejected by
# SLURM -- it is accepted and then held forever with reason PartitionTimeLimit,
# which reads like an ordinary queue wait. The check below compares the request
# against the partition's MaxTime and refuses up front.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL=""
TASKS=1
TRIALS=20
EPOCHS=50
WALLTIME="72:00:00"
PARTITION=""
CPUS=8
MEM_PER_CPU=4G
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --tasks)   TASKS="$2"; shift 2 ;;
        --trials)  TRIALS="$2"; shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --time)    WALLTIME="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --cpus)    CPUS="$2"; shift 2 ;;
        --mem-per-cpu) MEM_PER_CPU="$2"; shift 2 ;;
        --ude-args) export ADM1_UDE_ARGS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        # The whole leading comment block, however long it grows; a fixed
        # line range silently truncated the option list when options were added.
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
                       "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
fail() { echo "ERROR: $*" >&2; exit 1; }

command -v sbatch >/dev/null 2>&1 || fail "sbatch not found; is this a login node?"

[[ -n "${ADM1_VENV:-}" ]] || fail \
    "ADM1_VENV is not set. Point it at your virtualenv directory:
       export ADM1_VENV=\$PROJECT/envs/hybrid-adm1
     Build one with: bash scripts/hpc/setup_env.sh"

[[ -d "$ADM1_VENV" ]] || fail "ADM1_VENV=$ADM1_VENV does not exist"

# Activate exactly as the job will. `bash -c` is non-interactive and does not
# read ~/.bashrc; that does not break activation of a venv, but the import
# check below still earns its keep.
echo "checking activation as a batch job would see it..."
ACTIVATION_CHECK="$(
    ADM1_VENV="$ADM1_VENV" \
    bash -c '
        source scripts/hpc/activate_env.sh || exit 1
        python -c "import jax, optuna, equinox, diffrax, orbax.checkpoint" || exit 1
        python -c "
import jax
jax.config.update(\"jax_enable_x64\", True)
import jax.numpy as jnp
assert jnp.zeros(1).dtype == jnp.float64, \"float64 unavailable\"
" || exit 1
        echo "$ADM1_ACTIVATION"
    ' 2>&1
)" || {
    echo "$ACTIVATION_CHECK" >&2
    fail "the environment does not come up in a non-interactive shell (see above).
     Check that ADM1_VENV is the venv DIRECTORY -- the one containing bin/ --
     and that bash scripts/hpc/setup_env.sh completed without errors."
}
echo "  ok: ${ACTIVATION_CHECK##*$'\n'}"

[[ -f configs/split.json ]] || fail \
    "configs/split.json is missing. Run: python scripts/make_splits.py"

DATA_DIR="${ADM1_DATA_DIR:-$PWD/data}"
for campaign in LaRA-temp LaRA-substrate rice-bran; do
    [[ -d "$DATA_DIR/$campaign" ]] || fail \
        "$DATA_DIR/$campaign is missing. Set ADM1_DATA_DIR to where the data live."
done

# SLURM opens the output file before the script runs, so a missing log directory
# loses the job's entire output and the failure is invisible.
mkdir -p logs

# ---------------------------------------------------------------------------
# Wall clock against the partition's limit
# ---------------------------------------------------------------------------
# sbatch accepts a job asking for more time than the partition allows, then
# holds it forever with reason PartitionTimeLimit. In squeue that is a PD state
# with a plausible-looking reason, so it reads as a long queue wait rather than
# a job that will never start.
_slurm_seconds() {
    # SLURM time formats: "infinite", "MM", "MM:SS", "HH:MM:SS", "D-HH",
    # "D-HH:MM" and "D-HH:MM:SS".
    local spec="$1" days=0 rest
    case "${spec,,}" in
        infinite|unlimited|"") echo 999999999; return 0 ;;
    esac
    if [[ "$spec" == *-* ]]; then
        days="${spec%%-*}"
        rest="${spec#*-}"
    else
        rest="$spec"
    fi
    local -a parts
    IFS=':' read -ra parts <<< "$rest"
    local h=0 m=0 s=0
    case "${#parts[@]}" in
        3) h="${parts[0]}"; m="${parts[1]}"; s="${parts[2]}" ;;
        2) if [[ "$spec" == *-* ]]; then h="${parts[0]}"; m="${parts[1]}"
           else m="${parts[0]}"; s="${parts[1]}"; fi ;;
        1) if [[ "$spec" == *-* ]]; then h="${parts[0]}"; else m="${parts[0]}"; fi ;;
        *) echo 0; return 1 ;;
    esac
    echo $(( 10#${days:-0} * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${s:-0} ))
}

if command -v sinfo >/dev/null 2>&1; then
    if [[ -z "$PARTITION" ]]; then
        # The default partition is the one sinfo marks with a trailing '*'.
        PARTITION="$(sinfo -h -o '%P' 2>/dev/null | grep '\*' | tr -d '*' | head -1 || true)"
    fi
    if [[ -n "$PARTITION" ]]; then
        MAXTIME="$(sinfo -h -p "$PARTITION" -o '%l' 2>/dev/null | head -1 | tr -d ' ' || true)"
        if [[ -n "$MAXTIME" ]]; then
            REQUESTED_S="$(_slurm_seconds "$WALLTIME")"
            LIMIT_S="$(_slurm_seconds "$MAXTIME")"
            if (( REQUESTED_S > LIMIT_S )); then
                echo
                echo "Partitions on this cluster:" >&2
                sinfo -h -o '  %-20P  max %-14l  nodes %-6D' | sort -u >&2
                echo >&2
                fail "--time $WALLTIME exceeds partition '$PARTITION', whose limit is $MAXTIME.
     sbatch would accept this and then hold the job forever with reason
     PartitionTimeLimit, which looks like an ordinary queue wait.

     Nothing is lost by asking for less: a task killed at the wall clock loses
     only the trial it was running, because Optuna resumes from the shared
     study. Shorter jobs also backfill sooner. Either:
       bash scripts/hpc/submit_tune.sh --time $MAXTIME --tasks <more tasks>
     or pick a partition that allows longer:
       bash scripts/hpc/submit_tune.sh --partition <name> --time $WALLTIME"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Array geometry
# ---------------------------------------------------------------------------
if [[ -n "$MODEL" ]]; then
    case "$MODEL" in r2|r3|r4) ;; *) fail "--model must be r2, r3 or r4" ;; esac
    ARRAY_MAX=$(( TASKS - 1 ))
    export ADM1_MODEL="$MODEL"
    WHAT="$MODEL, $TASKS task(s)"
else
    ARRAY_MAX=$(( TASKS * 3 - 1 ))
    WHAT="r2+r3+r4, $TASKS task(s) each"
    # Do not let a value left over from an earlier --model run in this shell
    # pin every task to one variant.
    unset ADM1_MODEL
fi

# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------
# Measured on one workstation core, per fold per epoch, on a training set of
# twice the current size: R4 58 s, R3 81 s, R2 107 s. On the current training
# set an epoch takes about half that. Cluster cores differ, so treat this as an
# order of magnitude.
FOLDS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["k_folds"])' configs/split.json 2>/dev/null || echo 4)"
# Mirrors the sizing in slurm_tune.sh: Optuna runs PARALLEL trials at once and
# they all share one pool of WORKERS processes, so WORKERS is the true count of
# cores in use and is what the wall-clock estimate divides by.
PARALLEL=$(( CPUS / FOLDS ))
(( PARALLEL >= 1 )) || PARALLEL=1
WORKERS=$(( PARALLEL * FOLDS ))
(( WORKERS <= CPUS )) || WORKERS=$CPUS

estimate() {
    local per_epoch=$1
    # trials x folds x epochs x seconds, divided by the folds run in parallel
    echo $(( TRIALS * FOLDS * EPOCHS * per_epoch / WORKERS / 3600 ))
}

cat <<INFO
================================================================
submitting : $WHAT
array      : 0-$ARRAY_MAX
trials     : $TRIALS per task, $EPOCHS epochs per fold
partition  : ${PARTITION:-(cluster default)}${MAXTIME:+  (max $MAXTIME)}
resources  : $CPUS cores, $MEM_PER_CPU/core, $WALLTIME, $FOLDS folds, $PARALLEL trials in parallel, $WORKERS fold workers
env        : $ADM1_VENV
data       : $DATA_DIR
run dir    : ${ADM1_RUN_DIR:-\${SCRATCH:-$PWD}/adm1-runs}
storage    : ${ADM1_OPTUNA_STORAGE:-journal (file-based; safe on a shared FS)}

rough wall-clock per task, from workstation timings:
  r2  ~$(estimate 55) h      r3  ~$(estimate 40) h      r4  ~$(estimate 30) h
INFO

# Hours, parsed properly: "${WALLTIME%%:*}" reads "3-00:00:00" as 3.
WALLTIME_HOURS=$(( $(_slurm_seconds "$WALLTIME") / 3600 ))
if (( $(estimate 55) > WALLTIME_HOURS )); then
    cat <<WARN

WARNING: R2 may not finish $TRIALS trials within $WALLTIME (~${WALLTIME_HOURS} h).
         Optuna resumes from the shared study, so a task hitting the wall
         loses only its trial in flight -- but prefer more tasks with fewer
         trials each over one task that gets killed.
WARN
fi
echo "================================================================"

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
SBATCH_ARGS=(
    --array="0-$ARRAY_MAX"
    --time="$WALLTIME"
    --cpus-per-task="$CPUS"
    ${PARTITION:+--partition="$PARTITION"}
    # Must scale with --cpus: one process per core, each importing jax and
    # torch. The script's #SBATCH --mem is sized for the 8-core default only.
    --mem-per-cpu="$MEM_PER_CPU"
    # ALL carries the current environment, which is how ADM1_MODEL,
    # ADM1_DATA_DIR, ADM1_RUN_DIR and ADM1_OPTUNA_STORAGE reach the job. The
    # named ones are listed anyway so a misconfigured --export default on the
    # cluster cannot silently drop them.
    --export="ALL,N_TRIALS=$TRIALS,EPOCHS=$EPOCHS,ADM1_VENV=$ADM1_VENV,ADM1_REPO=$REPO_ROOT"
    scripts/hpc/slurm_tune.sh
)

if (( DRY_RUN )); then
    echo
    echo "dry run, would submit:"
    echo "  sbatch ${SBATCH_ARGS[*]}"
    exit 0
fi

sbatch "${SBATCH_ARGS[@]}"

cat <<'NEXT'

Watch it with:
  squeue --me
  tail -f logs/adm1-tune-*_0.out

When it finishes, the selected values are in
  runs/<model>/tuning_<array-job-id>/best_hyperparameters.json  (copied off scratch)
NEXT
