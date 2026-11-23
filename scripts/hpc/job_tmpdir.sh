# Give a batch job its own temporary directory on scratch instead of the node's /tmp.
#
#   source scripts/hpc/job_tmpdir.sh
#
# Source it first thing in a job script, before any here-string (<<<) or
# here-document (<<). Sets TMPDIR, TMP and TEMP; ADM1_TMP_BASE overrides where the
# per-job directories go (default $SCRATCH/tmp, else /scratch/$USER/tmp).
#
# Why
# ---
# Bash writes every here-string and here-document to a temporary file, and so do
# Python, JAX and the fold workers. They go to $TMPDIR, or /tmp without it, and /tmp
# on a compute node is small, local and shared with every other job on that node.
# Once someone has filled it, the job dies at its first here-string, before any of
# our code has run:
#   slurm_script: line 36: cannot create temp file for here-document: No space left on device
# Scratch has the room, and a directory per job keeps tasks of one array apart.
#
# Clean-up
# --------
# The directory is removed when the job exits, also at the wall clock (SLURM's
# SIGTERM). A SIGKILL skips that, so every job also removes the directories of its
# owner's jobs that squeue no longer lists. squeue's %A is the job id of each array
# task, which is what SLURM_JOB_ID holds, so running tasks of an array keep theirs.

_adm1_tmp_base=""
for _adm1_candidate in "${ADM1_TMP_BASE:-}" "${SCRATCH:+$SCRATCH/tmp}" "/scratch/${USER:-$(id -un)}/tmp"; do
    [[ -n "$_adm1_candidate" ]] || continue
    if mkdir -p "$_adm1_candidate" 2>/dev/null && [[ -w "$_adm1_candidate" ]]; then
        _adm1_tmp_base="$_adm1_candidate"
        break
    fi
done

if [[ -z "$_adm1_tmp_base" ]]; then
    echo "WARNING: no writable scratch for temporary files (set ADM1_TMP_BASE);" >&2
    echo "         keeping TMPDIR=${TMPDIR:-/tmp}, which may be the node's full /tmp." >&2
else
    # Directories of jobs that are gone; skipped when squeue does not answer, since
    # an empty list would make every job look gone.
    if _adm1_live="$(squeue -h -u "${USER:-$(id -un)}" -o '%A' 2>/dev/null)" && [[ -n "$_adm1_live" ]]; then
        for _adm1_dir in "$_adm1_tmp_base"/*/; do
            [[ -d "$_adm1_dir" ]] || continue
            _adm1_id="$(basename "$_adm1_dir")"
            case "$_adm1_id" in ''|*[!0-9]*) continue ;; esac
            [[ $'\n'"$_adm1_live"$'\n' == *$'\n'"$_adm1_id"$'\n'* ]] || rm -rf "$_adm1_dir"
        done
    fi

    ADM1_JOB_TMPDIR="$_adm1_tmp_base/${SLURM_JOB_ID:-local-$$}"
    mkdir -p "$ADM1_JOB_TMPDIR"
    export TMPDIR="$ADM1_JOB_TMPDIR" TMP="$ADM1_JOB_TMPDIR" TEMP="$ADM1_JOB_TMPDIR"
    # The path is fixed now, so the trap removes this directory whatever TMPDIR is later.
    trap "rm -rf '$ADM1_JOB_TMPDIR'" EXIT
    echo "temporary files: $TMPDIR (node /tmp: $(df -h --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' ') used)"
fi

unset _adm1_tmp_base _adm1_candidate _adm1_live _adm1_dir _adm1_id
