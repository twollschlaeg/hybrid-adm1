# Activate the project's virtualenv inside a batch job.
#
#   source scripts/hpc/activate_env.sh
#
# Requires ADM1_VENV, the virtualenv directory -- the one containing bin/.
#
# Linux only: it expects <venv>/bin/python, where a Windows venv puts
# <venv>/Scripts/python.exe. That is deliberate -- this runs on the cluster.
#
# Why a virtualenv and not conda
# ------------------------------
# Conda needs a shell function that `conda init` writes into ~/.bashrc. A batch
# script runs in a NON-INTERACTIVE shell and bash only reads ~/.bashrc when it is
# interactive, so that function does not exist in the job and `conda activate`
# fails with "run 'conda init' before 'conda activate'" -- after the queue wait.
#
# A virtualenv has none of that machinery. bin/activate is an ordinary POSIX
# script with nothing to initialise, and bin/python finds its own site-packages
# from its location regardless, so a job behaves exactly like a login shell.

: "${ADM1_VENV:?set ADM1_VENV to the virtualenv directory, e.g. \$PROJECT/envs/hybrid-adm1}"

if [[ ! -x "$ADM1_VENV/bin/python" ]]; then
    echo "ERROR: $ADM1_VENV/bin/python not found." >&2
    echo "       ADM1_VENV must be the venv DIRECTORY, the one containing bin/." >&2
    echo "       Build one with:  bash scripts/hpc/setup_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -f "$ADM1_VENV/bin/activate" ]]; then
    # bin/activate dereferences a few variables that may be unset, and callers
    # here run under `set -u`. Relax it for the source and put it back after.
    case "$-" in
        *u*) _adm1_reset_u=1; set +u ;;
        *)   _adm1_reset_u=0 ;;
    esac
    # shellcheck disable=SC1091
    source "$ADM1_VENV/bin/activate"
    [[ "$_adm1_reset_u" == 1 ]] && set -u
    unset _adm1_reset_u
    ADM1_ACTIVATION="venv ($ADM1_VENV/bin/activate)"
else
    # A venv built with --without-pip or copied incompletely still runs fine
    # from PATH; bin/python needs no activation to find its own site-packages.
    export PATH="$ADM1_VENV/bin:$PATH"
    export VIRTUAL_ENV="$ADM1_VENV"
    ADM1_ACTIVATION="PATH prefix (no bin/activate; this is fine)"
fi
export ADM1_ACTIVATION

# ---------------------------------------------------------------------------
# Check it actually took
# ---------------------------------------------------------------------------
# Catching this here costs a second; catching it later costs the queue wait plus
# a confusing ImportError from whichever interpreter answered instead.
# `-ef` compares device and inode, so this holds even when the venv reaches the
# interpreter by another name -- a symlinked home directory, an automounted
# path, a trailing slash. Comparing the strings would report a false failure on
# most clusters, where /home is a symlink into the parallel filesystem.
_adm1_python="$(command -v python || true)"
if [[ -z "$_adm1_python" || ! "$_adm1_python" -ef "$ADM1_VENV/bin/python" ]]; then
    echo "ERROR: activation did not take." >&2
    echo "       expected : $ADM1_VENV/bin/python" >&2
    echo "       got      : ${_adm1_python:-<no python on PATH>}" >&2
    echo "       route    : $ADM1_ACTIVATION" >&2
    return 1 2>/dev/null || exit 1
fi

unset _adm1_python
