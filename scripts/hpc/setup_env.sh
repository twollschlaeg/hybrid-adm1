#!/bin/bash
# One-time environment build on a cluster login node.
#
#   bash scripts/hpc/setup_env.sh                       # uses $PROJECT or $WORK
#   ADM1_VENV=/path/to/envs/hybrid-adm1 bash scripts/hpc/setup_env.sh
#   ADM1_PYTHON=/usr/bin/python3.12 bash scripts/hpc/setup_env.sh
#
# Run it on the login node, not in a job: compute nodes usually have no
# outbound network and pip would fail after the queue wait.
#
# This builds a plain virtualenv with pip, not a conda environment. On some
# clusters only an old conda is available, whose solver reads the whole of
# conda-forge's repodata into Python objects and needs several GB of memory to do
# it. Login nodes often cap per-user memory well below that, so the solve is
# OOM-killed partway through "Collecting package metadata" and leaves no error
# behind -- it just looks like a hang. pip resolves from one package's metadata at
# a time and stays far below such a cap.
#
# The venv goes into a project or scratch prefix rather than $HOME, because it
# is a couple of GB and the home quota is normally the smallest on the machine.

set -euo pipefail

# Resolved before the cd, so sourcing activate_env.sh below works no matter
# which directory the script was invoked from or how it was spelled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Where the environment goes
# ---------------------------------------------------------------------------
if [[ -z "${ADM1_VENV:-}" ]]; then
    for base in "${PROJECT:-}" "${WORK:-}" "${SCRATCH:-}" "$HOME"; do
        if [[ -n "$base" && -d "$base" ]]; then
            ADM1_VENV="$base/envs/hybrid-adm1"
            break
        fi
    done
fi
echo "venv               : $ADM1_VENV"
ENV_PARENT="$(dirname "$ADM1_VENV")"

# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
# jax 0.7 and numpy 2 need 3.11 or newer. On some clusters the environment
# modules stop at an older version, and a system Python such as
# /usr/bin/python3.12 is the newest. Every dependency supports 3.12.
PYTHON_MIN="3.11"

_usable_python() {
    local exe="$1"
    command -v "$exe" >/dev/null 2>&1 || return 1
    # venv needs ensurepip, which some distro interpreters ship separately.
    "$exe" -c "
import sys, importlib.util
need = tuple(int(p) for p in '$PYTHON_MIN'.split('.'))
sys.exit(0 if sys.version_info[:2] >= need
         and importlib.util.find_spec('venv')
         and importlib.util.find_spec('ensurepip') else 1)
" >/dev/null 2>&1
}

if [[ -n "${ADM1_PYTHON:-}" ]]; then
    PYTHON="$ADM1_PYTHON"
    _usable_python "$PYTHON" || {
        echo "ERROR: ADM1_PYTHON=$PYTHON is missing, older than $PYTHON_MIN," >&2
        echo "       or has no venv/ensurepip module." >&2
        exit 1
    }
else
    PYTHON=""
    for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
        if _usable_python "$candidate"; then PYTHON="$candidate"; break; fi
    done
fi

if [[ -z "$PYTHON" ]]; then
    cat >&2 <<MSG
ERROR: no Python $PYTHON_MIN+ with the venv and ensurepip modules on PATH.

Look for one, then pass it in:

    ls /usr/bin/python3.*
    module avail 2>&1 | grep -i python
    ADM1_PYTHON=/usr/bin/python3.12 bash scripts/hpc/setup_env.sh
MSG
    exit 1
fi
echo "interpreter        : $("$PYTHON" -c 'import platform; print(platform.python_version())') ($(command -v "$PYTHON"))"

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
# Some clusters reach PyPI only through a proxy. Without it pip does not fail
# fast -- it retries and hangs -- so say so now rather than after a five-minute
# stall.
if [[ -z "${https_proxy:-}" && -z "${HTTPS_PROXY:-}" ]]; then
    echo "NOTE               : no https_proxy is set. If this cluster needs one,"
    echo "                     pip will stall rather than report an error."
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
# Keep the wheel cache off the home quota as well; it is nearly as large as the
# environment itself.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ENV_PARENT/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"
echo "pip cache          : $PIP_CACHE_DIR"
echo

if [[ -d "$ADM1_VENV" ]]; then
    echo "Environment exists; updating it."
else
    "$PYTHON" -m venv "$ADM1_VENV"
fi

VENV_PY="$ADM1_VENV/bin/python"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install --upgrade -r requirements-hpc.txt

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
# Same activation path the batch job takes, so a problem shows up here rather
# than after the first queue wait.
export ADM1_VENV
source "$SCRIPT_DIR/activate_env.sh"

echo
echo "Verifying the installation ($ADM1_ACTIVATION):"
python - <<'PY'
import importlib
import platform

print(f"  python          {platform.python_version()}")
for name in ("jax", "jaxlib", "equinox", "diffrax", "optax",
             "orbax.checkpoint", "optuna", "numpy", "pandas", "torch"):
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise SystemExit(f"  MISSING {name}: {exc}")
    print(f"  {name:<15} {getattr(module, '__version__', 'unknown')}")

# The models are integrated in double precision; a float32 fallback would run
# and give quietly wrong answers rather than failing.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
assert jnp.zeros(1).dtype == jnp.float64, "float64 is not available"
print(f"  devices         {jax.devices()}")
print("  float64         ok")
PY

# A CUDA torch build on a CPU partition is GBs of wheels that will never be
# used, and it is the usual cause of a blown home quota here.
python - <<'PY'
import torch
if torch.version.cuda:
    print(f"\n  NOTE: torch was built against CUDA {torch.version.cuda}.")
    print("        requirements-hpc.txt points at PyTorch's CPU index, so pip")
    print("        should have taken a +cpu wheel. Harmless, but it wastes GBs.")
PY

# Not fatal. The environment is built and verified by this point, and a failing
# test must not swallow the banner below that says what to export.
if [[ -d tests ]]; then
    echo
    echo "Running the test suite (data-dependent tests skip if data/ is absent):"
    python -m pytest tests/ -q 2>&1 | tail -5 || echo "  (tests reported failures; the environment itself is fine)"
else
    echo
    echo "No tests/ directory in this checkout; skipping the test suite."
fi

cat <<EOF

================================================================
Environment ready.

Add this to your ~/.bashrc, or export it before every submit:

    export ADM1_VENV="$ADM1_VENV"

Then submit the search:

    bash scripts/hpc/submit_tune.sh --help
================================================================
EOF
