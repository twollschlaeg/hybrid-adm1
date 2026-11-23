"""Helpers the tests share; the fixtures are in conftest.py.

pytest puts this directory on ``sys.path``, so a test module imports them with
``from helpers import ...``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXPERIMENTS = ROOT / "configs" / "experiments"
CAMPAIGNS = ("LaRA-temp", "LaRA-substrate", "rice-bran")


def canonical_split():
    """The study's split (configs/split.json)."""
    from adm1_package.experiments.splits import load_split

    return load_split(ROOT / "configs" / "split.json")


def load_script(relative: str):
    """A script under scripts/ as a module, e.g. ``load_script("calibrate/calibrate_folds.py")``.

    Registered in ``sys.modules``, where a dataclass of the script resolves its annotations.
    """
    path = SCRIPTS / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stub_repository(tmp_path: Path, copies, files: dict[str, str], stubs: dict[str, str],
                    environment: dict[str, str] | None = None):
    """Shell scripts in a repository of their own, with programs stubbed out.

    ``copies`` are paths of this repository copied in; ``files`` are written (LF line
    endings); ``stubs`` are programs (python, sbatch) put first on the path. A stub that
    writes to ``$FAKE_LOG`` records what it was asked to do. Every ``ADM1_``/``SLURM_``
    variable, and the others a script reads, are cleared; runs go to the repository's runs/.

    Returns ``(root, run)``, ``run(script, *arguments, **environment)`` returning
    ``(result, calls)``: the completed process and the lines logged meanwhile.
    """
    import os

    root = tmp_path / "repo"
    for relative in copies:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, root / relative)
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8", newline="\n")
    for campaign in CAMPAIGNS:
        (root / "data" / campaign).mkdir(parents=True, exist_ok=True)
    programs = tmp_path / "bin"
    programs.mkdir()
    for name, text in stubs.items():
        (programs / name).write_text(text, encoding="utf-8", newline="\n")
    (tmp_path / "venv").mkdir()
    log = tmp_path / "calls.log"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("ADM1_", "SLURM_")) and key not in (
               "SCRATCH", "SEED", "HYPERPARAMETERS", "FOLD_HYPERPARAMETERS", "N_TRIALS", "EPOCHS")}
    env.update(PATH=str(programs) + os.pathsep + env.get("PATH", ""), FAKE_LOG=str(log),
               ADM1_RUN_DIR=str(root / "runs"), ADM1_REPO=str(root), ADM1_VENV=str(tmp_path / "venv"),
               **(environment or {}))

    def run(script, *arguments, **extra):
        result = subprocess.run(["bash", script, *arguments], cwd=root, env={**env, **extra},
                                capture_output=True, text=True, timeout=120)
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        log.unlink(missing_ok=True)
        return result, calls

    return root, run


def bash_works() -> bool:
    """Whether a working bash is on the path (on Windows, ``bash`` can be the WSL
    launcher, which fails without WSL)."""
    if shutil.which("bash") is None:
        return False
    try:
        return subprocess.run(["bash", "-c", "true"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Samples and networks
# ---------------------------------------------------------------------------
def first_days(inputs: dict, days: float) -> dict:
    """A sample's forcing (``x["input"]``) cut after ``days`` of its record."""
    knots = int(np.searchsorted(np.asarray(inputs["time"]), days)) + 1
    return {**inputs, "time": inputs["time"][:knots],
            "coeffs": tuple(c[: knots - 1] for c in inputs["coeffs"])}


def splined(inputs: dict) -> dict:
    """The same daily records joined by a cubic spline, the forcing before the smoothstep
    (since retired from the datasets): its slope at the records is not zero, so the
    temperature term of the gas output is exercised there."""
    import diffrax as dx
    import jax

    interp = dx.CubicInterpolation(inputs["time"], inputs["coeffs"])
    knots = jax.vmap(interp.evaluate)(inputs["time"])
    return {**inputs, "coeffs": dx.backward_hermite_coefficients(inputs["time"], knots)}


def correction_mlp(spec, in_size: int, correction=None, width: int = 16, depth: int = 2,
                   seed: int = 0, **options):
    """A tanh network of the width ``correction`` gives ``spec``'s hybrid, randomly
    initialised unless ``zero_output=True`` is among ``options``."""
    import jax
    import jax.random as jr

    from adm1_package.utils.nn_architectures import CorrectionMLP

    return CorrectionMLP(in_size, spec.correction_layout(correction).size, width, depth,
                         jax.nn.tanh, lambda z: z, True, key=jr.PRNGKey(seed), **options)


# ---------------------------------------------------------------------------
# Per-fold calibrations without the calibration
# ---------------------------------------------------------------------------
def write_benchmarks(root: Path, split=None, *, arrhenius: bool = False, bounds: dict | None = None,
                     forcing: str | None = "smoothstep", unfinished: int | None = None,
                     data: bool = False) -> Path:
    """A per-fold benchmark directory as calibrate_folds.py leaves it, without the fits.

    ``model.json`` records ``arrhenius``, ``bounds`` and ``forcing`` (``forcing=None``: made
    before the record, so no model.json). Every campaign of every fold is calibrated, except
    in fold ``unfinished``. With ``data``, each fold's data directory holds a marker naming
    the fold, beside a backup of an earlier calibration.
    """
    split = split or canonical_split()
    root.mkdir(parents=True)
    (root / "split.json").write_text(json.dumps(split.to_dict()), encoding="utf-8")
    if forcing is not None:
        (root / "model.json").write_text(json.dumps(
            {"arrhenius": arrhenius, "bounds": bounds or {}, "forcing": forcing}), encoding="utf-8")
    for fold in range(1, split.k_folds + 1):
        (root / f"fold_{fold}").mkdir()
        status = "running" if fold == unfinished else "calibrated"
        for campaign in CAMPAIGNS:
            (root / f"fold_{fold}" / f"{campaign}.json").write_text(
                json.dumps({"fold": fold, "campaign": campaign, "status": status}), encoding="utf-8")
            if data:
                directory = root / f"fold_{fold}" / "data" / campaign
                (directory / "calibration-backup").mkdir(parents=True)
                (directory / "marker.txt").write_text(f"outer {fold}", encoding="utf-8")
                (directory / "calibration-backup" / "old.json").write_text("{}", encoding="utf-8")
        if data:
            (root / f"fold_{fold}" / "held-out-starts").mkdir()
    return root
