"""Filesystem locations and run bookkeeping.

Every path resolves through environment variables, so that a run can move
between a workstation and a cluster and evaluation finds what training wrote:

``ADM1_DATA_DIR``
    Where the reactor data live. Defaults to ``<repo>/data``.
``ADM1_RUN_DIR``
    Where runs (checkpoints, metrics, split files, Optuna storage) are written.
    Defaults to ``<repo>/runs``. Point this at fast scratch on a cluster.
``ADM1_BENCHMARKS_DIR``
    The per-fold calibrations of ``scripts/calibrate/calibrate_folds.py``.
    Defaults to ``<run dir>/benchmarks``.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import warnings
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "PROJECT_ROOT",
    "RunPaths",
    "benchmarks_dir",
    "checkpoint_safe_dir",
    "data_dir",
    "environment_metadata",
    "run_dir",
]

#: Repository root (``adm1_package/experiments/config.py`` -> two levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Directory holding the reactor datasets."""
    return Path(os.environ.get("ADM1_DATA_DIR", PROJECT_ROOT / "data")).resolve()


def _needs_ascii_path(path: Path) -> bool:
    """Whether a path is likely to trip up the checkpoint writer on Windows."""
    text = str(path)
    return not text.isascii() or " " in text


def _short_path(path: Path) -> Path | None:
    """The Windows 8.3 short form of an existing path, if the volume provides one.

    ``C:\\Users\\Jane Müller\\...`` becomes a form such as ``C:\\Users\\JANEMU~1\\...``:
    the same directory, but free of spaces and non-ASCII characters. Returns ``None``
    when short-name generation is disabled on the volume (``fsutil 8dot3name``),
    when the path does not exist yet, or off Windows.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    try:
        get_short = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
    except (AttributeError, OSError):  # pragma: no cover - non-Windows
        return None
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD

    buffer = ctypes.create_unicode_buffer(1024)
    length = get_short(str(path), buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        return None
    return Path(buffer.value)


def run_dir() -> Path:
    """Root directory for run artifacts.

    Orbax writes checkpoints through tensorstore, a C++ library whose Windows
    path handling copes poorly with non-ASCII characters and, on some versions,
    with spaces. A repository under ``C:\\Users\\Jane Müller\\...`` hits both,
    and the failure surfaces late and unhelpfully as "access denied" partway
    through training.

    When the configured directory has either problem, the 8.3 short form of the
    *same* directory is handed back instead. Runs therefore stay next to the
    repository rather than being exiled to a temporary directory, where
    evaluation could only find them through hardcoded checkpoint paths.

    Set ``ADM1_RUN_DIR`` to an ASCII path without spaces (``D:\\adm1-runs``, or
    fast scratch on a cluster) to bypass all of this.
    """
    return checkpoint_safe_dir(
        Path(os.environ.get("ADM1_RUN_DIR", PROJECT_ROOT / "runs"))
    )


def benchmarks_dir(run_root: str | Path | None = None) -> Path:
    """The per-fold calibrations (``scripts/calibrate/calibrate_folds.py``).

    ``$ADM1_BENCHMARKS_DIR``, else ``benchmarks/`` under ``run_root`` or the run
    directory. On a cluster, where runs go to scratch, set the variable: the
    cluster job scripts point it at the repository's ``runs/benchmarks``.
    """
    configured = os.environ.get("ADM1_BENCHMARKS_DIR")
    if configured:
        return Path(configured)
    return Path(run_root if run_root is not None else run_dir()) / "benchmarks"


def checkpoint_safe_dir(path: str | Path) -> Path:
    """Create ``path`` and return a form the checkpoint writer can handle.

    See :func:`run_dir` for why this is needed. Applied to every run root,
    including one passed explicitly with ``--run-dir``.
    """
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)

    if os.name != "nt" or not _needs_ascii_path(resolved):
        return resolved

    short = _short_path(resolved)
    if short is not None and not _needs_ascii_path(short):
        return short

    warnings.warn(
        f"Run directory {resolved} contains characters that the checkpoint writer "
        "may reject on Windows (non-ASCII or spaces), and no 8.3 short form is "
        "available. If saving fails with an access error, set ADM1_RUN_DIR to a "
        "plain path, e.g.  set ADM1_RUN_DIR=C:\\adm1-runs",
        RuntimeWarning,
        stacklevel=2,
    )
    return resolved


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment_metadata() -> dict[str, Any]:
    """Provenance recorded next to every run, so a result can be traced back."""
    from adm1_package.utils.solver import solver_settings

    versions: dict[str, str] = {}
    for name in ("jax", "jaxlib", "equinox", "diffrax", "optax", "optuna", "numpy", "torch"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "unknown")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": versions,
        "solver": solver_settings(),
    }


@dataclass
class RunPaths:
    """Directory layout of a single run.

    ::

        <run_dir>/<model>/<run_name>/
            metadata.json       environment + configuration provenance
            split.json          copy of the split actually used
            hyperparameters.json
            fold_1/ ... fold_k/ trainer checkpoints and metrics
            results/            evaluation tables
    """

    model: str
    name: str
    root: Path = field(default_factory=run_dir)

    @property
    def base(self) -> Path:
        return self.root / self.model / self.name

    @property
    def results(self) -> Path:
        return self.base / "results"

    def fold(self, index: int) -> Path:
        return self.base / f"fold_{index}"

    @classmethod
    def create(cls, model: str, name: str | None = None, root: Path | None = None) -> "RunPaths":
        """Create the directory tree for a new run."""
        name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = cls(
            model=model,
            name=name,
            root=checkpoint_safe_dir(root) if root is not None else run_dir(),
        )
        paths.base.mkdir(parents=True, exist_ok=True)
        paths.results.mkdir(parents=True, exist_ok=True)
        return paths

    def write_metadata(self, **extra: Any) -> Path:
        """Record environment provenance plus any run-specific configuration."""
        payload = {**environment_metadata(), **extra}
        target = self.base / "metadata.json"
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target
