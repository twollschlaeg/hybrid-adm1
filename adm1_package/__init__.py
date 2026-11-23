"""Hybrid ADM1 models: reduced mechanistic variants coupled to neural corrections.

Subpackages
-----------
``models``
    One package per ADM1 variant (``base``, ``r2``, ``r3``, ``r4``), each holding
    the mechanistic model, the dataset and (not ``base``) the UDE.
``utils``
    Training loop, network, solver settings, loss scaling and I/O helpers.
``experiments``
    Splits, model registry, cross-validated training, hyperparameter search,
    metrics and table generation.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version

_SUBPACKAGES = ("experiments", "models", "utils")

try:
    __version__ = _version("hybrid-adm1")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "1.0.0"

__all__ = ["experiments", "models", "utils", "__version__"]


def __getattr__(name: str):
    """Import a subpackage on first use (``adm1_package.models``), so that importing the
    package, e.g. for the experiments' configuration, does not load JAX and PyTorch."""
    if name in _SUBPACKAGES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
