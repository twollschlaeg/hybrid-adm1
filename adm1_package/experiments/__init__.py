"""Reproducible experiment layer for the ADM1-UDE study.

The modules here hold, once for every variant, what the scripts share: where runs
are written, how the data are split, which hyperparameters each variant uses, how
a fold is trained, and how the resulting models are scored.

Typical use from the command line::

    python scripts/make_splits.py                 # write the canonical split
    python scripts/tune_hyperparameters.py --model r4
    python scripts/train_cv.py --model r4 --hyperparameters <trial.json>
    python scripts/evaluate.py --model r4 --run <run dir>
"""

from .config import RunPaths, data_dir, run_dir
from .registry import MODEL_KEYS, ModelSpec, get_spec
from .splits import DataSplit, load_split, make_split, save_split

__all__ = [
    "DataSplit",
    "MODEL_KEYS",
    "ModelSpec",
    "RunPaths",
    "data_dir",
    "get_spec",
    "load_split",
    "make_split",
    "run_dir",
    "save_split",
]
