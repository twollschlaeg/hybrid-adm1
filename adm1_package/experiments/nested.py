"""Nested cross-validation for the hyperparameter search.

A search that scores its trials on the folds the paper reports selects on those
folds' held-out reactors, and their scores then flatter the selected models. Nested,
every outer fold gets a search of its own: its training reactors are divided into
inner folds (:func:`.splits.nested_split`) and the trials are scored on those. The
selection is then trained on all of the outer fold's training reactors and scored on
its held-out reactors, which neither the selection nor the training saw.

A per-fold experiment anchors each fold on a calibration of that fold's training
reactors (``scripts/calibrate/calibrate_folds.py``). The inner folds of outer fold k
are built on outer fold k's calibration. It was fitted on exactly the reactors the inner
folds divide, so the outer held-out reactors stay out of the search. It has also seen
every inner fold's validation reactors, which flatters all trials' inner scores alike.
Calibrating each inner fold anew would take four times as many calibrations as the
outer folds did.

:func:`prepare_nested` lays the searches out on disk::

    <output>/nested.json                  what was prepared, from which calibrations
    <output>/outer_<k>/split.json         the inner split of outer fold k (--split)
    <output>/outer_<k>/benchmarks/        its per-fold calibrations (--benchmarks):
        split.json, model.json            the inner split; the outer calibrations' model
        fold_<j>/<campaign>.json          status "calibrated", derived from outer fold k
        fold_<j>/data/                    a copy of outer fold k's calibrated data
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .splits import DataSplit, nested_split

__all__ = ["MANIFEST", "outer_directory", "inner_split_path", "inner_benchmarks", "prepare_nested"]

#: What :func:`prepare_nested` prepared, in its output directory.
MANIFEST = "nested.json"


def outer_directory(root: Path | str, outer_fold: int) -> Path:
    """Where the search of outer fold ``outer_fold`` finds its split and calibrations."""
    return Path(root) / f"outer_{outer_fold}"


def inner_split_path(root: Path | str, outer_fold: int) -> Path:
    """The inner split of outer fold ``outer_fold``."""
    return outer_directory(root, outer_fold) / "split.json"


def inner_benchmarks(root: Path | str, outer_fold: int) -> Path:
    """The per-fold calibrations the inner folds of outer fold ``outer_fold`` are built on."""
    return outer_directory(root, outer_fold) / "benchmarks"


def prepare_nested(split: DataSplit, benchmarks: Path | str, output: Path | str,
                   k_folds: int = 4, seed: int | None = None) -> dict[int, DataSplit]:
    """Write every outer fold's inner split and calibrations under ``output``.

    ``benchmarks`` holds the outer folds' calibrations, made for ``split``. Each inner
    fold of outer fold k gets a copy of outer fold k's calibrated data, without the
    backups of earlier calibrations. ``seed`` (default: the split's) plus k deals outer
    fold k's reactors into its ``k_folds`` inner folds. Preparing the same again
    rewrites the same; other inner splits or calibrations are refused, since searches
    may already be running on these. Returns the inner splits by outer fold.
    """
    from .calibrations import fold_benchmark_data
    from .registry import DATASETS

    benchmarks = Path(benchmarks).absolute()
    output = Path(output)
    # Every outer calibration made for this split and finished, before anything is written.
    sources = {fold: fold_benchmark_data(benchmarks, split, fold)
               for fold in range(1, split.k_folds + 1)}
    inner_splits = {fold: nested_split(split, fold, k_folds, None if seed is None else seed + fold)
                    for fold in sources}
    manifest = {
        "note": "Inner splits and calibrations of a nested hyperparameter search, written by "
                "adm1_package.experiments.nested.prepare_nested. The inner folds of outer fold "
                "k are built on outer fold k's calibration.",
        "split": split.to_dict(),
        "benchmarks": str(benchmarks),
        "inner_folds": k_folds,
        "seed": seed,
        "outer": {str(fold): {"split": str(inner_split_path(output, fold)),
                              "benchmarks": str(inner_benchmarks(output, fold))}
                  for fold in sources},
    }
    record = output / MANIFEST
    if record.exists():
        existing = json.loads(record.read_text(encoding="utf-8"))
        changed = [key for key in ("split", "benchmarks", "inner_folds", "seed")
                   if existing.get(key) != manifest[key]]
        if changed:
            raise ValueError(f"{output} holds the inner splits of another preparation (it differs "
                             f"in {', '.join(changed)}); prepare into another directory")

    for fold, inner in inner_splits.items():
        where = inner_benchmarks(output, fold)
        where.mkdir(parents=True, exist_ok=True)
        text = json.dumps(inner.to_dict(), indent=2)
        inner_split_path(output, fold).write_text(text, encoding="utf-8")
        (where / "split.json").write_text(text, encoding="utf-8")
        if (benchmarks / "model.json").exists():
            shutil.copy2(benchmarks / "model.json", where / "model.json")
        for inner_fold in range(1, inner.k_folds + 1):
            directory = where / f"fold_{inner_fold}"
            shutil.copytree(sources[fold], directory / "data", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("calibration-backup"))
            for campaign in DATASETS:
                (directory / f"{campaign}.json").write_text(json.dumps({
                    "fold": inner_fold,
                    "campaign": campaign,
                    "status": "calibrated",
                    "outer_fold": fold,
                    "derived_from": str(benchmarks / f"fold_{fold}"),
                    "note": f"outer fold {fold}'s calibration, fitted on all of this inner "
                            "split's reactors, its validation reactors included",
                }, indent=2), encoding="utf-8")
    record.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return inner_splits
