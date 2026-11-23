"""The canonical train / validation / test split.

Design notes
------------
The split is defined **once**, written to disk as sample *labels*, and read back
by training, tuning and evaluation. It is deliberately not recomputed from a
seed at each step: folds derived by shuffling sample indices, e.g. with
``np.random.default_rng(42).shuffle(np.arange(len(dataset)))``, depend silently on
the order in which the loader enumerates files. Labels are stable, indices are not.

Two levels:

``test``
    A hold-out set that no training run and no hyperparameter search ever sees.
    One reactor per experimental campaign by default, so the test set covers a
    temperature shift, a substrate substitution and a rice-bran setup.

``folds``
    k-fold cross-validation over the remaining reactors. Hyperparameters are
    selected on the mean fold-validation loss, and the fold models give the
    mean +/- std figures reported in the paper. Folds are stratified by campaign
    so that no fold is drawn entirely from one experiment series.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = ["DataSplit", "make_split", "nested_split", "save_split", "load_split",
           "load_run_split", "default_split_path", "campaign_of"]


def campaign_of(label: str) -> str:
    """Experimental campaign a sample belongs to (``LaRA-temp_F3`` -> ``LaRA-temp``)."""
    return label.rsplit("_", 1)[0]


@dataclass
class DataSplit:
    """A split expressed in sample labels, plus the recipe that produced it."""

    test: list[str]
    folds: list[list[str]]
    seed: int
    stratified: bool = True
    note: str = ""
    all_labels: list[str] = field(default_factory=list)

    # -- derived views -----------------------------------------------------
    @property
    def k_folds(self) -> int:
        return len(self.folds)

    @property
    def development(self) -> list[str]:
        """Every label that is not in the test set (union of the folds)."""
        return [label for fold in self.folds for label in fold]

    def training_labels(self, fold_index: int) -> list[str]:
        """Labels used for training in fold ``fold_index`` (1-based)."""
        return [
            label
            for index, fold in enumerate(self.folds, start=1)
            if index != fold_index
            for label in fold
        ]

    def validation_labels(self, fold_index: int) -> list[str]:
        """Labels held out for validation in fold ``fold_index`` (1-based)."""
        return list(self.folds[fold_index - 1])

    def same_assignment(self, other: "DataSplit") -> bool:
        """Whether ``other`` holds out the same reactors in every fold and in the test set."""
        return (sorted(self.test) == sorted(other.test)
                and [sorted(fold) for fold in self.folds] == [sorted(fold) for fold in other.folds])

    # -- label -> index resolution ----------------------------------------
    def indices(self, labels: Sequence[str], dataset_labels: Sequence[str]) -> list[int]:
        """Resolve labels against a dataset's own label order.

        Raises if a label is missing, which is the point: a silently dropped
        sample would otherwise shrink a fold without anybody noticing.
        """
        lookup = {label: index for index, label in enumerate(dataset_labels)}
        missing = [label for label in labels if label not in lookup]
        if missing:
            raise KeyError(
                f"split refers to samples that the dataset does not contain: {missing}. "
                f"Dataset labels: {list(dataset_labels)}"
            )
        return [lookup[label] for label in labels]

    def fold_indices(
        self, fold_index: int, dataset_labels: Sequence[str]
    ) -> tuple[list[int], list[int]]:
        """``(train_indices, validation_indices)`` for one fold."""
        return (
            self.indices(self.training_labels(fold_index), dataset_labels),
            self.indices(self.validation_labels(fold_index), dataset_labels),
        )

    def test_indices(self, dataset_labels: Sequence[str]) -> list[int]:
        """Indices of the split's test reactors in ``dataset_labels``."""
        return self.indices(self.test, dataset_labels)

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        # Copies, so that editing the dictionary cannot change the split.
        return {
            "note": self.note,
            "seed": self.seed,
            "stratified": self.stratified,
            "k_folds": self.k_folds,
            "all_labels": list(self.all_labels),
            "test": list(self.test),
            "folds": {f"fold_{i}": list(fold) for i, fold in enumerate(self.folds, start=1)},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DataSplit":
        folds = payload["folds"]
        ordered = [folds[f"fold_{i}"] for i in range(1, payload["k_folds"] + 1)]
        return cls(
            test=list(payload["test"]),
            folds=[list(fold) for fold in ordered],
            seed=int(payload["seed"]),
            stratified=bool(payload.get("stratified", True)),
            note=payload.get("note", ""),
            all_labels=list(payload.get("all_labels", [])),
        )

    def summary(self) -> str:
        lines = [
            f"{self.k_folds}-fold split, seed={self.seed}, "
            f"stratified={'yes' if self.stratified else 'no'}",
            f"  test ({len(self.test)}): {', '.join(self.test)}",
        ]
        for index, fold in enumerate(self.folds, start=1):
            lines.append(f"  fold {index} validation ({len(fold)}): {', '.join(fold)}")
        return "\n".join(lines)


def _stratified_folds(
    labels: Sequence[str], k_folds: int, rng: np.random.Generator
) -> list[list[str]]:
    """Deal shuffled labels round-robin into folds, campaign by campaign.

    Dealing per campaign keeps each campaign spread as evenly as the counts
    allow, rather than letting one fold consist only of, say, rice-bran
    reactors. The starting fold rotates per campaign so that small campaigns do
    not all pile into fold 1.
    """
    folds: list[list[str]] = [[] for _ in range(k_folds)]
    offset = 0
    for campaign in sorted({campaign_of(label) for label in labels}):
        members = sorted(label for label in labels if campaign_of(label) == campaign)
        rng.shuffle(members)
        for position, label in enumerate(members):
            folds[(offset + position) % k_folds].append(label)
        offset = (offset + len(members)) % k_folds
    return [sorted(fold) for fold in folds]


def _random_folds(
    labels: Sequence[str], k_folds: int, rng: np.random.Generator
) -> list[list[str]]:
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return [sorted(fold.tolist()) for fold in np.array_split(np.array(shuffled), k_folds)]


def make_split(
    labels: Iterable[str],
    k_folds: int = 4,
    seed: int = 42,
    test_per_campaign: int = 1,
    stratified: bool = True,
    note: str = "",
) -> DataSplit:
    """Build the canonical split from a dataset's label list.

    Parameters
    ----------
    labels:
        Sample labels, e.g. ``LaRA-temp_F1``. Order is irrelevant; the split is
        sorted internally so the same labels always give the same split.
    k_folds:
        Number of cross-validation folds over the non-test reactors.
    seed:
        Seed of the shuffling. Recorded in the split file.
    test_per_campaign:
        How many reactors per campaign are held out as the test set. ``0``
        disables the test set (cross-validation over everything).
    stratified:
        Spread each campaign across the folds instead of shuffling globally.
    """
    labels = sorted(set(labels))
    if not labels:
        raise ValueError("no labels given")
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2")

    rng = np.random.default_rng(seed)

    test: list[str] = []
    if test_per_campaign > 0:
        for campaign in sorted({campaign_of(label) for label in labels}):
            members = sorted(label for label in labels if campaign_of(label) == campaign)
            if len(members) <= test_per_campaign:
                raise ValueError(
                    f"campaign {campaign!r} has only {len(members)} samples; "
                    f"cannot hold out {test_per_campaign}"
                )
            picked = rng.choice(members, size=test_per_campaign, replace=False)
            test.extend(sorted(picked.tolist()))
    test = sorted(test)

    development = [label for label in labels if label not in set(test)]
    if k_folds > len(development):
        raise ValueError(
            f"{k_folds} folds requested but only {len(development)} non-test samples"
        )

    builder = _stratified_folds if stratified else _random_folds
    folds = builder(development, k_folds, rng)

    empty = [index for index, fold in enumerate(folds, start=1) if not fold]
    if empty:
        raise ValueError(f"folds {empty} came out empty; reduce k_folds")

    return DataSplit(
        test=test,
        folds=folds,
        seed=seed,
        stratified=stratified,
        note=note,
        all_labels=labels,
    )


def nested_split(outer: DataSplit, outer_fold: int, k_folds: int = 4,
                 seed: int | None = None) -> DataSplit:
    """The inner split of outer fold ``outer_fold``, for a nested hyperparameter search.

    Its test set is everything the outer fold must not learn from: the outer fold's
    held-out reactors and the outer test set. Its ``k_folds`` folds divide the outer
    fold's training reactors, stratified by campaign as the outer folds are. A search
    on it selects without the reactors the outer fold is scored on, so that score
    stays an honest estimate. ``seed`` defaults to the outer seed plus ``outer_fold``,
    so every outer fold deals its reactors afresh.
    """
    if not 1 <= outer_fold <= outer.k_folds:
        raise ValueError(f"outer fold {outer_fold} does not exist; the split has {outer.k_folds}")
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2")
    development = sorted(outer.training_labels(outer_fold))
    if k_folds > len(development):
        raise ValueError(f"{k_folds} inner folds requested but outer fold {outer_fold} trains "
                         f"on only {len(development)} reactors")
    seed = outer.seed + outer_fold if seed is None else seed
    folds = _stratified_folds(development, k_folds, np.random.default_rng(seed))
    test = sorted(set(outer.test) | set(outer.validation_labels(outer_fold)))
    return DataSplit(
        test=test,
        folds=folds,
        seed=seed,
        stratified=True,
        note=(f"Inner split of outer fold {outer_fold} of a {outer.k_folds}-fold split (seed "
              f"{outer.seed}) for a nested search. Its test reactors are the ones that outer "
              "fold is scored on, which the search never sees; its folds divide the outer "
              "fold's training reactors."),
        all_labels=sorted(outer.all_labels or [*outer.development, *outer.test]),
    )


def default_split_path() -> Path:
    """Where the canonical split lives when no path is given."""
    from .config import PROJECT_ROOT

    return PROJECT_ROOT / "configs" / "split.json"


def save_split(split: DataSplit, path: str | Path | None = None) -> Path:
    """Write ``split`` to ``path`` (default: the canonical ``configs/split.json``)."""
    path = Path(path) if path is not None else default_split_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    return path


def load_split(path: str | Path | None = None) -> DataSplit:
    """Read a split file (default: the canonical ``configs/split.json``)."""
    path = Path(path) if path is not None else default_split_path()
    if not path.exists():
        raise FileNotFoundError(
            f"no split file at {path}. Create one with: python scripts/make_splits.py"
        )
    return DataSplit.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_run_split(run_directory: str | Path) -> DataSplit:
    """The split a run (or an ensemble of runs) was trained with, which it keeps as
    ``split.json``: never recomputed, so its folds hold out the reactors they did."""
    path = Path(run_directory) / "split.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Runs created before the split was persisted cannot be "
            "evaluated reliably, because the fold assignment cannot be recovered."
        )
    return DataSplit.from_dict(json.loads(path.read_text(encoding="utf-8")))
