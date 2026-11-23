"""What a cross-validated run predicts: its reactors simulated by its fold models.

:func:`collect_predictions` simulates a run's reactors with its fold models (an
ensemble's with the mean of its members), which :mod:`.figures`,
``scripts/manuscript_figures.py``, ``scripts/si_figures.py`` and :mod:`.inspection`
draw. Every fold is opened as evaluation opens it
(:func:`~adm1_package.experiments.evaluation.open_fold`), so a figure shows what was
scored. Re-simulating needs the reactor data, the checkpoints and the per-fold
calibrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.plotting import target_series
from .registry import get_spec
from .splits import DataSplit, campaign_of

__all__ = ["FoldPrediction", "collect_predictions", "mean_prediction", "predict_reactor"]


@dataclass
class FoldPrediction:
    """One reactor, simulated by one fold's hybrid model and by the mechanistic one."""

    label: str
    fold: int
    split: str
    times: np.ndarray
    hybrid: dict[str, np.ndarray]
    mechanistic: dict[str, np.ndarray]
    #: Calibrated mechanistic parameter values for this reactor.
    parameters: dict[str, float] = field(default_factory=dict)
    #: ``{component: (values, valid_mask)}`` for the measured quantities.
    measured: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    #: The days from the first time point that the run's scores leave out, the
    #: fold's ``spin_up_days``; the prediction figures shade them.
    spin_up_days: float = 0.0

    @property
    def campaign(self) -> str:
        return campaign_of(self.label)

    def scored(self, component: str) -> np.ndarray:
        """Where the scores count ``component``: measured, but not at the first point,
        which the initial state copies, and not in the spin-up
        (:func:`~adm1_package.experiments.metrics.score_solution`)."""
        _, valid = self.measured[component]
        scored = valid & (self.times >= self.times[0] + self.spin_up_days)
        scored[0] = False
        return scored


def _to_numpy(solution: Mapping[str, Any], keys: Sequence[str] | None) -> dict[str, np.ndarray]:
    """The entries ``keys`` of a solution as arrays; ``None`` keeps every entry."""
    out = {}
    for key in solution if keys is None else keys:
        if key in solution:
            out[key] = np.asarray(solution[key], dtype=float)
    return out


def collect_predictions(
    model_key: str,
    run_directory: Path,
    split: DataSplit,
    data_dir: Path,
    splits: Sequence[str] = ("validation",),
    folds: Sequence[int] | None = None,
    extra_keys: Sequence[str] = ("dynamic_rates", "nutrient_factors"),
    verbose: bool = True,
    checkpoint: str = "last",
    benchmarks: Path | None = None,
    held_out_starts: str = "fitted",
) -> list[FoldPrediction]:
    """Simulate a run's reactors with the fold models that held them out.

    Each fold is restored at ``checkpoint``, as evaluation scores it
    (:data:`~adm1_package.experiments.evaluation.CHECKPOINTS`), on its per-fold
    calibration, so its mechanistic curve is MECH-CV; ``benchmarks`` says where they are,
    and ``held_out_starts`` where the fold's held-out reactors start (see
    ``evaluation.open_fold``, which also refits the input scaler on the fold's training
    reactors, as training did).

    ``splits`` selects which reactors to simulate per fold:

    ``"validation"``
        the fold's held-out reactors. Each development reactor therefore appears
        exactly once across the returned records, predicted by a model that
        never saw it -- the honest picture of what the hybrid model does on an
        unseen reactor.
    ``"test"``
        the split's hold-out reactors, which every fold can be scored on. Each
        test reactor appears once per fold, so a figure can show the spread of
        the fold models on identical data.
    ``"train"``
        the fold's training reactors, for a goodness-of-fit view.

    An ensemble directory (:mod:`.ensembles`) is simulated by its member runs, and each
    reactor's hybrid curves are their mean (:func:`mean_prediction`).
    """
    from ..utils.solids import POOLS
    from .ensembles import member_runs
    from .evaluation import open_fold

    spec = get_spec(model_key)
    runs = member_runs(run_directory, spec.key)

    # the solids with the two pools they add to the model's own
    wanted_keys = (tuple(spec.drawn_outputs) + (POOLS if spec.diagnostics else ())
                   + tuple(extra_keys))

    fold_numbers = list(folds) if folds is not None else list(range(1, split.k_folds + 1))
    records: list[FoldPrediction] = []

    for fold in fold_numbers:
        if not all((run / f"fold_{fold}").exists() for run in runs):
            if verbose:
                print(f"[{spec.display_name}] fold {fold}: no checkpoint directory"
                      f"{'' if len(runs) == 1 else ' in every member'}, skipped")
            continue

        opened = [open_fold(spec, run, split, fold, data_dir, checkpoint=checkpoint,
                            benchmarks=benchmarks, held_out_starts=held_out_starts)
                  for run in runs]
        for split_name in splits:
            for index in opened[0].indices.get(split_name, []):
                records.append(mean_prediction([predict_reactor(member, index, split_name, wanted_keys)
                                                for member in opened]))

        if verbose:
            drawn = sum(1 for record in records if record.fold == fold)
            print(f"[{spec.display_name}] fold {fold}: simulated {drawn} reactors")

        del opened

    if not records:
        raise RuntimeError(f"no folds could be simulated under {run_directory}")
    return records


def predict_reactor(
    opened, index: int, split: str, keys: Sequence[str] | None = None
) -> FoldPrediction:
    """One reactor simulated by an opened fold's hybrid and mechanistic models.

    ``opened`` comes from :func:`~adm1_package.experiments.evaluation.open_fold`;
    ``split`` is what the reactor was to that fold, and ``keys`` the solution
    entries to keep (``None``: all of them).
    """
    inputs, targets = opened.dataset[int(index)]

    hybrid, _ = opened.hybrid(
        inputs["parameters"], inputs["initial_values"], inputs["input"], opened.model, opened.key
    )
    mechanistic, _ = opened.mechanistic(
        inputs["parameters"], inputs["initial_values"], inputs["input"]
    )

    return FoldPrediction(
        label=targets["label"],
        fold=opened.fold,
        split=split,
        times=np.asarray(inputs["input"]["time"], dtype=float),
        hybrid=_to_numpy(hybrid, keys),
        mechanistic=_to_numpy(mechanistic, keys),
        parameters={
            name: float(value)
            for name, value in inputs["parameters"].items()
            if np.ndim(value) == 0
        },
        measured={name: target_series(targets, name) for name in targets["object_indices"]},
        spin_up_days=float(opened.settings["spin_up_days"]),
    )


def mean_prediction(records: Sequence[FoldPrediction]) -> FoldPrediction:
    """One reactor as an ensemble predicts it: the members' hybrid outputs and learned
    corrections averaged; times, mechanistic curve and measurements are the members' own,
    which they share. A single record is returned as it is."""
    first = records[0]
    if len(records) == 1:
        return first
    for record in records[1:]:
        if (record.label, record.fold, record.split) != (first.label, first.fold, first.split):
            raise ValueError(f"cannot average {record.label} (fold {record.fold}) with "
                             f"{first.label} (fold {first.fold})")
    shared = [key for key in first.hybrid if all(key in record.hybrid for record in records)]
    return replace(first, hybrid={key: np.mean([record.hybrid[key] for record in records], axis=0)
                                  for key in shared})
