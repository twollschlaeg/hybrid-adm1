"""Score cross-validated hybrid models against their mechanistic counterparts.

The evaluation walks the folds of a run, restores each fold's network, and
scores the hybrid and the mechanistic variant on exactly the same reactors:

``train``
    the fold's training reactors -- goodness of fit;
``validation``
    the fold's held-out reactors -- the cross-validated generalization estimate;
``test``
    the split's hold-out reactors, identical for every fold and never used for
    training or hyperparameter selection.

Everything is reported per objective as well as pooled, and hybrid-vs-mechanistic
differences come with a paired test over folds.

Each fold is scored at its final epoch (``checkpoint="last"``). The epoch with the
lowest loss on the held-out reactors (``"best"``) was chosen by the reactors it would
be scored on, so its scores are optimistic and kept apart (:func:`results_name`).

Every hybrid is built on its fold's per-fold calibration, so scoring it needs the
per-fold benchmarks (``scripts/calibrate/calibrate_folds.py``, :mod:`.calibrations`).
Caveats:

* The installed benchmarks (``MECH``) are calibrated per campaign on all reactors of
  that campaign, so on every fold they have seen the held-out reactors while the
  hybrid model has not: a ceiling, not a fair comparison. The per-fold benchmarks
  (:data:`PER_FOLD_BENCHMARK`) are calibrated on each fold's training reactors only,
  as the fold's hybrid is trained.
* ``MECH`` starts a reactor from its installed start values, which the calibration
  fitted on that reactor's whole record. MECH-CV and the hybrid built on it start
  the fold's held-out reactors from values fitted on their first days only, with that
  calibration (:mod:`.held_out_starts`); ``held_out_starts="installed"`` starts them
  as MECH does, for comparison.
* On the ``test`` split the mechanistic scores are identical across folds (the
  same calibrated model is scored on the same reactors), so the fold-wise spread
  there measures the variability of the *hybrid* models alone, and the paired
  test compares k fold models against one fixed baseline. Read the test row as
  "how a fold model does on untouched reactors", not as k independent replicates.

An experiment is reported for the mean prediction of a few seeds, an ensemble
(:mod:`.ensembles`, :func:`evaluate_ensemble`), whose directory stands in for a run.

Runs record every option they were trained with. Those retired before this release are
read as :mod:`.retired` says: at the value that changed nothing they are ignored, at any
other the run is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .calibrations import (
    benchmark_model,
    check_calibration,
    check_forcing,
    fold_benchmark_data,
    installed_forcing,
)
from .ensembles import check_members, ensemble_members, ensemble_source
from .metrics import (
    count_parameters,
    paired_comparison,
    pooled_metrics,
    score_solution,
    simulate_sample,
)
from .registry import ModelSpec, get_spec
from .retired import drop_retired, retired
from .splits import DataSplit

__all__ = [
    "evaluate_run",
    "evaluate_ensemble",
    "ensemble_samples",
    "aggregate_folds",
    "compare_variants",
    "restore_fold",
    "open_fold",
    "fold_settings",
    "EvaluationResult",
    "RestoredFold",
    "CHECKPOINTS",
    "results_name",
    "PER_FOLD_BENCHMARK",
    "hybrid_data_dir",
]

#: Which of a fold's checkpoints is scored. ``last`` is the network after the fixed
#: number of epochs. ``best`` is the epoch with the lowest loss on the fold's
#: held-out reactors: selected on the reactors it is then scored on, so its scores
#: are optimistic, and they are kept apart (:func:`results_name`).
CHECKPOINTS = ("last", "best")

#: The mechanistic variant calibrated per fold on the fold's training reactors only
#: (scripts/calibrate/calibrate_folds.py). ``MECH`` is calibrated on every reactor of a
#: campaign, the fold's held-out ones included.
PER_FOLD_BENCHMARK = "MECH-CV"
#: The variants the hybrid is compared with, in table order.
BENCHMARKS = ("MECH", PER_FOLD_BENCHMARK)


def results_name(checkpoint: str, held_out_starts: str = "fitted",
                 spin_up_days: float | None = None, initial_state: str | None = None) -> str:
    """The results directory of a run scored at ``checkpoint``; scored otherwise than the
    paper's protocol, with the held-out reactors' installed starts, from a later day or from
    another initial state, in a directory of its own (``results-installed-starts``,
    ``results-from-day45``, ``results-warmup``)."""
    parts = [] if checkpoint == "last" else [checkpoint]
    if held_out_starts != "fitted":
        parts.append(f"{held_out_starts}-starts")
    if spin_up_days is not None:
        parts.append(f"from-day{float(spin_up_days):g}")
    if initial_state is not None:
        parts.append(initial_state)
    return "results" + "".join(f"-{part}" for part in parts)


@dataclass
class EvaluationResult:
    """Long-form fold rows plus the aggregates derived from them."""

    folds: pd.DataFrame
    objectives: pd.DataFrame
    aggregated: pd.DataFrame
    objectives_aggregated: pd.DataFrame
    comparisons: pd.DataFrame
    #: How the run was scored: the checkpoint and, per fold, the epoch it holds.
    settings: dict[str, Any] = field(default_factory=dict)

    def write(self, directory: Path) -> dict[str, Path]:
        """The ``cv_*.csv`` tables and ``evaluation.json`` into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, frame in (
            ("cv_fold_metrics", self.folds),
            ("cv_fold_objective_metrics", self.objectives),
            ("cv_aggregated_metrics", self.aggregated),
            ("cv_aggregated_objective_metrics", self.objectives_aggregated),
            ("cv_paired_comparisons", self.comparisons),
        ):
            path = directory / f"{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = path
        if self.settings:
            path = directory / "evaluation.json"
            path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
            written["evaluation"] = path
        return written


# ---------------------------------------------------------------------------
# Restoring a fold
# ---------------------------------------------------------------------------
#: What a fold's hparams.json records about the model it trained.
_FOLD_SETTINGS = ("correction", "initial_state", "spin_up_days")


def fold_settings(trainer) -> dict[str, Any]:
    """What a restored fold was trained as, from its ``hparams.json``.

    The correction and the initial state rebuild its UDE. The spin-up window is
    what its loss ignored, so its scores ignore it too. Evaluation starts the
    mechanistic benchmark from the same initial state, so the two differ by the
    model only.

    A fold trained with an option retired before this release is refused
    (:mod:`.retired`): with the memory, the pooled loss scale, the spline forcing (or
    none recorded, which was the spline) or on the installed calibration (or none
    recorded, which was the installed one).
    """
    from adm1_package.utils.corrections import CorrectionOptions
    from adm1_package.utils.forcing import resolve_forcing
    from adm1_package.utils.initial_state import resolve_initial_state

    where = str(getattr(trainer, "log_dir", "this checkpoint"))
    config = drop_retired(trainer.config, where)
    missing = [name for name in _FOLD_SETTINGS if name not in config]
    if missing:
        raise KeyError(
            f"{where} records no {', '.join(missing)}: it was trained with an earlier version "
            "of this code"
        )
    if config.get("calibration") != "per-fold":
        raise retired("a hybrid built on the installed calibration "
                      f"(calibration {config.get('calibration', 'installed')!r})", where)
    try:
        forcing = resolve_forcing(config.get("forcing"))
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
    return {
        "correction": CorrectionOptions.from_dict(config["correction"]),
        "initial_state": resolve_initial_state(config["initial_state"]),
        "spin_up_days": float(config["spin_up_days"]),
        "forcing": forcing,
        "calibration": config["calibration"],
        "calibration_fingerprint": config.get("calibration_fingerprint"),
    }


def restore_fold(spec: ModelSpec, fold_dir: Path, verbose: bool = False,
                 checkpoint: str = "last"):
    """Restore the trained network of one fold, in inference mode: ``(trainer, network)``.

    ``checkpoint`` is one of :data:`CHECKPOINTS`; the trainer's ``scored`` says which
    and at what epoch. The returned trainer's ``ude`` is rebuilt from the fold's
    record; see :func:`fold_settings`.
    """
    import equinox as eqx

    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {checkpoint!r}; expected one of "
                         f"{', '.join(CHECKPOINTS)}")
    fold_dir = Path(fold_dir)
    if not (fold_dir / "hparams.json").exists():
        raise FileNotFoundError(f"no trainer checkpoint in {fold_dir}")

    # Leaves the best checkpoint in trainer.state and the final one beside it.
    trainer = spec.trainer_class().load_from_checkpoint(str(fold_dir), verbose=verbose)
    if checkpoint == "last":
        network = getattr(trainer, "latest_model", None)
        if network is None:
            raise FileNotFoundError(f"no final checkpoint in {fold_dir}")
        epoch = int(trainer.checkpoint_manager.latest_step())
    else:
        network = trainer.state.model
        best = json.loads((fold_dir / "metrics" / "best_eval.json").read_text(encoding="utf-8"))
        epoch = int(best["epoch_idx"])
    trainer.scored = (checkpoint, epoch)
    settings = fold_settings(trainer)
    trainer.ude = spec.ude(settings["correction"], initial_state=settings["initial_state"])
    # Regularization is a training-time penalty; it must not enter a score.
    if hasattr(trainer, "regularization_lambda"):
        trainer.regularization_lambda = 0.0
    model = eqx.nn.inference_mode(network)
    return trainer, model


def hybrid_data_dir(settings: dict[str, Any], data_dir: Path, split: DataSplit, fold: int,
                    benchmarks: Path | None) -> Path:
    """Where a restored fold's hybrid reads its parameters and start values: the fold's
    per-fold calibration, which ``benchmarks`` must hold. ``data_dir``, the installed
    data, is not read."""
    if benchmarks is None:
        raise FileNotFoundError(
            f"fold {fold} was trained on its per-fold calibration; scoring it needs the "
            "per-fold benchmarks (scripts/calibrate/calibrate_folds.py, $ADM1_BENCHMARKS_DIR)")
    return fold_benchmark_data(benchmarks, split, fold)


def _fold_calibration(spec: ModelSpec, settings: dict[str, Any], data_dir: Path,
                      benchmarks: Path | None, split: DataSplit, fold: int, where: str,
                      held_out_starts: str, spin_up_days: float, diagnostics: bool = False):
    """Fold ``fold``'s per-fold calibration as its hybrid was built on it, ready to
    simulate: ``(dataset, arrhenius, fitted_days)``.

    Refused unless the calibration is finished and made for ``split``, was fitted to the
    forcing the fold trained with, and is the one it trained on (its fingerprint). The
    fold's held-out reactors start as ``held_out_starts`` says (:mod:`.held_out_starts`;
    ``fitted_days`` is the stretch the fitted starts came from, ``None`` as installed);
    a warm-up reads no start values. The input scaler is refitted on the fold's training
    reactors, as in training: refitted on any others, the network would read inputs
    normalized with statistics it never saw, and degrade without an error. ``arrhenius``
    is the calibration's model, which MECH-CV runs.
    """
    from .held_out_starts import held_out_view

    data = hybrid_data_dir(settings, data_dir, split, fold, benchmarks)
    check_forcing(settings, benchmarks, where)
    dataset = spec.dataset(data, diagnostics=diagnostics, forcing=settings["forcing"])
    check_calibration(dataset, settings, where)
    fitted_days = None
    if settings["initial_state"] != "warmup":
        dataset, fitted_days = held_out_view(dataset, spec.key, benchmarks, split, fold,
                                             held_out_starts, spin_up_days)
    train_indices, _ = split.fold_indices(fold, dataset.labels)
    dataset.refit_scaling(train_indices)
    return dataset, benchmark_model(benchmarks)["arrhenius"], fitted_days


@dataclass
class RestoredFold:
    """One fold of a run, ready to simulate any reactor of the dataset."""

    fold: int
    #: The variant's dataset, its input scaler refitted on the fold's training reactors.
    dataset: Any
    #: Dataset indices of the fold's ``train`` and ``validation`` reactors, and
    #: of the split's ``test`` reactors if it has any.
    indices: dict[str, list[int]]
    model: Any
    key: Any
    #: The fold's hybrid, built from its own record; see :func:`fold_settings`.
    hybrid: Any
    #: The mechanistic variant, started as the fold's hybrid is.
    mechanistic: Any
    settings: dict[str, Any]
    #: The checkpoint restored and its epoch, e.g. ``("last", 200)``.
    checkpoint: tuple[str, int]


def open_fold(
    spec: ModelSpec, run_directory: Path, split: DataSplit, fold: int, data_dir: Path,
    checkpoint: str = "last", benchmarks: Path | None = None, held_out_starts: str = "fitted",
) -> RestoredFold:
    """Restore one fold for prediction, as training saw its reactors.

    ``checkpoint`` is one of :data:`CHECKPOINTS`, as evaluation scores it. The fold is
    opened on its per-fold calibration (:func:`_fold_calibration`), so the mechanistic
    model beside its hybrid is MECH-CV. ``benchmarks`` defaults to
    :func:`~adm1_package.experiments.config.benchmarks_dir`. Its held-out reactors start
    as ``held_out_starts`` says (:mod:`.held_out_starts`), the hybrid and its anchor alike.
    The dataset carries the solids, which figures and inspections draw but no score pools.
    """
    from .held_out_starts import resolve_held_out_starts

    resolve_held_out_starts(held_out_starts)
    trainer, model = restore_fold(spec, Path(run_directory) / f"fold_{fold}",
                                  checkpoint=checkpoint)
    settings = fold_settings(trainer)
    if benchmarks is None:
        from .config import benchmarks_dir

        benchmarks = benchmarks_dir()
    dataset, arrhenius, _ = _fold_calibration(
        spec, settings, data_dir, benchmarks, split, fold, f"{run_directory}, fold {fold}",
        held_out_starts, settings["spin_up_days"], diagnostics=True)

    train_indices, validation_indices = split.fold_indices(fold, dataset.labels)
    indices = {"train": train_indices, "validation": validation_indices}
    if split.test:
        indices["test"] = split.test_indices(dataset.labels)
    return RestoredFold(
        fold=fold,
        dataset=dataset,
        indices=indices,
        model=model,
        key=trainer.state.key,
        hybrid=trainer.ude,
        mechanistic=spec.mechanistic(initial_state=settings["initial_state"], arrhenius=arrhenius),
        settings=settings,
        checkpoint=trainer.scored,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def ensemble_samples(
    spec: ModelSpec,
    dataset,
    indices: Sequence[int],
    members: Sequence[tuple[Any, Any, Any]],
    spin_up_days: float,
) -> list:
    """The reactors ``indices`` of ``dataset`` scored on the mean prediction of ``members``,
    ``(simulator, network, key)`` triples; one member is scored as
    :func:`~adm1_package.experiments.metrics.simulate_sample` scores it."""
    samples = []
    for index in indices:
        inputs, targets = dataset[int(index)]
        solutions = [simulator(inputs["parameters"], inputs["initial_values"], inputs["input"],
                               network, key)[0]
                     for simulator, network, key in members]
        mean = {name: np.mean([np.asarray(solution[name], dtype=float) for solution in solutions],
                              axis=0)
                for name in spec.outputs}
        samples.append(score_solution(mean, inputs, targets, spec.outputs,
                                      spin_up_days=spin_up_days))
    return samples


def evaluate_run(
    model_key: str,
    run_directory: Path,
    split: DataSplit,
    data_dir: Path,
    folds: Iterable[int] | None = None,
    include_test: bool = True,
    verbose: bool = True,
    checkpoint: str = "last",
    benchmarks: Path | None = None,
    held_out_starts: str = "fitted",
    spin_up_days: float | None = None,
    initial_state: str | None = None,
) -> EvaluationResult:
    """Evaluate every fold of one run and aggregate the result.

    Each fold is scored at ``checkpoint`` (:data:`CHECKPOINTS`): by default the
    network after the fixed number of epochs, which no held-out reactor chose.
    ``benchmarks``, the output of ``scripts/calibrate/calibrate_folds.py``, are required:
    every fold's hybrid is scored on the calibration it was built on, the one its
    per-fold benchmark (:data:`PER_FOLD_BENCHMARK`) uses, and is compared with both
    that and the installed benchmark.

    The fold's held-out reactors start from the values fitted on their first days
    (``held_out_starts="fitted"``, :mod:`.held_out_starts`), or as installed. The scores
    leave out the fold's spin-up, or the first ``spin_up_days``; every model starts from
    the fold's initial state, or from ``initial_state`` (``"warmup"``: its own warm-up,
    which reads no start values).
    """
    return _evaluate(model_key, [Path(run_directory)], split, data_dir, folds, include_test,
                     verbose, checkpoint, benchmarks, held_out_starts, spin_up_days,
                     initial_state)


def evaluate_ensemble(
    model_key: str,
    ensemble_directory: Path,
    split: DataSplit,
    data_dir: Path,
    folds: Iterable[int] | None = None,
    include_test: bool = True,
    verbose: bool = True,
    checkpoint: str = "last",
    benchmarks: Path | None = None,
    held_out_starts: str = "fitted",
    spin_up_days: float | None = None,
    initial_state: str | None = None,
) -> EvaluationResult:
    """Evaluate the mean prediction of an ensemble's member runs
    (:func:`~adm1_package.experiments.ensembles.create_ensemble`).

    Every fold is scored as :func:`evaluate_run` scores a run's, with the hybrid's
    prediction averaged over the members, which must have trained the same model on the
    same calibration; MECH and MECH-CV are theirs. The settings list the members and seeds.
    """
    members = ensemble_members(ensemble_directory)
    seeds = check_members(ensemble_source(ensemble_directory, model_key), members)
    result = _evaluate(model_key, members, split, data_dir, folds, include_test, verbose,
                       checkpoint, benchmarks, held_out_starts, spin_up_days, initial_state)
    result.settings.update(ensemble=[str(member) for member in members], seeds=seeds)
    return result


def _evaluate(
    model_key: str,
    runs: Sequence[Path],
    split: DataSplit,
    data_dir: Path,
    folds: Iterable[int] | None,
    include_test: bool,
    verbose: bool,
    checkpoint: str,
    benchmarks: Path | None,
    held_out_starts: str,
    spin_up_days: float | None,
    initial_state: str | None,
) -> EvaluationResult:
    """:func:`evaluate_run` for the mean prediction of ``runs``, one run or an ensemble's:
    every fold scored by :func:`_score_fold`, then aggregated over the folds."""
    from adm1_package.utils.initial_state import resolve_initial_state

    from .held_out_starts import resolve_held_out_starts

    resolve_held_out_starts(held_out_starts)
    if initial_state is not None:
        resolve_initial_state(initial_state)
    spec = get_spec(model_key)

    fold_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    epochs: dict[str, Any] = {}
    # how the folds were scored: where the hybrid's anchors came from, the spin-up, the
    # initial state, the held-out starts (and the days they were fitted on), the forcing
    protocol: dict[str, set] = {"calibration": set(), "spin_up_days": set(),
                                "initial_state": set(), "held_out_starts": set(),
                                "held_out_start_days": set(), "forcing": set()}

    fold_numbers = list(folds) if folds is not None else list(range(1, split.k_folds + 1))
    for fold in fold_numbers:
        if not all((run / f"fold_{fold}").exists() for run in runs):
            if verbose:
                print(f"[{spec.display_name}] fold {fold}: no checkpoint directory"
                      f"{'' if len(runs) == 1 else ' in every member'}, skipped")
            continue
        rows, objectives, epochs[str(fold)] = _score_fold(
            spec, runs, split, fold, data_dir, include_test, checkpoint, benchmarks,
            held_out_starts, spin_up_days, initial_state, protocol)
        fold_rows += rows
        objective_rows += objectives
        if verbose:
            summary = {row["split"]: row["nrmse"] for row in rows if row["variant"] == "UDE"}
            pretty = ", ".join(f"{name} NRMSE={value:.4f}" for name, value in summary.items())
            print(f"[{spec.display_name}] fold {fold} ({checkpoint}, epoch "
                  f"{epochs[str(fold)]}): {pretty}")

    if not fold_rows:
        raise RuntimeError(f"no folds could be evaluated under {runs[0]}")

    folds_frame = pd.DataFrame(fold_rows)
    objectives_frame = pd.DataFrame(objective_rows)

    def recorded(values):
        return values.pop() if len(values) == 1 else sorted(values)

    calibration = recorded(protocol.pop("calibration"))
    scoring = {name: recorded(values) for name, values in protocol.items() if values}
    return EvaluationResult(
        folds=folds_frame,
        objectives=objectives_frame,
        aggregated=aggregate_folds(folds_frame),
        objectives_aggregated=aggregate_folds(
            objectives_frame,
            group=["model", "model_key", "variant", "split", "objective", "reported"],
            value_columns=["rmse", "mae", "bias", "r2", "mape", "smape"],
        ),
        comparisons=compare_variants(folds_frame, objectives_frame),
        settings={"checkpoint": checkpoint, "epochs": epochs,
                  # Where the hybrid's parameters and start values came from.
                  "calibration": calibration,
                  "benchmarks": str(benchmarks),
                  # How the reactors started and from which day they were scored. The
                  # held-out starts are those of MECH-CV and of a hybrid built on it;
                  # MECH starts as installed.
                  **scoring},
    )


def _restore_members(spec: ModelSpec, runs: Sequence[Path], fold: int, checkpoint: str):
    """Fold ``fold`` of every run, restored: ``(restored, settings, epoch)``.

    ``restored`` holds ``(trainer, network)`` pairs; ``settings`` are what the first run's
    fold trained as (:func:`fold_settings`), which every other run's must share, since an
    ensemble's members are one experiment on one calibration; ``epoch`` is the epoch the
    checkpoint holds, a list where the members differ.
    """
    restored = [restore_fold(spec, run / f"fold_{fold}", verbose=False, checkpoint=checkpoint)
                for run in runs]
    scored_epochs = [trainer.scored[1] for trainer, _ in restored]
    epoch = scored_epochs[0] if len(set(scored_epochs)) == 1 else scored_epochs
    settings = fold_settings(restored[0][0])
    for (member, _), run in zip(restored[1:], runs[1:]):
        other = fold_settings(member)
        if other != settings:
            changed = ", ".join(sorted(name for name in settings if other[name] != settings[name]))
            raise ValueError(f"{run}, fold {fold}, trained another model than {runs[0]} "
                             f"({changed}): an ensemble's members are one experiment on one "
                             "calibration")
    return restored, settings, epoch


def _score_fold(spec: ModelSpec, runs: Sequence[Path], split: DataSplit, fold: int,
                data_dir: Path, include_test: bool, checkpoint: str, benchmarks: Path | None,
                held_out_starts: str, spin_up_days: float | None, initial_state: str | None,
                protocol: dict[str, set]):
    """One fold scored for the hybrid (the members' mean prediction), MECH and MECH-CV:
    ``(rows, objective_rows, epoch)``. How it was scored is added to ``protocol``.

    Every model is started and scored as the fold's hybrid is, unless ``spin_up_days`` or
    ``initial_state`` say otherwise, so the three differ by the model only.
    """
    restored, settings, epoch = _restore_members(spec, runs, fold, checkpoint)
    # An ensemble has every member's parameters.
    n_parameters = sum(count_parameters(network) for _, network in restored)
    # Each built per fold from the fold's own record, so a run is always scored under
    # the equations it was trained with.
    members = [(member.ude, network, member.state.key) for member, network in restored]
    days = settings["spin_up_days"] if spin_up_days is None else float(spin_up_days)
    if initial_state is not None and initial_state != settings["initial_state"]:
        settings = {**settings, "initial_state": initial_state}
        members = [(spec.ude(settings["correction"], initial_state=initial_state),
                    network, member.state.key)
                   for member, network in restored]
    where = f"{runs[0]}, fold {fold}"

    # The fold's own calibration, on the model and the forcing it was calibrated with:
    # the hybrid's anchors and MECH-CV.
    reference, arrhenius, fitted_days = _fold_calibration(
        spec, settings, data_dir, benchmarks, split, fold, where, held_out_starts, days)
    # MECH, on the installed calibration and the forcing it was fitted to.
    installed = spec.dataset(data_dir, forcing=installed_forcing(data_dir) or settings["forcing"])
    labels = installed.labels
    if list(reference.labels) != list(labels):
        raise ValueError(f"the per-fold benchmark of fold {fold} holds other reactors: "
                         f"{list(reference.labels)}")
    train_indices, validation_indices = split.fold_indices(fold, labels)
    installed.refit_scaling(train_indices)
    index_sets = {"train": train_indices, "validation": validation_indices}
    if include_test and split.test:
        index_sets["test"] = split.test_indices(labels)

    protocol["calibration"].add(settings["calibration"])
    protocol["spin_up_days"].add(days)
    protocol["initial_state"].add(settings["initial_state"])
    protocol["forcing"].add(settings["forcing"])
    if settings["initial_state"] == "warmup":
        protocol["held_out_starts"].add("warm-up")   # a warm-up reads no start values
    else:
        protocol["held_out_starts"].add(held_out_starts)
        if fitted_days is not None:
            protocol["held_out_start_days"].add(fitted_days)

    mechanistic = spec.mechanistic(initial_state=settings["initial_state"])
    per_fold = spec.mechanistic(initial_state=settings["initial_state"], arrhenius=arrhenius)

    def samples(variant: str, indices: Sequence[int]) -> list:
        if variant == "UDE":
            return ensemble_samples(spec, reference, indices, members, days)
        simulator, dataset = ((mechanistic, installed) if variant == "MECH"
                              else (per_fold, reference))
        return [simulate_sample(simulator, dataset, index, spec.outputs, spin_up_days=days)
                for index in indices]

    rows, objective_rows = [], []
    for split_name, indices in index_sets.items():
        for variant in ("UDE", "MECH", PER_FOLD_BENCHMARK):
            scores = pooled_metrics(
                samples(variant, indices),
                n_parameters=n_parameters if variant == "UDE" else 0,
                # Pool over the objectives the paper reports. Including the water
                # content would make the pooled RMSE a measurement of water alone.
                pooled_objectives=spec.reported_objectives,
            )
            rows.append(_fold_row(spec, variant, split_name, fold, scores))
            objective_rows += [
                {"model": spec.display_name, "model_key": spec.key, "variant": variant,
                 "split": split_name, "fold": fold, "objective": objective,
                 "reported": objective in spec.reported_objectives, **values}
                for objective, values in scores["per_objective"].items()
            ]
    return rows, objective_rows, epoch


def _fold_row(spec: ModelSpec, variant: str, split_name: str, fold: int,
              scores: dict[str, Any]) -> dict[str, Any]:
    """One row of ``cv_fold_metrics.csv``: a variant's pooled scores on one split of a fold."""
    return {
        "model": spec.display_name,
        "model_key": spec.key,
        "variant": variant,
        "split": split_name,
        "fold": fold,
        "nrmse": scores["nrmse"],
        "rmse": scores["rmse"],
        "mape": scores["mape"],
        "smape": scores["smape"],
        "r2": scores["r2"],
        "aic": scores["aic"],
        "n_parameters": scores["n_parameters"],
        "n_samples": scores["n_samples"],
        "n_observations": scores["n_observations"],
        "labels": ";".join(scores["labels"]),
    }


# ---------------------------------------------------------------------------
# Aggregation over folds
# ---------------------------------------------------------------------------
def aggregate_folds(
    frame: pd.DataFrame,
    group: Sequence[str] = ("model", "model_key", "variant", "split"),
    value_columns: Sequence[str] = ("nrmse", "rmse", "mape", "smape", "r2", "aic"),
) -> pd.DataFrame:
    """Mean, standard deviation and standard error over folds."""
    group = [column for column in group if column in frame.columns]
    value_columns = [column for column in value_columns if column in frame.columns]

    extra = {}
    if "n_parameters" in frame.columns:
        extra["n_parameters"] = ("n_parameters", "max")
    if "n_samples" in frame.columns:
        extra["n_samples"] = ("n_samples", "max")

    aggregated = frame.groupby(group, as_index=False, dropna=False).agg(
        **{
            f"{column}_{statistic}": (column, statistic)
            for column in value_columns
            for statistic in ("mean", "std")
        },
        **extra,
        n_folds=("fold", "count"),
    )
    for column in value_columns:
        aggregated[f"{column}_sem"] = aggregated[f"{column}_std"] / np.sqrt(
            aggregated["n_folds"].clip(lower=1)
        )
    return aggregated


def _paired_rows(
    frame: pd.DataFrame,
    keys: Sequence[str],
    metric: str,
    lower_is_better: bool,
    extra: dict[str, Any] | None = None,
    benchmark: str = "MECH",
) -> list[dict[str, Any]]:
    """The hybrid paired with ``benchmark`` on ``metric`` over the folds, per group of
    ``keys``; a group with fewer than two shared folds supports no test and is left out."""
    rows = []
    for values, group in frame.groupby(list(keys), as_index=False, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        hybrid = group[group["variant"] == "UDE"].set_index("fold")[metric]
        mechanistic = group[group["variant"] == benchmark].set_index("fold")[metric]
        shared = hybrid.index.intersection(mechanistic.index).sort_values()
        if len(shared) < 2:
            continue
        stats = paired_comparison(
            hybrid.loc[shared].to_numpy(),
            mechanistic.loc[shared].to_numpy(),
            lower_is_better=lower_is_better,
        )
        rows.append({**dict(zip(keys, values)), "metric": metric, "benchmark": benchmark,
                     **stats, **(extra or {})})
    return rows


def compare_variants(
    folds_frame: pd.DataFrame, objectives_frame: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Paired hybrid-vs-mechanistic statistics, per model.

    The hybrid is compared with every benchmark the frame scores (:data:`BENCHMARKS`);
    the ``benchmark`` column says which. The variants are not pooled into one test:
    they share one split, so their folds are not independent observations.
    """
    rows: list[dict[str, Any]] = []
    present = set(folds_frame["variant"]) if "variant" in folds_frame else set()

    for benchmark in (name for name in BENCHMARKS if name in present):
        for metric, lower_is_better in (("nrmse", True), ("rmse", True), ("mape", True),
                                        ("r2", False)):
            rows += _paired_rows(
                folds_frame, ["model", "model_key", "split"], metric, lower_is_better,
                {"objective": "all"}, benchmark=benchmark,
            )

        if objectives_frame is not None and not objectives_frame.empty:
            rows += _paired_rows(
                objectives_frame, ["model", "model_key", "split", "objective"], "rmse", True,
                benchmark=benchmark,
            )

    columns = [
        "model", "model_key", "split", "objective", "benchmark", "metric", "n_folds",
        "mean_hybrid", "mean_mechanistic", "mean_difference",
        "mean_relative_change_percent", "ci95_difference", "ci95_relative_change_percent",
        "wins", "ties", "sign_test_p", "wilcoxon_p",
    ]
    if not rows:
        # A single fold cannot support a paired test; return the empty frame with
        # its columns so downstream table code can report that cleanly.
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)
