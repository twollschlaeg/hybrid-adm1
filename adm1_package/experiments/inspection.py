"""Single reactors up close: the mechanistic variants, a trained run, the data.

The inspection helpers that ``scripts/manuscript_figures.py`` and
``scripts/si_figures.py`` use. What they show is computed the way the evaluation
computes it:

* a reactor is chosen by its label (``LaRA-temp_F3``), never by a position that
  depends on which campaigns were loaded;
* a run's fold is opened as evaluation opens it
  (:func:`~adm1_package.experiments.evaluation.open_fold`): the input scaler
  refitted on the fold's training reactors, the hybrid rebuilt from the fold's
  own record, the mechanistic model started as that hybrid is;
* errors are scored on the points evaluation scores
  (:func:`~adm1_package.experiments.metrics.score_solution`): not the initial
  point, not the spin-up, not the padded tail. The figures shade the spin-up.

Only runs that the current code can restore are offered (:func:`restorable_runs`), and
the ensembles of them that ``evaluate.py --ensemble`` made: these are simulated by their
member runs and show the members' mean prediction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..utils.plot_style import COLORS, LINES, WIDTH_DOUBLE, figure_style, shade_spin_up
from ..utils.plotting import axis_label, describe, forcing_series, target_series
from .calibrations import campaign_forcing
from .config import data_dir as default_data_dir
from .config import run_dir
from .ensembles import ENSEMBLE_FILE, ensemble_members, ensemble_source, is_ensemble, member_runs
from .evaluation import fold_settings, open_fold
from .figures import draw_folds
from .metrics import objective_metrics, score_solution
from .predictions import FoldPrediction, mean_prediction, predict_reactor
from .registry import get_spec
from .splits import DataSplit, campaign_of, load_run_split
from .training import Hyperparameters

__all__ = [
    "SPIN_UP_DAYS",
    "VARIANT_LINES",
    "Simulation",
    "RunInspection",
    "sample",
    "simulate_mechanistic",
    "restorable_runs",
    "latest_run",
    "load_run_split",
    "held_out_fold",
    "role_in_fold",
    "simulate_run",
    "scores",
    "overlay",
    "fold_overlay",
    "forcing_frame",
    "measurement_frame",
]

#: The days the training loss leaves out, and with it the evaluation's scores.
SPIN_UP_DAYS = Hyperparameters().spin_up_days

#: One line style per variant, the same in every figure whichever variants it
#: shows. Okabe-Ito blue, green and vermillion, and reddish purple for the full
#: ADM1: each at least 3:1 against white and apart under colour-vision
#: deficiency; the dash carries it in greyscale.
VARIANT_LINES: dict[str, dict[str, Any]] = {
    "r2": dict(color="#0072B2", linestyle="-", linewidth=1.3, zorder=2.2),
    "r3": dict(color="#009E73", linestyle=(0, (5, 1.8)), linewidth=1.3, zorder=2.1),
    "r4": dict(color="#D55E00", linestyle=(0, (3, 1.2, 1, 1.2)), linewidth=1.3, zorder=2.0),
    "base": dict(color="#CC79A7", linestyle=(0, (6, 1.5, 1, 1.5, 1, 1.5)), linewidth=1.3,
                 zorder=1.9),
}

#: The forcing ahead of the influent species: temperature and inflow.
_FORCING = ("T_op", "q_in")


# ---------------------------------------------------------------------------
# One simulated reactor
# ---------------------------------------------------------------------------
@dataclass
class Simulation:
    """One reactor simulated by one model, with the sample it was simulated from."""

    name: str
    #: The sample's ``x``: forcing, parameters and initial values.
    inputs: Mapping[str, Any]
    #: The sample's ``y``: measurements, their mask and the label.
    targets: Mapping[str, Any]
    #: Every entry of the model's solution, as arrays over the sample's time grid.
    solution: dict[str, np.ndarray]
    #: The measured outputs the model produces, the diagnostics (the solids) last.
    outputs: tuple[str, ...]
    #: How figures draw this model.
    style: dict[str, Any]
    #: The days from the first time point that the scores leave out, which figures
    #: shade: the spin-up its fold trained with, for a run's simulation.
    spin_up_days: float = SPIN_UP_DAYS

    @property
    def label(self) -> str:
        return self.targets["label"]

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self.inputs["input"]["time"], dtype=float)

    @property
    def length(self) -> int:
        """Points before the padded tail (the time grid is padded with its last day)."""
        return _length(self.times)


def _length(times: np.ndarray) -> int:
    """Points of a time grid before its padded tail, which repeats the last day."""
    return int(np.argmax(np.asarray(times))) + 1


def _data_dir(data_dir: str | Path | None) -> Path:
    """``data_dir``, else the default data directory."""
    return Path(data_dir) if data_dir is not None else default_data_dir()


def _arrays(solution: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Every entry of a solution as a float array."""
    return {name: np.asarray(value, dtype=float) for name, value in solution.items()}


def sample(dataset, label: str):
    """The sample ``(x, y)`` of the reactor ``label``."""
    return dataset[_index(dataset, label)]


def _index(dataset, label: str) -> int:
    """The position of reactor ``label`` in ``dataset``."""
    labels = list(dataset.labels)
    if label not in labels:
        raise KeyError(f"no reactor {label!r}; this dataset has {', '.join(labels)}")
    return labels.index(label)


def simulate_mechanistic(
    model_key: str,
    label: str,
    initial_state: str | None = None,
    data_dir: str | Path | None = None,
    forcing: str | None = None,
) -> Simulation:
    """One reactor simulated by a variant's mechanistic model.

    ``initial_state`` is ``"calibrated"`` (the default) or ``"warmup"``; see
    :mod:`adm1_package.utils.initial_state`. Only the reactor's campaign is
    loaded: the other campaigns change nothing but the network's input scaling,
    which a mechanistic model does not read, and the length of the padded tail.
    ``forcing`` defaults to the one the installed calibration was fitted to.
    """
    spec = get_spec(model_key)
    campaign = campaign_of(label)
    dataset = spec.dataset(_data_dir(data_dir), data_to_use=[campaign], diagnostics=True,
                           forcing=forcing or campaign_forcing(_data_dir(data_dir), campaign))
    inputs, targets = sample(dataset, label)
    solution, _ = spec.mechanistic(initial_state=initial_state)(
        inputs["parameters"], inputs["initial_values"], inputs["input"]
    )
    return Simulation(
        name=spec.display_name, inputs=inputs, targets=targets, solution=_arrays(solution),
        outputs=spec.drawn_outputs, style=dict(VARIANT_LINES[spec.key]),
    )


# ---------------------------------------------------------------------------
# Runs and folds
# ---------------------------------------------------------------------------
def _restorable(fold: Path) -> bool:
    """Whether the fold records everything its model is rebuilt from."""
    path = fold / "hparams.json"
    if not path.exists():
        return False
    try:
        fold_settings(SimpleNamespace(config=json.loads(path.read_text(encoding="utf-8"))))
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _restorable_run(run: Path) -> bool:
    folds = [fold for fold in run.glob("fold_*") if fold.is_dir()]
    return (run / "split.json").exists() and bool(folds) and all(map(_restorable, folds))


def _restorable_ensemble(directory: Path) -> bool:
    """Whether every member run of an ensemble is there and restorable; a transfer of
    another variant's runs, retired, is not."""
    try:
        ensemble_source(directory)
        members = ensemble_members(directory)
    except (OSError, KeyError, ValueError):
        return False
    return bool(members) and all(map(_restorable_run, members))


def _created(run: Path) -> float:
    """When a run was started: its metadata's timestamp, else its split file's; an
    ensemble, when it was made."""
    if is_ensemble(run):
        return (run / ENSEMBLE_FILE).stat().st_mtime
    try:
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        return datetime.fromisoformat(metadata["timestamp_utc"]).timestamp()
    except (OSError, KeyError, TypeError, ValueError):
        return (run / "split.json").stat().st_mtime


def restorable_runs(model_key: str, root: str | Path | None = None) -> list[Path]:
    """The variant's runs under ``root`` that this code can restore, newest first.

    ``root`` defaults to the run directory (``$ADM1_RUN_DIR`` or ``runs/``). A run
    qualifies if it kept its split and every fold recorded the model it trained;
    runs of earlier versions of this code are refused (see
    :func:`~adm1_package.experiments.evaluation.fold_settings`). An ensemble
    directory qualifies if every run it lists does.
    """
    base = Path(root if root is not None else run_dir()) / get_spec(model_key).key
    if not base.is_dir():
        return []
    runs = []
    for run in base.iterdir():
        if not (run / "split.json").exists():
            continue
        if _restorable_ensemble(run) if is_ensemble(run) else _restorable_run(run):
            runs.append(run)
    return sorted(runs, key=_created, reverse=True)


def latest_run(model_key: str, root: str | Path | None = None) -> Path:
    """The newest trained run of the variant that this code can restore; never an
    ensemble, which :func:`restorable_runs` lists as well."""
    runs = [run for run in restorable_runs(model_key, root) if not is_ensemble(run)]
    if not runs:
        base = Path(root if root is not None else run_dir()) / get_spec(model_key).key
        raise FileNotFoundError(
            f"no run under {base} that this code can restore; "
            f"train one with scripts/train_cv.py --model {get_spec(model_key).key}"
        )
    return runs[0]


def held_out_fold(split: DataSplit, label: str) -> int | None:
    """The fold that validated on ``label``; ``None`` for a test reactor, which
    every fold held out."""
    for fold in range(1, split.k_folds + 1):
        if label in split.validation_labels(fold):
            return fold
    if label in split.test:
        return None
    raise KeyError(f"{label!r} is not in this split; it has {', '.join(_split_labels(split))}")


def role_in_fold(split: DataSplit, fold: int, label: str) -> str:
    """What ``label`` was to ``fold``: ``"train"``, ``"validation"`` or ``"test"``."""
    if label in split.validation_labels(fold):
        return "validation"
    if label in split.test:
        return "test"
    if label in split.training_labels(fold):
        return "train"
    raise KeyError(f"{label!r} is not in this split; it has {', '.join(_split_labels(split))}")


def _split_labels(split: DataSplit) -> list[str]:
    return sorted(split.development + list(split.test))


@dataclass
class RunInspection:
    """One reactor simulated by one fold of a run, hybrid and mechanistic."""

    model_key: str
    run: Path
    fold: int
    #: What the reactor was to this fold: ``"train"``, ``"validation"`` or ``"test"``.
    role: str
    #: The correction, initial state, spin-up and forcing the fold trained with.
    settings: dict[str, Any]
    hybrid: Simulation
    mechanistic: Simulation
    #: The same simulation in the form the run figures take, e.g.
    #: :func:`~adm1_package.experiments.figures.kinetic_trajectories`.
    record: FoldPrediction
    #: The checkpoint restored and its epoch, e.g. ``("last", 200)``.
    checkpoint: tuple[str, int]


def simulate_run(
    model_key: str,
    label: str,
    run: str | Path | None = None,
    fold: int | None = None,
    data_dir: str | Path | None = None,
    checkpoint: str = "last",
    benchmarks: str | Path | None = None,
    held_out_starts: str = "fitted",
) -> RunInspection:
    """One reactor simulated by a trained fold and by its mechanistic benchmark.

    ``run`` defaults to the newest restorable run of the variant, and ``fold`` to
    the one that held the reactor out, so the hybrid is shown on a reactor it
    never trained on. A test reactor, which every fold held out, defaults to
    fold 1. ``checkpoint`` is the one evaluation scores, the final epoch; ``"best"``
    shows the epoch the held-out reactors chose. A run built on per-fold
    calibrations is shown on its fold's, beside MECH-CV; ``benchmarks`` says where
    they are (default :func:`~adm1_package.experiments.config.benchmarks_dir`), and
    ``held_out_starts`` where the fold's held-out reactors start, as evaluation
    starts them (``evaluation.open_fold``).

    ``run`` may be an ensemble directory: each member run opens its fold and the hybrid
    shown is their mean prediction (``settings["members"]``).
    """
    spec = get_spec(model_key)
    run = Path(run) if run is not None else latest_run(spec.key)
    split = load_run_split(run)
    if fold is None:
        fold = held_out_fold(split, label) or 1
    role = role_in_fold(split, fold, label)

    members = member_runs(run, spec.key)
    opened = [open_fold(spec, member, split, fold, _data_dir(data_dir), checkpoint=checkpoint,
                        benchmarks=None if benchmarks is None else Path(benchmarks),
                        held_out_starts=held_out_starts)
              for member in members]
    index = _index(opened[0].dataset, label)
    record = mean_prediction([predict_reactor(member, index, role) for member in opened])
    inputs, targets = opened[0].dataset[index]
    settings = dict(opened[0].settings)
    if is_ensemble(run):
        settings["members"] = [member.name for member in members]

    def simulation(name, solution, style):
        return Simulation(name=name, inputs=inputs, targets=targets, solution=solution,
                          outputs=spec.drawn_outputs, style=dict(style),
                          spin_up_days=settings["spin_up_days"])

    return RunInspection(
        model_key=spec.key, run=run, fold=fold, role=role, settings=settings,
        hybrid=simulation(f"{spec.display_name} hybrid", record.hybrid, LINES["ude"]),
        mechanistic=simulation(f"{spec.display_name} mechanistic", record.mechanistic,
                               LINES["mech"]),
        record=record,
        checkpoint=opened[0].checkpoint,
    )


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def scores(
    simulations: Sequence[Simulation],
    spin_up_days: float = SPIN_UP_DAYS,
    metric: str = "rmse",
) -> pd.DataFrame:
    """One error per measured output and model, scored as evaluation scores.

    ``metric`` is any of :func:`~adm1_package.experiments.metrics.objective_metrics`'
    (``rmse``, ``mae``, ``bias``, ``r2``, ``mape``, ``smape``). An output a model
    does not produce is left empty.
    """
    table: dict[str, dict[str, float]] = {}
    for simulation in simulations:
        outputs = [name for name in simulation.outputs if name in simulation.solution
                   and name in simulation.targets["object_indices"]]
        scored = score_solution(simulation.solution, simulation.inputs, simulation.targets,
                                outputs, spin_up_days=spin_up_days)
        values = objective_metrics(scored.predictions, scored.targets, scored.mask, outputs)
        for name in outputs:
            table.setdefault(name, {})[simulation.name] = values[name][metric]
    return pd.DataFrame.from_dict(table, orient="index").reindex(
        columns=[simulation.name for simulation in simulations])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def overlay(
    simulations: Sequence[Simulation],
    components: Sequence[str] | None = None,
    title: str | None = None,
    ncols: int = 3,
    panel_height: float = 1.7,
):
    """Models of one reactor against its measurements, a panel per output.

    One panel per component (by default every measured output of the models, in
    their order), then the temperature and the inflow, each on its own axis. Each
    model is drawn in its ``style``; the measurements come from the first model
    whose sample measured the component. The first model's ``spin_up_days`` are
    shaded on each component's panel.

    Returns the figure; nothing is saved and nothing is shown.
    """
    if not simulations:
        raise ValueError("no simulations to draw")

    def draw(ax, component):
        for simulation in simulations:
            if component in simulation.solution:
                ax.plot(simulation.times[: simulation.length],
                        simulation.solution[component][: simulation.length],
                        label=simulation.name, **simulation.style)

    return _output_figure(simulations, draw, components, title, ncols, panel_height)


def fold_overlay(
    inspections: Sequence[RunInspection],
    components: Sequence[str] | None = None,
    title: str | None = None,
    ncols: int = 3,
    panel_height: float = 1.7,
):
    """Every fold of a run on one reactor, hybrid and mechanistic, against its
    measurements.

    :func:`overlay` for all folds at once, drawn as
    :func:`~adm1_package.experiments.figures.kinetic_trajectories` draws the learned
    corrections: per model a band over the folds' range and a thin line per fold.
    The fold that held the reactor out is drawn bold in the model's style, as it
    alone shows the model on a reactor it never trained on; when no single fold
    held it out (a test reactor, held out by every fold), the folds' mean is. A
    model every fold simulates alike, such as a mechanistic model on the installed
    calibration rather than per-fold ones, is drawn once.

    ``inspections`` are :func:`simulate_run` of the same run and reactor, one per
    fold. Returns the figure; nothing is saved and nothing is shown.
    """
    if not inspections:
        raise ValueError("no folds to draw")
    labels = {one.hybrid.label for one in inspections}
    if len(labels) > 1:
        raise ValueError(f"the folds simulate different reactors: {', '.join(sorted(labels))}")
    held_out = [one for one in inspections if one.role != "train"]
    single = held_out[0] if len(held_out) == 1 else None

    def draw(ax, component):
        for model in ("mechanistic", "hybrid"):
            simulations = [getattr(one, model) for one in inspections]
            first = simulations[0]
            if component not in first.solution:
                continue
            length = min(simulation.length for simulation in simulations)
            times = first.times[:length]
            curves = np.stack([simulation.solution[component][:length]
                               for simulation in simulations])
            if single is None:
                draw_folds(ax, times, curves, first.style, first.name)
            else:
                draw_folds(ax, times, curves, first.style, first.name,
                           held_out=list(inspections).index(single),
                           held_out_label=f"{first.name}, fold {single.fold} ({single.role})")

    if title is None:
        run = inspections[0].run
        title = f"{inspections[0].hybrid.label}: {len(inspections)} folds of {run.name}"
    # three legend columns keep each model's bold line above its other folds
    return _output_figure([one.hybrid for one in inspections], draw, components, title,
                          ncols, panel_height, legend_columns=3)


def _output_figure(simulations, draw, components, title, ncols, panel_height,
                   legend_columns=4):
    """The grid :func:`overlay` and :func:`fold_overlay` share: ``draw(ax, component)``
    draws the models on each output's panel, then come the measurements (from the
    first of ``simulations`` that measured the component) and the first's spin-up,
    shaded, then the temperature, the inflow and one legend."""
    import matplotlib.pyplot as plt

    if components is None:
        components = list(dict.fromkeys(name for simulation in simulations
                                        for name in simulation.outputs))
    panels = list(components) + ["Temperature", "Inflow"]
    nrows = int(np.ceil(len(panels) / ncols))
    first = simulations[0]
    length = first.length
    times = first.times[:length]

    with figure_style():
        figure, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_DOUBLE, panel_height * nrows),
                                    layout="constrained", squeeze=False)
        flat = axes.ravel()

        for ax, component in zip(flat, components):
            limit = describe(component).detection_limit
            if limit is not None:
                ax.axhspan(0.0, limit, color="#000000", alpha=0.06, zorder=0,
                           label="Below detection limit")
            draw(ax, component)
            measured = next((simulation for simulation in simulations
                             if component in simulation.targets["object_indices"]), None)
            if measured is not None:
                values, valid = target_series(measured.targets, component)
                ax.plot(measured.times[valid], values[valid], label="Measured",
                        **LINES["data_line"])
            shade_spin_up(ax, times, first.spin_up_days)
            ax.set_title(describe(component).title)
            ax.set_xlabel("Time [d]")
            ax.set_ylabel(axis_label(component))

        _, celsius, inflow = forcing_series(first.inputs)
        for ax, name, values, unit in (
            (flat[len(components)], "Temperature", celsius, r"$T$ [°C]"),
            (flat[len(components) + 1], "Inflow", inflow, r"$q_{in}$ [L/d]"),
        ):
            ax.plot(times, values[:length], color=COLORS["data"], linewidth=1.1)
            ax.set_title(name)
            ax.set_xlabel("Time [d]")
            ax.set_ylabel(unit)

        for ax in flat[len(panels):]:
            ax.axis("off")

        handles: dict[str, Any] = {}
        for ax in flat[: len(components)]:
            for handle, name in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(name, handle)
        if handles:
            figure.legend(list(handles.values()), list(handles), loc="outside lower center",
                          ncols=min(len(handles), legend_columns))
        figure.suptitle(title if title is not None else first.label)

    return figure


# ---------------------------------------------------------------------------
# Tables of the data
# ---------------------------------------------------------------------------
def forcing_frame(dataset, influent_species: Sequence[str]) -> pd.DataFrame:
    """Every reactor's forcing as a long table, one row per reactor, day and variable.

    Columns ``label``, ``campaign``, ``time``, ``variable``, ``value``. The
    variables are the forcing's columns in order: ``T_op`` (in °C; the datasets
    hold Kelvin), ``q_in`` (L/d), then ``influent_species`` (g/L), which is
    :attr:`~adm1_package.experiments.registry.ModelSpec.influent_species` of the
    variant the dataset belongs to. The padded tail is dropped.
    """
    import diffrax as dx
    import jax

    names = (*_FORCING, *influent_species)
    frames = []
    for index in range(len(dataset)):
        inputs, targets = dataset[index]
        forcing = inputs["input"]
        times = np.asarray(forcing["time"], dtype=float)
        length = _length(times)
        interpolation = dx.CubicInterpolation(forcing["time"], forcing["coeffs"])
        values = np.array(jax.vmap(interpolation.evaluate)(forcing["time"][:length]), dtype=float)
        if values.shape[1] != len(names):
            raise ValueError(f"{targets['label']}: the forcing has {values.shape[1]} columns, "
                             f"expected {len(names)} ({', '.join(names)})")
        values[:, 0] -= 273.15
        for column, name in enumerate(names):
            frames.append(pd.DataFrame({
                "label": targets["label"], "campaign": campaign_of(targets["label"]),
                "time": times[:length], "variable": name, "value": values[:, column],
            }))
    return pd.concat(frames, ignore_index=True)


def measurement_frame(dataset) -> pd.DataFrame:
    """Every reactor's measured outputs as a long table, one row per reactor, day
    and output.

    Columns ``label``, ``campaign``, ``time``, ``output``, ``value``. These are
    the records the models are scored on: the lab values on their lab days and
    the gas on the days it was read (``plotting.target_series``). The padded tail
    is dropped.
    """
    frames = []
    for index in range(len(dataset)):
        inputs, targets = dataset[index]
        times = np.asarray(inputs["input"]["time"], dtype=float)
        for name in targets["object_indices"]:
            values, valid = target_series(targets, name)
            frames.append(pd.DataFrame({
                "label": targets["label"], "campaign": campaign_of(targets["label"]),
                "time": times[valid], "output": name, "value": values[valid],
            }))
    return pd.concat(frames, ignore_index=True)
