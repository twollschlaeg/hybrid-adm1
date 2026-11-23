"""Train one cross-validation fold of one ADM1 variant.

This is the single implementation behind ``scripts/train_cv.py`` and the Optuna
objective in :mod:`adm1_package.experiments.tuning`, for every variant.

Two properties are deliberate:

* **The augmentation seed is tied to the fold, not to the trial.** Optuna trials
  therefore see the same noise realizations and differ only in their
  hyperparameters, so the search compares configurations rather than luck.
* **The input scaler is refitted on the fold's training samples.** Held-out
  reactors contribute nothing to the z-score statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Subset

from .config import RunPaths
from .registry import ModelSpec
from .retired import drop_retired, retired
from .splits import DataSplit

__all__ = [
    "Hyperparameters",
    "FoldResult",
    "train_fold",
    "default_hyperparameters",
    "load_experiment",
    "apply_experiment",
    "check_experiments",
    "fold_data_dir",
    "fold_recipes",
    "loss_weights",
    "validation_interval",
    "CALIBRATIONS",
]

#: Where the reactors' parameters and start values come from: each fold's own
#: calibration of its training reactors (scripts/calibrate/calibrate_folds.py). Hybrids
#: built on the installed calibration in data/ were retired (:mod:`.retired`).
CALIBRATIONS = ("per-fold",)


def _default_correction() -> dict[str, Any]:
    # Imported here so building Hyperparameters does not pull in JAX.
    from adm1_package.utils.corrections import DEFAULT

    return DEFAULT.to_dict()


@dataclass
class Hyperparameters:
    """Everything the search may vary, plus the fixed training schedule."""

    width_size: int = 16
    depth: int = 4
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    regularization_lambda: float = 1e-4
    batch_size: int = 4
    warmup_steps: int = 100
    gradient_clip: float = 10.0
    num_epochs: int = 100
    # Curriculum over the simulated fraction of each time series; training on a
    # short prefix first keeps the early optimization out of local minima
    # (Kidger, 2022).
    length_schedule: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0)]
    )
    # Augmentation.
    noise_copies: int = 16
    noise_output: float = 0.1
    noise_input: float = 0.0
    # Overrides of the variant's loss weights (registry.py), e.g. {"pH": 0.0}: an
    # output with weight 0 is still simulated and scored, but not fitted.
    objective_weights: dict[str, float] = field(default_factory=dict)
    # lambda of a penalty lambda * mean(delta**2) on the network's raw outputs, which
    # pulls a calibrated hybrid back toward its calibration (utils/ude_trainer.py).
    correction_penalty: float = 0.0
    # Keep a checkpoint at every validation rather than the best and the last, so
    # a fold can later be scored at an epoch chosen without its held-out reactors.
    keep_checkpoints: bool = False

    # -- what the model is; an experiment file sets these -------------------
    # How network outputs become model parameters: parametrization, pruning and
    # physics options. See adm1_package/utils/corrections.py.
    correction: dict[str, Any] = field(default_factory=lambda: _default_correction())
    # Where each reactor's simulation starts: "calibrated", the per-reactor fit
    # of the *-IVs_*.csv files, or "warmup", the model's own quasi-steady state
    # at the first-week conditions (utils/initial_state.py).
    initial_state: str = "calibrated"
    # The first days follow from the initial state rather than from the model's
    # dynamics, so they enter neither the loss nor the scores.
    spin_up_days: float = 30.0
    # How the forcing runs between its daily records (utils/forcing.py FORCINGS): the
    # "smoothstep". The fold's calibration must have been fitted to the same
    # (fold_data_dir); recorded, so that a run is never scored on another.
    forcing: str = "smoothstep"
    # Degradability factors from a network of their own that reads only the
    # current forcing; the rates from one that reads everything
    # (utils/nn_architectures.py). False: one network for both.
    split_network: bool = True
    # Where the reactors' parameters and start values come from (CALIBRATIONS).
    # "per-fold" anchors fold k on the calibration of its training reactors only,
    # the one MECH-CV uses, so a calibrated hybrid at zero output *is* MECH-CV.
    calibration: str = "per-fold"
    # The bounds that per-fold calibration was searched within beyond the stored ones
    # (calibrate_folds.py --bound), e.g. {"k_hyd_ch": [0.1, 2.88]} for the fast-hydrolysis
    # anchors; a fold refuses calibrations fitted within other bounds (fold_data_dir).
    calibration_bounds: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.calibration == "installed":
            raise retired("a hybrid built on the installed calibration (calibration 'installed')")
        # Read without utils/corrections.py, which would pull in JAX; it refuses this as well.
        if (self.correction or {}).get("parametrization") == "global":
            raise retired("the global (literature-anchored) parametrization")
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"unknown calibration {self.calibration!r}; expected one of "
                             f"{', '.join(CALIBRATIONS)}")
        if self.correction_penalty < 0:
            raise ValueError(f"correction_penalty must not be negative, not {self.correction_penalty}")
        for name, bound in self.calibration_bounds.items():
            if len(bound) != 2 or not float(bound[0]) < float(bound[1]):
                raise ValueError(f"calibration_bounds of {name} must be [low, high] with "
                                 f"low < high, not {bound}")
        from adm1_package.utils.forcing import resolve_forcing

        resolve_forcing(self.forcing)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Hyperparameters":
        """A recipe from its dict form; a retired field reads as :func:`.retired.drop_retired`
        says, so recipes recorded before a field was retired still load."""
        payload = drop_retired(payload)
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise KeyError(f"unknown hyperparameters: {sorted(unknown)}")
        merged = {**asdict(cls()), **payload}
        merged["length_schedule"] = [tuple(pair) for pair in merged["length_schedule"]]
        return cls(**merged)

    def to_dict(self) -> dict[str, Any]:
        """Every field, as ``hyperparameters.json`` records it."""
        return asdict(self)


def load_experiment(path: Path | str) -> dict[str, Any]:
    """The settings an experiment file overrides, e.g.
    ``configs/experiments/fold_fast_narrow_tan.json``.

    An experiment says what the model is (network split, correction, anchors, ...)
    on top of a training recipe; a ``description`` entry is for people only.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload.pop("description", None)
    payload = drop_retired(payload, str(path))
    unknown = set(payload) - set(Hyperparameters.__dataclass_fields__)
    if unknown:
        raise KeyError(f"{path}: unknown hyperparameters {sorted(unknown)}")
    return payload


def apply_experiment(hyperparameters: Hyperparameters, experiment: dict[str, Any]) -> Hyperparameters:
    """The recipe with the experiment's settings put over it."""
    return Hyperparameters.from_dict({**hyperparameters.to_dict(), **experiment})


def fold_recipes(pattern: str, model_key: str, folds: Sequence[int],
                 experiment: dict[str, Any] | None = None,
                 num_epochs: int | None = None) -> dict[int, Hyperparameters]:
    """Each fold's own recipe, from a search per fold.

    ``pattern`` names the files, with ``{fold}`` for the fold number and ``{model}`` for
    the variant, e.g. a nested search's
    ``runs/{model}/tuning_nested-<name>_outer{fold}/best_hyperparameters.json``; a bare
    hyperparameter dictionary is read as well. The experiment goes over each, as over a
    single recipe, and ``num_epochs`` replaces the epochs. The folds may differ in what
    a search selects and in the warmup that follows the epochs, not in the model they
    train: a fold of a search over another model is refused.
    """
    from .tuning import SEARCH_SPACES

    if "{fold}" not in pattern:
        raise ValueError(f"{pattern} names no fold; put {{fold}} where the fold number goes")
    if not folds:
        raise ValueError("no folds to read hyperparameters for")
    recipes: dict[int, Hyperparameters] = {}
    sources: dict[int, Path] = {}
    for fold in folds:
        path = Path(pattern.replace("{model}", model_key).replace("{fold}", str(fold)))
        if not path.exists():
            raise FileNotFoundError(f"no hyperparameters for fold {fold}: {path} not found")
        payload = json.loads(path.read_text(encoding="utf-8"))
        recipe = Hyperparameters.from_dict(payload.get("hyperparameters", payload))
        if experiment is not None:
            recipe = apply_experiment(recipe, experiment)
        if num_epochs is not None:
            recipe.num_epochs = num_epochs
        recipes[fold], sources[fold] = recipe, path
    selected = {name for space in SEARCH_SPACES.values() for name in space}
    selected |= {"warmup_steps", "num_epochs"}
    first = folds[0]
    reference = recipes[first].to_dict()
    for fold, recipe in recipes.items():
        changed = sorted(name for name, value in recipe.to_dict().items()
                         if name not in selected and value != reference[name])
        if changed:
            raise ValueError(f"fold {fold}'s hyperparameters ({sources[fold]}) train another model "
                             f"than fold {first}'s ({sources[first]}): {', '.join(changed)} differ")
    return recipes


def default_hyperparameters(model_key: str) -> Hyperparameters:
    """A reasonable starting point per variant, used when no tuning result is given.

    These are the default configurations. They are a sensible place to start a
    search from, not tuned values: they were chosen for an earlier residual scaling
    and earlier solver tolerances. Prefer the output of
    ``scripts/tune_hyperparameters.py``.
    """
    defaults = {
        "r2": dict(width_size=32, depth=4, dropout_rate=0.3, learning_rate=1e-3, warmup_steps=400),
        "r3": dict(width_size=16, depth=5, dropout_rate=0.2, learning_rate=3e-3, warmup_steps=300),
        "r4": dict(width_size=12, depth=7, dropout_rate=0.2, learning_rate=5e-3, warmup_steps=300),
    }[model_key]
    return Hyperparameters(**defaults)


@dataclass
class FoldResult:
    """What a trained fold reports (``fold_result.json``, ``cv_results.csv``)."""

    fold: int
    train_loss: float
    val_loss: float
    train_labels: list[str]
    val_labels: list[str]
    checkpoint_dir: str
    n_parameters: int
    hyperparameters: dict[str, Any]


def loss_weights(spec: ModelSpec, hyperparameters: Hyperparameters) -> tuple[dict[str, float], list[str]]:
    """The variant's loss weights with the recipe's overrides, and the outputs they fit.

    The fitted outputs keep the variant's output order. An override may only name
    an output the variant produces, so a typo or a pH weight on R4 fails here
    rather than training a model with the registry's weights.
    """
    overrides = {name: float(value) for name, value in hyperparameters.objective_weights.items()}
    unknown = sorted(set(overrides) - set(spec.outputs))
    if unknown:
        raise KeyError(f"{spec.display_name} has no output {unknown}; it produces "
                       f"{', '.join(spec.outputs)}")
    negative = sorted(name for name, value in overrides.items() if value < 0)
    if negative:
        raise ValueError(f"negative loss weights for {negative}")
    weights = {**spec.objective_weights, **overrides}
    fitted = [name for name in spec.outputs if weights.get(name, 0.0) > 0.0]
    if not fitted:
        raise ValueError(f"the loss weights leave no output of {spec.display_name} to fit")
    return weights, fitted


def check_experiments(names: Sequence[str], model_keys: Sequence[str],
                      directory: Path | str | None = None, split: DataSplit | None = None,
                      benchmarks: Path | str | None = None) -> None:
    """Fail unless every experiment in ``configs/experiments/`` applies to every variant.

    The cluster job scripts call this before anything is queued: a run that only
    finds out on a compute node, e.g. a pH weight on R4, or per-fold calibrations
    that were never copied there, has waited for nothing. ``split``
    defaults to ``configs/split.json``, ``benchmarks`` to ``config.benchmarks_dir()``.
    """
    from adm1_package.utils.corrections import layout_for

    from .config import PROJECT_ROOT, benchmarks_dir
    from .registry import get_spec
    from .splits import default_split_path, load_split

    directory = Path(directory) if directory is not None else PROJECT_ROOT / "configs" / "experiments"
    problems = []
    for name in names:
        try:
            experiment = load_experiment(directory / f"{name}.json")
            recipe = Hyperparameters.from_dict(experiment)
        except (KeyError, ValueError) as error:
            problems.append(f"{name}: {error}")
            continue
        for key in model_keys:
            recipe_of_key = apply_experiment(default_hyperparameters(key), experiment)
            try:
                loss_weights(get_spec(key), recipe_of_key)
            except (KeyError, ValueError) as error:
                problems.append(f"{name} on {key}: {error}")
            # The network's outputs: a parameter the variant does not have, e.g. R2's
            # propionate uptake asked of R3, otherwise fails only once the run has started.
            try:
                layout_for(key, recipe_of_key.correction)
            except (KeyError, ValueError) as error:
                problems.append(f"{name} on {key}: {error}")
        folds_split = split if split is not None else load_split(default_split_path())
        where = Path(benchmarks) if benchmarks is not None else benchmarks_dir()
        for fold in range(1, folds_split.k_folds + 1):
            try:
                fold_data_dir(recipe, where, folds_split, fold, where)
            except (FileNotFoundError, ValueError) as error:
                problems.append(f"{name}: {error}")
                break
    if problems:
        raise ValueError("; ".join(problems))


def fold_data_dir(hyperparameters: Hyperparameters, data_dir: Path | str, split: DataSplit,
                  fold: int, benchmarks: Path | str | None = None) -> Path:
    """Where fold ``fold`` reads its reactors' parameters and start values: the fold's
    own calibration under ``benchmarks`` (default ``config.benchmarks_dir()``).

    Refused unless it was made for this split, is finished, and was made with the
    hybrid's Arrhenius physics, within its ``calibration_bounds`` and fitted to its
    ``forcing``. ``data_dir``, the installed data, is not read: the anchors are the fold's.
    """
    from adm1_package.utils.corrections import CorrectionOptions

    from .calibrations import benchmark_model, fold_benchmark_data
    from .config import benchmarks_dir

    where = Path(benchmarks) if benchmarks is not None else benchmarks_dir()
    data = fold_benchmark_data(where, split, fold)
    # Rates calibrated at 35 C with Arrhenius mean something else than rates fitted at the
    # reactors' temperature without it: the anchors need the physics they were fitted with.
    model = benchmark_model(where)
    calibrated_with = model["arrhenius"]
    anchored_with = "arrhenius" in CorrectionOptions.from_dict(hyperparameters.correction).physics
    if calibrated_with != anchored_with:
        raise ValueError(
            f"the per-fold calibrations under {where} were made {'with' if calibrated_with else 'without'} "
            f"Arrhenius, but the correction {'has' if anchored_with else 'lacks'} physics 'arrhenius'; "
            "a hybrid built on them needs the same (point --benchmarks or ADM1_BENCHMARKS_DIR "
            "at calibrations of its model, e.g. runs/benchmarks-smoothstep-arrhenius)")
    # The fast-hydrolysis anchors share their model with the Arrhenius calibration; only
    # the bounds they were fitted within tell the two apart.
    wanted = {name: [float(low), float(high)]
              for name, (low, high) in sorted(hyperparameters.calibration_bounds.items())}
    if model["bounds"] != wanted:
        raise ValueError(
            f"the per-fold calibrations under {where} were fitted within "
            f"{model['bounds'] or 'the stored bounds'}, but the experiment is built on "
            f"calibrations fitted within {wanted or 'the stored bounds'} (calibrate_folds.py "
            "--bound); point --benchmarks or ADM1_BENCHMARKS_DIR at those, e.g. "
            "runs/benchmarks-smoothstep-fast")
    # Parameters fitted to one forcing are not the anchors of a model given another.
    if model["forcing"] != hyperparameters.forcing:
        raise ValueError(
            f"the per-fold calibrations under {where} were fitted to the {model['forcing']} "
            f"forcing, but the experiment gives the model the {hyperparameters.forcing}; "
            "calibrate the folds again (calibrate_folds.py) into another --output")
    return data


def validation_interval(hyperparameters: Hyperparameters) -> int:
    """Epochs between two validations: ten per run.

    Never longer than the run: a run that never validates selects no checkpoint
    and cannot be restored.
    """
    return max(1, hyperparameters.num_epochs // 10)


def _build_fold_datasets(spec: ModelSpec, data_dir: Path, split: DataSplit, fold: int,
                         hyperparameters: Hyperparameters, seed: int):
    """Dataset, augmented training set and augmented validation set for one fold.

    The scaler is refitted on the training indices *before* the subsets are
    taken, so every sample the model sees is normalized with training-only
    statistics.
    """
    from adm1_package.utils.augmentation import add_noise

    dataset = spec.dataset(data_dir, forcing=hyperparameters.forcing)
    train_indices, val_indices = split.fold_indices(fold, dataset.labels)

    dataset.refit_scaling(train_indices)

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    training_set = add_noise(
        train_subset,
        size=hyperparameters.noise_copies,
        noise_input=hyperparameters.noise_input,
        noise_output=hyperparameters.noise_output,
        seed=seed,
    )
    validation_set = add_noise(
        val_subset,
        size=hyperparameters.noise_copies,
        noise_input=0.0,
        noise_output=hyperparameters.noise_output,
        seed=seed,
    )
    return dataset, training_set, validation_set


def _build_trainer(spec: ModelSpec, dataset, hyperparameters: Hyperparameters,
                   weights: dict[str, float], fitted: list[str], paths: RunPaths, fold: int,
                   seed: int, debug: bool, enable_progress_bar: bool, verbose: bool):
    """The fold's trainer: its network, its UDE and the record of both.

    Everything a fold's ``hparams.json`` must say to be rebuilt and scored later -- the
    correction, initial state, spin-up, forcing and the calibration with its fingerprint
    -- is passed in here (``evaluation.fold_settings`` reads it back).
    """
    import jax.nn as jnn

    from adm1_package.utils.corrections import CorrectionOptions

    from .calibrations import calibration_fingerprint

    correction = CorrectionOptions.from_dict(hyperparameters.correction)
    layout = spec.correction_layout(correction)
    forcing_width = spec.nn_in_size(dataset)
    n_factors = layout.n_factor_outputs if hyperparameters.split_network else 0

    return spec.trainer_class()(
        in_size=spec.nn_in_size(dataset),
        width_size=hyperparameters.width_size,
        out_size=layout.size,
        depth=hyperparameters.depth,
        activation=jnn.selu,
        final_activation=jnn.identity,
        # The output bias is the learned global base; zeroing the output layer
        # starts the model at that base.
        use_final_bias=True,
        dropout_rate=hyperparameters.dropout_rate,
        zero_output=True,
        factor_inputs=forcing_width if n_factors else 0,
        factor_outputs=n_factors,
        batch_size=hyperparameters.batch_size,
        ude=spec.ude(correction, initial_state=hyperparameters.initial_state),
        # Recorded in the fold's hparams.json, so a checkpoint says what model
        # it is. Evaluation rebuilds the fold's UDE and scores from them.
        correction=correction.to_dict(),
        initial_state=hyperparameters.initial_state,
        spin_up_days=hyperparameters.spin_up_days,
        forcing=hyperparameters.forcing,
        # Which calibration the reactors' parameters and starts came from, and a
        # hash of it, so evaluation scores the fold on the same one.
        calibration=hyperparameters.calibration,
        calibration_fingerprint=calibration_fingerprint(dataset),
        correction_penalty=hyperparameters.correction_penalty,
        keep_checkpoints=hyperparameters.keep_checkpoints,
        optimizer_hparams={
            "optimizer": "adam",
            "lr": hyperparameters.learning_rate,
            "lr_scheduler": "warmup_cosine_decay",
            "end_value": 0.1 * hyperparameters.learning_rate,
            "warmup": hyperparameters.warmup_steps,
            "gradient_clip": hyperparameters.gradient_clip,
            "length_schedule": [list(pair) for pair in hyperparameters.length_schedule],
        },
        regularization_lambda=hyperparameters.regularization_lambda,
        logger_params={"base_log_dir": str(paths.base), "log_dir": f"fold_{fold}"},
        check_val_every_n_epoch=validation_interval(hyperparameters),
        objective_weights=weights,
        objective_list=fitted,
        debug=debug,
        seed=seed,
        enable_progress_bar=enable_progress_bar,
        verbose=verbose,
    )


def train_fold(
    spec: ModelSpec,
    data_dir: Path,
    split: DataSplit,
    fold: int,
    hyperparameters: Hyperparameters,
    paths: RunPaths,
    seed: int = 42,
    verbose: bool = False,
    debug: bool = False,
    enable_progress_bar: bool = False,
    benchmarks: Path | None = None,
) -> FoldResult:
    """Train a single fold and return its losses and checkpoint location.

    ``benchmarks`` holds the per-fold calibrations a ``calibration="per-fold"``
    recipe is built on (default ``config.benchmarks_dir()``); see :func:`fold_data_dir`.
    The fold's checkpoints and ``fold_result.json`` go to ``paths.fold(fold)``.
    """
    import jax

    from adm1_package.utils.datasets import create_data_loaders, custom_collate_fn

    # Before the datasets are built, so a bad override fails at once.
    weights, fitted = loss_weights(spec, hyperparameters)
    source = fold_data_dir(hyperparameters, data_dir, split, fold, benchmarks)

    # Augmentation depends on the fold only, so trials stay comparable.
    augmentation_seed = seed + 1000 * fold

    dataset, training_set, validation_set = _build_fold_datasets(
        spec, source, split, fold, hyperparameters, augmentation_seed
    )
    train_loader, validation_loader = create_data_loaders(
        training_set,
        validation_set,
        train=[True, False],
        batch_size=hyperparameters.batch_size,
        num_workers=0,
        seed=augmentation_seed,
        collate_fn=custom_collate_fn,
    )
    trainer = _build_trainer(spec, dataset, hyperparameters, weights, fitted, paths, fold,
                             seed, debug, enable_progress_bar, verbose)
    metrics = trainer.train_model(
        train_loader, validation_loader, num_epochs=hyperparameters.num_epochs
    )

    fold_dir = paths.fold(fold)
    result = FoldResult(
        fold=fold,
        train_loss=float(metrics.get("train/loss", np.nan)),
        val_loss=float(metrics.get("val/loss", np.nan)),
        train_labels=split.training_labels(fold),
        val_labels=split.validation_labels(fold),
        checkpoint_dir=str(fold_dir),
        n_parameters=int(
            sum(
                leaf.size
                for leaf in jax.tree_util.tree_leaves(trainer.state.model)
                if hasattr(leaf, "size")
            )
        ),
        hyperparameters=hyperparameters.to_dict(),
    )

    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "fold_result.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )

    del trainer, train_loader, validation_loader
    jax.clear_caches()
    return result
