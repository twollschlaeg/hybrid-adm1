"""Hyperparameter search for the UDE variants.

Protocol
--------
Each trial trains one model per cross-validation fold and returns the mean
fold-validation loss, so selection uses validation data. Any hold-out test
reactors in the split file stay untouched by both training and selection --
though the committed split has none, in which case the same folds inform
selection and reporting and the reported errors are mildly optimistic.

Three properties make the search comparable across trials:

* the reported validation loss excludes the L2 penalty. Including it would let
  a trial lower the objective by lowering ``regularization_lambda`` alone,
  since the penalty falls with lambda whatever the network does;

* the augmentation seed depends on the fold only, so trials differ in their
  hyperparameters rather than in which noise draw they happened to get;
* folds are reported to Optuna in order, so the median pruner compares like with
  like. Pruning only saves compute when folds run sequentially
  (``--fold-workers 1``); with parallel folds every fold is already paid for by
  the time the trial can be pruned.

Running on a cluster
--------------------
Point ``--storage`` at a shared database and start several array tasks against
the same ``--study-name``; Optuna coordinates them. With a file-backed SQLite
store, keep ``--parallel-trials 1`` per task.
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import RunPaths
from .registry import ModelSpec, get_spec
from .retired import drop_retired
from .splits import DataSplit
from .training import Hyperparameters, apply_experiment, default_hyperparameters, train_fold

__all__ = ["suggest_hyperparameters", "default_search_values", "run_study", "tuning_run_name",
           "SEARCH_SPACES"]


#: Per-variant search ranges. The widths and depths bracket the default
#: configurations (:func:`.training.default_hyperparameters`). The network has to
#: stay small because the training set is 9-12 reactor time series.
#:
#: ``regularization_lambda`` spans 1e-5 to 1e-1, because below about 1e-3 the
#: penalty does nothing: its gradient is ``2*lambda*||W||``, and measured on a
#: trained R4 fold (``||W|| = 4.19``, data-gradient norm 0.349) that is 0.002% of
#: the data gradient at 1e-6, 0.24% at 1e-4 and 2.4% at 1e-3. Only from about
#: 1e-2 does it shape the fit. A range that stopped at 1e-3 could therefore only
#: ever report that regularization did not matter.
#:
#: The wide range is safe because the penalty is not part of the reported
#: validation loss (see Protocol). Otherwise a larger lambda would inflate the
#: very number the search minimizes, and the search would reward small lambdas
#: whatever they did to the fit. With the metric clean, an over-large lambda is
#: rejected on the evidence -- a worse data loss -- which is what makes the wide
#: range useful.
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "r2": {
        "width_size": ("int", 16, 64, 16),
        "depth": ("int", 3, 6, 1),
        "dropout_rate": ("float", 0.0, 0.4, 0.1),
        "learning_rate": ("log", 3e-4, 1e-2),
        "regularization_lambda": ("log", 1e-5, 1e-1),
        "batch_size": ("categorical", [2, 4, 8]),
    },
    "r3": {
        "width_size": ("int", 8, 48, 8),
        "depth": ("int", 3, 7, 1),
        "dropout_rate": ("float", 0.0, 0.4, 0.1),
        "learning_rate": ("log", 3e-4, 1e-2),
        "regularization_lambda": ("log", 1e-5, 1e-1),
        "batch_size": ("categorical", [2, 4, 8]),
    },
    "r4": {
        # Steps of 4, so the default width of 12 is a point of the search.
        "width_size": ("int", 8, 40, 4),
        "depth": ("int", 3, 8, 1),
        "dropout_rate": ("float", 0.0, 0.4, 0.1),
        "learning_rate": ("log", 3e-4, 1e-2),
        "regularization_lambda": ("log", 1e-5, 1e-1),
        "batch_size": ("categorical", [2, 4, 8]),
    },
}


def suggest_hyperparameters(trial, model_key: str, num_epochs: int,
                            base: Hyperparameters | None = None) -> Hyperparameters:
    """Draw one configuration from the variant's search space, over ``base``.

    Everything the search space does not cover comes from ``base`` (default:
    :class:`Hyperparameters`), and is so held fixed across the search: the
    correction, the network split and the anchors change what the model *is*, and
    a study that mixed them would compare different models by one loss.
    """
    space = SEARCH_SPACES[model_key]
    drawn: dict[str, Any] = {}
    for name, definition in space.items():
        kind = definition[0]
        if kind == "int":
            _, low, high, step = definition
            drawn[name] = trial.suggest_int(name, low, high, step=step)
        elif kind == "float":
            _, low, high, step = definition
            drawn[name] = trial.suggest_float(name, low, high, step=step)
        elif kind == "log":
            _, low, high = definition
            drawn[name] = trial.suggest_float(name, low, high, log=True)
        elif kind == "categorical":
            drawn[name] = trial.suggest_categorical(name, definition[1])
        else:
            raise ValueError(f"unsupported search-space entry {definition!r} for {name!r}")

    # The warmup is expressed in optimizer steps, so it has to follow the number
    # of steps an epoch actually takes rather than being tuned independently.
    drawn["num_epochs"] = num_epochs
    drawn["warmup_steps"] = max(50, int(3 * num_epochs))
    base = Hyperparameters() if base is None else base
    return Hyperparameters.from_dict({**base.to_dict(), **drawn})


def _contains(definition: tuple, value: Any) -> bool:
    """Whether ``value`` is a point of the search-space entry ``definition``."""
    kind = definition[0]
    if kind == "categorical":
        return value in definition[1]
    low, high = definition[1], definition[2]
    if not low <= value <= high:
        return False
    if kind in ("int", "float"):
        steps = (value - low) / definition[3]
        return abs(steps - round(steps)) < 1e-9
    return True


def default_search_values(model_key: str,
                          experiment: dict[str, Any] | None = None) -> dict[str, Any]:
    """The configuration the variant trains with when nothing is tuned, as values of its
    search space: its default hyperparameters (:func:`.training.default_hyperparameters`)
    with ``experiment`` put over them, as ``train_cv.py`` puts it. A search started from them
    selects them unless a trial does better.

    Refused if a value lies outside the search space, where the trial started from it would
    fail. With the correction penalty a dimension, for example, the experiment's value has
    to stand in for the class default of 0.0, which lies outside the logarithmic range.
    """
    defaults = default_hyperparameters(model_key)
    if experiment is not None:
        defaults = apply_experiment(defaults, experiment)
    space = SEARCH_SPACES[model_key]
    values = {name: getattr(defaults, name) for name in space}
    outside = {name: value for name, value in values.items() if not _contains(space[name], value)}
    if outside:
        raise ValueError(f"the default configuration of {model_key} lies outside its search "
                         f"space at {outside}; a trial started from it would fail")
    return values


def _fold_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entry point: build JAX objects inside the worker."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax

    jax.config.update("jax_enable_x64", True)

    spec = get_spec(job["model_key"])
    split = DataSplit.from_dict(job["split"])
    paths = RunPaths(model=job["model_key"], name=job["run_name"], root=Path(job["run_root"]))
    result = train_fold(
        spec=spec,
        data_dir=Path(job["data_dir"]),
        split=split,
        fold=job["fold"],
        hyperparameters=Hyperparameters.from_dict(job["hyperparameters"]),
        paths=paths,
        seed=job["seed"],
        verbose=job["verbose"],
        # .get: train_cv.py puts "debug" in its jobs, but the jobs run_study
        # builds do not.
        debug=job.get("debug", False),
        # Only train_cv.py's jobs name it; run_study's leave the default.
        benchmarks=Path(job.get("benchmarks")) if job.get("benchmarks") else None,
    )
    return asdict(result)


def tuning_run_name(explicit: str | None = None) -> str:
    """Directory name for one search, unique per job submission.

    In a shared ``<model>/tuning/`` directory every run would overwrite the previous
    one's ``best_hyperparameters.json`` and ``trials.csv``, and its trial directories
    would interleave with theirs.

    The identifier must be the same for **every process of one job submission** and
    different between job submissions, because an array job's tasks share one Optuna
    study through a file in this directory. Hence the order:

    ``explicit``
        whatever ``--tuning-name`` said. Pass the name of an existing search to
        add trials to it -- that is how a search resumes after a task is killed
        at the wall clock.
    ``$SLURM_ARRAY_JOB_ID``
        identical across the tasks of one array, which is exactly the grouping
        wanted. Checked before ``SLURM_JOB_ID``, which differs per array task
        and would scatter the tasks into one directory each, splitting the
        shared study into several private ones.
    ``$SLURM_JOB_ID``
        for a job submitted without an array.
    timestamp
        off a cluster. Resolution is seconds, so two searches started in the
        same second would collide.
    """
    import os
    from datetime import datetime

    if explicit:
        # removeprefix, not lstrip: lstrip strips any of those characters, so a
        # name like "nice" would come back as "ce".
        name = explicit.strip().removeprefix("tuning_")
        if not name:
            raise ValueError(f"tuning name {explicit!r} is empty once normalized")
        return f"tuning_{name}"

    for variable in ("SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"):
        value = os.environ.get(variable, "").strip()
        if value:
            return f"tuning_{value}"

    return f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_storage(storage: str | None, tuning_root: Path, study_name: str):
    """Turn a storage argument into something ``optuna.create_study`` accepts.

    ``None``
        a SQLite file next to the trials. Fine for one process on a local disk.
    ``"journal"``, or any path ending in ``.log``
        Optuna's journal storage. **This is the one to use on a cluster.**
        SQLite reaches across a network filesystem with POSIX locks that NFS and
        Lustre implement loosely or not at all, so concurrent array tasks get
        "database is locked" errors or, worse, a silently corrupted study. The
        journal backend appends and was written for exactly this case.
    anything else
        passed through unchanged, so ``postgresql://`` and ``mysql://`` URLs
        still work where a database server is available.
    """
    if storage is None:
        return f"sqlite:///{(tuning_root / f'{study_name}.db').as_posix()}"

    if storage == "journal" or storage.endswith(".log"):
        import optuna

        path = tuning_root / f"{study_name}.log" if storage == "journal" else Path(storage)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

            # The default lock creates a symlink, which needs a privilege
            # Windows does not grant by default and which NFSv2 was the reason
            # for. The open-based lock works on NFSv3+, on Lustre and on
            # Windows, so the same command line runs on the workstation and on
            # the cluster.
            backend = JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path)))
        except ImportError:  # optuna < 4.0
            from optuna.storages import JournalFileStorage, JournalFileOpenLock

            backend = JournalFileStorage(str(path), lock_obj=JournalFileOpenLock(str(path)))
        return optuna.storages.JournalStorage(backend)

    return storage


def run_study(
    model_key: str,
    split: DataSplit,
    data_dir: Path,
    run_root: Path,
    study_name: str | None = None,
    storage: str | None = None,
    n_trials: int = 30,
    num_epochs: int = 40,
    seed: int = 42,
    worker_id: int = 0,
    tuning_name: str | None = None,
    base: Hyperparameters | None = None,
    fold_workers: int = 1,
    parallel_trials: int = 1,
    verbose: bool = False,
    benchmarks: Path | None = None,
    enqueue: dict[str, Any] | None = None,
):
    """Run the search and return the Optuna study.

    ``seed`` fixes the *training* seed, identically for every trial and every
    worker, so trials differ by hyperparameters alone.

    ``worker_id`` offsets the *sampler* seed only, and must differ between
    processes that share one study. TPE draws its first ``n_startup_trials``
    at random from a seeded generator, so several array tasks created with the
    same sampler seed propose the same opening configurations and the first
    wave of an array job is spent computing the same thing several times over.
    Pass ``$SLURM_ARRAY_TASK_ID``.

    ``tuning_name`` names the directory this search writes into, under
    ``<run_root>/<model>/``. It defaults to something unique per job submission --
    see :func:`tuning_run_name` -- so a new search does not overwrite the
    previous one's results. Pass an existing name to add trials to that search
    instead of starting a fresh one.

    ``base`` holds everything the search does not vary fixed for every trial;
    see :func:`suggest_hyperparameters`.

    ``benchmarks`` holds the per-fold calibrations a per-fold ``base`` is built on
    (default ``config.benchmarks_dir()``); a nested search passes its outer fold's
    (:mod:`.nested`).

    ``enqueue`` is a configuration of the search space to try first, e.g. the default
    values (:func:`default_search_values`), so that the search selects it unless a
    trial does better. Only worker 0 enqueues it, and not into a study that has it.

    ``n_trials=0`` trains nothing and writes the selection from the trials the study
    holds: how a search is collected once its array tasks have ended, killed or not.
    """
    import optuna

    spec = get_spec(model_key)
    study_name = study_name or f"adm1_{model_key}_cv"
    run_name = tuning_run_name(tuning_name)
    tuning_root = Path(run_root) / model_key / run_name
    tuning_root.mkdir(parents=True, exist_ok=True)
    print(f"[{spec.display_name}] search directory: {tuning_root}")

    storage = _resolve_storage(storage, tuning_root, study_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed + worker_id),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )
    _check_signature(study, {
        "model_key": model_key,
        "num_epochs": num_epochs,
        "seed": seed,
        "split": split.to_dict(),
        "base": (base if base is not None else Hyperparameters()).to_dict(),
    }, f"study {study_name} in {tuning_root}")
    if study.trials:
        # Resumed: a sampler seeded as at the start would propose its opening trials again.
        study.sampler = optuna.samplers.TPESampler(seed=seed + worker_id + 7919 * len(study.trials))
    # One task enqueues, or an array would try the same configuration once per task. The
    # study remembers it, for the selection written by a task that was not told.
    if enqueue is not None and worker_id == 0:
        study.enqueue_trial(dict(enqueue), skip_if_exists=True)
        study.set_user_attr("started_from", dict(enqueue))

    fold_numbers = list(range(1, split.k_folds + 1))

    def make_job(trial_number: int, fold: int, hyperparameters: Hyperparameters) -> dict[str, Any]:
        return {
            "model_key": model_key,
            "split": split.to_dict(),
            "data_dir": str(data_dir),
            "run_root": str(run_root),
            # Same directory as the study itself, so trial checkpoints and trial
            # metadata land in one place.
            "run_name": f"{run_name}/trial_{trial_number:04d}",
            "fold": fold,
            "hyperparameters": hyperparameters.to_dict(),
            # Fixed per fold, so trials are compared on hyperparameters alone.
            "seed": seed,
            "verbose": verbose,
            # Passed on rather than left to $ADM1_BENCHMARKS_DIR, so the folds anchor where
            # the search was told to whatever the workers' environment says.
            "benchmarks": None if benchmarks is None else str(benchmarks),
        }

    def objective(trial, executor: ProcessPoolExecutor | None) -> float:
        hyperparameters = suggest_hyperparameters(trial, model_key, num_epochs, base)
        jobs = [make_job(trial.number, fold, hyperparameters) for fold in fold_numbers]
        results = _train_trial(trial, jobs, executor)
        return _record_trial(trial, tuning_root / f"trial_{trial.number:04d}", hyperparameters,
                             results)

    _optimize(study, objective, n_trials, fold_workers, parallel_trials)
    _write_selection(study, spec, study_name, tuning_root, split, num_epochs, seed, base,
                     benchmarks)
    return study


def _check_signature(study, signature: dict[str, Any], where: str) -> None:
    """Record what the trials of ``study`` are ranked on, or refuse a study that recorded
    something else.

    A study resumed under its name, by a resubmitted job or by the collection after an array,
    has to train the same model the same way, or it would rank losses of different things
    against each other. Compared through JSON, as the storage keeps it, so tuples and lists
    compare alike. A study started before the retirement records the retired fields at the
    values that changed nothing; they are read as this code reads a recipe (:mod:`.retired`).
    """
    signature = json.loads(json.dumps(signature))
    recorded = study.user_attrs.get("signature")
    if recorded is None:
        study.set_user_attr("signature", signature)
        return
    recorded = {**recorded, "base": drop_retired(recorded.get("base", {}), where)}
    if recorded != signature:
        changed = [key for key in signature if key != "base" and recorded.get(key) != signature[key]]
        changed += [name for name, value in signature["base"].items()
                    if recorded.get("base", {}).get(name) != value]
        raise ValueError(f"{where} was started with other settings "
                         f"({', '.join(changed)} differ); name this search otherwise")


def _train_trial(trial, jobs: list[dict[str, Any]],
                 executor: ProcessPoolExecutor | None) -> list[dict[str, Any]]:
    """Every fold of one trial trained (:func:`_fold_worker`), its results in fold order.

    Sequential, the mean loss is reported after each fold, so the pruner can stop a bad
    configuration before the remaining folds are paid for; parallel, every fold runs.
    """
    import optuna

    results: list[dict[str, Any]] = []
    if executor is None:
        for step, job in enumerate(jobs):
            results.append(_fold_worker(job))
            trial.report(float(np.nanmean([r["val_loss"] for r in results])), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    else:
        futures = {executor.submit(_fold_worker, job): job["fold"] for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
        results.sort(key=lambda item: item["fold"])
    return results


def _record_trial(trial, directory: Path, hyperparameters: Hyperparameters,
                  results: list[dict[str, Any]]) -> float:
    """Write a trial's recipe and fold results into ``directory``, record its spread on the
    trial, and return its score: the mean fold-validation loss (a trial without a finite
    one is pruned)."""
    import optuna

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hyperparameters.json").write_text(
        json.dumps(hyperparameters.to_dict(), indent=2), encoding="utf-8"
    )
    (directory / "fold_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    validation_losses = np.array([r["val_loss"] for r in results], dtype=float)
    training_losses = np.array([r["train_loss"] for r in results], dtype=float)
    trial.set_user_attr("val_loss_std", float(np.nanstd(validation_losses)))
    trial.set_user_attr("train_loss_mean", float(np.nanmean(training_losses)))
    trial.set_user_attr("fold_val_losses", validation_losses.tolist())

    score = float(np.nanmean(validation_losses))
    if not np.isfinite(score):
        raise optuna.TrialPruned()
    return score


def _optimize(study, objective: Callable, n_trials: int, fold_workers: int,
              parallel_trials: int) -> None:
    """Run ``n_trials`` trials of ``objective(trial, executor)``: with a process pool of
    ``fold_workers`` for the folds when there is more than one, else sequentially."""
    if n_trials <= 0:
        return
    if fold_workers > 1:
        import multiprocessing

        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        with ProcessPoolExecutor(max_workers=fold_workers) as executor:
            study.optimize(
                lambda trial: objective(trial, executor),
                n_trials=n_trials,
                n_jobs=parallel_trials,
            )
    else:
        study.optimize(lambda trial: objective(trial, None), n_trials=n_trials, n_jobs=1)


def _write_selection(study, spec: ModelSpec, study_name: str, tuning_root: Path,
                     split: DataSplit, num_epochs: int, seed: int,
                     base: Hyperparameters | None, benchmarks: Path | None) -> None:
    """Write the best trial's recipe (``best_hyperparameters.json``, which ``train_cv.py
    --hyperparameters`` reads) and every trial (``trials.csv``) into ``tuning_root``."""
    import optuna

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"no trial of study {study_name} in {tuning_root} has completed, so "
                           "there is nothing to select")
    best = study.best_trial
    best_hyperparameters = suggest_hyperparameters(
        optuna.trial.FixedTrial(best.params), spec.key, num_epochs, base
    )
    payload = {
        "model": spec.display_name,
        "model_key": spec.key,
        "study_name": study_name,
        "best_trial": best.number,
        "mean_val_loss": best.value,
        "val_loss_std": best.user_attrs.get("val_loss_std"),
        "fold_val_losses": best.user_attrs.get("fold_val_losses"),
        "n_trials_completed": len(completed),
        "search_space": SEARCH_SPACES[spec.key],
        "split_seed": split.seed,
        # What the selection was made on: the folds of this split (a nested search's
        # inner split leaves its outer fold's held-out reactors in "test"), built on
        # these calibrations, trained this long from this seed, and started from this.
        "split": split.to_dict(),
        "benchmarks": None if benchmarks is None else str(benchmarks),
        "num_epochs": num_epochs,
        "seed": seed,
        "started_from": study.user_attrs.get("started_from"),
        "hyperparameters": best_hyperparameters.to_dict(),
    }
    best_path = tuning_root / "best_hyperparameters.json"
    best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    trials_frame = study.trials_dataframe()
    trials_frame.to_csv(tuning_root / "trials.csv", index=False)

    print(f"\nBest trial {best.number}: mean validation loss {best.value:.6f}")
    print(json.dumps(best.params, indent=2))
    print(f"Wrote {best_path}")
