"""Ensembles: one experiment trained with several seeds, scored on its mean prediction.

A run's score varies with its training seed by more than arms of an experiment differ, so
an experiment is reported for the mean prediction of a few seeds. An ensemble directory
lists its member runs in :data:`ENSEMBLE_FILE` and stands in for a run wherever one is
scored or drawn: evaluation, figures and inspection open its members
(:func:`member_runs`) and average their hybrid predictions.

    <runs>/<model>/<experiment>_ensemble_s45-46-47/
        ensemble.json   {"model_key", "members" (relative to the parent), "seeds"}
        split.json      the members' split
        results/        as in a run directory
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from .retired import retired
from .splits import load_run_split

__all__ = [
    "ENSEMBLE_FILE",
    "create_ensemble",
    "default_ensemble_name",
    "ensemble_members",
    "ensemble_source",
    "is_ensemble",
    "member_runs",
]

#: The manifest of an ensemble directory (:func:`create_ensemble`).
ENSEMBLE_FILE = "ensemble.json"


# ---------------------------------------------------------------------------
# Reading an ensemble
# ---------------------------------------------------------------------------
def is_ensemble(directory: Path) -> bool:
    """Whether ``directory`` is an ensemble directory rather than a run."""
    return (Path(directory) / ENSEMBLE_FILE).exists()


def _manifest(directory: Path) -> dict[str, Any]:
    return json.loads((Path(directory) / ENSEMBLE_FILE).read_text(encoding="utf-8"))


def ensemble_source(directory: Path, model_key: str | None = None) -> str:
    """The variant an ensemble's member runs were trained with, which must be
    ``model_key`` where given. A transfer of runs of another variant (``source_model_key``
    in the manifest) was retired."""
    manifest = _manifest(directory)
    source = manifest.get("source_model_key", manifest["model_key"])
    if source != manifest["model_key"]:
        raise retired(f"a transfer of {source} networks into {manifest['model_key']}",
                      str(directory))
    if model_key is not None and source != model_key:
        raise ValueError(f"{directory} is an ensemble of {source}, not {model_key}")
    return source


def ensemble_members(directory: Path) -> list[Path]:
    """The member runs of an ensemble directory, resolved."""
    base = Path(directory).resolve().parent
    return [(base / member).resolve() for member in _manifest(directory)["members"]]


def member_runs(directory: Path, model_key: str) -> list[Path]:
    """The runs whose folds stand for ``directory``: its members if it is an ensemble of
    ``model_key`` runs, else the run itself."""
    directory = Path(directory)
    if not is_ensemble(directory):
        return [directory]
    ensemble_source(directory, model_key)
    return ensemble_members(directory)


# ---------------------------------------------------------------------------
# Making one
# ---------------------------------------------------------------------------
def _run_record(run: Path, name: str) -> dict[str, Any]:
    """A JSON record ``train_cv.py`` wrote into a run directory."""
    path = Path(run) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: {run} is no run directory of train_cv.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_name(run: Path) -> str | None:
    """The stem of the experiment file a run was trained with, from its metadata."""
    experiment = _run_record(run, "metadata.json").get("experiment")
    return Path(str(experiment).replace("\\", "/")).stem if experiment else None


def default_ensemble_name(members: Sequence[Path]) -> str:
    """``<experiment>_ensemble_s<seed>-<seed>...``, from the members' metadata;
    ``<experiment>_tuned_ensemble_...`` for runs trained with hyperparameters per fold
    (``train_cv.py --fold-hyperparameters``), so they do not take the name of the same
    experiment trained with one recipe."""
    experiments = {_experiment_name(member) for member in members}
    if len(experiments) != 1 or None in experiments:
        raise ValueError("the runs were not all trained from one experiment file; name the "
                         "ensemble with --name")
    metadata = [_run_record(member, "metadata.json") for member in members]
    tuned = "_tuned" if all(record.get("fold_hyperparameters") for record in metadata) else ""
    seeds = [record.get("seed") for record in metadata]
    return f"{experiments.pop()}{tuned}_ensemble_s{'-'.join(str(seed) for seed in seeds)}"


def _run_recipes(run: Path) -> dict[str, dict[str, Any]]:
    """The recipes a run was trained with, as this code reads them: ``{"all": recipe}`` for
    one recipe, ``{"<fold>": recipe, ...}`` for one per fold (``fold_hyperparameters.json``).
    A field added since a run was trained comes with its default, which the run trained with."""
    from .training import Hyperparameters

    if (Path(run) / "fold_hyperparameters.json").exists():
        per_fold = _run_record(run, "fold_hyperparameters.json")
        return {fold: Hyperparameters.from_dict(recipe).to_dict() for fold, recipe in per_fold.items()}
    return {"all": Hyperparameters.from_dict(_run_record(run, "hyperparameters.json")).to_dict()}


def _describe_recipes(recipes: dict[str, Any]) -> str:
    return ("one recipe for every fold" if "all" in recipes
            else f"a recipe per fold for folds {', '.join(sorted(recipes))}")


def check_members(model_key: str, members: Sequence[Path]) -> list:
    """Refuse runs that are not one experiment (one variant, split and recipe) trained with
    different seeds; return their seeds."""
    members = [Path(member) for member in members]
    if len(members) < 2 or len({member.resolve() for member in members}) != len(members):
        raise ValueError("an ensemble needs two or more different runs")
    seeds: list = []
    split = recipe = None
    for member in members:
        metadata = _run_record(member, "metadata.json")
        if metadata.get("model_key") != model_key:
            raise ValueError(f"{member} is a run of {metadata.get('model_key')}, not {model_key}")
        seed = metadata.get("seed")
        if seed in seeds:
            raise ValueError(f"{member} was trained with seed {seed}, as another member was: "
                             "the same network twice")
        seeds.append(seed)
        member_split = load_run_split(member)
        if split is None:
            split = member_split
        elif not member_split.same_assignment(split):
            raise ValueError(f"{member} was trained on another split than {members[0]}")
        member_recipes = _run_recipes(member)
        if recipe is None:
            recipe = member_recipes
            continue
        if set(member_recipes) != set(recipe):
            raise ValueError(f"{member} was trained with {_describe_recipes(member_recipes)}, "
                             f"{members[0]} with {_describe_recipes(recipe)}")
        for fold, expected in recipe.items():
            if member_recipes[fold] != expected:
                changed = ", ".join(sorted(name for name in expected
                                           if member_recipes[fold].get(name) != expected[name]))
                where = "" if fold == "all" else f" in fold {fold}"
                raise ValueError(f"{member} was trained with another recipe than {members[0]}{where}: "
                                 f"{changed}")
    return seeds


def create_ensemble(directory: Path, model_key: str, members: Sequence[Path]) -> Path:
    """Make ``directory`` the ensemble of ``members``: runs of one experiment, one recipe and
    split, trained with different seeds.

    It holds the members' split and :data:`ENSEMBLE_FILE`, which lists them relative to the
    directory's parent where they are its siblings (absolute on another drive), so runs and
    ensemble move together. Results and figures go into it as into a run directory. Made
    again from the same members it is reused; from others it is refused.
    """
    directory = Path(directory)
    members = [Path(member) for member in members]
    seeds = check_members(model_key, members)
    if is_ensemble(directory):
        if ensemble_members(directory) != [member.resolve() for member in members]:
            raise ValueError(f"{directory} is the ensemble of other members; choose another --name")
        return directory
    listed = []
    for member in members:
        try:
            listed.append(os.path.relpath(member.resolve(), directory.resolve().parent))
        except ValueError:  # on another drive
            listed.append(str(member.resolve()))
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(members[0] / "split.json", directory / "split.json")
    manifest = {"model_key": model_key, "members": listed, "seeds": seeds}
    (directory / ENSEMBLE_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory
