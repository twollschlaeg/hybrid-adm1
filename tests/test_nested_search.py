"""The nested hyperparameter search.

A search that selects on the folds the paper reports scores the selected models
optimistically. Nested, each outer fold's hyperparameters are chosen by a search of its
own on inner folds of the outer fold's training reactors, so the outer held-out reactors
inform neither the selection nor the training, and the outer scores stay honest.

The inner folds are built on the outer fold's per-fold calibration: it was fitted on the
same reactors the inner search may use, so nothing of the outer held-out reactors enters.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from helpers import CAMPAIGNS, EXPERIMENTS, ROOT, canonical_split, write_benchmarks

from adm1_package.experiments import calibrations, ensembles, nested, training, tuning
from adm1_package.experiments.splits import DataSplit, campaign_of, load_split, nested_split
from adm1_package.experiments.training import Hyperparameters

SPLIT = canonical_split()
#: The experiment each variant's nested search runs (scripts/hpc/submit_nested.sh).
NESTED_EXPERIMENTS = {"r4": "fold_fast_narrow_tan.json", "r3": "fold_fast_narrow_tan_no_ph.json",
                      "r2": "fold_fast_narrow_tan_no_ph.json"}


def _experiment(model_key: str) -> dict:
    return training.load_experiment(EXPERIMENTS / NESTED_EXPERIMENTS[model_key])


def _default_r4() -> dict:
    """Where R4's nested search starts: the paper's network under its experiment."""
    return tuning.default_search_values("r4", _experiment("r4"))


# ---------------------------------------------------------------------------
# The inner split
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("outer", range(1, SPLIT.k_folds + 1))
def test_the_inner_folds_divide_the_outer_training_reactors(outer):
    inner = nested_split(SPLIT, outer)
    assert sorted(inner.test) == sorted(SPLIT.validation_labels(outer))
    assert sorted(inner.development) == sorted(SPLIT.training_labels(outer))
    assert inner.k_folds == 4 and all(len(fold) == 3 for fold in inner.folds)
    assert sorted(inner.all_labels) == sorted(SPLIT.all_labels)


@pytest.mark.parametrize("outer", range(1, SPLIT.k_folds + 1))
def test_every_inner_fold_mixes_the_campaigns_as_evenly_as_they_allow(outer):
    inner = nested_split(SPLIT, outer)
    for campaign in CAMPAIGNS:
        counts = [sum(campaign_of(label) == campaign for label in fold) for fold in inner.folds]
        assert max(counts) - min(counts) <= 1, (campaign, inner.folds)


def test_the_inner_split_is_reproducible_and_says_what_it_is():
    first = nested_split(SPLIT, 2)
    assert first.to_dict() == nested_split(SPLIT, 2).to_dict()
    assert first.to_dict() == DataSplit.from_dict(json.loads(json.dumps(first.to_dict()))).to_dict()
    assert "outer fold 2" in first.note


def test_reactors_the_outer_split_tests_on_stay_out_of_the_inner_folds():
    outer = DataSplit(test=["LaRA-temp_F1"], folds=[list(fold) for fold in SPLIT.folds],
                      seed=SPLIT.seed, all_labels=list(SPLIT.all_labels))
    outer.folds[3].remove("LaRA-temp_F1")
    inner = nested_split(outer, 1)
    assert "LaRA-temp_F1" in inner.test and "LaRA-temp_F1" not in inner.development


@pytest.mark.parametrize("outer_fold, k_folds", [(0, 4), (6, 4), (1, 1), (1, 13)])
def test_an_impossible_inner_split_is_refused(outer_fold, k_folds):
    with pytest.raises(ValueError):
        nested_split(SPLIT, outer_fold, k_folds)


# ---------------------------------------------------------------------------
# The inner folds' calibrations
# ---------------------------------------------------------------------------
def _fake_benchmarks(root: Path, split: DataSplit = SPLIT, unfinished: int | None = None) -> Path:
    """Fast-hydrolysis per-fold calibrations, each fold's data a marker naming the fold."""
    return write_benchmarks(root, split, arrhenius=True, bounds={"k_hyd_ch": [0.1, 2.88]},
                            unfinished=unfinished, data=True)


def test_each_inner_fold_anchors_on_its_outer_folds_calibration(tmp_path):
    benchmarks = _fake_benchmarks(tmp_path / "benchmarks")
    output = tmp_path / "nested"
    inner_splits = nested.prepare_nested(SPLIT, benchmarks, output)
    assert sorted(inner_splits) == list(range(1, SPLIT.k_folds + 1))
    for outer, inner in inner_splits.items():
        assert load_split(nested.inner_split_path(output, outer)).to_dict() == inner.to_dict()
        where = nested.inner_benchmarks(output, outer)
        assert calibrations.benchmark_model(where) == calibrations.benchmark_model(benchmarks)
        for fold in range(1, inner.k_folds + 1):
            data = calibrations.fold_benchmark_data(where, inner, fold)
            for campaign in CAMPAIGNS:
                assert (data / campaign / "marker.txt").read_text(encoding="utf-8") == f"outer {outer}"
                # the record says where the calibration came from
                record = json.loads((where / f"fold_{fold}" / f"{campaign}.json").read_text(encoding="utf-8"))
                assert Path(record["derived_from"]) == benchmarks / f"fold_{outer}"
            # earlier calibrations are not the anchors and are left behind
            assert not list(data.rglob("calibration-backup"))
    manifest = json.loads((output / nested.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["inner_folds"] == 4 and Path(manifest["benchmarks"]) == benchmarks


def test_preparing_again_keeps_the_inner_splits_and_another_recipe_is_refused(tmp_path):
    benchmarks = _fake_benchmarks(tmp_path / "benchmarks")
    output = tmp_path / "nested"
    first = nested.prepare_nested(SPLIT, benchmarks, output)
    again = nested.prepare_nested(SPLIT, benchmarks, output)
    assert {k: v.to_dict() for k, v in first.items()} == {k: v.to_dict() for k, v in again.items()}
    with pytest.raises(ValueError, match="inner"):
        nested.prepare_nested(SPLIT, benchmarks, output, k_folds=3)


def test_benchmarks_of_another_split_are_refused(tmp_path):
    other = DataSplit(test=[], folds=[list(fold) for fold in reversed(SPLIT.folds)], seed=1,
                      all_labels=list(SPLIT.all_labels))
    benchmarks = _fake_benchmarks(tmp_path / "benchmarks", split=other)
    with pytest.raises(ValueError, match="another split"):
        nested.prepare_nested(SPLIT, benchmarks, tmp_path / "nested")


def test_an_unfinished_outer_calibration_is_refused(tmp_path):
    benchmarks = _fake_benchmarks(tmp_path / "benchmarks", unfinished=3)
    with pytest.raises(FileNotFoundError, match="fold 3"):
        nested.prepare_nested(SPLIT, benchmarks, tmp_path / "nested")


def test_the_final_experiments_accept_the_inner_folds_of_the_fast_calibrations(tmp_path,
                                                                               fast_benchmarks):
    """Everything fold_data_dir checks: split, status, Arrhenius, bounds and forcing. On the
    paper's calibrations, prepared into ``tmp_path``."""
    inner_splits = nested.prepare_nested(SPLIT, fast_benchmarks, tmp_path / "nested")
    for outer, inner in inner_splits.items():
        where = nested.inner_benchmarks(tmp_path / "nested", outer)
        training.check_experiments(["fold_fast_narrow_tan"], ["r4"], split=inner, benchmarks=where)
        training.check_experiments(["fold_fast_narrow_tan_no_ph"], ["r2", "r3"], split=inner,
                                   benchmarks=where)
        source = fast_benchmarks / f"fold_{outer}" / "data" / "LaRA-temp" / "LaRA-temp-parameters.json"
        copied = where / "fold_1" / "data" / "LaRA-temp" / "LaRA-temp-parameters.json"
        assert copied.read_bytes() == source.read_bytes()


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------
def _within(definition, value) -> bool:
    kind = definition[0]
    if kind == "categorical":
        return value in definition[1]
    low, high = definition[1], definition[2]
    if not low <= value <= high:
        return False
    if kind in ("int", "float"):
        steps = (value - low) / definition[3]
        return math.isclose(steps, round(steps), abs_tol=1e-9)
    return True


@pytest.mark.parametrize("model_key", sorted(tuning.SEARCH_SPACES))
def test_the_default_values_lie_in_the_search_space(model_key):
    """So a search can start from them, and select them when nothing does better."""
    values = tuning.default_search_values(model_key, _experiment(model_key))
    assert set(values) == set(tuning.SEARCH_SPACES[model_key])
    for name, value in values.items():
        assert _within(tuning.SEARCH_SPACES[model_key][name], value), (name, value)


@pytest.fixture
def penalty_searched(monkeypatch):
    """R4's search space with the correction penalty as a dimension, whether SEARCH_SPACES
    has it or not: the experiment sets it, the default network hyperparameters do not."""
    space = {**tuning.SEARCH_SPACES["r4"], "correction_penalty": ("log", 1e-5, 1e-1)}
    monkeypatch.setitem(tuning.SEARCH_SPACES, "r4", space)
    return training.load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json")


def test_the_default_values_take_what_the_experiment_sets(penalty_searched):
    """The default configuration is trained as train_cv.py trains it: the variant's
    default network with the experiment over it. A trial started from the class default
    of 0.0 would fail, outside the logarithmic range."""
    values = tuning.default_search_values("r4", penalty_searched)
    assert values["correction_penalty"] == penalty_searched["correction_penalty"] == 0.1
    assert values["width_size"] == 12 and values["depth"] == 7


def test_default_values_outside_the_search_space_are_refused_before_the_search(penalty_searched):
    with pytest.raises(ValueError, match="correction_penalty"):
        tuning.default_search_values("r4")


def test_the_search_starts_from_the_default_values_of_its_experiment(tmp_path, monkeypatch,
                                                                        penalty_searched):
    import tune_hyperparameters

    captured = {}
    monkeypatch.setattr(tuning, "run_study", lambda **options: captured.update(options))
    monkeypatch.setattr(sys, "argv", [
        "tune_hyperparameters.py", "--model", "r4", "--n-trials", "1", "--epochs", "2",
        "--experiment", str(EXPERIMENTS / "fold_fast_narrow_tan.json"),
        "--benchmarks", str(_fake_benchmarks(tmp_path / "benchmarks")),
        "--run-dir", str(tmp_path / "runs"), "--start-from-defaults"])
    assert tune_hyperparameters.main() == 0
    assert captured["enqueue"]["correction_penalty"] == 0.1


def test_a_nested_search_starts_from_the_default_values_on_its_own_calibrations(tmp_path,
                                                                                   fake_training):
    inner, benchmarks = nested_split(SPLIT, 1), tmp_path / "benchmarks"
    study = tuning.run_study("r4", inner, data_dir=tmp_path, run_root=tmp_path / "runs",
                             n_trials=3, num_epochs=7, storage="journal",
                             tuning_name="nested-x_outer1", study_name="nested_r4_outer1",
                             benchmarks=benchmarks, enqueue=_default_r4())
    assert study.trials[0].params == _default_r4() and len(study.trials) == 3
    assert {arguments["fold"] for arguments in fake_training} == {1, 2, 3, 4}
    assert all(arguments["benchmarks"] == benchmarks for arguments in fake_training)
    assert all(arguments["split"].same_assignment(inner) for arguments in fake_training)
    payload = json.loads((tmp_path / "runs" / "r4" / "tuning_nested-x_outer1"
                          / "best_hyperparameters.json").read_text(encoding="utf-8"))
    assert Path(payload["benchmarks"]) == benchmarks
    assert DataSplit.from_dict(payload["split"]).same_assignment(inner)
    assert payload["hyperparameters"]["num_epochs"] == 7


#: Where an R3 search started.
R3_START ={"width_size": 16, "depth": 5, "dropout_rate": 0.2, "learning_rate": 0.003,
            "regularization_lambda": 1e-4, "batch_size": 4}


def _finished_search(directory: Path, trials: list[tuple[dict, float, str]], best: int) -> None:
    """A collected search: best_hyperparameters.json started from R3_START, and trials.csv
    without a column for the correction penalty."""
    import pandas as pd

    directory.mkdir(parents=True)
    rows = [{"number": number, "value": value, "state": state,
             **{f"params_{name}": setting for name, setting in values.items()}}
            for number, (values, value, state) in enumerate(trials)]
    pd.DataFrame(rows).to_csv(directory / "trials.csv", index=False)
    values, value, _ = trials[best]
    payload = {"hyperparameters": {**values, "warmup_steps": 600, "num_epochs": 200,
                                   "correction_penalty": 0.1},
               "n_trials_completed": sum(state == "COMPLETE" for _, _, state in trials),
               "best_trial": best, "mean_val_loss": value, "started_from": R3_START}
    (directory / "best_hyperparameters.json").write_text(json.dumps(payload), encoding="utf-8")


def test_the_summary_compares_the_selection_with_where_its_search_started(tmp_path, monkeypatch):
    """Not with where a search would start now: the search space may have changed since,
    for example by a dimension such as the correction penalty that the search lacked."""
    import pandas as pd

    other = {**R3_START, "width_size": 32, "learning_rate": 0.001}
    # outer fold 1 selected where it started; in outer fold 2 the start failed
    _finished_search(tmp_path / "r3" / "outer1", [(R3_START, 0.5, "COMPLETE"),
                                                  (other, 0.6, "COMPLETE")], best=0)
    _finished_search(tmp_path / "r3" / "outer2", [(R3_START, math.nan, "FAIL"),
                                                  (other, 0.4, "COMPLETE")], best=1)
    output = tmp_path / "selections.csv"
    import nested_summary

    monkeypatch.setattr(sys, "argv", [
        "nested_summary.py", "--models", "r3", "--split", str(ROOT / "configs" / "split.json"),
        "--pattern", str(tmp_path / "{model}" / "outer{fold}" / "best_hyperparameters.json"),
        "--output", str(output)])
    assert nested_summary.main() == 0
    table = pd.read_csv(output).set_index("outer_fold")
    assert table.loc[1, "inner_loss_default"] == 0.5 and table.loc[1, "default_selected"]
    assert math.isnan(table.loc[2, "inner_loss_default"])
    assert not table.loc[2, "default_selected"]
    assert (table["correction_penalty"] == 0.1).all()


# ---------------------------------------------------------------------------
# The final models: each outer fold trained with its own selection
# ---------------------------------------------------------------------------
def _selections(root: Path, recipe=None, **per_fold) -> str:
    """best_hyperparameters.json per outer fold, with width 8 k in fold k; the pattern."""
    for fold in range(1, SPLIT.k_folds + 1):
        values = {**(recipe or Hyperparameters()).to_dict(), "width_size": 8 * fold,
                  **{name: value(fold) for name, value in per_fold.items()}}
        directory = root / "r4" / f"tuning_nested-x_outer{fold}"
        directory.mkdir(parents=True)
        (directory / "best_hyperparameters.json").write_text(
            json.dumps({"model_key": "r4", "hyperparameters": values}), encoding="utf-8")
    return str(root / "{model}" / "tuning_nested-x_outer{fold}" / "best_hyperparameters.json")


def test_each_fold_gets_its_own_selection_with_the_experiment_over_it(tmp_path):
    pattern = _selections(tmp_path)
    experiment = training.load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json")
    recipes = training.fold_recipes(pattern, "r4", [1, 2, 5], experiment, num_epochs=11)
    assert sorted(recipes) == [1, 2, 5]
    assert [recipes[fold].width_size for fold in (1, 2, 5)] == [8, 16, 40]
    assert all(recipe.num_epochs == 11 and recipe.calibration == "per-fold" for recipe in recipes.values())


def test_selections_from_searches_of_different_models_are_refused(tmp_path):
    pattern = _selections(tmp_path, spin_up_days=lambda fold: 30.0 if fold < 3 else 20.0)
    with pytest.raises(ValueError, match="spin_up_days"):
        training.fold_recipes(pattern, "r4", [1, 2, 3, 4, 5])


@pytest.mark.parametrize("pattern", ["runs/r4/best_hyperparameters.json", "missing_{fold}.json"])
def test_a_pattern_without_a_fold_or_a_missing_selection_is_refused(tmp_path, pattern):
    with pytest.raises((ValueError, FileNotFoundError)):
        training.fold_recipes(str(tmp_path / pattern), "r4", [1])


@pytest.fixture
def train_cv(monkeypatch):
    """scripts/train_cv.py with its folds recorded instead of trained."""
    import train_cv as module

    jobs = []

    def worker(job):
        jobs.append(job)
        return {"fold": job["fold"], "train_loss": 0.1, "val_loss": 0.2, "val_labels": [],
                "n_parameters": 1, "checkpoint_dir": ""}

    monkeypatch.setattr(module, "_worker", worker)

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["train_cv.py", *arguments])
        return module.main()

    return run, jobs


def test_train_cv_trains_each_fold_with_its_own_selection(tmp_path, train_cv):
    run, jobs = train_cv
    # every fold built on its per-fold calibration, of the model the experiment names
    recipe = training.apply_experiment(
        Hyperparameters(), training.load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json"))
    pattern = _selections(tmp_path / "searches", recipe)
    benchmarks = _fake_benchmarks(tmp_path / "benchmarks")
    assert run("--model", "r4", "--fold-hyperparameters", pattern, "--epochs", "3",
               "--run-dir", str(tmp_path / "runs"), "--run-name", "tuned",
               "--data-dir", str(tmp_path / "data"), "--benchmarks", str(benchmarks)) == 0
    assert sorted((job["fold"], job["hyperparameters"]["width_size"]) for job in jobs) == \
        [(fold, 8 * fold) for fold in range(1, SPLIT.k_folds + 1)]
    assert all(job["hyperparameters"]["num_epochs"] == 3 for job in jobs)
    run_directory = tmp_path / "runs" / "r4" / "tuned"
    recorded = json.loads((run_directory / "fold_hyperparameters.json").read_text(encoding="utf-8"))
    assert [recorded[str(fold)]["width_size"] for fold in range(1, 6)] == [8, 16, 24, 32, 40]
    # no single recipe that would misstate four of the five folds
    assert not (run_directory / "hyperparameters.json").exists()
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["fold_hyperparameters"] == pattern


def test_train_cv_takes_one_recipe_or_one_per_fold_not_both(tmp_path, train_cv):
    run, jobs = train_cv
    pattern = _selections(tmp_path / "searches")
    with pytest.raises(SystemExit):
        run("--model", "r4", "--fold-hyperparameters", pattern,
            "--hyperparameters", pattern.replace("{model}", "r4").replace("{fold}", "1"),
            "--run-dir", str(tmp_path / "runs"))
    assert jobs == []


# ---------------------------------------------------------------------------
# Ensembles of runs trained with a selection per fold
# ---------------------------------------------------------------------------
def _tuned_run(root: Path, name: str, seed: int, widths=(8, 16, 24, 32, 40)) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "split.json").write_text(json.dumps(SPLIT.to_dict()), encoding="utf-8")
    (run / "fold_hyperparameters.json").write_text(json.dumps(
        {str(fold): Hyperparameters(width_size=width).to_dict()
         for fold, width in enumerate(widths, start=1)}), encoding="utf-8")
    (run / "metadata.json").write_text(json.dumps({
        "model_key": "r4", "seed": seed, "experiment": "configs/experiments/fold_fast_narrow_tan.json",
        "fold_hyperparameters": "runs/{model}/tuning_nested-x_outer{fold}/best_hyperparameters.json"}),
        encoding="utf-8")
    return run


def test_an_ensemble_of_tuned_runs_is_named_apart_from_the_untuned_one(tmp_path):
    members = [_tuned_run(tmp_path, f"fold_fast_narrow_tan_seed{seed}_tuned", seed) for seed in (45, 46, 47)]
    name = ensembles.default_ensemble_name(members)
    assert name == "fold_fast_narrow_tan_tuned_ensemble_s45-46-47"
    ensembles.create_ensemble(tmp_path / name, "r4", members)


def test_tuned_runs_with_another_selection_in_one_fold_are_no_ensemble(tmp_path):
    members = [_tuned_run(tmp_path, "a", 45), _tuned_run(tmp_path, "b", 46, widths=(8, 16, 24, 32, 32))]
    with pytest.raises(ValueError, match="fold 5"):
        ensembles.create_ensemble(tmp_path / "ensemble", "r4", members)
