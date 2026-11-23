"""Scoring the mean prediction of several runs of one experiment, trained with different seeds.

One training run per fold varies with its seed by more than the arms of an experiment differ
from each other, so a result is reported for the mean prediction of a few seeds: an ensemble.
``evaluate.py --ensemble`` makes an ensemble directory beside the runs, which then stands in
for a run in ``evaluate.py --run`` and the figures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adm1_package.experiments import ensembles, evaluation, splits
from adm1_package.experiments.predictions import FoldPrediction, mean_prediction
from adm1_package.experiments.splits import load_split
from adm1_package.experiments.training import Hyperparameters

ROOT = Path(__file__).resolve().parents[1]
SPLIT = load_split(ROOT / "configs" / "split.json")


# ---------------------------------------------------------------------------
# The mean prediction
# ---------------------------------------------------------------------------
class _Reactors:
    """One reactor measured at 2.0 for 40 days, as a dataset hands it out."""

    def __getitem__(self, index):
        inputs = {"parameters": {}, "initial_values": None, "input": {"time": np.arange(40.0)}}
        targets = {"object_indices": {"q_gasout": 0}, "values": np.full((40, 1), 2.0),
                   "indices": np.ones((40, 1)), "label": "R1"}
        return inputs, targets


def _constant(value):
    return lambda parameters, initial, forcing, network, key: ({"q_gasout": np.full(40, value)}, None)


def test_the_ensemble_is_scored_on_the_mean_of_its_members_predictions():
    spec = SimpleNamespace(outputs=("q_gasout",))
    members = [(_constant(1.0), None, None), (_constant(3.0), None, None)]
    (sample,) = evaluation.ensemble_samples(spec, _Reactors(), [0], members, spin_up_days=30.0)
    scored = sample.mask[:, 0]
    np.testing.assert_array_equal(sample.predictions[scored, 0], 2.0)
    assert scored.sum() == 10  # days 30-39: the spin-up and the first point are not scored
    (single,) = evaluation.ensemble_samples(spec, _Reactors(), [0], members[:1], spin_up_days=30.0)
    np.testing.assert_array_equal(single.predictions[single.mask[:, 0], 0], 1.0)


def test_one_member_is_scored_as_a_single_run_is():
    """evaluate_run scores a run as an ensemble of one: it must score as simulate_sample does."""
    import jax.numpy as jnp

    from adm1_package.experiments.metrics import simulate_sample

    spec = SimpleNamespace(outputs=("q_gasout",))
    curve = jnp.linspace(1.0, 3.0, 40)
    member = (lambda parameters, initial, forcing, network, key: ({"q_gasout": curve}, None), "net", "key")
    (ensemble,) = evaluation.ensemble_samples(spec, _Reactors(), [0], [member], spin_up_days=30.0)
    single = simulate_sample(member[0], _Reactors(), 0, spec.outputs, model="net", key="key",
                             spin_up_days=30.0)
    np.testing.assert_array_equal(ensemble.predictions, single.predictions)
    np.testing.assert_array_equal(ensemble.mask, single.mask)


def _record(label, hybrid_value, rate_value):
    times = np.arange(5.0)
    return FoldPrediction(label=label, fold=1, split="validation", times=times,
                          hybrid={"q_gasout": np.full(5, hybrid_value), "dynamic_rates": np.full((5, 2), rate_value)},
                          mechanistic={"q_gasout": np.full(5, 9.0)})


def test_the_figures_draw_the_mean_of_the_members_and_their_learned_corrections():
    mean = mean_prediction([_record("R1", 1.0, 0.1), _record("R1", 3.0, 0.3)])
    np.testing.assert_allclose(mean.hybrid["q_gasout"], 2.0)
    np.testing.assert_allclose(mean.hybrid["dynamic_rates"], 0.2)
    np.testing.assert_allclose(mean.mechanistic["q_gasout"], 9.0)
    one = _record("R1", 1.0, 0.1)
    assert mean_prediction([one]) is one
    with pytest.raises(ValueError, match="R2"):
        mean_prediction([_record("R1", 1.0, 0.1), _record("R2", 3.0, 0.3)])


# ---------------------------------------------------------------------------
# The ensemble directory
# ---------------------------------------------------------------------------
def _fake_run(root: Path, name: str, seed: int, **recipe) -> Path:
    """A run directory as train_cv.py leaves it, without its folds."""
    run = root / name
    run.mkdir(parents=True)
    (run / "split.json").write_text(json.dumps(SPLIT.to_dict()), encoding="utf-8")
    (run / "hyperparameters.json").write_text(json.dumps({**Hyperparameters().to_dict(), **recipe}),
                                              encoding="utf-8")
    (run / "metadata.json").write_text(json.dumps({
        "model_key": "r4", "seed": seed,
        "experiment": "configs\\experiments\\fold_fast_narrow_tan.json"}), encoding="utf-8")
    return run


def test_an_ensemble_directory_lists_its_members_beside_it(tmp_path):
    members = [_fake_run(tmp_path, f"fold_fast_narrow_tan_seed{seed}_T", seed) for seed in (42, 43, 44)]
    name = ensembles.default_ensemble_name(members)
    assert name == "fold_fast_narrow_tan_ensemble_s42-43-44"
    directory = ensembles.create_ensemble(tmp_path / name, "r4", members)
    manifest = json.loads((directory / ensembles.ENSEMBLE_FILE).read_text(encoding="utf-8"))
    assert manifest["members"] == [member.name for member in members]  # relative: they move together
    assert manifest["seeds"] == [42, 43, 44] and manifest["model_key"] == "r4"
    assert ensembles.is_ensemble(directory) and not ensembles.is_ensemble(members[0])
    assert ensembles.ensemble_members(directory) == [member.resolve() for member in members]
    assert splits.load_run_split(directory).same_assignment(SPLIT)
    # made again from the same members it is the same ensemble; from others it is refused
    ensembles.create_ensemble(directory, "r4", members)
    with pytest.raises(ValueError, match="other members"):
        ensembles.create_ensemble(directory, "r4", members[:2])


def test_a_run_from_before_a_recipe_field_existed_is_the_same_experiment(tmp_path):
    """A run recorded before calibration_bounds was a recipe field lacks it."""
    old = _fake_run(tmp_path, "a", 42)
    recipe = json.loads((old / "hyperparameters.json").read_text(encoding="utf-8"))
    del recipe["calibration_bounds"]
    (old / "hyperparameters.json").write_text(json.dumps(recipe), encoding="utf-8")
    ensembles.create_ensemble(tmp_path / "ensemble", "r4", [old, _fake_run(tmp_path, "b", 43)])


@pytest.mark.parametrize("change, message", [
    ({"seed": 42}, "seed 42"),
    ({"recipe": {"num_epochs": 150}}, "num_epochs"),
    ({"model_key": "r3"}, "r3"),
    ({"split": True}, "split"),
])
def test_members_of_one_experiment_are_required(tmp_path, change, message):
    first = _fake_run(tmp_path, "a", 42)
    second = _fake_run(tmp_path, "b", change.get("seed", 43), **change.get("recipe", {}))
    if "model_key" in change:
        metadata = json.loads((second / "metadata.json").read_text(encoding="utf-8"))
        (second / "metadata.json").write_text(json.dumps({**metadata, "model_key": change["model_key"]}),
                                              encoding="utf-8")
    if change.get("split"):
        payload = SPLIT.to_dict()
        payload["folds"]["fold_1"], payload["folds"]["fold_2"] = (payload["folds"]["fold_2"],
                                                                  payload["folds"]["fold_1"])
        (second / "split.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        ensembles.create_ensemble(tmp_path / "ensemble", "r4", [first, second])


def test_an_ensemble_needs_two_runs(tmp_path):
    run = _fake_run(tmp_path, "a", 42)
    with pytest.raises(ValueError, match="two"):
        ensembles.create_ensemble(tmp_path / "ensemble", "r4", [run])
    with pytest.raises(ValueError, match="two"):
        ensembles.create_ensemble(tmp_path / "ensemble", "r4", [run, run])


def test_runs_of_different_experiments_need_a_name(tmp_path):
    first = _fake_run(tmp_path, "a", 42)
    second = _fake_run(tmp_path, "b", 43)
    metadata = json.loads((second / "metadata.json").read_text(encoding="utf-8"))
    metadata["experiment"] = "configs/experiments/fold_fast_narrow.json"
    (second / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="--name"):
        ensembles.default_ensemble_name([first, second])


def test_evaluate_takes_a_run_or_an_ensemble_of_two_or_more():
    for arguments, message in ((["--ensemble", "runs/r4/a"], "two"),
                               (["--run", "runs/r4/a", "--ensemble", "runs/r4/b", "runs/r4/c"], "not allowed")):
        result = subprocess.run([sys.executable, "scripts/evaluate.py", "--model", "r4", *arguments],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode != 0 and message in result.stderr, result.stderr
