"""Records of the paper's runs load although options were retired (``experiments/retired.py``).

Every run of the paper records each option it was trained with, the retired ones at the
values that changed nothing. Those are dropped. Any other value names a model this code
does not build, and is refused as retired before this release. So is a request for a
retired model: the literature-anchored ("global") hybrids, hybrids on the installed
calibration, the spline forcing, network transfer between variants and every optimizer
but Adam.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
from helpers import canonical_split, write_benchmarks

from adm1_package.experiments import calibrations, ensembles, evaluation
from adm1_package.experiments.retired import RETIRED_FIELDS, REFUSAL, drop_retired
from adm1_package.experiments.training import Hyperparameters, fold_data_dir, load_experiment
from adm1_package.utils.corrections import CorrectionOptions
from adm1_package.utils.forcing import FORCINGS, forcing_coefficients, resolve_forcing

#: The recipe of fold 1 of one of the paper's R4 runs (fold_fast_narrow_tan), as recorded
#: in its fold_hyperparameters.json (and in the nested search's best_hyperparameters.json).
RECORDED_RECIPE = {
    "width_size": 36, "depth": 3, "dropout_rate": 0.30000000000000004,
    "learning_rate": 0.003930268786823028, "regularization_lambda": 0.00023710570549623767,
    "batch_size": 4, "warmup_steps": 600, "gradient_clip": 10.0, "num_epochs": 200,
    "length_schedule": [[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]],
    "noise_copies": 16, "noise_output": 0.1, "noise_input": 0.0, "synthetic_ratio": 0.0,
    "campaign_balance": False, "validation_every": 0, "objective_weights": {},
    "loss_scale": "reactor", "correction_penalty": 0.1, "keep_checkpoints": False,
    "correction": {"parametrization": "calibrated", "prune": True,
                   "physics": ["arrhenius", "tan_inflow"], "references": {},
                   "corrected": ["f_TAN_in", "f_ch", "k_hyd_ch"],
                   "bound": 0.6931471805599453, "exact_anchors": True},
    "initial_state": "calibrated", "spin_up_days": 30.0, "forcing": "smoothstep",
    "split_network": True, "memory_days": 0.0, "calibration": "per-fold",
    "calibration_bounds": {"k_hyd_ch": [0.1, 2.88]},
}

#: What the same fold's hparams.json records about its model, beside the network.
RECORDED_FOLD = {
    "correction": RECORDED_RECIPE["correction"], "initial_state": "calibrated",
    "memory_days": 0.0, "spin_up_days": 30.0, "forcing": "smoothstep",
    "calibration": "per-fold", "calibration_fingerprint": "762538d7",
    "objective_scale": None, "correction_penalty": 0.1, "keep_checkpoints": False,
}

#: A retired field at a value that changed the model, per field.
CHANGED = {"memory_days": 10.0, "synthetic_ratio": 0.5, "campaign_balance": True,
           "validation_every": 5, "loss_scale": "pooled", "objective_scale": [1.0, 2.0]}

#: A retired model, as a recipe or a fold's record names it; ``None``: not recorded, which
#: was the retired choice when the record was written.
RETIRED_MODELS = [("calibration", "installed"), ("calibration", None), ("forcing", "spline"),
                  ("forcing", None)]


# ---------------------------------------------------------------------------
# Retired fields
# ---------------------------------------------------------------------------
def test_every_retired_field_has_a_changed_value_to_test():
    assert set(CHANGED) == set(RETIRED_FIELDS)


def test_a_retired_field_at_its_inert_value_is_dropped():
    kept = drop_retired({"memory_days": 0.0, "loss_scale": "reactor", "width_size": 8})
    assert kept == {"width_size": 8}


@pytest.mark.parametrize("name", sorted(CHANGED))
def test_a_retired_field_that_changed_the_model_is_refused(name):
    with pytest.raises(ValueError, match=REFUSAL) as error:
        drop_retired({name: CHANGED[name]}, "somewhere")
    assert "somewhere" in str(error.value) and name in str(error.value)


# ---------------------------------------------------------------------------
# A run's recipe
# ---------------------------------------------------------------------------
def test_a_recorded_recipe_of_the_papers_runs_loads_as_it_trained():
    recipe = Hyperparameters.from_dict(RECORDED_RECIPE)
    recorded = drop_retired(RECORDED_RECIPE)
    # every field that remains is read back unchanged
    assert {name: value for name, value in recipe.to_dict().items() if name != "length_schedule"} == {
        name: value for name, value in recorded.items() if name != "length_schedule"}
    assert recipe.length_schedule == [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0)]


@pytest.mark.parametrize("name, value", [
    ("memory_days", 10.0), ("calibration", "installed"), ("forcing", "spline"),
    ("correction", {**RECORDED_RECIPE["correction"], "parametrization": "global"}),
])
def test_a_recorded_recipe_with_a_retired_option_is_refused(name, value):
    with pytest.raises(ValueError, match=REFUSAL):
        Hyperparameters.from_dict({**RECORDED_RECIPE, name: value})


def test_an_unknown_field_or_forcing_is_still_refused():
    with pytest.raises(KeyError, match="widht"):
        Hyperparameters.from_dict({**RECORDED_RECIPE, "widht_size": 8})
    with pytest.raises(ValueError, match="smoothstep"):
        Hyperparameters(forcing="cubic")


def test_an_experiment_file_with_retired_fields_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"description": "-", "synthetic_ratio": 0.0, "memory_days": 0.0,
                                "calibration": "per-fold"}), encoding="utf-8")
    assert load_experiment(path) == {"calibration": "per-fold"}
    path.write_text(json.dumps({"memory_days": 10.0}), encoding="utf-8")
    with pytest.raises(ValueError, match=REFUSAL):
        load_experiment(path)


def test_the_recipes_of_a_run_read_through_the_retirement(tmp_path):
    (tmp_path / "fold_hyperparameters.json").write_text(
        json.dumps({"1": RECORDED_RECIPE, "2": RECORDED_RECIPE}), encoding="utf-8")
    recipes = ensembles._run_recipes(tmp_path)
    assert set(recipes) == {"1", "2"} and recipes["1"] == recipes["2"]
    assert "memory_days" not in recipes["1"]


# ---------------------------------------------------------------------------
# A fold's record
# ---------------------------------------------------------------------------
def _fold_settings(**changes):
    """The recorded fold with ``changes``; a change to ``None`` leaves the field out."""
    config = {name: value for name, value in {**RECORDED_FOLD, **changes}.items()
              if not (name in changes and value is None)}
    return evaluation.fold_settings(SimpleNamespace(config=config, log_dir="fold_1"))


def test_a_recorded_fold_reads_as_it_trained():
    settings = _fold_settings()
    assert settings["forcing"] == "smoothstep"
    assert settings["calibration"] == "per-fold"
    assert settings["spin_up_days"] == 30.0
    assert settings["correction"].corrected == ("f_TAN_in", "f_ch", "k_hyd_ch")
    assert "memory_days" not in settings


@pytest.mark.parametrize("name, value", [("memory_days", 10.0), ("objective_scale", [1.0, 2.0]),
                                         *RETIRED_MODELS])
def test_a_recorded_fold_with_a_retired_option_is_refused(name, value):
    with pytest.raises(ValueError, match=REFUSAL):
        _fold_settings(**{name: value})


# ---------------------------------------------------------------------------
# Retired models
# ---------------------------------------------------------------------------
def test_the_global_parametrization_was_retired():
    with pytest.raises(ValueError, match=REFUSAL):
        CorrectionOptions("global")
    with pytest.raises(ValueError, match=REFUSAL):
        CorrectionOptions.from_dict({"parametrization": "global"})


def test_the_retired_parametrization_flag_is_gone():
    import _bootstrap

    parser = argparse.ArgumentParser()
    _bootstrap.add_correction_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--ude", "global"])


def test_the_smoothstep_is_the_only_forcing():
    assert FORCINGS == ("smoothstep",)
    assert resolve_forcing("smoothstep") == "smoothstep"
    # a record that names the spline, or none, which was the spline
    for retired in ("spline", None):
        with pytest.raises(ValueError, match=REFUSAL):
            resolve_forcing(retired)
        with pytest.raises(ValueError, match=REFUSAL):
            forcing_coefficients([0.0, 1.0], [[1.0], [2.0]], retired)


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2", "base"])
def test_a_dataset_refuses_the_spline(key, data_dir):
    from adm1_package.experiments.registry import get_spec

    with pytest.raises(ValueError, match=REFUSAL):
        get_spec(key).dataset(data_dir, data_to_use=["LaRA-temp"], forcing="spline")


def test_per_fold_calibrations_on_the_spline_anchor_no_run(tmp_path):
    """Made before the forcing was recorded, they were fitted to the spline."""
    spline = write_benchmarks(tmp_path / "spline", forcing=None)
    with pytest.raises(ValueError, match=REFUSAL):
        calibrations.benchmark_model(spline)
    with pytest.raises(ValueError, match=REFUSAL):
        fold_data_dir(Hyperparameters(), tmp_path, canonical_split(), 1, spline)
    with pytest.raises(ValueError, match=REFUSAL):
        calibrations.check_forcing({"forcing": "smoothstep"}, spline, "fold 1")


def test_a_transfer_between_variants_was_retired(tmp_path):
    (tmp_path / ensembles.ENSEMBLE_FILE).write_text(
        json.dumps({"model_key": "r3", "source_model_key": "r4", "members": ["a"], "seeds": [42]}),
        encoding="utf-8")
    with pytest.raises(ValueError, match=REFUSAL):
        ensembles.ensemble_source(tmp_path)


def test_an_optimizer_other_than_adam_was_retired():
    from adm1_package.utils.ude_trainer import _check_optimizer

    _check_optimizer({"optimizer": "adam", "lr_scheduler": "warmup_cosine_decay"})
    for hparams in ({"optimizer": "lbfgs"}, {"optimizer": ["adam", "lbfgs"]},
                    {"optimizer": "adam", "lr_scheduler": "constant"}):
        with pytest.raises(ValueError, match=REFUSAL):
            _check_optimizer(hparams)
