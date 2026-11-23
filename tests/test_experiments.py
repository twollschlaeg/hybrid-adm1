"""What a run trains: the recipe defaults, the experiment files and the fold records.

The hyperparameters say how to train, an experiment file says what model to
train, and every fold records the model it trained so evaluation can rebuild it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import _bootstrap
import pytest
from helpers import EXPERIMENTS, canonical_split, write_benchmarks

from adm1_package.experiments import training
from adm1_package.experiments.calibrations import calibration_fingerprint
from adm1_package.experiments.config import RunPaths
from adm1_package.experiments.evaluation import fold_settings
from adm1_package.experiments.registry import get_spec
from adm1_package.experiments.training import (
    Hyperparameters,
    apply_experiment,
    check_experiments,
    default_hyperparameters,
    load_experiment,
    loss_weights,
    validation_interval,
)
from adm1_package.utils.corrections import DEFAULT, layout_for

SPLIT = canonical_split()
FAST_BOUNDS = {"k_hyd_ch": [0.1, 2.88]}


def test_a_recipe_without_the_model_settings_gets_the_current_defaults():
    recipe = Hyperparameters.from_dict({"width_size": 8})
    assert recipe.initial_state == "calibrated"
    assert recipe.spin_up_days == 30.0
    assert recipe.split_network is True
    assert recipe.calibration == "per-fold" and recipe.forcing == "smoothstep"
    assert recipe.correction["parametrization"] == "calibrated"
    assert recipe.objective_weights == {}


#: The experiments of the paper's runs. Every other was retired.
PAPER_EXPERIMENTS = ["fold_calibrated", "fold_calibrated_narrow", "fold_calibrated_narrow_no_ph",
                     "fold_arrhenius_narrow", "fold_fast_narrow", "fold_fast_narrow_tan",
                     "fold_fast_narrow_tan_linear", "fold_fast_narrow_tan_no_ph",
                     "fold_fast_acids_r3_no_ph", "fold_fast_acids_r2_no_ph",
                     "fold_fast_pruned_r3_no_ph", "fold_fast_pruned_r2_no_ph"]


def test_the_experiment_files_are_those_of_the_paper():
    assert sorted(path.stem for path in EXPERIMENTS.glob("*.json")) == sorted(PAPER_EXPERIMENTS)


@pytest.mark.parametrize("name", PAPER_EXPERIMENTS)
def test_every_experiment_anchors_a_split_network_on_the_per_fold_calibration(name):
    experiment = load_experiment(EXPERIMENTS / f"{name}.json")
    assert experiment["calibration"] == "per-fold"
    assert experiment["correction"]["parametrization"] == "calibrated"
    assert experiment["correction"]["exact_anchors"] is True
    assert experiment["split_network"] is True
    assert experiment["initial_state"] == "calibrated" and experiment["spin_up_days"] == 30.0


def test_validation_runs_ten_times_per_run():
    assert validation_interval(Hyperparameters(num_epochs=200)) == 20
    # a run shorter than ten epochs still validates, or it has no checkpoint to restore
    assert validation_interval(Hyperparameters(num_epochs=5)) == 1


def test_an_experiment_overrides_the_model_and_keeps_the_training_recipe():
    recipe = default_hyperparameters("r4")
    run = apply_experiment(recipe, load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json"))
    assert run.correction["physics"] == ["arrhenius", "tan_inflow"] and run.split_network
    assert (run.width_size, run.depth, run.learning_rate) == (recipe.width_size, recipe.depth,
                                                              recipe.learning_rate)


def test_an_experiment_with_an_unknown_setting_is_refused(tmp_path):
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"spin_up_day": 10.0}), encoding="utf-8")
    with pytest.raises(KeyError, match="spin_up_day"):
        load_experiment(path)


def test_a_fold_is_rebuilt_from_its_record():
    config = {"correction": DEFAULT.to_dict(), "initial_state": "warmup", "spin_up_days": 30.0,
              "forcing": "smoothstep", "calibration": "per-fold", "calibration_fingerprint": "f"}
    settings = fold_settings(SimpleNamespace(config=config))
    assert settings == {"correction": DEFAULT, "initial_state": "warmup", "spin_up_days": 30.0,
                        "forcing": "smoothstep", "calibration": "per-fold",
                        "calibration_fingerprint": "f"}


def test_a_checkpoint_from_an_earlier_version_is_refused():
    with pytest.raises(KeyError, match="earlier version"):
        fold_settings(SimpleNamespace(config={"correction": DEFAULT.to_dict()}))


def test_the_banner_says_what_network_a_run_trains():
    split = _bootstrap.describe_model(
        apply_experiment(Hyperparameters(), load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json")))
    single = _bootstrap.describe_model(Hyperparameters(split_network=False))
    assert "split" in split and "day 30" in split
    assert "single" in single


def test_the_banner_says_where_the_anchors_come_from():
    text = _bootstrap.describe_model(Hyperparameters(correction_penalty=0.1, keep_checkpoints=True))
    assert "per-fold calibration" in text
    assert "correction penalty 0.1" in text and "checkpoint at every validation" in text
    plain = _bootstrap.describe_model(Hyperparameters())
    assert "penalty" not in plain
    fast = _bootstrap.describe_model(Hyperparameters(calibration_bounds={"k_hyd_ch": [0.1, 2.88]}))
    assert "per-fold calibration fitted with k_hyd_ch in [0.1, 2.88]" in fast


@pytest.mark.parametrize("key", ["r2", "r3", "r4"])
def test_water_content_is_scored_but_not_fitted(key):
    # Water varies little within a reactor and the models miss it by far more than that, so
    # in a loss scaled by each reactor's range it would outweigh gas and TAN together.
    spec = get_spec(key)
    assert "S_h2o" in spec.outputs
    assert spec.objective_weights.get("S_h2o", 0.0) == 0.0
    assert "S_h2o" not in spec.objective_list


@pytest.mark.parametrize("key", ["r2", "r3"])
def test_the_ph_experiment_scores_ph_without_fitting_it(key):
    # Both models miss pH by a nearly constant offset, which a loss scaled by each reactor's
    # small pH range turns into a large share of the R2/R3 loss, as it does for water.
    spec = get_spec(key)
    run = apply_experiment(default_hyperparameters(key),
                           load_experiment(EXPERIMENTS / "fold_calibrated_narrow_no_ph.json"))
    weights, fitted = loss_weights(spec, run)
    assert weights["pH"] == 0.0 and "pH" not in fitted
    assert "pH" in spec.outputs  # still simulated, so still scored
    assert fitted == [name for name in spec.objective_list if name != "pH"]
    assert {name: value for name, value in weights.items() if name != "pH"} == {
        name: value for name, value in spec.objective_weights.items() if name != "pH"}


@pytest.mark.parametrize("key", ["r2", "r3", "r4"])
def test_without_overrides_a_run_fits_what_the_registry_fits(key):
    spec = get_spec(key)
    assert loss_weights(spec, default_hyperparameters(key)) == (
        dict(spec.objective_weights), list(spec.objective_list))


def test_a_loss_weight_for_an_output_the_variant_lacks_is_refused():
    with pytest.raises(KeyError, match="pH"):
        loss_weights(get_spec("r4"), Hyperparameters(objective_weights={"pH": 0.0}))


def test_a_negative_loss_weight_is_refused():
    with pytest.raises(ValueError, match="S_TAN"):
        loss_weights(get_spec("r4"), Hyperparameters(objective_weights={"S_TAN": -1.0}))


def test_no_experiment_keeps_a_checkpoint_at_every_validation():
    """Scoring each fold at an epoch chosen from the other folds' curves gains nothing over the
    final epoch, while ten checkpoints per fold would bloat every run. A fold keeps its final
    and its best."""
    keeping = sorted(path.stem for path in EXPERIMENTS.glob("*.json")
                     if load_experiment(path).get("keep_checkpoints"))
    assert keeping == []


def test_the_fold_calibrated_arms_each_change_one_thing():
    """per-fold anchors -> a narrow correction with a penalty."""
    anchored = load_experiment(EXPERIMENTS / "fold_calibrated.json")
    narrow = load_experiment(EXPERIMENTS / "fold_calibrated_narrow.json")
    # Exact anchors make "at zero output it is MECH-CV" hold where a calibrated
    # fraction lies outside the logit band (e.g. at 1.0).
    assert anchored["correction"]["exact_anchors"] is True
    assert narrow["correction"] == {**anchored["correction"], "corrected": ["f_ch", "k_hyd_ch"],
                                    "bound": pytest.approx(math.log(2.0))}
    assert narrow["correction_penalty"] > 0
    assert ({key: value for key, value in narrow.items()
             if key not in ("correction", "correction_penalty")}
            == {key: value for key, value in anchored.items() if key != "correction"})


def test_the_temperature_arms_are_the_narrow_arm_on_temperature_dependent_anchors():
    """narrow with Arrhenius, built on the Arrhenius calibration
    (runs/benchmarks-smoothstep-arrhenius) or on the fast-hydrolysis one
    (runs/benchmarks-smoothstep-fast)."""
    narrow = load_experiment(EXPERIMENTS / "fold_calibrated_narrow.json")
    warm = load_experiment(EXPERIMENTS / "fold_arrhenius_narrow.json")
    fast = load_experiment(EXPERIMENTS / "fold_fast_narrow.json")
    assert warm == {**narrow, "correction": {**narrow["correction"], "physics": ["arrhenius"]}}
    assert fast == {**warm, "calibration_bounds": {"k_hyd_ch": [0.1, 2.88]}}


def test_the_tan_arm_is_the_fast_arm_with_a_lever_on_influent_tan():
    """R4 ties TAN to carbohydrate degradation through biomass growth, so the corrections that
    fix the gas also move TAN. The influent TAN factor of physics tan_inflow moves TAN alone,
    within the same factor of two."""
    fast = load_experiment(EXPERIMENTS / "fold_fast_narrow.json")
    tan = load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json")
    assert tan == {**fast, "correction": {**fast["correction"], "physics": ["arrhenius", "tan_inflow"],
                                          "corrected": ["f_ch", "k_hyd_ch", "f_TAN_in"]}}


def test_the_linear_baseline_is_the_tan_arm_with_a_linear_map_for_a_network():
    """Is the network needed? The TAN arm with each network a single linear layer and no
    dropout: the same anchors, links, bounds, penalty and loss, with the corrections a linear
    map of the inputs."""
    tan = load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json")
    linear = load_experiment(EXPERIMENTS / "fold_fast_narrow_tan_linear.json")
    assert linear == {**tan, "depth": 1, "dropout_rate": 0.0}
    recipe = default_hyperparameters("r4")
    run = apply_experiment(recipe, linear)
    assert (run.depth, run.dropout_rate) == (1, 0.0)
    assert (run.learning_rate, run.warmup_steps) == (recipe.learning_rate, recipe.warmup_steps)


def test_each_r4_arm_runs_on_its_calibrations_only(tmp_path):
    """The plain arms on calibrations without Arrhenius, the temperature arm on those with
    it, the fast arms on those with it and the fast-hydrolysis bounds."""
    plain = write_benchmarks(tmp_path / "plain")
    warm = write_benchmarks(tmp_path / "warm", arrhenius=True)
    fast = write_benchmarks(tmp_path / "fast", arrhenius=True, bounds=FAST_BOUNDS)
    plain_arms = ["fold_calibrated", "fold_calibrated_narrow"]
    warm_arms = ["fold_arrhenius_narrow"]
    fast_arms = ["fold_fast_narrow", "fold_fast_narrow_tan", "fold_fast_narrow_tan_linear"]
    for arms, benchmarks in ((plain_arms, plain), (warm_arms, warm), (fast_arms, fast)):
        check_experiments(arms, ["r4"], split=SPLIT, benchmarks=benchmarks)
    for arms, benchmarks, message in ((fast_arms, warm, "bounds"), (fast_arms, plain, "Arrhenius"),
                                      (warm_arms, fast, "bounds"), (warm_arms, plain, "Arrhenius"),
                                      (plain_arms, warm, "Arrhenius")):
        with pytest.raises(ValueError, match=message):
            check_experiments(arms, ["r4"], split=SPLIT, benchmarks=benchmarks)


@pytest.mark.parametrize("arm", ["fold_calibrated_narrow", "fold_fast_narrow_tan"])
def test_the_r3_r2_arms_are_the_r4_arms_with_ph_scored_but_not_fitted(arm):
    """R2 and R3 miss pH by a nearly constant offset, a large share of their loss, which the
    narrow corrections cannot move."""
    no_ph = load_experiment(EXPERIMENTS / f"{arm}_no_ph.json")
    assert no_ph.pop("objective_weights") == {"pH": 0.0}
    assert no_ph == load_experiment(EXPERIMENTS / f"{arm}.json")


def test_the_r3_r2_arms_run_on_their_calibrations_and_not_on_r4(tmp_path):
    plain = write_benchmarks(tmp_path / "plain")
    fast = write_benchmarks(tmp_path / "fast", arrhenius=True, bounds=FAST_BOUNDS)
    check_experiments(["fold_calibrated_narrow_no_ph"], ["r3", "r2"], split=SPLIT, benchmarks=plain)
    check_experiments(["fold_fast_narrow_tan_no_ph"], ["r3", "r2"], split=SPLIT, benchmarks=fast)
    with pytest.raises(ValueError, match="on r4"):
        check_experiments(["fold_fast_narrow_tan_no_ph"], ["r4"], split=SPLIT, benchmarks=fast)
    # refused before anything of the batch queues
    with pytest.raises(ValueError, match="fold_calibrated_narrow_no_ph on r4"):
        check_experiments(["fold_calibrated_narrow_no_ph"], ["r3", "r4"], split=SPLIT,
                          benchmarks=plain)


#: The R3 and R2 arms that correct more than the R4 parameters (f_ch, k_hyd_ch, f_TAN_in):
#: "acids" adds the growth kinetics each variant adds to R4; "pruned" holds only what the
#: pruning holds for structural reasons (corrections.PRUNED), but not the acidogens' decay,
#: which the same reasoning calls dead.
LESS_PRUNED = {
    "fold_fast_acids_r3_no_ph": ("r3", ["f_ch", "k_hyd_ch", "k_m_ac", "f_TAN_in"]),
    "fold_fast_acids_r2_no_ph": ("r2", ["f_ch", "k_hyd_ch", "k_m_pro", "k_m_ac", "f_TAN_in"]),
    "fold_fast_pruned_r3_no_ph": ("r3", ["f_ch", "f_pr", "k_hyd_ch", "k_m_ac", "k_dec_Xac",
                                         "f_TAN_in"]),
    "fold_fast_pruned_r2_no_ph": ("r2", ["f_ch", "f_pr", "k_hyd_ch", "k_m_pro", "k_m_ac",
                                         "k_dec_Xpro", "k_dec_Xac", "f_TAN_in"]),
}


@pytest.mark.parametrize("arm", sorted(LESS_PRUNED))
def test_the_less_pruned_arms_differ_from_the_final_r3_r2_arm_in_what_they_correct(arm):
    key, outputs = LESS_PRUNED[arm]
    final = load_experiment(EXPERIMENTS / "fold_fast_narrow_tan_no_ph.json")
    less = load_experiment(EXPERIMENTS / f"{arm}.json")
    corrected = less["correction"].pop("corrected")
    final["correction"].pop("corrected")
    assert less == final
    assert list(layout_for(key, {**less["correction"], "corrected": corrected}).outputs) == outputs


@pytest.mark.parametrize("arm", sorted(LESS_PRUNED))
def test_a_less_pruned_arm_is_refused_on_a_variant_without_its_parameters(arm, tmp_path):
    fast = write_benchmarks(tmp_path / "fast", arrhenius=True, bounds=FAST_BOUNDS)
    key, _ = LESS_PRUNED[arm]
    check_experiments([arm], [key], split=SPLIT, benchmarks=fast)
    others = ["r4", "r3"] if key == "r2" else ["r4"]
    for other in others:
        with pytest.raises(ValueError, match="no correctable parameter"):
            check_experiments([arm], [other], split=SPLIT, benchmarks=fast)


def test_the_banner_says_when_the_loss_weights_change():
    def banner(name):
        return _bootstrap.describe_model(
            apply_experiment(Hyperparameters(), load_experiment(EXPERIMENTS / f"{name}.json")))

    assert "loss weights: pH 0" in banner("fold_calibrated_narrow_no_ph")
    assert "loss weights" not in banner("fold_calibrated_narrow")


@pytest.fixture
def recorded_trainer(monkeypatch):
    """What train_fold hands the trainer, which stands in for it without training."""
    import jax

    captured = {}

    class Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.state = SimpleNamespace(model=[])

        def train_model(self, *args, **kwargs):
            return {"train/loss": 0.0, "val/loss": 0.0}

    monkeypatch.setattr(type(get_spec("r4")), "trainer_class", lambda self: Recorder)
    # nothing was compiled for the next tests to compile again
    monkeypatch.setattr(jax, "clear_caches", lambda: None)
    return captured


@pytest.mark.requires_data
def test_a_fold_trains_with_the_overridden_loss_weights(recorded_trainer, tmp_path, data_dir,
                                                        installed_benchmarks):
    spec = get_spec("r3")
    benchmarks = installed_benchmarks(tmp_path / "benchmarks", SPLIT, [1])
    recipe = Hyperparameters(num_epochs=1, noise_copies=1, objective_weights={"pH": 0.0})
    training.train_fold(spec, data_dir, SPLIT, 1, recipe, RunPaths("r3", "weights", root=tmp_path),
                        benchmarks=benchmarks)
    assert recorded_trainer["objective_weights"]["pH"] == 0.0
    assert recorded_trainer["objective_weights"]["S_TAN"] == spec.objective_weights["S_TAN"]
    assert recorded_trainer["objective_list"] == [name for name in spec.objective_list
                                                  if name != "pH"]


@pytest.mark.requires_data
def test_a_per_fold_fold_trains_on_its_calibration_and_records_it(
        recorded_trainer, monkeypatch, tmp_path, data_dir, installed_benchmarks):
    spec, read = get_spec("r4"), []
    dataset_of = type(spec).dataset

    def spy(self, directory, *args, **kwargs):
        read.append(Path(directory))
        return dataset_of(self, directory, *args, **kwargs)

    monkeypatch.setattr(type(spec), "dataset", spy)
    benchmarks = installed_benchmarks(tmp_path / "benchmarks", SPLIT, [1])
    # keep_checkpoints set against its default, so that its passing on is tested
    recipe = apply_experiment(Hyperparameters(num_epochs=1, noise_copies=1),
                              {**load_experiment(EXPERIMENTS / "fold_calibrated_narrow.json"),
                               "keep_checkpoints": True})
    training.train_fold(spec, data_dir, SPLIT, 1, recipe, RunPaths("r4", "anchored", root=tmp_path),
                        benchmarks=benchmarks)

    assert read == [benchmarks / "fold_1" / "data"]
    dataset = dataset_of(spec, benchmarks / "fold_1" / "data", forcing="smoothstep")
    captured = recorded_trainer
    assert captured["forcing"] == "smoothstep"
    assert captured["calibration"] == "per-fold"
    assert captured["calibration_fingerprint"] == calibration_fingerprint(dataset)
    assert captured["correction"]["corrected"] == ["f_ch", "k_hyd_ch"]
    assert captured["correction_penalty"] == 0.1 and captured["keep_checkpoints"] is True
    # the retired options are no longer recorded
    assert "memory_days" not in captured and "objective_scale" not in captured
