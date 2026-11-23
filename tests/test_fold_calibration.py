"""Anchoring each fold's hybrid on the fold's own calibration (``calibration="per-fold"``).

The installed calibration in data/ has seen every reactor, the held-out ones too, so a
calibrated hybrid built on it would start from parameters fitted to the reactors it is
scored on; those hybrids were retired (test_retired.py). Built on the calibration of its
fold's training reactors (the one MECH-CV uses, scripts/calibrate/calibrate_folds.py), a
calibrated hybrid at zero output *is* MECH-CV, and the comparison with MECH-CV is fair by
construction. Which experiment runs on which calibrations is in test_experiments.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import canonical_split, write_benchmarks

from adm1_package.experiments import calibrations, evaluation
from adm1_package.experiments.registry import DATASETS
from adm1_package.experiments.splits import DataSplit
from adm1_package.experiments.training import Hyperparameters, check_experiments, fold_data_dir
from adm1_package.utils.corrections import DEFAULT

SPLIT = canonical_split()
PER_FOLD = Hyperparameters(calibration="per-fold")


# ---------------------------------------------------------------------------
# Where a fold reads its parameters and start values
# ---------------------------------------------------------------------------
def test_a_per_fold_run_trains_and_is_scored_on_its_folds_calibration(tmp_path):
    benchmarks = write_benchmarks(tmp_path / "benchmarks")
    assert (fold_data_dir(PER_FOLD, tmp_path / "data", SPLIT, 2, benchmarks)
            == benchmarks / "fold_2" / "data")
    settings = {"calibration": "per-fold", "calibration_fingerprint": None}
    assert (evaluation.hybrid_data_dir(settings, tmp_path / "data", SPLIT, 3, benchmarks)
            == benchmarks / "fold_3" / "data")


def test_a_per_fold_run_without_benchmarks_says_how_to_make_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate_folds"):
        fold_data_dir(PER_FOLD, tmp_path, SPLIT, 1, tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="per-fold"):
        evaluation.hybrid_data_dir({"calibration": "per-fold"}, tmp_path, SPLIT, 1, None)


def test_a_per_fold_experiment_is_refused_before_its_benchmarks_exist(tmp_path):
    """The cluster's job-submission scripts check this before anything is queued."""
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "anchored.json").write_text(json.dumps({"calibration": "per-fold"}),
                                               encoding="utf-8")
    with pytest.raises(ValueError, match="calibrate_folds"):
        check_experiments(["anchored"], ["r4"], experiments, split=SPLIT,
                          benchmarks=tmp_path / "missing")
    check_experiments(["anchored"], ["r4"], experiments, split=SPLIT,
                      benchmarks=write_benchmarks(tmp_path / "benchmarks"))


def test_benchmarks_of_another_split_are_refused(tmp_path):
    payload = SPLIT.to_dict()
    payload["folds"]["fold_1"], payload["folds"]["fold_2"] = (payload["folds"]["fold_2"],
                                                              payload["folds"]["fold_1"])
    benchmarks = write_benchmarks(tmp_path / "benchmarks", DataSplit.from_dict(payload))
    with pytest.raises(ValueError, match="another split"):
        calibrations.fold_benchmark_data(benchmarks, SPLIT, 1)


def test_an_unfinished_per_fold_benchmark_is_refused(tmp_path):
    benchmarks = write_benchmarks(tmp_path / "benchmarks", unfinished=1)
    with pytest.raises(FileNotFoundError, match="LaRA-substrate"):
        calibrations.fold_benchmark_data(benchmarks, SPLIT, 1)


def test_benchmarks_say_what_model_they_calibrated_and_within_what_bounds(tmp_path):
    root = write_benchmarks(tmp_path / "benchmarks")
    # made before the bounds were recorded: within the stored ones
    (root / "model.json").write_text(json.dumps({"arrhenius": True, "forcing": "smoothstep"}),
                                     encoding="utf-8")
    assert calibrations.benchmark_model(root) == {"arrhenius": True, "bounds": {},
                                                "forcing": "smoothstep"}
    bounds = {"k_hyd_ch": [0.1, 2.88]}
    (root / "model.json").write_text(json.dumps({"arrhenius": True, "bounds": bounds,
                                                 "forcing": "smoothstep"}), encoding="utf-8")
    assert calibrations.benchmark_model(root) == {"arrhenius": True, "bounds": bounds,
                                                "forcing": "smoothstep"}


@pytest.mark.parametrize("bounds", [{"k_hyd_ch": [0.3, 0.1]}, {"k_hyd_ch": [0.1]}])
def test_calibration_bounds_must_be_a_range(bounds):
    with pytest.raises(ValueError, match="k_hyd_ch"):
        Hyperparameters(calibration="per-fold", calibration_bounds=bounds)


def test_the_installed_calibration_says_which_forcing_it_was_made_with(tmp_path):
    for campaign in DATASETS:
        (tmp_path / campaign).mkdir()
    # calibrated before the smoothstep, so recorded nothing
    assert calibrations.installed_forcing(tmp_path) == "spline"
    for campaign in DATASETS:
        (tmp_path / campaign / "calibration.json").write_text(json.dumps({"forcing": "smoothstep"}),
                                                              encoding="utf-8")
    assert calibrations.installed_forcing(tmp_path) == "smoothstep"
    (tmp_path / DATASETS[0] / "calibration.json").write_text(json.dumps({"forcing": "spline"}),
                                                             encoding="utf-8")
    with pytest.raises(ValueError, match="forcing"):
        calibrations.installed_forcing(tmp_path)


# ---------------------------------------------------------------------------
# A fold records the calibration it trained on, and evaluation checks it
# ---------------------------------------------------------------------------
class _Dataset:
    """Two reactors with the given parameters, as far as a fingerprint reads them."""

    labels = ["LaRA-temp_F1", "rice-bran_R30"]

    def __init__(self, parameters, start=(1.0, 2.0, 3.0)):
        self._sample = ({"parameters": dict(parameters), "initial_values": np.asarray(start)}, {})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self._sample


def test_the_fingerprint_tells_calibrations_apart():
    fingerprint = calibrations.calibration_fingerprint
    reference = fingerprint(_Dataset({"k_hyd_ch": 0.05, "V_liq": 10.0}))
    assert reference == fingerprint(_Dataset({"V_liq": 10.0, "k_hyd_ch": 0.05}))
    assert reference != fingerprint(_Dataset({"k_hyd_ch": 0.06, "V_liq": 10.0}))
    assert reference != fingerprint(_Dataset({"k_hyd_ch": 0.05, "V_liq": 10.0}, start=(1, 2, 4)))


def test_the_fingerprint_ignores_the_last_bits_of_a_float():
    """Another machine's CSV parser may round the last bit differently."""
    fingerprint = calibrations.calibration_fingerprint
    assert fingerprint(_Dataset({"k": 0.1 + 0.2})) == fingerprint(_Dataset({"k": 0.3}))


def test_a_per_fold_run_records_its_calibration():
    config = {"correction": DEFAULT.to_dict(), "initial_state": "calibrated", "spin_up_days": 30.0,
              "forcing": "smoothstep", "calibration": "per-fold", "calibration_fingerprint": "abc"}
    settings = evaluation.fold_settings(SimpleNamespace(config=config))
    assert (settings["calibration"], settings["calibration_fingerprint"]) == ("per-fold", "abc")


def test_a_per_fold_run_needs_calibrations_made_with_its_forcing(tmp_path):
    benchmarks = write_benchmarks(tmp_path / "benchmarks")
    calibrations.check_forcing({"forcing": "smoothstep"}, benchmarks, "fold 1")


def test_a_calibration_changed_since_training_is_refused():
    dataset = _Dataset({"k": 1.0})
    trained = {"calibration": "per-fold",
               "calibration_fingerprint": calibrations.calibration_fingerprint(dataset)}
    calibrations.check_calibration(dataset, trained, "fold 1")
    with pytest.raises(ValueError, match="changed"):
        calibrations.check_calibration(_Dataset({"k": 1.1}), trained, "fold 1")
