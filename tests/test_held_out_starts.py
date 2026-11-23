"""Starting a fold's held-out reactors from start values fitted on their first days only
(adm1_package/experiments/held_out_starts.py).

The installed start values of a held-out reactor were fitted on its whole record, the part
that is scored included, with parameters calibrated on it. A per-fold benchmark copies them
as they are; ``scripts/calibrate/fit_held_out_starts.py`` fits them again from the unscored
first days, with the fold's own parameters, and evaluation starts the reactor there.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from adm1_package.experiments import evaluation, held_out_starts, splits
from adm1_package.experiments.registry import get_spec
from adm1_package.experiments.splits import DataSplit

SPLIT = DataSplit(
    test=[],
    folds=[["LaRA-temp_F1", "rice-bran_R30"], ["LaRA-temp_F2", "rice-bran_R60"]],
    seed=0,
)


class _Reactors:
    """Two reactors, laid out as the datasets keep them, with ``size`` states each: the
    first starts at 1, 2, 3, ..., the second at 101, 102, ..."""

    def __init__(self, size=3):
        self.data = [{"initial_values": jnp.arange(1.0, size + 1), "input": {"time": 0}},
                     {"initial_values": jnp.arange(101.0, size + 101), "input": {"time": 0}}]
        self.targets = [{"label": "LaRA-temp_F1"}, {"label": "LaRA-temp_F2"}]

    @property
    def labels(self):
        return [target["label"] for target in self.targets]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def _write(benchmarks: Path, fold: int, campaign: str, starts: dict, days: float = 30.0) -> Path:
    path = held_out_starts.starts_file(benchmarks, fold, campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fold": fold, "campaign": campaign, "days": days,
                                "variants": {key: {label: {"start": values}
                                                   for label, values in by_label.items()}
                                             for key, by_label in starts.items()}}),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Putting fitted start values into a dataset
# ---------------------------------------------------------------------------
def test_fitted_starts_replace_only_the_unmeasured_states_of_their_reactor():
    reactors = _Reactors()
    started = held_out_starts.with_starts(reactors, {"LaRA-temp_F2": {"X_ch": 50.0}},
                                          ("S_TAN", "X_ch", "X_bac"))
    np.testing.assert_array_equal(started[1][0]["initial_values"], [101.0, 50.0, 103.0])
    np.testing.assert_array_equal(started[0][0]["initial_values"], [1.0, 2.0, 3.0])
    # the dataset it was given starts as before
    np.testing.assert_array_equal(reactors[1][0]["initial_values"], [101.0, 102.0, 103.0])


def test_starts_laid_out_for_another_state_vector_are_refused():
    with pytest.raises(ValueError, match="3 states"):
        held_out_starts.with_starts(_Reactors(), {"LaRA-temp_F1": {"X_ch": 50.0}},
                                    held_out_starts.state_names("r4"))


def test_starts_for_a_reactor_the_dataset_lacks_are_refused():
    with pytest.raises(KeyError, match="LaRA-temp_F9"):
        held_out_starts.with_starts(_Reactors(), {"LaRA-temp_F9": {"X_ch": 1.0}},
                                    ("S_TAN", "X_ch", "X_bac"))


def test_a_variants_states_are_laid_out_as_its_dataset_keeps_them():
    assert held_out_starts.state_names("r4") == ("S_ch4", "S_IC", "S_TAN", "S_h2o", "X_ch",
                                                 "X_pr", "X_li", "X_bac", "S_gas_ch4",
                                                 "S_gas_co2")
    assert "X_ac" in held_out_starts.state_names("r3")
    assert "X_pro" in held_out_starts.state_names("r2")


# ---------------------------------------------------------------------------
# Reading them
# ---------------------------------------------------------------------------
def test_the_starts_of_a_folds_held_out_reactors_are_read_per_variant(tmp_path):
    _write(tmp_path, 1, "LaRA-temp", {"r4": {"LaRA-temp_F1": {"X_ch": 7.0, "X_bac": 0.5}},
                                      "r3": {"LaRA-temp_F1": {"X_ch": 8.0}}}, days=30.0)
    _write(tmp_path, 1, "rice-bran", {"r4": {"rice-bran_R30": {"X_ch": 0.2}}}, days=25.0)
    starts, days = held_out_starts.read_starts(tmp_path, 1, "r4",
                                               ["LaRA-temp_F1", "rice-bran_R30"])
    assert starts == {"LaRA-temp_F1": {"X_ch": 7.0, "X_bac": 0.5},
                      "rice-bran_R30": {"X_ch": 0.2}}
    assert days == 30.0   # the longest window any of them was fitted on


def test_a_held_out_reactor_without_fitted_starts_says_how_to_fit_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit_held_out_starts.py"):
        held_out_starts.read_starts(tmp_path, 1, "r4", ["LaRA-temp_F1"])
    _write(tmp_path, 1, "LaRA-temp", {"r3": {"LaRA-temp_F1": {"X_ch": 8.0}}})
    with pytest.raises(FileNotFoundError, match="fit_held_out_starts.py"):
        held_out_starts.read_starts(tmp_path, 1, "r4", ["LaRA-temp_F1"])


@pytest.mark.parametrize("value", ["fitted", "installed"])
def test_the_two_ways_to_start_a_held_out_reactor(value):
    assert held_out_starts.resolve_held_out_starts(value) == value


def test_another_way_to_start_a_held_out_reactor_is_refused():
    with pytest.raises(ValueError, match="fitted"):
        held_out_starts.resolve_held_out_starts("warm")


# ---------------------------------------------------------------------------
# A fold's dataset, its held-out reactors started as asked
# ---------------------------------------------------------------------------
def test_a_folds_held_out_reactors_start_from_their_fitted_values(tmp_path):
    _write(tmp_path, 1, "LaRA-temp", {"r4": {"LaRA-temp_F1": {"X_ch": 7.0}}})
    reactors = _Reactors(size=10)
    started, days = held_out_starts.held_out_view(reactors, "r4", tmp_path, SPLIT, 1, "fitted",
                                                  spin_up_days=30.0)
    names = held_out_starts.state_names("r4")
    assert float(started[0][0]["initial_values"][names.index("X_ch")]) == 7.0
    assert days == 30.0
    # the training reactor keeps its own
    np.testing.assert_array_equal(started[1][0]["initial_values"], np.arange(101.0, 111.0))


def test_installed_starts_leave_the_dataset_as_it_is(tmp_path):
    reactors = _Reactors()
    started, days = held_out_starts.held_out_view(reactors, "r4", tmp_path, SPLIT, 1,
                                                  "installed", spin_up_days=30.0)
    assert started is reactors and days is None


def test_starts_fitted_on_days_that_are_scored_are_refused(tmp_path):
    """A fit over the first 45 days used points that scoring from day 30 counts."""
    _write(tmp_path, 1, "LaRA-temp", {"r4": {"LaRA-temp_F1": {"X_ch": 7.0}}}, days=45.0)
    with pytest.raises(ValueError, match="45"):
        held_out_starts.held_out_view(_Reactors(size=10), "r4", tmp_path, SPLIT, 1, "fitted",
                                      spin_up_days=30.0)


# ---------------------------------------------------------------------------
# On the real datasets: a fitted start is a start file with other biology
# ---------------------------------------------------------------------------
@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2"])
def test_fitted_starts_start_a_reactor_as_a_start_file_with_that_biology_would(key, data_dir,
                                                                              tmp_path):
    """The datasets set the chemistry from the first measurements; it must not depend on
    the biology, or replacing the biology after loading would start elsewhere."""
    spec = get_spec(key)
    campaign, reactor = "LaRA-temp", "F3"
    names = held_out_starts.state_names(key)
    copy = tmp_path / campaign
    shutil.copytree(Path(data_dir) / campaign, copy,
                    ignore=shutil.ignore_patterns("calibration-backup", "*.xlsx"))
    iv_file = copy / f"{key.upper()}-IVs_{reactor}.csv"
    stored = pd.read_csv(iv_file, index_col=0).iloc[:, 0]
    biology = {name: float(stored[name]) * 1.7 + 0.3 for name in names if name.startswith("X_")}
    edited = stored.copy()
    for name, value in biology.items():
        edited[name] = value
    edited.to_csv(iv_file)

    installed = spec.dataset(data_dir, data_to_use=[campaign])
    rebuilt = spec.dataset(tmp_path, data_to_use=[campaign])
    label = f"{campaign}_{reactor}"
    started = held_out_starts.with_starts(installed, {label: biology}, names)
    for index, name in enumerate(installed.labels):
        np.testing.assert_allclose(np.asarray(started[index][0]["initial_values"]),
                                   np.asarray(rebuilt[index][0]["initial_values"]),
                                   rtol=1e-12, atol=0.0, err_msg=name)


# ---------------------------------------------------------------------------
# Scoring a run with them
# ---------------------------------------------------------------------------
def test_results_scored_otherwise_go_to_a_directory_of_their_own():
    assert evaluation.results_name("last") == "results"
    assert evaluation.results_name("best") == "results-best"
    assert evaluation.results_name("last", held_out_starts="installed") == "results-installed-starts"
    assert evaluation.results_name("last", spin_up_days=45.0) == "results-from-day45"
    assert evaluation.results_name("last", initial_state="warmup") == "results-warmup"
    assert evaluation.results_name("best", held_out_starts="installed", spin_up_days=60.0,
                                   initial_state="warmup") == (
        "results-best-installed-starts-from-day60-warmup")


@pytest.mark.requires_data
def test_a_per_fold_hybrid_starts_where_its_anchor_does(data_dir, paper_r4_run,
                                                        fast_benchmarks):
    """Opened for figures and inspection, a fold's held-out reactors start from the fitted
    values, the hybrid and its anchor alike; ``installed`` starts them from the fold's copy.
    On a run of the paper and the calibrations it was built on."""
    run, benchmarks = paper_r4_run, fast_benchmarks
    spec = get_spec("r4")
    split = splits.load_run_split(run)
    names = held_out_starts.state_names("r4")
    held = split.validation_labels(1)
    fitted, _ = held_out_starts.read_starts(benchmarks, 1, "r4", held)
    copied = spec.dataset(benchmarks / "fold_1" / "data")
    for how in ("fitted", "installed"):
        opened = evaluation.open_fold(spec, run, split, 1, data_dir, benchmarks=benchmarks,
                                      held_out_starts=how)
        for label in held:
            start = opened.dataset[opened.dataset.labels.index(label)][0]["initial_values"]
            expected = (fitted[label] if how == "fitted" else
                        {name: float(copied[copied.labels.index(label)][0]["initial_values"][
                            names.index(name)]) for name in fitted[label]})
            for name, value in expected.items():
                assert float(start[names.index(name)]) == pytest.approx(value, rel=1e-12), (
                    how, label, name)
