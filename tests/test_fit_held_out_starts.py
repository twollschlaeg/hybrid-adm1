"""Fitting a fold's held-out reactors' start values on their first days
(scripts/calibrate/fit_held_out_starts.py).

The fit may read a held-out reactor's forcing and its measurements before the scores begin,
and nothing else of it: not its later record, and not its installed start values, which the
calibration on all reactors fitted on that later record.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import canonical_split, load_script

from adm1_package.experiments import held_out_starts
from adm1_package.experiments.registry import get_spec

SPLIT = canonical_split()
fit = load_script("calibrate/fit_held_out_starts.py")

REACTOR = "LaRA-temp_F5"
MAXITER = 8


def _held_out_fold(label: str) -> int:
    return next(fold for fold in range(1, SPLIT.k_folds + 1)
                if label in SPLIT.validation_labels(fold))


@pytest.fixture(scope="module")
def reactors(installed_dataset):
    """R4's LaRA-temp reactors as installed: ``(dataset, x, y, ranges)`` for REACTOR, with
    the loss scales of the others."""
    dataset = installed_dataset("r4", ["LaRA-temp"])
    labels = dataset.labels
    others = [dataset[i] for i, label in enumerate(labels) if label != REACTOR]
    x, y = dataset[labels.index(REACTOR)]
    return dataset, x, y, fit.campaign_ranges(get_spec("r4"), others)


def _fitted(x, y, ranges):
    return fit.fit_reactor("r4", x, y, ranges, arrhenius=False, maxiter=MAXITER)


def _changed_after(x, y, day, factor):
    """``y`` with every target from ``day`` of the record on scaled by ``factor``."""
    time = np.asarray(x["input"]["time"])
    later = jnp.asarray(time >= time[0] + day)[:, None]
    return {**y, "values": jnp.where(later, y["values"] * factor, y["values"])}


@pytest.mark.requires_data
def test_the_fit_reads_nothing_of_the_reactor_after_its_first_days(reactors):
    _, x, y, ranges = reactors
    first = _fitted(x, y, ranges)
    # another later record, and other installed biology
    names = held_out_starts.state_names("r4")
    biology = jnp.array([names.index(name) for name in ("X_ch", "X_pr", "X_li", "X_bac")])
    x_other = {**x, "initial_values": x["initial_values"].at[biology].multiply(3.0)}
    again = _fitted(x_other, _changed_after(x, y, fit.DAYS, 2.0), ranges)
    assert again["start"] == pytest.approx(first["start"], rel=1e-9)
    assert again["loss"]["fitted"] == pytest.approx(first["loss"]["fitted"], rel=1e-9)
    assert again["loss"]["starts"] == pytest.approx(first["loss"]["starts"], rel=1e-9)


@pytest.mark.requires_data
def test_the_fit_follows_the_reactors_first_days(reactors):
    _, x, y, ranges = reactors
    first = _fitted(x, y, ranges)
    other = _fitted(x, _changed_after(x, y, 1.0, 1.5), ranges)
    assert other["start"] != pytest.approx(first["start"], rel=1e-6)


@pytest.mark.requires_data
def test_the_fitted_starts_lie_within_their_bounds_and_fit_the_first_days_better(reactors):
    _, x, y, ranges = reactors
    result = _fitted(x, y, ranges)
    assert set(result["start"]) == {"X_ch", "X_pr", "X_li", "X_bac"}
    for name, cap in result["caps"].items():
        assert 0.0 <= result["start"][name] <= cap * (1 + 1e-12)
    assert result["start"]["X_bac"] > 0.0
    assert result["loss"]["fitted"] <= result["loss"]["guess"]
    assert result["days"] == fit.DAYS


@pytest.mark.requires_data
def test_the_fit_keeps_the_better_of_the_warm_up_and_the_training_reactors_starts(reactors):
    """A warm-up can wash a model's acid degraders out (R2 at fast hydrolysis on LaRA-temp),
    and a fit from there stays in that basin; the median start of the campaign's training
    reactors is a second place to start from, and the better fit of the first days wins."""
    dataset, x, y, ranges = reactors
    names = held_out_starts.state_names("r4")
    others = [dataset[i] for i, label in enumerate(dataset.labels) if label != REACTOR]
    median = fit.training_medians("r4", others)
    assert set(median) == {"X_ch", "X_pr", "X_li", "X_bac"}
    result = fit.fit_reactor("r4", x, y, ranges, arrhenius=False, maxiter=MAXITER,
                             alternatives={"training median": median})
    assert set(result["loss"]["starts"]) == {"warm-up", "training median"}
    assert result["from"] in result["loss"]["starts"]
    assert result["loss"]["fitted"] <= min(result["loss"]["starts"].values())
    assert result["loss"]["guess"] == result["loss"]["starts"]["warm-up"]
    expected = float(np.median([float(sample[0]["initial_values"][names.index("X_bac")])
                                for sample in others]))
    assert median["X_bac"] == pytest.approx(expected)


@pytest.mark.requires_data
def test_r2_at_fast_hydrolysis_does_not_keep_the_washed_out_warm_up(fast_benchmarks):
    """The case that makes the training reactors' start necessary: R2 on the fast-hydrolysis
    calibration of fold 2, LaRA-temp_F3. From the warm-up alone the first 30 days score
    poorly, its acids piling up for the rest of the record. Read from the paper's
    calibrations."""
    from adm1_package.experiments.calibrations import benchmark_model

    benchmarks = fast_benchmarks
    data = benchmarks / "fold_2" / "data"

    spec = get_spec("r2")
    dataset = spec.dataset(data, data_to_use=["LaRA-temp"],
                           forcing=benchmark_model(benchmarks)["forcing"])
    held = [label for label in SPLIT.validation_labels(2) if label.startswith("LaRA-temp")]
    training = [dataset[i] for i, label in enumerate(dataset.labels) if label not in held]
    x, y = dataset[dataset.labels.index("LaRA-temp_F3")]
    result = fit.fit_reactor("r2", x, y, fit.campaign_ranges(spec, training), arrhenius=True,
                             maxiter=40, ceilings=fit.training_ceilings("r2", training),
                             alternatives={"training median": fit.training_medians("r2", training)})
    assert result["loss"]["fitted"] < 0.6
    assert result["start"]["X_ac"] > 0.1


@pytest.mark.requires_data
def test_a_folds_held_out_starts_are_written_for_each_variant(data_dir, tmp_path,
                                                             installed_benchmarks):
    fold = _held_out_fold(REACTOR)
    benchmarks = installed_benchmarks(tmp_path / "benchmarks", SPLIT, [fold])
    written = fit.fit_fold(benchmarks, SPLIT, fold, "LaRA-temp", variants=("r4",),
                           arrhenius=False, maxiter=2)
    assert written == held_out_starts.starts_file(benchmarks, fold, "LaRA-temp")
    record = json.loads(written.read_text(encoding="utf-8"))
    assert (record["fold"], record["campaign"], record["days"]) == (fold, "LaRA-temp", fit.DAYS)
    held = [label for label in SPLIT.validation_labels(fold) if label.startswith("LaRA-temp")]
    starts, days = held_out_starts.read_starts(benchmarks, fold, "r4", held)
    assert sorted(starts) == sorted(held) and days == fit.DAYS


def test_a_campaign_the_fold_holds_nothing_out_of_has_nothing_to_fit(tmp_path):
    # two folds hold no rice-bran reactor out
    fold, campaign = next(
        (fold, campaign) for fold in range(1, SPLIT.k_folds + 1)
        for campaign in ("LaRA-temp", "LaRA-substrate", "rice-bran")
        if not any(label.startswith(campaign) for label in SPLIT.validation_labels(fold)))
    assert fit.fit_fold(tmp_path, SPLIT, fold, campaign, variants=("r4",), arrhenius=False) is None


def test_a_biomass_start_may_reach_twice_what_the_training_reactors_started_with():
    """The calibration lets a start move up to twice its stored value; a held-out reactor has
    none it may read, so the largest start the fold's calibration gave the campaign's
    training reactors stands in. The particulates stay at most at their first-week steady
    state."""
    bounds = fit.start_bounds("r4", guess={"X_ch": 5.0, "X_pr": 1.0, "X_li": 0.5, "X_bac": 2.0},
                              caps={"X_ch": 5.0, "X_pr": 1.0, "X_li": 0.5},
                              ceilings={"X_bac": 18.0})
    assert bounds == [(0.0, 5.0), (0.0, 1.0), (0.0, 0.5), (0.0, 36.0)]
    # never below the calibration's own floor of 10 g/L
    (_, _, _, biomass) = fit.start_bounds("r4", guess={"X_ch": 5.0, "X_pr": 1.0, "X_li": 0.5,
                                                       "X_bac": 2.0},
                                          caps={"X_ch": 5.0, "X_pr": 1.0, "X_li": 0.5},
                                          ceilings={"X_bac": 1.0})
    assert biomass == (0.0, 10.0)


def test_the_window_ends_before_the_first_scored_day():
    time = np.arange(0.0, 60.0)
    measured = np.ones((60, 2), dtype=bool)
    window = fit.window_mask(measured, time, 30.0)
    assert not window[0].any()                    # the start, which the datasets set
    assert window[1:30].all()
    assert not window[30:].any()                  # scored from day 30 on
