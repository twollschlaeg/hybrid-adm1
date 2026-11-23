"""Metric definitions and the paired statistics used for the significance claims."""

from __future__ import annotations

import numpy as np
import pytest

from adm1_package.experiments.metrics import (
    SamplePrediction,
    objective_metrics,
    paired_comparison,
    pooled_metrics,
    simulate_sample,
)

OBJECTIVES = ("q_gasout", "S_TAN")


def _sample(predictions, targets, mask=None, label="reactor"):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if mask is None:
        mask = np.ones_like(predictions, dtype=bool)
    return SamplePrediction(
        label=label,
        objectives=OBJECTIVES,
        predictions=predictions,
        targets=targets,
        mask=np.asarray(mask, dtype=bool),
    )


def test_perfect_prediction_scores_zero_error():
    targets = [[10.0, 1.0], [12.0, 1.5], [14.0, 2.0]]
    scores = pooled_metrics([_sample(targets, targets)])
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["nrmse"] == pytest.approx(0.0)
    assert scores["r2"] == pytest.approx(1.0)


def test_rmse_is_computed_per_objective():
    targets = np.array([[10.0, 1.0], [10.0, 1.0]])
    predictions = np.array([[12.0, 1.0], [12.0, 1.0]])
    per_objective = objective_metrics(
        predictions, targets, np.ones_like(targets, bool), OBJECTIVES
    )
    assert per_objective["q_gasout"]["rmse"] == pytest.approx(2.0)
    assert per_objective["S_TAN"]["rmse"] == pytest.approx(0.0)


def test_masked_points_are_excluded():
    targets = np.array([[10.0, 1.0], [10.0, 1.0], [999.0, 999.0]])
    predictions = np.array([[10.0, 1.0], [10.0, 1.0], [0.0, 0.0]])
    mask = np.array([[True, True], [True, True], [False, False]])
    scores = pooled_metrics([_sample(predictions, targets, mask)])
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["n_observations"] == 4


def test_nrmse_normalizes_by_the_observed_range():
    # Objective 1 spans 100, objective 2 spans 1; an error of 10% of each range
    # must contribute equally.
    targets = np.array([[0.0, 0.0], [100.0, 1.0]])
    predictions = np.array([[10.0, 0.1], [110.0, 1.1]])
    scores = pooled_metrics([_sample(predictions, targets)])
    assert scores["nrmse"] == pytest.approx(0.1)


class _OneReactor:
    """One record of 60 days starting on day 5, for simulate_sample."""

    def __getitem__(self, index):
        inputs = {"parameters": {}, "initial_values": None, "input": {"time": 5.0 + np.arange(60.0)}}
        targets = {"values": np.zeros((60, 2)), "indices": np.ones((60, 2)), "label": "reactor",
                   "object_indices": {"q_gasout": 0, "S_TAN": 1}}
        return inputs, targets


def _constant_simulator(parameters, initial_values, inputs):
    return {"q_gasout": np.ones(60), "S_TAN": np.ones(60)}, None


def test_the_spin_up_window_is_not_scored():
    """Days 5-34 of a record that starts on day 5; the first point is never scored."""
    sample = simulate_sample(_constant_simulator, _OneReactor(), 0, OBJECTIVES, spin_up_days=30.0)
    assert not sample.mask[:30].any()
    assert sample.mask[30:].all()
    assert simulate_sample(_constant_simulator, _OneReactor(), 0, OBJECTIVES).mask[1:].all()


def test_pooled_objectives_restrict_the_pooled_scores_only():
    # A huge-scale objective must not be able to swamp the pooled RMSE when it is
    # excluded, while still being reported per objective.
    targets = np.array([[10.0, 1000.0], [12.0, 1000.0]])
    predictions = np.array([[10.0, 1100.0], [12.0, 1100.0]])
    scores = pooled_metrics([_sample(predictions, targets)], pooled_objectives=["q_gasout"])
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["per_objective"]["S_TAN"]["rmse"] == pytest.approx(100.0)
    assert scores["pooled_objectives"] == ["q_gasout"]


def test_residuals_are_pooled_across_reactors():
    first = _sample([[10.0, 1.0]], [[10.0, 1.0]], label="a")
    second = _sample([[14.0, 1.0]], [[10.0, 1.0]], label="b")
    scores = pooled_metrics([first, second])
    assert scores["n_samples"] == 2
    assert scores["labels"] == ["a", "b"]
    # Two zero residuals and one of 4 over four scored points.
    assert scores["rmse"] == pytest.approx(np.sqrt((4.0**2) / 4))


def test_smape_stays_bounded_near_zero_targets():
    targets = np.array([[0.0, 0.0]])
    predictions = np.array([[5.0, 5.0]])
    per_objective = objective_metrics(
        predictions, targets, np.ones_like(targets, bool), OBJECTIVES
    )
    assert per_objective["q_gasout"]["smape"] <= 200.0


def test_paired_comparison_detects_a_consistent_improvement():
    hybrid = [0.10, 0.12, 0.09, 0.11]
    mechanistic = [0.20, 0.22, 0.19, 0.21]
    result = paired_comparison(hybrid, mechanistic, lower_is_better=True)
    assert result["wins"] == 4
    assert result["n_folds"] == 4
    assert result["mean_difference"] < 0
    assert result["sign_test_p"] == pytest.approx(0.0625)
    low, high = result["ci95_difference"]
    assert low <= result["mean_difference"] <= high


def test_the_difference_has_the_t_interval_over_the_folds():
    """A bootstrap over five folds has 126 distinct resamples and undercovers."""
    from scipy import stats

    hybrid = np.array([0.10, 0.12, 0.09, 0.11, 0.15])
    mechanistic = np.array([0.20, 0.22, 0.19, 0.21, 0.16])
    difference = hybrid - mechanistic
    half = stats.t.ppf(0.975, 4) * difference.std(ddof=1) / np.sqrt(5)
    result = paired_comparison(hybrid, mechanistic)
    assert result["ci95_difference"] == pytest.approx(
        (difference.mean() - half, difference.mean() + half))


def test_the_relative_change_is_the_mean_of_the_folds_changes_with_its_t_interval():
    from scipy import stats

    hybrid = np.array([0.10, 0.12, 0.09, 0.11, 0.15])
    mechanistic = np.array([0.20, 0.22, 0.19, 0.21, 0.16])
    change = (hybrid - mechanistic) / mechanistic * 100.0
    half = stats.t.ppf(0.975, 4) * change.std(ddof=1) / np.sqrt(5)
    result = paired_comparison(hybrid, mechanistic)
    assert result["mean_relative_change_percent"] == pytest.approx(change.mean())
    assert result["ci95_relative_change_percent"] == pytest.approx(
        (change.mean() - half, change.mean() + half))


def test_paired_comparison_respects_the_direction_of_the_metric():
    hybrid = [0.9, 0.8, 0.85]
    mechanistic = [0.5, 0.4, 0.45]
    higher_better = paired_comparison(hybrid, mechanistic, lower_is_better=False)
    lower_better = paired_comparison(hybrid, mechanistic, lower_is_better=True)
    assert higher_better["wins"] == 3
    assert lower_better["wins"] == 0


def test_paired_comparison_ignores_non_finite_pairs():
    result = paired_comparison([0.1, np.nan, 0.2], [0.3, 0.1, 0.4])
    assert result["n_folds"] == 2


def test_paired_comparison_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_comparison([0.1, 0.2], [0.3])
