"""Prediction metrics and the paired statistics that go with cross-validation.

Design choices
--------------
* **Per-objective RMSE is kept**, so the per-objective tables can be recomputed
  under cross-validation.
* **The padded tail, the initial time point and the spin-up are excluded.**
  Targets are padded with edge values to a common length; scoring those is
  meaningless. The initial point is excluded because the dataset copies measured
  values into the initial state for some objectives, which would otherwise be a
  free match. The first ``spin_up_days`` follow from the initial state rather
  than from the model's dynamics, and the training loss ignores them too.
* **NRMSE follows the paper's definition.** The normalizing range has no floor:
  flooring it at 1 (``max(ptp, std, 1)``) would leave objectives whose observed
  range is below 1 -- most acid concentrations, in g/L -- effectively
  unnormalized.
* **Paired tests across folds.** Cross-validation gives one paired
  (UDE, mechanistic) observation per fold, which is what
  :func:`paired_comparison` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

__all__ = [
    "SamplePrediction",
    "simulate_sample",
    "score_solution",
    "objective_metrics",
    "pooled_metrics",
    "paired_comparison",
    "count_parameters",
]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class SamplePrediction:
    """One simulated reactor: predictions, targets and a validity mask."""

    label: str
    objectives: tuple[str, ...]
    predictions: np.ndarray  # (time, objective)
    targets: np.ndarray      # (time, objective)
    mask: np.ndarray         # (time, objective), True where the point is scored


def simulate_sample(
    simulator,
    dataset,
    index: int,
    objectives: Sequence[str],
    model=None,
    key=None,
    skip_initial: bool = True,
    spin_up_days: float = 0.0,
) -> SamplePrediction:
    """Run one sample through a mechanistic model or a UDE.

    ``model`` and ``key`` are ``None`` for the mechanistic variants and carry the
    trained network for the hybrid ones. Points in the first ``spin_up_days``
    days of the record are not scored.
    """
    inputs, targets = dataset[int(index)]

    if model is None:
        solution, _ = simulator(inputs["parameters"], inputs["initial_values"], inputs["input"])
    else:
        solution, _ = simulator(
            inputs["parameters"], inputs["initial_values"], inputs["input"], model, key
        )
    return score_solution(solution, inputs, targets, objectives,
                          skip_initial=skip_initial, spin_up_days=spin_up_days)


def score_solution(
    solution,
    inputs,
    targets,
    objectives: Sequence[str],
    skip_initial: bool = True,
    spin_up_days: float = 0.0,
) -> SamplePrediction:
    """Score a simulated reactor on the points :func:`simulate_sample` scores."""
    object_indices = targets["object_indices"]
    columns = [object_indices[name] for name in objectives]

    predictions = jnp.stack([solution[name] for name in objectives], axis=-1)
    measured = targets["values"][..., columns]
    mask = targets["indices"][..., columns] > 0
    if "measured" in targets:
        # only the records that hold a measurement, not the interpolation between them
        mask = mask & (targets["measured"][..., columns] > 0)

    predictions = np.asarray(predictions, dtype=float)
    measured = np.asarray(measured, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if skip_initial:
        mask = mask.copy()
        mask[0, :] = False
    if spin_up_days > 0:
        time = np.asarray(inputs["input"]["time"], dtype=float)
        mask = mask & (time >= time[0] + spin_up_days)[:, None]

    # Anything the solver could not produce is dropped rather than scored as 0.
    mask &= np.isfinite(predictions) & np.isfinite(measured)

    return SamplePrediction(
        label=targets["label"],
        objectives=tuple(objectives),
        predictions=predictions,
        targets=measured,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _masked(values: np.ndarray, mask: np.ndarray, column: int) -> np.ndarray:
    return values[mask[:, column], column]


def objective_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    objectives: Sequence[str],
    mape_epsilon: float = 2e-2,
) -> dict[str, dict[str, float]]:
    """Per-objective RMSE, MAE, bias, R^2, MAPE and sMAPE."""
    result: dict[str, dict[str, float]] = {}
    for column, name in enumerate(objectives):
        y_hat = _masked(predictions, mask, column)
        y = _masked(targets, mask, column)
        if y.size == 0:
            result[name] = {key: float("nan") for key in
                            ("rmse", "mae", "bias", "r2", "mape", "smape", "n")}
            continue

        residual = y_hat - y
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        denominator = np.abs(y_hat) + np.abs(y)

        result[name] = {
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "mape": float(np.mean(np.abs(residual / (y + mape_epsilon))) * 100.0),
            # Symmetric MAPE is bounded at 200% and does not blow up near zero,
            # which matters here because the acid measurements sit close to the
            # detection limit for long stretches.
            "smape": float(
                np.mean(np.where(denominator > 0, 2.0 * np.abs(residual) / denominator, 0.0)) * 100.0
            ),
            "n": float(y.size),
        }
    return result


def pooled_metrics(
    samples: Sequence[SamplePrediction],
    n_parameters: int = 0,
    mape_epsilon: float = 2e-2,
    pooled_objectives: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate a set of simulated reactors into one score set.

    Residuals are pooled over reactors, as the paper's NRMSE pools them.

    ``pooled_objectives`` restricts the *pooled* scores (RMSE, R^2, AIC) to a
    subset; per-objective scores always cover everything the model produces.
    This matters because the water content ``S_h2o`` sits near 960 g/L while the
    acid concentrations sit below 5 g/L, so a pooled RMSE over all objectives is
    simply a measurement of the water term. Range-normalized NRMSE is immune to
    this, but the raw pooled numbers are not.
    """
    if not samples:
        raise ValueError("no samples to score")

    objectives = samples[0].objectives
    for sample in samples:
        if sample.objectives != objectives:
            raise ValueError("cannot pool samples with different objectives")

    predictions = np.concatenate([s.predictions for s in samples], axis=0)
    targets = np.concatenate([s.targets for s in samples], axis=0)
    mask = np.concatenate([s.mask for s in samples], axis=0)

    per_objective = objective_metrics(predictions, targets, mask, objectives, mape_epsilon)

    if pooled_objectives is None:
        pooled_columns = list(range(len(objectives)))
        pooled_names = list(objectives)
    else:
        unknown = set(pooled_objectives) - set(objectives)
        if unknown:
            raise KeyError(f"pooled_objectives not simulated: {sorted(unknown)}")
        pooled_names = [name for name in objectives if name in set(pooled_objectives)]
        pooled_columns = [objectives.index(name) for name in pooled_names]

    predictions = predictions[:, pooled_columns]
    targets = targets[:, pooled_columns]
    mask = mask[:, pooled_columns]
    objectives = tuple(pooled_names)

    # NRMSE as the paper defines it: residuals scaled by the full observed range of each
    # objective, then averaged over objectives and time.
    normalized_mse = []
    for column in range(len(objectives)):
        y_hat = _masked(predictions, mask, column)
        y = _masked(targets, mask, column)
        if y.size == 0:
            continue
        span = float(y.max() - y.min())
        if span <= 0:
            # A constant objective cannot be range-normalized; fall back to its
            # scale so the term stays finite.
            span = max(float(np.abs(y).mean()), 1e-12)
        normalized_mse.append(float(np.mean(((y_hat - y) / span) ** 2)))

    nrmse = float(np.sqrt(np.mean(normalized_mse))) if normalized_mse else float("nan")

    valid_predictions = predictions[mask]
    valid_targets = targets[mask]
    residual = valid_predictions - valid_targets
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((valid_targets - valid_targets.mean()) ** 2))

    n_observations = int(valid_targets.size)
    # Gaussian-likelihood AIC with the RMSE as the scale estimate. Reported for
    # continuity with the earlier evaluation; it is dominated by the network
    # parameter count and should not be used to rank hybrid against mechanistic.
    log_likelihood = -n_observations * np.log(rmse + 1e-12)
    aic = 2 * n_parameters - 2 * log_likelihood

    return {
        "nrmse": nrmse,
        "rmse": rmse,
        "pooled_objectives": list(objectives),
        "mape": float(np.mean([per_objective[name]["mape"] for name in objectives])),
        "smape": float(np.mean([per_objective[name]["smape"] for name in objectives])),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "aic": float(aic),
        "n_parameters": int(n_parameters),
        "n_observations": n_observations,
        "n_samples": len(samples),
        "labels": [s.label for s in samples],
        "per_objective": per_objective,
    }


def count_parameters(model) -> int:
    """Number of trainable floats in an Equinox module."""
    import equinox as eqx
    import jax

    if model is None:
        return 0
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return int(sum(leaf.size for leaf in leaves))


# ---------------------------------------------------------------------------
# Paired statistics across folds
# ---------------------------------------------------------------------------
def paired_comparison(
    hybrid: Sequence[float],
    mechanistic: Sequence[float],
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Compare two models across folds on paired per-fold scores.

    Cross-validation yields one (hybrid, mechanistic) pair per fold, evaluated on
    exactly the same held-out reactors, so the pairing is genuine and a paired
    test is the right instrument. With the handful of folds available here the
    exact Wilcoxon signed-rank test and the sign test are the honest choices. The
    size of the difference, and the mean of the folds' relative changes, get
    95% intervals from the t distribution over the folds: a bootstrap over five
    folds has 126 distinct resamples and undercovers.

    With k folds the smallest attainable one-sided p value is ``2**-k``. The
    variants share one split, so pooling their folds would count the same folds as
    independent observations several times.
    """
    hybrid_scores = np.asarray(list(hybrid), dtype=float)
    mechanistic_scores = np.asarray(list(mechanistic), dtype=float)
    if hybrid_scores.shape != mechanistic_scores.shape:
        raise ValueError("paired comparison needs one score per model per fold")

    finite = np.isfinite(hybrid_scores) & np.isfinite(mechanistic_scores)
    hybrid_scores, mechanistic_scores = hybrid_scores[finite], mechanistic_scores[finite]
    n = hybrid_scores.size
    if n == 0:
        raise ValueError("no finite score pairs")

    differences = hybrid_scores - mechanistic_scores
    improvement = -differences if lower_is_better else differences
    n_wins = int(np.sum(improvement > 0))
    n_ties = int(np.sum(improvement == 0))

    # Exact one-sided sign test over the non-tied pairs.
    from scipy import stats

    n_effective = n - n_ties
    sign_p = (
        float(stats.binomtest(n_wins, n_effective, 0.5, alternative="greater").pvalue)
        if n_effective > 0
        else float("nan")
    )

    try:
        wilcoxon_p = float(
            stats.wilcoxon(
                hybrid_scores,
                mechanistic_scores,
                alternative="less" if lower_is_better else "greater",
                zero_method="zsplit",
            ).pvalue
        )
    except ValueError:
        wilcoxon_p = float("nan")

    def t_interval(values):
        values = values[np.isfinite(values)]
        if values.size < 2:
            return (float("nan"), float("nan"))
        half = stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)
        return (float(values.mean() - half), float(values.mean() + half))

    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(
            mechanistic_scores != 0, differences / mechanistic_scores * 100.0, np.nan
        )

    return {
        "n_folds": int(n),
        "mean_hybrid": float(hybrid_scores.mean()),
        "mean_mechanistic": float(mechanistic_scores.mean()),
        "mean_difference": float(differences.mean()),
        "mean_relative_change_percent": float(np.nanmean(relative)),
        "ci95_difference": t_interval(differences),
        "ci95_relative_change_percent": t_interval(relative),
        "wins": n_wins,
        "ties": n_ties,
        "sign_test_p": sign_p,
        "wilcoxon_p": wilcoxon_p,
    }
