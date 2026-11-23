"""Aggregation and paired-comparison bookkeeping, on synthetic fold tables.

These exercise the parts of the evaluation that turn per-fold rows into the
manuscript's tables, without needing trained checkpoints.

A run is scored against two benchmarks. ``MECH`` is calibrated on every reactor of a
campaign, so on a fold's held-out reactors it is in-sample. ``MECH-CV`` is calibrated per
fold on the fold's training reactors only (scripts/calibrate/calibrate_folds.py): it
learned from the same reactors as the fold's hybrid, which makes it the fair comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adm1_package.experiments.evaluation import (
    EvaluationResult,
    aggregate_folds,
    compare_variants,
)
from adm1_package.experiments.reporting import (
    comparison_table,
    objective_rmse_table,
    overall_metrics_table,
    write_tables,
)


def _fold_frame(models=("ADM1-R4",), n_folds=4):
    rows = []
    rng = np.random.default_rng(0)
    for model in models:
        key = model.split("-")[-1].lower()
        for fold in range(1, n_folds + 1):
            for split in ("train", "validation"):
                for variant, base in (("UDE", 0.10), ("MECH", 0.20)):
                    rows.append(
                        {
                            "model": model,
                            "model_key": key,
                            "variant": variant,
                            "split": split,
                            "fold": fold,
                            "nrmse": base + 0.01 * rng.random(),
                            "rmse": base * 10,
                            "mape": base * 100,
                            "smape": base * 90,
                            "r2": 1 - base,
                            "aic": 100 * base,
                            "n_parameters": 960 if variant == "UDE" else 0,
                            "n_samples": 3,
                        }
                    )
    return pd.DataFrame(rows)


def _objective_frame(models=("ADM1-R4",), n_folds=4):
    rows = []
    for model in models:
        key = model.split("-")[-1].lower()
        for fold in range(1, n_folds + 1):
            for split in ("train", "validation"):
                for variant, base in (("UDE", 1.0), ("MECH", 2.0)):
                    for objective in ("q_gasout", "ch4_out", "S_TAN"):
                        rows.append(
                            {
                                "model": model,
                                "model_key": key,
                                "variant": variant,
                                "split": split,
                                "fold": fold,
                                "objective": objective,
                                "reported": True,
                                "rmse": base,
                                "mae": base * 0.8,
                                "bias": 0.0,
                                "r2": 0.5,
                                "mape": base * 10,
                                "smape": base * 9,
                            }
                        )
    return pd.DataFrame(rows)


def test_aggregation_reports_one_row_per_model_variant_split():
    aggregated = aggregate_folds(_fold_frame())
    assert len(aggregated) == 4  # 2 variants x 2 splits
    assert set(aggregated["n_folds"]) == {4}


def test_aggregation_carries_the_parameter_count():
    aggregated = aggregate_folds(_fold_frame())
    ude = aggregated[aggregated["variant"] == "UDE"]
    assert set(ude["n_parameters"]) == {960}


def test_standard_error_follows_from_the_standard_deviation():
    aggregated = aggregate_folds(_fold_frame())
    row = aggregated.iloc[0]
    assert row["nrmse_sem"] == pytest.approx(row["nrmse_std"] / np.sqrt(row["n_folds"]))


def test_comparison_is_produced_per_model_and_split():
    comparisons = compare_variants(_fold_frame(), _objective_frame())
    nrmse = comparisons[
        (comparisons["metric"] == "nrmse") & (comparisons["objective"] == "all")
    ]
    assert set(nrmse["split"]) == {"train", "validation"}
    # The synthetic hybrid is uniformly better.
    assert (nrmse["wins"] == 4).all()
    assert (nrmse["mean_difference"] < 0).all()


def test_per_objective_comparisons_are_included():
    comparisons = compare_variants(_fold_frame(), _objective_frame())
    per_objective = comparisons[comparisons["objective"] == "q_gasout"]
    assert not per_objective.empty
    assert per_objective["mean_difference"].to_numpy() == pytest.approx(-1.0)


def test_no_pooled_row_for_a_single_variant():
    comparisons = compare_variants(_fold_frame(), _objective_frame())
    assert "pooled" not in set(comparisons["model_key"])


def test_the_variants_are_not_pooled_into_one_test():
    """The variants share one split: pooled, the same folds would count as independent
    observations three times."""
    models = ("ADM1-R2", "ADM1-R3", "ADM1-R4")
    comparisons = compare_variants(_fold_frame(models), _objective_frame(models))
    assert "pooled" not in set(comparisons["model_key"])
    assert set(comparisons["model"]) == set(models)


def test_single_fold_yields_an_empty_but_well_formed_comparison_frame():
    comparisons = compare_variants(_fold_frame(n_folds=1), _objective_frame(n_folds=1))
    assert comparisons.empty
    for column in ("model", "split", "metric", "objective", "sign_test_p"):
        assert column in comparisons.columns


def test_tables_render_valid_latex():
    folds = _fold_frame(("ADM1-R2", "ADM1-R4"))
    objectives = _objective_frame(("ADM1-R2", "ADM1-R4"))
    aggregated = aggregate_folds(folds)
    objectives_aggregated = aggregate_folds(
        objectives,
        group=["model", "model_key", "variant", "split", "objective", "reported"],
        value_columns=["rmse", "mae", "bias", "r2", "mape", "smape"],
    )
    comparisons = compare_variants(folds, objectives)

    for latex in (
        objective_rmse_table(objectives_aggregated, "validation"),
        overall_metrics_table(aggregated, "validation"),
        comparison_table(comparisons, "validation"),
    ):
        assert latex.count(r"\begin{tabular}") == 1
        assert latex.count(r"\end{tabular}") == 1
        assert r"\toprule" in latex and r"\bottomrule" in latex
        # Every body row must have the same number of columns as the header.
        body = latex.split(r"\midrule")[1].split(r"\bottomrule")[0]
        widths = {line.count("&") for line in body.strip().splitlines() if line.strip()}
        header = latex.split(r"\toprule")[1].split(r"\midrule")[0]
        assert widths == {header.count("&")}


def test_comparison_table_refuses_a_split_it_has_no_rows_for():
    comparisons = compare_variants(_fold_frame(), _objective_frame())
    with pytest.raises(ValueError):
        comparison_table(comparisons, split="test")


# ---------------------------------------------------------------------------
# The per-fold benchmark beside the installed one
# ---------------------------------------------------------------------------
def _benchmark_frame(values: dict[str, list[float]]) -> pd.DataFrame:
    """R4's validation scores per fold, ``values`` per variant."""
    rows = []
    for variant, numbers in values.items():
        for fold, value in enumerate(numbers, start=1):
            rows.append({"model": "ADM1-R4", "model_key": "r4", "variant": variant,
                         "split": "validation", "fold": fold, "nrmse": value, "rmse": value,
                         "mape": value, "smape": value, "r2": 1 - value, "aic": 0.0,
                         "n_parameters": 100 if variant == "UDE" else 0, "n_samples": 3})
    return pd.DataFrame(rows)


THREE = {"UDE": [0.10, 0.12, 0.11], "MECH": [0.09, 0.10, 0.10], "MECH-CV": [0.12, 0.14, 0.13]}
INSTALLED_ONLY = {key: THREE[key] for key in ("UDE", "MECH")}


def test_the_hybrid_is_compared_with_every_benchmark_scored():
    comparisons = compare_variants(_benchmark_frame(THREE))
    rows = comparisons[(comparisons["metric"] == "nrmse")
                       & (comparisons["objective"] == "all")].set_index("benchmark")
    assert rows.loc["MECH", "mean_mechanistic"] == pytest.approx(0.29 / 3)
    assert rows.loc["MECH-CV", "mean_mechanistic"] == pytest.approx(0.13)
    assert rows.loc["MECH", "wins"] == 0 and rows.loc["MECH-CV", "wins"] == 3


def test_without_a_per_fold_benchmark_only_the_installed_one_is_compared():
    assert set(compare_variants(_benchmark_frame(INSTALLED_ONLY))["benchmark"]) == {"MECH"}


def test_each_benchmark_gets_its_own_paired_table():
    comparisons = compare_variants(_benchmark_frame(THREE))
    installed = comparison_table(comparisons, "validation")
    per_fold = comparison_table(comparisons, "validation", benchmark="MECH-CV")
    assert "0.0967" in installed and "0.1300" not in installed
    assert "0.1300" in per_fold and "MECH-CV" in per_fold


def test_results_written_before_the_per_fold_benchmark_still_render():
    comparisons = compare_variants(_benchmark_frame(INSTALLED_ONLY)).drop(columns="benchmark")
    assert "0.0967" in comparison_table(comparisons, "validation")


def test_the_overall_table_lists_the_per_fold_benchmark_between_the_others():
    table = overall_metrics_table(aggregate_folds(_benchmark_frame(THREE)), "validation")
    assert (table.index("ADM1-R4 MECH &") < table.index("ADM1-R4 MECH-CV &")
            < table.index("ADM1-R4 UDE &"))


def test_a_per_fold_comparison_table_is_written_when_there_is_one(tmp_path):
    frame = _benchmark_frame(THREE)
    result = EvaluationResult(
        folds=frame, objectives=pd.DataFrame(), aggregated=aggregate_folds(frame),
        objectives_aggregated=pd.DataFrame(columns=["split"]),
        comparisons=compare_variants(frame))
    names = {path.name for path in write_tables(result, tmp_path)}
    assert {"paired_comparison_validation.tex", "paired_comparison_cv_validation.tex"} <= names
