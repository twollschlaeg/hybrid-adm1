"""Merge the evaluations of several variants into the manuscript's tables.

The manuscript lists ADM1-R2, R3 and R4 side by side, so the tables have to be
built from all three runs together:

    python scripts/combine_results.py \\
        --results runs/r2/<run>/results runs/r3/<run>/results runs/r4/<run>/results \\
        --output evaluations/model_comparison

Combining also makes the pooled paired test meaningful: with k folds a single
variant can at best reach a one-sided p of 2**-k, whereas the folds of all three
variants share one split and can be pooled into one study-level statement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_common_arguments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Results directories written by scripts/evaluate.py.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the merged tables.")
    arguments = parser.parse_args()

    import pandas as pd

    from adm1_package.experiments.evaluation import (
        EvaluationResult,
        aggregate_folds,
        compare_variants,
    )
    from adm1_package.experiments.reporting import write_tables

    def read_all(name: str) -> pd.DataFrame:
        frames = []
        for directory in arguments.results:
            path = directory / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; run scripts/evaluate.py first")
            frames.append(pd.read_csv(path))
        return pd.concat(frames, ignore_index=True)

    folds = read_all("cv_fold_metrics")
    objectives = read_all("cv_fold_objective_metrics")

    duplicated = folds.duplicated(subset=["model_key", "variant", "split", "fold"])
    if duplicated.any():
        raise SystemExit(
            "the same model, split and fold appears in more than one results directory; "
            "pass one directory per variant"
        )

    result = EvaluationResult(
        folds=folds,
        objectives=objectives,
        aggregated=aggregate_folds(folds),
        objectives_aggregated=aggregate_folds(
            objectives,
            group=["model", "model_key", "variant", "split", "objective", "reported"],
            value_columns=["rmse", "mae", "bias", "r2", "mape", "smape"],
        ),
        comparisons=compare_variants(folds, objectives),
    )

    written = result.write(arguments.output)
    tables = write_tables(result, arguments.output)

    variants = ", ".join(sorted(folds["model"].unique()))
    print(f"Combined {variants}")
    print("\nAggregated metrics:")
    print(
        result.aggregated[
            ["model", "variant", "split", "nrmse_mean", "nrmse_std", "r2_mean", "n_folds"]
        ].to_string(index=False)
    )

    pooled = result.comparisons
    pooled = pooled[(pooled["model_key"] == "pooled") & (pooled["metric"] == "nrmse")]
    if not pooled.empty:
        print("\nPooled paired test (all variants, NRMSE):")
        print(
            pooled[
                ["split", "benchmark", "mean_mechanistic", "mean_hybrid", "wins", "n_folds",
                 "sign_test_p", "wilcoxon_p"]
            ].to_string(index=False)
        )

    print("\nWrote:")
    for path in list(written.values()) + tables:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
