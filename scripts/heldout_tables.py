"""Tables 4 and 5 of the manuscript: the held-out accuracy of the final hybrid models.

    python scripts/heldout_tables.py
    python scripts/heldout_tables.py --latex

Every variant's final hybrid (:data:`manuscript_figures.FINAL`, a three-seed ensemble) is
scored fold by fold against the plain MECH-CV, its base model and MECH, as the evaluations
score them (``manuscript_figures.fold_scores``):

* Table 4: the RMSE of every objective and the NRMSE on held-out and training reactors,
  mean and standard deviation over the five folds, for MECH-CV and the hybrid;
* Table 5: the paired comparison of the held-out NRMSE with each reference: the mean
  difference with a 95% t-interval, the mean of the per-fold relative changes, the folds
  won, and the one-sided sign and Wilcoxon tests (``metrics.paired_comparison``).

The variants share one split, so their folds are not independent observations: the pooled
line counts the folds won over all variants and has no test. Variants whose final ensemble
has not been evaluated yet are listed as missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_common_arguments  # noqa: E402
from manuscript_figures import FINAL, PLAIN, fold_scores  # noqa: E402

#: Table 4's columns, in its order; a variant without the objective gets a dash.
OBJECTIVES = ("q_gasout", "ch4_out", "S_TAN", "S_ac", "S_pro", "pH")
#: Table 5's compared models, in its order: their names in ``fold_scores``.
REFERENCES = ("MECH-CV", "Base", "MECH")
VARIANTS = ("r4", "r3", "r2")


def _per_fold(frame, variant: str, column: str, objective: str | None = None) -> np.ndarray:
    rows = frame[frame["variant"] == variant]
    if objective is not None:
        rows = rows[rows["objective"] == objective]
    return rows.sort_values("fold")[column].to_numpy(dtype=float)


def _training_nrmse(run: Path, variant: str) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(run / "results" / "cv_fold_metrics.csv")
    rows = frame[(frame["split"] == "train") & (frame["variant"] == variant)]
    return rows.sort_values("fold")["nrmse"].to_numpy(dtype=float)


def _spread(values: np.ndarray, digits: int) -> str:
    return f"${values.mean():.{digits}f} \\pm {values.std(ddof=1):.{digits}f}$"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--latex", action="store_true", help="Also print the rows as LaTeX.")
    arguments = parser.parse_args()

    from adm1_package.experiments.metrics import paired_comparison
    from adm1_package.experiments.registry import get_spec

    table_4, table_5 = [], []
    pooled_wins, pooled_folds = 0, 0
    for key in VARIANTS:
        spec = get_spec(key)
        if not (arguments.runs / FINAL[key] / "results" / "cv_fold_metrics.csv").exists():
            print(f"{spec.display_name}: {FINAL[key]} not evaluated yet")
            continue
        folds, objectives = fold_scores(arguments.runs, key)
        hybrid = _per_fold(folds, "UDE", "nrmse")
        training = {"MECH-CV": _training_nrmse(arguments.runs / PLAIN[key], "MECH-CV"),
                    "UDE": _training_nrmse(arguments.runs / FINAL[key], "UDE")}

        print(f"\n{spec.display_name}: {FINAL[key]}")
        for variant in ("MECH-CV", "UDE"):
            cells = []
            for name in OBJECTIVES:
                values = _per_fold(objectives, variant, "rmse", name)
                cells.append(_spread(values, 2) if values.size else "--")
            cells.append(_spread(_per_fold(folds, variant, "nrmse"), 3))
            cells.append(_spread(training[variant], 3))
            r2 = _per_fold(folds, variant, "r2")
            print(f"  {variant:8s} " + "  ".join(f"{name} {cell}" for name, cell in
                                                  zip((*OBJECTIVES, "NRMSE", "NRMSE train"), cells))
                  + f"  R2 {r2.mean():.2f}")
            label = spec.display_name if variant == "MECH-CV" else ""
            table_4.append(f"{label:7s} & {variant:7s} & " + " & ".join(cells) + r" \\")

        for reference in REFERENCES:
            scores = _per_fold(folds, reference, "nrmse")
            result = paired_comparison(hybrid, scores)
            low, high = result["ci95_difference"]
            print(f"  vs {reference:8s} {result['mean_mechanistic']:.4f} -> {result['mean_hybrid']:.4f}: "
                  f"difference {result['mean_difference']:+.4f} ({low:+.4f}, {high:+.4f}), "
                  f"{result['mean_relative_change_percent']:+.1f}%, {result['wins']}/{result['n_folds']} "
                  f"folds, p sign {result['sign_test_p']:.3f}, Wilcoxon {result['wilcoxon_p']:.3f}")
            label = spec.display_name if reference == "MECH-CV" else ""
            table_5.append(" & ".join([
                f"{label:7s}", f"{reference:7s}", f"{result['mean_mechanistic']:.4f}",
                f"{result['mean_hybrid']:.4f}",
                f"${result['mean_difference']:+.4f}$ (${low:+.4f}$, ${high:+.4f}$)",
                f"${result['mean_relative_change_percent']:+.1f}\\%$",
                f"{result['wins']}/{result['n_folds']}",
                f"{result['sign_test_p']:.3f}", f"{result['wilcoxon_p']:.3f}"]) + r" \\")
            if reference == "MECH-CV":
                pooled_wins += result["wins"]
                pooled_folds += result["n_folds"]

    if pooled_folds:
        print(f"\nPooled: more accurate than MECH-CV in {pooled_wins} of {pooled_folds} folds "
              f"(no test: the variants share one split)")
    if arguments.latex:
        print("\nTable 4\n" + "\n".join(table_4) + "\n\nTable 5\n" + "\n".join(table_5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
