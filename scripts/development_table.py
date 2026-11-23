"""Table S4 of the Supporting Information: how the final hybrid ADM1-R4 came about, from
its ensembles.

    python scripts/development_table.py
    python scripts/development_table.py --latex

Every row is a three-seed ensemble of one experiment (:data:`ROWS`), evaluated like the
final models (``evaluate.py --ensemble``): held-out reactors started from their first 30
days, scores at the measurement times from day 30. Each row is compared fold by fold with

* MECH-CV, the plain fold-wise calibration, on the same reactors. The runs built on it
  (:data:`REFERENCE`) score it as their MECH-CV;
* its own anchor, the calibration it was trained on, i.e. the hybrid at zero network
  output, which its own evaluation calls MECH-CV.

Changes are means of the per-fold relative changes, in percent, with 95% t-intervals over
the five folds; "folds" counts those in which the hybrid had the lower held-out NRMSE.
Rows whose ensemble has not been evaluated yet are listed as missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_common_arguments  # noqa: E402

#: (label, anchor, corrected parameters, ensemble directory under runs/), in the order of
#: the table.
ROWS = (
    ("Fold cal.", r"$f_{ch}, f_{pr}, k_{hyd,ch}$, wide", "r4/fold_calibrated_ensemble_s45-46-47"),
    ("Fold cal.", r"$f_{ch}, k_{hyd,ch}$, $\times 2$", "r4/fold_calibrated_narrow_ensemble_s45-46-47"),
    ("Fold cal., Arr.", r"$f_{ch}, k_{hyd,ch}$, $\times 2$", "r4/fold_arrhenius_narrow_ensemble_s45-46-47"),
    ("Fold cal., Arr.$^{+}$", r"$f_{ch}, k_{hyd,ch}$, $\times 2$", "r4/fold_fast_narrow_ensemble_s45-46-47"),
    ("Fold cal., Arr.$^{+}$", r"$f_{ch}, k_{hyd,ch}, f_{TAN,in}$, $\times 2$",
     "r4/fold_fast_narrow_tan_ensemble_s45-46-47"),
    ("Fold cal., Arr.$^{+}$", r"as above, linear$^{a}$", "r4/fold_fast_narrow_tan_linear_ensemble_s45-46-47"),
)
#: The runs whose evaluation scores the plain fold-wise calibration as MECH-CV.
REFERENCE = "r4/fold_calibrated_narrow_ensemble_s45-46-47"
OBJECTIVES = (("q_gasout", "biogas"), ("ch4_out", "methane"), ("S_TAN", "TAN"))


def _frames(run: Path):
    import pandas as pd

    results = run / "results"
    return (pd.read_csv(results / "cv_fold_metrics.csv"),
            pd.read_csv(results / "cv_fold_objective_metrics.csv"))


def _per_fold(frame, variant: str, split: str, column: str = "nrmse", objective: str | None = None):
    rows = frame[(frame["variant"] == variant) & (frame["split"] == split)]
    if objective is not None:
        rows = rows[rows["objective"] == objective]
    return rows.sort_values("fold")[column].to_numpy(dtype=float)


def compare(row: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Per-fold relative change of ``row`` against ``reference``, in percent."""
    from adm1_package.experiments.metrics import paired_comparison

    result = paired_comparison(row, reference)
    low, high = result["ci95_relative_change_percent"]
    return {"change": result["mean_relative_change_percent"], "low": low, "high": high,
            "wins": result["wins"], "n": result["n_folds"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--latex", action="store_true", help="Also print the rows as LaTeX.")
    arguments = parser.parse_args()

    reference_folds, reference_objectives = _frames(arguments.runs / REFERENCE)
    plain = {split: _per_fold(reference_folds, "MECH-CV", split) for split in ("train", "validation")}
    plain_objectives = {name: _per_fold(reference_objectives, "MECH-CV", "validation", "rmse", name)
                        for name, _ in OBJECTIVES}

    lines = []
    for letter, (anchor, corrected, run) in zip("abcdefgh", ROWS):
        directory = arguments.runs / run
        if not (directory / "results" / "cv_fold_metrics.csv").exists():
            print(f"{letter}  {run}: not evaluated yet")
            continue
        folds, objectives = _frames(directory)
        training = compare(_per_fold(folds, "UDE", "train"), plain["train"])
        held_out = compare(_per_fold(folds, "UDE", "validation"), plain["validation"])
        own = compare(_per_fold(folds, "UDE", "validation"), _per_fold(folds, "MECH-CV", "validation"))
        changes = {label: compare(_per_fold(objectives, "UDE", "validation", "rmse", name),
                                  plain_objectives[name])
                   for name, label in OBJECTIVES}
        print(f"{letter}  {run}")
        print(f"   NRMSE training {training['change']:+.1f}%; held out {held_out['change']:+.1f}% "
              f"({held_out['low']:+.1f} to {held_out['high']:+.1f}), better in "
              f"{held_out['wins']}/{held_out['n']} folds; against its own anchor "
              f"{own['change']:+.1f}% ({own['low']:+.1f} to {own['high']:+.1f}), {own['wins']}/{own['n']}")
        print("   held-out RMSE " + ", ".join(f"{label} {change['change']:+.1f}% ({change['wins']}/5)"
                                              for label, change in changes.items()))
        lines.append(" & ".join([
            letter, anchor, corrected,
            f"${training['change']:+.1f}$",
            f"${held_out['change']:+.1f}$ (${held_out['low']:+.1f}$, ${held_out['high']:+.1f}$)",
            f"{held_out['wins']}/{held_out['n']}",
            *(f"${changes[label]['change']:+.1f}$" for _, label in OBJECTIVES),
            f"${own['change']:+.1f}$",
        ]) + r" \\")
    if arguments.latex:
        print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
