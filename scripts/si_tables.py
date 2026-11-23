"""The tables of the Supporting Information that no other script prints.

    python scripts/si_tables.py
    python scripts/si_tables.py --latex

* calibrations: the parameters of every fold-wise calibration (MECH-CV and the two with
  Arrhenius-corrected hydrolysis, :data:`SETS`) for each dataset, those of ADM1-R4 and the
  kinetics that ADM1-R3 and ADM1-R2 add (``runs/<set>/fold_<k>/data/<campaign>``). A
  dataset none of whose reactors a fold holds out is calibrated on all its reactors there;
  the folds that share such a calibration share a row;
* mechanistic scores: every output of the mechanistic models calibrated on all reactors
  (MECH), pooled over the 15 reactors (``benchmark_numbers.py --json``, :data:`NUMBERS`);
* variants: the corrections of ADM1-R3 and ADM1-R2 compared as Table S4 compares those of
  ADM1-R4 (``development_table.py``): the three corrections of ADM1-R4 and two extensions,
  each a three-seed ensemble with the network hyperparameters of the first version;
* metrics, one table per variant: RMSE, MAE, bias, R^2, MAPE and sMAPE of every output of
  the final hybrids, the plain MECH-CV, the base model and MECH on held-out reactors, as means
  over the five folds (``manuscript_figures.fold_scores``). Variants without an evaluated
  final run are listed as missing.

The development rows' scores after operating changes come from ``change_windows_table.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import add_common_arguments  # noqa: E402
from development_table import _frames, _per_fold, compare  # noqa: E402
from manuscript_figures import DATASET, FINAL, PLAIN, UNTUNED, fold_scores, name_of  # noqa: E402

#: The fold-wise calibrations, by the names of Table 3, and the variants listed for each.
SETS = (("MECH-CV", "benchmarks-smoothstep", ("r4", "r3", "r2")),
        ("Arr.", "benchmarks-smoothstep-arrhenius", ("r4",)),
        ("Arr.$^{+}$", "benchmarks-smoothstep-fast", ("r4", "r3", "r2")))
CAMPAIGNS = ("rice-bran", "LaRA-temp", "LaRA-substrate")
#: The parameters each variant calibrates (ADM1-R3 and ADM1-R2 keep those of the variant
#: before them).
PARAMETERS = {"r4": ("f_ch", "f_pr", "f_li", "k_hyd_ch", "k_hyd_pr", "k_hyd_li"),
              "acids": ("k_m_ac", "K_ac", "k_m_va", "k_m_bu", "k_m_pro", "K_va", "K_bu", "K_pro")}
#: A value within this fraction of its search range from a bound is marked as at the bound.
AT_BOUND = 1e-3
NUMBERS = Path("plots/manuscript/benchmark_numbers.json")
OUTPUT_NAMES = {"q_gasout": "Biogas production", "ch4_out": "Methane production",
                "S_TAN": "TAN", "S_ac": "Acetic acid", "S_pro": "Propionic acid",
                "S_bu": "Butyric acid", "S_va": "Valeric acid", "pH": "pH",
                "S_h2o": "Water content"}
UNITS = {"q_gasout": r"NL\,d$^{-1}$", "ch4_out": r"NL\,d$^{-1}$", "pH": "--"}
#: The corrections of ADM1-R3 and ADM1-R2: (corrected parameters, run), the first the final
#: correction of ADM1-R4.
VARIANTS = {
    "r3": ((r"$f_{ch}, k_{hyd,ch}, f_{TAN,in}$", UNTUNED["r3"]),
           (r"$+\,k_{m,ac}$", "r3/fold_fast_acids_r3_no_ph_ensemble_s45-46-47"),
           (r"$+\,k_{m,ac}, f_{pr}, k_{dec,ac}$", "r3/fold_fast_pruned_r3_no_ph_ensemble_s45-46-47")),
    "r2": ((r"$f_{ch}, k_{hyd,ch}, f_{TAN,in}$", UNTUNED["r2"]),
           (r"$+\,k_{m,pro}, k_{m,ac}$", "r2/fold_fast_acids_r2_no_ph_ensemble_s45-46-47"),
           (r"$+\,k_{m,pro}, k_{m,ac}, f_{pr}, k_{dec,pro}, k_{dec,ac}$",
            "r2/fold_fast_pruned_r2_no_ph_ensemble_s45-46-47")),
}
VARIANT_OBJECTIVES = ("q_gasout", "ch4_out", "S_TAN", "S_ac", "S_pro")
#: The metrics table's columns, and the order of its outputs and models (their names in
#: ``fold_scores``).
METRICS = ("rmse", "mae", "bias", "r2", "mape", "smape")
METRIC_OUTPUTS = ("q_gasout", "ch4_out", "S_TAN", "S_ac", "S_pro", "pH", "S_h2o")
REFERENCES = ("MECH-CV", "Base", "UDE", "MECH")


def _significant(value: float, sign: bool = False) -> str:
    """Three significant digits, trailing zeros kept (2.00, not 2)."""
    return f"{value:{'+' if sign else ''}#.3g}".rstrip(".")


def _number(value: float, marked: bool = False) -> str:
    """Three significant digits, powers of ten for small values; ``marked`` adds an
    asterisk (at a bound)."""
    mark = "{}^{*}" if marked else ""
    if value == 0:
        return f"$0{mark}$"
    exponent = int(np.floor(np.log10(abs(value))))
    if exponent < -2:
        mantissa = value / 10 ** exponent
        return f"${mantissa:.2f} \\times 10^{{{exponent}}}{mark}$"
    return f"${_significant(value)}{mark}$" if marked else _significant(value)


def calibrations(runs: Path) -> dict[str, list[str]]:
    """LaTeX rows of the fold-wise parameters: ``r4`` and ``acids``."""
    rows: dict[str, list[str]] = {"r4": [], "acids": []}
    for set_name, directory, variants in SETS:
        print(f"\n{set_name} ({directory})")
        for group, names in PARAMETERS.items():
            if group == "acids" and "r3" not in variants:
                continue
            columns = len(names) + 2
            rows[group].append(rf"\multicolumn{{{columns}}}{{@{{}}l}}{{\textit{{{set_name}}}}} \\")
        for campaign in CAMPAIGNS:
            seen: dict[tuple, list[int]] = {}
            values_of: dict[tuple, dict] = {}
            for fold in range(1, 6):
                meta = json.loads((runs / directory / f"fold_{fold}" / f"{campaign}.json").read_text())
                held_out = tuple(meta.get("held_out", ()))
                stored = json.loads((runs / directory / f"fold_{fold}" / "data" / campaign
                                     / f"{campaign}-parameters.json").read_text())
                seen.setdefault(held_out, []).append(fold)
                values_of[held_out] = stored
            for held_out, folds in seen.items():
                stored = values_of[held_out]
                reactors = ", ".join(name_of(label) for label in held_out) or "none"
                label = f"{DATASET[campaign]} & {', '.join(map(str, folds))} ({reactors})"
                for group, names in PARAMETERS.items():
                    if group == "acids" and "r3" not in variants:
                        continue
                    cells = []
                    for name in names:
                        entry = stored[name]
                        value, low, high = entry["value"], entry["min"], entry["max"]
                        bounded = (np.isfinite(low) and np.isfinite(high)
                                   and min(value - low, high - value) <= AT_BOUND * (high - low))
                        cells.append(_number(value, bounded))
                    rows[group].append(f"{label} & " + " & ".join(cells) + r" \\")
                print(f"  {label.replace(' & ', ' fold ')}: "
                      + ", ".join(f"{name} {stored[name]['value']:.4g}" for name in PARAMETERS["r4"]))
    return rows


def mechanistic_scores(numbers: Path) -> list[str]:
    """LaTeX rows: every output of MECH pooled over all reactors."""
    data = json.loads(numbers.read_text(encoding="utf-8"))
    rows = []
    for key, model in (("r4", "ADM1-R4"), ("r3", "ADM1-R3"), ("r2", "ADM1-R2"), ("base", "ADM1")):
        scores = data[f"mech_pooled_{key}"]
        first = True
        for name in ("q_gasout", "ch4_out", "S_TAN", "S_ac", "S_pro", "S_bu", "S_va", "pH", "S_h2o"):
            if name not in scores:
                continue
            score = scores[name]
            unit = UNITS.get(name, r"g\,L$^{-1}$")
            rows.append(" & ".join([model if first else "", OUTPUT_NAMES[name], unit,
                                    _significant(score["rmse"]), f"${_significant(score['bias'], True)}$",
                                    f"${score['r2']:.2f}$", str(score["n"])]) + r" \\")
            first = False
        rows.append(r"\midrule")
    return rows[:-1]


def variants(runs: Path) -> list[str]:
    """LaTeX rows: the corrections of ADM1-R3 and ADM1-R2 against the plain MECH-CV."""
    rows = []
    for key, entries in VARIANTS.items():
        reference_folds, reference_objectives = _frames(runs / PLAIN[key])
        plain = {split: _per_fold(reference_folds, "MECH-CV", split) for split in ("train", "validation")}
        for position, (corrected, run) in enumerate(entries):
            directory = runs / run
            if not (directory / "results" / "cv_fold_metrics.csv").exists():
                print(f"{key} {run}: not evaluated yet")
                continue
            folds, objectives = _frames(directory)
            training = compare(_per_fold(folds, "UDE", "train"), plain["train"])
            held_out = compare(_per_fold(folds, "UDE", "validation"), plain["validation"])
            own = compare(_per_fold(folds, "UDE", "validation"), _per_fold(folds, "MECH-CV", "validation"))
            changes = {}
            for name in VARIANT_OBJECTIVES:
                values = _per_fold(objectives, "UDE", "validation", "rmse", name)
                if values.size:
                    changes[name] = compare(values, _per_fold(reference_objectives, "MECH-CV",
                                                              "validation", "rmse", name))
            print(f"{key} {run}: held out {held_out['change']:+.1f}% ({held_out['low']:+.1f} to "
                  f"{held_out['high']:+.1f}), {held_out['wins']}/{held_out['n']}; vs base model "
                  f"{own['change']:+.1f}%; " + ", ".join(f"{name} {change['change']:+.1f}% "
                                                         f"({change['wins']}/5)"
                                                         for name, change in changes.items()))
            model = {"r3": "ADM1-R3", "r2": "ADM1-R2"}[key] if position == 0 else ""
            rows.append(" & ".join([
                model, corrected, f"${training['change']:+.1f}$",
                f"${held_out['change']:+.1f}$ (${held_out['low']:+.1f}$, ${held_out['high']:+.1f}$)",
                f"{held_out['wins']}/{held_out['n']}",
                *(f"${changes[name]['change']:+.1f}$" if name in changes else "--"
                  for name in VARIANT_OBJECTIVES),
                f"${own['change']:+.1f}$"]) + r" \\")
        rows.append(r"\midrule")
    return rows[:-1]


def metrics(runs: Path) -> dict[str, list[str]]:
    """LaTeX rows, one table per variant: every metric of every output on held-out reactors,
    mean over the folds."""
    from adm1_package.experiments.registry import get_spec

    tables = {}
    for key in ("r4", "r3", "r2"):
        spec = get_spec(key)
        if not (runs / FINAL[key] / "results" / "cv_fold_metrics.csv").exists():
            print(f"{spec.display_name}: {FINAL[key]} not evaluated yet")
            continue
        _, objectives = fold_scores(runs, key)
        rows = []
        for name in METRIC_OUTPUTS:
            if name not in set(objectives["objective"]):
                continue
            if rows:
                rows.append(r"\addlinespace")
            unit = UNITS.get(name, r"g\,L$^{-1}$")
            for position, reference in enumerate(REFERENCES):
                frame = objectives[(objectives["objective"] == name) & (objectives["variant"] == reference)]
                means = {metric: float(frame[metric].mean()) for metric in METRICS}
                cells = [_significant(means["rmse"]), _significant(means["mae"]),
                         f"${_significant(means['bias'], True)}$",
                         f"${means['r2']:.2f}$", f"{means['mape']:.1f}", f"{means['smape']:.1f}"]
                short = OUTPUT_NAMES[name].removesuffix(" production")
                label = f"{short} ({unit})" if position == 0 else ""
                rows.append(" & ".join([label, reference, *cells]) + r" \\")
                print(f"{spec.display_name} {name} {reference}: "
                      + ", ".join(f"{metric} {value:.4g}" for metric, value in means.items()))
        tables[spec.display_name] = rows
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--latex", action="store_true", help="Also print the rows as LaTeX.")
    arguments = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    tables = {}
    calibrated = calibrations(arguments.runs)
    tables["fold-wise calibrations, ADM1-R4"] = calibrated["r4"]
    tables["fold-wise calibrations, ADM1-R3 and ADM1-R2"] = calibrated["acids"]
    tables["mechanistic scores"] = mechanistic_scores(project / NUMBERS)
    print()
    tables["corrections of ADM1-R3 and ADM1-R2"] = variants(arguments.runs)
    print()
    for model, rows in metrics(arguments.runs).items():
        tables[f"further metrics, {model}"] = rows
    if arguments.latex:
        for name, rows in tables.items():
            print(f"\n{name}\n" + "\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
