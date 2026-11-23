"""The numbers of the manuscript's Section 3.1 and Table 3: the mechanistic benchmarks.

    python scripts/benchmark_numbers.py
    python scripts/benchmark_numbers.py --json plots/manuscript/benchmark_numbers.json

Nothing here depends on a hybrid's training, so these numbers are final once the
calibrations are:

* MECH: every variant and the full ADM1 as calibrated on all reactors of a campaign (the
  installed calibration in ``data/``), simulated on every reactor;
* MECH-CV: each fold's calibration on its training reactors, simulated on the fold's
  held-out reactors, which start from their first 30 days (the held-out starts): the
  plain calibration and the two with Arrhenius-corrected hydrolysis (:data:`SETS`).

Every score counts the measurements from day 30, as ``evaluate.py`` does. A fold's score
pools its three held-out reactors; Table 3 gives the mean and the sample standard
deviation over the five folds. Before anything is printed, the fold scores are checked
against the evaluations of runs built on them, which score MECH and, as their own
MECH-CV, the calibration they are built on (:data:`CHECKED`). The Arrhenius calibration
without the bound and the full ADM1, which no run scores, take the same path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_common_arguments, configure_jax  # noqa: E402
from manuscript_figures import (DATASET, PLAIN, UNTUNED, forcing, name_of,  # noqa: E402
                                operating_changes, windows)

SPIN_UP_DAYS = 30.0
#: The per-fold calibrations of Table 3, by its row names, and the variants it lists them for.
SETS = {
    "MECH-CV": ("benchmarks-smoothstep", ("r4", "r3", "r2")),
    "MECH-CV, Arr.": ("benchmarks-smoothstep-arrhenius", ("r4",)),
    "MECH-CV, Arr.+": ("benchmarks-smoothstep-fast", ("r4", "r3", "r2")),
}
#: Evaluated runs that score a set as their own MECH-CV, and MECH beside it. The untuned
#: final configuration stands for the fast set: every variant has it on these anchors.
CHECKED = {"MECH-CV": PLAIN, "MECH-CV, Arr.+": UNTUNED}
VARIANTS = ("r4", "r3", "r2", "base")
#: Table 3's columns.
TABLE = ("q_gasout", "ch4_out", "S_TAN", "S_ac", "S_pro", "pH")
NAMES = {"q_gasout": "biogas", "ch4_out": "methane", "S_TAN": "TAN", "S_ac": "acetic acid",
         "S_pro": "propionic acid", "pH": "pH", "S_h2o": "water"}


@dataclass
class Simulated:
    """One simulated reactor: its scored sample and its time axis."""

    label: str
    times: np.ndarray
    sample: Any
    inputs: dict

    def series(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulated, measured and scored, for one objective."""
        column = self.sample.objectives.index(name)
        return (self.sample.predictions[:, column], self.sample.targets[:, column],
                self.sample.mask[:, column])


def simulate(spec, dataset, simulator, labels) -> dict[str, Simulated]:
    from adm1_package.experiments.metrics import simulate_sample

    result = {}
    for label in labels:
        index = list(dataset.labels).index(label)
        inputs, _ = dataset[index]
        sample = simulate_sample(simulator, dataset, index, spec.outputs, spin_up_days=SPIN_UP_DAYS)
        result[label] = Simulated(label, np.asarray(inputs["input"]["time"], dtype=float), sample,
                                  inputs)
    return result


def mech(key: str, data_dir: Path) -> dict[str, Simulated]:
    """MECH on every reactor, as evaluate.py scores it."""
    from adm1_package.experiments.calibrations import installed_forcing
    from adm1_package.experiments.registry import get_spec

    spec = get_spec(key)
    dataset = spec.dataset(data_dir, forcing=installed_forcing(data_dir))
    return simulate(spec, dataset, spec.mechanistic(initial_state="calibrated"), dataset.labels)


def mech_cv(key: str, where: Path, split) -> dict[str, Simulated]:
    """Each fold's calibration in ``where`` on the fold's held-out reactors."""
    from adm1_package.experiments.calibrations import benchmark_model, fold_benchmark_data
    from adm1_package.experiments.held_out_starts import held_out_view
    from adm1_package.experiments.registry import get_spec

    spec = get_spec(key)
    model = benchmark_model(where)
    simulator = spec.mechanistic(initial_state="calibrated", arrhenius=model["arrhenius"])
    result = {}
    for fold in range(1, split.k_folds + 1):
        dataset = spec.dataset(fold_benchmark_data(where, split, fold), forcing=model["forcing"])
        dataset, _ = held_out_view(dataset, key, where, split, fold, "fitted", SPIN_UP_DAYS)
        result.update(simulate(spec, dataset, simulator, split.validation_labels(fold)))
    return result


def fold_scores(key: str, simulated: dict[str, Simulated], split) -> list[dict[str, float]]:
    """Per fold, its held-out reactors pooled: the NRMSE and every output's RMSE."""
    from adm1_package.experiments.metrics import pooled_metrics
    from adm1_package.experiments.registry import get_spec

    spec = get_spec(key)
    rows = []
    for fold in range(1, split.k_folds + 1):
        scores = pooled_metrics([simulated[label].sample for label in split.validation_labels(fold)],
                                pooled_objectives=spec.reported_objectives)
        rows.append({"fold": fold, "nrmse": scores["nrmse"],
                     **{name: scores["per_objective"][name]["rmse"] for name in spec.outputs}})
    return rows


def check(rows: list[dict[str, float]], run: Path, variant: str) -> float:
    """The largest relative difference from ``run``'s evaluation; stops above 1e-6."""
    import pandas as pd

    results = run / "results"
    folds = pd.read_csv(results / "cv_fold_metrics.csv")
    folds = folds[(folds["variant"] == variant) & (folds["split"] == "validation")].set_index("fold")
    objectives = pd.read_csv(results / "cv_fold_objective_metrics.csv")
    objectives = objectives[(objectives["variant"] == variant) & (objectives["split"] == "validation")]
    worst = 0.0
    for row in rows:
        worst = max(worst, abs(row["nrmse"] / folds.loc[row["fold"], "nrmse"] - 1))
        for _, line in objectives[objectives["fold"] == row["fold"]].iterrows():
            if line["objective"] in row and np.isfinite(line["rmse"]) and line["rmse"] > 0:
                worst = max(worst, abs(row[line["objective"]] / line["rmse"] - 1))
    if worst > 1e-6:
        raise SystemExit(f"{variant} differs from the evaluation in {run} by {worst:.1e}")
    return worst


def summary(rows: list[dict[str, float]], name: str) -> tuple[float, float]:
    """Mean and sample standard deviation over the folds."""
    values = np.array([row[name] for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def pooled(simulated: dict[str, Simulated], labels, name: str) -> dict[str, float]:
    """RMSE, bias and R^2 of one output over the scored points of ``labels``, pooled."""
    sim, meas = [], []
    for label in labels:
        s, m, mask = simulated[label].series(name)
        sim.append(s[mask])
        meas.append(m[mask])
    sim, meas = np.concatenate(sim), np.concatenate(meas)
    residual = sim - meas
    total = float(np.sum((meas - meas.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(residual ** 2))), "bias": float(residual.mean()),
            "r2": float(1 - np.sum(residual ** 2) / total) if total > 0 else float("nan"),
            "n": int(meas.size)}


def gas_ratio(reactor: Simulated, where: np.ndarray) -> float:
    """Simulated over measured biogas on the scored points ``where`` selects."""
    sim, meas, mask = reactor.series("q_gasout")
    keep = mask & where
    return float(sim[keep].sum() / meas[keep].sum())


def gas_rmse(reactor: Simulated, where: np.ndarray) -> float:
    sim, meas, mask = reactor.series("q_gasout")
    keep = mask & where
    return float(np.sqrt(np.mean((sim[keep] - meas[keep]) ** 2))) if keep.any() else float("nan")


def in_spans(times: np.ndarray, spans) -> np.ndarray:
    inside = np.zeros_like(times, dtype=bool)
    for start, end in spans:
        inside |= (times >= start) & (times < end)
    return inside


def campaign_of(label: str) -> str:
    return label.rsplit("_", 1)[0]


def next_change(changes: dict[str, list[float]], day: float, last: float) -> float:
    """The first operating change after ``day``, else ``last``."""
    upcoming = [d for kind in ("temperature", "load", "feed", "feed_stop") for d in changes[kind]
                if d > day]
    return min(upcoming) if upcoming else last


def latest_report(campaign_dir: Path) -> dict:
    """The newest calibration report under a campaign's calibration-backup."""
    reports = sorted((campaign_dir / "calibration-backup").glob("*/report.json"))
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs",
                        help="Where the per-fold calibrations and the evaluated final runs are.")
    parser.add_argument("--json", type=Path, default=None, help="Also write every number here.")
    arguments = parser.parse_args()
    configure_jax()

    from adm1_package.experiments.registry import get_spec
    from adm1_package.experiments.splits import load_split

    split = load_split(arguments.split)
    record: dict[str, Any] = {}
    species = get_spec("r4").influent_species

    # -- simulations -------------------------------------------------------------------
    print("simulating MECH ...", flush=True)
    mechs = {key: mech(key, arguments.data_dir) for key in VARIANTS}
    print("simulating MECH-CV ...", flush=True)
    cvs = {(row, key): mech_cv(key, arguments.runs / where, split)
           for row, (where, keys) in SETS.items() for key in keys}

    # -- fold scores, checked against the evaluations ------------------------------------
    folds = {("MECH", key): fold_scores(key, mechs[key], split) for key in VARIANTS}
    folds.update({(row, key): fold_scores(key, simulated, split) for (row, key), simulated in cvs.items()})
    for row, runs in CHECKED.items():
        for key, run in runs.items():
            for variant, name in (("MECH", "MECH"), ("MECH-CV", row)):
                worst = check(folds[(name, key)], arguments.runs / run, variant)
                print(f"  {name:15s} {key}: matches {run} (largest relative difference {worst:.1e})")

    # -- Table 3 -------------------------------------------------------------------------
    print("\nTable 3 (held-out reactors, mean +- SD over folds)")
    table = {}
    for key in VARIANTS:
        spec = get_spec(key)
        for row in ("MECH", *SETS):
            if (row, key) not in folds:
                continue
            rows = folds[(row, key)]
            cells = {name: summary(rows, name) for name in TABLE if name in spec.outputs}
            cells["nrmse"] = summary(rows, "nrmse")
            table[f"{key} {row}"] = cells
            text = "  ".join(f"{NAMES.get(name, name)} {m:.2f}+-{s:.2f}" if name != "nrmse"
                             else f"NRMSE {m:.3f}+-{s:.3f}" for name, (m, s) in cells.items())
            print(f"  {spec.display_name:8s} {row:15s} {text}")
    record["table_3"] = table

    # -- MECH-CV against MECH ------------------------------------------------------------
    print("\nMECH-CV against MECH (held-out NRMSE per fold)")
    for key in ("r4", "r3", "r2"):
        cv = np.array([row["nrmse"] for row in folds[("MECH-CV", key)]])
        installed = np.array([row["nrmse"] for row in folds[("MECH", key)]])
        change = cv / installed - 1
        record[f"mech_cv_vs_mech_{key}"] = {"mean_change": float(change.mean()),
                                            "min": float(change.min()), "max": float(change.max()),
                                            "higher_in_folds": int((change > 0).sum())}
        print(f"  {key}: {100 * change.mean():+.1f}% on average (per fold {100 * change.min():+.1f} to "
              f"{100 * change.max():+.1f}%), higher in {(change > 0).sum()}/5 folds")
    for row in ("MECH-CV, Arr.", "MECH-CV, Arr.+"):
        cv = np.array([r["nrmse"] for r in folds[("MECH-CV", "r4")]])
        arrhenius = np.array([r["nrmse"] for r in folds[(row, "r4")]])
        change = arrhenius / cv - 1
        record[f"{row} vs MECH-CV r4"] = float(change.mean())
        print(f"  r4 {row} against MECH-CV: {100 * change.mean():+.1f}% on average, "
              f"better in {(change < 0).sum()}/5 folds")

    # -- MECH on every reactor, pooled -------------------------------------------------------
    print("\nMECH on all 15 reactors, pooled (RMSE, bias, R2)")
    labels = list(split.all_labels)
    for key in VARIANTS:
        spec = get_spec(key)
        scores = {name: pooled(mechs[key], labels, name) for name in spec.outputs}
        record[f"mech_pooled_{key}"] = scores
        print(f"  {spec.display_name:8s} " + "  ".join(
            f"{NAMES.get(name, name)} {s['rmse']:.3f} ({s['bias']:+.3f}, R2 {s['r2']:.2f})"
            for name, s in scores.items()))
    for campaign, letter in sorted(DATASET.items(), key=lambda item: item[1]):
        members = [label for label in labels if campaign_of(label) == campaign]
        line = []
        for key in VARIANTS:
            scores = {name: pooled(mechs[key], members, name) for name in ("q_gasout", "ch4_out", "S_TAN")}
            record[f"mech_dataset{letter}_{key}"] = scores
            line.append(f"{key} biogas {scores['q_gasout']['rmse']:.2f} TAN {scores['S_TAN']['rmse']:.2f}")
        print(f"  dataset {letter}: " + "; ".join(line))

    # -- the situations of Fig. 3 ------------------------------------------------------------
    print("\nMECH, simulated over measured biogas")
    for label in labels:
        reactor = mechs["r4"][label]
        changes = operating_changes(reactor.inputs, species)
        times = reactor.times
        whole = {key: gas_ratio(mechs[key][label], np.ones_like(times, dtype=bool)) for key in VARIANTS}
        line = ", ".join(f"{key} {value:.2f}" for key, value in whole.items())
        text = f"  {name_of(label)} ({label}) whole record: {line}"
        if changes["feed"]:
            day = changes["feed"][0]
            after = {key: gas_ratio(mechs[key][label], times >= day) for key in VARIANTS}
            record[f"substitution_{label}"] = {"day": day, **after}
            text += f"; from day {day:.0f}: " + ", ".join(f"{key} {value:.2f}" for key, value in after.items())
        record[f"whole_record_{label}"] = whole
        print(text)

    print("\nTemperature steps in dataset B (measured biogas; 21-day windows, merged)")
    for label in [label for label in labels if campaign_of(label) == "LaRA-temp"]:
        reactor = mechs["r4"][label]
        times = reactor.times
        changes = operating_changes(reactor.inputs, species)
        if not changes["temperature"]:
            print(f"  {name_of(label)} ({label}): no temperature step after day 40")
            continue
        _, series = forcing(reactor.inputs, species)
        temperature = series["T_op"]
        index = {round(day): int(np.argmin(np.abs(times[:len(temperature)] - day))) for day in changes["temperature"]}
        sizes = [float(temperature[i] - temperature[i - 1]) for i in index.values()]
        _, measured, mask = reactor.series("q_gasout")
        results = []
        for start, end in windows(changes["temperature"]):
            before = mask & (times >= start - 21) & (times < start)
            inside = mask & (times >= start) & (times < end)
            base = measured[before].mean()
            window_mean = measured[inside].mean() / base - 1
            # the highest three-day mean of the measured days in the window
            days = measured[inside]
            peak = max(days[i:i + 3].mean() for i in range(max(1, days.size - 2))) / base - 1
            results.append({"start": start, "end": end, "window_mean": float(window_mean),
                            "peak_3day": float(peak)})
        record[f"steps_{label}"] = {"days": changes["temperature"], "sizes": sizes, "windows": results}
        print(f"  {name_of(label)} ({label}): steps on days "
              f"{', '.join(f'{d:.0f}' for d in changes['temperature'])} "
              f"of {', '.join(f'{s:+.1f}' for s in sizes)} K; load changes "
              f"{', '.join(f'{d:.0f}' for d in changes['load']) or 'none'}")
        for result in results:
            print(f"    window {result['start']:.0f}-{result['end']:.0f}: mean "
                  f"{100 * result['window_mean']:+.0f}%, highest 3-day mean "
                  f"{100 * result['peak_3day']:+.0f}% against the 21 days before")

    # -- the Arrhenius calibrations ----------------------------------------------------------
    print("\nArrhenius calibrations in dataset B, held out: biogas RMSE in each 21-day window and "
          "from its end to the next operating change, and simulated/measured gas there")
    for label in [label for label in labels if campaign_of(label) == "LaRA-temp"]:
        plain = cvs[("MECH-CV", "r4")][label]
        times = plain.times
        last = float(times.max()) + 1
        changes = operating_changes(plain.inputs, species)
        entry: dict[str, Any] = {"spans": []}
        print(f"  {name_of(label)} ({label}):")
        for start, end in windows(changes["temperature"]):
            inside = (times >= start) & (times < end)
            stop = next_change(changes, end, last)
            after = (times >= end) & (times < stop)
            span = {"start": start, "end": end, "after_until": stop}
            text = []
            for row in SETS:
                reactor = cvs[(row, "r4")][label]
                span[row] = {"rmse_window": gas_rmse(reactor, inside), "rmse_after": gas_rmse(reactor, after),
                             "ratio_after": gas_ratio(reactor, after) if (after & reactor.series("q_gasout")[2]).any()
                             else float("nan")}
            for row in ("MECH-CV, Arr.", "MECH-CV, Arr.+"):
                within = span[row]["rmse_window"] / span["MECH-CV"]["rmse_window"] - 1
                outside = span[row]["rmse_after"] / span["MECH-CV"]["rmse_after"] - 1
                span[row].update({"window_change": float(within), "after_change": float(outside)})
                text.append(f"{row} window {100 * within:+.0f}%, after {100 * outside:+.0f}%")
            ratios = ", ".join(f"{row} {span[row]['ratio_after']:.2f}" for row in SETS)
            entry["spans"].append(span)
            print(f"    window {start:.0f}-{end:.0f}, after until {stop:.0f}: {'; '.join(text)}; "
                  f"simulated/measured after: {ratios}")
        if changes["load"]:
            low = times < changes["load"][0]
            ratios = {row: gas_ratio(cvs[(row, "r4")][label], low) for row in SETS}
            entry["low_load_ratio"] = ratios
            print("    simulated/measured before the first load change (day "
                  f"{changes['load'][0]:.0f}): " + ", ".join(f"{row} {value:.2f}" for row, value in ratios.items()))
        record[f"arrhenius_{label}"] = entry

    print("\nB6 (LaRA-temp F6) during the steps to 55 C (measured and MECH ADM1-R4 biogas, NL/d)")
    reactor = mechs["r4"]["LaRA-temp_F6"]
    sim, meas, mask = reactor.series("q_gasout")
    times = reactor.times
    periods = {"before the steps (60-85)": (60, 85), "days 100-185": (100, 185), "days 185-196": (185, 196)}
    f6 = {}
    for name, (start, end) in periods.items():
        keep = mask & (times >= start) & (times < end)
        f6[name] = {"measured": float(meas[keep].mean()), "simulated": float(sim[keep].mean()),
                    "measured_min": float(meas[keep].min()), "measured_max": float(meas[keep].max())}
        print(f"  {name}: measured {f6[name]['measured']:.2f} (range {f6[name]['measured_min']:.1f} to "
              f"{f6[name]['measured_max']:.1f}), MECH {f6[name]['simulated']:.2f}")
    record["f6_depression"] = f6

    print("\npH: mean deviation of MECH by period (pooled over the reactors)")
    for key in ("r3", "r2", "base"):
        line = []
        for start, end in ((30, 90), (90, 150), (150, 400)):
            sim, meas = [], []
            for label in labels:
                s, m, mask = mechs[key][label].series("pH")
                keep = mask & (mechs[key][label].times >= start) & (mechs[key][label].times < end)
                sim.append(s[keep])
                meas.append(m[keep])
            residual = np.concatenate(sim) - np.concatenate(meas)
            record[f"ph_bias_{key}_{start}_{end}"] = float(residual.mean())
            line.append(f"days {start}-{end}: {residual.mean():+.2f} (n {residual.size})")
        print(f"  {key}: " + "; ".join(line))

    print("\nCalibrated hydrolysis of dataset B per fold, and the R4 loss the fit ended at")
    for fold in range(1, split.k_folds + 1):
        line = [f"  fold {fold}:"]
        losses = {}
        for row in SETS:
            campaign_dir = arguments.runs / SETS[row][0] / f"fold_{fold}" / "data" / "LaRA-temp"
            parameters = json.loads((campaign_dir / "LaRA-temp-parameters.json").read_text(encoding="utf-8"))
            report = latest_report(campaign_dir)["variants"]["r4"]
            losses[row] = report["loss"]["hybrids"]["kept"]
            line.append(f"{row}: k_hyd_ch {parameters['k_hyd_ch']['value']:.3f}, f_ch "
                        f"{parameters['f_ch']['value']:.3f}, loss {losses[row]:.4f};")
        cost = losses["MECH-CV, Arr.+"] / losses["MECH-CV, Arr."] - 1
        record[f"loss_cost_fold{fold}"] = {"losses": losses, "arr_plus_vs_arr": cost}
        line.append(f"Arr.+ over Arr.: {100 * cost:+.0f}%")
        print(" ".join(line))

    print("\nInstalled calibration: parameters at a bound of their search range (within 0.1% "
          "of the range), or close to one (within 2%)")
    for campaign in DATASET:
        for key, report in latest_installed_reports(arguments.data_dir / campaign).items():
            at = []
            for name, entry in report.get("parameters", {}).items():
                low, high = entry["bounds"]
                value = entry["after"]
                span = high - low
                for bound, side in ((low, "lower"), (high, "upper")):
                    distance = abs(value - bound) / span
                    if distance <= 0.02:
                        at.append(f"{name} {'at' if distance <= 1e-3 else 'near'} its {side} bound "
                                  f"{bound:g} ({value:.4g})")
            record[f"bounds_{campaign}_{key}"] = at
            print(f"  {DATASET[campaign]} ({campaign}) {key}: {'; '.join(at) or 'none'}")

    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
        print(f"\nWrote {arguments.json}")
    return 0


def latest_installed_reports(campaign_dir: Path) -> dict[str, dict]:
    """Every variant's report of the calibration installed in ``campaign_dir``: the newest
    report that names the variant and was fitted to the installed forcing, among the
    backups of the calibration runs."""
    from adm1_package.experiments.calibrations import campaign_forcing

    installed = campaign_forcing(campaign_dir.parent, campaign_dir.name)
    reports: dict[str, dict] = {}
    for path in sorted((campaign_dir / "calibration-backup").glob("*/report.json")):
        for key, report in json.loads(path.read_text(encoding="utf-8")).get("variants", {}).items():
            if report.get("forcing") == installed:
                reports[key] = report
    return reports


if __name__ == "__main__":
    raise SystemExit(main())
