"""Draw the manuscript's Figures 3 to 6 from the evaluated runs.

    python scripts/manuscript_figures.py
    python scripts/manuscript_figures.py --figures 4 5 --format pdf png

Each figure is composed with the figure layer of :mod:`adm1_package.experiments.figures`:

* Figure 3 -- the mechanistic models calibrated on all reactors of a campaign (MECH) and
  the full ADM1, on the reactors they fail on (``inspection.simulate_mechanistic``);
* Figure 4 -- held-out NRMSE per fold of the final hybrids against MECH-CV and their
  base model, and the change per objective (the CSVs of ``evaluate.py`` only);
* Figure 5 -- held-out biogas after operating changes: MECH-CV, the base model and the
  final ADM1-R4;
* Figure 6 -- the corrections the final ADM1-R4 learned on held-out reactors.

MECH-CV is the plain fold-wise calibration (``runs/benchmarks-smoothstep``). The final
hybrids are built on another one (``runs/benchmarks-smoothstep-fast``: Arrhenius,
``k_hyd_ch >= 0.1``), which their own evaluation calls MECH-CV and the figures call the
base model: the hybrid at zero network output. The plain one is therefore read from
(Figure 4) or simulated by (Figure 5) the runs built on it, :data:`PLAIN`. The linear
baseline (:data:`LINEAR`) is printed beside Figure 4 where it has been evaluated.

Every reactor a hybrid draws is predicted by the fold that held it out. Each figure goes
to ``--output`` (default ``plots/manuscript``) as ``Figure_<n>``, and the numbers the
captions and the text need are printed beside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_common_arguments, configure_jax  # noqa: E402

#: The final hybrids: three-seed ensembles, retrained with seeds that took no part in any
#: selection and with every fold's own hyperparameters from the nested search, built on
#: :data:`FINAL_BENCHMARKS`. ADM1-R3 and ADM1-R2 also correct the growth of the degraders of
#: the acids they model.
FINAL = {
    "r4": "r4/fold_fast_narrow_tan_tuned_ensemble_s45-46-47",
    "r3": "r3/fold_fast_acids_r3_no_ph_tuned_ensemble_s45-46-47",
    "r2": "r2/fold_fast_acids_r2_no_ph_tuned_ensemble_s45-46-47",
}
#: The same configuration for every variant with the default hyperparameters: row e of the
#: development table (SI Table S4), and the runs that check the fast calibrations in
#: ``benchmark_numbers.py``.
UNTUNED = {
    "r4": "r4/fold_fast_narrow_tan_ensemble_s45-46-47",
    "r3": "r3/fold_fast_narrow_tan_no_ph_ensemble_s45-46-47",
    "r2": "r2/fold_fast_narrow_tan_no_ph_ensemble_s45-46-47",
}
FINAL_BENCHMARKS = "benchmarks-smoothstep-fast"
#: Runs built on the plain fold-wise calibration, whose MECH-CV is the paper's.
PLAIN = {
    "r4": "r4/fold_calibrated_narrow_ensemble_s45-46-47",
    "r3": "r3/fold_calibrated_narrow_no_ph_ensemble_s45-46-47",
    "r2": "r2/fold_calibrated_narrow_no_ph_ensemble_s45-46-47",
}
PLAIN_BENCHMARKS = "benchmarks-smoothstep"
#: The final ADM1-R4 with a linear map for each network, on the same anchors.
LINEAR = {"r4": "r4/fold_fast_narrow_tan_linear_ensemble_s45-46-47"}

#: The manuscript's letters for the campaigns, its datasets A to C.
DATASET = {"rice-bran": "A", "LaRA-temp": "B", "LaRA-substrate": "C"}
#: Rice bran's setups, each named after the last of its three replicates (R10 to R30: R30).
SETUP = {"R30": 1, "R60": 2, "R90": 3}

FIGURE_3 = ("LaRA-substrate_F4", "LaRA-temp_F5", "rice-bran_R60")
FIGURE_5 = ("LaRA-temp_F5", "LaRA-temp_F6", "LaRA-substrate_F4", "LaRA-substrate_F3")
FIGURE_6 = ("LaRA-temp_F5", "LaRA-substrate_F4", "rice-bran_R60")

#: Days after a temperature step that the change-window scores cover.
WINDOW_DAYS = 21.0
#: Figure 4(b), in this order; pH is scored but not fitted.
OBJECTIVES = ("q_gasout", "ch4_out", "S_TAN", "S_ac", "S_pro", "pH")
OBJECTIVE_NAMES = {"q_gasout": "Biogas", "ch4_out": "Methane", "S_TAN": "TAN",
                   "S_ac": "Acetic\nacid", "S_pro": "Propionic\nacid", "pH": "pH\n(not fitted)"}
BIOGAS_LABEL = r"Biogas [NL d$^{-1}$]"


def name_of(label: str) -> str:
    """The manuscript's name of a reactor: its dataset's letter and its number in it
    (``LaRA-temp_F5`` -> ``B5``, ``rice-bran_R60`` -> ``A2``)."""
    from adm1_package.experiments.splits import campaign_of

    campaign = campaign_of(label)
    reactor = label[len(campaign) + 1:]
    number = SETUP[reactor] if campaign == "rice-bran" else int(reactor.removeprefix("F"))
    return f"{DATASET[campaign]}{number}"


def title_of(label: str) -> str:
    """``LaRA-temp_F5`` -> ``Reactor B5``."""
    return f"Reactor {name_of(label)}"


# ---------------------------------------------------------------------------
# Operating changes, read off the forcing
# ---------------------------------------------------------------------------
def forcing(inputs, influent_species) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The forcing on the time grid before the padded tail: temperature in degrees
    Celsius, inflow and the influent species."""
    import diffrax as dx
    import jax

    signal = inputs["input"]
    times = np.asarray(signal["time"], dtype=float)
    length = int(np.argmax(times)) + 1
    interpolation = dx.CubicInterpolation(signal["time"], signal["coeffs"])
    values = np.asarray(jax.vmap(interpolation.evaluate)(signal["time"][:length]), dtype=float)
    names = ("T_op", "q_in", *influent_species)
    series = {name: values[:, column] for column, name in enumerate(names)}
    series["T_op"] = series["T_op"] - 273.15
    return times[:length], series


def operating_changes(inputs, influent_species, after: float = 40.0) -> dict[str, list[float]]:
    """Days on which the operation changed, after the start-up (``after``).

    ``temperature``: the temperature moved by more than 0.8 K from one day to the next
    (the controllers wander by a few tenths of a kelvin). ``load``: the inflow changed by
    more than 5 %, left empty for a reactor fed intermittently (rice bran), where every
    feed would count. ``feed``: the carbohydrate content of the influent changed by more than 5 % (in
    LaRA-substrate, the substitution of the corn silage and each later change of the
    co-substrate's share); ``feed_stop``: the feeding ended.
    """
    times, series = forcing(inputs, influent_species)
    running = times > after

    def steps(values, relative: bool, threshold: float) -> list[int]:
        change = np.abs(np.diff(values))
        if relative:
            change = change / np.maximum(np.abs(values[:-1]), 1e-9)
        return [index + 1 for index in np.flatnonzero(change > threshold)
                if times[index + 1] > after]

    temperature = [float(times[i]) for i in steps(series["T_op"], False, 0.8)]
    intermittent = np.mean(series["q_in"][running] < 1e-6) > 0.1
    load = [] if intermittent else [float(times[i]) for i in steps(series["q_in"], True, 0.05)
                                    if series["q_in"][i] > 1e-6]
    feed, feed_stop = [], []
    for index in steps(series["X_ch"], True, 0.05):
        (feed if series["X_ch"][index] > 1e-6 else feed_stop).append(float(times[index]))
    return {"temperature": temperature, "load": load, "feed": feed, "feed_stop": feed_stop}


def windows(days: list[float], length: float = WINDOW_DAYS) -> list[tuple[float, float]]:
    """The union of ``[day, day + length]`` over ``days``, as disjoint spans."""
    spans: list[list[float]] = []
    for day in sorted(days):
        if spans and day <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], day + length)
        else:
            spans.append([day, day + length])
    return [(start, end) for start, end in spans]


def mark_changes(ax, changes: dict[str, list[float]], loads: bool = True) -> None:
    """Vertical lines at the operating changes: the first temperature step and the
    first feed change solid, later ones dotted; load changes and the end of feeding
    thin."""
    style = dict(color="#555555", zorder=0.8)
    for kind, first_width in (("temperature", 0.9), ("feed", 0.9)):
        for position, day in enumerate(changes[kind]):
            ax.axvline(day, linewidth=first_width if position == 0 else 0.6,
                       linestyle="-" if position == 0 else (0, (1, 1.5)), **style)
    if loads:
        for day in changes["load"]:
            ax.axvline(day, linewidth=0.5, linestyle=(0, (1, 2.5)), alpha=0.7, **style)
    for day in changes["feed_stop"]:
        ax.axvline(day, linewidth=0.6, linestyle=(0, (4, 2)), **style)


def describe_changes(label: str, changes: dict[str, list[float]], temperature: np.ndarray
                     ) -> str:
    parts = []
    if changes["temperature"]:
        parts.append(f"T steps on days {_days(changes['temperature'])} "
                     f"({temperature.min():.1f} -> {temperature.max():.1f} C)")
    if changes["feed"]:
        parts.append(f"feed changes on days {_days(changes['feed'])}")
    if changes["load"]:
        parts.append(f"load changes on days {_days(changes['load'])}")
    if changes["feed_stop"]:
        parts.append(f"feeding ends on day {_days(changes['feed_stop'])}")
    return f"{name_of(label)} ({label}): " + ("; ".join(parts) or "no operating change")


def _days(days: list[float]) -> str:
    return ", ".join(f"{day:.0f}" for day in days)


def _length(times: np.ndarray) -> int:
    return int(np.argmax(times)) + 1


# ---------------------------------------------------------------------------
# Figure 3: the calibrated mechanistic models where they fail
# ---------------------------------------------------------------------------
def figure_3(labels, data_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from adm1_package.experiments import inspection
    from adm1_package.experiments.registry import get_spec
    from adm1_package.utils.plot_style import (LINES, SPIN_UP_LABEL, WIDTH_DOUBLE,
                                               figure_style, label_panels, shade_spin_up)
    from adm1_package.utils.plotting import target_series

    variants = ("r4", "r3", "r2", "base")
    species = get_spec("r4").influent_species
    simulations = {label: [inspection.simulate_mechanistic(key, label, data_dir=data_dir)
                           for key in variants] for label in labels}

    with figure_style():
        figure, axes = plt.subplots(1, len(labels), figsize=(WIDTH_DOUBLE, 2.3),
                                    layout="constrained", squeeze=False)
        for ax, label in zip(axes.ravel(), labels):
            runs = simulations[label]
            first = runs[0]
            length = first.length
            times = first.times[:length]
            changes = operating_changes(first.inputs, species)
            values, valid = target_series(first.targets, "q_gasout")
            valid = valid[:length]
            ax.plot(times[valid], values[:length][valid], label="Measured", **LINES["data_line"])
            for run in runs:
                ax.plot(times, run.solution["q_gasout"][:length], label=run.name, **run.style)
            shade_spin_up(ax, times, inspection.SPIN_UP_DAYS)
            mark_changes(ax, changes)
            ax.set_xlim(times[0], times[-1])
            ax.set_title(title_of(label))
            ax.set_xlabel("Time [d]")
            ax.set_ylabel(BIOGAS_LABEL)

            # What the text says about these reactors: simulated over measured gas.
            _, series = forcing(first.inputs, species)
            print(describe_changes(label, changes, series["T_op"]))
            scored = valid & (times >= times[0] + inspection.SPIN_UP_DAYS)
            scored[0] = False
            change = (changes["feed"] or changes["temperature"] or [times[0]])[0]
            for run in runs:
                simulated = run.solution["q_gasout"][:length]
                measured = values[:length]
                after = scored & (times >= change)
                rmse = float(np.sqrt(np.mean((simulated[scored] - measured[scored]) ** 2)))
                print(f"  {run.name:8s} simulated/measured gas: whole record "
                      f"{simulated[scored].sum() / measured[scored].sum():.2f}, from day "
                      f"{change:.0f} {simulated[after].sum() / measured[after].sum():.2f}; "
                      f"RMSE {rmse:.2f} NL/d")

        label_panels(axes.ravel())
        handles = [Line2D([], [], label="Measured", **LINES["data_line"])]
        handles += [Line2D([], [], label=f"{run.name} (MECH)" if run.name != "ADM1"
                           else "Full ADM1", **run.style) for run in runs]
        handles += [patch for patch in axes.ravel()[0].patches
                    if patch.get_label() == SPIN_UP_LABEL][:1]
        handles += [Line2D([], [], color="#555555", linewidth=0.9, label="Operating change")]
        figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    return figure


# ---------------------------------------------------------------------------
# Figure 4: fold-wise comparison
# ---------------------------------------------------------------------------
def _read(run: Path, name: str):
    import pandas as pd

    frame = pd.read_csv(run / "results" / name)
    return frame[frame["split"] == "validation"]


def fold_scores(runs_root: Path, key: str):
    """Held-out per-fold NRMSE and per-objective RMSE of the final hybrid (UDE), its
    anchor, the plain MECH-CV and MECH, one frame each. The final run's own "MECH-CV" is
    the calibration it is built on, called "Base" (the paper's base model)."""
    import pandas as pd

    final, plain = runs_root / FINAL[key], runs_root / PLAIN[key]
    tables = {}
    for name in ("cv_fold_metrics.csv", "cv_fold_objective_metrics.csv"):
        own, other = _read(final, name), _read(plain, name)
        # MECH is calibrated on every reactor, so both runs must score it alike: the
        # check that they were evaluated on the same data.
        index = ["fold"] if "objective" not in own else ["fold", "objective"]
        mech = own[own["variant"] == "MECH"].set_index(index)["rmse"].sort_index()
        mech_other = other[other["variant"] == "MECH"].set_index(index)["rmse"].sort_index()
        if not np.allclose(mech.to_numpy(), mech_other.loc[mech.index].to_numpy(), rtol=1e-9):
            raise ValueError(f"{final.name} and {plain.name} score MECH differently")
        parts = [own[own["variant"].isin(["UDE", "MECH"])],
                 own[own["variant"] == "MECH-CV"].assign(variant="Base"),
                 other[other["variant"] == "MECH-CV"]]
        linear = runs_root / LINEAR[key] if key in LINEAR else None
        if linear is not None and (linear / "results").exists():
            # on the same anchors, so only its hybrid is new
            frame = _read(linear, name)
            parts.append(frame[frame["variant"] == "UDE"].assign(variant="Linear"))
        tables[name] = pd.concat(parts)
    return tables["cv_fold_metrics.csv"], tables["cv_fold_objective_metrics.csv"]


def figure_4(runs_root: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from adm1_package.experiments.inspection import VARIANT_LINES
    from adm1_package.experiments.registry import get_spec
    from adm1_package.utils.plot_style import COLORS, WIDTH_DOUBLE, figure_style, label_panels

    keys = ("r4", "r3", "r2")
    scores = {key: fold_scores(runs_root, key) for key in keys}

    with figure_style():
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(WIDTH_DOUBLE, 2.7), layout="constrained",
            gridspec_kw={"width_ratios": [1.0, 1.25]})

        # (a) paired NRMSE per fold: MECH-CV, the anchor (the hybrid at zero output), UDE
        stages = ("MECH-CV", "Base", "UDE")
        ticks, tick_labels = [], []
        for group, key in enumerate(keys):
            folds = scores[key][0]
            nrmse = {variant: folds[folds["variant"] == variant].set_index("fold")["nrmse"]
                     for variant in (*stages, "MECH")}
            shared = nrmse["MECH-CV"].index.intersection(nrmse["Base"].index).intersection(
                nrmse["UDE"].index).sort_values()
            x0 = 3.6 * group
            xs = [x0, x0 + 1, x0 + 2]
            color = VARIANT_LINES[key]["color"]
            won = won_anchor = 0
            for fold in shared:
                values = [float(nrmse[stage][fold]) for stage in stages]
                won += values[2] < values[0]
                won_anchor += values[2] < values[1]
                left.plot(xs, values, marker="o", markersize=3.0, linewidth=0.9, alpha=0.85,
                          zorder=2, color=color,
                          linestyle="-" if values[2] < values[0] else (0, (3, 1.5)))
            # fold numbers beside the UDE ends, pushed apart where two ends nearly meet
            ends = sorted((float(nrmse["UDE"][fold]), fold) for fold in shared)
            placed: list[float] = []
            for end, fold in ends:
                spot = max(end, placed[-1] + 0.0065) if placed else end
                placed.append(spot)
                left.annotate(f"{fold}", xy=(x0 + 2, end), xytext=(x0 + 2.12, spot),
                              textcoords="data", va="center", fontsize=6, color="#666666")
            means = [float(nrmse[stage][shared].mean()) for stage in stages]
            left.plot(xs, means, color="black", linewidth=1.8, marker="o", markersize=4.2,
                      zorder=3)
            mech = float(nrmse["MECH"][shared].mean())
            left.plot([x0 - 0.25, x0 + 2.25], [mech, mech], color=COLORS["reference"],
                      linestyle=(0, (4, 2)), linewidth=1.0, zorder=1.5)
            left.text(x0 + 1.0, 1.0, f"{get_spec(key).display_name}\n"
                      f"UDE better in {won}/{len(shared)}",
                      transform=left.get_xaxis_transform(), ha="center", va="bottom",
                      fontsize=7)
            ticks += xs
            tick_labels += list(stages)
            print(f"{get_spec(key).display_name}: held-out NRMSE MECH-CV {means[0]:.4f}, "
                  f"anchor {means[1]:.4f}, UDE {means[2]:.4f}, MECH {mech:.4f}; UDE better "
                  f"than MECH-CV in {won}/{len(shared)} folds, than the anchor in "
                  f"{won_anchor}/{len(shared)}")
            if "Linear" in set(folds["variant"]):
                linear = folds[folds["variant"] == "Linear"].set_index("fold")["nrmse"]
                change = (linear[shared] - nrmse["Base"][shared]) / nrmse["Base"][shared] * 100
                print(f"  linear baseline: NRMSE {float(linear[shared].mean()):.4f}, "
                      f"{change.mean():+.1f}% vs the anchor per fold "
                      f"(better in {(change < 0).sum()}/{len(shared)})")
        left.set_xticks(ticks, tick_labels, rotation=35, ha="right", rotation_mode="anchor",
                        fontsize=6.5)
        left.set_xlim(-0.5, 3.6 * (len(keys) - 1) + 2.6)
        left.grid(axis="x", visible=False)
        left.set_ylabel("NRMSE, held out")

        # (b) change of the RMSE per objective, per fold
        width = 0.26
        positions = np.arange(len(OBJECTIVES))
        for offset, key in zip((-width, 0.0, width), keys):
            objectives = scores[key][1]
            pivot = objectives.pivot_table(index=["objective", "fold"], columns="variant",
                                           values="rmse")
            change = ((pivot["UDE"] - pivot["MECH-CV"]) / pivot["MECH-CV"] * 100.0).dropna()
            color = VARIANT_LINES[key]["color"]
            for position, objective in zip(positions, OBJECTIVES):
                if objective not in change.index.get_level_values("objective"):
                    continue
                values = change.loc[objective].to_numpy()
                right.bar(position + offset, values.mean(), width=width * 0.92, color=color,
                          alpha=0.55, linewidth=0, zorder=2)
                jitter = np.linspace(-0.06, 0.06, len(values))
                right.scatter(position + offset + jitter, values, s=7, color=color,
                              edgecolors="white", linewidths=0.3, zorder=3)
                print(f"  {get_spec(key).display_name} {objective}: {values.mean():+.1f}% "
                      f"(better in {(values < 0).sum()}/{len(values)} folds)")
        right.axhline(0.0, color="black", linewidth=0.7, zorder=2.5)
        right.set_xticks(positions, [OBJECTIVE_NAMES[objective] for objective in OBJECTIVES],
                         fontsize=6.5)
        right.grid(axis="x", visible=False)
        right.set_ylabel("Change in RMSE, UDE vs. MECH-CV [%]")
        right.text(1.0, 0.02, "negative: hybrid better", transform=right.transAxes,
                   ha="right", va="bottom", fontsize=6, color="#666666")

        label_panels([left, right], y=1.12)
        handles = [Line2D([], [], color="#666666", marker="o", markersize=3,
                          linewidth=0.9, label="Fold, UDE better"),
                   Line2D([], [], color="#666666", marker="o", markersize=3,
                          linewidth=0.9, linestyle=(0, (3, 1.5)), label="Fold, UDE worse"),
                   Line2D([], [], color="black", marker="o", markersize=4, linewidth=1.8,
                          label="Mean of folds"),
                   Line2D([], [], color=COLORS["reference"], linestyle=(0, (4, 2)),
                          linewidth=1.0, label="MECH (all reactors), mean")]
        handles += [Line2D([], [], color=VARIANT_LINES[key]["color"], marker="s",
                           linestyle="none", markersize=5, alpha=0.7,
                           label=get_spec(key).display_name) for key in keys]
        figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    return figure


# ---------------------------------------------------------------------------
# Figures 5 and 6: held-out reactors of the final ADM1-R4
# ---------------------------------------------------------------------------
class HeldOut:
    """Opens the folds of a run lazily and predicts held-out reactors with every member."""

    def __init__(self, key: str, run: Path, benchmarks: Path, data_dir: Path):
        from adm1_package.experiments.ensembles import member_runs
        from adm1_package.experiments.registry import get_spec
        from adm1_package.experiments.splits import load_run_split

        self.spec = get_spec(key)
        self.run = run
        self.members = member_runs(run, self.spec.key)
        self.split = load_run_split(run)
        self.benchmarks = benchmarks
        self.data_dir = data_dir
        self._opened: dict[int, list] = {}

    def opened(self, fold: int) -> list:
        from adm1_package.experiments.evaluation import open_fold

        if fold not in self._opened:
            self._opened[fold] = [open_fold(self.spec, member, self.split, fold, self.data_dir,
                                            benchmarks=self.benchmarks)
                                  for member in self.members]
        return self._opened[fold]

    def fold_of(self, label: str) -> int:
        from adm1_package.experiments.inspection import held_out_fold

        return held_out_fold(self.split, label)

    def members_of(self, label: str) -> list:
        """One :class:`FoldPrediction` per member, by the fold that held ``label`` out."""
        from adm1_package.experiments.predictions import predict_reactor

        opened = self.opened(self.fold_of(label))
        index = list(opened[0].dataset.labels).index(label)
        return [predict_reactor(member, index, "validation", keys=None) for member in opened]

    def mechanistic(self, label: str) -> tuple[np.ndarray, dict]:
        """The fold's mechanistic model alone (its anchor): times and solution."""
        opened = self.opened(self.fold_of(label))[0]
        index = list(opened.dataset.labels).index(label)
        inputs, _ = opened.dataset[index]
        solution, _ = opened.mechanistic(inputs["parameters"], inputs["initial_values"],
                                         inputs["input"])
        return (np.asarray(inputs["input"]["time"], dtype=float),
                {name: np.asarray(value, dtype=float) for name, value in solution.items()})

    def inputs(self, label: str):
        opened = self.opened(self.fold_of(label))[0]
        return opened.dataset[list(opened.dataset.labels).index(label)][0]


def _rmse(simulated, measured, mask) -> float:
    return float(np.sqrt(np.mean((simulated[mask] - measured[mask]) ** 2))) if mask.any() \
        else float("nan")


def figure_5(final: HeldOut, plain: HeldOut, labels):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from adm1_package.utils.plot_style import (COLORS, LINES, SPIN_UP_LABEL, WIDTH_DOUBLE,
                                               figure_style, label_panels, shade_spin_up)

    species = final.spec.influent_species
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    with figure_style():
        figure, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_DOUBLE, 1.95 * nrows),
                                    layout="constrained", squeeze=False)
        for ax, label in zip(axes.ravel(), labels):
            members = final.members_of(label)
            first = members[0]
            length = _length(first.times)
            times = first.times[:length]
            plain_times, plain_solution = plain.mechanistic(label)
            if not np.allclose(plain_times[:length], times):
                raise ValueError(f"{label}: the two runs' time grids differ")
            mech_cv = plain_solution["q_gasout"][:length]
            # the hybrid at zero output: the calibration the final hybrid is built on
            anchor = final.mechanistic(label)[1]["q_gasout"][:length]
            seeds = np.stack([member.hybrid["q_gasout"][:length] for member in members])
            hybrid = seeds.mean(axis=0)
            values, valid = first.measured["q_gasout"]
            values, valid = values[:length], valid[:length]

            inputs = final.inputs(label)
            changes = operating_changes(inputs, species)
            spans = windows(changes["temperature"])
            for start, end in spans:
                ax.axvspan(start, min(end, times[-1]), color=COLORS["ude"], alpha=0.10,
                           linewidth=0, zorder=0.5, label="21 days after a temperature step")
            mark_changes(ax, changes)
            ax.plot(times[valid], values[valid], label="Measured", **LINES["data_line"])
            ax.plot(times, mech_cv, label="MECH-CV", **LINES["mech"])
            ax.plot(times, anchor, label="Base model", **LINES["anchor"])
            ax.fill_between(times, seeds.min(axis=0), seeds.max(axis=0), color=COLORS["ude"],
                            alpha=0.22, linewidth=0, zorder=2.1)
            ax.plot(times, hybrid, label="UDE", **LINES["ude"])
            shade_spin_up(ax, times, first.spin_up_days)
            ax.set_xlim(times[0], times[-1])
            ax.set_title(f"{title_of(label)}, fold {first.fold}")
            ax.set_xlabel("Time [d]")
            ax.set_ylabel(BIOGAS_LABEL)

            _, series = forcing(inputs, species)
            print(describe_changes(label, changes, series["T_op"]))
            scored = valid & (times >= times[0] + first.spin_up_days)
            scored[0] = False
            print(f"  RMSE whole record: MECH-CV {_rmse(mech_cv, values, scored):.2f}, "
                  f"anchor {_rmse(anchor, values, scored):.2f}, "
                  f"UDE {_rmse(hybrid, values, scored):.2f} NL/d")
            for start, end in spans:
                inside = scored & (times >= start) & (times <= end)
                mech, anc, ude = (_rmse(curve, values, inside) for curve in (mech_cv, anchor, hybrid))
                print(f"  days {start:.0f}-{end:.0f}: MECH-CV {mech:.2f}, anchor {anc:.2f}, "
                      f"UDE {ude:.2f} NL/d (UDE {(ude - mech) / mech * 100:+.0f}% vs MECH-CV, "
                      f"{(ude - anc) / anc * 100:+.0f}% vs the anchor)")
            if changes["feed"]:
                after = scored & (times >= changes["feed"][0])
                ratios = {name: curve[after].sum() / values[after].sum()
                          for name, curve in (("MECH-CV", mech_cv), ("anchor", anchor), ("UDE", hybrid))}
                rmses = {name: _rmse(curve, values, after)
                         for name, curve in (("MECH-CV", mech_cv), ("anchor", anchor), ("UDE", hybrid))}
                print(f"  from day {changes['feed'][0]:.0f}: simulated/measured gas "
                      + ", ".join(f"{name} {ratio:.2f}" for name, ratio in ratios.items())
                      + "; RMSE " + ", ".join(f"{name} {rmse:.2f}" for name, rmse in rmses.items()))

        for ax in axes.ravel()[len(labels):]:
            ax.axis("off")
        label_panels(axes.ravel())
        handles = [Line2D([], [], label="Measured", **LINES["data_line"]),
                   Line2D([], [], label="MECH-CV", **LINES["mech"]),
                   Line2D([], [], label="Base model (UDE at zero output)", **LINES["anchor"]),
                   Line2D([], [], label="UDE, mean of three seeds", **LINES["ude"]),
                   Patch(facecolor=COLORS["ude"], alpha=0.22, label="UDE, range of seeds"),
                   Patch(facecolor=COLORS["ude"], alpha=0.10,
                         label="21 days after a temperature step"),
                   Line2D([], [], color="#555555", linewidth=0.9, label="Operating change")]
        handles += [patch for patch in axes.ravel()[0].patches
                    if patch.get_label() == SPIN_UP_LABEL][:1]
        figure.legend(handles=handles, loc="outside lower center", ncols=4)
    return figure


#: Figure 6's columns, the corrections of the final ADM1-R4: (name, title, unit). Rates are
#: drawn at the 35 C their anchors refer to.
CORRECTIONS = (
    ("f_ch", r"Degradability $f_{ch}$", "[-]"),
    ("k_hyd_ch", r"Hydrolysis $k_{hyd,ch}$ at 35 °C", r"[d$^{-1}$]"),
    ("f_TAN_in", r"Influent TAN factor $f_{TAN,in}$", "[-]"),
)


def figure_6(final: HeldOut, labels, columns=CORRECTIONS, width: float | None = None,
             row_height: float = 1.5):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from adm1_package.utils.corrections import (FRACTION_BAND, activation_energies,
                                                arrhenius_factor)
    from adm1_package.utils.plot_style import (COLORS, LINES, WIDTH_DOUBLE, figure_style,
                                               shade_spin_up)

    spec = final.spec
    bound = float(np.log(2.0))
    energies = activation_energies(spec.key)
    columns = [(name, f"({letter}) {title}", unit)
               for letter, (name, title, unit) in zip("abcdefgh", columns)]

    def sigmoid(value):
        return 1.0 / (1.0 + np.exp(-value))

    def admissible(name: str, anchor: float) -> tuple[float, float]:
        """The range the bounded link can reach from ``anchor`` (``bound`` = ln 2)."""
        if name == "f_ch":
            low, high = FRACTION_BAND
            clipped = min(max(anchor, low), high)
            logit = np.log(clipped) - np.log1p(-clipped)
            offset = anchor - clipped
            return sigmoid(logit - bound) + offset, sigmoid(logit + bound) + offset
        return anchor / 2.0, anchor * 2.0

    with figure_style():
        figure, axes = plt.subplots(len(labels), len(columns),
                                    figsize=(width or WIDTH_DOUBLE, row_height * len(labels) + 0.45),
                                    layout="constrained", squeeze=False)
        for row, label in enumerate(labels):
            members = final.members_of(label)
            first = members[0]
            length = _length(first.times)
            times = first.times[:length]
            changes = operating_changes(final.inputs(label), spec.influent_species)
            temperature = first.hybrid["T_op"][:length]
            print(f"{name_of(label)} ({label}, fold {first.fold}):")
            for column, (name, title, unit) in enumerate(columns):
                ax = axes[row, column]
                if name in spec.rate_names:
                    curves = np.stack([member.hybrid["dynamic_rates"][:length,
                                                                      spec.rate_names.index(name)]
                                       for member in members])
                    # the rate integrated at T_op, back at the 35 C its anchor refers to
                    if name in energies:
                        curves = curves / np.asarray(arrhenius_factor(temperature, energies[name]))
                elif name in spec.factor_names:
                    curves = np.stack([member.hybrid["nutrient_factors"][:length,
                                                                         spec.factor_names.index(name)]
                                       for member in members])
                else:
                    curves = np.stack([member.hybrid[name][:length] for member in members])
                anchor = float(first.parameters.get(name, 1.0))
                low, high = admissible(name, anchor)
                ax.axhspan(low, high, color="#000000", alpha=0.05, linewidth=0, zorder=0.3)
                for edge in (low, high):
                    ax.axhline(edge, color="#999999", linewidth=0.5, zorder=0.4)
                ax.axhline(anchor, **LINES["reference"])
                for curve in curves:
                    ax.plot(times, curve, color=COLORS["ude"], linewidth=0.5, alpha=0.55,
                            zorder=2.0)
                ax.plot(times, curves.mean(axis=0), color=COLORS["ude"], linewidth=1.4,
                        zorder=2.2)
                mark_changes(ax, changes)
                shade_spin_up(ax, times, first.spin_up_days)
                ax.set_xlim(times[0], times[-1])
                # the whole admissible range, so it shows how much of it is used
                margin = 0.08 * (high - low)
                ax.set_ylim(min(low, curves.min()) - margin, max(high, curves.max()) + margin)
                if row == 0:
                    ax.set_title(title)
                if row == len(labels) - 1:
                    ax.set_xlabel("Time [d]")
                ax.set_ylabel(f"{name_of(label)}\n{unit}" if column == 0 else unit)

                mean = curves.mean(axis=0)
                scored = times >= times[0] + first.spin_up_days
                print(f"  {name}: anchor {anchor:.4g}, admissible {low:.4g}-{high:.4g}; mean "
                      f"of seeds after spin-up {mean[scored].min():.4g}-{mean[scored].max():.4g}"
                      f"; seed spread up to {np.ptp(curves[:, scored], axis=0).max():.3g}")

        figure.legend(
            handles=[Line2D([], [], color=COLORS["ude"], linewidth=1.4,
                            label="UDE, mean of three seeds"),
                     Line2D([], [], color=COLORS["ude"], linewidth=0.5, alpha=0.55,
                            label="Individual seeds"),
                     Line2D([], [], label="Calibrated anchor", **LINES["reference"]),
                     Patch(facecolor="#000000", alpha=0.05, label="Admissible range (factor 2)"),
                     Line2D([], [], color="#555555", linewidth=0.9, label="Operating change")],
            loc="outside lower center", ncols=5)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--figures", nargs="+", type=int, default=[3, 4, 5, 6],
                        choices=[3, 4, 5, 6])
    parser.add_argument("--output", type=Path, default=None,
                        help="Figure directory (default: plots/manuscript).")
    parser.add_argument("--format", nargs="+", default=["pdf"],
                        choices=["pdf", "png", "svg", "eps"])
    arguments = parser.parse_args()

    configure_jax()

    from adm1_package.experiments.config import PROJECT_ROOT
    from adm1_package.utils.plot_style import save_figure

    runs_root = arguments.run_dir
    output = arguments.output or PROJECT_ROOT / "plots" / "manuscript"
    data_dir = arguments.data_dir

    final = plain = None
    if {5, 6} & set(arguments.figures):
        final = HeldOut("r4", runs_root / FINAL["r4"], runs_root / FINAL_BENCHMARKS, data_dir)
        plain = HeldOut("r4", runs_root / PLAIN["r4"], runs_root / PLAIN_BENCHMARKS, data_dir)
        # the plain run's mechanistic model only, so one member suffices
        plain.members = plain.members[:1]

    drawers = {
        3: lambda: figure_3(FIGURE_3, data_dir),
        4: lambda: figure_4(runs_root),
        5: lambda: figure_5(final, plain, FIGURE_5),
        6: lambda: figure_6(final, FIGURE_6),
    }
    for number in arguments.figures:
        print(f"\n=== Figure {number}")
        for path in save_figure(drawers[number](), output / f"Figure_{number}",
                                formats=arguments.format):
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
