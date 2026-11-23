"""Draw the figures of the Supporting Information from the calibrations and the evaluated runs.

    python scripts/si_figures.py
    python scripts/si_figures.py --kinds heldout corrections --models r4

One figure per dataset (A to C), with one row per reactor:

* ``mechanistic`` -- the mechanistic models calibrated on all reactors of the dataset
  (MECH) and the full ADM1: biogas, TAN, acetic and propionic acid and pH
  (``inspection.simulate_mechanistic``, as Figure 3);
* ``heldout`` -- every reactor predicted by the fold that held it out: MECH-CV, the anchor
  and the final hybrid of each of ``--models`` (``manuscript_figures.FINAL``), for every
  output the variant fits and the pH, as Figure 5;
* ``corrections`` -- the corrections the final hybrid learned for those reactors, as
  Figure 6.

The figures are drawn at the text width of the Supporting Information (:data:`WIDTH`) and go
to ``--output`` (default ``plots/si``) as ``S_<kind>[_<model>]_<dataset>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_common_arguments, configure_jax  # noqa: E402
from manuscript_figures import (CORRECTIONS, DATASET, FINAL, FINAL_BENCHMARKS,  # noqa: E402
                                PLAIN, PLAIN_BENCHMARKS, HeldOut, _length, figure_6,
                                mark_changes, name_of, operating_changes)

#: The text width of the Supporting Information, in inches (160 mm).
WIDTH = 160 / 25.4
#: The reactors of each campaign, in the order of the manuscript's Table 1.
REACTORS = {
    "rice-bran": ("rice-bran_R30", "rice-bran_R60", "rice-bran_R90"),
    "LaRA-temp": tuple(f"LaRA-temp_F{number}" for number in range(1, 7)),
    "LaRA-substrate": tuple(f"LaRA-substrate_F{number}" for number in range(1, 7)),
}
TITLES = {"q_gasout": r"Biogas [NL d$^{-1}$]", "ch4_out": r"Methane [NL d$^{-1}$]",
          "S_TAN": r"TAN [g L$^{-1}$]", "S_ac": r"Acetic acid [g L$^{-1}$]",
          "S_pro": r"Propionic acid [g L$^{-1}$]", "pH": "pH [-]"}
#: The outputs drawn for the mechanistic models.
MECHANISTIC_OUTPUTS = ("q_gasout", "S_TAN", "S_ac", "S_pro", "pH")
#: The outputs drawn for each variant's held-out predictions: those it fits, and the pH.
HELD_OUT_OUTPUTS = {"r4": ("q_gasout", "ch4_out", "S_TAN"),
                    "r3": ("q_gasout", "ch4_out", "S_TAN", "S_ac", "pH"),
                    "r2": ("q_gasout", "ch4_out", "S_TAN", "S_ac", "S_pro", "pH")}
#: The corrections each variant's final hybrid learns (Section 2.4.1 of the manuscript). Four
#: or five columns need their titles on two lines; R4's are Figure 6's.
WRAPPED_CORRECTIONS = (
    ("f_ch", "Degradability\n$f_{ch}$", "[-]"),
    ("k_hyd_ch", "Hydrolysis $k_{hyd,ch}$\nat 35 °C", r"[d$^{-1}$]"),
    ("f_TAN_in", "Influent TAN\nfactor $f_{TAN,in}$", "[-]"),
)
VARIANT_CORRECTIONS = {
    "r4": CORRECTIONS,
    "r3": (*WRAPPED_CORRECTIONS, ("k_m_ac", "Growth $k_{m,ac}$\nat 35 °C", r"[d$^{-1}$]")),
    "r2": (*WRAPPED_CORRECTIONS, ("k_m_pro", "Growth $k_{m,pro}$\nat 35 °C", r"[d$^{-1}$]"),
           ("k_m_ac", "Growth $k_{m,ac}$\nat 35 °C", r"[d$^{-1}$]")),
}
#: Height of one row of panels, and of the legend below them, in inches.
ROW_HEIGHT = 1.2
LEGEND_HEIGHT = 0.45


def _measured(ax, times, values, valid, name: str) -> None:
    """The gas production, read daily, as a line; the weekly analyses as markers."""
    from adm1_package.utils.plot_style import LINES

    style = LINES["data_line"] if name in ("q_gasout", "ch4_out") else LINES["data"]
    ax.plot(times[valid], values[valid], label="Measured", **style)


def _frame(axes, row: int, column: int, label: str, name: str, times, spin_up: float,
           changes, nrows: int, fold: int | None = None) -> None:
    from adm1_package.utils.plot_style import shade_spin_up

    ax = axes[row, column]
    shade_spin_up(ax, times, spin_up)
    mark_changes(ax, changes)
    ax.set_xlim(times[0], times[-1])
    if row == 0:
        # six columns leave no room for a title and its unit on one line
        ax.set_title(TITLES[name] if axes.shape[1] <= 5 else TITLES[name].replace(" [", "\n["))
    if row == nrows - 1:
        ax.set_xlabel("Time [d]")
    if column == 0:
        ax.set_ylabel(name_of(label) if fold is None else f"{name_of(label)} (fold {fold})")


def mechanistic_figure(labels, data_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from adm1_package.experiments import inspection
    from adm1_package.experiments.registry import get_spec
    from adm1_package.utils.plot_style import LINES, SPIN_UP_LABEL, figure_style
    from adm1_package.utils.plotting import target_series

    variants = ("r4", "r3", "r2", "base")
    species = get_spec("r4").influent_species
    outputs = MECHANISTIC_OUTPUTS
    with figure_style():
        figure, axes = plt.subplots(len(labels), len(outputs),
                                    figsize=(WIDTH, ROW_HEIGHT * len(labels) + LEGEND_HEIGHT),
                                    layout="constrained", squeeze=False)
        for row, label in enumerate(labels):
            runs = [inspection.simulate_mechanistic(key, label, data_dir=data_dir)
                    for key in variants]
            # the full ADM1's sample measures every output drawn; the forcing is read off
            # ADM1-R4's, whose columns ``species`` names
            reference = runs[-1]
            length = reference.length
            times = reference.times[:length]
            changes = operating_changes(runs[0].inputs, species)
            for column, name in enumerate(outputs):
                ax = axes[row, column]
                values, valid = target_series(reference.targets, name)
                _measured(ax, times, values[:length], valid[:length], name)
                for run in runs:
                    if name in run.solution:
                        ax.plot(times, run.solution[name][:length], **run.style)
                _frame(axes, row, column, label, name, times, inspection.SPIN_UP_DAYS, changes,
                       len(labels))
            print(f"  {name_of(label)} ({label}) drawn")

        handles = [Line2D([], [], label="Measured (daily)", **LINES["data_line"]),
                   Line2D([], [], label="Measured (weekly)", **LINES["data"])]
        handles += [Line2D([], [], label=f"{run.name} (MECH)" if run.name != "ADM1"
                           else "Full ADM1 (MECH)", **run.style) for run in runs]
        handles += [patch for patch in axes.ravel()[0].patches
                    if patch.get_label() == SPIN_UP_LABEL][:1]
        handles += [Line2D([], [], color="#555555", linewidth=0.9, label="Operating change")]
        figure.legend(handles=handles, loc="outside lower center", ncols=4)
    return figure


def heldout_figure(final: HeldOut, plain: HeldOut, labels, outputs):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from adm1_package.utils.plot_style import COLORS, LINES, SPIN_UP_LABEL, figure_style

    species = final.spec.influent_species
    with figure_style():
        figure, axes = plt.subplots(len(labels), len(outputs),
                                    figsize=(WIDTH, ROW_HEIGHT * len(labels) + LEGEND_HEIGHT),
                                    layout="constrained", squeeze=False)
        for row, label in enumerate(labels):
            members = final.members_of(label)
            first = members[0]
            length = _length(first.times)
            times = first.times[:length]
            plain_times, mech_cv = plain.mechanistic(label)
            if not np.allclose(plain_times[:length], times):
                raise ValueError(f"{label}: the two runs' time grids differ")
            # the hybrid at zero output: the calibration the final hybrid is built on
            anchor = final.mechanistic(label)[1]
            changes = operating_changes(final.inputs(label), species)
            for column, name in enumerate(outputs):
                ax = axes[row, column]
                values, valid = first.measured[name]
                _measured(ax, times, np.asarray(values)[:length], np.asarray(valid)[:length], name)
                seeds = np.stack([member.hybrid[name][:length] for member in members])
                ax.plot(times, mech_cv[name][:length], **LINES["mech"])
                ax.plot(times, anchor[name][:length], **LINES["anchor"])
                ax.fill_between(times, seeds.min(axis=0), seeds.max(axis=0), color=COLORS["ude"],
                                alpha=0.22, linewidth=0, zorder=2.1)
                ax.plot(times, seeds.mean(axis=0), **LINES["ude"])
                _frame(axes, row, column, label, name, times, first.spin_up_days, changes,
                       len(labels), first.fold)
            print(f"  {name_of(label)} ({label}, fold {first.fold}) drawn")

        handles = [Line2D([], [], label="Measured (daily)", **LINES["data_line"]),
                   Line2D([], [], label="Measured (weekly)", **LINES["data"]),
                   Line2D([], [], label="MECH-CV", **LINES["mech"]),
                   Line2D([], [], label="Base model (UDE at zero output)", **LINES["anchor"]),
                   Line2D([], [], label="UDE, mean of three seeds", **LINES["ude"]),
                   Patch(facecolor=COLORS["ude"], alpha=0.22, label="UDE, range of seeds"),
                   Line2D([], [], color="#555555", linewidth=0.9, label="Operating change")]
        handles += [patch for patch in axes.ravel()[0].patches
                    if patch.get_label() == SPIN_UP_LABEL][:1]
        figure.legend(handles=handles, loc="outside lower center", ncols=4)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--kinds", nargs="+", default=["mechanistic", "heldout", "corrections"],
                        choices=["mechanistic", "heldout", "corrections"])
    parser.add_argument("--models", nargs="+", default=["r4"], choices=["r4", "r3", "r2"])
    parser.add_argument("--datasets", nargs="+", default=list(REACTORS), choices=list(REACTORS))
    parser.add_argument("--output", type=Path, default=None,
                        help="Figure directory (default: plots/si).")
    parser.add_argument("--format", nargs="+", default=["pdf"],
                        choices=["pdf", "png", "svg", "eps"])
    arguments = parser.parse_args()

    configure_jax()

    from adm1_package.experiments.config import PROJECT_ROOT
    from adm1_package.utils.plot_style import save_figure

    runs_root, data_dir = arguments.run_dir, arguments.data_dir
    output = arguments.output or PROJECT_ROOT / "plots" / "si"

    def save(figure, name: str) -> None:
        for path in save_figure(figure, output / name, formats=arguments.format):
            print(f"  wrote {path}")

    if "mechanistic" in arguments.kinds:
        for campaign in arguments.datasets:
            print(f"\n=== mechanistic, dataset {DATASET[campaign]}")
            save(mechanistic_figure(REACTORS[campaign], data_dir), f"S_mechanistic_{DATASET[campaign]}")

    for key in arguments.models if {"heldout", "corrections"} & set(arguments.kinds) else ():
        final = HeldOut(key, runs_root / FINAL[key], runs_root / FINAL_BENCHMARKS, data_dir)
        plain = HeldOut(key, runs_root / PLAIN[key], runs_root / PLAIN_BENCHMARKS, data_dir)
        # the plain run's mechanistic model only, so one member suffices
        plain.members = plain.members[:1]
        for campaign in arguments.datasets:
            dataset = DATASET[campaign]
            if "heldout" in arguments.kinds:
                print(f"\n=== held out, {key}, dataset {dataset}")
                save(heldout_figure(final, plain, REACTORS[campaign], HELD_OUT_OUTPUTS[key]),
                     f"S_heldout_{key}_{dataset}")
            if "corrections" in arguments.kinds:
                print(f"\n=== corrections, {key}, dataset {dataset}")
                save(figure_6(final, REACTORS[campaign], VARIANT_CORRECTIONS[key], width=WIDTH,
                              row_height=ROW_HEIGHT), f"S_corrections_{key}_{dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
