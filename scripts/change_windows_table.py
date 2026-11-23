"""The tables of held-out biogas production after operating changes: Table 6 of the
manuscript (Section 3.4) and Table S5 of the Supporting Information, its development rows.

    python scripts/change_windows_table.py
    python scripts/change_windows_table.py --latex

Every reactor is predicted by the fold that held it out (``manuscript_figures.HeldOut``), by
MECH-CV, the base model of the final hybrids, the final ADM1-R4 (:data:`manuscript_figures.FINAL`)
and the hybrids of the development table's rows b to e (:data:`ROWS`); every hybrid is the
mean of its three seeds. Measurements count as they do in the scores: from day 30, on the
days they were read. Each cell is the biogas RMSE of the span:

* temperature steps, dataset B: within 21 days after the steps of the heating program,
  merged into one window per reactor (``manuscript_figures.windows``). Only the first window
  counts: a short thermostat dip late in B3's record, which ``operating_changes`` takes
  for a step of its own, comes long after the heating;
* loading, dataset B: from the first increase of the loading rate to the end of the record;
* substitution, dataset C: from the substitution of the corn silage on day 84 to the end of
  the record, for the four reactors whose co-substrate was replaced and, on the same days,
  for the two that kept their feed.

The printout adds the simulated over the measured gas production of every span.
"""

from __future__ import annotations

import argparse

import numpy as np

from _bootstrap import add_common_arguments, configure_jax  # noqa: E402
from manuscript_figures import (FINAL, FINAL_BENCHMARKS, PLAIN, PLAIN_BENCHMARKS,  # noqa: E402
                                UNTUNED, HeldOut, _length, _rmse, name_of, operating_changes,
                                windows)

#: The development table's rows, with the benchmarks they are built on: the narrow
#: correction on the plain calibration (b), on the Arrhenius calibration within the original
#: bounds (c) and with fast hydrolysis (d), and the final configuration with the network
#: hyperparameters of the first version of the study (e).
ROWS = {
    "b": (PLAIN["r4"], PLAIN_BENCHMARKS),
    "c": ("r4/fold_arrhenius_narrow_ensemble_s45-46-47", "benchmarks-smoothstep-arrhenius"),
    "d": ("r4/fold_fast_narrow_ensemble_s45-46-47", FINAL_BENCHMARKS),
    "e": (UNTUNED["r4"], FINAL_BENCHMARKS),
}
#: The day on which the corn silage of dataset C was replaced (Table 1).
SUBSTITUTION_DAY = 84.0
#: The columns of the manuscript's table and of the Supporting Information's.
COLUMNS = ("MECH-CV", "Base", "UDE")
DEVELOPMENT_COLUMNS = ("MECH-CV", "b", "c", "d", "e", "UDE")
#: Where the heating program of each reactor of dataset B ended (Table 1 gives the steps);
#: B1 stayed at 37 C.
HEATING = {"LaRA-temp_F2": r"to \SI{42}{\degreeCelsius}", "LaRA-temp_F3": r"to \SI{46}{\degreeCelsius}",
           "LaRA-temp_F4": r"to \SI{46}{\degreeCelsius}", "LaRA-temp_F5": r"to \SI{46}{\degreeCelsius}",
           "LaRA-temp_F6": r"to \SI{55}{\degreeCelsius}"}
TEMPERATURE = {"LaRA-temp_F1": r"at \SI{37}{\degreeCelsius}", "LaRA-temp_F2": r"at \SI{42}{\degreeCelsius}",
               "LaRA-temp_F3": r"at \SI{46}{\degreeCelsius}", "LaRA-temp_F4": r"at \SI{46}{\degreeCelsius}",
               "LaRA-temp_F5": r"at \SI{46}{\degreeCelsius}", "LaRA-temp_F6": r"at \SI{55}{\degreeCelsius}"}
FEED = {"LaRA-substrate_F1": "CS", "LaRA-substrate_F2": "CS, CoS",
        "LaRA-substrate_F3": r"CoS $\rightarrow$ SCM", "LaRA-substrate_F4": r"CoS $\rightarrow$ DM",
        "LaRA-substrate_F5": r"CoS $\rightarrow$ GS", "LaRA-substrate_F6": r"CoS $\rightarrow$ RS"}


class Reactor:
    """One held-out reactor: its measured biogas, scored days and every column's biogas."""

    def __init__(self, runs: dict[str, HeldOut], label: str):
        final = runs["UDE"]
        members = final.members_of(label)
        first = members[0]
        length = _length(first.times)
        self.label = label
        self.fold = first.fold
        self.times = first.times[:length]
        values, valid = first.measured["q_gasout"]
        self.values = np.asarray(values)[:length]
        self.scored = np.asarray(valid, dtype=bool)[:length] & (
            self.times >= self.times[0] + first.spin_up_days)
        self.scored[0] = False
        self.gas = {"MECH-CV": runs["b"].mechanistic(label)[1]["q_gasout"][:length],
                    "Base": final.mechanistic(label)[1]["q_gasout"][:length],
                    "UDE": np.mean([member.hybrid["q_gasout"][:length] for member in members], axis=0)}
        for row in ROWS:
            self.gas[row] = np.mean([member.hybrid["q_gasout"][:length]
                                     for member in runs[row].members_of(label)], axis=0)
        self.changes = operating_changes(final.inputs(label), final.spec.influent_species)

    def span(self, start: float, end: float) -> dict[str, tuple[float, float]]:
        """RMSE and simulated/measured gas of every column on the scored days in
        [start, end]."""
        mask = self.scored & (self.times >= start) & (self.times <= end)
        return {column: (_rmse(curve, self.values, mask),
                         float(curve[mask].sum() / self.values[mask].sum()))
                for column, curve in self.gas.items()}

    def row(self, what: str, start: float, end: float) -> dict[str, str]:
        """Print the span's scores; return its LaTeX row for each table."""
        scores = self.span(start, end)
        change = 100 * (scores["UDE"][0] / scores["MECH-CV"][0] - 1)
        print(f"  {name_of(self.label)} (fold {self.fold}) {what}, days {start:.0f}-{end:.0f}: "
              + ", ".join(f"{column} {rmse:.2f} ({ratio:.2f})"
                          for column, (rmse, ratio) in scores.items())
              + f"; UDE {change:+.0f}% vs MECH-CV")
        return {table: " & ".join(f"{scores[column][0]:.2f}" for column in columns)
                for table, columns in (("main", COLUMNS), ("development", DEVELOPMENT_COLUMNS))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--latex", action="store_true", help="Also print the rows as LaTeX.")
    arguments = parser.parse_args()

    configure_jax()

    runs_root, data_dir = arguments.run_dir, arguments.data_dir
    runs = {"UDE": HeldOut("r4", runs_root / FINAL["r4"], runs_root / FINAL_BENCHMARKS, data_dir)}
    for row, (run, benchmarks) in ROWS.items():
        runs[row] = HeldOut("r4", runs_root / run, runs_root / benchmarks, data_dir)

    print("Biogas RMSE [NL/d] (simulated/measured gas)")
    blocks: dict[str, list[tuple[str, str, dict[str, str]]]] = {
        "Temperature steps": [], "Loading": [], "Substitution": []}
    for number in range(1, 7):
        reactor = Reactor(runs, f"LaRA-temp_F{number}")
        name = name_of(reactor.label)
        if reactor.label in HEATING:
            start, end = windows(reactor.changes["temperature"])[0]
            blocks["Temperature steps"].append(
                (f"{name} ({HEATING[reactor.label]})", f"{start:.0f}--{end:.0f}",
                 reactor.row("21 days after the steps", start, end)))
        start, end = reactor.changes["load"][0], float(reactor.times[-1])
        blocks["Loading"].append((f"{name} ({TEMPERATURE[reactor.label]})", f"{start:.0f}--{end:.0f}",
                                  reactor.row("loading", start, end)))
    for number in range(1, 7):
        reactor = Reactor(runs, f"LaRA-substrate_F{number}")
        end = float(reactor.times[-1])
        blocks["Substitution"].append(
            (f"{name_of(reactor.label)} ({FEED[reactor.label]})", f"{SUBSTITUTION_DAY:.0f}--{end:.0f}",
             reactor.row("substitution", SUBSTITUTION_DAY, end)))

    if arguments.latex:
        for table, columns in (("main", COLUMNS), ("development", DEVELOPMENT_COLUMNS)):
            print(f"\n{table}: Reactor & Days & " + " & ".join(columns))
            for block, rows in blocks.items():
                print(f"\\multicolumn{{{len(columns) + 2}}}{{@{{}}l}}{{\\textit{{{block}}}}} \\\\")
                for reactor, days, cells in rows:
                    print(f"{reactor} & {days} & {cells[table]} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
