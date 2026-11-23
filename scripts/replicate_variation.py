"""The replicate variation of dataset A that Section 2.1 of the manuscript reports.

    python scripts/replicate_variation.py
    python scripts/replicate_variation.py --variants

The models see each setup of dataset A as the mean of its three replicates (the datasets
average them, ``ADM1Dataset.REPLICATES``): R10 to R30 are setup A1, R40 to R60 A2, R70 to R90 A3
(:data:`SETUPS`). On every day on which all three replicates of a setup were read, the
coefficient of variation of their biogas production rates is the sample standard
deviation (n - 1) over the mean. The manuscript gives its median over the days, per setup.

The days count as they do in the scores: from day 30, the spin-up being left out, and only
the days on which the gas was read (``datasets.measured_points``). That leaves out the
readings removed in processing as faulty (manuscript Appendix A) and the days without a reading,
which the records fill with the following reading. A day counts only if all three
replicates were read on it. The rates are those of the processed records,
``data/rice-bran/output_R*.csv``: the laboratory's 'Biogasrate (NmL/d)', i.e. the volume
since the previous reading over its interval, in NL/d.

``--variants`` adds what these choices change: all days, the population standard
deviation, and the mean instead of the median.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_common_arguments, configure_jax  # noqa: E402
from manuscript_figures import SETUP  # noqa: E402

#: Each setup's replicates, by the manuscript's name of the setup; the models name a setup
#: after its last replicate (``rice-bran_R30`` is A1).
SETUPS = {f"A{number}": tuple(f"R{int(last[1:]) - 10 * k}" for k in (2, 1, 0))
          for last, number in SETUP.items()}
#: The days at the start of a record that no score counts.
SPIN_UP_DAYS = 30.0


def read(rates) -> np.ndarray:
    """Which records of one replicate hold a gas reading: finite, and not a fill, i.e. not
    equal to the next record (``datasets.measured_points``, as the scores count them)."""
    from adm1_package.utils.datasets import measured_points

    values = np.asarray(rates, dtype=float)[:, None]
    return np.asarray(measured_points(values, ("q_gasout",)))[:, 0] > 0.5


def daily_cv(times, rates, from_day: float = SPIN_UP_DAYS, ddof: int = 1) -> np.ndarray:
    """The coefficient of variation of the replicates on every day from ``from_day`` after
    the first record on which all of them were read; ``rates`` holds one row per
    replicate."""
    times = np.asarray(times, dtype=float)
    rates = np.asarray(rates, dtype=float)
    days = np.all([read(replicate) for replicate in rates], axis=0) & (times >= times[0] + from_day)
    return np.std(rates[:, days], axis=0, ddof=ddof) / np.mean(rates[:, days], axis=0)


def load(data_dir: Path, members) -> tuple[np.ndarray, np.ndarray]:
    """Times and biogas production rates of a setup's replicates, one row per replicate,
    cut to the shortest record as the averaging does."""
    import pandas as pd

    records = [pd.read_csv(Path(data_dir) / "rice-bran" / f"output_{member}.csv",
                           usecols=["time", "q_gasout"]) for member in members]
    length = min(len(record) for record in records)
    times = records[0]["time"].to_numpy(dtype=float)[:length]
    for member, record in zip(members, records):
        if not np.array_equal(record["time"].to_numpy(dtype=float)[:length], times):
            raise ValueError(f"{member} has other record times than {members[0]}")
    return times, np.stack([record["q_gasout"].to_numpy(dtype=float)[:length] for record in records])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--variants", action="store_true",
                        help="Also print what the choices of days, deviation and average change.")
    arguments = parser.parse_args()

    configure_jax()

    medians = []
    for setup, members in SETUPS.items():
        times, rates = load(arguments.data_dir, members)
        cv = daily_cv(times, rates) * 100
        medians.append(float(np.median(cv)))
        print(f"{setup} ({', '.join(members)}): median CV {medians[-1]:.1f}% over {cv.size} days "
              f"from day {SPIN_UP_DAYS:.0f}; quartiles {np.percentile(cv, 25):.1f} and "
              f"{np.percentile(cv, 75):.1f}%, largest {cv.max():.0f}%")
        if arguments.variants:
            every_day = daily_cv(times, rates, from_day=0.0) * 100
            population = daily_cv(times, rates, ddof=0) * 100
            print(f"    all days: median {np.median(every_day):.1f}% over {every_day.size} days; "
                  f"population standard deviation: median {np.median(population):.1f}%; "
                  f"mean instead of median: {cv.mean():.1f}%")
    print(f"\nSection 2.1: a median coefficient of variation of {min(medians):.1f} to "
          f"{max(medians):.1f}% per setup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
