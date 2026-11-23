"""The organic loading rates of the manuscript's Table 1.

    python scripts/loading_rates.py
    python scripts/loading_rates.py --steps

The organic loading rate (OLR) is the volatile solids fed per litre of reactor and day,

    OLR = q * VS_in / V_liq,

with the feed q (kg fresh mass per day), the volatile solids of the feed VS_in (g per kg
fresh mass) and the liquid volume V_liq (L, from the campaign's parameter file, as the
models take it). Feed and composition are those of the processed records,
``data/<campaign>/input_*.csv``, i.e. what the models are given. A setup of dataset A is the
mean of its three replicates (``replicate_variation.SETUPS``), as the models average them.

The volatile solids are the sum of the carbohydrates, proteins and lipids. The Weender
analysis calculates the carbohydrates as the organic matter less proteins and lipids, so the
three make up the organic dry matter (oTS) of every analysed substrate.

The OLR was raised in steps (:func:`steps`). Table 1 gives their range, from the lowest to
the highest step. The printout adds the minimum and maximum over all days with feed;
``--steps`` lists every step's first day and OLR. Dataset A was not fed on one day a week,
so its OLR on the days with feed is 7/6 of its weekly mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import add_common_arguments  # noqa: E402
from replicate_variation import SETUPS  # noqa: E402

#: Table 1's reactors, in its order: (campaign, records); a setup of dataset A is the mean of
#: its replicates' records.
REACTORS = {
    **{name: ("rice-bran", members) for name, members in SETUPS.items()},
    **{f"B{number}": ("LaRA-temp", (f"F{number}",)) for number in range(1, 7)},
    **{f"C{number}": ("LaRA-substrate", (f"F{number}",)) for number in range(1, 7)},
}


def volatile_solids(influent) -> np.ndarray:
    """The feed's volatile solids in g/kg fresh mass: its carbohydrates, proteins and lipids."""
    return np.asarray(influent["X_ch"] + influent["X_pr"] + influent["X_li"], dtype=float)


def organic_loading(inflow, solids, volume: float) -> np.ndarray:
    """g VS per litre and day from the feed (kg/d), its volatile solids (g/kg) and the
    liquid volume (L)."""
    return np.asarray(inflow, dtype=float) * np.asarray(solids, dtype=float) / volume


def steps(times, olr, tolerance: float = 0.05, days: int = 3) -> list[tuple[float, float]]:
    """The loading steps: (first day, OLR) of every plateau on the days with feed. A plateau
    is ``days`` fed days within ``tolerance`` of each other. Three days take the last step
    of B2, which its record ends after three days, but not two irregular feeding days of
    A2."""
    fed = np.asarray(olr, dtype=float) > 0
    times, olr = np.asarray(times, dtype=float)[fed], np.asarray(olr, dtype=float)[fed]
    found: list[tuple[float, float]] = []
    for start in range(len(olr) - days + 1):
        window = olr[start:start + days]
        if np.ptp(window) > tolerance * window.min():
            continue
        level = float(np.median(window))
        if not found or abs(level - found[-1][1]) > tolerance * found[-1][1]:
            found.append((float(times[start]), level))
    return found


def step_range(times, olr) -> tuple[float, float]:
    """Table 1's range: the OLR of the lowest and of the highest loading step."""
    levels = [level for _, level in steps(times, olr)]
    return min(levels), max(levels)


def load(data_dir: Path, campaign: str, members) -> tuple[np.ndarray, np.ndarray]:
    """Times and daily OLR of a reactor, or of a setup from the mean feed of its replicates,
    cut to the shortest record as the averaging does."""
    import pandas as pd

    directory = Path(data_dir) / campaign
    stored = json.loads((directory / f"{campaign}-parameters.json").read_text(encoding="utf-8"))
    records = [pd.read_csv(directory / f"input_{member}.csv") for member in members]
    length = min(len(record) for record in records)
    times = records[0]["time"].to_numpy(dtype=float)[:length]
    for member, record in zip(members, records):
        if not np.array_equal(record["time"].to_numpy(dtype=float)[:length], times):
            raise ValueError(f"{member} has other record times than {members[0]}")
    inflow = np.mean([record["inflow"].to_numpy(dtype=float)[:length] for record in records], axis=0)
    solids = np.mean([volatile_solids(record)[:length] for record in records], axis=0)
    return times, organic_loading(inflow, solids, float(stored["V_liq"]["value"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--steps", action="store_true", help="Also print every reactor's loading steps.")
    arguments = parser.parse_args()

    print("OLR on the days with feed, g VS/(L d): Table 1 (lowest to highest loading step)")
    for name, (campaign, members) in REACTORS.items():
        times, olr = load(arguments.data_dir, campaign, members)
        low, high = step_range(times, olr)
        fed = olr[olr > 0]
        print(f"  {name}: {low:.1f}--{high:.1f}   (minimum {fed.min():.2f}, maximum {fed.max():.2f}, "
              f"{fed.size} days with feed)")
        if arguments.steps:
            print("      steps (day: OLR) " + ", ".join(f"{day:.0f}: {value:.2f}"
                                                  for day, value in steps(times, olr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
