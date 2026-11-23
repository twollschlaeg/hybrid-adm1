"""The organic loading rates of the manuscript's Table 1 (``scripts/loading_rates.py``).

The loading rate is the volatile solids fed per litre of reactor and day. The feed's volatile
solids are the sum of its carbohydrates, proteins and lipids, since the Weender analysis
calculates the carbohydrates as the organic matter less proteins and lipids. Table 1 gives
the range of the loading steps, from the lowest to the highest.
"""

from __future__ import annotations

import loading_rates
import numpy as np
import pandas as pd
import pytest
from manuscript_figures import name_of


def test_the_loading_rate_is_the_volatile_solids_fed_per_litre_and_day():
    # 0.2 kg of feed a day with 100 g VS per kg, in 8 L
    assert loading_rates.organic_loading(np.array([0.2]), np.array([100.0]), 8.0) == pytest.approx([2.5])


def test_the_volatile_solids_are_the_carbohydrates_proteins_and_lipids():
    influent = pd.DataFrame({"X_ch": [70.0], "X_pr": [20.0], "X_li": [10.0], "S_TAN": [3.0]})
    assert loading_rates.volatile_solids(influent) == pytest.approx([100.0])


def test_the_range_spans_the_lowest_and_the_highest_step():
    olr = np.array([2.0] * 7 + [0.0] + [3.0] * 7 + [2.5] * 7)
    assert loading_rates.step_range(np.arange(olr.size, dtype=float), olr) == pytest.approx((2.0, 3.0))


def test_a_step_needs_three_days_with_feed():
    # two irregular days start no step; a last step of three days at the end of a record does
    olr = np.array([2.0] * 10 + [4.0] * 2 + [2.0] * 10 + [3.0] * 3)
    steps = loading_rates.steps(np.arange(olr.size, dtype=float), olr)
    assert steps == [(0.0, pytest.approx(2.0)), (22.0, pytest.approx(3.0))]


def test_the_steps_are_the_plateaus_of_the_loading_rate():
    olr = np.array([2.0] * 7 + [2.5] * 7 + [3.0] * 14)
    olr[3] = 0.0     # a day without feed
    olr[20] = 4.0    # a single day off its plateau
    steps = loading_rates.steps(np.arange(olr.size, dtype=float), olr)
    assert steps == [(0.0, pytest.approx(2.0)), (7.0, pytest.approx(2.5)), (14.0, pytest.approx(3.0))]


def test_the_reactors_are_those_of_table_1():
    assert list(loading_rates.REACTORS) == ["A1", "A2", "A3", *(f"B{i}" for i in range(1, 7)),
                                            *(f"C{i}" for i in range(1, 7))]
    for name, (campaign, members) in loading_rates.REACTORS.items():
        assert name_of(f"{campaign}_{members[-1]}") == name


@pytest.mark.requires_data
def test_the_ranges_table_1_gives(data_dir):
    cells = {name: "{:.1f}--{:.1f}".format(*loading_rates.step_range(*loading_rates.load(data_dir, *reactor)))
             for name, reactor in loading_rates.REACTORS.items()}
    assert cells == {"A1": "0.6--2.4", "A2": "0.5--2.7", "A3": "0.6--2.8",
                     "B1": "2.1--7.2", "B2": "2.1--9.2", "B3": "2.1--10.3", "B4": "2.1--10.3",
                     "B5": "2.1--10.3", "B6": "2.1--10.3",
                     "C1": "1.6--2.7", "C2": "1.6--2.6", "C3": "1.6--2.6", "C4": "1.6--2.6",
                     "C5": "1.6--2.6", "C6": "1.6--2.6"}
