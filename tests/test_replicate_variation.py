"""The replicate variation of dataset A that Section 2.1 of the manuscript reports.

Each setup of dataset A is the mean of three replicates. The manuscript gives, per setup,
the median over the days of the coefficient of variation of the replicates' biogas
production rates (``scripts/replicate_variation.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
import replicate_variation
from manuscript_figures import name_of


def test_the_cv_of_a_day_is_the_sample_standard_deviation_over_the_mean():
    times = np.array([0.0, 31.0, 32.0])
    rates = np.array([[5.0, 100.0, 10.0], [6.0, 103.0, 20.0], [7.0, 97.0, 30.0]])
    # day 31: standard deviation 3 (n - 1), mean 100; day 32: 10 over 20
    assert replicate_variation.daily_cv(times, rates) == pytest.approx([0.03, 0.5])


def test_the_days_of_the_spin_up_do_not_count():
    times = np.array([0.0, 29.0, 30.0, 31.0])
    rates = np.array([[5.0, 1.0, 100.0, 10.0], [6.0, 50.0, 103.0, 20.0], [7.0, 99.0, 97.0, 30.0]])
    assert replicate_variation.daily_cv(times, rates) == pytest.approx([0.03, 0.5])


def test_a_day_without_a_reading_does_not_count():
    # the records fill a day without a reading with the following reading
    times = np.array([0.0, 31.0, 32.0, 33.0])
    rates = np.array([[5.0, 100.0, 100.0, 50.0], [6.0, 103.0, 103.0, 55.0], [7.0, 97.0, 97.0, 45.0]])
    assert replicate_variation.daily_cv(times, rates) == pytest.approx([0.03, 0.1])


def test_a_day_on_which_one_replicate_has_no_valid_reading_does_not_count():
    times = np.array([0.0, 31.0, 32.0])
    rates = np.array([[5.0, 100.0, 50.0], [6.0, np.nan, 55.0], [7.0, 97.0, 45.0]])
    assert replicate_variation.daily_cv(times, rates) == pytest.approx([0.1])


def test_the_setups_are_the_replicate_triples_the_models_average():
    assert replicate_variation.SETUPS == {"A1": ("R10", "R20", "R30"),
                                          "A2": ("R40", "R50", "R60"),
                                          "A3": ("R70", "R80", "R90")}
    for setup, members in replicate_variation.SETUPS.items():
        assert name_of(f"rice-bran_{members[-1]}") == setup


@pytest.mark.requires_data
def test_the_medians_section_2_1_reports(data_dir):
    medians, days = {}, {}
    for setup, members in replicate_variation.SETUPS.items():
        cv = replicate_variation.daily_cv(*replicate_variation.load(data_dir, members))
        medians[setup], days[setup] = round(float(np.median(cv)) * 100, 1), cv.size
    assert medians == {"A1": 3.8, "A2": 2.7, "A3": 3.5}
    assert days == {"A1": 173, "A2": 220, "A3": 208}
