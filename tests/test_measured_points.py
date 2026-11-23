"""Scores at the measurement times.

The loss fits the targets interpolated onto the daily grid, but the scores count only the
records that hold a measurement: the weekly lab values on their lab days, and the gas on
the days it was read. The records copy a gas reading back onto the days without one
and onto the first days, so a gas rate equal to the next record's is a fill.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from adm1_package.experiments.metrics import score_solution
from adm1_package.utils.datasets import measured_points

NAN = float("nan")


def test_a_record_without_a_value_is_not_a_measurement():
    lab = jnp.array([[1.2], [NAN], [NAN], [1.4], [NAN]])
    np.testing.assert_array_equal(measured_points(lab, ["S_TAN"])[:, 0], [1, 0, 0, 1, 0])


def test_a_gas_rate_copied_from_the_next_record_is_a_fill():
    # days 0-1 filled from day 2, day 4 inserted before the 2-day reading of day 5
    gas = jnp.array([[3.0], [3.0], [3.0], [2.0], [5.0], [5.0], [4.0]])
    np.testing.assert_array_equal(measured_points(gas, ["q_gasout"])[:, 0], [0, 0, 1, 1, 0, 1, 1])


def test_lab_values_may_repeat():
    """A pH of 7.5 two weeks running is two measurements."""
    lab = jnp.array([[7.5], [7.5], [NAN]])
    np.testing.assert_array_equal(measured_points(lab, ["pH"])[:, 0], [1, 1, 0])


def test_each_output_column_is_judged_on_its_own():
    values = jnp.array([[3.0, 1.0, 1.0], [3.0, 1.0, NAN], [2.0, 1.0, NAN]])
    np.testing.assert_array_equal(measured_points(values, ["ch4_out", "S_ac", "S_TAN"]),
                                  [[0, 1, 1], [1, 1, 0], [1, 1, 0]])


def test_the_scores_count_only_measured_points():
    time = jnp.arange(6.0)
    targets = {"values": jnp.ones((6, 2)), "indices": jnp.ones((6, 2)),
               "measured": jnp.array([[1, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]], dtype=float),
               "object_indices": {"q_gasout": 0, "S_TAN": 1}, "label": "a"}
    scored = score_solution({"q_gasout": jnp.ones(6), "S_TAN": jnp.ones(6)},
                            {"input": {"time": time}}, targets, ["q_gasout", "S_TAN"])
    # the first point is never scored
    np.testing.assert_array_equal(scored.mask, [[0, 0], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]])


def test_the_figures_draw_the_measured_points():
    """And the parity plots and the change windows score them, as the scores do."""
    from adm1_package.utils.plotting import target_series

    targets = {"values": jnp.arange(5.0)[:, None], "indices": jnp.ones((5, 1)),
               "measured": jnp.array([[1.0], [0.0], [1.0], [0.0], [0.0]]),
               "object_indices": {"S_TAN": 0}}
    values, valid = target_series(targets, "S_TAN")
    np.testing.assert_array_equal(valid, [True, False, True, False, False])
    np.testing.assert_array_equal(values, np.arange(5.0))
