"""The forcing the models are given between the daily records
(adm1_package/utils/forcing.forcing_coefficients).

The feed, the temperature and the influent are recorded once a day. A cubic spline
through the records overshot a step by about 15% the next day and drove the inflow
negative after a day without feed. The forcing now moves from one record to the next
along a smoothstep: it passes through every record at its time stamp, stays within the
two values, reads at most one record ahead, and keeps the spline's smoothness, so the
solver takes as many steps as before.
"""

from __future__ import annotations

from pathlib import Path

import diffrax as dx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from adm1_package.utils.forcing import forcing_coefficients, smoothstep_coefficients

TIMES = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
VALUES = jnp.array([[1.0, 0.0], [1.0, 5.0], [3.0, 0.0], [3.0, 5.0], [2.0, 5.0], [2.0, 0.0]])


def _forcing(times=TIMES, values=VALUES):
    return dx.CubicInterpolation(times, forcing_coefficients(times, values))


def test_each_record_is_reached_at_its_time_stamp():
    forcing = _forcing()
    for day in range(len(TIMES)):
        np.testing.assert_allclose(forcing.evaluate(TIMES[day]), VALUES[day], atol=1e-12)
    # half way to the next record, half the change
    np.testing.assert_allclose(forcing.evaluate(2.5), (VALUES[2] + VALUES[3]) / 2, atol=1e-12)
    # two equal records hold their value between them
    np.testing.assert_allclose(forcing.evaluate(0.5)[0], VALUES[0, 0], atol=1e-12)


def test_the_forcing_stays_between_the_two_records():
    rng = np.random.default_rng(0)
    times = jnp.cumsum(jnp.asarray(rng.uniform(0.8, 1.2, 40)))
    values = jnp.asarray(rng.choice([0.0, 0.0, 1.0, 4.0], size=(40, 3)) * rng.uniform(0.5, 1.5, (40, 3)))
    forcing = _forcing(times, values)
    for day in range(39):
        inside = jnp.linspace(times[day], times[day + 1], 25)
        path = jax.vmap(forcing.evaluate)(inside)
        low = jnp.minimum(values[day], values[day + 1])
        high = jnp.maximum(values[day], values[day + 1])
        assert bool(jnp.all(path >= low - 1e-12)) and bool(jnp.all(path <= high + 1e-12))


def test_a_day_without_feed_never_drives_the_inflow_negative():
    path = jax.vmap(_forcing().evaluate)(jnp.linspace(0.0, 5.0, 501))
    assert float(path[:, 1].min()) >= 0.0


def test_the_forcing_reads_at_most_one_record_ahead():
    """Up to a record's time stamp nothing later than the next record matters; on the
    way to the next record the forcing already moves toward it."""
    later = VALUES.at[3:].set(VALUES[3:] * 7.0 + 1.0)
    before = jnp.linspace(0.0, float(TIMES[2]), 61)
    np.testing.assert_array_equal(jax.vmap(_forcing().evaluate)(before),
                                  jax.vmap(_forcing(TIMES, later).evaluate)(before))
    assert not np.allclose(_forcing().evaluate(2.5), _forcing(TIMES, later).evaluate(2.5))


def test_the_forcing_is_the_smoothstep_through_its_values():
    """So a forcing built again from its values at the records (the memory, the input
    noise) is the forcing itself."""
    for built, through in zip(forcing_coefficients(TIMES, VALUES),
                              smoothstep_coefficients(TIMES, VALUES)):
        np.testing.assert_array_equal(built, through)


def test_the_forcing_has_no_kinks_for_the_solver_to_find():
    forcing = _forcing()
    for knot in TIMES[1:-1]:
        np.testing.assert_allclose(forcing.derivative(knot - 1e-9), 0.0, atol=1e-6)
        np.testing.assert_allclose(forcing.derivative(knot + 1e-9), 0.0, atol=1e-6)


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2", "base"])
def test_every_variants_dataset_gives_its_model_this_forcing(key, data_dir, installed_dataset):
    """Temperature (in K), inflow and the influent columns of a reactor's record, each at
    its time stamp."""
    dataset = installed_dataset(key, ["LaRA-temp"])
    assert dataset.forcing == "smoothstep"
    x, _ = dataset[dataset.labels.index("LaRA-temp_F3")]
    record = pd.read_csv(Path(data_dir) / "LaRA-temp" / "input_F3.csv")
    n = len(record)
    times = x["input"]["time"][:n]
    forcing = dx.CubicInterpolation(times, tuple(c[: n - 1] for c in x["input"]["coeffs"]))
    at_records = np.asarray(jax.vmap(forcing.evaluate)(times))
    np.testing.assert_allclose(at_records[:, 0], record["temperature"].to_numpy() + 273.15,
                               rtol=1e-12)
    np.testing.assert_allclose(at_records[:, 1], record["inflow"].to_numpy(), rtol=1e-12,
                               atol=1e-15)
