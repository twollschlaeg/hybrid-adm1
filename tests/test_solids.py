"""Total and volatile solids of the reduced models (utils/solids.py).

The feed's non-degradable organics and its ash only wash out, so the two pools are
a linear filter of the feed, integrated after the solve. The filter is checked
against its closed form, the pools against hand-computed steady states, and the
models against the measured start they are anchored to.
"""

from __future__ import annotations

import importlib

import diffrax as dx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import pytest
from helpers import correction_mlp, first_days

from adm1_package.experiments.registry import MODEL_KEYS, get_spec
from adm1_package.experiments.registry import SOLIDS as REGISTRY_SOLIDS
from adm1_package.utils.corrections import DEFAULT
from adm1_package.utils.datasets import measured_targets, read_outputs
from adm1_package.utils.solids import PARTICULATES, POOLS, SOLIDS, biomass, solids, washout

V_LIQ = 8.0


# ---------------------------------------------------------------------------
# The washout filter
# ---------------------------------------------------------------------------
def test_constant_feed_relaxes_exponentially_onto_it():
    times = jnp.arange(0.0, 101.0)
    inflow = jnp.full(times.shape, 0.2)
    feed = jnp.stack([jnp.full(times.shape, 20.0), jnp.full(times.shape, 5.0)], axis=1)
    start = jnp.array([44.0, 15.0])

    pools = np.asarray(washout(times, inflow, feed, V_LIQ, start))

    decay = np.exp(-0.2 / V_LIQ * np.asarray(times))[:, None]
    expected = np.array([20.0, 5.0]) + (np.asarray(start) - np.array([20.0, 5.0])) * decay
    np.testing.assert_allclose(pools, expected, rtol=1e-12)


def test_no_inflow_holds_the_pools():
    times = jnp.arange(0.0, 11.0)
    pools = washout(times, jnp.zeros(times.shape), jnp.full((11, 1), 300.0), V_LIQ,
                    jnp.array([7.0]))
    np.testing.assert_array_equal(np.asarray(pools), 7.0)


def test_the_padded_tail_changes_nothing():
    """The datasets pad the time grid with its last day; no time passes there."""
    times = jnp.concatenate([jnp.arange(0.0, 6.0), jnp.full(4, 5.0)])
    pools = np.asarray(washout(times, jnp.full(times.shape, 0.5), jnp.full((10, 1), 10.0),
                               V_LIQ, jnp.array([2.0])))
    np.testing.assert_array_equal(pools[5:, 0], pools[5, 0])


def test_the_filter_is_linear():
    times = jnp.arange(0.0, 30.0)
    rng = np.random.default_rng(0)
    inflow = jnp.asarray(rng.uniform(0.0, 0.4, times.shape))
    first, second = (jnp.asarray(rng.uniform(0, 50, (30, 1))) for _ in range(2))
    a, b = jnp.array([3.0]), jnp.array([11.0])
    together = washout(times, inflow, first + second, V_LIQ, a + b)
    apart = washout(times, inflow, first, V_LIQ, a) + washout(times, inflow, second, V_LIQ, b)
    np.testing.assert_allclose(np.asarray(together), np.asarray(apart), rtol=1e-12)


def test_a_varying_feed_matches_a_fine_integration():
    """Daily knots, feed and inflow changing between them: second order in the step."""
    from scipy.integrate import solve_ivp

    times = np.arange(0.0, 61.0)
    inflow = 0.2 + 0.15 * np.sin(times / 5.0)
    feed = 30.0 + 10.0 * np.cos(times / 7.0)
    pools = np.asarray(washout(jnp.asarray(times), jnp.asarray(inflow),
                               jnp.asarray(feed)[:, None], V_LIQ, jnp.array([12.0])))[:, 0]

    def rhs(t, y):
        q = np.interp(t, times, inflow)
        load = np.interp(t, times, inflow * feed)
        return [(load - q * y[0]) / V_LIQ]

    reference = solve_ivp(rhs, (0.0, 60.0), [12.0], t_eval=times, rtol=1e-10, atol=1e-12,
                          max_step=0.05).y[0]
    np.testing.assert_allclose(pools, reference, rtol=2e-4)


# ---------------------------------------------------------------------------
# The solids of a model solution
# ---------------------------------------------------------------------------
def _sample(days=101, inflow=0.2, start=None, no_feed=()):
    """An R4-shaped forcing: temperature, inflow, S_TAN, S_h2o, X_ch, X_pr, X_li.

    The feed is 100 g/L dry matter, 80 of it volatile (50 carbohydrates, 20 proteins,
    10 lipids) and 20 ash. Days in ``no_feed`` carry no inflow and no composition at
    all, as the records of LaRA-substrate_F1 hold them: water 0, particulates 0.
    """
    times = np.arange(0.0, float(days))
    forcing = np.tile([310.0, inflow, 2.0, 900.0, 50.0, 20.0, 10.0], (days, 1))
    for day in no_feed:
        forcing[day, 1:] = 0.0
    times_j = jnp.asarray(times)
    sample = {"time": times_j,
              "coeffs": dx.backward_hermite_coefficients(times_j, jnp.asarray(forcing))}
    if start is not None:
        sample["start"] = {name: jnp.asarray(value) for name, value in start.items()}
    return sample


def _solution(days=101):
    """A model solution holding the degradable particulates and one biomass state."""
    flat = lambda value: jnp.full(days, value)
    return {"time": jnp.arange(0.0, float(days)), "X_ch": flat(10.0), "X_pr": flat(2.0),
            "X_li": flat(1.0), "X_bac": flat(3.0)}


FACTORS = {"f_ch": 0.8, "f_pr": 0.5, "f_li": 1.0}
#: The feed's non-degradable organics: 0.2 * 50 + 0.5 * 20 + 0 * 10.
INERT_IN = 20.0
ASH_IN = 20.0


def test_the_pools_start_at_the_measurement_and_relax_onto_the_feed():
    out = solids(_solution(), _sample(start={"TS": 75.0, "VS": 60.0}), {"V_liq": V_LIQ},
                 FACTORS, biomass=("X_bac",))
    t = np.arange(0.0, 101.0)
    decay = np.exp(-0.2 / V_LIQ * t)
    # The measured VS minus the model's own solids (10 + 2 + 1 + 3) at the start
    np.testing.assert_allclose(out["X_inert"], INERT_IN + (44.0 - INERT_IN) * decay, rtol=1e-12)
    np.testing.assert_allclose(out["X_ash"], ASH_IN + (15.0 - ASH_IN) * decay, rtol=1e-12)
    np.testing.assert_allclose(out["VS"], 16.0 + out["X_inert"], rtol=1e-12)
    np.testing.assert_allclose(out["TS"], out["VS"] + out["X_ash"], rtol=1e-12)
    assert out["VS"][0] == pytest.approx(60.0)
    assert out["TS"][0] == pytest.approx(75.0)


def test_without_a_measured_start_the_pools_start_at_the_first_feed():
    out = solids(_solution(), _sample(), {"V_liq": V_LIQ}, FACTORS, biomass=("X_bac",))
    np.testing.assert_allclose(out["X_inert"], INERT_IN, rtol=1e-12)
    np.testing.assert_allclose(out["X_ash"], ASH_IN, rtol=1e-12)


def test_measured_solids_below_the_models_leave_no_negative_inerts():
    out = solids(_solution(), _sample(start={"TS": 14.0, "VS": 12.0}), {"V_liq": V_LIQ},
                 FACTORS, biomass=("X_bac",))
    assert out["X_inert"][0] == 0.0
    assert out["X_ash"][0] == pytest.approx(2.0)


def test_days_without_feed_bring_no_ash():
    """LaRA-substrate_F1's empty days read as water 0: no 1000 g/L of ash may leak in."""
    fed = solids(_solution(), _sample(), {"V_liq": V_LIQ}, FACTORS, biomass=("X_bac",))
    gaps = solids(_solution(), _sample(no_feed=(10, 11, 40)), {"V_liq": V_LIQ}, FACTORS,
                  biomass=("X_bac",))
    assert np.max(gaps["X_ash"]) <= ASH_IN + 1e-9
    # without inflow the pools are held, so the gaps only delay the relaxation
    np.testing.assert_allclose(gaps["X_ash"], fed["X_ash"], rtol=1e-12)


def test_degradability_may_change_over_time():
    """A hybrid's corrected factors come as one value per time point."""
    days = 101
    factors = {name: jnp.full(days, value) for name, value in FACTORS.items()}
    varying = solids(_solution(), _sample(), {"V_liq": V_LIQ}, factors, biomass=("X_bac",))
    constant = solids(_solution(), _sample(), {"V_liq": V_LIQ}, FACTORS, biomass=("X_bac",))
    for name in POOLS + SOLIDS:
        np.testing.assert_allclose(varying[name], constant[name], rtol=1e-12)

    less = {**factors, "f_ch": jnp.where(jnp.arange(days) < 50, 0.8, 0.6)}
    out = solids(_solution(), _sample(), {"V_liq": V_LIQ}, less, biomass=("X_bac",))
    # a less degradable feed raises the inert pool towards 0.4 * 50 + 10 = 30
    assert out["X_inert"][49] == pytest.approx(INERT_IN)
    assert INERT_IN < out["X_inert"][-1] < 30.0


# ---------------------------------------------------------------------------
# Loading the measurements
# ---------------------------------------------------------------------------
def test_older_output_files_hold_total_solids_as_water(tmp_path):
    path = tmp_path / "output_F1.csv"
    pd.DataFrame({"time": [0.0, 1.0], "q_gasout": [1.0, 2.0], "S_h2o": [930.0, 935.5]}).to_csv(
        path, index=False)
    frame = read_outputs(str(path), ["time", "q_gasout", "TS", "VS"])
    assert list(frame.columns) == ["time", "q_gasout", "TS", "VS"]
    np.testing.assert_allclose(frame["TS"], [70.0, 64.5])
    assert frame["VS"].isna().all()


def test_output_files_with_solids_are_read_as_they_are(tmp_path):
    path = tmp_path / "output_F1.csv"
    pd.DataFrame({"time": [0.0], "S_h2o": [930.0], "TS": [70.25], "VS": [50.5]}).to_csv(
        path, index=False)
    frame = read_outputs(str(path), ["time", "S_h2o", "TS", "VS"])
    assert frame.iloc[0].tolist() == [0.0, 930.0, 70.25, 50.5]


def test_a_missing_measurement_is_masked_not_scored():
    values, indices = measured_targets(jnp.array([[1.0, jnp.nan], [2.0, jnp.nan]]))
    np.testing.assert_array_equal(np.asarray(values), [[1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_array_equal(np.asarray(indices), [[1.0, 0.0], [1.0, 0.0]])


def test_every_reduced_variant_draws_the_solids_but_trains_without_them():
    assert REGISTRY_SOLIDS == SOLIDS
    for key in MODEL_KEYS:
        spec = get_spec(key)
        assert spec.diagnostics == SOLIDS
        assert not set(SOLIDS) & set(spec.outputs)
        assert spec.drawn_outputs == tuple(spec.outputs) + SOLIDS
        # solids() reads the water and the particulates as the forcing's last four columns
        assert spec.influent_species[-4:] == ("S_h2o", *PARTICULATES)
    assert get_spec("base").diagnostics == ()


def test_biomass_is_every_particulate_that_is_not_fed():
    states = ("S_ac", "X_ch", "X_pr", "X_li", "X_bac", "X_ac", "S_gas_ch4")
    assert biomass(states) == ("X_bac", "X_ac")


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------
REACTOR = "LaRA-temp_F4"
DAYS = 40.0


def _reactor(dataset):
    """REACTOR's first DAYS: ``(x, y)``."""
    x, y = dataset[dataset.labels.index(REACTOR)]
    return {**x, "input": first_days(x["input"], DAYS)}, y


def _states(key):
    return importlib.import_module(f"adm1_package.models.{key}.model").STATES.names


@pytest.mark.requires_data
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_the_solids_leave_every_other_output_as_it_was(key, installed_dataset):
    spec = get_spec(key)
    plain, y_plain = _reactor(installed_dataset(key, ["LaRA-temp"]))
    drawn, y_drawn = _reactor(installed_dataset(key, ["LaRA-temp"], diagnostics=True))
    assert list(y_plain["object_indices"]) == list(spec.outputs)
    assert list(y_drawn["object_indices"]) == list(spec.drawn_outputs)

    model = spec.mechanistic()
    before, _ = model(plain["parameters"], plain["initial_values"], plain["input"])
    after, _ = model(drawn["parameters"], drawn["initial_values"], drawn["input"])
    for name in before:
        if name not in SOLIDS + POOLS:
            np.testing.assert_array_equal(np.asarray(after[name]), np.asarray(before[name]))

    held = sum(np.asarray(after[name]) for name in PARTICULATES + biomass(_states(key)))
    np.testing.assert_allclose(after["VS"], held + np.asarray(after["X_inert"]), rtol=1e-12)
    np.testing.assert_allclose(after["TS"], np.asarray(after["VS"]) + np.asarray(after["X_ash"]),
                               rtol=1e-12)
    assert np.all(np.asarray(after["X_inert"]) >= 0) and np.all(np.asarray(after["X_ash"]) >= 0)

    start = drawn["input"]["start"]
    np.testing.assert_allclose(float(start["TS"]),
                               1000.0 - float(y_drawn["values"][0, y_drawn["object_indices"]["S_h2o"]]),
                               rtol=1e-12)
    if np.isfinite(float(start["VS"])):
        # the pools start where the first measurement puts them
        assert float(after["VS"][0]) == pytest.approx(max(float(start["VS"]), float(held[0])))
        assert float(after["X_ash"][0]) == pytest.approx(float(start["TS"]) - float(start["VS"]))


@pytest.mark.requires_data
def test_a_hybrid_feeds_the_inerts_with_its_own_degradabilities(installed_dataset):
    spec = get_spec("r4")
    dataset = installed_dataset("r4", ["LaRA-temp"], diagnostics=True)
    x, _ = _reactor(dataset)
    mlp = correction_mlp(spec, spec.nn_in_size(dataset), DEFAULT)
    solution, _ = spec.ude(DEFAULT)(x["parameters"], x["initial_values"], x["input"], mlp,
                                    jr.PRNGKey(1))
    factors = np.asarray(solution["nutrient_factors"])
    assert np.ptp(factors[:, 0]) > 0, "the network should move f_ch over time"
    expected = solids(solution, x["input"], x["parameters"],
                      dict(zip(spec.factor_names, factors.T)), biomass(_states("r4")))
    for name in SOLIDS + POOLS:
        np.testing.assert_allclose(np.asarray(solution[name]), np.asarray(expected[name]),
                                   rtol=1e-12)
    constant = solids(solution, x["input"], x["parameters"], x["parameters"],
                      biomass(_states("r4")))
    assert not np.allclose(np.asarray(solution["X_inert"]), np.asarray(constant["X_inert"]))
