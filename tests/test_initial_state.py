"""Where a simulation starts.

``calibrated`` starts from the per-reactor states in the ``*-IVs_*.csv`` files,
which were fitted to each reactor's own data and do not exist for a new one.
``warmup`` starts from the model's own quasi-steady state at the reactor's
first-week operating conditions, with the measured TAN, water and pH held at
their first measured values; measured states and the gas phase then come from
the first measurements, as they do for ``calibrated``.
"""

from __future__ import annotations

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import pytest
from helpers import correction_mlp, first_days

from adm1_package.experiments.registry import get_spec
from adm1_package.models.r2.model import STATES as R2_STATES
from adm1_package.models.r3.model import STATES as R3_STATES
from adm1_package.models.r4.model import STATES as R4_STATES
from adm1_package.utils import initial_state
from adm1_package.utils.corrections import DEFAULT
from adm1_package.utils.datasets import custom_collate_fn
from adm1_package.utils.gas import gas_transfer_rates

LAYOUTS = {"r4": R4_STATES, "r3": R3_STATES, "r2": R2_STATES}
REACTOR = "LaRA-temp_F4"


@pytest.fixture(scope="module")
def sample(installed_dataset):
    """``sample(key, label=REACTOR)``: a reactor of a variant's installed dataset."""
    def get(key, label=REACTOR):
        dataset = installed_dataset(key)
        return dataset[dataset.labels.index(label)]

    return get


def _start_of_simulation(simulator, x, days=5.0, network=()):
    _, sol = simulator(x["parameters"], x["initial_values"], first_days(x["input"], days), *network)
    return np.asarray(sol.ys[0])


def _raw_warm_up(key, x):
    """The warm-up's own end state, before the measured values are put back."""
    model = get_spec(key).mechanistic()
    interp = dx.CubicInterpolation(x["input"]["time"], x["input"]["coeffs"])
    t0 = x["input"]["time"][0]
    forcing = initial_state.ConstantForcing(initial_state.first_week_forcing(interp, t0))
    state = initial_state.warm_up(LAYOUTS[key], model._rhs, lambda f: (x["parameters"], f),
                                  x["parameters"], x["initial_values"], x["input"]["start"],
                                  forcing, t0)
    return model, forcing, np.asarray(state)


# -- the options -----------------------------------------------------------------------
def test_an_unknown_initial_state_is_rejected():
    with pytest.raises(ValueError, match="initial state"):
        initial_state.resolve_initial_state("steady")


def test_a_simulator_without_the_option_starts_from_the_calibrated_states():
    assert initial_state.resolve_initial_state(None) == "calibrated"


# -- the forcing -----------------------------------------------------------------------
def test_the_warm_up_runs_at_the_mean_of_the_first_week():
    ts = jnp.arange(0.0, 21.0)
    values = jnp.stack([ts, 2 * ts + 1], axis=1)
    interp = dx.CubicInterpolation(ts, dx.backward_hermite_coefficients(ts, values))
    # days 0..6 and 5..11
    np.testing.assert_allclose(np.asarray(initial_state.first_week_forcing(interp, 0.0)), [3.0, 7.0])
    np.testing.assert_allclose(np.asarray(initial_state.first_week_forcing(interp, 5.0)), [8.0, 17.0])


# -- what the datasets hand over -------------------------------------------------------
MEASURED = {   # and S_ch4, which starts in equilibrium with the measured gas phase
    "r4": ["S_ch4", "S_TAN", "S_h2o", "S_gas_ch4", "S_gas_co2"],
    "r3": ["S_ac", "S_ch4", "S_TAN", "S_h2o", "S_gas_ch4", "S_gas_co2"],
    "r2": ["S_va", "S_bu", "S_pro", "S_ac", "S_ch4", "S_TAN", "S_h2o", "S_gas_ch4", "S_gas_co2"],
}


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2"])
def test_the_calibrated_files_name_the_models_states(key, data_dir):
    """The datasets pick the calibrated states out of these files by name."""
    names = set(LAYOUTS[key].names)
    files = [path for campaign in ("LaRA-temp", "LaRA-substrate", "rice-bran")
             for path in sorted((data_dir / campaign).glob(f"R{key[1]}-IVs_*.csv"))]
    assert files
    for path in files:
        stored = set(pd.read_csv(path, index_col=0).index)
        assert stored == names, f"{path.name}: {sorted(stored ^ names)}"


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2"])
def test_the_dataset_marks_the_start_states_it_took_from_measurements(key, sample):
    x, y = sample(key)
    names = LAYOUTS[key].names
    expected = [1.0 if name in MEASURED[key] else 0.0 for name in names]
    np.testing.assert_array_equal(np.asarray(x["input"]["start"]["measured"]), expected)
    if key == "r4":
        assert np.isnan(float(x["input"]["start"]["pH"]))
    else:
        assert float(x["input"]["start"]["pH"]) == float(y["values"][0, y["object_indices"]["pH"]])


def test_a_batch_carries_the_start_of_each_sample():
    def sample(pH, label):
        start = {"measured": jnp.asarray([1.0, 0.0, 1.0]), "pH": jnp.asarray(pH)}
        inputs = {"time": jnp.arange(3.0), "coeffs": [jnp.zeros((2, 4))] * 4,
                  "means": jnp.zeros(4), "stds": jnp.ones(4), "start": start}
        x = {"input": inputs, "initial_values": jnp.zeros(3), "parameters": {"V_liq": jnp.asarray(8.0)}}
        y = {"values": jnp.zeros((3, 2)), "indices": jnp.ones((3, 2)), "label": label,
             "object_indices": {"q_gasout": 0, "ch4_out": 1}}
        return x, y

    x, _ = custom_collate_fn([sample(7.8, "a"), sample(8.0, "b")])
    np.testing.assert_array_equal(np.asarray(x["input"]["start"]["pH"]), [7.8, 8.0])
    assert x["input"]["start"]["measured"].shape == (2, 3)


# -- the measured start chemistry (R2/R3) ----------------------------------------------
# The constants of the campaigns' parameter files, the van 't Hoff enthalpies and molar
# masses of the models' right-hand sides.
DISSOCIATION = {"K_a_co2": 4.466835921509635e-07, "K_a_TAN": 5.623413251903491e-10,
                "K_w": 1e-13, "K_a_ac": 1.7378008287493744e-05,
                "K_a_pro": 1.3182567385564052e-05, "K_a_bu": 1.5135612484362051e-05,
                "K_a_va": 1.3803842646028827e-05}
ACIDS = {   # ionized species: (total, molar mass, constant)
    "r3": {"S_ac_i": ("S_ac", 60, "K_a_ac")},
    "r2": {"S_va_i": ("S_va", 102, "K_a_va"), "S_bu_i": ("S_bu", 88, "K_a_bu"),
           "S_pro_i": ("S_pro", 74, "K_a_pro"), "S_ac_i": ("S_ac", 60, "K_a_ac")},
}
START_T = 310.15
START_PH = 7.8
#: A made-up start state of the R3/R2 kind. Its own chemistry is far off the measured pH
#: (about pH 12), as a calibrated file's can be. R2 adds its acids.
START_STATE = {"S_ac": 0.5, "S_va": 0.02, "S_bu": 0.03, "S_pro": 0.05, "S_ch4": 0.01,
               "S_IC": 5.0, "S_TAN": 2.0, "S_h2o": 900.0, "X_ch": 10.0, "X_pr": 0.1,
               "X_li": 0.1, "X_bac": 0.5, "X_va": 0.2, "X_bu": 0.2, "X_pro": 0.3,
               "X_ac": 1.0, "S_an": 0.0, "S_cat": 0.3, "S_va_i": 0.02, "S_bu_i": 0.0,
               "S_pro_i": 0.01, "S_ac_i": 0.01, "S_hco3_i": 5.0, "S_nh3": 0.1,
               "S_gas_ch4": 0.4, "S_gas_co2": 0.5}


def _van_t_hoff(enthalpy, temperature=START_T):
    return np.exp(-enthalpy / 8.3145 * (1 / temperature - 1 / 298.15))


def _start_state(key):
    return jnp.asarray([START_STATE[name] for name in LAYOUTS[key].names])


def _measured_chemistry(key, y=None):
    y = _start_state(key) if y is None else y
    params = {name: jnp.asarray(value) for name, value in DISSOCIATION.items()}
    return np.asarray(initial_state.measured_chemistry(LAYOUTS[key], y, params,
                                                       jnp.asarray(START_T), jnp.asarray(START_PH)))


def _charge_balance_ph(key, y):
    """The pH the models compute from a state: S_H^2 + phi S_H - K_w = 0."""
    s = dict(zip(LAYOUTS[key].names, y))
    phi = s["S_cat"] - s["S_an"] + (s["S_TAN"] - s["S_nh3"]) / 17 - s["S_hco3_i"] / 44
    for ion, (_, molar_mass, _) in ACIDS[key].items():
        phi -= s[ion] / molar_mass
    K_w = DISSOCIATION["K_w"] * _van_t_hoff(55900)
    return -np.log10(-phi / 2 + np.sqrt(phi**2 + 4 * K_w) / 2)


@pytest.mark.parametrize("key", ["r3", "r2"])
def test_the_measured_chemistry_reproduces_the_ph_through_the_charge_balance(key):
    assert _charge_balance_ph(key, _start_state(key)) > 11      # what the state's own chemistry gives
    assert _charge_balance_ph(key, _measured_chemistry(key)) == pytest.approx(START_PH, abs=1e-9)


@pytest.mark.parametrize("key", ["r3", "r2"])
def test_the_measured_chemistry_starts_the_co2_in_equilibrium_with_the_head_space(key):
    s = dict(zip(LAYOUTS[key].names, _measured_chemistry(key)))
    _, to_gas = gas_transfer_rates(0.0, s["S_IC"] - s["S_hco3_i"], s["S_gas_ch4"],
                                   s["S_gas_co2"], START_T, 200.0)
    assert float(to_gas) == pytest.approx(0.0, abs=1e-10)
    # the alkalinity the measured pH and head space imply: more than the state's own 5.0
    assert 8.0 < s["S_IC"] < 12.0


@pytest.mark.parametrize("key", ["r3", "r2"])
def test_the_measured_chemistry_puts_every_acid_base_pair_in_equilibrium(key):
    s = dict(zip(LAYOUTS[key].names, _measured_chemistry(key)))
    S_H = 10.0 ** -START_PH
    K_co2 = DISSOCIATION["K_a_co2"] * _van_t_hoff(7646)
    K_TAN = DISSOCIATION["K_a_TAN"] * _van_t_hoff(51965)
    assert s["S_hco3_i"] == pytest.approx(K_co2 / (K_co2 + S_H) * s["S_IC"], rel=1e-12)
    assert s["S_nh3"] == pytest.approx(K_TAN / (K_TAN + S_H) * s["S_TAN"], rel=1e-12)
    for ion, (total, _, constant) in ACIDS[key].items():
        K_a = DISSOCIATION[constant]
        assert s[ion] == pytest.approx(K_a / (K_a + S_H) * s[total], rel=1e-12), ion


@pytest.mark.parametrize("key", ["r3", "r2"])
def test_the_measured_chemistry_does_not_read_the_calibrated_chemistry(key):
    names = LAYOUTS[key].names
    chemistry = [names.index(name) for name in ("S_IC", "S_an", "S_cat", "S_hco3_i", "S_nh3",
                                                *ACIDS[key])]
    y = _start_state(key)
    scrambled = y.at[jnp.asarray(chemistry)].set(3 * y[jnp.asarray(chemistry)] + 1)
    np.testing.assert_array_equal(_measured_chemistry(key, scrambled), _measured_chemistry(key, y))
    # and leaves everything else as it was
    rest = [i for i in range(len(names)) if i not in chemistry]
    np.testing.assert_array_equal(_measured_chemistry(key)[rest], np.asarray(y)[rest])


# -- the calibrated start the datasets hand over ---------------------------------------
@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r3", "r2"])
@pytest.mark.parametrize("label", ["LaRA-temp_F4", "LaRA-substrate_F2", "rice-bran_R60"])
def test_the_calibrated_start_reproduces_the_first_measured_ph(key, label, sample):
    # A calibrated file's own chemistry can start far off the measured pH.
    x, y = sample(key, label)
    out, _ = get_spec(key).mechanistic()(x["parameters"], x["initial_values"],
                                         first_days(x["input"], 5.0))
    measured = float(y["values"][0, y["object_indices"]["pH"]])
    assert float(out["pH"][0]) == pytest.approx(measured, abs=1e-8)


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2"])
@pytest.mark.parametrize("label", ["LaRA-temp_F4", "LaRA-substrate_F3", "rice-bran_R30"])
def test_the_dissolved_gases_start_in_equilibrium_with_the_gas_phase(key, label, sample):
    x, _ = sample(key, label)
    s = dict(zip(LAYOUTS[key].names, np.asarray(x["initial_values"])))
    co2 = s["S_IC"] if key == "r4" else s["S_IC"] - s["S_hco3_i"]   # what each transfers
    forcing = dx.CubicInterpolation(x["input"]["time"], x["input"]["coeffs"])
    T = float(forcing.evaluate(x["input"]["time"][0])[0])
    to_gas = gas_transfer_rates(s["S_ch4"], co2, s["S_gas_ch4"], s["S_gas_co2"], T, 200.0)
    np.testing.assert_allclose(np.asarray(to_gas, dtype=float), 0.0, atol=1e-10)


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r3", "r2"])
@pytest.mark.parametrize("label", ["LaRA-temp_F4", "LaRA-substrate_F2", "rice-bran_R60"])
def test_the_calibrated_start_holds_its_acid_base_pairs_at_rest(key, label, sample):
    """The right-hand side moves no ionized species at the start: each sits at its
    equilibrium for the pH the start's charge balance gives, at the start's temperature."""
    x, _ = sample(key, label)
    model = get_spec(key).mechanistic()
    forcing = dx.CubicInterpolation(x["input"]["time"], x["input"]["coeffs"])
    dy = np.asarray(model._rhs(x["input"]["time"][0], x["initial_values"], (x["parameters"], forcing)))
    names = LAYOUTS[key].names
    for ion in ("S_hco3_i", "S_nh3", *ACIDS[key]):
        # With a calibrated file's own chemistry these would move fast (k_AB = 1e10).
        # R2's right-hand side floors its states
        # at 1e-12, which puts the ion that is zero into the charge balance at 1e-12 mol/L:
        # up to ~1e-3 g/L/d of bicarbonate around pH 7.5.
        assert abs(dy[names.index(ion)]) < 1e-2, ion


# -- the warm-up itself ----------------------------------------------------------------
@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3"])
def test_the_warm_up_ends_in_a_steady_state(key, sample):
    x, _ = sample(key)
    model, forcing, state = _raw_warm_up(key, x)
    layout = LAYOUTS[key]
    dy = np.asarray(model._rhs(0.0, jnp.asarray(state), (x["parameters"], forcing)))
    free = [i for i, name in enumerate(layout.names) if name not in layout.held]
    relative = np.abs(dy[free]) / np.maximum(np.abs(state[free]), 1e-3)
    assert np.max(relative) < 1e-3, layout.names[free[int(np.argmax(relative))]]


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3"])
def test_the_warm_up_holds_tan_and_water_at_their_first_measurement(key, sample):
    x, _ = sample(key)
    _, _, state = _raw_warm_up(key, x)
    layout = LAYOUTS[key]
    for name in ("S_TAN", "S_h2o"):
        assert state[layout.index(name)] == float(x["initial_values"][layout.index(name)]), name


@pytest.mark.requires_data
def test_the_warm_up_runs_at_the_first_measured_ph(sample):
    """The inorganic carbon ends up speciated at the measured pH, which it only
    does if the right-hand side saw that pH throughout the warm-up."""
    x, _ = sample("r3")
    _, forcing, state = _raw_warm_up("r3", x)
    temperature = float(forcing.value[0])
    K_a = float(x["parameters"]["K_a_co2"]) * np.exp(-7646 / 8.3145 * (1 / temperature - 1 / 298.15))
    S_H = 10 ** -float(x["input"]["start"]["pH"])
    layout = LAYOUTS["r3"]
    fraction = state[layout.index("S_hco3_i")] / state[layout.index("S_IC")]
    assert fraction == pytest.approx(K_a / (K_a + S_H), rel=1e-6)


# -- the start it hands to the simulation ----------------------------------------------
@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2"])
def test_the_warm_start_does_not_read_the_calibrated_states(key, sample):
    x, _ = sample(key)
    measured = x["input"]["start"]["measured"]
    scrambled = {**x, "initial_values": jnp.where(measured > 0, x["initial_values"],
                                                  3 * x["initial_values"] + 1)}
    simulator = get_spec(key).mechanistic(initial_state="warmup")
    np.testing.assert_array_equal(_start_of_simulation(simulator, x),
                                  _start_of_simulation(simulator, scrambled))


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2"])
def test_the_warm_start_keeps_the_measured_states_and_the_gas_phase(key, sample):
    x, _ = sample(key)
    start = _start_of_simulation(get_spec(key).mechanistic(initial_state="warmup"), x)
    measured = np.asarray(x["input"]["start"]["measured"]) > 0
    np.testing.assert_array_equal(start[measured], np.asarray(x["initial_values"])[measured])
    assert not np.array_equal(start[~measured], np.asarray(x["initial_values"])[~measured])


@pytest.mark.requires_data
def test_the_calibrated_start_is_the_datasets_state(sample):
    x, _ = sample("r4")
    for simulator in (get_spec("r4").mechanistic(), get_spec("r4").mechanistic(initial_state="calibrated")):
        np.testing.assert_array_equal(_start_of_simulation(simulator, x), np.asarray(x["initial_values"]))


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r3", "r2"])
def test_the_warm_start_reproduces_the_first_measured_ph(key, sample):
    x, y = sample(key)
    out, _ = get_spec(key).mechanistic(initial_state="warmup")(
        x["parameters"], x["initial_values"], first_days(x["input"], 5.0))
    measured = float(y["values"][0, y["object_indices"]["pH"]])
    assert float(out["pH"][0]) == pytest.approx(measured, abs=1e-8)


@pytest.mark.requires_data
def test_a_warm_up_without_a_measured_ph_fails_loudly(sample):
    x, _ = sample("r3")
    start = {**x["input"]["start"], "pH": jnp.asarray(jnp.nan)}
    unmeasured = {**x, "input": {**x["input"], "start": start}}
    with pytest.raises(Exception, match="measured pH"):
        _start_of_simulation(get_spec("r3").mechanistic(initial_state="warmup"), unmeasured)


# -- the hybrid ------------------------------------------------------------------------
def _hybrid(spec, dataset, seed):
    ude = spec.ude(DEFAULT, initial_state="warmup")
    return ude, correction_mlp(spec, spec.nn_in_size(dataset), DEFAULT, seed=seed)


@pytest.mark.requires_data
def test_the_hybrid_warms_up_with_its_own_network(sample, installed_dataset):
    """Its start is its own steady state, so it moves with the network, and the
    training gradient reaches the network through it."""
    spec = get_spec("r4")
    dataset = installed_dataset("r4")
    x, _ = sample("r4")
    ude, mlp = _hybrid(spec, dataset, 0)
    _, other = _hybrid(spec, dataset, 1)
    key = jr.PRNGKey(2)
    unmeasured = np.asarray(x["input"]["start"]["measured"]) == 0
    first = _start_of_simulation(ude, x, network=(mlp, key))
    second = _start_of_simulation(ude, x, network=(other, key))
    assert not np.allclose(first[unmeasured], second[unmeasured])

    def unmeasured_start(network):
        _, sol = ude(x["parameters"], x["initial_values"], first_days(x["input"], 5.0), network, key)
        return jnp.sum(sol.ys[0][jnp.asarray(unmeasured)])

    gradient = eqx.filter_grad(unmeasured_start)(mlp)
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(gradient)]
    assert all(np.all(np.isfinite(leaf)) for leaf in leaves)
    assert any(np.any(leaf != 0) for leaf in leaves)


@pytest.mark.requires_data
def test_a_training_batch_warms_up_each_reactor_from_its_own_start(sample, installed_dataset):
    spec = get_spec("r4")
    dataset = installed_dataset("r4")
    ude, mlp = _hybrid(spec, dataset, 0)
    key = jr.PRNGKey(2)
    samples = [sample("r4", label) for label in ("LaRA-temp_F4", "LaRA-substrate_F2")]
    short = [({**x, "input": first_days(x["input"], 5.0)}, y) for x, y in samples]
    batch, _ = custom_collate_fn(short)
    _, sol = eqx.filter_vmap(ude, in_axes=(0, 0, 0, None, None))(
        batch["parameters"], batch["initial_values"], batch["input"], mlp, key)
    for row, (x, _) in enumerate(short):
        alone = _start_of_simulation(ude, x, network=(mlp, key))
        np.testing.assert_allclose(np.asarray(sol.ys[row, 0]), alone, rtol=1e-10, atol=1e-12)
