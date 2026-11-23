"""The full ADM1, ``adm1_package.models.base``.

Its substrates, water and biomass are mass concentrations in g/L, and its head space
holds hydrogen, methane and carbon dioxide in g per litre of gas. So every process in its
Petersen matrix conserves mass: weighted by the volume each state fills, a process's
coefficients sum to zero. The rest of the state vector carries no mass of its own: the
anions and cations (mol/L), and the dissociated shares of the acids, which only the
acid-base processes change.

Its gas is reported as the reduced variants' is (see test_biogas_output.py): norm
volumes, the dry moles leaving the head space per day at 22.414 L/mol, from the
head-space mole balance. The hydrogen counts towards the dry gas.
"""

from __future__ import annotations

import diffrax as dx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import first_days, splined

from adm1_package.experiments.registry import MODEL_KEYS, get_spec
from adm1_package.models.base import ADM1Base, BaseDataset
from adm1_package.models.base.dataset import correct_gas_ivs
from adm1_package.models.base.model import STATES
from adm1_package.utils.gas import gas_transfer_rates, hydrogen_transfer_rate
from adm1_package.utils.initial_state import warm_start

R = 0.083145

#: Illustrative yields. The fixed-coefficient matrix was computed for others (its
#: substrate entries are -1/Y), so the two branches see different yields.
YIELDS = {"Y_su": 0.0734, "Y_aa": 0.0964, "Y_fa": 0.12, "Y_va": 0.09, "Y_bu": 0.0373,
          "Y_pro": 0.0289, "Y_ac": 0.0360, "Y_h2": 0.34}
V_LIQ, V_GAS = 8.0, 1.0

#: S_su ... X_h2: the substrates, water and the biomass, in g per litre of liquid.
LIQUID = slice(0, 23)
#: S_gas_h2, S_gas_ch4, S_gas_co2, in g per litre of head space.
GAS = slice(31, 34)
#: The acid-base processes, which change only the dissociated shares.
ACID_BASE = range(19, 25)
#: The yield branch closes sugar and amino-acid uptake worst: at the fixed branch's yields
#: its S_IC entry for the one and its S_ac entry for the other are 0.0029 and 0.0094 low.
#: Both come from the thesis, whose Table A.8 prints them so; Table A.9 closes.
KNOWN_GAPS = {3: 3.5e-4, 4: 8e-4}


@pytest.mark.parametrize("use_yield_coefficients", [True, False], ids=["yields", "fixed"])
def test_every_process_conserves_mass(use_yield_coefficients):
    model = ADM1Base(options={"use_yield_coefficients": use_yield_coefficients})
    params = {name: jnp.array(value)
              for name, value in {**YIELDS, "V_liq": V_LIQ, "V_gas": V_GAS}.items()}
    petersen = np.asarray(model._petersen_matrix(params))

    volume = np.zeros(petersen.shape[0])
    volume[LIQUID] = V_LIQ
    volume[GAS] = V_GAS
    processes = [j for j in range(petersen.shape[1]) if j not in ACID_BASE]
    mass = volume[:, None] * petersen[:, processes]       # g per unit of each process rate
    gap = np.abs(mass.sum(axis=0)) / -np.minimum(mass, 0).sum(axis=0)
    # Within 2e-4 of the mass a process converts, which a typo such as 0.5777 -> 0.5277
    # or 4.4683 -> 4.4863 exceeds. Hydrolysis without the water it takes up would be 10%
    # (carbohydrates) and 5% (lipids) off.
    np.testing.assert_array_less(gap, [KNOWN_GAPS.get(j, 2e-4) for j in processes])


# -- simulating the reactors -----------------------------------------------------------
REPORTED_OUTPUTS = ["q_gasout", "ch4_out", "S_ac", "S_pro", "S_TAN", "pH"]
REACTOR = "LaRA-temp_F4"


def _length(times):
    """Points before the padded tail (the dataset pads the time grid with its last day)."""
    return int(np.argmax(np.asarray(times))) + 1


@pytest.fixture(scope="module")
def reported(data_dir):
    return BaseDataset(root_dir=str(data_dir), data_to_use=["LaRA-temp"], output=REPORTED_OUTPUTS)


@pytest.fixture(scope="module")
def with_water(data_dir):
    return BaseDataset(root_dir=str(data_dir), data_to_use=["LaRA-temp"],
                       output=REPORTED_OUTPUTS + ["S_h2o"])


@pytest.mark.requires_data
def test_simulates_every_reactor_of_a_campaign(reported):
    # A failure shows as "Terms are not compatible with solver!": diffrax traces the
    # right-hand side before it solves and reports any error raised there that way.
    model = ADM1Base()
    for index, label in enumerate(reported.labels):
        x, _ = reported[index]
        solution, _ = model(x["parameters"], x["initial_values"], x["input"])
        length = _length(x["input"]["time"])
        for name in REPORTED_OUTPUTS:
            assert np.all(np.isfinite(np.asarray(solution[name])[:length])), (label, name)


@pytest.mark.requires_data
def test_the_right_hand_side_prints_nothing_under_vmap(reported, capfd):
    # A check for non-finite values written as a lax.cond would print here: vmap turns it
    # into a select that runs the printing branch on every call, finite states or not, and
    # a batched solve of a campaign stalls.
    x, _ = reported[0]
    interp = dx.CubicInterpolation(x["input"]["time"], x["input"]["coeffs"])
    states = jnp.stack([x["initial_values"], 1.01 * x["initial_values"]])
    dy = jax.vmap(lambda y: ADM1Base()._rhs(0.0, y, (x["parameters"], interp)))(states)
    jax.effects_barrier()
    assert np.all(np.isfinite(np.asarray(dy)))
    assert capfd.readouterr().out == ""


def _rhs_at(sample, state, **parameters):
    """The right-hand side at the start of ``sample``, with some parameters changed."""
    interp = dx.CubicInterpolation(sample["input"]["time"], sample["input"]["coeffs"])
    params = {**sample["parameters"], **{k: jnp.array(v) for k, v in parameters.items()}}
    return np.asarray(ADM1Base()._rhs(0.0, state, (params, interp)))


@pytest.mark.requires_data
def test_butyrate_uptake_competes_with_valerate_not_propionate(reported):
    # Weinrich (2017), thesis Table A.8a: k_m,bu S_bu / (K_S,bu + S_bu) X_bu S_bu / (S_bu + S_va).
    # With S_pro in place of S_va, propionate piling up throttled the uptake of butyrate.
    x, _ = reported[0]
    state = x["initial_values"].at[4].set(0.5)                       # S_bu, g/L
    scarce = _rhs_at(x, state.at[5].set(0.01))                       # S_pro
    piled_up = _rhs_at(x, state.at[5].set(5.0))
    assert piled_up[4] == pytest.approx(scarce[4], rel=1e-12)


@pytest.mark.requires_data
def test_lcfa_uptake_is_inhibited_by_ph_as_acidogenesis_is(reported):
    # Thesis Table A.8d: I_fa = I_pH,aa I_IN,lim I_h2,fa. With I_pH,ac in its place, the
    # acetoclasts' pH limits throttled the uptake of long-chain fatty acids.
    x, _ = reported[0]
    state = x["initial_values"].at[2].set(0.5)                       # S_fa, g/L
    as_calibrated = _rhs_at(x, state)
    acetoclasts_off = _rhs_at(x, state, pK_l_ac=9.0, pK_u_ac=10.0)  # at pH ~8
    assert acetoclasts_off[2] == pytest.approx(as_calibrated[2], rel=1e-12)


@pytest.mark.requires_data
def test_the_water_content_starts_at_the_measurement_and_follows_the_influent(with_water):
    x, y = with_water[with_water.labels.index(REACTOR)]
    solution, _ = ADM1Base()(x["parameters"], x["initial_values"], x["input"])
    length = _length(x["input"]["time"])
    simulated = np.asarray(solution["S_h2o"])[:length]
    measured = np.asarray(y["values"])[:length, y["object_indices"]["S_h2o"]]
    assert simulated[0] == pytest.approx(measured[0], rel=1e-12)
    # Without the water the influent brings it would wash out at q_in / V_liq, a residence
    # time of about 30 days.
    np.testing.assert_array_less(np.abs(simulated / measured - 1), 0.2)


# -- the gas ---------------------------------------------------------------------------
GAS_PARAMS = {"k_p": 5e4, "p_norm": 1.013, "V_liq": 8.0, "V_gas": 1.0}


class _Forcing:
    def __init__(self, temperature):
        self.temperature = temperature

    def evaluate(self, t):
        return jnp.array([self.temperature, 0.1])


def _pressure_law(report, params):
    """The right-hand side's outflow ``k_p (P - p_norm) P / p_norm``, from a report of the
    head space, in litres of dry gas at standard conditions; and its methane."""
    p_norm = float(params["p_norm"])
    head_space = np.maximum(0, float(params["k_p"]) * (report["p_gas"] - p_norm)
                            * report["p_gas"] / p_norm)
    dry = report["p_h2"] + report["p_ch4"] + report["p_co2"]
    norm = head_space * dry / 1.01325 * 273.15 / report["T_op"]
    return norm, norm * report["p_ch4"] / dry


def _head_space(states, temperature):
    """The partial pressures of the gas states, the last three of the state vector."""
    p_h2, p_ch4, p_co2 = (states[i] * R * temperature / molar_mass
                          for i, molar_mass in ((-3, 2), (-2, 16), (-1, 44)))
    p_w = 0.0313 * np.exp(5290 * (1 / 298.15 - 1 / temperature))
    return {"p_h2": p_h2, "p_ch4": p_ch4, "p_co2": p_co2, "p_gas": p_h2 + p_ch4 + p_co2 + p_w,
            "T_op": temperature}


@pytest.mark.parametrize("temperature", [310.0, 328.15])
def test_the_initial_gas_phase_reproduces_the_first_measurement(temperature):
    # A first measurement of, say, 7.0 NL/d of biogas with 4.3 NL/d of methane. The head
    # space starts with 2e-5 bar of hydrogen, 10-12% of the overpressure that drives it out.
    initial = correct_gas_ivs(jnp.zeros(34), GAS_PARAMS, _Forcing(temperature),
                              jnp.array([[7.0, 4.3, 0.2]]))
    q, ch4 = _pressure_law(_head_space(np.asarray(initial), temperature), GAS_PARAMS)
    assert q == pytest.approx(7.0, rel=1e-10)
    assert ch4 == pytest.approx(4.3, rel=1e-10)


@pytest.mark.requires_data
def test_the_first_reported_flows_are_the_first_measurement(reported):
    model = ADM1Base()
    for index, label in enumerate(reported.labels):
        x, y = reported[index]
        solution, _ = model(x["parameters"], x["initial_values"], x["input"])
        for name in ("q_gasout", "ch4_out"):
            measured = float(y["values"][0, y["object_indices"][name]])
            assert float(solution[name][0]) == pytest.approx(measured, rel=1e-8), (label, name)


@pytest.mark.requires_data
def test_reported_biogas_differs_from_the_pressure_form_only_by_the_head_space_lag(reported):
    # In head-space litres q_pressure - q_mass_balance = -V_gas (dP/dt) / (P - p_w) along
    # any state; the hydrogen is part of P and of the moles that leave. On a spline: the
    # output is reported at the forcing's daily knots, where the smoothstep is flat and
    # would never exercise the temperature term.
    x, _ = reported[reported.labels.index(REACTOR)]
    params, inputs = x["parameters"], splined(first_days(x["input"], 40.0))
    model = ADM1Base()
    solution, sol = model(params, x["initial_values"], inputs)
    report = {name: np.asarray(value) for name, value in solution.items()}
    q_pressure, ch4_pressure = _pressure_law(report, params)

    # dP/dt from the model's own right-hand side, not from the output function. lax.map
    # rather than vmap, under which the right-hand side's NaN check prints every state.
    interp = dx.CubicInterpolation(inputs["time"], inputs["coeffs"])
    dy = np.asarray(jax.lax.map(lambda ty: model._rhs(ty[0], ty[1], (params, interp)),
                                (sol.ts, sol.ys)))
    ys = np.asarray(sol.ys)
    temperature = np.asarray(jax.vmap(interp.evaluate)(sol.ts))[:, 0]
    warming = np.asarray(jax.vmap(interp.derivative)(sol.ts))[:, 0]
    p_w = 0.0313 * np.exp(5290 * (1 / 298.15 - 1 / temperature))
    n = ys[:, -3] / 2 + ys[:, -2] / 16 + ys[:, -1] / 44
    dn = dy[:, -3] / 2 + dy[:, -2] / 16 + dy[:, -1] / 44
    dP = R * temperature * dn + (n * R + p_w * 5290 / temperature**2) * warming
    P = n * R * temperature + p_w
    lag = -float(params["V_gas"]) * dP / (P - p_w)

    later = slice(1, None)
    assert np.all(P[later] > float(params["p_norm"])), "the scenario should have gas flowing out"
    assert np.max(np.abs(warming)) > 0.01, "the scenario must exercise the temperature term"
    to_norm = (P - p_w) / 1.01325 * 273.15 / temperature
    np.testing.assert_allclose((q_pressure - report["q_gasout"])[later], (lag * to_norm)[later],
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose((ch4_pressure - report["ch4_out"])[later],
                               (lag * to_norm * report["p_ch4"] / (P - p_w))[later],
                               rtol=0, atol=1e-8)


# -- the calibration ---------------------------------------------------------------------
def test_the_registry_builds_the_full_adm1_outside_the_study():
    spec = get_spec("base")
    assert isinstance(spec.mechanistic(initial_state="calibrated"), ADM1Base)
    assert "base" not in MODEL_KEYS               # never trained, tuned or scored
    # calibrated after R2, on the same objectives
    assert spec.objective_list == get_spec("r2").objective_list


@pytest.mark.requires_data
def test_the_start_takes_its_chemistry_from_the_first_measurements(reported):
    # As R2 and R3 start: at the measured pH, with nothing to exchange with the head space,
    # whatever the calibrated file held.
    model = ADM1Base()
    for index, label in enumerate(reported.labels):
        x, y = reported[index]
        start = np.asarray(x["initial_values"])
        inputs = first_days(x["input"], 2.0)
        interp = dx.CubicInterpolation(inputs["time"], inputs["coeffs"])
        temperature = float(interp.evaluate(inputs["time"][0])[0])
        to_gas = [*gas_transfer_rates(start[8], start[9] - start[29], start[32], start[33],
                                      temperature, 1.0),
                  hydrogen_transfer_rate(start[7], start[31], temperature, 1.0)]
        np.testing.assert_allclose(np.asarray(to_gas, dtype=float), 0.0, atol=1e-12, err_msg=label)
        solution, _ = model(x["parameters"], x["initial_values"], inputs)
        measured_pH = float(y["values"][0, y["object_indices"]["pH"]])
        assert float(solution["pH"][0]) == pytest.approx(measured_pH, rel=1e-9), label
        # the warm-up behind the calibration's particulate caps keeps what the data set
        marked = {STATES.names[i] for i in np.flatnonzero(np.asarray(x["input"]["start"]["measured"]))}
        assert {"S_gas_h2", "S_gas_ch4", "S_gas_co2", "S_h2", "S_ch4"} <= marked, label


@pytest.mark.requires_data
def test_a_warmup_start_is_the_models_own_first_week_steady_state(reported):
    # as R2-R4 start with initial_state="warmup"
    x, _ = reported[reported.labels.index(REACTOR)]
    inputs = first_days(x["input"], 10.0)            # the warm-up reads the first week
    model = get_spec("base").mechanistic(initial_state="warmup")
    expected = warm_start(STATES, model._rhs, lambda forcing: (x["parameters"], forcing),
                          x["parameters"], x["initial_values"], inputs)
    _, sol = model(x["parameters"], x["initial_values"], inputs)
    np.testing.assert_allclose(np.asarray(sol.ys[0]), np.asarray(expected), rtol=1e-10)
    assert not np.allclose(np.asarray(sol.ys[0]), np.asarray(x["initial_values"]))
