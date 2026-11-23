"""The reported biogas and methane flows.

The campaigns report their gas as norm volumes, dry gas at 0 degC and 1.01325 bar (NmL),
and so do the models: the dry moles leaving the head space per day, at R T_N / P_N =
22.414 L/mol. Read as wet head-space gas at 37 degC the same flow would be 21% larger
for biogas and 14% for methane, and more the warmer the reactor.

The moles come from the head-space mole balance rather than from the overpressure
``k_p (P - p_norm)``. In head-space litres, along any state of the model, the two differ by

    q_pressure - q_mass_balance = -V_gas * (dP/dt) / (P - p_w)          (exact)

so the mass balance drops only the ~2 s lag of the head-space pressure, but never
subtracts two pressures near 1 bar. The model tests check that identity against each
model's own right-hand side, which also proves that the output functions use the same
gas-transfer rates as the ODE.
"""

from __future__ import annotations

import diffrax as dx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import correction_mlp, first_days, splined

from adm1_package.experiments.registry import get_spec
from adm1_package.utils import solver
from adm1_package.utils.corrections import DEFAULT
from adm1_package.utils.gas import biogas_outflow, correct_gas_initial_values, gas_transfer_rates

R = 0.083145
PARAMS = {"k_p": 5e4, "p_norm": 1.013, "V_liq": 8.0, "V_gas": 1.0}
T = 310.0
P_W = 0.06167316492211745       # 0.0313 exp(5290 (1/298.15 - 1/310))
MOLAR_VOLUME = R * 273.15 / 1.01325            # L/mol of dry gas at standard conditions
TO_NORM = 0.96 / 1.01325 * 273.15 / T          # head-space litres -> NL at 0.96 bar dry gas


def _water(temperature):
    return 0.0313 * np.exp(5290 * (1 / 298.15 - 1 / temperature))


def _flows(p_ch4, p_co2, dT_dt=0.0, rho_ch4=0.32, rho_co2=0.88, temperature=T, **hydrogen):
    """Two time points of the same state, so index 0 (the initial state) and index 1 differ
    only in how the helper treats them. ``hydrogen`` is the full ADM1's ``p_h2``, ``rho_h2``."""
    as_series = lambda value: jnp.full(2, value)
    q, ch4 = biogas_outflow(
        as_series(p_ch4), as_series(p_co2), as_series(_water(temperature)),
        as_series(temperature), as_series(dT_dt), as_series(rho_ch4), as_series(rho_co2),
        PARAMS, **{name: as_series(value) for name, value in hydrogen.items()},
    )
    return np.asarray(q), np.asarray(ch4)


def _pressure_law(out, params):
    """The right-hand side's outflow ``k_p (P - p_norm) P / p_norm``, from a model's
    reported pressures, in litres of dry gas at standard conditions."""
    p_norm = float(params["p_norm"])
    head_space = np.maximum(0, float(params["k_p"]) * (out["p_gas"] - p_norm) * out["p_gas"] / p_norm)
    dry = out["p_ch4"] + out["p_co2"]
    norm = head_space * dry / 1.01325 * 273.15 / out["T_op"]
    return norm, norm * out["p_ch4"] / dry


# -- the helper ------------------------------------------------------------------------
def test_outflow_is_the_transfer_into_the_head_space():
    # 0.32 g/L/d CH4 + 0.88 g/L/d CO2 = 0.04 mol/L/d; times 8 L, at 22.414 L/mol. The
    # methane is the head space's share of the dry gas.
    q, ch4 = _flows(0.60, 0.36)
    assert q[1] == pytest.approx(0.32 * MOLAR_VOLUME, rel=1e-12)
    assert ch4[1] == pytest.approx(q[1] * 0.60 / 0.96, rel=1e-12)


def test_the_norm_volume_does_not_depend_on_the_head_space():
    # Wet head-space gas at 55 degC would read 18% more than at 37 degC for the same moles.
    cool, _ = _flows(0.60, 0.36)
    hot, _ = _flows(0.55, 0.40, temperature=328.15)
    assert hot[1] == pytest.approx(cool[1], rel=1e-12)


def test_warming_expels_gas_on_top_of_the_transfer():
    # at 2 K/d the head space loses 5.0e-4 mol/L/d of dry gas by expansion and evaporation
    q, ch4 = _flows(0.60, 0.36, dT_dt=2.0)
    n = 0.96 / (R * T)                                             # mol/L of dry gas
    expelled = (P_W * 5290 / T**2 + n * R) * 2.0 / (R * T)
    assert expelled == pytest.approx(5.0e-4, rel=0.01)
    assert q[1] == pytest.approx((0.32 + PARAMS["V_gas"] * expelled) * MOLAR_VOLUME, rel=1e-12)
    assert ch4[1] == pytest.approx(q[1] * 0.60 / 0.96, rel=1e-12)


def test_the_initial_state_reports_the_pressure_form():
    # The data set the initial gas phase so that the right-hand side's outflow, the pressure
    # law, reproduces the first measurement; an initial state can sit far from
    # equilibrium, where the mass balance is not the outflow yet.
    q, ch4 = _flows(0.60, 0.36)
    P = 0.96 + P_W
    assert q[0] == pytest.approx(5e4 * (P - 1.013) * P / 1.013 * TO_NORM, rel=1e-12)
    assert ch4[0] == pytest.approx(q[0] * 0.60 / 0.96, rel=1e-12)


def test_hydrogen_counts_as_dry_biogas():
    # The full ADM1's head space also holds hydrogen. 0.32 g/L/d CH4 + 0.88 g/L/d CO2 +
    # 0.002 g/L/d H2 = 0.041 mol/L/d, and the methane is its share of the dry gas.
    q, ch4 = _flows(0.60, 0.36, p_h2=1e-4, rho_h2=0.002)
    assert q[1] == pytest.approx(8.0 * 0.041 * MOLAR_VOLUME, rel=1e-12)
    assert ch4[1] == pytest.approx(q[1] * 0.60 / 0.9601, rel=1e-12)
    # and in the pressure law of the initial state
    P = 0.9601 + P_W
    assert q[0] == pytest.approx(5e4 * (P - 1.013) * P / 1.013 * 0.9601 / 1.01325 * 273.15 / T,
                                 rel=1e-12)
    assert ch4[0] == pytest.approx(q[0] * 0.60 / 0.9601, rel=1e-12)


def test_the_initial_state_reports_no_outflow_below_ambient_pressure():
    q, ch4 = _flows(0.60, 0.30)                         # P = 0.962 bar < p_norm
    assert q[0] == 0.0
    assert ch4[0] == 0.0


def test_mass_balance_is_not_switched_off_by_a_pressure_dip():
    # The sign of P - p_norm is as unreliable as its size: at rtol = 1e-4 the computed
    # pressure dips below p_norm on days when the exact one never does.
    q, _ = _flows(0.60, 0.35)                           # P = 1.0117 bar < p_norm
    assert q[1] == pytest.approx(0.32 * MOLAR_VOLUME, rel=1e-12)


class _Forcing:
    def __init__(self, temperature):
        self.temperature = temperature

    def evaluate(self, t):
        return jnp.array([self.temperature, 0.1])


@pytest.mark.parametrize("temperature", [310.0, 328.15])
def test_the_initial_gas_phase_reproduces_the_first_measurement(temperature):
    # a first measurement of, say, 7.0 NL/d of biogas with 4.3 NL/d of methane
    initial = correct_gas_initial_values(jnp.zeros(4), PARAMS, _Forcing(temperature),
                                         jnp.array([[7.0, 4.3, 1.5]]))
    p_ch4 = initial[-2] * R * temperature / 16
    p_co2 = initial[-1] * R * temperature / 44
    q, ch4 = _flows(p_ch4, p_co2, temperature=temperature)
    assert q[0] == pytest.approx(7.0, rel=1e-10)
    assert ch4[0] == pytest.approx(4.3, rel=1e-10)


@pytest.mark.parametrize("temperature", [310.0, 328.15])
def test_the_dissolved_methane_starts_in_equilibrium_with_the_head_space(temperature):
    # A start far above saturation would degas within hours.
    initial = correct_gas_initial_values(jnp.array([9.96, 0.0, 0.0, 0.0]), PARAMS,
                                         _Forcing(temperature), jnp.array([[7.0, 4.3, 1.5]]),
                                         ch4_index=0)
    to_gas, _ = gas_transfer_rates(initial[0], 0.0, initial[-2], initial[-1], temperature, 200.0)
    assert float(to_gas) == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < float(initial[0]) < 0.05


@pytest.mark.parametrize("temperature", [310.0, 328.15])
def test_the_dissolved_co2_starts_in_equilibrium_with_the_head_space(temperature):
    # R4 transfers all of S_IC. A start of several g/L is far above saturation and would
    # degas within minutes.
    initial = correct_gas_initial_values(jnp.array([0.0, 6.2, 0.0, 0.0]), PARAMS,
                                         _Forcing(temperature), jnp.array([[7.0, 4.3, 1.5]]),
                                         co2_index=1)
    _, to_gas = gas_transfer_rates(0.0, initial[1], initial[-2], initial[-1], temperature, 200.0)
    assert float(to_gas) == pytest.approx(0.0, abs=1e-12)
    assert 0.1 < float(initial[1]) < 1.0


def test_no_outflow_while_the_liquid_takes_gas_up():
    q, ch4 = _flows(0.60, 0.36, rho_ch4=-0.32, rho_co2=-0.88)
    assert q[1] == 0.0
    assert ch4[1] == 0.0


def test_gradients_stay_finite_without_gas():
    def total(p):
        q, ch4 = biogas_outflow(p, p, jnp.full(3, P_W), jnp.full(3, T), jnp.zeros(3),
                                jnp.zeros(3), jnp.zeros(3), PARAMS)
        return jnp.sum(q + ch4)

    grad = jax.grad(total)(jnp.zeros(3))
    assert np.all(np.isfinite(np.asarray(grad)))


# -- the models ------------------------------------------------------------------------
MODELS = [(key, kind) for key in ("r4", "r3", "r2") for kind in ("mechanistic", "ude")]
REACTOR = "LaRA-temp_F4"
DAYS = 40.0
def _solve(key, kind, dataset):
    spec = get_spec(key)
    x, _ = dataset[dataset.labels.index(REACTOR)]
    # On a spline: the output is reported at the forcing's daily knots, where the
    # smoothstep is flat and would never exercise the temperature term.
    inputs = splined(first_days(x["input"], DAYS))
    params, y0 = x["parameters"], x["initial_values"]
    interp = dx.CubicInterpolation(inputs["time"], inputs["coeffs"])
    if kind == "mechanistic":
        model = spec.mechanistic()
        _, sol = model(params, y0, inputs)
        return model, sol, (params, interp)
    model = spec.ude(DEFAULT)
    # a random network, so the corrections are active
    mlp = correction_mlp(spec, spec.nn_in_size(dataset), DEFAULT)
    key_ = jr.PRNGKey(1)
    _, sol = model(params, y0, inputs, mlp, key_)
    return model, sol, (params, interp, mlp, key_, (inputs["means"], inputs["stds"]))


@pytest.fixture(scope="module", params=MODELS, ids=[f"{k}-{kind}" for k, kind in MODELS])
def solved(request, installed_dataset):
    key, kind = request.param
    return _solve(key, kind, installed_dataset(key))


def _report(model, sol, args):
    return {name: np.asarray(value) for name, value in model._calculate_output(sol, args).items()}


@pytest.mark.requires_data
def test_reported_biogas_differs_from_the_pressure_form_only_by_the_head_space_lag(solved):
    model, sol, args = solved
    params, interp = args[0], args[1]
    default = _report(model, sol, args)
    q_pressure, ch4_pressure = _pressure_law(default, params)

    # dP/dt from the model's own right-hand side, not from the output function
    dy = np.asarray(jax.vmap(lambda t, y: model._rhs(t, y, args))(sol.ts, sol.ys))
    ys = np.asarray(sol.ys)
    temperature = np.asarray(jax.vmap(interp.evaluate)(sol.ts))[:, 0]
    warming = np.asarray(jax.vmap(interp.derivative)(sol.ts))[:, 0]
    p_w = 0.0313 * np.exp(5290 * (1 / 298.15 - 1 / temperature))
    n = ys[:, -2] / 16 + ys[:, -1] / 44
    dP = R * temperature * (dy[:, -2] / 16 + dy[:, -1] / 44) + (n * R + p_w * 5290 / temperature**2) * warming
    P = n * R * temperature + p_w
    lag = -float(params["V_gas"]) * dP / (P - p_w)

    later = slice(1, None)
    assert np.all(P[later] > float(params["p_norm"])), "the scenario should have gas flowing out"
    assert np.max(np.abs(warming)) > 0.01, "the scenario must exercise the temperature term"
    to_norm = (P - p_w) / 1.01325 * 273.15 / temperature
    np.testing.assert_allclose((q_pressure - default["q_gasout"])[later], (lag * to_norm)[later],
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose((ch4_pressure - default["ch4_out"])[later],
                               (lag * to_norm * default["p_ch4"] / (P - p_w))[later],
                               rtol=0, atol=1e-8)


@pytest.mark.requires_data
def test_reported_biogas_is_robust_to_the_solver_tolerance(installed_dataset, monkeypatch):
    # LaRA-substrate_F4 is where the pressure law is furthest off an accurate solve.
    spec = get_spec("r4")
    dataset = installed_dataset("r4")
    x, y = dataset[dataset.labels.index("LaRA-substrate_F4")]
    model = spec.mechanistic()
    reported = np.asarray(model(x["parameters"], x["initial_values"], x["input"])[0]["q_gasout"])
    monkeypatch.setattr(solver, "RTOL", 1e-10)
    monkeypatch.setattr(solver, "ATOL", 1e-12)
    out = model(x["parameters"], x["initial_values"], x["input"])[0]
    accurate, _ = _pressure_law({name: np.asarray(value) for name, value in out.items()},
                                x["parameters"])

    scored = np.asarray(y["indices"][:, y["object_indices"]["q_gasout"]]) > 0
    scored[0] = False
    error = np.abs(reported - accurate)[scored]
    assert np.max(error) < 3.0
    assert np.percentile(error, 95) < 0.05


@pytest.mark.requires_data
def test_mass_balance_matches_the_pressure_form_on_an_accurate_solution(installed_dataset,
                                                                          monkeypatch):
    monkeypatch.setattr(solver, "RTOL", 1e-10)
    monkeypatch.setattr(solver, "ATOL", 1e-12)
    model, sol, args = _solve("r4", "mechanistic", installed_dataset("r4"))
    default = _report(model, sol, args)
    pressure, _ = _pressure_law(default, args[0])
    gap = np.abs(default["q_gasout"] - pressure)[1:]
    assert np.max(gap) < 1e-3                        # L/d; the head-space lag is ~2 s
    assert np.max(gap / pressure[1:]) < 1e-4
