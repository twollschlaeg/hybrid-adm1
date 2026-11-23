"""Temperature dependence of the reduced models' kinetics (``arrhenius``).

ADM1-R2, -R3 and -R4 dropped the Arrhenius factors the full ADM1 applies, so their
biology does not respond to temperature. After the LaRA-temp temperature steps the
measured biogas rose for a while and came back, which kinetics without temperature
dependence cannot follow. ``arrhenius`` scales every rate that has an activation energy
(hydrolysis with the sugar, amino-acid and fatty-acid energies, acid uptake with its own;
Weinrich 2017) by
exp(-E/R (1/T - 1/T_ref)), T_ref = 35 C, so calibrated rates are rates at 35 C. The
mechanistic models and the hybrids' ``physics: arrhenius`` scale the same rates by the
same factors: a calibrated hybrid at zero output is its mechanistic model.
"""

from __future__ import annotations

import diffrax as dx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import EXPERIMENTS

from adm1_package.experiments.registry import MODEL_KEYS, get_spec
from adm1_package.experiments.training import load_experiment
from adm1_package.utils.corrections import (
    CATALOGUE,
    CorrectionOptions,
    activation_energies,
    arrhenius_factor,
    layout_for,
    with_arrhenius,
)
from adm1_package.utils.nn_architectures import CorrectionMLP

#: Illustrative values of R4's corrected parameters.
PARAMS = {"f_ch": 0.7, "f_pr": 0.8, "f_li": 0.95, "k_hyd_ch": 0.05, "k_m_ac": 0.3,
          "K_I_nh3": 0.03}
SCALED = {
    "r4": {"k_hyd_ch", "k_hyd_pr", "k_hyd_li"},
    "r3": {"k_hyd_ch", "k_hyd_pr", "k_hyd_li", "k_m_ac"},
    "r2": {"k_hyd_ch", "k_hyd_pr", "k_hyd_li", "k_m_va", "k_m_bu", "k_m_pro", "k_m_ac"},
}


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_the_rates_with_an_activation_energy_are_the_hybrids(model_key):
    energies = activation_energies(model_key)
    assert set(energies) == SCALED[model_key]
    catalogue = {t.name: t.activation_energy for t in CATALOGUE[model_key]}
    assert energies == {name: catalogue[name] for name in energies}


def test_at_the_reference_temperature_nothing_changes():
    params = {"k_hyd_ch": 0.1, "k_hyd_pr": 2.0, "k_dec_Xbac": 0.02, "V_liq": 10.0}
    scaled = with_arrhenius(params, 308.15, "r4")
    assert {name: float(value) for name, value in scaled.items()} == pytest.approx(params)


def test_only_the_rates_with_an_activation_energy_are_scaled():
    params = {"k_hyd_ch": 0.1, "k_dec_Xbac": 0.02, "V_liq": 10.0}
    scaled = with_arrhenius(params, 318.15, "r4")
    energy = activation_energies("r4")["k_hyd_ch"]
    assert float(scaled["k_hyd_ch"]) == pytest.approx(0.1 * float(arrhenius_factor(318.15, energy)))
    assert scaled["k_dec_Xbac"] == 0.02 and scaled["V_liq"] == 10.0
    assert params["k_hyd_ch"] == 0.1  # the caller's parameters are left alone


def test_arrhenius_speeds_rates_up_with_temperature():
    """Acetoclastic growth roughly doubles from 35 to 55 C with the Weinrich energy."""
    assert float(arrhenius_factor(328.15, 27362.37923619)) == pytest.approx(1.92, abs=0.01)


def test_the_hybrids_scale_pruned_rates_too():
    """Temperature dependence is physics, independent of whether the network corrects."""
    layout = layout_for("r4", CorrectionOptions(prune=True, physics=("arrhenius",)))
    cold, hot = (layout.resolve(jnp.zeros(layout.size), PARAMS, temperature)["k_hyd_pr"]
                 for temperature in (308.15, 328.15))
    assert float(hot) > float(cold)


def test_decay_rates_carry_no_temperature_dependence():
    """The full ADM1 applies Arrhenius to growth only."""
    layout = layout_for("r4", CorrectionOptions(prune=False, physics=("arrhenius",)))
    cold, hot = (layout.resolve(jnp.zeros(layout.size), PARAMS, temperature)["k_dec_Xbac"]
                 for temperature in (308.15, 328.15))
    assert float(hot) == pytest.approx(float(cold))


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_the_mechanistic_models_are_temperature_independent_unless_asked(model_key):
    spec = get_spec(model_key)
    assert not spec.mechanistic().options.get("arrhenius", False)
    assert spec.mechanistic(arrhenius=True).options["arrhenius"] is True


@pytest.mark.requires_data
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_the_mechanistic_right_hand_side_scales_the_rates(model_key, installed_dataset):
    spec = get_spec(model_key)
    dataset = installed_dataset(model_key, ["LaRA-temp"])
    x, _ = dataset[0]
    interp = dx.CubicInterpolation(x["input"]["time"], x["input"]["coeffs"])
    t = 50.0
    temperature = interp.evaluate(t)[0]
    assert abs(float(temperature) - 308.15) > 1.0  # else the factors would all be ~1
    plain, warm = spec.mechanistic(), spec.mechanistic(arrhenius=True)
    y = x["initial_values"]
    expected = plain._rhs(t, y, (with_arrhenius(x["parameters"], temperature, model_key), interp))
    np.testing.assert_allclose(np.asarray(warm._rhs(t, y, (x["parameters"], interp))),
                               np.asarray(expected), rtol=1e-12, atol=1e-15)
    assert not np.allclose(np.asarray(plain._rhs(t, y, (x["parameters"], interp))),
                           np.asarray(expected), rtol=1e-9)


def _tan_arm():
    """The correction of the TAN arm, as its experiment file sets it."""
    return CorrectionOptions.from_dict(
        load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json")["correction"])


@pytest.mark.requires_data
@pytest.mark.parametrize("options", [
    CorrectionOptions("calibrated", exact_anchors=True, physics=("arrhenius",)),
    pytest.param("tan arm", id="tan-arm"),
])
def test_a_calibrated_hybrid_at_zero_output_is_its_mechanistic_model_with_arrhenius(short_r4_samples,
                                                                                   options):
    """Also the TAN arm, whose influent TAN factor starts at 1."""
    dataset, samples = short_r4_samples
    x, _ = samples[0]
    spec = get_spec("r4")
    options = _tan_arm() if options == "tan arm" else options
    layout = spec.correction_layout(options)
    width = spec.nn_in_size(dataset)
    mlp = CorrectionMLP(width, layout.size, 8, 2, jax.nn.selu, lambda z: z, True, zero_output=True,
                        factor_inputs=width, factor_outputs=layout.n_factor_outputs,
                        key=jr.PRNGKey(0))
    hybrid, _ = spec.ude(options)(x["parameters"], x["initial_values"], x["input"], mlp, jr.PRNGKey(1))
    mechanistic, _ = spec.mechanistic(arrhenius=True)(x["parameters"], x["initial_values"], x["input"])
    plain, _ = spec.mechanistic()(x["parameters"], x["initial_values"], x["input"])
    # Within the solver's accuracy (rtol 1e-4): roundoff can change its step sequence and
    # move a point by ~1e-6. Without the Arrhenius factor the gas differs by more than 1e-4.
    for name in spec.outputs:
        np.testing.assert_allclose(np.asarray(hybrid[name]), np.asarray(mechanistic[name]),
                                   rtol=1e-5, atol=1e-9, err_msg=name)
    assert not np.allclose(np.asarray(plain["q_gasout"]), np.asarray(mechanistic["q_gasout"]), rtol=1e-4)
