"""The Monod half-saturation constants of R2 and R3.

The network does not correct them. They are resolved with the other parameters:
the campaign's own where its parameter file has them, else the literature.
"""

from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import correction_mlp

from adm1_package.experiments.registry import get_spec
from adm1_package.utils.corrections import DEFAULT, CorrectionOptions, layout_for

#: The half-saturation constants each variant's uptake rates use.
USED = {"r3": ("K_ac",), "r2": ("K_va", "K_bu", "K_pro", "K_ac")}
#: Made-up values of two campaigns' parameter files.
ONE_CAMPAIGN = {"K_ac": 1.2, "K_pro": 0.2, "K_va": 0.15, "K_bu": 0.1}
OTHER_CAMPAIGN = {"K_ac": 0.35, "K_pro": 0.07, "K_va": 0.1, "K_bu": 0.12}
#: ADM1 defaults (Batstone et al. 2002) in kg COD/m3, and g COD per g of acid.
ADM1_COD = {"K_ac": 0.15, "K_pro": 0.1, "K_bu": 0.2, "K_va": 0.2}
COD_PER_GRAM = {"K_ac": 64 / 60.05, "K_pro": 112 / 74.08, "K_bu": 160 / 88.11, "K_va": 208 / 102.13}


def _resolve(model_key, options, params):
    layout = layout_for(model_key, options)
    return layout.resolve(jnp.zeros(layout.size), params, 310.0)


@pytest.mark.parametrize("model_key", ["r2", "r3"])
def test_each_campaign_keeps_its_own_half_saturation_constants(model_key):
    """Calibrated, the default, keeps them at the campaign's values."""
    one = _resolve(model_key, DEFAULT, ONE_CAMPAIGN)
    other = _resolve(model_key, DEFAULT, OTHER_CAMPAIGN)
    for name in USED[model_key]:
        assert (float(one[name]) == ONE_CAMPAIGN[name]
                and float(other[name]) == OTHER_CAMPAIGN[name]), name


@pytest.mark.parametrize("model_key", ["r2", "r3"])
def test_a_campaign_without_them_gets_the_adm1_defaults_in_grams_of_acid(model_key):
    values = _resolve(model_key, DEFAULT, {})
    for name in USED[model_key]:
        expected = ADM1_COD[name] / COD_PER_GRAM[name]
        assert float(values[name]) == pytest.approx(expected, rel=5e-3), name


def test_a_reference_override_sets_a_half_saturation_constant_the_campaign_lacks():
    options = CorrectionOptions(references=(("K_ac", 0.25),))
    assert float(_resolve("r3", options, {})["K_ac"]) == 0.25
    assert float(_resolve("r3", options, ONE_CAMPAIGN)["K_ac"]) == ONE_CAMPAIGN["K_ac"]


@pytest.mark.requires_data
@pytest.mark.parametrize("model_key", ["r2", "r3"])
def test_the_ude_takes_its_half_saturation_constants_from_the_campaign(model_key,
                                                                      installed_dataset):
    spec = get_spec(model_key)
    dataset = installed_dataset(model_key, ["LaRA-temp"])
    x, _ = dataset[0]
    ude = spec.ude(DEFAULT)
    mlp = correction_mlp(spec, spec.nn_in_size(dataset), DEFAULT, width=8, depth=1)
    interp = dx.CubicInterpolation(x["input"]["time"], x["input"]["coeffs"])
    scaling = (x["input"]["means"], x["input"]["stds"])

    def rhs(params):
        args = (params, interp, mlp, jr.PRNGKey(1), scaling)
        return np.asarray(ude._rhs(10.0, x["initial_values"], args))

    tripled = {**x["parameters"], **{name: 3 * x["parameters"][name] for name in USED[model_key]}}
    assert not np.array_equal(rhs(x["parameters"]), rhs(tripled))
