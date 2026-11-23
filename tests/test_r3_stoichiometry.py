"""The stoichiometry of ADM1-R3 and its hybrid.

The mass-based ADM1-R3 (Weinrich, 2017, thesis Table A.12a) has six biochemical
processes: the hydrolysis of X_ch, X_pr and X_li, the uptake of acetate by the
acetoclastic methanogens X_ac, and the decay of X_bac and of X_ac. Each one is a
closed mass balance over the ten components it converts. Decayed biomass, X_bac
and X_ac alike, returns to the particulates as 0.19 X_ch, 0.77 X_pr and 0.04 X_li;
the uptake of acetate grows X_ac and makes no particulates.

The tests evaluate the right-hand side where a single process runs: every other
rate constant, the in- and outflow and the gas exchange are zero, so ``dy`` is
that process's column of the Petersen matrix times its rate.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import correction_mlp

from adm1_package.experiments.registry import get_spec
from adm1_package.models.r3.model import STATES
from adm1_package.utils.corrections import CorrectionOptions
from adm1_package.utils.initial_state import ConstantForcing

KINDS = ("mechanistic", "ude")
#: What the biochemical processes convert into one another, all in g/L.
CONVERTED = ("S_ac", "S_ch4", "S_IC", "S_TAN", "S_h2o", "X_ch", "X_pr", "X_li", "X_bac", "X_ac")
PARTICULATES = ("X_ch", "X_pr", "X_li")
#: The rate constant of each biochemical process.
PROCESSES = {
    "hydrolysis of X_ch": "k_hyd_ch",
    "hydrolysis of X_pr": "k_hyd_pr",
    "hydrolysis of X_li": "k_hyd_li",
    "acetate uptake": "k_m_ac",
    "decay of X_bac": "k_dec_Xbac",
    "decay of X_ac": "k_dec_Xac",
}

#: Illustrative values, with every process switched off: no reaction, dissociation, gas
#: transfer or gas outflow.
PARAMS = {
    **{rate: 0.0 for rate in PROCESSES.values()},
    "k_AB_ac": 0.0, "k_AB_co2": 0.0, "k_AB_TAN": 0.0, "k_La": 0.0, "k_p": 0.0,
    "K_ac": 1.0, "K_I_TAN": 0.002, "K_I_nh3": 0.03, "n_ac": 3.0, "pK_l_ac": 6.0, "pK_u_ac": 7.0,
    "K_a_ac": 1.74e-5, "K_a_co2": 4.47e-7, "K_a_TAN": 5.62e-10, "K_w": 1e-13,
    "p_norm": 1.013, "V_liq": 8.0, "V_gas": 1.0, "f_ch": 1.0, "f_pr": 0.5, "f_li": 0.9,
}
#: An illustrative start, with some of both biomasses.
STATE = {
    "S_ac": 0.5, "S_ch4": 0.01, "S_IC": 9.0, "S_TAN": 2.0, "S_h2o": 950.0,
    "X_ch": 10.0, "X_pr": 1.0, "X_li": 2.0, "X_bac": 1.0, "X_ac": 0.5,
    "S_an": 0.0, "S_cat": 0.1, "S_ac_i": 0.5, "S_hco3_i": 9.0, "S_nh3": 0.1,
    "S_gas_ch4": 0.4, "S_gas_co2": 0.5,
}
#: 37 degC and no feed, then [S_ac, S_TAN, S_h2o, X_ch, X_pr, X_li] of the influent.
FORCING = ConstantForcing(jnp.array([310.15, 0.0, 1.0, 3.0, 900.0, 40.0, 10.0, 5.0]))
#: The network multiplies each calibrated rate constant, so one that is zero
#: stays zero whatever the network says.
CALIBRATED = CorrectionOptions("calibrated")


@functools.lru_cache(maxsize=None)
def _only(kind: str, process: str) -> dict[str, float]:
    """``dy`` per component when ``process`` is the only one running."""
    spec = get_spec("r3")
    params = {name: jnp.asarray(value)
              for name, value in {**PARAMS, PROCESSES[process]: 0.05}.items()}
    y = jnp.array([STATE[name] for name in STATES.names])
    if kind == "mechanistic":
        dy = spec.mechanistic()._rhs(0.0, y, (params, FORCING))
    else:
        width = FORCING.value.shape[0]
        mlp = correction_mlp(spec, width, CALIBRATED)
        scaling = (jnp.zeros(width), FORCING.value + 1.0)
        args = (params, FORCING, mlp, jr.PRNGKey(1), scaling)
        dy = spec.ude(CALIBRATED)._rhs(0.0, y, args)
    return dict(zip(STATES.names, np.asarray(dy).tolist()))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("process", PROCESSES)
def test_every_biochemical_process_conserves_mass(kind, process):
    dy = _only(kind, process)
    turnover = max(abs(dy[name]) for name in CONVERTED)
    assert turnover > 0, "the process should run"
    assert sum(dy[name] for name in CONVERTED) == pytest.approx(0.0, abs=1e-12 * turnover)


@pytest.mark.parametrize("kind", KINDS)
def test_decayed_acetate_degraders_return_their_mass_as_particulates(kind):
    dy = _only(kind, "decay of X_ac")
    assert dy["X_ac"] < 0
    assert sum(dy[name] for name in PARTICULATES) == pytest.approx(-dy["X_ac"], rel=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_both_biomasses_decay_into_the_same_particulates(kind):
    def shares(biomass):
        dy = _only(kind, f"decay of {biomass}")
        return [dy[name] / -dy[biomass] for name in PARTICULATES]

    np.testing.assert_allclose(shares("X_ac"), shares("X_bac"), rtol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_acetate_uptake_makes_no_particulates(kind):
    dy = _only(kind, "acetate uptake")
    assert dy["X_ac"] > 0, "the acetoclastic methanogens should grow"
    assert [dy[name] for name in PARTICULATES] == [0.0, 0.0, 0.0]
