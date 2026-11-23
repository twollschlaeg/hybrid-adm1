"""The physical chemistry every ADM1 variant shares: gas law, Henry and dissociation
constants, the vapour pressure of water, the pH inhibition and the head-space outflow.

Each helper returns the expression the right-hand sides have always written, operation
for operation, so naming a constant here does not change a single bit of a result: a
literal such as ``1.4 * 10**(-3)`` is kept as that product, not folded into ``1.4e-3``,
and every sum and product keeps its order. The Henry constants live in
:mod:`adm1_package.utils.gas`, which the datasets share.

Each variant keeps its own state vector, rates and Petersen matrix in its ``model.py``;
what differs between variants (the clip bounds, which constants are corrected) is set
there.
"""

from __future__ import annotations

import jax.numpy as jnp

from adm1_package.utils.gas import R_GAS, henry_ch4, henry_co2, henry_h2

__all__ = [
    "R_GAS", "T_REF", "KELVIN", "M_H2", "M_CH4", "M_CO2",
    "P_H2O_REF", "H2O_VAN_T_HOFF", "DH_CO2", "DH_TAN", "DH_W",
    "C4_COMPETITION", "DECAY_YIELDS",
    "henry_h2", "henry_ch4", "henry_co2", "water_vapour_pressure", "van_t_hoff",
    "partial_pressure", "proton_concentration", "ph_inhibition", "pressure_law_outflow",
    "P_GAS_BOUNDS", "Q_GAS_BOUNDS", "S_H_BOUNDS",
]

#: Reference temperature of the Henry constants, the dissociation constants and the
#: vapour pressure of water (25 C), in K.
T_REF = 298.15
#: 0 C in K: the datasets record the temperature in C.
KELVIN = 273.15
#: Molar masses in g/mol, which turn the gas states (g per litre of head space) into moles.
M_H2, M_CH4, M_CO2 = 2, 16, 44
#: Vapour pressure of water at T_REF in bar, and the van 't Hoff coefficient of its
#: temperature dependence in K (Rosen & Jeppsson 2006); about 0.062 bar at 37 C.
P_H2O_REF = 0.0313
H2O_VAN_T_HOFF = 5290
#: Reaction enthalpies of the dissociation of CO2 and ammonium and of water, in J/mol.
#: The van 't Hoff terms divide them by R * 100 (100 J = 1 L bar).
DH_CO2 = 7646
DH_TAN = 51965
DH_W = 55900
#: Keeps the competition of the valerate and butyrate degraders, S_va / (S_va + S_bu),
#: finite when both are gone.
C4_COMPETITION = 10**(-8)
#: Where decaying biomass goes: the shares of carbohydrates, proteins and lipids
#: (Weinrich 2017), the decay columns of every Petersen matrix.
DECAY_YIELDS = (0.19, 0.77, 0.04)

#: Bounds of the bounded variants (the hybrids, R2 and the full ADM1): the proton
#: concentration (pH 2 to 12), the head-space pressure in bar and its outflow in L/d.
S_H_BOUNDS = (1e-12, 1e-2)
P_GAS_BOUNDS = (1e-6, 10)
Q_GAS_BOUNDS = (0, 1000)


def water_vapour_pressure(T):
    """Partial pressure of water vapour in the head space in bar (Rosen & Jeppsson 2006)."""
    return P_H2O_REF * jnp.exp(H2O_VAN_T_HOFF * (1/T_REF - 1/T))


def van_t_hoff(K_ref, enthalpy, T):
    """A dissociation constant ``K_ref`` (at T_REF) at temperature ``T``."""
    R = jnp.array(R_GAS)
    return K_ref * jnp.exp(-enthalpy/(R* 100) * (1/T - 1/T_REF))


def partial_pressure(S_gas, molar_mass, T):
    """Partial pressure in bar of a head-space gas held at ``S_gas`` g/L."""
    R = jnp.array(R_GAS)
    return S_gas * R * T / molar_mass


def proton_concentration(phi, K_w):
    """S_H in mol/L from the charge balance S_H^2 + phi S_H - K_w = 0, where ``phi`` is
    the net charge of every ion but H+ and OH-."""
    return - phi / 2 + jnp.sqrt(phi**2 + 4 * K_w) / 2


def ph_inhibition(S_H, params, group):
    """The pH inhibition of a group of degraders (``ac``, ``aa`` or ``h2``), between the
    lower and upper pK of its parameters, with the Hill coefficient ``n_<group>``."""
    n = params[f'n_{group}']
    pK_l, pK_u = params[f'pK_l_{group}'], params[f'pK_u_{group}']
    return (10**(-n*(pK_l + pK_u)/2) /
            (10**(-n*(pK_l + pK_u)/2) + S_H**n))


def pressure_law_outflow(p_gas, params, bounded):
    """The head-space outflow in L/d from the pressure law of ADM1 (BSM2), as the
    right-hand sides integrate it.

    ``bounded`` clips the pressure to :data:`P_GAS_BOUNDS` and the flow to
    :data:`Q_GAS_BOUNDS`; otherwise the flow is only kept non-negative.
    """
    if bounded:
        p_gas = jnp.clip(p_gas, *P_GAS_BOUNDS)
    q_gas = params['k_p'] * (p_gas - params['p_norm']) * p_gas / params['p_norm']
    if bounded:
        return jnp.clip(q_gas, *Q_GAS_BOUNDS)
    return jnp.maximum(0, q_gas)
