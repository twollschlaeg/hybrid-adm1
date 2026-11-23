"""The biogas and methane flows the ADM1 variants report.

Every variant closes its head space with the pressure law of ADM1 (BSM2),

    q = k_p (P - p_norm) P / p_norm,

which holds the head-space pressure P within a fraction of a millibar of p_norm. The
law sets how fast the head space responds, not how much gas leaves it. Dividing the
two gas-phase balances by their molar masses and adding them gives, exactly,

    q = (V_liq * Sigma - V_gas * dn/dt) / n,
    n = (P - p_w) / (R T),   Sigma = rho_ch4 / 16 + rho_co2 / 44,

outflow = transfer out of the liquid - accumulation in the head space. (The full ADM1's
head space holds hydrogen as well: a third balance, rho_h2 / 2 in Sigma.) Splitting dn/dt
into its temperature part and (dP/dt) / (R T) then gives

    q + tau * dq/dt = q_mb,   tau = V_gas p_norm / (k_p (2P - p_norm) (P - p_w)),

where ``q_mb`` is the same balance with dn/dt taken at constant pressure. For the 1-L
head spaces here tau is about 2 s, so the two differ by roughly 1e-4 L/d.

They are not equally robust, though. The pressure law subtracts two pressures that
agree to about 1e-4 of their size, and the solver resolves P only to its tolerance
(rtol = 1e-4), so on some days the pressure law's biogas is off by as much as the
signal. The mass balance reads the pressures only through ratios, so the models
report it. It needs tau to be negligible, which holds for a head space of litres but
not for a full-scale digester.

Both give litres of head-space gas: wet, at the reactor temperature. The campaigns
report norm volumes instead, dry gas at 0 degC and 1.01325 bar (NmL; the methane
content is "korr", i.e. of the dry gas), so the models report the dry moles leaving per
day at R T_N / P_N = 22.414 L/mol. For the same moles the head-space volume is
(T / T_N) P_N / (P - p_w) times larger: 1.21 at 37 degC, 1.43 at 55 degC.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

__all__ = ["gas_transfer_rates", "hydrogen_transfer_rate", "biogas_outflow",
           "initial_gas_phase", "correct_gas_initial_values", "henry_ch4", "dissolved_ch4_at_equilibrium", "henry_co2",
           "dissolved_co2_at_equilibrium", "henry_h2", "dissolved_h2_at_equilibrium"]

R_GAS = 0.083145   # L bar / (K mol)
T_NORM = 273.15    # K    } standard conditions (DIN 1343), at which the campaigns
P_NORM = 1.01325   # bar  } report their gas volumes


def henry_ch4(T):
    """Henry constant of methane in mol/(L bar), as the right-hand sides write it."""
    return 1.4 * 10**(-3) * jnp.exp(1600 * (1/T - 1/298.15))


def dissolved_ch4_at_equilibrium(p_ch4, T):
    """Dissolved methane in g/L in equilibrium with the partial pressure ``p_ch4``."""
    return 16 * henry_ch4(T) * p_ch4


def henry_co2(T):
    """Henry constant of carbon dioxide in mol/(L bar), as the right-hand sides write it."""
    return 3.4 * 10**(-2) * jnp.exp(2300 * (1/T - 1/298.15))


def dissolved_co2_at_equilibrium(p_co2, T):
    """Dissolved CO2 in g/L in equilibrium with the partial pressure ``p_co2``."""
    return 44 * henry_co2(T) * p_co2


def gas_transfer_rates(S_ch4, S_co2, S_gas_ch4, S_gas_co2, T, k_La):
    """Liquid-to-gas transfer of CH4 and CO2 in g/L/d, as the right-hand sides write it.

    ``S_co2`` is the species a variant transfers: ``S_IC`` in R4, dissolved CO2 in R2
    and R3. Pass the states clipped the way that variant's right-hand side clips them,
    so these are the rates the ODE integrated.
    """
    R = jnp.array(R_GAS)
    K_H_ch4 = henry_ch4(T)
    K_H_co2 = henry_co2(T)
    p_ch4 = S_gas_ch4 * R * T / 16
    p_co2 = S_gas_co2 * R * T / 44
    return (k_La * (S_ch4 - 16 * (K_H_ch4 * p_ch4)),
            k_La * (S_co2 - 44 * (K_H_co2 * p_co2)))


def henry_h2(T):
    """Henry constant of hydrogen in mol/(L bar), as the full ADM1's right-hand side writes it."""
    return 7.8 * 10**(-4) * jnp.exp(530 * (1/T - 1/298.15))


def dissolved_h2_at_equilibrium(p_h2, T):
    """Dissolved hydrogen in g/L in equilibrium with the partial pressure ``p_h2``."""
    return 2 * henry_h2(T) * p_h2


def hydrogen_transfer_rate(S_h2, S_gas_h2, T, k_La):
    """Liquid-to-gas transfer of H2 in g/L/d, as the full ADM1's right-hand side writes it.

    Only the full ADM1 carries hydrogen. Pass the states clipped the way its right-hand
    side clips them, as for :func:`gas_transfer_rates`.
    """
    R = jnp.array(R_GAS)
    K_H_h2 = henry_h2(T)
    p_h2 = S_gas_h2 * R * T / 2
    return k_La * (S_h2 - 2 * (K_H_h2 * p_h2))


def biogas_outflow(p_ch4, p_co2, p_h2o, T, dT_dt, rho_ch4, rho_co2,
                   params: Mapping[str, Any], p_h2=0.0, rho_h2=0.0):
    """Biogas and methane outflow along a solution, in litres of dry gas at standard
    conditions per day, from the mole balance.

    Time is the leading axis and index 0 is the initial state. There the pressure law
    is exact -- the data set the initial gas phase so that it reproduces the first
    measurement (:func:`initial_gas_phase`) -- while the state may sit far from a
    quasi-steady head space, so index 0 reports the pressure law.

    The full ADM1's head space also holds hydrogen: ``p_h2`` and its transfer from the
    liquid ``rho_h2`` (g/L/d, :func:`hydrogen_transfer_rate`) count towards the dry gas
    like the methane and the carbon dioxide.
    """
    P = p_ch4 + p_co2 + p_h2 + p_h2o
    q_pressure = jnp.maximum(0, params['k_p'] * (P - params['p_norm']) * P / params['p_norm'])

    dry = p_ch4 + p_co2 + p_h2                             # P - p_h2o = n R T
    has_gas = dry > 0
    dry = jnp.where(has_gas, dry, 1.0)
    n = dry / (R_GAS * T)
    dn_dt = -(p_h2o * 5290 / T**2 + n * R_GAS) * dT_dt / (R_GAS * T)   # at constant P
    q = (params['V_liq'] * (rho_ch4 / 16 + rho_co2 / 44 + rho_h2 / 2)
         - params['V_gas'] * dn_dt) / n
    # Unlike the pressure law this does not stop the outflow below p_norm: the sign of
    # P - p_norm is as unreliable as its size, and the computed pressure dips below
    # p_norm on days when the exact one never does. After fast cooling the head space
    # really does refill for a few minutes, which this slightly overstates.
    q = jnp.where(has_gas, jnp.maximum(q, 0.0), 0.0)
    q = q.at[0].set(q_pressure[0])
    # Head-space litres hold n mol of dry gas each, the right-hand side's convention.
    q = q * n * R_GAS * T_NORM / P_NORM
    return q, q * p_ch4 / dry


def initial_gas_phase(q_gas, q_ch4, T, params: Mapping[str, Any], p_h2=0.0):
    """CH4 and CO2 partial pressures whose outflow is the first measurement.

    Inverts index 0 of :func:`biogas_outflow`: ``q_gas`` and ``q_ch4`` are the first
    measured biogas and methane flows in NL/d, ``T`` the temperature then. The pressure
    law in head-space litres, ``k_p (P - p_norm) P / p_norm``, holds (P - p_w) / (R T)
    mol of dry gas per litre, which makes P the root of a cubic. P lies within 1e-3 bar
    of p_norm, where each pass below gains three to four digits.

    ``p_h2`` is the hydrogen in the full ADM1's head space, which nothing measures. It
    takes its part of the dry gas from the carbon dioxide.
    """
    p_h2o = 0.0313 * jnp.exp(5290 * (1 / 298.15 - 1 / T))
    moles = q_gas * P_NORM / (R_GAS * T_NORM)            # mol/d of dry gas
    P = params['p_norm']
    for _ in range(3):
        head_space = moles * R_GAS * T / (P - p_h2o)      # L/d at P and T
        P = params['p_norm'] / 2 + jnp.sqrt(
            params['p_norm'] ** 2 / 4 + params['p_norm'] / params['k_p'] * head_space)
    dry = P - p_h2o
    p_ch4 = q_ch4 / q_gas * dry
    return p_ch4, dry - p_ch4 - p_h2


def correct_gas_initial_values(initial_values, params, forcing, targets, ch4_index=None,
                               co2_index=None):
    """Derive the initial gas-phase concentrations from the first measurement.

    The gas-phase states ``S_gas_ch4`` and ``S_gas_co2`` are not measured. They
    are recovered from the first observed biogas and methane production rates by
    inverting the head-space pressure balance (:func:`initial_gas_phase`), which is
    far better than starting them at zero and letting the solver relax over the
    first days. The rates are norm volumes, as the models report them.

    With ``ch4_index``, the position of ``S_ch4``, the dissolved methane starts in
    equilibrium with that gas phase. A fitted start value can otherwise sit far above
    saturation and degas within hours. No score sees that, so no fit corrects it.
    ``co2_index`` does the same for a dissolved CO2 state: R4's ``S_IC``, all of which it
    transfers. R2 and R3 split ``S_IC`` by
    the pH, so they set it with the rest of their chemistry
    (:func:`adm1_package.utils.initial_state.measured_chemistry`).

    Shared by ADM1-R2, R3 and R4, which carry the two gas states last. The full
    ADM1 keeps its own version because its head space also contains hydrogen.
    """
    T_op = forcing.evaluate(0)[0]
    p_ch4, p_co2 = initial_gas_phase(targets[0, 0], targets[0, 1], T_op, params)

    gas_constant = jnp.array(0.083145)  # L bar / (K mol)
    initial_values = initial_values.at[-2].set(p_ch4 * 16 / (gas_constant * T_op))
    initial_values = initial_values.at[-1].set(p_co2 * 44 / (gas_constant * T_op))
    if ch4_index is not None:
        initial_values = initial_values.at[ch4_index].set(dissolved_ch4_at_equilibrium(p_ch4, T_op))
    if co2_index is not None:
        initial_values = initial_values.at[co2_index].set(dissolved_co2_at_equilibrium(p_co2, T_op))
    return initial_values
