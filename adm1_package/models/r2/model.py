"""ADM1-R2 (Weinrich 2017): R3 with valerate, butyrate and propionate as further
intermediates, and their degraders X_va, X_bu and X_pro.

This module is the variant's one place for its states, its Petersen matrix and its
mechanistic rates; the hybrid (ude.py) reuses the first two. The solve and the
reported output are shared (models/simulation.py).
"""

from collections import OrderedDict

import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from adm1_package.models.chemistry import (
    C4_COMPETITION, DECAY_YIELDS, DH_CO2, DH_TAN, DH_W, M_CH4, M_CO2, S_H_BOUNDS, henry_ch4,
    henry_co2, partial_pressure, ph_inhibition, pressure_law_outflow, proton_concentration,
    van_t_hoff, water_vapour_pressure,
)
from adm1_package.models.simulation import (
    Mechanistic, biogas, column, columns, solution_arrays, state_output, unpack,
)
from adm1_package.utils.gas import gas_transfer_rates
from adm1_package.utils.initial_state import StateLayout

#: The state vector, shared with the UDE, for the warm-up initialisation.
STATES = StateLayout(
    names=('S_va', 'S_bu', 'S_pro', 'S_ac', 'S_ch4', 'S_IC', 'S_TAN', 'S_h2o',
           'X_ch', 'X_pr', 'X_li', 'X_bac', 'X_va', 'X_bu', 'X_pro', 'X_ac',
           'S_an', 'S_cat', 'S_va_i', 'S_bu_i', 'S_pro_i', 'S_ac_i', 'S_hco3_i', 'S_nh3',
           'S_gas_ch4', 'S_gas_co2'),
    held=('S_TAN', 'S_h2o', 'S_an', 'S_cat'),
    acids=(('S_va', 'S_va_i', 102.0, 'K_a_va'), ('S_bu', 'S_bu_i', 88.0, 'K_a_bu'),
           ('S_pro', 'S_pro_i', 74.0, 'K_a_pro'), ('S_ac', 'S_ac_i', 60.0, 'K_a_ac')),
)

#: R2 clips its states to [1e-12, 1e8], mechanistic and hybrid alike, and bounds the
#: proton concentration, the head-space pressure and the outflow.
STATE_BOUNDS = (1e-12, 1e8)


def petersen_matrix(params) -> Float[Array, "26 20"]:
    '''Stoichiometric coefficients of the 26 states (rows) in the 20 processes (columns):
    hydrolysis of carbohydrates, proteins and lipids (0-2), growth of X_va, X_bu, X_pro
    and X_ac (3-6), decay of X_bac, X_va, X_bu, X_pro and X_ac (7-11), the acid-base
    equilibria of valerate, butyrate, propionate, acetate, CO2 and ammonia (12-17), and
    the transfer of CH4 and CO2 to the head space (18, 19).'''
    d_ch, d_pr, d_li = DECAY_YIELDS
    A = jnp.zeros((26, 20))
    # The conversion processes 0-11 of the states S_va ... X_ac (rows 0-15).
    B = jnp.array([
        [0, 0.1588, 0, -10.1452, 0, 0, 0, 0, 0, 0, 0, 0],                         # S_va
        [0.0763, 0.2014, 0.0093, 0, -10.9274, 0, 0, 0, 0, 0, 0, 0],               # S_bu
        [0.1903, 0.0465, 0.0231, 6.9368, 0, -14.4449, 0, 0, 0, 0, 0, 0],          # S_pro
        [0.41, 0.5278, 1.7353, 5.6494, 14.0023, 11.2133, -26.5447, 0, 0, 0, 0, 0],# S_ac
        [0.0477, 0.0199, 0.1872, 0.6864, 0.8790, 2.1242, 6.7367, 0, 0, 0, 0, 0],  # S_ch4
        [0.2255, 0.1835, -0.6470, -2.6138, -3.0468, 1.5366, 18.4808, 0, 0, 0, 0, 0],  # S_IC
        [-0.0139, 0.1813, -0.0240, -0.1506, -0.1506, -0.1506, -0.1506, 0, 0, 0, 0, 0],  # S_TAN
        [-0.0283, -0.4092, -0.4434, -1.3630, -1.7566, -1.2786, 0.4778, 0, 0, 0, 0, 0],   # S_h2o
        [-1, 0, 0, 0, 0, 0, 0, d_ch, d_ch, d_ch, d_ch, d_ch],                     # X_ch
        [0, -1, 0, 0, 0, 0, 0, d_pr, d_pr, d_pr, d_pr, d_pr],                     # X_pr
        [0, 0, -1, 0, 0, 0, 0, d_li, d_li, d_li, d_li, d_li],                     # X_li
        [0.0923, 0.0900, 0.1597, 0, 0, 0, 0, -1, 0, 0, 0, 0],                     # X_bac
        [0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0],                                    # X_va
        [0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0],                                    # X_bu
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],                                    # X_pro
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1]                                     # X_ac
    ])
    A = A.at[:16, :12].set(B)
    # S_ch4 and S_IC leave to the head space
    A = A.at[4:6, 18:].set(jnp.array([[-1, 0], [0, -1]]))
    # S_va_i, S_bu_i, S_pro_i, S_ac_i, S_hco3_i and S_nh3 in their acid-base equilibria
    A = A.at[18:24, 12:18].set(-1 * jnp.identity(6))
    # S_gas_ch4 and S_gas_co2 receive them
    A = A.at[24:26, 18:].set(jnp.identity(2) * params['V_liq'] / params['V_gas'])
    return A


def transfer_rates(clipped, T_op, params):
    '''CH4 and CO2 transfer along a solution of clipped states; R2 transfers the
    dissolved CO2, S_IC less bicarbonate.'''
    def c(name):
        return column(STATES, clipped, name)
    return gas_transfer_rates(c('S_ch4'), jnp.clip(c('S_IC') - c('S_hco3_i'), 0, None),
                              c('S_gas_ch4'), c('S_gas_co2'), T_op, params['k_La'])


#: The states the algebraic outputs read; R2 reports these columns as read.
READ = ('S_IC', 'S_TAN', 'S_an', 'S_cat', 'S_va_i', 'S_bu_i', 'S_pro_i', 'S_ac_i',
        'S_hco3_i', 'S_nh3', 'S_gas_ch4', 'S_gas_co2')


def algebraic_output(s, y, t, interp, T_op, params, K_I_nh3) -> dict:
    '''pH, partial pressures, gas flows and inhibition along a solution (mechanistic and
    UDE), from the states ``s`` (:data:`READ`). ``K_I_nh3`` is the ammonia inhibition
    constant the right-hand side used. The pH is the charge balance's on the unclipped
    states.'''
    K_w = van_t_hoff(params['K_w'], DH_W, T_op)
    p_h2o = water_vapour_pressure(T_op)

    S_nh4_i = s.S_TAN - s.S_nh3
    S_co2 = s.S_IC - s.S_hco3_i
    phi = (s.S_cat + S_nh4_i / 17 - s.S_hco3_i / 44 - s.S_ac_i / 60 - s.S_pro_i / 74
            - s.S_bu_i / 88 - s.S_va_i / 102 - s.S_an)
    S_H = proton_concentration(phi, K_w)
    pH = - jnp.log10(S_H)
    p_ch4 = partial_pressure(s.S_gas_ch4, M_CH4, T_op)
    p_co2 = partial_pressure(s.S_gas_co2, M_CO2, T_op)
    p_gas = p_ch4 + p_co2 + p_h2o
    q_gasout, ch4_out = biogas(y, t, interp, T_op, params, (p_ch4, p_co2), p_h2o,
                               STATE_BOUNDS, transfer_rates)

    I_TAN = s.S_TAN / (s.S_TAN + params['K_I_TAN'])
    I_nh3 = K_I_nh3 / (K_I_nh3 + s.S_nh3)
    I_ph_ac = ph_inhibition(S_H, params, 'ac')
    I_ph_aa = ph_inhibition(S_H, params, 'aa')
    return {
        'T_op': T_op, 'pH': pH, 'S_nh4': S_nh4_i, 'S_co2': S_co2, 'p_ch4': p_ch4,
        'p_co2': p_co2, 'p_gas': p_gas, 'q_gasout': q_gasout, 'ch4_out': ch4_out,
        'I_TAN': I_TAN, 'I_nh3': I_nh3, 'I_ph_ac': I_ph_ac, 'I_ph_aa': I_ph_aa,
    }


class ADM1R2(Mechanistic):
    '''ADM1-R2 model according to the thesis from S. Weinrich.'''

    VARIANT = "r2"
    STATES = STATES
    STATE_BOUNDS = STATE_BOUNDS
    BOUNDED = True

    def _rhs(self, t, y: Float[Array, "26"],
             args: tuple[dict, dx.CubicInterpolation]) -> Float[Array, "26"]:
        '''Evaluate right hand side of model ODE'''
        params, interp = args
        forcing = interp.evaluate(t)
        T_op = forcing[0]
        q_in = forcing[1]
        # the influent: S_va, S_bu, S_pro, S_ac, S_TAN, S_h2o, X_ch, X_pr, X_li
        y_in = forcing[2:]
        params = self._at_temperature(params, T_op)

        K_H_ch4 = henry_ch4(T_op)
        K_H_co2 = henry_co2(T_op)
        K_a_co2 = van_t_hoff(params['K_a_co2'], DH_CO2, T_op)
        K_a_TAN = van_t_hoff(params['K_a_TAN'], DH_TAN, T_op)
        K_w = van_t_hoff(params['K_w'], DH_W, T_op)
        p_h2o = water_vapour_pressure(T_op)
        K_I_nh3 = params['K_I_nh3']

        y = jnp.clip(y, *self.STATE_BOUNDS) # avoid NaN and Inf values
        s = unpack(STATES, y)

        # negative concentrations are not allowed
        S_nh4_i = jnp.clip(s.S_TAN - s.S_nh3, 0, None)
        S_co2 = jnp.clip(s.S_IC - s.S_hco3_i, 0, None)
        phi = (s.S_cat + S_nh4_i / 17 - s.S_hco3_i / 44 - s.S_ac_i / 60 - s.S_pro_i / 74
                - s.S_bu_i / 88 - s.S_va_i / 102 - s.S_an)
        S_H = proton_concentration(phi, K_w)
        if self.BOUNDED:
            S_H = jnp.clip(S_H, *S_H_BOUNDS)
        p_ch4 = partial_pressure(s.S_gas_ch4, M_CH4, T_op)
        p_co2 = partial_pressure(s.S_gas_co2, M_CO2, T_op)
        p_gas = p_ch4 + p_co2 + p_h2o
        q_gas = pressure_law_outflow(p_gas, params, self.BOUNDED)

        I_TAN = s.S_TAN / (s.S_TAN + params['K_I_TAN'])
        I_nh3 = K_I_nh3 / (K_I_nh3 + s.S_nh3)
        I_ph_ac = ph_inhibition(S_H, params, 'ac')
        I_ph_aa = ph_inhibition(S_H, params, 'aa')
        I_ac = I_TAN * I_nh3 * I_ph_ac
        I_va = I_TAN * I_ph_aa
        I_bu = I_TAN * I_ph_aa
        I_pro = I_TAN * I_ph_aa

        p = params
        rates = jnp.array([
            p['k_hyd_ch'] * s.X_ch,
            p['k_hyd_pr'] * s.X_pr,
            p['k_hyd_li'] * s.X_li,
            p['k_m_va'] * s.X_va * s.S_va / (s.S_va + s.S_bu + C4_COMPETITION) * s.S_va / (p['K_va'] + s.S_va) * I_va,
            p['k_m_bu'] * s.X_bu * s.S_bu / (s.S_bu + s.S_va + C4_COMPETITION) * s.S_bu / (p['K_bu'] + s.S_bu) * I_bu,
            p['k_m_pro'] * s.X_pro * s.S_pro / (p['K_pro'] + s.S_pro) * I_pro,
            p['k_m_ac'] * s.X_ac * s.S_ac / (p['K_ac'] + s.S_ac) * I_ac,
            p['k_dec_Xbac'] * s.X_bac,
            # One decay rate for both C4 degraders, as calibrated (the hybrid has two).
            p['k_dec_Xc4'] * s.X_va,
            p['k_dec_Xc4'] * s.X_bu,
            p['k_dec_Xpro'] * s.X_pro,
            p['k_dec_Xac'] * s.X_ac,
            p['k_AB_va'] * (s.S_va_i * (p['K_a_va'] + S_H) - p['K_a_va'] * s.S_va),
            p['k_AB_bu'] * (s.S_bu_i * (p['K_a_bu'] + S_H) - p['K_a_bu'] * s.S_bu),
            p['k_AB_pro'] * (s.S_pro_i * (p['K_a_pro'] + S_H) - p['K_a_pro'] * s.S_pro),
            p['k_AB_ac'] * (s.S_ac_i * (p['K_a_ac'] + S_H) - p['K_a_ac'] * s.S_ac),
            p['k_AB_co2'] * (s.S_hco3_i * (K_a_co2 + S_H) - K_a_co2 * s.S_IC),
            p['k_AB_TAN'] * (s.S_nh3 * (K_a_TAN + S_H) - K_a_TAN * s.S_TAN),
            p['k_La'] * (s.S_ch4 - 16 * (K_H_ch4 * p_ch4)),
            p['k_La'] * (S_co2 - 44 * (K_H_co2 * p_co2))
        ])
        A = petersen_matrix(params)

        # S_TAN and S_h2o enter as fed, the particulates scaled by their degradabilities
        inflow_factors = jnp.array([1.0, 1.0, params['f_ch'],params['f_pr'], params['f_li']])
        in_out_flow = jnp.concat([
            q_in / params['V_liq'] * (y_in[:4] - y[:4]),
            q_in / params['V_liq'] * (- y[4:6]),
            q_in / params['V_liq'] * (inflow_factors * y_in[4:] - y[6:11]),
            q_in / params['V_liq'] * (-y[11:16]),
            q_in / params['V_liq'] * (-y[16:18]),
            0 * y[18:24],
            q_gas / params['V_gas'] * (-y[24:26])
        ], axis=0)

        return jnp.dot(A, rates) + in_out_flow

    def _petersen_matrix(self, params: dict) -> Float[Array, "26 20"]:
        return petersen_matrix(params)

    def _calculate_output(self, sol: dx.Solution,
                          args: tuple[dict, dx.CubicInterpolation]) -> dict:
        '''Calculate the output of the model.'''
        params, interp = args
        t, y = solution_arrays(sol)
        s = columns(STATES, y, *READ)
        T_op = jax.vmap(interp.evaluate)(t)[:,0]
        algebraic = algebraic_output(s, y, t, interp, T_op, params, params['K_I_nh3'])
        return OrderedDict(state_output(t, y, STATES, read=s) | algebraic)
