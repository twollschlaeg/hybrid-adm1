"""ADM1-R3 (Weinrich 2017): R4 with acetate as an intermediate, acetoclastic
methanogens X_ac, and the acid-base chemistry that gives the pH.

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
    DECAY_YIELDS, DH_CO2, DH_TAN, DH_W, M_CH4, M_CO2, S_H_BOUNDS, henry_ch4, henry_co2,
    partial_pressure, ph_inhibition, pressure_law_outflow, proton_concentration, van_t_hoff,
    water_vapour_pressure,
)
from adm1_package.models.simulation import (
    Mechanistic, biogas, column, columns, solution_arrays, state_output, unpack,
)
from adm1_package.utils.gas import gas_transfer_rates
from adm1_package.utils.initial_state import StateLayout

#: The state vector, shared with the UDE, for the warm-up initialisation.
STATES = StateLayout(
    names=('S_ac', 'S_ch4', 'S_IC', 'S_TAN', 'S_h2o', 'X_ch', 'X_pr', 'X_li', 'X_bac', 'X_ac',
           'S_an', 'S_cat', 'S_ac_i', 'S_hco3_i', 'S_nh3', 'S_gas_ch4', 'S_gas_co2'),
    held=('S_TAN', 'S_h2o', 'S_an', 'S_cat'),
    acids=(('S_ac', 'S_ac_i', 60.0, 'K_a_ac'),),
)


def petersen_matrix(params) -> Float[Array, "17 11"]:
    '''Stoichiometric coefficients of the 17 states (rows) in the 11 processes (columns):
    hydrolysis of carbohydrates, proteins and lipids, growth of X_ac, decay of X_bac and
    X_ac, the acid-base equilibria of acetate, CO2 and ammonia, and the transfer of CH4
    and CO2 to the head space.'''
    d_ch, d_pr, d_li = DECAY_YIELDS
    return jnp.array([
        [0.6555, 0.9947, 1.7651, -26.5447, 0, 0, 0, 0, 0, 0, 0],                    # S_ac
        [0.0818, 0.0696, 0.1913, 6.7367, 0, 0, 0, 0, 0, -1, 0],                     # S_ch4
        [0.2245, 0.1029, -0.6472, 18.4808, 0, 0, 0, 0, 0, 0, -1],                   # S_IC
        [-0.0169, 0.1746, -0.0244, -0.1506, 0, 0, 0, 0, 0, 0, 0],                   # S_TAN
        [-0.0574, -0.4767, -0.4469, 0.4778, 0, 0, 0, 0, 0, 0, 0],                   # S_h2o
        [-1, 0, 0, 0, d_ch, d_ch, 0, 0, 0, 0, 0],                                   # X_ch
        [0, -1, 0, 0, d_pr, d_pr, 0, 0, 0, 0, 0],                                   # X_pr
        [0, 0, -1, 0, d_li, d_li, 0, 0, 0, 0, 0],                                   # X_li
        [0.1125, 0.1349, 0.1621, 0, -1, 0, 0, 0, 0, 0, 0],                          # X_bac
        [0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],                                         # X_ac
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                                          # S_an
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                                          # S_cat
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],                                         # S_ac_i
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],                                         # S_hco3_i
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],                                         # S_nh3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, params['V_liq'] / params['V_gas'], 0],          # S_gas_ch4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, params['V_liq'] / params['V_gas']]           # S_gas_co2
    ])


def transfer_rates(clipped, T_op, params):
    '''CH4 and CO2 transfer along a solution of clipped states; R3 transfers the
    dissolved CO2, S_IC less bicarbonate.'''
    def c(name):
        return column(STATES, clipped, name)
    return gas_transfer_rates(c('S_ch4'), jnp.clip(c('S_IC') - c('S_hco3_i'), 0, None),
                              c('S_gas_ch4'), c('S_gas_co2'), T_op, params['k_La'])


#: The states the algebraic outputs read.
READ = ('S_IC', 'S_TAN', 'S_an', 'S_cat', 'S_ac_i', 'S_hco3_i', 'S_nh3', 'S_gas_ch4',
        'S_gas_co2')


def algebraic_output(s, y, t, interp, T_op, params, K_I_nh3, state_bounds) -> dict:
    '''pH, partial pressures, gas flows and inhibition along a solution (mechanistic and
    UDE), from the states ``s`` (:data:`READ`). ``K_I_nh3`` is the ammonia inhibition
    constant the right-hand side used, and ``state_bounds`` its clip bounds. The pH is the
    charge balance's on the unclipped states.'''
    K_w = van_t_hoff(params['K_w'], DH_W, T_op)
    p_h2o = water_vapour_pressure(T_op)

    S_nh4_i = s.S_TAN - s.S_nh3
    S_co2 = s.S_IC - s.S_hco3_i
    phi = (s.S_cat + S_nh4_i / 17 - s.S_hco3_i / 44 - s.S_ac_i / 60 - s.S_an)
    S_H = proton_concentration(phi, K_w)
    pH = - jnp.log10(S_H)

    p_ch4 = partial_pressure(s.S_gas_ch4, M_CH4, T_op)
    p_co2 = partial_pressure(s.S_gas_co2, M_CO2, T_op)
    p_gas = p_ch4 + p_co2 + p_h2o
    q_gasout, ch4_out = biogas(y, t, interp, T_op, params, (p_ch4, p_co2), p_h2o,
                               state_bounds, transfer_rates)

    I_TAN = s.S_TAN / (s.S_TAN + params['K_I_TAN'])
    I_nh3 = K_I_nh3 / (K_I_nh3 + s.S_nh3)
    I_ph_ac = ph_inhibition(S_H, params, 'ac')
    I_ac = I_TAN * I_nh3 * I_ph_ac
    return {
        'T_op': T_op, 'pH': pH, 'S_nh4': S_nh4_i, 'S_co2': S_co2, 'p_ch4': p_ch4,
        'p_co2': p_co2, 'p_gas': p_gas, 'q_gasout': q_gasout, 'ch4_out': ch4_out,
        # R3 has always reported the TAN limitation as I_IN (R2 and the full ADM1: I_TAN)
        'I_IN': I_TAN, 'I_nh3': I_nh3, 'I_ph_ac': I_ph_ac, 'I_ac': I_ac,
    }


class ADM1R3(Mechanistic):
    '''ADM1-R3 model according to the thesis from S. Weinrich.'''

    VARIANT = "r3"
    STATES = STATES
    #: The mechanistic R3 clips its states to [0, 1e10] and nothing else: neither the
    #: proton concentration nor the head-space pressure, and the outflow only to >= 0.
    STATE_BOUNDS = (0, 1e10)
    BOUNDED = False

    def _rhs(self, t, y: Float[Array, "17"],
             args: tuple[dict, dx.CubicInterpolation]) -> Float[Array, "17"]:
        '''Evaluate right hand side of model ODE'''
        params, interp = args
        forcing = interp.evaluate(t)
        T_op = forcing[0]
        q_in = forcing[1]
        # the influent: S_ac, S_TAN, S_h2o, X_ch, X_pr, X_li
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
        phi = s.S_cat + S_nh4_i / 17 - s.S_hco3_i / 44 - s.S_ac_i / 60 - s.S_an
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
        I_ac = I_TAN * I_nh3 * I_ph_ac

        rates = jnp.array([
            params['k_hyd_ch'] * s.X_ch,
            params['k_hyd_pr'] * s.X_pr,
            params['k_hyd_li'] * s.X_li,
            params['k_m_ac'] * s.X_ac * s.S_ac / (params['K_ac'] + s.S_ac) * I_ac,
            params['k_dec_Xbac'] * s.X_bac,
            params['k_dec_Xac'] * s.X_ac,
            params['k_AB_ac'] * (s.S_ac_i * (params['K_a_ac'] + S_H) - params['K_a_ac'] * s.S_ac),
            params['k_AB_co2'] * (s.S_hco3_i * (K_a_co2 + S_H) - K_a_co2 * s.S_IC),
            params['k_AB_TAN'] * (s.S_nh3 * (K_a_TAN + S_H) - K_a_TAN * s.S_TAN),
            params['k_La'] * (s.S_ch4 - 16 * (K_H_ch4 * p_ch4)),
            params['k_La'] * (S_co2 - 44 * (K_H_co2 * p_co2))
        ])
        A = petersen_matrix(params)

        degradabilities = jnp.array([params['f_ch'],params['f_pr'], params['f_li']])
        in_out_flow = jnp.concat([
            q_in / params['V_liq'] * (y_in[:1] - y[:1]),
            q_in / params['V_liq'] * (- y[1:3]),
            q_in / params['V_liq'] * (y_in[1:3] - y[3:5]),
            q_in / params['V_liq'] * (degradabilities * y_in[3:6] - y[5:8]),
            q_in / params['V_liq'] * (-y[8:12]),
            0 * y[12:15],
            q_gas / params['V_gas'] * (-y[15:17])
        ], axis=0)

        return jnp.dot(A, rates) + in_out_flow

    def _petersen_matrix(self, params: dict) -> Float[Array, "17 11"]:
        return petersen_matrix(params)

    def _calculate_output(self, sol: dx.Solution,
                          args: tuple[dict, dx.CubicInterpolation]) -> dict:
        '''Calculate the output of the model.'''
        params, interp = args
        t, y = solution_arrays(sol)
        s = columns(STATES, y, *READ)
        T_op = jax.vmap(interp.evaluate)(t)[:,0]
        algebraic = algebraic_output(s, y, t, interp, T_op, params, params['K_I_nh3'],
                                     self.STATE_BOUNDS)
        return OrderedDict(state_output(t, y, STATES) | algebraic)
