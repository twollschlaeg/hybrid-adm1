"""ADM1-R4 (Weinrich 2017): hydrolysis of carbohydrates, proteins and lipids straight to
biogas, with one biomass X_bac.

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
    DECAY_YIELDS, M_CH4, M_CO2, henry_ch4, henry_co2, partial_pressure, pressure_law_outflow,
    water_vapour_pressure,
)
from adm1_package.models.simulation import (
    Mechanistic, biogas, column, columns, solution_arrays, state_output, unpack,
)
from adm1_package.utils.gas import gas_transfer_rates
from adm1_package.utils.initial_state import StateLayout

#: The state vector, shared with the UDE, for the warm-up initialisation.
STATES = StateLayout(
    names=('S_ch4', 'S_IC', 'S_TAN', 'S_h2o', 'X_ch', 'X_pr', 'X_li', 'X_bac',
           'S_gas_ch4', 'S_gas_co2'),
)

#: R4 clips its states to [0, 1e10], mechanistic and hybrid alike, and keeps only the
#: head-space outflow non-negative (``pressure_law_outflow(..., bounded=False)``).
STATE_BOUNDS = (0, 1e10)


def petersen_matrix(params) -> Float[Array, "10 6"]:
    '''Stoichiometric coefficients of the 10 states (rows) in the 6 processes (columns):
    hydrolysis of carbohydrates, proteins and lipids, decay of X_bac, and the transfer
    of CH4 and CO2 to the head space.'''
    d_ch, d_pr, d_li = DECAY_YIELDS
    return jnp.array([
        [0.2482, 0.3221, 0.6393, 0, -1, 0],                     # S_ch4
        [0.6809, 0.7954, 0.5817, 0, 0, -1],                     # S_IC
        [-0.0207, 0.1689, -0.0344, 0, 0, 0],                    # S_TAN
        [-0.0456, -0.4588, -0.4152, 0, 0, 0],                   # S_h2o
        [-1, 0, 0, d_ch, 0, 0],                                 # X_ch
        [0, -1, 0, d_pr, 0, 0],                                 # X_pr
        [0, 0, -1, d_li, 0, 0],                                 # X_li
        [0.1125, 0.1723, 0.2286, -1, 0, 0],                     # X_bac
        [0, 0, 0, 0, params['V_liq'] / params['V_gas'], 0],     # S_gas_ch4
        [0, 0, 0, 0, 0, params['V_liq'] / params['V_gas']]      # S_gas_co2
    ])


def transfer_rates(clipped, T_op, params):
    '''CH4 and CO2 transfer along a solution of clipped states; R4 transfers all of S_IC.'''
    def c(name):
        return column(STATES, clipped, name)
    return gas_transfer_rates(c('S_ch4'), c('S_IC'), c('S_gas_ch4'), c('S_gas_co2'),
                              T_op, params['k_La'])


#: The states the algebraic outputs read.
READ = ('S_gas_ch4', 'S_gas_co2')


def algebraic_output(s, y, t, interp, T_op, params) -> dict:
    '''Temperature, partial pressures and gas flows along a solution (mechanistic and UDE),
    from the states ``s`` (:data:`READ`).'''
    p_h2o = water_vapour_pressure(T_op)
    p_ch4 = partial_pressure(s.S_gas_ch4, M_CH4, T_op)
    p_co2 = partial_pressure(s.S_gas_co2, M_CO2, T_op)
    p_gas = p_ch4 + p_co2 + p_h2o
    q_gasout, ch4_out = biogas(y, t, interp, T_op, params, (p_ch4, p_co2), p_h2o,
                               STATE_BOUNDS, transfer_rates)
    return {'T_op': T_op, 'p_ch4': p_ch4, 'p_co2': p_co2, 'p_gas': p_gas,
            'q_gasout': q_gasout, 'ch4_out': ch4_out}


class ADM1R4(Mechanistic):
    '''ADM1-R4 model according to the thesis from S. Weinrich.'''

    VARIANT = "r4"
    STATES = STATES
    STATE_BOUNDS = STATE_BOUNDS
    BOUNDED = False

    def _rhs(self, t, y: Float[Array, "10"],
             args: tuple[dict, dx.CubicInterpolation]) -> Float[Array, "10"]:
        '''Evaluate right hand side of model ODE'''
        params, interp = args
        forcing = interp.evaluate(t)
        T_op = forcing[0]
        q_in = forcing[1]
        # the influent: S_TAN, S_h2o, X_ch, X_pr, X_li
        y_in = forcing[2:]
        params = self._at_temperature(params, T_op)

        K_H_ch4 = henry_ch4(T_op)
        K_H_co2 = henry_co2(T_op)
        p_h2o = water_vapour_pressure(T_op)

        y = jnp.clip(y, *self.STATE_BOUNDS) # avoid NaN and Inf values
        s = unpack(STATES, y)

        p_ch4 = partial_pressure(s.S_gas_ch4, M_CH4, T_op)
        p_co2 = partial_pressure(s.S_gas_co2, M_CO2, T_op)
        p_gas = p_ch4 + p_co2 + p_h2o
        q_gas = pressure_law_outflow(p_gas, params, self.BOUNDED)

        rates = jnp.array([
            params['k_hyd_ch'] * s.X_ch,
            params['k_hyd_pr'] * s.X_pr,
            params['k_hyd_li'] * s.X_li,
            params['k_dec_Xbac'] * s.X_bac,
            params['k_La'] * (s.S_ch4 - 16 * (K_H_ch4 * p_ch4)),
            params['k_La'] * (s.S_IC - 44 * (K_H_co2 * p_co2))
        ])
        A = petersen_matrix(params)

        # The degradabilities scale the particulates of the influent.
        degradabilities = jnp.array([params['f_ch'], params['f_pr'], params['f_li']])
        inflow_factors = jnp.ones(5)
        inflow_factors = inflow_factors.at[2:5].set(degradabilities)
        in_out_flow = jnp.concat([
            - q_in / params['V_liq'] * (y[:2]),
            q_in / params['V_liq'] * (inflow_factors * y_in - y[2:7]),
            - q_in / params['V_liq'] * (y[7:8]),
            q_gas / params['V_gas'] * (-y[8:10])
        ], axis=0)

        return jnp.dot(A, rates) + in_out_flow

    def _petersen_matrix(self, params: dict) -> Float[Array, "10 6"]:
        return petersen_matrix(params)

    def _calculate_output(self, sol: dx.Solution,
                          args: tuple[dict, dx.CubicInterpolation]) -> dict:
        '''Calculate the output of the model.'''
        params, interp = args
        t, y = solution_arrays(sol)
        s = columns(STATES, y, *READ)
        T_op = jax.vmap(interp.evaluate)(t)[:,0]
        algebraic = algebraic_output(s, y, t, interp, T_op, params)
        return OrderedDict(state_output(t, y, STATES) | algebraic)
