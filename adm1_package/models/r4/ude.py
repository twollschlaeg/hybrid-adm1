"""ADM1-R4 UDE: the network corrects the hydrolysis and decay rates and the
degradabilities of the influent (utils/corrections.py)."""

from collections import OrderedDict
from typing import Callable

import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from adm1_package.models.chemistry import (
    M_CH4, M_CO2, henry_ch4, henry_co2, partial_pressure, pressure_law_outflow,
    water_vapour_pressure,
)
from adm1_package.models.simulation import (
    Hybrid, columns, solution_arrays, state_output, unpack,
)
from .model import READ, STATE_BOUNDS, STATES, algebraic_output, petersen_matrix


class ADM1R4UDE(Hybrid):
    '''ADM1-R4 model where an MLP is used to predict rate
    changes resulting from substrate concentrations.'''

    VARIANT = "r4"
    STATES = STATES
    STATE_BOUNDS = STATE_BOUNDS
    BOUNDED = False

    def _rhs(self, t, y: Float[Array, "10"],
             args: tuple[dict, dx.CubicInterpolation, Callable, jax.Array, tuple]
             ) -> Float[Array, "10"]:
        '''Evaluate right hand side of model ODE'''
        params, interp, mlp, key, scaling = args
        forcing = interp.evaluate(t)
        T_op = forcing[0]
        q_in = forcing[1]
        # the influent: S_TAN, S_h2o, X_ch, X_pr, X_li
        y_in = forcing[2:]
        mlp_out = self._network(T_op, q_in, y_in, params, mlp, key, scaling)

        K_H_ch4 = henry_ch4(T_op)
        K_H_co2 = henry_co2(T_op)
        p_h2o = water_vapour_pressure(T_op)

        y = jnp.clip(y, *self.STATE_BOUNDS) # avoid NaN and Inf values
        s = unpack(STATES, y)

        p_ch4 = partial_pressure(s.S_gas_ch4, M_CH4, T_op)
        p_co2 = partial_pressure(s.S_gas_co2, M_CO2, T_op)
        p_gas = p_ch4 + p_co2 + p_h2o
        q_gas = pressure_law_outflow(p_gas, params, self.BOUNDED)

        # The parameters the network sets (utils/corrections.py).
        corrected = self._corrections().resolve(mlp_out, params, T_op)
        rates = jnp.array([
            corrected['k_hyd_ch'] * s.X_ch,
            corrected['k_hyd_pr'] * s.X_pr,
            corrected['k_hyd_li'] * s.X_li,
            corrected['k_dec_Xbac'] * s.X_bac,
            params['k_La'] * (s.S_ch4 - 16 * (K_H_ch4 * p_ch4)),
            params['k_La'] * (s.S_IC - 44 * (K_H_co2 * p_co2))
        ])
        A = petersen_matrix(params)

        # The degradabilities scale the particulates of the influent, and the optional
        # TAN-inflow correction its first entry, S_TAN.
        degradabilities = jnp.stack([corrected['f_ch'], corrected['f_pr'], corrected['f_li']])
        inflow_factors = jnp.ones(5)
        inflow_factors = inflow_factors.at[2:5].set(degradabilities)
        if 'f_TAN_in' in corrected:
            inflow_factors = inflow_factors.at[0].set(corrected['f_TAN_in'])
        in_out_flow = jnp.concat([
            q_in / params['V_liq'] * (- y[:2]),
            q_in / params['V_liq'] * (inflow_factors * y_in - y[2:7]),
            q_in / params['V_liq']* (-y[7:8]),
            q_gas / params['V_gas'] * (-y[8:10])
        ], axis=0)

        return jnp.dot(A, rates) + in_out_flow

    def _petersen_matrix(self, params: dict) -> Float[Array, "10 6"]:
        return petersen_matrix(params)

    def _calculate_output(
            self, sol: dx.Solution,
            args: tuple[dict, dx.CubicInterpolation, Callable, jax.Array, tuple]
            ) -> dict:
        '''Calculate the output of the model.'''
        params, interp, mlp, key, scaling = args
        t, y = solution_arrays(sol)
        s = columns(STATES, y, *READ)
        forcing = jax.vmap(interp.evaluate)(t)
        T_op, network = self._corrections_along(forcing, params, mlp, key, scaling)
        algebraic = algebraic_output(s, y, t, interp, T_op, params)
        return OrderedDict(state_output(t, y, STATES) | algebraic
                           | self._network_output(*network))
