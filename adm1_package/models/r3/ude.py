"""ADM1-R3 UDE: the network corrects the hydrolysis, growth and decay rates and the
degradabilities of the influent (utils/corrections.py)."""

from collections import OrderedDict
from typing import Callable

import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from adm1_package.models.chemistry import (
    DH_CO2, DH_TAN, DH_W, M_CH4, M_CO2, S_H_BOUNDS, henry_ch4, henry_co2, partial_pressure,
    ph_inhibition, pressure_law_outflow, proton_concentration, van_t_hoff,
    water_vapour_pressure,
)
from adm1_package.models.simulation import (
    Hybrid, columns, solution_arrays, state_output, unpack,
)
from .model import READ, STATES, algebraic_output, petersen_matrix


class ADM1R3UDE(Hybrid):
    '''ADM1-R3 UDE model.'''

    VARIANT = "r3"
    STATES = STATES
    #: Unlike the mechanistic R3, the hybrid clips its states to [1e-12, 1e8] and
    #: bounds the proton concentration, the head-space pressure and the outflow.
    STATE_BOUNDS = (1e-12, 1e8)
    BOUNDED = True

    def _rhs(self, t, y: Float[Array, "17"],
             args: tuple[dict, dx.CubicInterpolation, Callable, jax.Array, tuple]
             ) -> Float[Array, "17"]:
        '''Evaluate right hand side of model ODE'''
        params, interp, mlp, key, scaling = args
        forcing = interp.evaluate(t)
        T_op = forcing[0]
        q_in = forcing[1]
        # the influent: S_ac, S_TAN, S_h2o, X_ch, X_pr, X_li
        y_in = forcing[2:]
        mlp_out = self._network(T_op, q_in, y_in, params, mlp, key, scaling)
        # The parameters the network sets (utils/corrections.py).
        corrected = self._corrections().resolve(mlp_out, params, T_op)

        K_H_ch4 = henry_ch4(T_op)
        K_H_co2 = henry_co2(T_op)
        K_a_co2 = van_t_hoff(params['K_a_co2'], DH_CO2, T_op)
        K_a_TAN = van_t_hoff(params['K_a_TAN'], DH_TAN, T_op)
        K_w = van_t_hoff(params['K_w'], DH_W, T_op)
        p_h2o = water_vapour_pressure(T_op)
        # Network-corrected under the nh3_inhibition option, else the literature
        # constant, which is shared by every campaign.
        K_I_nh3 = corrected.get('K_I_nh3', params['K_I_nh3'])

        y = jnp.clip(y, *self.STATE_BOUNDS) # avoid numerical problems
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

        # K_ac: the campaign's own, else the literature (utils/corrections.py).
        k = corrected
        rates = jnp.array([
            k['k_hyd_ch'] * s.X_ch,
            k['k_hyd_pr'] * s.X_pr,
            k['k_hyd_li'] * s.X_li,
            k['k_m_ac'] * s.X_ac * s.S_ac / (k['K_ac'] + s.S_ac) * I_ac,
            k['k_dec_Xbac'] * s.X_bac,
            k['k_dec_Xac'] * s.X_ac,
            params['k_AB_ac'] * (s.S_ac_i * (params['K_a_ac'] + S_H) - params['K_a_ac'] * s.S_ac),
            params['k_AB_co2'] * (s.S_hco3_i * (K_a_co2 + S_H) - K_a_co2 * s.S_IC),
            params['k_AB_TAN'] * (s.S_nh3 * (K_a_TAN + S_H) - K_a_TAN * s.S_TAN),
            params['k_La'] * (s.S_ch4 - 16 * (K_H_ch4 * p_ch4)),
            params['k_La'] * (S_co2 - 44 * (K_H_co2 * p_co2))
        ])
        A = petersen_matrix(params)

        degradabilities = jnp.stack([k['f_ch'], k['f_pr'], k['f_li']])
        # The optional TAN-inflow correction scales S_TAN, entry 1 of the influent.
        soluble_in = y_in[1:3]
        if 'f_TAN_in' in corrected:
            soluble_in = soluble_in.at[0].multiply(corrected['f_TAN_in'])
        in_out_flow = jnp.concat([
            q_in / params['V_liq'] * (y_in[:1] - y[:1]),
            q_in / params['V_liq'] * (- y[1:3]),
            q_in / params['V_liq'] * (soluble_in - y[3:5]),
            q_in / params['V_liq'] * (degradabilities * y_in[3:6] - y[5:8]),
            q_in / params['V_liq'] * (-y[8:12]),
            0 * y[12:15],
            q_gas / params['V_gas'] * (-y[15:17])
        ], axis=0)

        return jnp.dot(A, rates) + in_out_flow

    def _petersen_matrix(self, params: dict) -> Float[Array, "17 11"]:
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
        corrected = network[1]
        # Network-corrected under the nh3_inhibition option, else the literature constant.
        K_I_nh3 = corrected.get('K_I_nh3', params['K_I_nh3'])
        algebraic = algebraic_output(s, y, t, interp, T_op, params, K_I_nh3,
                                     self.STATE_BOUNDS)
        return OrderedDict(state_output(t, y, STATES) | algebraic
                           | self._network_output(*network))
