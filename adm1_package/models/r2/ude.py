"""ADM1-R2 UDE: the network corrects the hydrolysis, growth and decay rates and the
degradabilities of the influent (utils/corrections.py)."""

from collections import OrderedDict
from typing import Callable

import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from adm1_package.models.chemistry import (
    C4_COMPETITION, DH_CO2, DH_TAN, DH_W, M_CH4, M_CO2, S_H_BOUNDS, henry_ch4, henry_co2,
    partial_pressure, ph_inhibition, pressure_law_outflow, proton_concentration, van_t_hoff,
    water_vapour_pressure,
)
from adm1_package.models.simulation import (
    Hybrid, columns, solution_arrays, state_output, unpack,
)
from .model import READ, STATE_BOUNDS, STATES, algebraic_output, petersen_matrix


class ADM1R2UDE(Hybrid):
    '''ADM1-R2 UDE model.'''

    VARIANT = "r2"
    STATES = STATES
    STATE_BOUNDS = STATE_BOUNDS
    BOUNDED = True

    def _rhs(self, t, y: Float[Array, "26"],
             args: tuple[dict, dx.CubicInterpolation, Callable, jax.Array, tuple]
             ) -> Float[Array, "26"]:
        '''Evaluate right hand side of model ODE'''
        params, interp, mlp, key, scaling = args
        forcing = interp.evaluate(t)
        T_op = forcing[0]
        q_in = forcing[1]
        # the influent: S_va, S_bu, S_pro, S_ac, S_TAN, S_h2o, X_ch, X_pr, X_li
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

        # The half-saturation constants: the campaign's own, else the literature
        # (utils/corrections.py). Valerate and butyrate degraders decay at rates of their
        # own (the mechanistic R2 shares k_dec_Xc4).
        k, p = corrected, params
        rates = jnp.array([
            k['k_hyd_ch'] * s.X_ch,
            k['k_hyd_pr'] * s.X_pr,
            k['k_hyd_li'] * s.X_li,
            k['k_m_va'] * s.X_va * s.S_va / (s.S_va + s.S_bu + C4_COMPETITION) * s.S_va / (k['K_va'] + s.S_va) * I_va,
            k['k_m_bu'] * s.X_bu * s.S_bu / (s.S_bu + s.S_va + C4_COMPETITION) * s.S_bu / (k['K_bu'] + s.S_bu) * I_bu,
            k['k_m_pro'] * s.X_pro * s.S_pro / (k['K_pro'] + s.S_pro) * I_pro,
            k['k_m_ac'] * s.X_ac * s.S_ac / (k['K_ac'] + s.S_ac) * I_ac,
            k['k_dec_Xbac'] * s.X_bac,
            k['k_dec_Xva'] * s.X_va,
            k['k_dec_Xbu'] * s.X_bu,
            k['k_dec_Xpro'] * s.X_pro,
            k['k_dec_Xac'] * s.X_ac,
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

        degradabilities = jnp.stack([k['f_ch'], k['f_pr'], k['f_li']])
        # The optional TAN-inflow correction scales S_TAN, entry 4 of the influent.
        soluble_in = y_in[4:6]
        if 'f_TAN_in' in corrected:
            soluble_in = soluble_in.at[0].multiply(corrected['f_TAN_in'])
        in_out_flow = jnp.concat([
            q_in / params['V_liq'] * (y_in[:4] - y[:4]),
            q_in / params['V_liq'] * (- y[4:6]),
            q_in / params['V_liq'] * (soluble_in - y[6:8]),
            q_in / params['V_liq'] * (degradabilities * y_in[6:] - y[8:11]),
            q_in / params['V_liq'] * (-y[11:16]),
            q_in / params['V_liq'] * (-y[16:18]),
            0 * y[18:24],
            q_gas / params['V_gas'] * (-y[24:26])
        ], axis=0)

        return jnp.dot(A, rates) + in_out_flow

    def _petersen_matrix(self, params: dict) -> Float[Array, "26 20"]:
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
        algebraic = algebraic_output(s, y, t, interp, T_op, params, K_I_nh3)
        return OrderedDict(state_output(t, y, STATES, read=s) | algebraic
                           | self._network_output(*network))
