"""The full, mass-based ADM1 (Weinrich 2017), which R2-R4 reduce: sugars, amino acids,
long-chain fatty acids and hydrogen as further intermediates, and their degraders.

This module is the variant's one place for its states, its Petersen matrix and its
rates. The solve and the reported output are shared (models/simulation.py). It has no
hybrid.
"""

from collections import OrderedDict

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from adm1_package.models.chemistry import (
    C4_COMPETITION, DECAY_YIELDS, DH_CO2, DH_TAN, DH_W, M_CH4, M_CO2, M_H2, S_H_BOUNDS,
    henry_ch4, henry_co2, henry_h2, partial_pressure, ph_inhibition, pressure_law_outflow,
    proton_concentration, van_t_hoff, water_vapour_pressure,
)
from adm1_package.models.simulation import (
    Mechanistic, biogas, column, columns, solution_arrays, state_output, unpack,
)
from adm1_package.utils.gas import R_GAS, gas_transfer_rates, hydrogen_transfer_rate
from adm1_package.utils.initial_state import StateLayout

#: The state vector, for the dataset, the measured start and the calibration's warm-up.
STATES = StateLayout(
    names=('S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac', 'S_h2', 'S_ch4', 'S_IC',
           'S_TAN', 'S_h2o', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa', 'X_va', 'X_bu',
           'X_pro', 'X_ac', 'X_h2', 'S_an', 'S_cat', 'S_va_i', 'S_bu_i', 'S_pro_i', 'S_ac_i',
           'S_hco3_i', 'S_nh3', 'S_gas_h2', 'S_gas_ch4', 'S_gas_co2'),
    held=('S_TAN', 'S_h2o', 'S_an', 'S_cat'),
    acids=(('S_va', 'S_va_i', 102.0, 'K_a_va'), ('S_bu', 'S_bu_i', 88.0, 'K_a_bu'),
           ('S_pro', 'S_pro_i', 74.0, 'K_a_pro'), ('S_ac', 'S_ac_i', 60.0, 'K_a_ac')),
)

#: Activation energies in J/mol of the growth of X_su, X_aa, X_fa, X_va, X_bu, X_pro,
#: X_ac and X_h2, from the growth rates in Weinrich (2017); the rates are those at
#: T_GROWTH. The full ADM1's growth always depends on temperature.
GROWTH_ENERGIES = (33447.56500805, 13282.43293645, 20165.1320716, 16005.96578368,
                   16005.96578368, 17005.40065541, 27362.37923619, 0.0)
#: Reference temperature of the growth rates (35 C), K.
T_GROWTH = 308.15
#: Hydrogen partial pressure in bar at the start, which nothing measures.
P_H2_START = 2e-5


def growth_temperature_factors(T_op):
    '''Arrhenius factors of the eight growth rates at ``T_op``.'''
    R = jnp.array(R_GAS)
    energies_growth = jnp.array(GROWTH_ENERGIES)
    return jnp.exp(-energies_growth/ (R* 100) * (1/T_op- 1/T_GROWTH))


def transfer_rates(clipped, T_op, params):
    '''CH4, CO2 and H2 transfer along a solution of clipped states; the dissolved CO2 is
    S_IC less bicarbonate.'''
    def c(name):
        return column(STATES, clipped, name)
    rho_ch4, rho_co2 = gas_transfer_rates(
        c('S_ch4'), jnp.clip(c('S_IC') - c('S_hco3_i'), 0, None), c('S_gas_ch4'),
        c('S_gas_co2'), T_op, params['k_La'])
    return rho_ch4, rho_co2, hydrogen_transfer_rate(c('S_h2'), c('S_gas_h2'), T_op,
                                                    params['k_La'])


#: The states the algebraic outputs read, in this order.
READ = ('S_h2', 'S_TAN', 'S_IC', 'S_an', 'S_cat', 'S_va_i', 'S_bu_i', 'S_pro_i', 'S_ac_i',
        'S_hco3_i', 'S_nh3', 'S_gas_h2', 'S_gas_ch4', 'S_gas_co2')


class ADM1Base(Mechanistic):
    '''Base class for mass-based ADM1 model according to the thesis from S. Weinrich.

    ``options``: ``use_yield_coefficients`` (default True) builds the Petersen matrix
    from the yields in the parameters (Weinrich's Table A.8, with Table A.9's water);
    False takes the fixed coefficients of the thesis instead. The ``arrhenius`` option of
    the reduced variants is a no-op here: the full ADM1's growth rates always depend on
    temperature (:func:`growth_temperature_factors`), and its hydrolysis never does.
    '''

    VARIANT = "base"
    STATES = STATES
    DEFAULT_OPTIONS = {'use_yield_coefficients': True}
    #: The full ADM1 clips its states to [1e-12, 1e8] and bounds the proton
    #: concentration, the head-space pressure and the outflow.
    STATE_BOUNDS = (1e-12, 1e8)
    BOUNDED = True
    #: Nothing scores its total and volatile solids.
    REPORTS_SOLIDS = False

    def _rhs(self, t, y: Float[Array, "34"],
             args: tuple[dict, dx.CubicInterpolation]) -> Float[Array, "34"]:
        '''Right-hand side of the ODE system.
        This function computes the rate of change of the state variables y at time t.'''
        params, interp = args
        forcing = interp.evaluate(t)
        T_op = forcing[0]
        q_in = forcing[1]
        # the influent: S_va, S_bu, S_pro, S_ac, S_TAN, S_h2o, X_ch, X_pr, X_li
        y_in = forcing[2:]

        K_H_h2 = henry_h2(T_op)
        K_H_ch4 = henry_ch4(T_op)
        K_H_co2 = henry_co2(T_op)
        K_a_co2 = van_t_hoff(params['K_a_co2'], DH_CO2, T_op)
        K_a_TAN = van_t_hoff(params['K_a_TAN'], DH_TAN, T_op)
        K_w = van_t_hoff(params['K_w'], DH_W, T_op)
        p_h2o = water_vapour_pressure(T_op)
        arrhenius = growth_temperature_factors(T_op)
        K_I_nh3 = params['K_I_nh3']

        y = jnp.clip(y, *self.STATE_BOUNDS) # avoid NaN and Inf values
        s = unpack(STATES, y)

        S_nh4_i = jnp.clip(s.S_TAN - s.S_nh3, 0, None)
        S_co2 = jnp.clip(s.S_IC - s.S_hco3_i, 0, None)
        phi = (s.S_cat + S_nh4_i / 17 - s.S_hco3_i / 44 - s.S_ac_i / 60 - s.S_pro_i / 74
               - s.S_bu_i / 88 - s.S_va_i / 102 - s.S_an)
        S_H = proton_concentration(phi, K_w)
        if self.BOUNDED:
            S_H = jnp.clip(S_H, *S_H_BOUNDS)
        p_h2 = partial_pressure(s.S_gas_h2, M_H2, T_op)
        p_ch4 = partial_pressure(s.S_gas_ch4, M_CH4, T_op)
        p_co2 = partial_pressure(s.S_gas_co2, M_CO2, T_op)
        p_gas = p_h2 + p_ch4 + p_co2 + p_h2o
        q_gas = pressure_law_outflow(p_gas, params, self.BOUNDED)

        p = params
        I_TAN = s.S_TAN / (s.S_TAN + p['K_I_TAN'])
        I_h2_fa = p['K_I_fa'] / (p['K_I_fa'] + s.S_h2)
        I_h2_c4 = p['K_I_c4'] / (p['K_I_c4'] + s.S_h2)
        I_h2_pro = p['K_I_pro'] / (p['K_I_pro'] + s.S_h2)
        I_ph_aa = ph_inhibition(S_H, p, 'aa')
        I_ph_ac = ph_inhibition(S_H, p, 'ac')
        I_ph_h2 = ph_inhibition(S_H, p, 'h2')
        I_nh3 = K_I_nh3 / (K_I_nh3 + s.S_nh3)

        rates = jnp.array([
            p['k_hyd_ch'] * s.X_ch,
            p['k_hyd_pr'] * s.X_pr,
            p['k_hyd_li'] * s.X_li,
            (p['k_m_su'] * arrhenius[0] * s.S_su / (p['K_su'] + s.S_su)
             * s.X_su * I_ph_aa * I_TAN),
            (p['k_m_aa'] * arrhenius[1] * s.S_aa / (p['K_aa'] + s.S_aa)
             * s.X_aa * I_ph_aa * I_TAN),
            (p['k_m_fa'] * arrhenius[2] * s.S_fa / (p['K_fa'] + s.S_fa)
             * s.X_fa * I_ph_aa * I_TAN * I_h2_fa),
            (p['k_m_va'] * arrhenius[3] * s.S_va / (p['K_va'] + s.S_va) * s.X_va
             * s.S_va / (s.S_va + s.S_bu + C4_COMPETITION) * I_ph_aa * I_TAN * I_h2_c4),
            (p['k_m_bu'] * arrhenius[4] * s.S_bu / (p['K_bu'] + s.S_bu) * s.X_bu
             * s.S_bu / (s.S_bu + s.S_va + C4_COMPETITION) * I_ph_aa * I_TAN * I_h2_c4),
            (p['k_m_pro'] * arrhenius[5] * s.S_pro / (p['K_pro'] + s.S_pro) * s.X_pro
             * I_ph_aa * I_TAN * I_h2_pro),
            (p['k_m_ac'] * arrhenius[6] * s.S_ac / (p['K_ac'] + s.S_ac)
             * s.X_ac * I_ph_ac * I_TAN * I_nh3),
            (p['k_m_h2'] * arrhenius[7] * s.S_h2 / (p['K_h2'] + s.S_h2)
             * s.X_h2 * I_ph_h2 * I_TAN),
            p['k_dec_Xsu'] * s.X_su,
            p['k_dec_Xaa'] * s.X_aa,
            p['k_dec_Xfa'] * s.X_fa,
            p['k_dec_Xc4'] * s.X_va,
            p['k_dec_Xc4'] * s.X_bu,
            p['k_dec_Xpro'] * s.X_pro,
            p['k_dec_Xac'] * s.X_ac,
            p['k_dec_Xh2'] * s.X_h2,
            p['k_AB_va'] * (s.S_va_i * (p['K_a_va'] + S_H) - p['K_a_va'] * s.S_va),
            p['k_AB_bu'] * (s.S_bu_i * (p['K_a_bu'] + S_H) - p['K_a_bu'] * s.S_bu),
            p['k_AB_pro'] * (s.S_pro_i * (p['K_a_pro'] + S_H) - p['K_a_pro'] * s.S_pro),
            p['k_AB_ac'] * (s.S_ac_i * (p['K_a_ac'] + S_H) - p['K_a_ac'] * s.S_ac),
            p['k_AB_co2'] * (s.S_hco3_i * (K_a_co2 + S_H) - K_a_co2 * s.S_IC),
            p['k_AB_TAN'] * (s.S_nh3 * (K_a_TAN + S_H) - K_a_TAN * s.S_TAN),
            p['k_La'] * (s.S_h2 - 2 * (K_H_h2 * p_h2)),
            p['k_La'] * (s.S_ch4 - 16 * (K_H_ch4 * p_ch4)),
            p['k_La'] * (S_co2 - 44 * (K_H_co2 * p_co2))
        ])

        A = self._petersen_matrix(params)

        # S_TAN and S_h2o enter as fed, the particulates scaled by their degradabilities
        inflow_factors = jnp.array([1.0, 1.0, params['f_ch'],params['f_pr'], params['f_li']])
        in_out_flows = jnp.concat([
            q_in / params['V_liq'] * (- y[:3]),
            q_in / params['V_liq'] * (y_in[:4] - y[3:7]),
            q_in / params['V_liq'] * (- y[7:10]),
            q_in / params['V_liq'] * (inflow_factors * y_in[4:] - y[10:15]),
            q_in / params['V_liq'] * (-y[15:25]), # no flow for the solids
            0 * y[25:31], # no flow for the ions
            q_gas /params['V_gas'] * (-y[31:34]),
        ])

        return jnp.dot(A, rates) + in_out_flows

    def _petersen_matrix(self, params: dict) -> Float[Array, "34 28"]:
        '''Stoichiometric coefficients of the 34 states (rows) in the 28 processes
        (columns): hydrolysis of carbohydrates, proteins and lipids (0-2), growth of X_su,
        X_aa, X_fa, X_va, X_bu, X_pro, X_ac and X_h2 (3-10), their decay (11-18), the
        acid-base equilibria of valerate, butyrate, propionate, acetate, CO2 and ammonia
        (19-24), and the transfer of H2, CH4 and CO2 to the head space (25-27).'''
        A = jnp.zeros((34, 28))
        # The soluble states S_su ... S_h2o (rows 0-11) in hydrolysis and growth.
        if self.options['use_yield_coefficients']:
            Y = {name: params[name] for name in
                 ('Y_su', 'Y_aa', 'Y_fa', 'Y_va', 'Y_bu', 'Y_pro', 'Y_ac', 'Y_h2')}
            row = STATES.index
            A = A.at[row('S_su'), 0:4].set([1.111, 0, 0.13482, -1/Y['Y_su']])
            A = A.at[row('S_aa'), 1:5].set([1, 0, 0, -1/Y['Y_aa']])
            A = A.at[row('S_fa'), 2:6].set([0.95115, 0, 0, -1/Y['Y_fa']])
            A = A.at[row('S_va'), 4:7].set([(0.1726/Y['Y_aa'] - 0.1597), 0, -1/Y['Y_va']])
            A = A.at[row('S_bu'), 3:8].set([(0.0763/Y['Y_su'] - 0.1013),
                                            (0.2189/Y['Y_aa'] - 0.2025), 0, 0, -1/Y['Y_bu']])
            A = A.at[row('S_pro'), 3:9].set([(0.1903/Y['Y_su'] - 0.2526),
                                             (0.0505/Y['Y_aa'] - 0.0468), 0,
                                             (0.7274/Y['Y_va'] - 0.5052), 0, -1/Y['Y_pro']])
            A = A.at[row('S_ac'), 3:10].set([(0.4100/Y['Y_su'] - 0.5442),
                                             (0.5737/Y['Y_aa'] - 0.5398),
                                             (1.8852/Y['Y_fa'] - 0.9291),
                                             (0.5924/Y['Y_va'] - 0.4114),
                                             (1.3632/Y['Y_bu'] - 1.0618),
                                             (0.8086/Y['Y_pro'] - 0.7565), - 1/Y['Y_ac']])
            A = A.at[row('S_h2'), 3:11].set([(0.0255/Y['Y_su'] - 0.0339),
                                             (0.0116/Y['Y_aa'] - 0.0107),
                                             (0.1085/Y['Y_fa'] - 0.0535),
                                             (0.0385/Y['Y_va'] - 0.0267),
                                             (0.0458/Y['Y_bu'] - 0.0356),
                                             (0.0819/Y['Y_pro'] - 0.0766), 0, - 1/Y['Y_h2']])
            A = A.at[row('S_ch4'), 9:11].set([(0.2671/Y['Y_ac'] - 0.3546),
                                              (1.9895/Y['Y_h2'] - 0.3546)])
            A = A.at[row('S_IC'), 2:11].set([-0.02933, (0.3731/Y['Y_su'] - 0.4977),
                                             (0.2739/Y['Y_aa'] - 0.3350),
                                             -(0.0172/Y['Y_fa'] + 0.5836),
                                             -(0.0102/Y['Y_va'] + 0.4419), -0.3891,
                                             (0.5971/Y['Y_pro'] - 0.8365),
                                             (0.7329/Y['Y_ac'] - 0.9727),
                                             -(5.4579/Y['Y_h2'] + 0.9727)])
            A = A.at[row('S_TAN'), 3:11].set([-0.1506, (0.1949/Y['Y_aa'] - 0.1506), -0.1506,
                                              -0.1506, -0.1506, - 0.1506, -0.1506, -0.1506])
            # Weinrich's (2017) Table A.8 leaves out the water hydrolysis takes up; the
            # first three entries are Table A.9's, which close those mass balances
            A = A.at[row('S_h2o'), 0:11].set([-0.1111, 0, -0.0566,
                                              (-0.0753/Y['Y_su'] +0.5777),
                                              (-0.4962/Y['Y_aa'] + 0.4362),
                                              (-0.9766/Y['Y_fa'] + 0.7167),
                                              (-0.3481/Y['Y_va'] + 0.5358),
                                              (-0.4089/Y['Y_bu'] + 0.6371),
                                              (-0.4876/Y['Y_pro'] + 0.8202),
                                              0.4778, (4.4683/Y['Y_h2'] + 0.4778)])
        else:
            # Fixed coefficients for the thesis' yields (its substrate entries are -1/Y).
            B = jnp.array([
                [1.111, 0, 0.13482, -13.2724, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, -11.5665, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.95115, 0, 0, -8.2136, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1.8371, 0, -11.5757, 0, 0, 0, 0],
                [0, 0, 0, 0.9113, 2.3289, 0, 0, -12.9817, 0, 0, 0],
                [0, 0, 0, 2.2734, 0.53795, 0, 7.9149, 0, -23.3892, 0, 0],
                [0, 0, 0, 4.8975,  6.1053, 14.5554, 6.4459, 16.6347, 18.1566, -26.5447, 0],
                [0, 0, 0, 0.30475, 0.12297, 0.83761, 0.41881, 0.55841, 1.8392, 0, -2.9703],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 6.7367, 5.5548],
                [0, 0, -0.0293, 4.4571, 2.8335, -0.72457, -0.55945, -0.38907, 13.1283, 18.4808, -17.1839],
                [0, 0, 0, -0.1506, 2.1033, -0.1506, -0.1506, -0.1506, -0.1506, -0.1506, -0.1506],
                [-0.1111, 0, -0.0566, -0.4211, -5.3026, -7.3043, -3.4940, -4.6718, -10.5843, 0.4778, 13.7499]
            ])
            A = A.at[:12, :11].set(B)
        # S_h2, S_ch4 and S_IC leave to the head space
        A = A.at[7:10, 25:28].set(-1 * jnp.identity(3))
        # The particulates X_ch, X_pr, X_li are hydrolysed, the biomass X_su ... X_h2 grows
        # (rows 12-22, processes 0-10) ...
        A = A.at[12:23, :11].set(
            jnp.identity(11) * jnp.array([-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1])
            )
        # ... and decays into the particulates (processes 11-18).
        d_ch, d_pr, d_li = DECAY_YIELDS
        A = A.at[12:23, 11:19].set(jnp.array([
            [d_ch] * 8,
            [d_pr] * 8,
            [d_li] * 8,
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1]
            ]))
        # The ionized acids, bicarbonate and ammonia in their acid-base equilibria, and
        # the head space receiving H2, CH4 and CO2 (rows 25-33, processes 19-27).
        A = A.at[25:, 19:].set(
            jnp.identity(9)
            * jnp.array([-1, -1, -1, -1, -1, -1, params['V_liq']/params['V_gas'],
                         params['V_liq']/params['V_gas'], params['V_liq']/params['V_gas']])
            )
        return A

    def _calculate_output(self, sol: dx.Solution,
                          args: tuple[dict, dx.CubicInterpolation]) -> dict:
        '''Calculate the output of the model: its states, pH, partial pressures, gas
        flows and inhibition terms along the solution.'''
        params, interp = args
        t, y = solution_arrays(sol)
        s = columns(STATES, y, *READ)
        T_op = jax.vmap(interp.evaluate)(t)[:,0]

        K_w = van_t_hoff(params['K_w'], DH_W, T_op)
        p_h2o = water_vapour_pressure(T_op)
        K_I_nh3 = params['K_I_nh3']

        S_nh4_i = s.S_TAN - s.S_nh3
        S_co2 = s.S_IC - s.S_hco3_i
        phi = (s.S_cat + S_nh4_i / 17 - s.S_hco3_i / 44 - s.S_ac_i / 60 - s.S_pro_i / 74
               - s.S_bu_i / 88 - s.S_va_i / 102 - s.S_an)
        S_H = proton_concentration(phi, K_w)
        p_h2 = partial_pressure(s.S_gas_h2, M_H2, T_op)
        p_ch4 = partial_pressure(s.S_gas_ch4, M_CH4, T_op)
        p_co2 = partial_pressure(s.S_gas_co2, M_CO2, T_op)
        p_gas = p_h2 + p_ch4 + p_co2 + p_h2o
        pH = -jnp.log10(S_H)
        q_gasout, ch4_out = biogas(y, t, interp, T_op, params, (p_ch4, p_co2, p_h2), p_h2o,
                                   self.STATE_BOUNDS, transfer_rates)

        p = params
        inhibition = {
            'I_TAN': s.S_TAN / (s.S_TAN + p['K_I_TAN']),
            'I_h2_fa': p['K_I_fa'] / (p['K_I_fa'] + s.S_h2),
            'I_h2_c4': p['K_I_c4'] / (p['K_I_c4'] + s.S_h2),
            'I_h2_pro': p['K_I_pro'] / (p['K_I_pro'] + s.S_h2),
            'I_ph_aa': ph_inhibition(S_H, p, 'aa'),
            'I_ph_ac': ph_inhibition(S_H, p, 'ac'),
            'I_ph_h2': ph_inhibition(S_H, p, 'h2'),
            'I_nh3': K_I_nh3 / (K_I_nh3 + s.S_nh3),
        }
        algebraic = {
            'T_op': T_op, 'pH': pH, 'S_nh4': S_nh4_i, 'S_H': S_H, 'S_co2': S_co2, 'p_h2': p_h2,
            'p_ch4': p_ch4, 'p_co2': p_co2, 'p_gas': p_gas, 'q_gasout': q_gasout, 'ch4_out': ch4_out
        }
        return OrderedDict(state_output(t, y, STATES) | algebraic | inhibition)

    @eqx.filter_jit
    def __call__(self, params: dict, initial: Float[Array, "34"],
                 input_data: dict) -> tuple[dict, dx.Solution]:
        '''Solve the model ODE (jitted); see :meth:`Mechanistic.__call__`.'''
        return Mechanistic.__call__(self, params, initial, input_data)
