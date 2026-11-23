"""The full ADM1's samples (models/dataset.py), with hydrogen in the head space."""

import jax.numpy as jnp

from adm1_package.models.chemistry import M_CH4, M_CO2, M_H2
from adm1_package.models.dataset import ADM1Dataset
from adm1_package.utils.gas import (
    R_GAS,
    dissolved_ch4_at_equilibrium,
    dissolved_h2_at_equilibrium,
    initial_gas_phase,
)
from .model import P_H2_START, STATES


def correct_gas_ivs(initial_values, params, input_data, targets):
    '''Corrects the initial values of the gas concentrations using the initial values of
    the output data.

    The first measured biogas and methane flows are norm volumes, as the model reports
    them, so the gas phase is the one whose outflow reproduces them
    (:func:`adm1_package.utils.gas.initial_gas_phase`), as for the reduced variants. The
    head space also holds hydrogen, which nothing measures (``P_H2_START``). The
    dissolved methane and hydrogen start in equilibrium with the head space, as R2-R4's
    methane does. A fitted start value of hydrogen can otherwise sit many orders of
    magnitude above saturation.'''
    T_op = input_data.evaluate(0)[0]
    p_h2 = P_H2_START
    p_ch4, p_co2 = initial_gas_phase(targets[0,0], targets[0,1], T_op, params, p_h2=p_h2)
    R = jnp.array(R_GAS)
    S_gas_ch4 = p_ch4 * M_CH4 / (R * T_op)
    S_gas_co2 = p_co2 * M_CO2 / (R * T_op)
    S_gas_h2 = p_h2 * M_H2 / (R * T_op)

    initial_values = initial_values.at[STATES.index('S_gas_ch4')].set(S_gas_ch4)
    initial_values = initial_values.at[STATES.index('S_gas_co2')].set(S_gas_co2)
    initial_values = initial_values.at[STATES.index('S_gas_h2')].set(S_gas_h2)
    initial_values = initial_values.at[STATES.index('S_ch4')].set(
        dissolved_ch4_at_equilibrium(p_ch4, T_op))
    initial_values = initial_values.at[STATES.index('S_h2')].set(
        dissolved_h2_at_equilibrium(p_h2, T_op))

    return initial_values


class BaseDataset(ADM1Dataset):
    '''The full ADM1's samples (see ``ADM1Dataset``).

    Its calibration was fitted with three choices of its own, kept here: it reads every
    parameter but those of the variants' hydrolysis kinetics (``DROPPED_PARAMETERS``),
    its outputs strictly and every point scored (``MASK_MISSING = False``), and its gas
    phase with hydrogen (:func:`correct_gas_ivs`).
    '''

    STATES = STATES
    INFLUENT = ('S_va', 'S_bu', 'S_pro', 'S_ac', 'S_TAN', 'S_h2o', 'X_ch', 'X_pr', 'X_li')
    DROPPED_PARAMETERS = ('k_dis', 'K_ch', 'K_pr', 'K_li', 'hillcoefficient_nh3')
    IV_PREFIX = 'base'
    MEASURED_STATES = ('S_ac', 'S_va', 'S_bu', 'S_pro', 'S_TAN', 'S_h2o')
    MASK_MISSING = False
    DEFAULT_OUTPUT = ('q_gasout', 'ch4_out', 'S_ac', 'S_va', 'S_bu', 'S_pro', 'S_TAN',
                      'S_h2o', 'pH')

    def _initial_gas_phase(self, initial_values, params, forcing, measurements):
        return correct_gas_ivs(initial_values, params, forcing, measurements)

    def _start_measurements(self, measured, output, measurements) -> dict:
        start = super()._start_measurements(measured, output, measurements)
        # the hydrogen in the head space and in the liquid is set with the rest of the gas
        # phase, which start_measurements knows only for R2-R4
        hydrogen = jnp.array([STATES.index('S_gas_h2'), STATES.index('S_h2')])
        start['measured'] = start['measured'].at[hydrogen].set(1.0)
        return start
