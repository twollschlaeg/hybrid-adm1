"""The ADM1-R4 samples (models/dataset.py)."""

from adm1_package.models.dataset import ADM1Dataset
from .model import STATES


class R4Dataset(ADM1Dataset):
    '''The ADM1-R4 samples of the LaRA and/or rice-bran reactors (see ``ADM1Dataset``).'''

    STATES = STATES
    INFLUENT = ('S_TAN', 'S_h2o', 'X_ch', 'X_pr', 'X_li')
    PARAMETERS = ('k_p', 'p_norm', 'k_hyd_ch', 'k_hyd_pr', 'k_hyd_li', 'k_dec_Xbac',
                  'k_La', 'V_liq', 'V_gas', 'f_ch', 'f_pr', 'f_li')
    IV_PREFIX = 'R4'
    MEASURED_STATES = ('S_TAN', 'S_h2o')
    # R4 transfers all of S_IC, so it starts in equilibrium with the gas phase as well.
    S_IC_FROM_GAS = True
    DEFAULT_OUTPUT = ('q_gasout', 'ch4_out', 'S_TAN', 'S_h2o')
