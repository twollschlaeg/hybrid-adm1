"""The ADM1-R2 samples (models/dataset.py)."""

from adm1_package.models.dataset import ADM1Dataset
from .model import STATES


class R2Dataset(ADM1Dataset):
    '''The ADM1-R2 samples of the LaRA and/or rice-bran reactors (see ``ADM1Dataset``).'''

    STATES = STATES
    INFLUENT = ('S_va', 'S_bu', 'S_pro', 'S_ac', 'S_TAN', 'S_h2o', 'X_ch', 'X_pr', 'X_li')
    PARAMETERS = ('K_a_co2', 'K_a_TAN', 'K_w', 'K_I_nh3', 'k_p', 'p_norm',
                  'K_I_TAN', 'n_ac', 'pK_u_ac', 'pK_l_ac', 'n_aa', 'pK_l_aa',
                  'pK_u_aa', 'k_hyd_ch', 'k_hyd_pr', 'k_hyd_li', 'k_m_va',
                  'K_va', 'k_m_bu', 'K_bu', 'k_m_pro', 'K_pro', 'k_m_ac', 'K_ac',
                  'k_dec_Xbac', 'k_dec_Xc4', 'k_dec_Xpro',
                  'k_dec_Xac', 'k_AB_va', 'K_a_va', 'k_AB_bu', 'K_a_bu',
                  'k_AB_pro', 'K_a_pro', 'k_AB_ac', 'K_a_ac', 'k_AB_co2',
                  'k_AB_TAN', 'k_La', 'V_liq', 'V_gas', 'f_ch', 'f_pr', 'f_li')
    IV_PREFIX = 'R2'
    MEASURED_STATES = ('S_ac', 'S_va', 'S_bu', 'S_pro', 'S_TAN', 'S_h2o')
    DEFAULT_OUTPUT = ('q_gasout', 'ch4_out', 'S_ac', 'S_va', 'S_bu', 'S_pro', 'S_TAN',
                      'S_h2o', 'pH')
