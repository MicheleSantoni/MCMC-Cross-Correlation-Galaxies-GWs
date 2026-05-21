#-----------------------------------------------------------------------------------------
#                                CONFIGURATION FILE: MODEL
#-----------------------------------------------------------------------------------------
COSMO_PARAMS = {
    # --- Background (CLASS wants Omega_b + Omega_cdm, not Omega_m directly) ---
    'h': 0.67810,
    'T_cmb': 2.7255,
    'YHe': 0.2454,
    'N_ur': 3.044,
    'N_ncdm': 0,
    'omega_b': 0.0220704,
    'omega_m': 0.142544079,
    'omega_cdm':0.120473679,
    'Omega_k': 0.0,

    # --- Primordial ---
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,

    # --- Model ---
    'dark_energy_model': 'fluid',   # dynamic back ---> need to specify w0, wa 


}

# --- Gravity Model: propto_omega ---
# In EFTCAMB: RPH (Remapped Parametrized Horndeski)
# EFTflag=4 (FullMappingEFT) + FullMappingEFTmodel=1 (RPH)
# alpha functions: kineticity (K), braiding (B), Planck mass run rate (M), tensor speed excess (T)
#    EFTwDE  = 2  →  CPL:  w(a) = w0 + wa*(1-a)

EFTCAMB_PARAMS = {
    'EFTflag'          : 2,
    'AltParEFTmodel'   : 1,

    'RPHintegratefromtoday' : False,
    'RPHusealphaM'          : True,   # usa alpha_M direttamente invece di M^2

    # suffisso _ODE — versione solutore ODE
    'RPHkineticitymodel_ODE' : 2,     # propto Omega_DE
    'RPHbraidingmodel_ODE'   : 2,
    'RPHalphaMmodel_ODE'     : 2,
    'RPHtensormodel_ODE'     : 2,

    # ampiezze — suffisso _ODE0
    'RPHkineticity_ODE0' : 0.0,   # alpha_K
    'RPHbraiding_ODE0'   : 0.0,   # alpha_B
    'RPHalphaM_ODE0'     : 0.0,   # alpha_M
    'RPHtensor_ODE0'     : 0.0,   # alpha_T

    'EFTCAMB_back_turn_on'  : 1e-8,
    'EFTCAMB_turn_on_time'  : 1e-8,
    'EFTCAMB_skip_stability': False,
    'feedback_level'        : 0,
}



