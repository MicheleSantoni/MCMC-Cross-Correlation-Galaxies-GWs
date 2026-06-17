#-----------------------------------------------------------------------------------------
#                                CONFIGURATION FILE: MODEL
#-----------------------------------------------------------------------------------------
COSMO_PARAMS = {
    # --- Background ---
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
    'dark_energy_model':'EFTCAMB',
}

# --- Gravity Model: propto_omega ---
EFTCAMB_PARAMS = {
    'EFTflag'          : 2,
    'AltParEFTmodel'   : 1,
    'RPHintegratefromtoday' : False,
    'RPHusealphaM'          : True,
    # Modelli ODE
    'RPHkineticitymodel_ODE' : 2,
    'RPHbraidingmodel_ODE'   : 2,
    'RPHalphaMmodel_ODE'     : 2,
    'RPHtensormodel_ODE'     : 2,
    # Fiduciali
    'RPHkineticity_ODE0' : 1.0,
    'RPHbraiding_ODE0'   : 0.0,
    'RPHalphaM_ODE0'     : 0.0,
    'RPHtensor_ODE0'     : 0.0,
    # Dynamic Background
    'RPHwDE'             : 2,      # CPL
    'RPHw0'              : -1.0,
    'RPHwa'              : 0.0,

    'EFTCAMB_stability_time'     : 1./(1.+7.), #redshift range
    'feedback_level'             : 0,
}


