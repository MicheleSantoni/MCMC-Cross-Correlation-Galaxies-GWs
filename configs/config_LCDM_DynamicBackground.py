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
    'dark_energy_model': 'EFTCAMB',   # dynamic background
}

EFTCAMB_PARAMS = {
    # ── Flag principale ──────────────────────────────────────────────────────
    'EFTflag'      : 1,    # Pure EFT
    'PureEFTmodel' : 1,    # standard

    # ── Background DE ────────────────────────────────────────────────────────
    'EFTwDE' : 2,          # CPL: w(a) = w0 + wa*(1-a)
    'EFTw0'  : -1.0,       # ΛCDM background
    'EFTwa'  : 0.0,        # static

    # ── Omega(a): effective gravitational coupling ──────────────────────────
    'PureEFTmodelOmega' : 0,    # 0=zero, 1=costante, 2=lineare, 3=power law
    'EFTOmega0'         : 0.0,  # Omega today
    'EFTOmegaExp'       : 0.0,  # exponent (used only if model=3 o 4)

    # ── Gamma functions: perturbazioni oltre Omega ───────────────────────────
    #    = 0 to ignore extra-terms (GR-like)
    'PureEFTmodelGamma1' : 0,
    'PureEFTmodelGamma2' : 0,
    'PureEFTmodelGamma3' : 0,
    'PureEFTmodelGamma4' : 0,
    'PureEFTmodelGamma5' : 0,
    'PureEFTmodelGamma6' : 0,

    # ── Restrizione a Horndeski (opzionale) ──────────────────────────────────
    #    If True, gamma4/5/6  would be treated internally (Horndeski condition)
    'PureEFTHorndeski' : False,
}


#EFTCAMB_PARAMS = {
#    'EFTflag'           : 0}

"""
EFTCAMB_PARAMS = {
    'EFTflag'              : 2,
    'FullMappingEFTmodel'  : 1,   # RPH

    # ── Background CPL: w(a) = w0 + wa*(1-a) ─────────────────────────────
    'EFTwDE' : 2,        # CPL
    'EFTw0'  : -1.0,     # ΛCDM background
    'EFTwa'  : 0.0,      # static

    # ── Alpha functions  ───────────────────────────────────────
    'RPHmassPmodel'        : 2,
    'RPHkineticitymodel'   : 2,
    'RPHbraidingmodel'     : 2,
    'RPHtensormodel'       : 2,

    'RPHmassP0'      : 0.0,   # c_M
    'RPHkineticity0' : 0.0,   # c_K
    'RPHbraiding0'   : 0.0,   # c_B
    'RPHtensor0'     : 0.0,   # c_T
}


"""
