#-----------------------------------------------------------------------------------------
#                                CONFIGURATION FILE: MODEL
#-----------------------------------------------------------------------------------------
COSMO_PARAMS = {
     # --- Background & Thermodynamics ---
    'h': 0.67810,
    'T_cmb': 2.7255,
    'YHe': 0.2454,  # replaced invalid 'BBN' , 0.2454
    'N_ur': 3.044,
    'N_ncdm':0,
    'omega_m':0.142544079,
    'omega_b':0.0220704,
    'Omega_k': 0.0,

    # --- Primordial Power Spectrum ---
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,

    # --- Reionization / Thermodynamics ---
    "recombination": "RECFAST",
    "reio_parametrization": "reio_camb",

    # --- Power spectrum / Fourier ---
    "P_k_max_h/Mpc": 10.0,
    "z_pk": 0,
    "z_max_pk": 7.0,
    #"non_linear": "halofit",
    "output": "mPk",

    # --- Sampling and Integration ---
    #'start_small_k_at_tau_c_over_tau_h': 1e-4,
    #'start_large_k_at_tau_h_over_tau_k': 1e-4,
    #'perturbations_sampling_stepsize': 0.10,
    #'l_logstep': 1.045,
    #'l_linstep': 50,

    # --- Verbosity (opzionale) ---
    "input_verbose": 0,
    "background_verbose": 0,
    "thermodynamics_verbose": 0,
    "perturbations_verbose": 0,
    "transfer_verbose": 0,
    "primordial_verbose": 0,
    "harmonic_verbose": 0,
    "fourier_verbose": 0,
    "output_verbose": 0,
}



