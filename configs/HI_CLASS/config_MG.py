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
    #'Omega_Lambda': 0.0,
    #'Omega_fld': 0.0,
    'Omega_smg': 0.7,


    # --- Primordial Power Spectrum ---
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,

    # --- Gravity Model ---
    # x_k, x_b, x_m, x_t, M*^2_ini
    'gravity_model': 'propto_omega',
    'parameters_smg': '1.0, 0.1, 0.1, 0.0, 1.0',
    #'expansion_model' : 'wowa',
    #'expansion_smg' : '0.7, -1.0, 0.0',  #Omega_smg, w0, wa
    'expansion_model':'lcdm',
    'expansion_smg': '0.7',
    #'want_lcmb_full_limber' : 'no',

    # --- Modified Gravity: Stability ---
    'output_background_smg': 1,    #1 -> alpha functions, stability parameters (c_s^2, D)
    #2e) Parameter controling how much smg information do you want on the background.dat file
    #(works as _verbose parameters, all lower priority are included)
    #0 -> rho,p
    #1 -> alpha functions, stability parameters (c_s^2, D)
    #2 -> phi, phi', phi'' and Friedmann constraint equations (only covariant theories)
    #3 -> functions useful to calculate perturbations (lambda_i, ...)

    'method_qs_smg': 'quasi_static',
    'skip_stability_tests_smg': 'no',
    'cs2_safe_smg': 0.0,
    'D_safe_smg': 0.0,
    'ct2_safe_smg': 0.0,
    'M2_safe_smg': 0.0,
    'a_min_stability_test_smg': 1e-3,
    'kineticity_safe_smg': 1e-4,
    #'pert_initial_conditions_smg': 'ext_field_attr',


    # --- Reionization / Thermodynamics ---
    "recombination": "RECFAST",
    "reio_parametrization": "reio_camb",

    # --- Power spectrum / Fourier ---
    "P_k_max_h/Mpc": 10.0,
    "z_pk": 0,
    "z_max_pk": 7.0,
    "non_linear": "halofit",
    #'extra_metric_transfer_functions': 'no',
    "output": "mPk",

    # --- Sampling and Integration ---
    'start_small_k_at_tau_c_over_tau_h': 1e-4,
    'start_large_k_at_tau_h_over_tau_k': 1e-4,
    'perturbations_sampling_stepsize': 0.10,
    'l_logstep': 1.045,
    'l_linstep': 50,

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



