# MCMC Cross-Correlation between Galaxies and Gravitational Waves

This repository contains a Python-based MCMC pipeline designed to probe deviations from General Relativity through the angular cross-correlation between galaxy catalogs and gravitational-wave (GW) events.

The code combines a modified Boltzmann solver (hi_class), custom window function construction, Limber integration, and Bayesian parameter inference via Markov Chain Monte Carlo (MCMC).

---

## Scientific Motivation

The large-scale structure of the Universe and gravitational-wave observations provide complementary probes of cosmology and gravity. 

This project computes angular power spectra between:

- Galaxy number density fluctuations  
- Gravitational-wave luminosity distance fluctuations  

within a Limber approximation framework, allowing constraints on modified gravity parameters (e.g., running Planck mass).

The pipeline is modular and configurable, enabling flexible exploration of cosmological models.

---

## Main Features

- Custom construction of galaxy and GW window functions
- Support for modified gravity parameterizations (e.g. α_M, M2_running)
- Limber-based angular power spectrum computation
- hi_class integration for background and transfer functions
- Full likelihood evaluation
- Configurable MCMC sampling
- Resume functionality for interrupted chains
- Structured output management

---

## Repository Structure
MCMC_template/
│
├── colibri/ # Limber integrator and window function modules
├── config/ # Configuration files (cosmology + MCMC settings)
├── cosmology/ # Cosmological parameter handling
├── chains/ # MCMC output (ignored by git)
├── outputs/ # Generated spectra and intermediate results
├── runMCMC_template.py # Main execution script
├── requirements.txt
└── README.md
