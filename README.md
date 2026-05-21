# MCMC Cross-Correlation: Galaxies × Gravitational Waves

A pipeline for **full MCMC-based cosmological inference** using the cross-correlation of gravitational-wave (GW) standard sirens with galaxy catalogues, including modified gravity extensions via **EFTCAMB**.

---

## Overview

This pipeline provides:

- MCMC sampling of cosmological and modified gravity parameters
- Matter power spectra via **EFTCAMB** (modified gravity) or standard CAMB (ΛCDM)
- Angular power spectra C(ℓ) for GW × galaxy cross-correlations
- GW source distributions and window functions including α_M corrections to luminosity distance
- Galaxy clustering window functions with redshift-dependent bias
- Full posterior analysis, convergence diagnostics, and corner plots

This is the full posterior sampling counterpart to Fisher-matrix forecasting, and can be used to validate Fisher results or explore non-Gaussian and degenerate parameter spaces.

---

## Repository Structure

```
MCMC-Cross-Correlation-Galaxies-GWs/
│
├── main.py                         # Main entry point
├── MCMC.py                         # MCMC runner for standard or modified gravity
│
├── functions_cross_correlation.py  # GW merger rate, source distributions
├── functions_extra_main.py         # Cl computation, likelihood, diagnostics
├── likelihood_functions.py         # Log-likelihood and Fisher utilities
│
├── colibri/                        # Core cosmology library
│   ├── cosmology_MG.py             # EFTCAMB cosmology class (MG_pk, camb_Pk, ...)
│   ├── cosmology.py                # Standard CAMB cosmology class
│   ├── limber_GW.py                # Limber integrator for GW window functions
│   ├── limber_angular_PS.py        # Angular power spectra
│   └── ...                         # Other utilities (fourier, halo, RSD, ...)
│
├── configs/                        # Pipeline configuration files (.py)
│   ├── config_LCDM_Camb.py         # ΛCDM via CAMB
│   ├── config_EFT.py               # Modified gravity via EFTCAMB (alpha_M, alpha_B)
│   ├── config_Alpha_DynamicBackground.py # Modified gravity via EFTCAMB (alpha_M, alpha_B) and Dynamic Background (w0, wa)
│   ├── config_template_detectors.py # All the settings for the detectors
│   └── ...
│
├── settings.json                   # MCMC settings for ΛCDM runs
├── settings_MG.json                # MCMC settings for MG runs
│
├── det_param/                      # GW detector parameters (.npy)
│   └── log_dl_*, log_loc_*, ...    # ET, CE, LVK configurations
│
├── EFTCAMB/                        # EFTCAMB source and Python interface
├── hi_class/                       # hi_CLASS source (legacy, not used in main pipeline)
│
└── process_chains.ipynb            # Notebook for posterior analysis
```

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/MicheleSantoni/MCMC-Cross-Correlation-Galaxies-GWs.git
cd MCMC-Cross-Correlation-Galaxies-GWs
```

### Install dependencies

```bash
pip install numpy scipy matplotlib astropy h5py
pip install emcee corner arviz tqdm numdifftools
```

EFTCAMB must be installed separately from the `EFTCAMB/` directory:

```bash
cd EFTCAMB
pip install -e .
cd ..
```

---

## Usage

### Run the MCMC

```bash
python main.py --config='config_template_detectors,config_LCDM_Camb' --fout='results/my_run'
```

For a modified gravity run:

```bash
python main.py --config='config_template_detectors,config_EFT' --fout='results/MG_run'
```

### Configuration

MCMC settings are controlled via two separate files:

**`settings.json` / `settings_MG.json`** — MCMC runtime parameters:
```json
{
  "nwalkers": 32,
  "nsteps": 1000,
  "params_inference": ["H0", "Omega_m", "alpha_M"],
  "priors": { ... }
}
```

**`configs/config_*.py`** — cosmological and pipeline parameters:
```python
COSMO_PARAMS = {
    'h': 0.6781,
    'omega_b': 0.0220704,
    'omega_cdm': 0.120473679,
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    ...
}

EFTCAMB_PARAMS = {
    'EFTflag': 1,           # 0 = ΛCDM, 1 = Pure EFT
    'PureEFTmodel': 1,
    'RPHmassPparams': [0.0],    # alpha_M
    'RPHbraidingparams': [0.0], # alpha_B
    'EFTw0': -1.0,
    'EFTwa': 0.0,
    ...
}
```

---

## Inference Parameters

The following parameters can be included in `params_inference`:

| Parameter | Description |
|---|---|
| `H0` | Hubble constant [km/s/Mpc] |
| `Omega_m` | Total matter density |
| `Omega_b` | Baryon density |
| `A_s` | Primordial amplitude |
| `n_s` | Spectral index |
| `alpha_M` | Planck mass run rate (EFTCAMB) |
| `alpha_B` | Braiding parameter (EFTCAMB) |
| `w0` | Dark energy equation of state |
| `wa` | Dark energy time evolution |
| `A_GW`, `gamma_GW` | GW bias power-law amplitude and slope |
| `A_gal`, `gamma_gal` | Galaxy bias power-law amplitude and slope |

---

## Outputs

Each run produces in the `--fout` directory:

| File | Description |
|---|---|
| `sampler.h5` | Full MCMC chains (emcee backend) |
| `summary.txt` | Median, ±1σ credible intervals per parameter |
| `arviz_summary.csv` | ArviZ diagnostics including R̂ convergence |
| `traceplots.png` | Walker traces with burn-in line |
| `corner_<nsteps>.png` | Corner plot with 68% and 90% contours |

---

## Fisher vs MCMC

| | Fisher Matrix | MCMC |
|---|---|---|
| Speed | Fast | Slower |
| Validity | Gaussian posteriors | Full non-Gaussian posteriors |
| Output | Covariance matrix | Posterior chains |
| Use case | Quick forecasts | Robust inference & validation |

---

## GW Detector Configurations

Pre-computed detector parameters in `det_param/` support:

- **Einstein Telescope**: 2L and Triangle geometries
- **Cosmic Explorer**: 1CE and 2CE configurations
- **LVK**: current network
- With and without luminosity-distance cuts (`_cut`, `_hardcut`)

---

## License

MIT License

---

## Citation

If you use this pipeline in academic work, please cite the original COLIBRI_GW project and the relevant EFTCAMB and CAMB papers.
