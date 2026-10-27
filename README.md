# MCMC Cross-Correlation: Galaxies × Gravitational Waves

A pipeline for **full MCMC-based cosmological inference** from the angular cross-correlation of gravitational-wave (GW) standard sirens with galaxy catalogues, including modified-gravity and dynamical dark-energy extensions via **EFTCAMB**.

---

## Overview

The pipeline provides:

- Matter power spectra via **EFTCAMB**: ΛCDM (`EFTflag = 0`) or Horndeski-like modified gravity in the α-basis (`EFTflag = 2`, `AltParEFTmodel = 1`), optionally with a CPL dark-energy background (w₀, wₐ)
- GW source distributions (merger rate, redshift/luminosity-distance distributions) and window functions, including the α_M correction to the GW luminosity distance
- Galaxy clustering window functions with redshift-dependent bias (Euclid photometric/spectroscopic, SKA)
- Angular power spectra C(ℓ) for GG, GW-GW and G×GW, with density, RSD and (optionally) lensing terms
- Gaussian likelihood built on a fiducial data vector and pre-computed inverse covariance
- MCMC sampling with **emcee** (HDF5 backend, multiprocessing or MPI), autocorrelation-based stopping, ArviZ diagnostics, trace and corner plots

It is the full-posterior counterpart to Fisher-matrix forecasting and can be used to validate Fisher results or to explore non-Gaussian and degenerate parameter spaces.

---

## Repository Structure

```
MCMC-Cross-Correlation-Galaxies-GWs/
│
├── main.py                         # Step 1: fiducial C(ℓ), noise, binning, inverse covariance
├── MCMC.py                         # Step 2: MCMC sampling (EFTCAMB backend)
├── main_hi_class.py                # Legacy: Step 1 with hi_class
├── MCMC_hi_class.py                # Legacy: Step 2 with hi_class
│
├── functions_cross_correlation.py  # GW merger rate, source distributions
├── functions_extra_main.py         # C(ℓ) helpers, detector loading, plots, diagnostics
├── likelihood_functions.py         # Log-likelihood and Fisher utilities
│
├── colibri/                        # Core cosmology library (adapted from COLIBRI)
│   ├── cosmology_MG.py             # EFTCAMB cosmology class (MG_pk, ...)
│   ├── cosmology.py                # Standard cosmology class
│   ├── cosmology_hi_class.py       # hi_class cosmology class (legacy)
│   ├── limber_GW.py                # Limber integrator for GW/galaxy window functions
│   ├── limber_angular_PS.py        # Angular power spectra
│   └── ...                         # fourier, halo, RSD, nonlinear, galaxy, constants, ...
│
├── configs/
│   ├── config_template_detectors.py      # Survey/detector/binning settings (template)
│   ├── config_detectors.py               # Example detector config (ET_2L_1CE)
│   ├── config_LCDM_Camb.py               # ΛCDM (EFTflag = 0)
│   ├── config_LCDM_DynamicBackground.py  # GR + CPL background (w0, wa)
│   ├── config_EFT.py                     # Modified gravity: alpha_M, alpha_B
│   ├── config_Alpha_DynamicBackground.py # Modified gravity + CPL background
│   └── HI_CLASS/                         # Legacy configs for the hi_class pipeline
│
├── settings.json                   # MCMC settings: ΛCDM + bias parameters
├── settings_MG.json                # MCMC settings: ΛCDM + alpha_M, alpha_B
├── settings_Dyn.json               # MCMC settings: ΛCDM + bias + w0, wa
│
└── det_param/                      # Pre-computed GW detector parameters (.npy)
    └── log_dl_*, log_delta_dl_*, log_loc_*
```

> EFTCAMB is **not** included in this repository and must be installed separately (see below).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MicheleSantoni/MCMC-Cross-Correlation-Galaxies-GWs.git
cd MCMC-Cross-Correlation-Galaxies-GWs
```

### 2. Install Python dependencies

```bash
pip install numpy scipy matplotlib astropy h5py
pip install emcee corner getdist arviz tqdm numdifftools
pip install schwimmbad mpi4py   # only if running with MPI
```

### 3. Install EFTCAMB

Install EFTCAMB (Python interface, imported as `camb`) following the instructions on the [official EFTCAMB repository](https://github.com/EFTCAMB/EFTCAMB).

### 4. Set local paths

Some paths are currently hard-coded and must be edited before the first run:

| File | Variable / line | Set to |
|---|---|---|
| `configs/config_*detectors.py` | `colibri_path` | absolute path to `colibri/` in your clone |
| `colibri/cosmology_MG.py` | `sys.path.insert(0, ".../EFTCAMB/")` | your EFTCAMB installation |

On start-up, `cosmology_MG.py` prints `USING CAMB FROM: ...` so you can check which CAMB build is being imported.

---

## Usage

The workflow has **two steps**. Both scripts must be launched from the repository root (configs are read from `configs/`).

`--config` always takes **two comma-separated config names, in this order**: the detector/survey config, then the cosmology/gravity model config (no `.py` extension).

### Step 1 — Fiducial data vector and covariance (`main.py`)

```bash
python main.py --config='config_template_detectors,config_LCDM_Camb' --fout='ET_2L_2CE'
```

This computes binning, source distributions, noise, fiducial C(ℓ) and inverse covariance matrices, and saves them as `.npy` files (plus diagnostic plots) in `--fout`.

> **Important:** `MCMC.py` loads these products from a folder named after the detector network (`GW_det` in the detector config, e.g. `ET_2L_2CE/`) relative to the working directory. Use `--fout=<GW_det>` in Step 1, or move/symlink the output folder accordingly.

### Step 2 — MCMC sampling (`MCMC.py`)

The output folder must already exist.

```bash
mkdir -p results/LCDM_run
python MCMC.py --settings='settings.json' \
               --config='config_template_detectors,config_LCDM_Camb' \
               --fout='results/LCDM_run'
```

Modified gravity (α_M, α_B):

```bash
python MCMC.py --settings='settings_MG.json' \
               --config='config_template_detectors,config_EFT' \
               --fout='results/MG_run'
```

Dynamical dark energy (w₀, wₐ):

```bash
python MCMC.py --settings='settings_Dyn.json' \
               --config='config_template_detectors,config_LCDM_DynamicBackground' \
               --fout='results/Dyn_run'
```

#### Command-line arguments of `MCMC.py`

| Argument | Description |
|---|---|
| `--settings` | Path to the JSON settings file (omit when resuming) |
| `--config` | `<detector_config>,<model_config>` |
| `--fout` | Output folder (must exist) |
| `--resume` | `1` to continue a previous chain from `<fout>/chains.h5` (default `0`) |
| `--nsteps` | Number of steps when resuming |

When resuming, the original settings are read from `<fout>/settings_original.json`, and new chains/logs are written as `chains_run<N>.h5` / `logfile_run<N>.txt`.

---

## Configuration

### Detector / survey config (`configs/config_template_detectors.py`)

Main options:

| Variable | Description |
|---|---|
| `GW_det` | GW network: `ET_Delta`, `ET_2L`, `ET_Delta_1CE`, `ET_2L_1CE`, `ET_Delta_2CE`, `ET_2L_2CE`, `LVK` (optionally `_cut`, `_hardcut`) |
| `gal_det` | Galaxy survey: `euclid_photo`, `euclid_spectro`, `ska` |
| `yr` | Years of GW observation |
| `n_bins_z`, `n_bins_dl` | Number of galaxy (z) and GW (d_L) bins |
| `z_m_bin`, `z_M_bin` / `z_m_bin_GW`, `z_M_bin_GW` | Galaxy / GW bin ranges in redshift |
| `bin_strategy` | `right_cosmo`, `wrong_cosmo`, `equal_pop`, `equal_space` |
| `Lensing` | Include lensing contributions |
| `name_component` | Data vector used in the likelihood: `GW`, `Gal`, `Cross` or `Full` |
| `A_GW`, `gamma_GW`, `A_gal`, `gamma_gal` | Fiducial bias parameters |
| `f_sky`, `f_sky_GW` | Sky fractions of the galaxy survey and GW network |
| `l_min`, `delta_ell` | Minimum multipole and ℓ step |
| `sig_gal` | Galaxy photo-z error |

### Model config (`configs/config_*.py`)

Each model config defines `COSMO_PARAMS` and `EFTCAMB_PARAMS`. Example (`config_EFT.py`):

```python
COSMO_PARAMS = {
    'h': 0.6781,
    'omega_b': 0.0220704,
    'omega_cdm': 0.120473679,
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'dark_energy_model': 'EFTCAMB',
    ...
}

EFTCAMB_PARAMS = {
    'EFTflag': 2,                 # 0 = GR/ΛCDM, 2 = alternative parametrization
    'AltParEFTmodel': 1,          # α-basis (RPH)
    'RPHusealphaM': True,
    'RPHalphaMmodel_ODE': 2,      # α_i ∝ Ω_DE
    'RPHbraidingmodel_ODE': 2,
    'RPHalphaM_ODE0': 0.0,        # α_M today
    'RPHbraiding_ODE0': 0.0,      # braiding today
    ...
}
```

`config_Alpha_DynamicBackground.py` and `config_LCDM_DynamicBackground.py` additionally set a CPL background via `RPHwDE = 2`, `RPHw0`, `RPHwa`.

### MCMC settings (`settings*.json`)

```json
{
  "fiducial_vals":   { "H0": 67.81, "Omega_m": 0.31, "alpha_M": 0.5, ... },
  "params_inference": ["H0", "Omega_m", "Omega_b", "A_s", "n_s", "alpha_M", "alpha_B"],
  "prior_limits":    { "H0": [60, 80], "alpha_M": [-1, 1], ... },
  "params_fixed":    [],
  "all_params":      ["H0", "Omega_m", ...],
  "nprocesses": 14,
  "nwalkers": 14,
  "eps": 0.15,
  "ntaus": 50,
  "maxsteps": 100000,
  "mpi": 0
}
```

| Key | Description |
|---|---|
| `fiducial_vals` | Fiducial values; walkers are initialised around them |
| `params_inference` | Parameters sampled |
| `prior_limits` | Flat (uniform) prior ranges |
| `params_fixed` | Parameters held at their fiducial value |
| `nwalkers`, `nprocesses` | Number of walkers and parallel processes |
| `eps` | Relative spread of the initial ball **and** tolerance on the change of τ for convergence |
| `ntaus` | Required chain length in units of the autocorrelation time τ |
| `maxsteps` | Maximum number of steps |
| `ncheck` | (optional, default 500) Steps between convergence checks |
| `params_labels` | (optional) LaTeX labels for the corner plot |
| `mpi` | `0` = multiprocessing, `1` = MPI via `schwimmbad` |

Sampling stops when `iteration > ntaus · τ` and `|Δτ|/τ < eps` for all parameters, or at `maxsteps`.

---

## Inference Parameters

| Parameter | Description |
|---|---|
| `H0` | Hubble constant [km/s/Mpc] |
| `Omega_m` | Total matter density |
| `Omega_b` | Baryon density |
| `A_s` | Primordial amplitude |
| `n_s` | Spectral index |
| `alpha_M` | Planck-mass run rate today (→ `RPHalphaM_ODE0`) |
| `alpha_B` | Braiding today (→ `RPHbraiding_ODE0 = -α_B/2`) |
| `w0`, `wa` | CPL dark-energy equation of state (→ `RPHw0`, `RPHwa`) |
| `A_GW`, `gamma_GW` | GW bias power-law amplitude and slope |
| `A_gal`, `gamma_gal` | Galaxy bias power-law amplitude and slope |

---

## Outputs

### Step 1 (`main.py`), in `--fout`

| File | Description |
|---|---|
| `*_original.py` | Copies of the configs used |
| `bin_edges_*.npy`, `nz_gal.npy`, `ndl_GW.npy`, `z_GW.npy` | Binning and source distributions |
| `bias_fiducial_*.npy`, `noise_*.npy` | Fiducial biases and noise |
| `l_max*.npy`, `ll*.npy` | Multipole ranges per bin |
| `Cl_*.npy` | Fiducial angular power spectra (with/without noise) |
| `vec_fid*.npy`, `cov_mat_inverse*.npy` | Data vectors and inverse covariances (full and per component) |
| `*.png`, `*.txt` | Bin-distribution plots and diagnostics |

### Step 2 (`MCMC.py`), in `--fout`

| File | Description |
|---|---|
| `settings_original.json` | Copy of the settings used |
| `logfile.txt` | Full run log (stdout + stderr) |
| `chains.h5` | MCMC chains (emcee HDF5 backend) |
| `summary.txt` | Burn-in, thinning, τ, ESS, median and ±1σ per parameter |
| `arviz_summary.csv` | ArviZ diagnostics including R̂ |
| `traceplots.png` | Walker traces with burn-in line |
| `corner_.png` | Corner plot (GetDist) |

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

Pre-computed detector parameters in `det_param/` (luminosity distance, its error, and localization) support:

- **Einstein Telescope**: 2L and Triangle (`Delta`) geometries
- **ET + Cosmic Explorer**: with 1 or 2 CE detectors
- **LVK**: current network
- With and without luminosity-distance cuts (`_cut`, `_hardcut`)

---

## Legacy: hi_class pipeline

`main_hi_class.py`, `MCMC_hi_class.py`, `colibri/cosmology_hi_class.py` and `configs/HI_CLASS/` implement an earlier version based on **hi_class** (`classy`). They are kept for reference and are not used in the main EFTCAMB pipeline.

---

## License

MIT License

---

## Citation

If you use this pipeline in academic work, please cite the original COLIBRI_GW project and the relevant EFTCAMB, CAMB, emcee and GetDist papers.
