#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April-May 2026

@author: Michele Santoni
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import multiprocessing
# Needed for mpipool not to stall when trying to write on a file (do not ask me why)
#multiprocessing.set_start_method("spawn",force=True)
#import zeus
import emcee
import corner
import argparse
import shutil
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import arviz as az

####
import functions_cross_correlation as fcc
import functions_extra_main as fem
import colibri.cosmology_MG as cc_MG
import colibri.limber_GW as LLG
import likelihood_functions as LH_fun
import importlib
import scipy.interpolate as si
from scipy.interpolate import interp1d
import time
from copy import deepcopy


#####################################################################################
# AUXILIARY STUFF
#####################################################################################

# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.flush()
        self.log.close()
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False
    


class Distribution(ABC):
    
    def __init__(self,):
        pass
    
    @abstractmethod
    def sample(self, nsamples):
        pass
    
    @abstractmethod
    def logpdf(self, x):
        pass
    
    def pdf(self, x):
        return np.exp(self.logpdf(x))
        

class Uniform(Distribution):
    ''' 
    Uncorrelated uniform distribution in N dimensions
    '''
    
    def __init__(self, lims = [  (0,1), (0,1), ]):
        self.lims=lims
        self.lows = [ l[0] for l in lims]
        self.highs = [ l[1] for l in lims]
        self.D = len(lims)
        Distribution.__init__(self)

        
    def sample(self, nsamples):
        return np.random.uniform( low=self.lows, high=self.highs, size=[nsamples, self.D]).T

    
    def logpdf(self, x, return_sum=True):   
        '''
        x has shape ( D, n_samples)
        works also for x of shape (n_samples) (n points in 1D distribution) and (D, 1) (1 point in D dimensional distribution)
        '''
        x=np.array(x)

        more_points_D_dim=False
        one_point_D_dim=False
        more_points_1D=False
        if not np.isscalar(x) and x.shape[0]==self.D and np.ndim(x)>1:
            more_points_D_dim = True
        elif x.shape[0]==self.D and np.ndim(x)==1:
            one_point_D_dim = True
        elif x.shape[0]!=self.D and self.D==1:
            more_points_1D = True
        
        lp = np.empty(x.shape)
        
        if more_points_1D:
            # samples are 1D, x has more than one point
            lp = np.where( (x<self.highs[0]) & (x>=self.lows[0]), 0 , np.inf ) 
        elif more_points_D_dim:
            lp = np.array([ np.where( (x[d, :]<self.highs[d]) & (x[d, :]>=self.lows[d]), 0 , np.inf ) for d in range(self.D)])
        elif one_point_D_dim:
            lp = np.array([ 0 if ( x[d]<np.nan_to_num(self.highs[d])) & (x[d]>=np.nan_to_num(self.lows[d])) else np.inf for d in range(self.D)])
        if return_sum and not more_points_1D:
            return lp.sum(axis=0)
        else:
            return lp



def get_theta(x, pinf, pfix, allpars):
    # pfix is a dictionary {parameter_name: parameter_value}
    # pinf is a list of parameters names
    # allpars is a list with names of all parameters in the correct order (i.e. the same order as theta)
    
    allp = pfix
    for i,p in enumerate(pinf):
        allp[p] =  x[i]
    
    return np.array([ allp[pname] for pname in allpars ])


def get_init_point_flat(prior, expected_vals, nwalkers, ndim, eps, verbose=False):
    allinit = np.empty((nwalkers, ndim))
    for i in range(ndim):
        if expected_vals[i] != 0:
            linf = expected_vals[i] - np.abs(expected_vals[i]) * eps
            lsup = expected_vals[i] + np.abs(expected_vals[i]) * eps
        else:
            linf = -eps
            lsup = eps
        pinf = max(linf, prior.lims[i][0])
        psup = min(lsup, prior.lims[i][1])
        if psup <= pinf:
            raise ValueError(f"Param {i}: empty interval after clipping on prior "
                             f"(pinf={pinf}, psup={psup}, central={expected_vals[i]}). "
                             f"Check eps or prior limits.")
        if verbose:
            print('For param %s, eps=%s, min=%s, max=%s, central value=%s' % (i, eps, pinf, psup, expected_vals[i]))
        for k in range(nwalkers):
            allinit[k, i] = np.random.uniform(low=pinf, high=psup)
    assert np.all(~np.isnan(allinit))
    return allinit

def find_valid_start(log_post_fn, prior, expected_vals, nwalkers, ndim, eps, max_tries=10000, verbose=True):
    valid = []
    tries = 0

    # Genera prima una ball attorno al fiduciale
    center = np.array(expected_vals)
    scales = np.array([
        abs(center[i]) * eps if center[i] != 0 else eps
        for i in range(ndim)
    ])

    while len(valid) < nwalkers and tries < max_tries:
        # proposta nella ball fiduciale (non in tutto il prior)
        proposal = center + scales * np.random.uniform(-1, 1, ndim)

        # check che sia dentro il prior
        in_prior = all(
            prior.lims[i][0] <= proposal[i] <= prior.lims[i][1]
            for i in range(ndim)
        )
        if not in_prior:
            tries += 1
            continue

        lp = log_post_fn(proposal)
        if np.isfinite(lp):
            valid.append(proposal)
            if verbose:
                print(f"  Walker {len(valid)}/{nwalkers} found! Attempt number: {tries}; logpost={lp:.2f}")
        tries += 1

    if len(valid) < nwalkers:
        raise RuntimeError(
            f"Only {len(valid)}/{nwalkers} valid walker after {max_tries} attempts.\n"
            f"Check the EFTCamb parameters in the config file. The fiducial point could be unstable."
        )

    print(f"Initialization completed: valid walkers {nwalkers}.")
    return np.array(valid)



def diagnostic(sampler, iterations, settings):
    """
        Diagnostic of the chains
    """
    try:
        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        n_steps_used = iterations - burnin
        ESS = settings["nwalkers"] * n_steps_used / tau
        ratio = iterations / np.max(tau)

        print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        print(f"Autocorrelation time: {tau}")
        print(f"Burnin: {burnin}, thin: {thin}")
        print(f"Effective Sample Size per parametro: {ESS.astype(int)}")
        print(f"Steps / tau_max = {ratio:.1f}  (vuoi > 50)")
        if ratio < 50:
            print("The chain is too short: need to increase the number of steps")

    except emcee.autocorr.AutocorrError:
        print("The chain is too short for evaluating the autocorrelation: burnin=0, thin=1")
        tau, burnin, thin, ESS, ratio = None, 0, 1, None, None

    try:
        idata = az.from_emcee(sampler, var_names=settings["params_inference"])
        az_summary = az.summary(idata)
        print(az_summary)
        converged = (az_summary["r_hat"] < 1.01).all()
        print(f"Convergenza (R̂ < 1.01): {'✓' if converged else '⚠️ NO'}")
        az_summary.to_csv(os.path.join(FLAGS.fout, "arviz_summary.csv"))
    except Exception as e:
        print(f"Arviz non disponibile o errore: {e}")

    return burnin, thin, tau, ESS, ratio


def summary(sampler, burnin, thin, tau, ESS, ratio, iterations, settings, fout):
    """
            Summary of the MCMC and the outcomes
    """
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    print(f"Samples extracted: {samples.shape}")

    summary_path = os.path.join(fout, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}\n")
        f.write(f"Steps completed: {iterations}\n")
        if tau is not None:
            f.write(f"Autocorrelation time: {tau}\n")
            f.write(f"Burnin: {burnin}, thin: {thin}\n")
            f.write(f"ESS: {ESS.astype(int)}\n")
            f.write(f"Steps / tau_max: {ratio:.1f}\n\n")
        f.write("Parameter | median | +err | -err\n")
        for i, name in enumerate(settings["params_inference"]):
            q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
            f.write(f"{name:15s}: {q50:.6f} +{q84 - q50:.6f} -{q50 - q16:.6f}\n")
            print(f"{name:15s}: {q50:.6f} +{q84 - q50:.6f} -{q50 - q16:.6f}")

    return samples


def traceplots(sampler, burnin, settings, fout):
    """
                Draw the traceplots for each parameter
    """
    ndim = len(settings["params_inference"])
    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)

    fig, axes = plt.subplots(ndim, 1, figsize=(12, 2.5 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, settings["params_inference"])):
        ax.plot(chain[:, :, i], alpha=0.3, lw=0.5)
        ax.axvline(burnin, color='red', lw=1.5, linestyle='--', label='burnin' if i == 0 else '')
        ax.set_ylabel(name)
        ax.yaxis.set_label_coords(-0.08, 0.5)
    axes[-1].set_xlabel("Step")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fout, "traceplots.png"), dpi=150)
    plt.close()


def plot_corrmatrix(corr_matrix, labels, names, Save=False, out_path=None, nsteps=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([f"${l}$" for l in labels], rotation=45, ha='right')
    ax.set_yticklabels([f"${l}$" for l in labels])

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr_matrix[i,j]:.2f}",
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(corr_matrix[i,j]) > 0.6 else 'black')

    plt.tight_layout()
    if Save:
        fname = f"corr_matrix_{nsteps}.png" if nsteps is not None else "corr_matrix.png"
        dest  = os.path.join(out_path, fname) if out_path else fname
        plt.savefig(dest, dpi=150, bbox_inches='tight')
    plt.show()

def plot_corner(samples, settings, fiducials, myPrior, out_path, nsteps):
    try:
        print('Plotting corner...')

        names  = settings["params_inference"]       # nomi brevi, es. ['om', 's8']
        labels = settings.get("params_labels", names)  # label LaTeX, es. [r'\Omega_m', r'\sigma_8']

        # Range: stesso approccio margin del codice originale
        margin   = 0.05 * (samples.max(axis=0) - samples.min(axis=0))
        plot_range = {
            n: (samples[:, i].min() - margin[i], samples[:, i].max() + margin[i])
            for i, n in enumerate(names)
        }

        mcmc_samples = MCSamples(
            samples=samples,
            names=names,
            labels=labels,
            label='MCMC',
            ranges=plot_range,
        )
        mcmc_samples.updateSettings({'smooth_scale_2D': 0.4, 'smooth_scale_1D': 0.4})

        g = gdplt.get_subplot_plotter(width_inch=12)
        g.settings.axes_fontsize   = 13
        g.settings.legend_fontsize = 13
        g.settings.lab_fontsize    = 15

        g.triangle_plot(
            mcmc_samples,
            names,
            filled=True,
            contour_colors=['darkred'],
            contour_ls='-',
            contour_lws=1.5,
            legend_labels=[f'MCMC ({nsteps} steps)'],
            markers={n: fiducials[i] for i, n in enumerate(names)},
            title_limit=1,
        )

        plt.savefig(os.path.join(out_path, f'corner_{nsteps}.png'), dpi=150, bbox_inches='tight')
        plt.close('all')
        print("Corner ok")

        # --- Bonus: correlation matrix ---
        latex_names = {name: f"${label}$" for name, label in zip(names, labels)}
        print_latex_table(samples, names, latex_names)

        corr_matrix = mcmc_samples.corr()
        df_corr = pd.DataFrame(corr_matrix, index=names, columns=names)
        print(df_corr.round(3))
        plot_corrmatrix(corr_matrix, labels, names,
                        Save=True, out_path=out_path, nsteps=nsteps)

    except Exception as e:
        print(f"Corner failed: {e}")



def get_pool(mpi=False, threads=None):
    """
    Always returns a pool object with a `map()` method. By default,
        returns a `SerialPool()` -- `SerialPool.map()` just calls the built-in
        Python function `map()`. If `mpi=True`, will attempt to import the 
        `MPIPool` implementation from `emcee`. If `threads` is set to a 
        number > 1, it will return a Python multiprocessing pool.
        Parameters
        ----------
        mpi : bool (optional)
            Use MPI or not. If specified, ignores the threads kwarg.
        threads : int (optional)
            If mpi is False and threads is specified, use a Python
            multiprocessing pool with the specified number of threads.
    """

    if mpi:
        from schwimmbad import MPIPool
        print('Using MPI...')
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    elif threads > 1:
        print('Using multiprocessing with %s processes...' %(threads-1))
        pool = multiprocessing.Pool(threads-1)

    else:
        raise ValueError('Called get pool with threads=1. No need')
        #pool = SerialPool()

    return pool


#####################################################################################
# MAIN
#####################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--settings", default='', type=str, required=False) # path to config file, in.json format
parser.add_argument("--config", default='', type=str, required=False) # path to config file, in.json format
parser.add_argument("--fout", default='', type=str, required=True) # path to output folder
parser.add_argument("--resume", default=0, type=int, required=False) #continue a pre-existing chain (0=False, 1=True)
parser.add_argument("--nsteps", default=0, type=int, required=False) # only if continuing a chain
FLAGS = parser.parse_args()


tin=time.time()

if FLAGS.resume and FLAGS.settings!='':
    raise ValueError('If continuing a pre-existing chain, you do not have to specify the settings file.')

if not os.path.exists(FLAGS.fout):
    raise ValueError('Path to output folder does not exist. Value entered: %s' %FLAGS.fout)

baseChainName='chains'
baselogfileName= 'logfile'
if FLAGS.resume:
    nResume = 1
    chainName = baseChainName + '_run' + str(nResume)
    logfile = os.path.join(FLAGS.fout, baselogfileName + '_run' + str(nResume) + '.txt')
    while os.path.exists(logfile):
        nResume += 1   # increment FIRST
        chainName = baseChainName + '_run' + str(nResume)
        logfile = os.path.join(FLAGS.fout, baselogfileName + '_run' + str(nResume) + '.txt')
else:   
    logfile = os.path.join(FLAGS.fout, 'logfile.txt')
    chainName = baseChainName

myLog = Logger(logfile)
sys.stdout = myLog
sys.stderr = myLog

print()
print('='*40)
print('Setting up...')
print('='*40)


if not FLAGS.resume:
    print('\nImporting settings...')
    shutil.copy(os.path.join( FLAGS.settings), os.path.join(FLAGS.fout, 'settings_original.json'))
    settings_path=FLAGS.settings
else:
    settings_path = os.path.join(FLAGS.fout, 'settings_original.json')
    print('\nImporting original settings from %s...' %settings_path)

with open(settings_path, 'r') as fp: # import settings for the analysis 
    settings = json.load(fp)

################
# Define prior and likelihood
print()
print('='*40)
print('Defining prior and likelihood...')
print('='*40)


params_fixed_dict = { p: settings["fiducial_vals"][p] for p in settings["params_fixed"] }

print('\nAll parameters: %s' %(settings["all_params"]))
print('\nRunning inference on %s' %settings["params_inference"])
#print('\nFixed parameters and their values: ')
#print(params_fixed_dict)

myPrior = Uniform( [ settings['prior_limits'][par] for par in settings["params_inference"] ] )


###############################################################################################################################################################
print()
print('='*40)
print('CONFIGURATION FILES')
print('='*40)
# Dictionary to hold imported config modules
configspath = 'configs/'
sys.path.append(configspath)

configs = {}
config_names = FLAGS.config.split(',')  # if FLAGS.config is a comma-separated string
print(config_names)
for cfg_name in config_names:
    # Copy the config file into the output directory
    src = os.path.join(configspath, f"{cfg_name}.py")
    # Dynamically import the config module
    configs[cfg_name] = importlib.import_module(cfg_name)

cosmo=config_names[1]
detectors= config_names[0]

# Initialize cosmology for power spectrum calculation
cosmo_params = configs[cosmo].COSMO_PARAMS
eftcamb_params = configs[config_names[1]].EFTCAMB_PARAMS

if eftcamb_params["EFTflag"] !=0:
    print("Modified Gravity model is on")

GW_det = configs[detectors].GW_det # GW detector (ET_Delta_2CE, ET_2L_2CE, ET_Delta_1CE, ET_2L_1CE, ET_Delta, ET_2L, LVK)
yr = configs[detectors].yr # Years of observation
gw_params = fem.load_detector_params(GW_det, yr)

main_config = GW_det  # ET_Delta_2CE, ET_2L_2CE, ET_Delta, ET_2L, LVK

print()
print('='*40)
print('CONFIGURATION CONSIDERED')
print('='*40)
name_component = configs[detectors].name_component
if name_component in ["GW","Gal","Cross"]:
    print(name_component)
else:
    print("Component not recognized \t -----------> \t Full considered")
print("The detector network for GWS is",GW_det)

# Fraction of the sky covered from the survey
f_sky = configs[detectors].f_sky
f_sky_GW = configs[detectors].f_sky_GW

delta_ell = configs[detectors].delta_ell

# Define the redshift total range
z_m = configs[detectors].z_m
z_M = configs[detectors].z_M

# Define the number of bins
n_bins_z = configs[detectors].n_bins_z
n_bins_dl = configs[detectors].n_bins_dl

autoX_len = n_bins_z ** 2 - np.sum(range(n_bins_z))
crossXY_len = n_bins_z * n_bins_dl
autoY_len = n_bins_dl ** 2 - np.sum(range(n_bins_dl))

# Define the luminosity distance total range
dlm = configs[detectors].dlm
dlM = configs[detectors].dlM

l_min = configs[detectors].l_min

l_max_nl_=np.load(os.path.join(main_config, 'l_max_nl.npy'))
l_max_loc= np.load(os.path.join(main_config, 'l_max_loc.npy'))

l_max= np.load(os.path.join(main_config, 'l_max.npy')) #
ll_subset = np.load(os.path.join(main_config, 'll.npy')) #
ll_total = np.arange(l_min, l_max + 1) #

bin_edges_dl = np.load(os.path.join(main_config, 'bin_edges_GW.npy')) #
bin_edges    = np.load(os.path.join(main_config, 'bin_edges_gal.npy')) #

nz_gal = np.load(os.path.join(main_config, 'nz_gal.npy')) #
z_GW   = np.load(os.path.join(main_config, 'z_GW.npy')) #

bias_GW  = np.load(os.path.join(main_config, 'bias_fiducial_GW.npy')) #
bias_gal = np.load(os.path.join(main_config, 'bias_fiducial_gal.npy')) #

noise_loc_mat_auto = np.load(os.path.join(main_config, 'noise_loc_auto.npy')) #
noise_loc_mat      = np.load(os.path.join(main_config, 'noise_loc_cross.npy')) #

noise_GW  = np.load(os.path.join(main_config, 'noise_GW.npy')) #
noise_gal = np.load(os.path.join(main_config, 'noise_gal.npy')) #

if name_component == "GW":
    Cl_fid = np.load(os.path.join(main_config, 'vec_fid_GWGW.npy'))  #
    F = np.load(os.path.join(main_config, 'cov_mat_inverse_GWGW.npy'))  #

elif name_component == "Gal":
    Cl_fid = np.load(os.path.join(main_config, 'vec_fid_GG.npy'))  #
    F = np.load(os.path.join(main_config, 'cov_mat_inverse_GG.npy'))  #

elif name_component == "Cross":
    Cl_fid = np.load(os.path.join(main_config, 'vec_fid_GGW.npy'))  #
    F = np.load(os.path.join(main_config, 'cov_mat_inverse_GGW.npy'))  #

else:
    Cl_fid = np.load(os.path.join(main_config, 'vec_fid.npy'))  #
    F = np.load(os.path.join(main_config, 'cov_mat_inverse.npy'))  #

F_t    = F.transpose(2, 0, 1)

z_mean_GW = np.load(os.path.join(main_config, 'z_mean_GW.npy')) #
z_mean_gal = np.load(os.path.join(main_config, 'z_mean_gal.npy')) #

z_gal = np.linspace(z_m, z_M, 2000)  # Redshift grid for galaxy distribution
dl_GW = np.linspace(dlm, dlM, 2000)  # Luminosity distance grid for gravitational wave sources in Mpc

Lensing = configs[detectors].Lensing

if Lensing:
    def Cl_func(universe, params, eftcamb_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll_subset, b_gal, b_GW,
                n_points=13, n_points_x=20, grid_x='lin', z_min=1e-05, n_low=5, n_high=5):

        S = LLG.limber(cosmology=universe, z_limits=[z_m, z_M])

        # Define power spectrum grids
        kk = np.geomspace(1e-4, 10, 500)  # in [h/Mpc]
        zz = np.linspace(0, z_M, 100)
        # Compute nonlinear matter power spectrum
        bg, kk_out, zz_out, pkz = universe.MG_pk(kk, zz, eftcamb_params, True)  # pkz (Mpc/h)^3

        if bg == 0 and pkz == 0:
            return np.array(0), np.array(0), np.array(0)
        else:
            # print("pkz shape",pkz.shape)
            # print("bg",bg.keys())
            S.load_power_spectra(z=zz_out, k=kk_out, power_spectra=pkz)

            # Generate GW distribution from fiducial parameters
            A = gw_params['A']
            Alpha = gw_params['Alpha']
            log_delta_dl = gw_params['log_delta_dl']
            log_dl = gw_params['log_dl']
            Z_0 = gw_params['Z_0']
            Beta = gw_params['Beta']

            h = params['h']
            H_0 = params['h'] * 100  # if you want H0 in km/s/Mpc
            Omega_m = round(params['omega_m'] / (params['h'] ** 2), 2)
            Omega_b = round(params['omega_b'] / (params['h'] ** 2), 3)

            z_bg = np.asarray(bg['z'])
            H_bg = np.asarray(bg["H"], dtype=float)  # 1/Mpc ----> From EFTCamb, H=[1/Mpc]
            chi_bg = np.asarray(bg["chi"], dtype=float)  # Mpc ---> default via EFTCamb
            Omega_m_z_bg = np.asarray(bg["Omega_m_z"], dtype=float)

            H_interp = interp1d(
                z_bg, H_bg,
                kind="cubic",
                bounds_error=False,
                fill_value=(H_bg[0], H_bg[-1])  # clamp outside range
            )

            Omz_interp = interp1d(
                z_bg, Omega_m_z_bg,
                kind="cubic",
                bounds_error=False,
                fill_value=(Omega_m_z_bg[0], Omega_m_z_bg[-1])  # clamp outside range
            )
            chi_interp = interp1d(
                z_bg, chi_bg,
                kind="cubic",
                bounds_error=False,
                fill_value=(chi_bg[0], chi_bg[-1])  # clamp outside range
            )

            alpha_bg = np.asarray(bg["alpha_M"], dtype=float)
            if np.any(np.abs(alpha_bg) > 1e-10):
                alpha_M_interp = interp1d(
                    z_bg, alpha_bg,
                    kind="cubic",
                    bounds_error=False,
                    fill_value=(alpha_bg[0], alpha_bg[-1])
                )
            else:
                alpha_M_interp = interp1d(
                    z_bg,
                    np.zeros_like(z_bg),
                    kind="cubic",
                    bounds_error=False,
                    fill_value=0.0
                )

            # Generate GW source distribution
            # z_GW no dim; bin_GW_converted ; ndl_GW 1/Gpc; n_GW [Gpc]; total no dim
            z_GW, bin_GW_converted, ndl_GW, n_GW, total = fcc.merger_rate_dl_new(
                chi=chi_interp,
                alpha_M_interp=alpha_M_interp,
                z_max=z_M,
                dl=dl_GW,  # Mpc
                bin_dl=bin_edges_dl,
                log_dl=log_dl,
                log_delta_dl=log_delta_dl,
                A=A,
                Z_0=Z_0,
                Alpha=Alpha,
                Beta=Beta,
                normalize=False
            )

            # Load bin edges for all observables
            S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='lensing_gal', name_2='lensing_GW')
            S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='galaxy', name_2='GW')
            S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='rsd', name_2='lsd')

            # Compute galaxy magnification slope parameter beta
            s_a, s_b, s_c, s_d = [gw_params[k] for k in ['s_a', 's_b', 's_c', 's_d']]
            be_a, be_b, be_c, be_d = [gw_params[k] for k in ['be_a', 'be_b', 'be_c', 'be_d']]

            beta = fem.compute_beta(z_gal, H_interp, chi_interp, Omega_m, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d)

            print('\nLoading the window functions...\n')
            # Load window functions for each observable
            S.load_galaxy_clustering_window_functions(H_interp, z=z_gal, n_z=nz_gal, ll=ll_subset, bias=b_gal, name='galaxy')
            S.load_gravitational_wave_window_functions(H_interp, chi_interp, alpha_M_interp, z=z_GW, n_dl=ndl_GW, ll=ll_subset,
                                                       bias=b_GW, name='GW')

            S.load_rsd_window_functions(H_interp, Omz_interp, z=z_gal, n_z=nz_gal, ll=ll_subset, name='rsd')  ### OK
            S.load_lsd_window_functions(H_interp, chi_interp, alpha_M_interp, Omz_interp, z=z_GW, n_dl=ndl_GW, ll=ll_subset,
                                        name='lsd')  ### OK

            S.load_galaxy_lensing_window_functions(z=z_gal, n_z=nz_gal, H_0=H_0, Omega_m=Omega_m, ll=ll_subset,
                                                   name='lensing_gal')  ### Mpc   OK
            S.load_gw_lensing_window_functions(chi_interp, alpha_M_interp, z=z_GW, n_dl=ndl_GW, H_0=H_0,
                                               Omega_m=Omega_m, ll=ll_subset, name='lensing_GW')  ### OK

            print('Computing the angular power spectra...')
            # Compute all angular power spectra using Limber integrals
            Cl = S.limber_angular_power_spectra(H_interp, chi_interp, h, l=ll_subset, windows=['galaxy', 'GW', 'rsd', 'lsd'])

            # t0 = time.time()
            print('\nComputing the angular power spectra autocorrelation...\n')
            Cl_lens = S.limber_angular_power_spectra_lensing_auto(H_interp, chi_interp, alpha_M_interp,
                                                                  l=ll_subset, s_gal=s_gal, beta=beta,
                                                                  windows=['lensing_gal', 'lensing_GW'],
                                                                  n_points=n_points, n_points_x=n_points_x,
                                                                  z_min=z_min, grid_x=grid_x, n_low=n_low,
                                                                  n_high=n_high)
            # print(f"  Cl_lens_auto: {time.time() - t0:.3f}s")

            print('\nComputing the angular power spectra cross-correlation...\n')
            # t0 = time.time()
            Cl_lens_cross = S.limber_angular_power_spectra_lensing_cross(H_interp, chi_interp, alpha_M_interp, l=ll_subset,
                                                                         s_gal=s_gal, beta=beta,
                                                                         windows=None, n_points=n_points,
                                                                         n_points_x=n_points_x,
                                                                         z_min=z_min, grid_x=grid_x,
                                                                         n_low=n_low,
                                                                         n_high=n_high)
            # print(f"  Cl_lens_cross: {time.time() - t0:.3f}s")

            # Extract auto and cross angular power spectra
            Cl_delta_GG = Cl['galaxy-galaxy']
            Cl_delta_GWGW = Cl['GW-GW']
            Cl_delta_GGW = Cl['galaxy-GW']

            Cl_len_GG = Cl_lens['lensing_gal-lensing_gal']
            Cl_len_GWGW = Cl_lens['lensing_GW-lensing_GW']
            Cl_len_GGW = Cl_lens['lensing_gal-lensing_GW']

            Cl_RSD_GG = Cl['rsd-rsd']
            Cl_RSD_GWGW = Cl['lsd-lsd']
            Cl_RSD_GGW = Cl['rsd-lsd']

            Cl_delta_len_GG = Cl_lens_cross['galaxy-lensing_gal']
            Cl_delta_len_GWGW = Cl_lens_cross['GW-lensing_GW']
            Cl_delta_len_GGW = Cl_lens_cross['galaxy-lensing_GW']
            Cl_delta_len_GWG = Cl_lens_cross['GW-lensing_gal']

            Cl_delta_RSD_GG = Cl['galaxy-rsd']
            Cl_delta_RSD_GWGW = Cl['GW-lsd']
            Cl_delta_RSD_GGW = Cl['galaxy-lsd']
            Cl_delta_RSD_GWG = Cl['GW-rsd']

            Cl_RSD_len_GG = Cl_lens_cross['rsd-lensing_gal']
            Cl_RSD_len_GWGW = Cl_lens_cross['lsd-lensing_GW']
            Cl_RSD_len_GGW = Cl_lens_cross['rsd-lensing_GW']
            Cl_RSD_len_GWG = Cl_lens_cross['lsd-lensing_gal']

            # Ensure matrix symmetry where needed
            Cl_delta_len_GWG = np.swapaxes(Cl_delta_len_GWG, 0, 1)
            Cl_delta_RSD_GWG = np.swapaxes(Cl_delta_RSD_GWG, 0, 1)
            Cl_RSD_len_GWG = np.swapaxes(Cl_RSD_len_GWG, 0, 1)

            # Combine all contributions to total angular power spectra
            Cl_GG = Cl_delta_GG + Cl_len_GG + Cl_RSD_GG + 2 * Cl_delta_len_GG + 2 * Cl_delta_RSD_GG + 2 * Cl_RSD_len_GG
            Cl_GWGW = Cl_delta_GWGW + Cl_len_GWGW + Cl_RSD_GWGW + 2 * Cl_delta_len_GWGW + 2 * Cl_delta_RSD_GWGW + 2 * Cl_RSD_len_GWGW
            Cl_GGW = (Cl_delta_GGW + Cl_len_GGW + Cl_RSD_GGW +
                      Cl_delta_len_GGW + Cl_delta_len_GWG +
                      Cl_delta_RSD_GGW + Cl_delta_RSD_GWG +
                      Cl_RSD_len_GGW + Cl_RSD_len_GWG)

        return Cl_GG, Cl_GWGW, Cl_GGW

else:
    # Define function to compute Cl from galaxy and GW clustering only
    def Cl_func(universe, params,eftcamb_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll_subset, b_gal, b_GW):
        S = LLG.limber(cosmology=universe, z_limits=[z_m, z_M])

        # Define power spectrum grids
        kk = np.geomspace(1e-4, 10, 500)  # in [h/Mpc]
        zz = np.linspace(0, z_M, 100)
        # Compute nonlinear matter power spectrum
        bg, kk_out, zz_out, pkz = universe.MG_pk(kk, zz, eftcamb_params, True)  # pkz (Mpc/h)^3

        if bg == 0 and pkz == 0:
            return np.array(0), np.array(0), np.array(0)
        else:
            #print("pkz shape", pkz.shape)
            #print("bg", bg.keys())
            S.load_power_spectra(z=zz_out, k=kk_out, power_spectra=pkz)

            # Generate GW distribution from fiducial parameters
            A = gw_params['A']
            Alpha = gw_params['Alpha']
            log_delta_dl = gw_params['log_delta_dl']
            log_dl = gw_params['log_dl']
            Z_0 = gw_params['Z_0']
            Beta = gw_params['Beta']

            h = params['h']

            z_bg = np.asarray(bg['z'])
            H_bg = np.asarray(bg["H"], dtype=float)  # 1/Mpc ----> From EFTCamb, H=[1/Mpc]
            chi_bg = np.asarray(bg["chi"], dtype=float)  # Mpc ---> default via EFTCamb

            H_interp = interp1d(
                z_bg, H_bg,
                kind="cubic",
                bounds_error=False,
                fill_value=(H_bg[0], H_bg[-1])  # clamp outside range
            )

            chi_interp = interp1d(
                z_bg, chi_bg,
                kind="cubic",
                bounds_error=False,
                fill_value=(chi_bg[0], chi_bg[-1])  # clamp outside range
            )

            alpha_bg = np.asarray(bg["alpha_M"], dtype=float)
            if np.any(np.abs(alpha_bg) > 1e-10):
                alpha_M_interp = interp1d(
                    z_bg, alpha_bg,
                    kind="cubic",
                    bounds_error=False,
                    fill_value=(alpha_bg[0], alpha_bg[-1])
                )
            else:
                alpha_M_interp = interp1d(
                    z_bg,
                    np.zeros_like(z_bg),
                    kind="cubic",
                    bounds_error=False,
                    fill_value=0.0
                )

            # Generate GW source distribution
            # z_GW no dim; bin_GW_converted ; ndl_GW 1/Gpc; n_GW [Gpc]; total no dim
            z_GW, bin_GW_converted, ndl_GW, n_GW, total = fcc.merger_rate_dl_new(
                chi=chi_interp,
                alpha_M_interp=alpha_M_interp,
                z_max=z_M,
                dl=dl_GW,  # Mpc
                bin_dl=bin_edges_dl,
                log_dl=log_dl,
                log_delta_dl=log_delta_dl,
                A=A,
                Z_0=Z_0,
                Alpha=Alpha,
                Beta=Beta,
                normalize=False
            )

            #print('Loading the window functions...')

            # Load binning and window functions
            S.load_bin_edges(bin_edges, bin_GW_converted)
            S.load_galaxy_clustering_window_functions(H_interp, z=z_gal, n_z=nz_gal, ll=ll_subset, bias=b_gal,name='galaxy')
            S.load_gravitational_wave_window_functions(H_interp, chi_interp, alpha_M_interp, z=z_GW,n_dl=ndl_GW, ll=ll_subset, bias=b_GW, name='GW')

            #print('Computing the angular power spectra...')
            # Compute angular power spectra (density terms only)
            Cl = S.limber_angular_power_spectra(H_interp, chi_interp, h, l=ll_subset, windows=None)

            # Galaxy-GW
            Cl_delta_GGW = Cl['galaxy-GW']

            # Galaxy-Galaxy
            Cl_delta_GG = Cl['galaxy-galaxy']

            # GW-GW
            Cl_delta_GWGW = Cl['GW-GW']

            return Cl_delta_GG, Cl_delta_GWGW, Cl_delta_GGW


if name_component in ["GW","Gal","Cross"]:
    def Cl_UPDATE(cosmo_params, eftcamb_params, theta_dict, z_mean_GW, z_mean_gal, b_gal, b_GW,
                  noise_loc_mat_auto,noise_loc_mat,noise_GW, noise_gal, names):
        """
            Given parameter vector θ, compute the theory spectra vector

                C(ℓ; θ) = [C_ℓ^{GG}, C_ℓ^{GWGW}, C_ℓ^{GGW}]

            on the same ell-grid as the fiducial model, and return it as
            a 2D array of shape (3, N_ell).

            Steps:
              1) Build cosmo_params from θ.
              2) Call hi_CLASS via cc.cosmo(**params).
              3) Call your Cl_func(...) with these settings.
        """
        # print('\nUpdating parameters...')
        params = deepcopy(cosmo_params)
        #print("proposed upgrade",theta_dict)
        #print("Previous params", cosmo_params)

        if 'H0' in theta_dict:
            params['h'] = theta_dict['H0'] / 100

        for name in names:
            # Upgrading Cosmology
            if name == 'A_GW' or name == 'gamma_GW':
                # Compute mean redshift for each GW bin and corresponding GW bias
                b_GW = theta_dict['A_GW'] * (1. + z_mean_GW) ** theta_dict['gamma_GW']
            elif name == 'A_gal' or name == 'gamma_gal':
                # Compute galaxy bias using polynomial model
                b_gal = theta_dict['A_gal'] * (1. + z_mean_gal) ** theta_dict['gamma_gal']
            elif name == 'H0':
                pass #params['h'] = theta_dict['H0'] / 100
            elif name == 'Omega_m':
                params['omega_m'] = theta_dict['Omega_m'] * params['h'] ** 2
            elif name == 'Omega_b':
                params['omega_b'] = theta_dict['Omega_b'] * params['h'] ** 2
            elif name == "alpha_M":
                eftcamb_params['RPHalphaM_ODE0'] = theta_dict['alpha_M']
            elif name == "alpha_B":
                eftcamb_params['RPHbraiding_ODE0'] = -0.5*theta_dict['alpha_B']
            elif name == "wa":
                #params['wa'] = theta_dict['wa']
                #eftcamb_params['EFTwa'] = theta_dict['wa']
                eftcamb_params['RPHwa'] = theta_dict['wa']
            elif name == "w0":
                #params['w'] = theta_dict['w']
                #eftcamb_params['EFTw0'] = theta_dict['w0']
                eftcamb_params['RPHw0'] = theta_dict['w0']
            else:
                params[f'{name}'] = theta_dict[f'{name}']


        # Cosmology object from CLASS / hi_class
        universe = cc_MG.cosmo(**params)

        Cl_GG_update, Cl_GWGW_update, Cl_GGW_update = Cl_func(universe, params, eftcamb_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll_subset, b_gal, b_GW)

        if (np.allclose(Cl_GG_update, 0.0) and
                np.allclose(Cl_GWGW_update, 0.0) and
                np.allclose(Cl_GGW_update, 0.0)):
            return np.array(0)

        else:
            Cl_GWGW_total = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))
            for i in range(n_bins_dl):
                for ii in range(n_bins_dl):
                    Cl_GWGW_interp = si.interp1d(ll_subset, Cl_GWGW_update[i, ii], kind='cubic')
                    Cl_GWGW_total[i, ii] = Cl_GWGW_interp(ll_total)

            Cl_GWGW_total = Cl_GWGW_total * noise_loc_mat_auto
            Cl_GWGW_total += noise_GW

            Cl_GG_total = np.zeros(shape=(n_bins_z, n_bins_z, len(ll_total)))
            for i in range(n_bins_z):
                for ii in range(n_bins_z):
                    Cl_GG_interp = si.interp1d(ll_subset, Cl_GG_update[i, ii], kind='cubic')
                    Cl_GG_total[i, ii] = Cl_GG_interp(ll_total)

            Cl_GG_total += noise_gal

            Cl_GGW_total = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))
            for i in range(n_bins_z):
                for ii in range(n_bins_dl):
                    Cl_GGW_interp = si.interp1d(ll_subset, Cl_GGW_update[i, ii], kind='cubic')
                    Cl_GGW_total[i, ii] = Cl_GGW_interp(ll_total)
            Cl_GGW_total = Cl_GGW_total * noise_loc_mat

            Cl_vector_updated, _, _ = LH_fun.build_vector(n_bins_z, n_bins_dl, Cl_GGW_total, Cl_GWGW_total, Cl_GG_total,
                                                          l_max_nl_, l_max_loc)

            if name_component == "GW":
                Cl_vec = Cl_vector_updated[autoX_len + crossXY_len:]
                # vec_idx_GWGW = vec_idx[autoX_len + crossXY_len:]

            elif name_component == "Gal":
                Cl_vec = Cl_vector_updated[:autoX_len]
                # vec_idx_GG = vec_idx[:autoX_len]
            else:
                Cl_vec = Cl_vector_updated[autoX_len:autoX_len + crossXY_len]
                # vec_idx_GGW = vec_idx[autoX_len:autoX_len + crossXY_len]

            return Cl_vec
else:
    def Cl_UPDATE(cosmo_params, eftcamb_params, theta_dict, z_mean_GW, z_mean_gal, b_gal, b_GW, noise_loc_mat_auto,
                  noise_loc_mat,noise_GW, noise_gal, names):
        """
            Given parameter vector θ, compute the theory spectra vector

                C(ℓ; θ) = [C_ℓ^{GG}, C_ℓ^{GWGW}, C_ℓ^{GGW}]

            on the same ell-grid as the fiducial model, and return it as
            a 2D array of shape (3, N_ell).

            Steps:
              1) Build cosmo_params from θ.
              2) Call EFTCamb via cc.cosmo(**params).
              3) Call your Cl_func(...) with these settings.
        """
        # print('\nUpdating parameters...')
        params = deepcopy(cosmo_params)

        #print("proposed upgrade", theta_dict)
        #print("Previous params", cosmo_params)
        #print('eftcamb_params pre', eftcamb_params)


        # aggiungi questo debug dentro log_likelihood
        #print(f"h in params prima del loop: {params['h']:.4f}")
        #print(f"H0 in theta_dict: {theta_dict.get('H0', 'NON PRESENTE')}")

        if 'H0' in theta_dict:
            params['h'] = theta_dict['H0'] / 100

        #print(f"h dopo update: {params['h']:.4f}")

        for name in names:
            # Upgrading Cosmology
            if name == 'A_GW' or name == 'gamma_GW':
                # Compute mean redshift for each GW bin and corresponding GW bias
                b_GW = theta_dict['A_GW'] * (1. + z_mean_GW) ** theta_dict['gamma_GW']
            elif name == 'A_gal' or name == 'gamma_gal':
                # Compute galaxy bias using polynomial model
                b_gal = theta_dict['A_gal'] * (1. + z_mean_gal) ** theta_dict['gamma_gal']
            elif name == 'H0':
                pass #params['h'] = theta_dict['H0'] / 100
            elif name == 'Omega_m':
                params['omega_m'] = theta_dict['Omega_m'] * params['h'] ** 2
            elif name == 'Omega_b':
                params['omega_b'] = theta_dict['Omega_b'] * params['h'] ** 2
            elif name == "alpha_M":
                eftcamb_params['RPHalphaM_ODE0'] = theta_dict['alpha_M']
            elif name == "alpha_B":
                eftcamb_params['RPHbraiding_ODE0'] = -0.5*theta_dict['alpha_B']
            elif name == "wa":
                # params['wa'] = theta_dict['wa']
                # eftcamb_params['EFTwa'] = theta_dict['wa']
                eftcamb_params['RPHwa'] = theta_dict['wa']
            elif name == "w0":
                # params['w'] = theta_dict['w']
                # eftcamb_params['EFTw0'] = theta_dict['w0']
                eftcamb_params['RPHw0'] = theta_dict['w0']
            else:
                params[f'{name}'] = theta_dict[f'{name}']


        # Cosmology object from EFTCAMB
        universe = cc_MG.cosmo(**params)

        Cl_GG_update, Cl_GWGW_update, Cl_GGW_update = Cl_func(universe, params, eftcamb_params, gw_params, dl_GW,
                                                              bin_edges_dl, z_gal, ll_subset, b_gal, b_GW)

        if (np.allclose(Cl_GG_update, 0.0) and
                np.allclose(Cl_GWGW_update, 0.0) and
                np.allclose(Cl_GGW_update, 0.0)):
            return np.array(0)

        else:
            Cl_GG_total = np.zeros(shape=(n_bins_z, n_bins_z, len(ll_total)))
            Cl_GWGW_total = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))
            Cl_GGW_total = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))

            for i in range(n_bins_z):
                for ii in range(n_bins_z):
                    Cl_GG_interp = si.interp1d(ll_subset, Cl_GG_update[i, ii], kind='cubic')
                    Cl_GG_total[i, ii] = Cl_GG_interp(ll_total)

            for i in range(n_bins_dl):
                for ii in range(n_bins_dl):
                    Cl_GWGW_interp = si.interp1d(ll_subset, Cl_GWGW_update[i, ii], kind='cubic')
                    Cl_GWGW_total[i, ii] = Cl_GWGW_interp(ll_total)

            for i in range(n_bins_z):
                for ii in range(n_bins_dl):
                    Cl_GGW_interp = si.interp1d(ll_subset, Cl_GGW_update[i, ii], kind='cubic')
                    Cl_GGW_total[i, ii] = Cl_GGW_interp(ll_total)

            Cl_GWGW_total = Cl_GWGW_total * noise_loc_mat_auto
            Cl_GGW_total = Cl_GGW_total * noise_loc_mat

            Cl_GWGW_total += noise_GW
            Cl_GG_total += noise_gal

            Cl_vector_updated, _, _ = LH_fun.build_vector(n_bins_z, n_bins_dl, Cl_GGW_total, Cl_GWGW_total, Cl_GG_total,
                                                          l_max_nl_, l_max_loc)

            return Cl_vector_updated


def log_likelihood(theta,names, verbose=False):

    theta_dict={}
    for i,name in enumerate(names):
        theta_dict[f'{name}']= theta[i]

    diff=theta_dict['Omega_m']-theta_dict['Omega_b']
    # Compute theory Cl's for this θ
    if diff<1e-10:
        return -np.inf

    #if theta_dict['w0'] >= -1 / 3:  # acceleration
    #    return -np.inf

    #if theta_dict['w0'] + theta_dict['wa'] >= 0:  # stability at high z
    #    return -np.inf

    Cl_new = Cl_UPDATE(cosmo_params,eftcamb_params, theta_dict, z_mean_GW, z_mean_gal, bias_gal, bias_GW, noise_loc_mat_auto,noise_loc_mat,noise_GW, noise_gal, names)  # shape (3, N_ell)
    # Reject failed / invalid outputs robustly
    if (Cl_new is None) or (not np.isfinite(Cl_new).all()) or (np.all(Cl_new == 0)):
        return -np.inf

    # Difference with fiducial mock data
    Delta = Cl_new - Cl_fid  # shape (3, N_ell)

    #print("Delta shape:", Delta.shape)
    #print("F_t shape:  ", F_t.shape)
    #print("Cl_fid shape:", Cl_fid.shape)
    #print("Cl_new shape:", Cl_new.shape)

    # Dentro log_likelihood
    logL = -0.5  * np.einsum('li,lik,lk->l', Delta.T,F_t,Delta.T)  # +0.5* np.linalg.slogdet(F.transpose(2, 0, 1))[1]

    if not np.isfinite(logL).all():
        return -np.inf

    s = np.sum(logL)
    if not np.isfinite(s):
        return -np.inf

    return float(s)



def log_post(x, verbose=False):
     
     # x has len (D) with D=number of parameters of mcmc
     logprior = myPrior.logpdf(x)
     
     if not np.isfinite(logprior):
         if verbose:
             print('Neg infinite logprior=%s'%logprior)
         return -np.inf

     th = get_theta( x, settings["params_inference"], params_fixed_dict, settings["all_params"])
     
     if verbose:
         print('theta log post: %s' %th)
     
     ll = log_likelihood(th,settings["all_params"], verbose=verbose)
     
     if verbose:
         print('prior, likelihood:')
         print(logprior, ll)
     
     return ll+logprior   


print()
print('='*40)
print('Running mcmc...')
print('='*40)

#cb0 = zeus.callbacks.AutocorrelationCallback( ncheck=50, dact=settings["eps"], nact=settings["ntaus"], discard=0.)
ndim = len(settings["params_inference"])

if not FLAGS.resume:
    print('\nInitializing chain within uniform prior...')
    p0 = get_init_point_flat( myPrior, [ settings["fiducial_vals"][p] for p in settings["params_inference"] ], settings["nwalkers"], ndim, settings["eps"], verbose=True)
    #cb2 = zeus.callbacks.MinIterCallback(nmin=500)
else:
    print('Resuming from last point of the chain. Doing at least %s extra steps' %FLAGS.nsteps)
    #cb2 = zeus.callbacks.MinIterCallback(nmin=FLAGS.nsteps)
    print('Reading chains from %s...'%os.path.join(FLAGS.fout, "chains.h5"))
    with h5py.File(os.path.join(FLAGS.fout, "chains.h5"), "r") as hf:
        all_samples = np.copy(hf['samples'])
    p0 = all_samples[-1, :, :]

print('Saving chains in %s ' %os.path.join(FLAGS.fout, chainName+".h5"))
#cbsave = zeus.callbacks.SaveProgressCallback( os.path.join(FLAGS.fout, chainName+".h5"), ncheck=10)
#cbs = [cb0, cb2 , cbsave]

print('Using %s parallel processes' %settings["nprocesses"])
myPool =  get_pool( mpi=settings["mpi"], threads=settings["nprocesses"]+1)

with myPool as pool:

    backend = emcee.backends.HDFBackend(os.path.join(FLAGS.fout, chainName + ".h5"))
    if not FLAGS.resume:
        backend.reset(settings["nwalkers"], ndim)

    sampler = emcee.EnsembleSampler(
        settings["nwalkers"], ndim, log_post,
        pool=pool, backend=backend,
    )

    # ── Check convergence
    ncheck  = settings.get("ncheck", 500)
    old_tau = np.inf

    for sample in sampler.sample(p0, iterations=settings["maxsteps"],
                                  progress=True, skip_initial_state_check=False):
        if sampler.iteration % ncheck:
            continue

        tau = sampler.get_autocorr_time(tol=0)
        cond1 = np.all(sampler.iteration > settings["ntaus"] * tau)
        cond2 = np.all(np.abs(old_tau - tau) / tau < settings["eps"])
        old_tau = tau

        #print(f"Step {sampler.iteration}: tau_max={np.max(tau):.1f}, "f"steps/tau={sampler.iteration / np.max(tau):.1f}, "f"cond1={cond1}, cond2={cond2}")

        if cond1 and cond2:
            #iterations=sampler.iteration
            print(f"The chains have converged. # of steps (approx): {sampler.iteration}")
            break
    iterations = sampler.iteration
    burnin, thin, tau, ESS, ratio = diagnostic(sampler, iterations, settings)

    samples = summary(sampler, burnin, thin, tau, ESS, ratio, iterations, settings, FLAGS.fout)

    traceplots(sampler, burnin, settings, FLAGS.fout)

    plot_corner(samples, settings, [settings["fiducial_vals"][p] for p in settings["params_inference"]], myPrior,FLAGS.fout, nsteps='')


""" to use when need to look for stable initial points
print('Using %s parallel processes' %settings["nprocesses"])
myPool =  get_pool( mpi=settings["mpi"], threads=settings["nprocesses"]+1)

with myPool as pool:

    backend = emcee.backends.HDFBackend(os.path.join(FLAGS.fout, chainName + ".h5"))
    if not FLAGS.resume:
        backend.reset(settings["nwalkers"], ndim)

    sampler = emcee.EnsembleSampler(
        settings["nwalkers"], ndim, log_post,
        pool=pool, backend=backend,
    )

    # inizializzazione DENTRO il pool, così i worker sono già attivi
    print("Finding valid starting points...")
    p0 = find_valid_start(
        log_post_fn=sampler.log_prob_fn,  # usa il sampler che già distribuisce sul pool
        prior=myPrior,
        expected_vals=[settings["fiducial_vals"][p] for p in settings["params_inference"]],
        nwalkers=settings["nwalkers"],
        ndim=ndim,
        eps=settings["eps"],
        verbose=True
    )


    # ── Check convergence
    ncheck  = settings.get("ncheck", 500)
    old_tau = np.inf

    for sample in sampler.sample(p0, iterations=settings["maxsteps"],
                                  progress=True, skip_initial_state_check=False):
        if sampler.iteration % ncheck:
            continue

        tau = sampler.get_autocorr_time(tol=0)
        cond1 = np.all(sampler.iteration > settings["ntaus"] * tau)
        cond2 = np.all(np.abs(old_tau - tau) / tau < settings["eps"])
        old_tau = tau

        #print(f"Step {sampler.iteration}: tau_max={np.max(tau):.1f}, "f"steps/tau={sampler.iteration / np.max(tau):.1f}, "f"cond1={cond1}, cond2={cond2}")

        if cond1 and cond2:
            iterations=sampler.iteration
            print(f"The chains have converged. # of steps (approx): {iterations}")
            break

    burnin, thin, tau, ESS, ratio = diagnostic(sampler, iterations, settings)

    samples = summary(sampler, burnin, thin, tau, ESS, ratio, iterations, settings, FLAGS.fout)

    traceplots(sampler, burnin, settings, FLAGS.fout)

    plot_corner(samples, settings, [settings["fiducial_vals"][p] for p in settings["params_inference"]], myPrior,FLAGS.fout, nsteps='')

"""
    
if settings["mpi"] == 'mpi':
    pool.close()
    sys.exit(0)
    
print('='*40)
print('Done in %s sec.'%str( time.time()-tin ) )
print('='*40)
    
myLog.close()


##################### TEST FUNCTIONS
'''
def fake_MG_pk(kk, zz, eftcamb_params, nonlinear):
    """
    Mock di MG_pk. Restituisce un background e un P(k,z) analitico
    basato sul modello di Harrison-Zel'dovich con crescita ΛCDM approssimata.
    """
    # ── Background farlocco ───────────────────────────────────────────────────
    H0 = 67.81 / 2.99792458e5   # 1/Mpc
    Omega_m = 0.31
    Omega_L = 1 - Omega_m

    H_z   = H0 * np.sqrt(Omega_m * (1 + zz)**3 + Omega_L)
    chi_z = np.array([
        np.trapezoid(1.0 / (H0 * np.sqrt(Omega_m * (1 + np.linspace(0, z, 500))**3 + Omega_L)),
                 np.linspace(0, z, 500))
        for z in zz
    ])

    bg = {
        'z'      : zz,
        'H'      : H_z,
        'chi'    : chi_z,
        'alpha_M': np.zeros_like(zz),   # GR → alpha_M = 0
    }

    # ── P(k, z) farlocco ─────────────────────────────────────────────────────
    # Spettro di potenza: P(k) ∝ k * T(k)^2 con transfer function BBKS
    # scalato per crescita lineare D(z) ∝ 1/(1+z)  (approssimazione piatta)
    k0 = 0.01      # Mpc^-1, scala pivot
    A  = 1e4       # ampiezza arbitraria

    def transfer_BBKS(k, Omega_m=0.31, h=0.6781):
        q = k / (Omega_m * h**2)
        return np.log(1 + 2.34*q) / (2.34*q) * (
            1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
        )**(-0.25)

    Tk  = transfer_BBKS(kk)
    Pk0 = A * kk * Tk**2          # shape (n_k,)

    # Fattore di crescita lineare approssimato D(z) = 1/(1+z)
    Dz  = 1.0 / (1.0 + zz)        # shape (n_z,)

    # pkz[i, j] = P(k_j, z_i)
    pkz = np.outer(Dz**2, Pk0)    # shape (n_z, n_k)

    return bg, None, None, pkz
'''

'''
# test log_likelihoood
def log_likelihood(theta, names, verbose=False):

    theta_dict = {}
    for i, name in enumerate(names):
        theta_dict[name] = theta[i]

    if theta_dict['Omega_m'] < theta_dict['Omega_b']:
        return -np.inf

    # Gaussiana multivariata centrata sui valori fiduciali
    fiducials = np.array([settings["fiducial_vals"][n] for n in names])
    sigmas    = np.array([
        abs(settings["fiducial_vals"][n]) * 0.05 if settings["fiducial_vals"][n] != 0 else 0.05
        for n in names
    ])

    logL = -0.5 * np.sum(((theta - fiducials) / sigmas) ** 2)

    return float(logL)
'''

