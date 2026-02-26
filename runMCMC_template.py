#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:46:14 2022

@author: Michi
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import multiprocessing
# Needed for mpipool not to stall when trying to write on a file (do not ask me why)
#multiprocessing.set_start_method("spawn",force=True)
import zeus
import corner
import argparse
import shutil
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py


####
import functions_cross_correlation as fcc
import functions_extra_main as fem
import colibri.cosmology as cc
import colibri.limber_GW as LLG
import likelihood_functions as LH_fun
import importlib
import scipy.interpolate as si
from scipy.interpolate import interp1d
import time




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
        self.log.close
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
    allinit = np.empty( (nwalkers, ndim))
    for i in range(ndim):    
        if expected_vals[i]!=0:
            linf = expected_vals[i]*np.abs( (1-eps) )
            lsup = expected_vals[i]*np.abs( (1+eps) )
        else:
            linf = -eps
            lsup = eps
        pinf = max( linf, prior.lims[i][0] )
        psup = min( lsup, prior.lims[i][1])
        if verbose:
            print('For param %s, eps=%s, min=%s, max=%s, central value=%s' %(i, eps, pinf, psup, expected_vals[i]))
        for k in range(nwalkers):
            allinit[k, i] = np.random.uniform( low= pinf, high=psup, size=1) 
    assert np.all(~np.isnan(allinit))
    return allinit



def plot_corner(samples, settings, fiducials, myPrior, out_path, nsteps):
    
    try:
        print('Plotting corner...')
        
        eps=0.0005
        
        myrange=[ ( samples[:, i].min()*(1-eps), samples[:, i].max()*(1+eps)) for i in range(samples.shape[1])] #range(samples.shape[0]) ]
        
        _ = corner.corner(samples, labels=settings["params_inference"],
                               range=myrange, 
                                truths=fiducials,
                               quantiles=[0.05, 0.95],
                               show_titles=True, title_kwargs={"fontsize": 12},
                               smooth=0.5, color='darkred',
                               levels=[0.68, 0.90],
                               density=True,
                               verbose=False, 
                               plot_datapoints=True, 
                               fill_contours=True,
                               )
        
      
        
        plt.savefig(os.path.join(out_path, 'corner_%s.png'%nsteps) )
        plt.close('all')
        print("Corner ok")
    except Exception as e:
        print(e)
        print('No corner for this event!')




def get_pool(mpi=False, threads=None):
    """ Always returns a pool object with a `map()` method. By default,
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
    logfile=os.path.join(FLAGS.fout, baselogfileName+'_run1.txt')
    nResume=1
    while os.path.exists(logfile):
        logfileName=baselogfileName+'_run'+str(nResume)+'.txt'
        logfile=os.path.join(FLAGS.fout, logfileName)
        chainName = baseChainName+'_run'+str(nResume)
        nResume += 1
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

# Dictionary to hold imported config modules
configspath = 'configs/'
sys.path.append(configspath)

configs = {}
config_names = FLAGS.config.split(',')  # if FLAGS.config is a comma-separated string
print('configuration files:',config_names)
for cfg_name in config_names:
    # Copy the config file into the output directory
    src = os.path.join(configspath, f"{cfg_name}.py")
    # Dynamically import the config module
    configs[cfg_name] = importlib.import_module(cfg_name)

cosmo=config_names[1]
detectors= config_names[0]

# Initialize cosmology for power spectrum calculation
cosmo_params = configs[cosmo].COSMO_PARAMS

GW_det = configs[detectors].GW_det # GW detector (ET_Delta_2CE, ET_2L_2CE, ET_Delta_1CE, ET_2L_1CE, ET_Delta, ET_2L, LVK)
yr = configs[detectors].yr # Years of observation
gw_params = fem.load_detector_params(GW_det, yr)


main_config = GW_det  # ET_Delta_2CE, ET_2L_2CE, ET_Delta, ET_2L, LVK

# Define the redshift total range
z_m = configs[detectors].z_m
z_M = configs[detectors].z_M

# Define the number of bins
n_bins_z = configs[detectors].n_bins_z
n_bins_dl = configs[detectors].n_bins_dl

# Define the luminosity distance total range
dlm = configs[detectors].dlm
dlM = configs[detectors].dlM

l_min = configs[detectors].l_min

l_max= np.load(os.path.join(main_config, 'l_max.npy')) #
ll = np.load(os.path.join(main_config, 'll.npy')) #
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

Cl_fid = np.load(os.path.join(main_config, 'vec_fid.npy')) #
F      = np.load(os.path.join(main_config, 'cov_mat_inverse.npy')) #

z_mean_GW = np.load(os.path.join(main_config, 'z_mean_GW.npy')) #
z_mean_gal = np.load(os.path.join(main_config, 'z_mean_gal.npy')) #


# Define function to compute Cl from galaxy and GW clustering only
def Cl_func(Hi_Cosmo,params,gw_params,dl_GW,bin_edges_dl,z_gal,ll,b_gal, b_GW):

    S = LLG.limber(cosmology=Hi_Cosmo, z_limits=[z_m, z_M])

    # Define power spectrum grids
    kk = np.geomspace(1e-4, 10, 200)
    zz = np.linspace(0, z_M, 100)

    # Compute nonlinear matter power spectrum
    #t0 = time.time()
    bg,_,_, pkz = Hi_Cosmo.hi_class_pk(kk, zz, True) # pkz (Mpc/1)^3
    #t1 = time.time()
    #print(f"\t\t\tTime HI_CLASS POWER SPECTRUM: {t1 - t0:.4f} s")

    if bg == 0 and pkz == 0:
        #print('unstable cosmology')
        return np.array(0),np.array(0),np.array(0)
    else:
        #print('bg keys', bg.keys())
        #t0 = time.time()
        S.load_power_spectra(z=zz, k=kk, power_spectra=pkz)  # 0.05
        #t1 = time.time()
        #print(f"\t\t\tTime load_power_spectra: {t1 - t0:.4f} LOAD POWER SPECTRUM s")

        # Generate GW distribution from fiducial parameters
        A = gw_params['A']
        Alpha = gw_params['Alpha']
        log_delta_dl = gw_params['log_delta_dl']
        log_dl = gw_params['log_dl']
        Z_0 = gw_params['Z_0']
        Beta = gw_params['Beta']

        h = params['h']

        z_bg = np.asarray(bg['z'])
        H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False,fill_value="extrapolate")  # [1/Mpc]
        chi_interp = interp1d(z_bg, bg['comov. dist.'], kind='cubic', bounds_error=False,fill_value="extrapolate")  # Mpc

        if 'M2_running_smg' in bg:
            alpha_M_interp = interp1d(
                z_bg,
                bg['M2_running_smg'],
                kind='cubic',
                bounds_error=False,
                fill_value="extrapolate"
            )
        else:
            alpha_M_interp = interp1d(
                z_bg,
                np.zeros_like(z_bg),
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )

        #t0 = time.time()
        # Generate GW source distribution
        # z_GW no dim; bin_GW_converted ; ndl_GW 1/Gpc; n_GW [Gpc]; total no dim
        z_GW, bin_GW_converted, ndl_GW, n_GW, total = fcc.merger_rate_dl_new(
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
            C=Hi_Cosmo,
            normalize=False
        )
        #t1 = time.time()
        #print(f"\t\t\tTime merger_rate_dl_new: {t1 - t0:.4f}  BIN + OTHER GWs")

        #print('\nLoading the window functions...')

        #t0 = time.time()
        # Load binning and window functions ----> 0.3
        S.load_bin_edges(bin_edges, bin_GW_converted)
        #t1 = time.time()
        #print(f"\t\t\tTime loading BIN EDGES: {t1 - t0:.4f}")

        #t0 = time.time()
        S.load_galaxy_clustering_window_functions(H_interp, z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal, name='galaxy')
        #t1 = time.time()
        #print(f"\t\t\tTime loading GALAXY windows: {t1 - t0:.4f}")

        #t0 = time.time()
        S.load_gravitational_wave_window_functions(H_interp,chi_interp, alpha_M_interp, C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, bias=b_GW,name='GW')
        #t1 = time.time()
        #print(f"\t\t\tTime loading GW windows: {t1 - t0:.4f}")

        #print('\nComputing the angular power spectra...')
        # Compute angular power spectra (density terms only)
        #print('ll',ll)
        #start = time.time()
        Cl = S.limber_angular_power_spectra(H_interp,chi_interp, h, l=ll, windows=None)
        #end = time.time()
        #print(f"\t\t\tTime took {end - start:.4f} seconds for LIMBER\n")

        # Galaxy-GW
        Cl_delta_GGW = Cl['galaxy-GW']

        # Galaxy-Galaxy
        Cl_delta_GG = Cl['galaxy-galaxy']

        # GW-GW
        Cl_delta_GWGW = Cl['GW-GW']

        return Cl_delta_GG, Cl_delta_GWGW, Cl_delta_GGW


def Cl_UPDATE(cosmo_params, theta_dict, z_mean_GW, z_mean_gal, b_gal, b_GW, noise_loc_mat_auto, noise_loc_mat,names):
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
    #print('\nUpdating parameters...')
    params = cosmo_params
    for name in names:
        # Upgrading Cosmology
        if name == 'A_GW' or name == 'A_gal' or name == 'gamma_GW' or name == 'gamma_gal':
            # Compute mean redshift for each GW bin and corresponding GW bias
            b_GW = theta_dict['A_GW'] * (1. + z_mean_GW) ** theta_dict['gamma_GW']

            # Compute galaxy bias using polynomial model
            b_gal = theta_dict['A_gal'] * (1. + z_mean_gal) ** theta_dict['gamma_gal']

        elif name == "alpha_M" or name == "alpha_B":
            params['parameters_smg'] = f"10.0,{theta_dict['alpha_B']},{theta_dict['alpha_M']},0.0,1.0"  # x_k, x_b, x_m, x_t, (M_*)^ 2_ini
        else:
            params[f'{name}'] = theta_dict[f'{name}']

    z_gal = np.linspace(z_m, z_M, 2000)  # Redshift grid for galaxy distribution
    dl_GW = np.linspace(dlm, dlM, 2000)  # Luminosity distance grid for gravitational wave sources in Mpc
    # Define multipole vector with increasing step sizes at higher l

    # Cosmology object from CLASS / hi_class
    Hi_Cosmo = cc.cosmo(**params)

    Cl_GG_update, Cl_GWGW_update, Cl_GGW_update = Cl_func(Hi_Cosmo, params, gw_params, dl_GW, bin_edges_dl, z_gal,ll, b_gal, b_GW)

    if (np.allclose(Cl_GG_update, 0.0) and
            np.allclose(Cl_GWGW_update, 0.0) and
            np.allclose(Cl_GGW_update, 0.0)):
        return np.array(0)  # IMPORTANT: correct shape

    else:
        Cl_GG_total = np.zeros(shape=(n_bins_z, n_bins_z, len(ll_total)))
        Cl_GWGW_total = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))
        Cl_GGW_total = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))

        for i in range(n_bins_z):
            for ii in range(n_bins_z):
                Cl_GG_interp = si.interp1d(ll, Cl_GG_update[i, ii])
                Cl_GG_total[i, ii] = Cl_GG_interp(ll_total)

        for i in range(n_bins_dl):
            for ii in range(n_bins_dl):
                Cl_GWGW_interp = si.interp1d(ll, Cl_GWGW_update[i, ii])
                Cl_GWGW_total[i, ii] = Cl_GWGW_interp(ll_total)

        for i in range(n_bins_z):
            for ii in range(n_bins_dl):
                Cl_GGW_interp = si.interp1d(ll, Cl_GGW_update[i, ii])
                Cl_GGW_total[i, ii] = Cl_GGW_interp(ll_total)

        Cl_GWGW_total = Cl_GWGW_total * noise_loc_mat_auto
        Cl_GGW_total = Cl_GGW_total * noise_loc_mat

        Cl_GWGW_total += noise_GW
        Cl_GG_total += noise_gal

        Cl_vector_updated, _, _ = LH_fun.build_vector(n_bins_z, n_bins_dl, Cl_GGW_total, Cl_GWGW_total, Cl_GG_total)

        return Cl_vector_updated


def log_likelihood(theta,names, verbose=False):


    theta_dict={}
    for i,name in enumerate(names):
        #print('dictionary', name,theta[i])
        theta_dict[f'{name}']= theta[i]
    #print(theta_dict)
    #start=time.time()
    # Compute theory Cl's for this θ
    if theta_dict['omega_m']<theta_dict['omega_b']:
        #end = time.time()
        #print(f"time for L {end-start}")
        return -np.inf
    Cl_new = Cl_UPDATE(cosmo_params, theta_dict, z_mean_GW, z_mean_gal, bias_gal, bias_GW, noise_loc_mat_auto,noise_loc_mat,names)  # shape (3, N_ell)

    # Reject failed / invalid outputs robustly
    if (Cl_new is None) or (not np.isfinite(Cl_new).all()) or (np.all(Cl_new == 0)):
        #end = time.time()
        #print(f"time for L {end-start}")
        return -np.inf

    # Difference with fiducial mock data
    Delta = Cl_new - Cl_fid  # shape (3, N_ell)
    logL = -0.5 * np.einsum('li,lik,lk->l', Delta.T, F.transpose(2, 0, 1),Delta.T)  # +0.5* np.linalg.slogdet(F.transpose(2, 0, 1))[1]
    if not np.isfinite(logL).all():
        #end = time.time()
        #print(f"time for L {end-start}")
        return -np.inf

    s = np.sum(logL)
    if not np.isfinite(s):
        return -np.inf

    #end = time.time()
    #print(f"time for L {end - start}")

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



cb0 = zeus.callbacks.AutocorrelationCallback( ncheck=50, 
                                             dact=settings["eps"], nact=settings["ntaus"], discard=0.)
ndim = len(settings["params_inference"])

if not FLAGS.resume:
    
    print('\nInitializing chain within uniform prior...')
    p0 = get_init_point_flat( myPrior, [ settings["fiducial_vals"][p] for p in settings["params_inference"] ], settings["nwalkers"], ndim, settings["eps"], verbose=True)
    cb2 = zeus.callbacks.MinIterCallback(nmin=500)
        
            
else:
    print('Resuming from last point of the chain. Doing at least %s extra steps' %FLAGS.nsteps)
    cb2 = zeus.callbacks.MinIterCallback(nmin=FLAGS.nsteps)
    print('Reading chains from %s...'%os.path.join(FLAGS.fout, "chains.h5"))
    with h5py.File(os.path.join(FLAGS.fout, "chains.h5"), "r") as hf:
        all_samples = np.copy(hf['samples'])
    
    p0 = all_samples[-1, :, :]

print('Saving chains in %s ' %os.path.join(FLAGS.fout, chainName+".h5"))
cbsave = zeus.callbacks.SaveProgressCallback( os.path.join(FLAGS.fout, chainName+".h5"), ncheck=10)

cbs = [cb0, cb2 , cbsave] 


print('Using %s parallel processes' %settings["nprocesses"])
myPool =  get_pool( mpi=settings["mpi"], threads=settings["nprocesses"]+1)

with myPool as pool:
    
    
    # do stuff
    
    sampler = zeus.EnsembleSampler( settings["nwalkers"], ndim, log_post, 
                                            pool=pool, 
                                            #blobs_dtype=dtype, 
                                            #moves=zeus.moves.GlobalMove(),
                                            vectorize=False, maxiter=10**3, 
                                            )

    
    

    sampler.run_mcmc(p0, settings["maxsteps"], callbacks=cbs, progress=True, )
        
    tau = cb0.estimates
      
    nbi = int(2 * np.max(tau))
    
    if np.all( ~np.isnan(tau)):
        burnin = nbi #int(2 * np.max(self.tau))
        try:
            thin = int( np.min(tau) )
        except:
            thin = int( np.min(tau) )
    else:
        burnin=0
        thin=1
    
    
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    allsamples = sampler.get_chain()
    
    
    print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(tau)))
    print('Samples extracted: %s' %str(samples.shape))
    

    plot_corner(samples, settings, settings["fiducial_vals"], myPrior, FLAGS.fout, nsteps='')
        
        
    
    

    
if settings["mpi"] == 'mpi':
    pool.close()
    sys.exit(0)
    
print('='*40)
print('Done in %s sec.'%str( time.time()-tin ) )
print('='*40)
    
myLog.close()

