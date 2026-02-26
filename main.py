#!/usr/bin/env python3
#-----------------------------------------------------------------------------------------
from typing import Any

import os

import argparse
import shutil
import json
import sys
import importlib
from inspect import getmembers, ismodule, isfunction
from tabulate import tabulate
from scipy.interpolate import interp1d
import itertools


import functions_cross_correlation as fcc
import functions_extra_main as fem
import colibri.cosmology as cc
import colibri.limber_GW as LLG
import likelihood_functions as LH_fun

from astropy.cosmology import FlatLambdaCDM
import astropy.cosmology.units as cu
from astropy import units as u

from functools import partial
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from scipy.integrate import trapezoid, simpson
from scipy.stats import multivariate_normal
import scipy.linalg
from scipy.interpolate import RectBivariateSpline
import scipy.interpolate as si

import time
import cProfile
plt.rc('font',size=20,family='serif')

configspath = 'configs/'

#-----------------------------------------------------------------------------------------

"""
Reads user inputs at runtime:
    --config = name of the config file (without .py).
    --fout = output folder path (where to save results).
"""
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='', type=str, required=False) # path to config file, in.json format
parser.add_argument("--fout", default='', type=str, required=True) # path to output folder
FLAGS = parser.parse_args()
#-----------------------------------------------------------------------------------------


if __name__=='__main__':
	os.makedirs(FLAGS.fout, exist_ok=True)

	sys.path.append(configspath)

	# Dictionary to hold imported config modules
	configs = {}
	config_names = FLAGS.config.split(',')  # if FLAGS.config is a comma-separated string

	for cfg_name in config_names:
		# Copy the config file into the output directory
		src = os.path.join(configspath, f"{cfg_name}.py")
		dst = os.path.join(FLAGS.fout, f"{cfg_name}_original.py")
		shutil.copy(src, dst)

		# Dynamically import the config module
		configs[cfg_name] = importlib.import_module(cfg_name)

	# import colibri
	sys.path.insert(0, configs['config_template_detectors'].colibri_path)

	# Initialize cosmology for power spectrum calculation
	cosmo_params = configs['config_template_LCDM_alphas'].COSMO_PARAMS

	#-----------------------------------------------------------------------------------------
    #					INITIAL SETTINGS FROM CONFIG(s)
	#-----------------------------------------------------------------------------------------
	# GW detector (ET_Delta_2CE, ET_2L_2CE, ET_Delta_1CE, ET_2L_1CE, ET_Delta, ET_2L, LVK)
	GW_det = configs['config_template_detectors'].GW_det

	# Years of observation
	yr = configs['config_template_detectors'].yr

	# Parameters for GW bias model
	A_GW = configs['config_template_detectors'].A_GW
	gamma_GW = configs['config_template_detectors'].gamma_GW

	# Parameters for Galaxy bias model
	A_gal = configs['config_template_detectors'].A_gal
	gamma_gal = configs['config_template_detectors'].gamma_gal

	# Define the number of bins
	n_bins_z = configs['config_template_detectors'].n_bins_z
	n_bins_dl = configs['config_template_detectors'].n_bins_dl

	# Define the galaxy bin range
	z_m_bin = configs['config_template_detectors'].z_m_bin
	z_M_bin = configs['config_template_detectors'].z_M_bin

	# Define the GW bin range in redshift (will be converted in dl using the fiducial model)
	z_m_bin_GW = configs['config_template_detectors'].z_m_bin_GW
	z_M_bin_GW = configs['config_template_detectors'].z_M_bin_GW

	# Set the binning strategy (right_cosmo, wrong_cosmo(H0=65, Om0=0.32), equal_pop, equal_space)
	bin_strategy = configs['config_template_detectors'].bin_strategy

	# Include the lensing
	Lensing = configs['config_template_detectors'].Lensing
	computed = configs['config_template_detectors'].computed
	computed_single_cov_inv=configs['config_template_detectors'].computed_single_cov_inv
	full = configs['config_template_detectors'].full

	# Fraction of the sky covered from the survey
	f_sky = configs['config_template_detectors'].f_sky
	f_sky_GW = configs['config_template_detectors'].f_sky_GW

	# Errors on the galaxy distribution
	sig_gal = configs['config_template_detectors'].sig_gal

	# galaxy survey (euclid_photo, euclid_spectro, ska)
	gal_det = configs['config_template_detectors'].gal_det

	l_min = configs['config_template_detectors'].l_min

	# Compute power spectra (True)
	fourier = configs['config_template_detectors'].fourier

	# Define the redshift total range
	z_m = configs['config_template_detectors'].z_m
	z_M = configs['config_template_detectors'].z_M

	# Define the luminosity distance total range
	dlm = configs['config_template_detectors'].dlm
	dlM = configs['config_template_detectors'].dlM

	# "True" values of the cosmological parameters
	H0_true = cosmo_params['h'] * 100  # if you want H0 in km/s/Mpc
	Omega_m_true = round(cosmo_params['omega_m']/(cosmo_params['h']**2),2)
	Omega_b_true = round(cosmo_params['omega_b']/(cosmo_params['h']**2),3)
	A_s = cosmo_params['A_s']*10**(9)
	n_s = cosmo_params['n_s']

	#print('H0_true:', cosmo_params['h']*100,'Omega_m', round(cosmo_params['omega_m']/(cosmo_params['h']**2),2),'Omega_b',round(cosmo_params['omega_b']/(cosmo_params['h']**2),3))

	parameters_smg = cosmo_params['parameters_smg']
	#print(type(parameters_smg), parameters_smg)  # This should show <class 'str'>
	parameters_smg = list(map(float, parameters_smg.split(','))) 	# Convert to list of floats
	alpha_B = parameters_smg[1]
	alpha_M = parameters_smg[2]
	#print(alpha_M,alpha_B)

	expansion_smg = cosmo_params['expansion_smg']
	#print(type(expansion_smg_smg), expansion_smg_smg)
	expansion_smg_list = list(map(float, expansion_smg.split(',')))  # Convert to list of floats
	if len(expansion_smg_list)==1:
		Omega_DE = expansion_smg_list[0]
		w_0 = -1
		w_a = 0
	else:
		Omega_DE = expansion_smg_list[0]
		w_0 = expansion_smg_list[1]
		w_a = expansion_smg_list[2]

	#print(Omega_DE,w_0,w_a)

	#-----------------------------------------------------------------------------------------
    #                   LOADING GW AND GALAXY PARAMETERS
	#-----------------------------------------------------------------------------------------
	# Load gravitational wave detector parameters
	gw_params = fem.load_detector_params(GW_det, yr)

	# Load galaxy detector parameters
	gal_params = fem.load_galaxy_detector_params(gal_det)

	# Call of the single parameters: first GW and second Galaxies
	A = gw_params['A']
	Alpha = gw_params['Alpha']
	log_loc = gw_params['log_loc']
	log_delta_dl = gw_params['log_delta_dl']
	log_dl = gw_params['log_dl']
	Z_0=gw_params['Z_0']
	Beta=gw_params['Beta']
	s_a, s_b, s_c, s_d = [gw_params[k] for k in ['s_a', 's_b', 's_c', 's_d']]
	be_a, be_b, be_c, be_d = [gw_params[k] for k in ['be_a', 'be_b', 'be_c', 'be_d']]

	spline = gal_params['spline']
	bg0 = gal_params['bg0']
	bg1 = gal_params['bg1']
	bg2 = gal_params['bg2']
	bg3 = gal_params['bg3']
	sg0 = gal_params['sg0']
	sg1= gal_params['sg1']
	sg2= gal_params['sg2']
	sg3= gal_params['sg3']
	sig_gal = gal_params.get('sig_gal', None)  # may not exist for all detectors

	# -----------------------------------------------------------------------------------------
	# 	DEFINE FIDUCIAL COSMOLOGICAL MODEL AND COMPUTE CORRESPONDING LUMINOSITY DISTANCES
	# -----------------------------------------------------------------------------------------
	# Luminosity distance interval, equal to the redshift one assuming fiducial cosmology
	print("\nComputing the cosmology ...")
	fiducial_universe = FlatLambdaCDM(H0=H0_true, Om0=Omega_m_true, Ob0=Omega_b_true)

	dlm_bin = fiducial_universe.luminosity_distance(z_m_bin_GW).value  # Minimum luminosity distance from fiducial model
	dlM_bin = fiducial_universe.luminosity_distance(z_M_bin_GW).value  # Maximum luminosity distance from fiducial model

	z_gal = np.linspace(z_m, z_M, 2000)  # Redshift grid for galaxy distribution
	dl_GW = np.linspace(dlm, dlM, 2000)  # Luminosity distance grid for gravitational wave sources in Mpc

	# -----------------------------------------------------------------------------------------
	#							BIN STRATEGY
	# -----------------------------------------------------------------------------------------
	bin_int = np.linspace(z_m_bin, z_M_bin, n_bins_z * 1000)  # Fine redshift grid for binning
	bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl * 1000)  # Fine luminosity distance grid for GW binning (in Gpc)

	# Compute bin edges using the specified strategy and cosmology
	bin_edges, bin_edges_dl = fem.compute_bin_edges(bin_strategy, n_bins_dl, n_bins_z, bin_int, z_M_bin, dlM_bin, z_m_bin, fiducial_universe, A, Z_0, Alpha, Beta, spline)

	# Convert luminosity distance bin edges to redshift using the fiducial cosmology
	bin_z_fiducial = (bin_edges_dl * u.Gpc).to(cu.redshift,cu.redshift_distance(fiducial_universe, kind="luminosity")).value

	# Compute redshift distribution and total number of galaxies
	nz_gal, gal_tot = fem.compute_nz_gal_and_total(gal_det, z_gal, bin_edges, sig_gal, gal_params['spline'])

	gal_tot[gal_tot < 0] = 0  # Remove negative values (if any)
	n_tot_gal = trapezoid(gal_tot, z_gal)  # Integrate total galaxy distribution

	# Compute fraction of galaxies in each redshift bin
	bin_frac_gal = np.zeros(shape=(n_bins_z))
	for i in range(n_bins_z):
		bin_frac_gal[i] = simpson(nz_gal[i], z_gal)

	shot_noise_gal = 1 / bin_frac_gal
	n_gal_bins = np.sum(bin_frac_gal)  # Sum of galaxy fractions across bins

	# Save bin edges for later use

	np.save(os.path.join(FLAGS.fout, 'bin_edges_gal_fiducial.npy'), bin_z_fiducial)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW.npy'), bin_edges_dl)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_gal.npy'), bin_edges)
	np.save(os.path.join(FLAGS.fout, 'nz_gal.npy'), nz_gal)

	# -----------------------------------------------------------------------------------------
	#           PLOTTING AND SAVING THE GALAXY BIN DISTRIBUTION AND INFORMATION
	# -----------------------------------------------------------------------------------------
	#fem.plot_galaxy_bin_distributions(z_gal, nz_gal, gal_tot, bin_edges, n_bins_z, z_m_bin, z_M_bin, FLAGS.fout)

	# Print statistics about galaxy bins
	#print('\nthe total number of galaxies across all redshift: ', n_tot_gal * 4 * np.pi * f_sky)
	#print('the total number of galaxies in our bins: ', n_gal_bins * 4 * np.pi * f_sky)
	#print('mean number of galaxies in each bin: ', np.mean(bin_frac_gal))
	#print('mean shot noise in each bin: ', np.mean(shot_noise_gal))

	'''
	#with open(os.path.join(FLAGS.fout, "galaxy_bin_distributions.txt"), "w") as f:
		f.write("Diagnostics for this run: z_gal,nz_gal, gal_tot, bin_frac_gal, shot_noise_gal, n_bins_z, z_m_bin, z_M_bin and info galaxies and shot noise \n\n")
		f.write("z_gal =" + str(z_gal.tolist()) + "\n\n")
		f.write("nz_gal =" + str(nz_gal.tolist()) + "\n\n")
		f.write("gal_tot = " + str(gal_tot.tolist()) + "\n\n")
		f.write("bin_frac_gal = " + str(bin_frac_gal.tolist()) + "\n\n")
		f.write("shot_noise_gal = " + str(shot_noise_gal.tolist()) + "\n\n")
		f.write("n_bins_z         = " + str(n_bins_z) + "\n\n")
		f.write("z_m_bin            = " + str(z_m_bin) + "\n\n")
		f.write("z_M_bin     = " + str(z_M_bin) + "\n\n")
		f.write("the total number of galaxies across all redshift  = " + str(n_tot_gal * 4 * np.pi * f_sky) + "\n\n")
		f.write("the total number of galaxies in our bins     = " + str(n_gal_bins * 4 * np.pi * f_sky) + "\n\n")
		f.write("mean number of galaxies in each bin     = " + str(np.mean(bin_frac_gal)) + "\n\n")
		f.write("mean shot noise in each bin    = " + str(np.mean(shot_noise_gal)) + "\n\n")

	print("\nDiagnostics GALAXY BIN DISTRIBUTION saved!\n")
	'''
	# -----------------------------------------------------------------------------------------
	# 		DETERMINE REPRESENTATIVE REDSHIFTS AND COMPUTE NONLINEAR POWER SPECTRUM
	# -----------------------------------------------------------------------------------------
	print('\nComputing power spectrum...\n')
	# Initialize array to store the peak redshift of each galaxy bin
	redshift = np.zeros(shape=n_bins_z)
	for i in range(n_bins_z):
		a = np.argmax(nz_gal[i])  # Index of maximum value in the redshift distribution
		redshift[i] = z_gal[a]  # Assign corresponding redshift

	# Define k and z arrays for evaluating the nonlinear power spectrum
	kk_nl_input = np.geomspace(1e-4, 10, 200)  # Logarithmically spaced k values [h/Mpc]
	zz_nl_input = np.linspace(z_m_bin_GW, z_M_bin_GW, 100)  # Linearly spaced redshift values

	# Compute nonlinear matter power spectrum using HI_CLASS
	Hi_Cosmo = cc.cosmo(**cosmo_params)

	# Compute nonlinear matter power spectrum using HI_CLASS
	bg, kk_nl, zz_nl, P_vals = Hi_Cosmo.hi_class_pk(kk_nl_input, zz_nl_input,True) # kk_nl in [1/Mpc]

	#print('\nSaving the background...')
	np.save(os.path.join(FLAGS.fout, 'background.npy'), bg)

	# Interpolate power spectrum over redshift and k
	P_interp = RectBivariateSpline(kk_nl, zz_nl, P_vals)

	# Use peak redshifts of bins as centers for computing k_max
	z_centers_use = redshift

	# Compute maximum usable wavenumber at each redshift bin center
	k_max = fem.compute_k_max(z_centers_use, P_interp, kk_nl)    # [1/Mpc]

	#print('K_max',k_max)

	# -----------------------------------------------------------------------------------------
	#			COMPUTING MULTIPOLE LIMITS AND GW BIN DISTRIBUTION STATISTICS
	# -----------------------------------------------------------------------------------------
	# Compute maximum multipole l for each bin using comoving distance and k_max
	l_max_nl = np.asarray([fiducial_universe.comoving_distance(z_centers_use[i]).value * k_ for i, k_ in enumerate(k_max)]).astype(int)
	#print('l_max_nl',l_max_nl)

	# Compute localization error parameters for GW bins
	sigma_sn_GW, l_max_loc = fcc.loc_error_param(bin_edges_dl, log_loc, log_dl, l_min, 10000)

	# Determine lengths of arrays
	n = len(l_max_nl)
	m = len(l_max_loc)

	# Extend l_max_nl to match length of l_max_loc
	l_max_nl_ = np.concatenate((l_max_nl, l_max_loc[-(m - n):]))

	# Compute final l_max per bin as the minimum between localization and nonlinear limits
	l_max_bin = np.minimum(l_max_loc, l_max_nl_)

	# Determine whether the limiting factor is localization (0) or nonlinear scale (1)
	loc_or_nl = np.where(l_max_loc <= l_max_nl_, 0, 1)

	# Compute overall maximum multipole
	l_max = np.max(l_max_nl_)

	print('l_max',l_max)

	# Define multipole vector with increasing step sizes at higher l
	ll = np.sort(np.unique(np.concatenate([
		np.arange(l_min, 20, step=2),
		np.arange(20, 50, step=5),
		np.arange(50, 100, step=10),
		np.arange(100, l_max + 1, step=25)])))
	#print('ll', ll)
	ll[-1] = l_max  # Ensure maximum l is included
	ll_total = np.arange(l_min, l_max + 1)

	#print('ll',ll)
	#print('ll_total',ll_total)

	# Compute normalization factor for Cl's
	c = ll * (ll + 1.) / (2. * np.pi)

	# Save computed arrays
	np.save(os.path.join(FLAGS.fout, 'l_max'), l_max)
	np.save(os.path.join(FLAGS.fout, 'll'), ll)
	np.save(os.path.join(FLAGS.fout, 'll_total'), ll_total)

	#np.save(os.path.join(FLAGS.fout, 'ell_max.npy'), l_max_bin)
	#np.save(os.path.join(FLAGS.fout, 'loc_nl.npy'), loc_or_nl)

	# -----------------------------------------------------------------------------------------
	#            COMPUTING AND PLOTTING GW BIN DISTRIBUTION STATISTICS
	# -----------------------------------------------------------------------------------------
	# Compute the merger rate distribution and related quantities from luminosity distance bins
	z_GW, bin_convert, ndl_GW, n_GW, merger_rate_tot = fcc.merger_rate_dl( #_new
		dl=dl_GW, #Mpc
		bin_dl=bin_edges_dl,
		log_dl=log_dl,
		log_delta_dl=log_delta_dl,
		H0=H0_true,
		omega_m=Omega_m_true,
		omega_b=Omega_b_true,
		A=A,
		Z_0=Z_0,
		Alpha=Alpha,
		Beta=Beta,
		normalize=False
	)

	np.save(os.path.join(FLAGS.fout, 'z_GW'), z_GW) #
	#np.save(os.path.join(FLAGS.fout, 'ndl_GW'), ndl_GW)
	#np.save(os.path.join(FLAGS.fout, 'n_GW'), n_GW)

	# Integrate the total merger rate over the full luminosity distance range (in Gpc)
	n_tot_GW = trapezoid(merger_rate_tot, dl_GW / 1000) * 4 * np.pi   # Mpc/1000 = Gpc
	#print('\nthe total number of GW across all distance: ', n_tot_GW)

	# Calculate the fraction of GW sources in each luminosity distance bin
	bin_frac_GW = np.zeros(shape=n_bins_dl)
	for i in range(n_bins_dl):
		bin_frac_GW[i] = trapezoid(ndl_GW[i], dl_GW / 1000)

	# Sum all bin fractions to get the total number in bins (should match total GW if complete)
	n_GW_bins = np.sum(bin_frac_GW)
	#print('the total number of GW in our bins: ', n_GW_bins * 4 * np.pi)

	#fem.plot_gw_bin_distributions(	dl_GW=dl_GW,ndl_GW=ndl_GW,merger_rate_tot=merger_rate_tot,bin_edges_dl=bin_edges_dl,n_bins_dl=n_bins_dl,output_path=FLAGS.fout)

	# Print per-bin and mean statistics for GW shot noise
	#print('\nfraction of GW per sterad in each bin', bin_frac_GW)
	shot_noise_GW = 1 / bin_frac_GW
	#print('shot noise per bin', shot_noise_GW)
	#print('mean number of GW in each bin: ', np.mean(bin_frac_GW))
	#print('mean shot noise in each bin: ', np.mean(shot_noise_GW))

	#-----------------------------------------------------------------------------------------
    #        FIGURES FOR COMPARING GALAXY AND GW DISTRIBUTIONS
	#-----------------------------------------------------------------------------------------
	'''
	fem.plot_distribution_comparison(
		z_gal, nz_gal, gal_tot, bin_edges, n_bins_z, z_m_bin, z_M_bin,
		z_GW, ndl_GW, merger_rate_tot, bin_convert, n_bins_dl, FLAGS.fout)
	'''

	#-----------------------------------------------------------------------------------------
    #					COMPUTING MEAN REDSHIFT FOR FIDUCIAL BIASES
	#-----------------------------------------------------------------------------------------
	# Compute mean redshift for each GW bin and corresponding GW bias
	z_mean_GW = (bin_z_fiducial[:-1] + bin_z_fiducial[1:]) * 0.5
	bias_GW = A_GW * (1. + z_mean_GW) ** gamma_GW

	# Compute mean redshift for each galaxy bin and galaxy bias using polynomial model
	z_mean_gal = (bin_edges[:-1] + bin_edges[1:]) * 0.5
	bias_gal =A_gal* (1. + z_mean_gal) ** gamma_gal

	# Compute magnification slope s(z) depending on galaxy detector
	s_gal= fem.compute_s_gal(z_gal, gal_det, sg0, sg1, sg2, sg3)

	np.save(os.path.join(FLAGS.fout, 'bias_fiducial_gal'), bias_gal)
	np.save(os.path.join(FLAGS.fout, 'bias_fiducial_GW'), bias_GW)
	np.save(os.path.join(FLAGS.fout, 'z_mean_gal'), z_mean_gal)
	np.save(os.path.join(FLAGS.fout, 'z_mean_GW'), z_mean_GW)

	'''
	#with open(os.path.join(FLAGS.fout, "fisher_info_diagnostics.txt"), "w") as f:
		f.write("Diagnostics for this run: dl_GW, z_centers_use, k_max, ll, l_max_bin, ll_total, l_max_nl, z_mean_GW, z_mean_gal \n\n")
		f.write("dl_GW ="+str(dl_GW.tolist())+"\n\n")
		f.write("z_centers_use = " + str(z_centers_use.tolist()) + "\n\n")
		f.write("k_max         = " + str(k_max.tolist()) + "\n\n")
		f.write("ll            = " + str(ll.tolist()) + "\n\n")
		f.write("l_max_bin     = " + str(l_max_bin.tolist()) + "\n\n")
		f.write("ll_total      = " + str(ll_total.tolist()) + "\n\n")
		f.write("l_max_nl      = " + str(l_max_nl.tolist()) + "\n\n")
		f.write("z_mean_GW     = " + str(z_mean_GW.tolist()) + "\n\n")
		f.write("z_mean_gal    = " + str(z_mean_gal.tolist()) + "\n\n")
	print("\nDiagnostics for FISHER saved!")
	'''

	#-----------------------------------------------------------------------------------------
	#				COMPUTING LOCALIZATION NOISE MATRICES
	#-----------------------------------------------------------------------------------------
	print('\nComputing localization noise matrices...')

	noise_gal = fcc.shot_noise_mat_auto(shot_noise_gal, ll_total)
	noise_GW = fcc.shot_noise_mat_auto(shot_noise_GW, ll_total)

	noise_loc = np.zeros(shape=(n_bins_dl, len(ll_total)))
	noise_loc_auto = np.zeros(shape=(n_bins_dl, len(ll_total)))

	for i in range(n_bins_dl):
		for l in range(len(ll_total)):
			if (ll_total[l] * (ll_total[l] + 1) * (sigma_sn_GW[i] / (2 * np.pi) ** (3 / 2))) < 30:
				noise_loc[i, l] = np.exp(-ll_total[l] * (ll_total[l] + 1) * (sigma_sn_GW[i] / (2 * np.pi) ** (3 / 2)))
				noise_loc_auto[i, l] = np.exp(-2 * ll_total[l] * (ll_total[l] + 1) * (sigma_sn_GW[i] / (2 * np.pi) ** (3 / 2)))
			else:
				noise_loc[i, l] = np.exp(-30)
				noise_loc_auto[i, l] = np.exp(-30)

	noise_loc_mat = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))
	noise_loc_mat_auto = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))

	for i in range(n_bins_z):
		for ii in range(n_bins_dl):
			noise_loc_mat[i, ii, :] = noise_loc[ii, :]

	for i in range(n_bins_dl):
		for ii in range(i, n_bins_dl):
			noise_loc_mat_auto[i, ii, :] = noise_loc_auto[ii, :]

	for i in range(n_bins_dl):
		for ii in range(i + 1, n_bins_dl):
			noise_loc_mat_auto[ii, i] = noise_loc_mat_auto[i, ii]

	np.save(os.path.join(FLAGS.fout, 'noise_GW'), noise_GW)
	np.save(os.path.join(FLAGS.fout, 'noise_gal'), noise_gal)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_auto'), noise_loc_mat_auto)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_cross'), noise_loc_mat)

	# -----------------------------------------------------------------------------------------
	#       DEFINITION OF Cl_func DEPENDING ON THE PRESENCE OF THE LENSING
	# -----------------------------------------------------------------------------------------
	# If lensing is included in the analysis
	if Lensing:
		# Define function to compute Cl including lensing, clustering, and RSD contributions
		def Cl_func(Hi_Cosmo, params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal, b_GW, save, n_points=13,
					n_points_x=20, grid_x='lin', z_min=1e-05, n_low=5, n_high=5):

			# Define cosmology and initialize Limber integrator
			# print('params contains:',params)
			S = LLG.limber(cosmology=Hi_Cosmo, z_limits=[z_m, z_M])

			# Define k and z grids for power spectrum
			kk = np.geomspace(1e-4, 10, 200)
			zz = np.linspace(0, z_M, 101)

			# Compute nonlinear matter power spectrum with HI_CLASS
			bg, _, _, pkz = Hi_Cosmo.hi_class_pk(kk, zz, True)

			# np.save(os.path.join(FLAGS.fout, 'background'),bg )

			S.load_power_spectra(k=kk, z=zz, power_spectra=pkz)

			# Generate GW distribution from fiducial parameters
			A = gw_params['A']
			Alpha = gw_params['Alpha']
			log_delta_dl = gw_params['log_delta_dl']
			log_dl = gw_params['log_dl']
			Z_0 = gw_params['Z_0']
			Beta = gw_params['Beta']

			h = params['h']
			H_0 = params['h'] * 100
			Omega_m = params['Omega_m']
			Omega_b = params['Omega_b']

			# z_GW no dim; bin_GW_converted no dim; ndl_GW 1/Gpc; n_GW [Gpc]; total no dim
			z_GW, bin_GW_converted, ndl_GW, n_GW, total = fcc.merger_rate_dl_new(
				alpha_M_interp=alpha_M_interp,
				dl=dl_GW,
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
			# Load bin edges for all observables
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='lensing_gal', name_2='lensing_GW')
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='galaxy', name_2='GW')
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='rsd', name_2='lsd')

			# Compute galaxy magnification slope parameter beta
			s_a, s_b, s_c, s_d = [gw_params[k] for k in ['s_a', 's_b', 's_c', 's_d']]
			be_a, be_b, be_c, be_d = [gw_params[k] for k in ['be_a', 'be_b', 'be_c', 'be_d']]

			print('QUI BETA HA FlatLambdaCDM')
			beta = fem.compute_beta(H_0, Omega_m, Omega_b, z_gal, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d)

			# beta_new =fem.compute_beta_new(Hi_Cosmo,z_gal,Omega_m, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d)
			# print('Relative dif % beta', ((beta_new/beta)-1 )*100)

			print('\nLoading the window functions...\n')
			# Load window functions for each observable
			S.load_galaxy_clustering_window_functions(bg, h, z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal,
													  name='galaxy')  ### OK
			S.load_gravitational_wave_window_functions(bg, C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, bias=b_GW,
													   name='GW')  ### OK

			S.load_rsd_window_functions(bg, h, z=z_gal, n_z=nz_gal, ll=ll, name='rsd')  ### OK
			S.load_lsd_window_functions(bg, h, C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, name='lsd')  ### OK

			S.load_galaxy_lensing_window_functions(z=z_gal, n_z=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll,
												   name='lensing_gal')  ### in units Mpc   OK
			S.load_gw_lensing_window_functions(bg=bg, C=Hi_Cosmo, z=z_GW, h=h, n_dl=ndl_GW, H_0=H_0, omega_m=Omega_m,
											   ll=ll, name='lensing_GW')  ### OK

			print('\nComputing the angular power spectra...\n')
			# Compute all angular power spectra using Limber integrals

			Cl = S.limber_angular_power_spectra(bg, h=h, l=ll, windows=['galaxy', 'GW', 'rsd', 'lsd'])

			print('\nComputing the angular power spectra cross-correlation...\n')

			Cl_lens_cross = S.limber_angular_power_spectra_lensing_cross(bg, l=ll, s_gal=s_gal, beta=beta,
																		 windows=None, n_points=n_points,
																		 n_points_x=n_points_x,
																		 z_min=z_min, grid_x=grid_x,
																		 n_low=n_low,
																		 n_high=n_high)

			print('\nComputing the angular power spectra autocorrelation...\n')
			Cl_lens = S.limber_angular_power_spectra_lensing_auto(bg,
																  l=ll, s_gal=s_gal, beta=beta,
																  windows=['lensing_gal', 'lensing_GW'],
																  n_points=n_points, n_points_x=n_points_x,
																  z_min=z_min, grid_x=grid_x, n_low=n_low,
																  n_high=n_high)

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

			if save:
				print('\nSaving all the Cl results...\n')
				# Galaxy-Galaxy
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GG'), Cl_delta_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_len_GG'), Cl_len_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_GG'), Cl_RSD_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_len_GG'), Cl_delta_len_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_RSD_GG'), Cl_delta_RSD_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_len_GG'), Cl_RSD_len_GG)

				# GW-GW
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GWGW'), Cl_delta_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_len_GWGW'), Cl_len_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_GWGW'), Cl_RSD_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_len_GWGW'), Cl_delta_len_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_RSD_GWGW'), Cl_delta_RSD_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_len_GWGW'), Cl_RSD_len_GWGW)

				# Galaxy-GW
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GGW'), Cl_delta_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_len_GGW'), Cl_len_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_GGW'), Cl_RSD_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_len_GGW'), Cl_delta_len_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_RSD_GGW'), Cl_delta_RSD_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_len_GGW'), Cl_RSD_len_GGW)

			return Cl_GG, Cl_GWGW, Cl_GGW

	# If lensing is not included, compute only density clustering spectra
	else:
		# Define function to compute Cl from galaxy and GW clustering only
		def Cl_func(Hi_Cosmo, params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal, b_GW, save):

			S = LLG.limber(cosmology=Hi_Cosmo, z_limits=[z_m, z_M])

			# Define power spectrum grids
			kk = np.geomspace(1e-4, 10, 200)
			zz = np.linspace(0, z_M, 100)

			# t0 = time.time()
			# Compute nonlinear matter power spectrum
			bg, _, _, pkz = Hi_Cosmo.hi_class_pk(kk, zz, True)  # pkz (Mpc/1)^3
			# t1 = time.time()
			# print(f"\t Time hi_class_pk: {t1 - t0:.4f} s")

			if bg == 0 and pkz == 0:
				return np.array(0), np.array(0), np.array(0)
			else:
				# t0 = time.time()
				S.load_power_spectra(z=zz, k=kk, power_spectra=pkz)  # 0.05
				# t1 = time.time()
				# print(f"\t\t\tTime load_power_spectra: {t1 - t0:.4f} LOAD POWER SPECTRUM s")

				# Generate GW distribution from fiducial parameters
				A = gw_params['A']
				Alpha = gw_params['Alpha']
				log_delta_dl = gw_params['log_delta_dl']
				log_dl = gw_params['log_dl']
				Z_0 = gw_params['Z_0']
				Beta = gw_params['Beta']

				h = params['h']

				z_bg = np.asarray(bg['z'])
				H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False,
									fill_value="extrapolate")  # [1/Mpc]
				chi_interp = interp1d(z_bg, bg['comov. dist.'], kind='cubic', bounds_error=False,
									  fill_value="extrapolate")  # Mpc
				alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False,
										  fill_value="extrapolate")

				# t0 = time.time()
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
				# t1 = time.time()
				# print(f"\t\t\tTime merger_rate_dl_new: {t1 - t0:.4f}  BIN + OTHER GWs")

				print('Loading the window functions...')

				# t0 = time.time()
				# Load binning and window functions ----> 0.3
				S.load_bin_edges(bin_edges, bin_GW_converted)
				S.load_galaxy_clustering_window_functions(H_interp, z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal,
														  name='galaxy')
				S.load_gravitational_wave_window_functions(H_interp, chi_interp, alpha_M_interp, C=Hi_Cosmo, z=z_GW,
														   n_dl=ndl_GW, ll=ll, bias=b_GW, name='GW')
				# t1 = time.time()
				# print(f"\t\t\tTime loading windows: {t1 - t0:.4f} LOADING STUFF")

				print('Computing the angular power spectra...')
				# Compute angular power spectra (density terms only)

				ll = np.sort(np.unique(np.concatenate(
					[np.arange(l_min, 20, step=2), np.arange(20, 50, step=5), np.arange(50, 100, step=10),
					 np.arange(100, l_max + 1, step=25)])))

				# print('ll used',ll)
				# start = time.time()
				Cl = S.limber_angular_power_spectra(H_interp, chi_interp, h, l=ll, windows=None)
				# end = time.time()
				# print(f"\t\t\tTime took {end - start:.4f} seconds for LIMBER\n")

				# Galaxy-GW
				Cl_delta_GGW = Cl['galaxy-GW']

				# Galaxy-Galaxy
				Cl_delta_GG = Cl['galaxy-galaxy']

				# GW-GW
				Cl_delta_GWGW = Cl['GW-GW']

				if save:
					print('\nSaving all the Cl results...')
					np.save(os.path.join(FLAGS.fout, 'Cl_delta_GG'), Cl_delta_GG)
					np.save(os.path.join(FLAGS.fout, 'Cl_delta_GWGW'), Cl_delta_GWGW)
					np.save(os.path.join(FLAGS.fout, 'Cl_delta_GGW'), Cl_delta_GGW)
				return Cl_delta_GG, Cl_delta_GWGW, Cl_delta_GGW

	# -----------------------------------------------------------------------------------------
	#					COMPUTING THE POWER SPECTRUM
	# -----------------------------------------------------------------------------------------
	# Print status message for power spectrum computation
	print('\nComputing the Power Spectrum...')

	# Compute angular power spectra from Cl_func with fiducial cosmological and bias parameters
	Cl_GG, Cl_GWGW, Cl_GGW = Cl_func(Hi_Cosmo, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal,b_GW=bias_GW, save=True)

	Cl_GG_total = np.zeros(shape=(n_bins_z, n_bins_z, len(ll_total)))
	Cl_GWGW_total = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))
	Cl_GGW_total = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))

	for i in range(n_bins_z):
		for ii in range(n_bins_z):
			Cl_GG_interp = si.interp1d(ll, Cl_GG[i, ii])
			Cl_GG_total[i, ii] = Cl_GG_interp(ll_total)

	for i in range(n_bins_dl):
		for ii in range(n_bins_dl):
			Cl_GWGW_interp = si.interp1d(ll, Cl_GWGW[i, ii])
			Cl_GWGW_total[i, ii] = Cl_GWGW_interp(ll_total)

	for i in range(n_bins_z):
		for ii in range(n_bins_dl):
			Cl_GGW_interp = si.interp1d(ll, Cl_GGW[i, ii])
			Cl_GGW_total[i, ii] = Cl_GGW_interp(ll_total)

	Cl_GWGW_total = Cl_GWGW_total * noise_loc_mat_auto
	Cl_GGW_total = Cl_GGW_total * noise_loc_mat

	Cl_GWGW_total += noise_GW
	Cl_GG_total += noise_gal

	Cl_GG_total_with_noise = Cl_GG_total
	Cl_GWGW_total_with_noise = Cl_GWGW_total
	Cl_GGW_total_with_noise = Cl_GGW_total

	# -----------------------------------------------------------------------------------------
	#					COMPUTING FIDUCIAL VECTOR AND FIDUCIAL COVARIANCE MATRIX
	# -----------------------------------------------------------------------------------------
	print('\nComputing fiducial vector for covariance matrix...')
	# BE CAREFUL FOR THE ORDER OF THE CLs
	# vec_fid,vec_dict,vec_idx=LH_fun.build_vector(n_bins_z,n_bins_dl,Cl_GGW,Cl_GWGW,Cl_GG) # NO NOISE
	vec_fid, vec_dict,vec_idx = LH_fun.build_vector(n_bins_z, n_bins_dl, Cl_GGW_total_with_noise,Cl_GWGW_total_with_noise, Cl_GG_total_with_noise)  # NOISE

	print('\nComputing fiducial covariance matrix...')
	_, cov_inv = LH_fun.build_cov_and_inv(n_bins_z, n_bins_dl, vec_dict, vec_idx, ll_total)

	# FOR EACH L SHOULD BE SYMMETRIC
	for ell in range(cov_inv.shape[2]):
		A = cov_inv[:, :, ell]
		A_T = cov_inv[:, :, ell].T
	print("Check symmetric:",np.max(A-A_T))

	# Save the fiducial covariance matrix to file and its inverse
	np.save(os.path.join(FLAGS.fout, 'vec_fid'), vec_fid)
	# Save the fiducial covariance matrix to file and its inverse
	np.save(os.path.join(FLAGS.fout, 'cov_mat_inverse'), cov_inv)

	print('''
		  ________  ________   _______   ______ 
		 /_  __/ / / / ____/  / ____/ | / / __ \\
		  / / / /_/ / __/    / __/ /  |/ / / / /
		 / / / __  / /___   / /___/ /|  / /_/ / 
		/_/ /_/ /_/_____/  /_____/_/ |_/_____/  
		''')
