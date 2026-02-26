# bg= background,

#-----------------------------------------------------------------------------------------
# ANGULAR SPECTRA
#-----------------------------------------------------------------------------------------
def limber_angular_power_spectra(self,bg, h, l, windows = None):
    """
    This function computes the angular power spectra (using the Limber's and the flat-sky approximations) for the window function specified.
    Given two redshift bins `i` and `j` the equation is

    .. math::

      C^{(ij)}(\ell) = \int_0^\infty dz \ \\frac{1}{H(z)} \ \\frac{W^{(i)}(z) W^{(j)}(z)}{f_K^2[\chi(z)]} \ P\left(\\frac{\ell}{f_K[\chi(z)]}, z\\right),

    where :math:`P(k,z)` is the matter power spectrum and :math:`W^{(i)}(z)` are the window functions.

    :param l: Multipoles at which to compute the shear power spectra.
    :type l: array

    :param windows: which spectra (auto and cross) must be computed. If set to ``None`` all the spectra will be computed.
    :type windows: list of strings, default = ``None``

    :return: dictionary whose keys are combinations of window functions specified in ``windows``. Each key is a 3-D array whose entries are ``Cl[bin i, bin j, multipole l]``
    """

    def geometric_factor_bg(bg, z, z_bg, h, K_h=0.0, z0=0.0):
        """
        Return f_K(chi[z]-chi[z0]) in units of Mpc/h.
        bg['comov. dist.'] is assumed in Mpc (flat comoving distance).
        K_h must be in (h/Mpc)^2.
        """
        chi_interp = interp1d(z_bg, bg['comov. dist.'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        chi_Mpc = chi_interp(z) - chi_interp(z0)  # Mpc
        chi_h = h * chi_Mpc  # Mpc/h

        if np.allclose(K_h, 0.0):
            return chi_h  # flat: f_K(chi)=chi  [Mpc/h]

        if np.any(K_h > 0):
            # closed: f_K = sin(sqrt(K)*chi)/sqrt(K)
            return np.sin(np.sqrt(K_h) * chi_h) / np.sqrt(K_h)  # Mpc/h
        else:
            # open:   f_K = sinh(sqrt(|K|)*chi)/sqrt(|K|)
            Kabs = np.abs(K_h)
            return np.sinh(np.sqrt(Kabs) * chi_h) / np.sqrt(Kabs)  # Mpc/h

    # 1) Define lengths and quantities
    zz = self.z_integration
    n_l = len(np.atleast_1d(l))
    n_z = self.n_z_integration

    ################################################
    # the previous term was: cH_chi2  = self.c_over_H_over_chi_squared where the curvature is considered
    # Factor c/H(z)/f_K(z)^2
    # self.c_over_H_over_chi_squared = const.c/self.Hubble/self.geometric_factor**2

    # --- bg interpolators
    z_bg = np.asarray(bg['z'])
    H_interp = interp1d(z_bg, bg['H [1/Mpc]']/h, kind='cubic', bounds_error=False, fill_value="extrapolate") #[h/Mpc]

    # set the right range
    H_interp_z_int = H_interp(zz)  # h/Mpc
    H_inverse = (1 / H_interp_z_int) * (1 / (geometric_factor_bg(bg, zz, z_bg,h) ** 2)) #[Mpc/h]*[h^2/Mpc^2] = [h/Mpc]

    # Check existence of power spectrum
    try: self.power_spectra_interpolator
    except AttributeError: raise AttributeError("Power spectra have not been loaded yet")

    # Check convergence with (l, k, z):
    assert np.atleast_1d(l).min() > self.k_min*h*geometric_factor_bg(bg, self.z_min, z_bg,h), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
    assert np.atleast_1d(l).max() < self.k_max*h*geometric_factor_bg(bg, self.z_max, z_bg,h), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

    # Check window functions to use
    if windows is None:
        windows_to_use = self.window_function
    else:
        windows_to_use = {}
        for key in windows:
            try:             windows_to_use[key] = self.window_function[key]
            except KeyError: raise KeyError("Requested window function '%s' not known" %key)

    # Check window functions
    keys   = windows_to_use.keys()
    if len(keys) == 0: raise AttributeError("No window function has been computed!")
    n_keys  = len(keys)
    n_bins = [len(windows_to_use[key]) for key in keys]

    Cl = {}
    for i,ki in enumerate(keys):
        for j,kj in enumerate(keys):
            Cl['%s-%s' %(ki,kj)] = np.zeros((n_bins[i],n_bins[j], n_l))

    # distance in Mpc/h
    geom_int = geometric_factor_bg(bg, zz, z_bg,h, z0=0.0)  # shape: (n_z,)

    # 2) Load power spectra: use geom_int instead of self.geometric_factor
    power_spectra = self.power_spectra_interpolator
    PS_lz = np.zeros((n_l, n_z))
    for il in xrange(n_l):
        for iz in range(n_z):
            k_hMpc = (l[il] + 0.5) / geom_int[iz]  # k in h/Mpc
            PS_lz[il, iz] = power_spectra([(k_hMpc, zz[iz])]) #PS_lz in (Mpc/h)^3

    # --- curvature correction ---
    if self.cosmology.K != 0.:
        KK = self.cosmology.K  # should be in (h/Mpc)^2 to match geom_int in Mpc/h
        factor = np.zeros((n_l, n_z))
        for il, ell in enumerate(l):
            # note: ((ell+0.5)/geom_int)**2 broadcasts over z
            factor[il] = (1 - np.sign(KK) * ell ** 2 / (((ell + 0.5) / geom_int) ** 2 + KK)) ** -0.5
        PS_lz *= factor

    #print('\n LIMBER NEW')
    #print('zz', zz)
    #print('n_z', n_z)
    #print('n_l', n_l)
    #print('cH_chi2', H_inverse.tolist())
    #print('Cl', Cl)
    #print('PS_lz', PS_lz.tolist())

    # 3) load Cls given the source functions
    # 1st key (from 1 to N_keys)
    for index_X in xrange(n_keys):
        key_X = list(keys)[index_X]
        W_X = np.array([[windows_to_use[key_X][i,j](zz) for j in range(n_l)] for i in range(n_bins[index_X])])
        # 2nd key (from 1st key to N_keys)
        for index_Y in xrange(index_X,n_keys):
            key_Y = list(keys)[index_Y]
            W_Y = np.array([[windows_to_use[key_Y][i,j](zz) for j in range(n_l)] for i in range(n_bins[index_Y])])
            # Symmetry C_{AA}^{ij} == C_{AA}^{ji}
            if key_X == key_Y:
                for bin_i in xrange(n_bins[index_X]):
                    for bin_j in xrange(bin_i, n_bins[index_Y]):
                        Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j] = [simpson(H_inverse*W_X[bin_i,xx]*W_Y[bin_j,xx]*PS_lz[xx], x = zz) for xx in range(n_l)]
                        Cl['%s-%s' %(key_X,key_Y)][bin_j,bin_i] = Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j]
            # Symmetry C_{AB}^{ij} == C_{BA}^{ji}
            else:
                for bin_i in xrange(n_bins[index_X]):
                    for bin_j in xrange(n_bins[index_Y]):
                        #                                                    [h/Mpc] *  [h/Mpc] * [h/Mpc] * (Mpc/h)^3
                        Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j] = [simpson(H_inverse*W_X[bin_i,xx]*W_Y[bin_j,xx]*PS_lz[xx], x = zz) for xx in range(n_l)]
                        Cl['%s-%s' %(key_Y,key_X)][bin_j,bin_i] = Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j]

    return Cl