
#-----------------------------------------------------------------------------------------
# GALAXY CLUSTERING WINDOW FUNCTION
#-----------------------------------------------------------------------------------------
def load_galaxy_clustering_window_functions(self,bg, z, n_z, ll, bias = 1.0, name = 'galaxy'):
    """
    This function computes the window function for galaxy clustering given a galaxy distribution.
    The function automatically normalizes the galaxy distribution such that the integral over
    redshifts is 1.
    The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
    Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

    .. math::

       W^{(i)}_\mathrm{G}(z) = b(z) \ n^{(i)}(z) \\frac{H(z)}{c}


    :param z: array or list of redshift at which the galaxy distribution ``n_z`` is evaluated
    :type z: 1-D array, default = None

    :param n_z: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
    :type n_z: 2-D array with shape ``(n_bins, len(z))``, default = None

    :param bias: Galaxy bias.
    :type bias: float or array, same length of ``n_z``, default = 1

    :param name: name of the key to add to the dictionary
    :type name: string, default = 'galaxy'


    An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``z_min``, ``z_max``:

    .. code-block:: python

       bin_edges = [0.00, 0.72, 1.11, 5.00]
       nbins     = len(bin_edges)-1
       z_w       = np.linspace(0., 6., 1001)
       n_z_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, z_min = bin_edges[i], z_max = bin_edges[i+1]) for i in range(nbins)]
       S.load_galaxy_clustering_window_functions(z = z_w, n_z = n_z_w, bias = 1)

    :return: A key of a given name is added to the ``self.window_function`` dictionary

    """
    ####################### QUA
    print('z \n',len(z))
    z_bg=bg['z']

    print('z_bg \n',len(z_bg))

    H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate")

    print('\nHubble\n',(self.Hubble))
    print('\nHubble\n',(H_interp))

    n_z = np.array(n_z)
    z  = np.array(z)
    n_bins = len(n_z)
    norm_const = simpson(n_z, x = z, axis = 1)

    assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
    assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
    assert (n_z.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
    assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
    assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

    if   isinstance(bias, float): bias = bias*np.ones(n_bins)
    elif isinstance(bias, int)  : bias = np.float(bias)*np.ones(n_bins)
    else:                         assert len(bias)==n_bins, "Number of bias factors different from number of bins"

    # Initialize window
    self.window_function[name] = []

    # Compute window
    for galaxy_bin in xrange(n_bins):
        tmp_interp = si.interp1d(z, n_z[galaxy_bin], kind='cubic',
                                 bounds_error=False, fill_value=0.)

        # evaluate H(z) on the integration grid
        H_on_grid = H_interp(self.z_integration)

        w_vals = (tmp_interp(self.z_integration) * H_on_grid /
                  const.c / norm_const[galaxy_bin] * bias[galaxy_bin])

        self.window_function[name].append(
            si.interp1d(self.z_integration, w_vals,
                        kind='cubic', bounds_error=False, fill_value=0.)
        )

    self.window_function[name] = np.tile(
        np.array(self.window_function[name]).reshape(-1, 1),
        (1, len(ll))
    )

    # Initialize window
    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        tmp_interp = si.interp1d(z,n_z[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
        #to_append = si.interp1d(self.z_integration, tmp_interp(self.z_integration)*self.Hubble/const.c/norm_const[galaxy_bin]*bias[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.)
        self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration)*self.Hubble/const.c/norm_const[galaxy_bin]*bias[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))

        ####################### QUA
        #self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration)*H_interp/const.c/norm_const[galaxy_bin]*bias[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))


    self.window_function[name]=np.tile(np.array(self.window_function[name]).reshape(-1,1), (1,len(ll)))

# -----------------------------------------------------------------------------------------
# GALAXY CLUSTERING WINDOW FUNCTION — compare H from bg vs self.Hubble
# -----------------------------------------------------------------------------------------
def load_galaxy_clustering_window_functions(self, bg, z, n_z, ll, bias=1.0, name='galaxy'):
    """
    Builds galaxy clustering window functions W_G^i(z) = b(z) n^i(z) H(z)/c
    using:
      (a) H from `bg` (interpolated), and
      (b) H from `self.Hubble`.
    Prints per-bin diffs and stores results as:
      self.window_function[name + "_bg"]
      self.window_function[name + "_self"]
    """

    # --- inputs & guards
    z_bg = np.asarray(bg['z'])
    H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic',bounds_error=False, fill_value="extrapolate")

    n_z = np.asarray(n_z)
    z = np.asarray(z)
    n_bins = len(n_z)
    norm_const = simpson(n_z, x=z, axis=1)

    assert np.all(np.diff(z) < self.dz_windows), \
        "For convergence reasons, dz must be <= %.3f" % (self.dz_windows)
    assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
    assert n_z.shape[1] == z.shape[0], "Each n_z[i] must have length len(z)"
    assert z.min() <= self.z_min, "z.min() must be <= z_min integration bound"
    assert z.max() >= self.z_max, "z.max() must be >= z_max integration bound"

    if isinstance(bias, (float, int, np.floating, np.integer)):
        bias = float(bias) * np.ones(n_bins)
    else:
        assert len(bias) == n_bins, "Number of bias factors must equal number of bins"

    # --- Build both versions and compare (unit-consistent)
    z_int = np.asarray(self.z_integration)
    H_bg_on_grid = H_interp(z_int)  # already in 1/Mpc

    # --- Normalize self.Hubble (= H/h) to 1/Mpc
    h = float(self.cosmology.h)

    if np.isscalar(self.Hubble):
        # self.Hubble is H/h (km/s/Mpc)/h  -> multiply by h, then divide by c
        H_self_on_grid = np.full_like(H_bg_on_grid, float(self.Hubble) * h) / const.c
    else:
        H_self_arr = np.asarray(self.Hubble)
        if H_self_arr.shape == z_int.shape:
            H_self_on_grid = (H_self_arr * h) / const.c
        else:
            # If self.Hubble is tabulated on a different z-grid, you need *that* grid to interpolate.
            # Assuming you actually meant it's already on self.z_integration; if not, replace `src_z`
            # with the correct source grid for self.Hubble.
            src_z = z_int  # TODO: set to the true grid of self.Hubble if different
            H_self_on_grid = (np.interp(z_int, src_z, H_self_arr) * h) / const.c

    windows_bg, windows_self = [], []

    print(f"\n[load_galaxy_clustering_window_functions] Comparing H(bg: 1/Mpc) vs H(self)/c on {len(z_int)} z-points")
    for i_bin in range(n_bins):
        nz_i_interp = si.interp1d(z, n_z[i_bin], kind='cubic', bounds_error=False, fill_value=0.0)

        nz_on_grid = nz_i_interp(z_int)
        # NOTE: common factor has NO /c; each branch supplies its own 1/Mpc factor
        common_fac = (nz_on_grid / norm_const[i_bin]) * bias[i_bin]

        w_bg_vals = common_fac * H_bg_on_grid  # (1/Mpc) from bg
        w_self_vals = common_fac * H_self_on_grid  # (1/Mpc) from self/c

        windows_bg.append(si.interp1d(z_int, w_bg_vals, kind='cubic', bounds_error=False, fill_value=0.0))
        windows_self.append(si.interp1d(z_int, w_self_vals, kind='cubic', bounds_error=False, fill_value=0.0))

        # robust diffs (ignore tiny denominators to avoid 1e5 spikes)
        denom = np.maximum(np.maximum(np.abs(w_bg_vals), np.abs(w_self_vals)), 1e-12)
        abs_diff = float(np.max(np.abs(w_bg_vals - w_self_vals)))
        rel_diff = float(np.max(np.abs((w_bg_vals - w_self_vals) / denom)))
        print(f"  bin {i_bin:02d}: max_abs_diff={abs_diff:.3e}, max_rel_diff={rel_diff:.3e}")

    # --- store both versions without overwriting each other
    self.window_function[name + "_bg"] = np.tile(
        np.array(windows_bg).reshape(-1, 1), (1, len(ll))
    )
    self.window_function[name + "_self"] = np.tile(
        np.array(windows_self).reshape(-1, 1), (1, len(ll))
    )

    # (Optional) keep backward-compatibility: choose one as the default 'name'
    # Here we pick the bg version; change to windows_self if you prefer.
    self.window_function[name] = self.window_function[name + "_bg"]

# -----------------------------------------------------------------------------------------
    # GRAVITATIONAL WAVES CLUSTERING WINDOW FUNCTION — compare bg vs self/LCDM
    # -----------------------------------------------------------------------------------------
    def load_gravitational_wave_window_functions(self, bg, z, n_dl, H_0, omega_m, omega_b, ll, bias=1.0, name='GW'):
        """
        Compare GW clustering windows built from:
          (a) bg:   H_bg(z) [1/Mpc] and r_bg(z) [Mpc] from 'bg'
          (b) self: H_self(z) = (self.Hubble * h)/c [1/Mpc] and dL_LCDM(z) from FlatLambdaCDM(H0, Om0, Ob0)
        Window formula per bin i:
            W_i(z) = b_i * n_dl_i(d_L(z)) * [ d d_L/dz ] * H(z) / N_i
          with d d_L/dz = (1+z)/H(z) + r(z), and N_i = ∫ n_dl(d_L) d d_L.
        """

        # --- bg interpolators
        z_bg = np.asarray(bg['z'])
        H_bg_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        chi_bg_interp = interp1d(z_bg, bg['comov. dist.'], kind='cubic', bounds_error=False,
                                 fill_value="extrapolate")

        # --- LCDM converter for the "self" branch normalization (as in your current code)
        conversion = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b)

        # inputs & guards
        z = np.asarray(z)
        n_dl = np.asarray(n_dl)  # shape (n_bins, len(z))
        n_bins = len(n_dl)

        assert np.all(np.diff(z) < self.dz_windows), \
            f"For convergence reasons, dz must be <= {self.dz_windows:.3f}"
        assert n_dl.ndim == 2, "'n_dl' must be 2-dimensional"
        assert n_dl.shape[1] == z.shape[0], "Each n_dl[i] must have length len(z)"
        assert z.min() <= self.z_min, f"z.min() must be <= {self.z_min:.3f}"
        assert z.max() >= self.z_max, f"z.max() must be >= {self.z_max:.3f}"

        # bias vector
        if isinstance(bias, (float, int, np.floating, np.integer)):
            bias = float(bias) * np.ones(n_bins)
        else:
            assert len(bias) == n_bins, "Number of bias factors must equal number of bins"

        # --- Normalizations (one per branch to keep each internally consistent)
        # bg normalization uses dL_bg(z) = (1+z) * r_bg(z)
        dL_bg_on_z = (1.0 + z) * chi_bg_interp(z)  # [Mpc]
        norm_bg = simpson(n_dl, x=dL_bg_on_z, axis=1)

        # self/LCDM normalization uses dL_LCDM(z) from the provided FlatLambdaCDM
        dL_lcdm_on_z = conversion.luminosity_distance(z).value  # [Mpc]
        norm_self = simpson(n_dl, x=dL_lcdm_on_z, axis=1)

        # --- Build both windows on the *same* integration grid
        z_int = np.asarray(self.z_integration)

        # bg branch: H and r from bg
        H_bg_on_z = H_bg_interp(z_int)  # [1/Mpc]
        r_bg_on_z = chi_bg_interp(z_int)  # [Mpc]
        jac_dL_bg = (1.0 + z_int) / H_bg_on_z + r_bg_on_z  # [Mpc]

        # self/LCDM branch:
        # self.Hubble was defined in __init__ as H/h (km/s/Mpc)/h; convert to 1/Mpc via *h/c
        h = float(self.cosmology.h)
        if np.isscalar(self.Hubble):
            H_self_on_z = np.full_like(z_int, float(self.Hubble) * h) / const.c
        else:
            H_self_on_z = (np.asarray(self.Hubble) * h) / const.c  # already on z_int
        # r from LCDM conversion for consistency with your original "self" expression
        dL_lcdm_int = conversion.luminosity_distance(z_int).value  # [Mpc]
        r_lcdm_on_z = dL_lcdm_int / (1.0 + z_int)  # [Mpc]
        jac_dL_self = (1.0 + z_int) / H_self_on_z + r_lcdm_on_z  # [Mpc]

        windows_bg, windows_self = [], []

        print(f"\n[load_gravitational_wave_window_functions] Comparing bg vs self on {len(z_int)} z-points")
        for i_bin in range(n_bins):
            ndl_i = si.interp1d(z, n_dl[i_bin], kind='cubic', bounds_error=False, fill_value=0.0)
            ndl_on_z = ndl_i(z_int)

            # W_i(z) = b_i * n_dl_i(d_L(z)) * (ddL/dz) * H(z) / N_i
            w_bg_vals = bias[i_bin] * ndl_on_z * jac_dL_bg * H_bg_on_z / norm_bg[i_bin]
            w_self_vals = bias[i_bin] * ndl_on_z * jac_dL_self * H_self_on_z / norm_self[i_bin]

            windows_bg.append(
                si.interp1d(z_int, w_bg_vals, kind='cubic', bounds_error=False, fill_value=0.0)
            )
            windows_self.append(
                si.interp1d(z_int, w_self_vals, kind='cubic', bounds_error=False, fill_value=0.0)
            )

            # robust diffs (avoid division by ~0)
            denom = np.maximum(np.maximum(np.abs(w_bg_vals), np.abs(w_self_vals)), 1e-12)
            abs_diff = float(np.max(np.abs(w_bg_vals - w_self_vals)))
            rel_diff = float(np.max(np.abs((w_bg_vals - w_self_vals) / denom)))
            print(f"  bin {i_bin:02d}: max_abs_diff={abs_diff:.3e}, max_rel_diff={rel_diff:.3e}")

        # --- store both versions without overwriting each other
        self.window_function[name + "_bg"] = np.tile(np.array(windows_bg).reshape(-1, 1), (1, len(ll)))
        self.window_function[name + "_self"] = np.tile(np.array(windows_self).reshape(-1, 1), (1, len(ll)))

        # choose one as the default (bg is usually the most consistent with your pipeline)
        self.window_function[name] = self.window_function[name + "_bg"]


'''
    def limber_angular_power_spectra_parallel(self, l, windows = None):
        """
        This function computes the angular power spectra (using the Limber's and the flat-sky approximations) for the window function specified.
        Given two redshift bins `i` and `j` the equation is

        .. math::

          C^{(ij)}(\ell) = \int_0^\infty dz \ \frac{c}{H(z)} \ \frac{W^{(i)}(z) W^{(j)}(z)}{f_K^2[\chi(z)]} \ P\left(\frac{\ell}{f_K[\chi(z)]}, z\right),

        where :math:`P(k,z)` is the matter power spectrum and :math:`W^{(i)}(z)` are the window functions.

        :param l: Multipoles at which to compute the shear power spectra.
        :type l: array

        :param windows: which spectra (auto and cross) must be computed. If set to ``None`` all the spectra will be computed.
        :type windows: list of strings, default = ``None``

        :return: dictionary whose keys are combinations of window functions specified in ``windows``. Each key is a 3-D array whose entries are ``Cl[bin i, bin j, multipole l]``
        """

        from concurrent.futures import ProcessPoolExecutor
        import numpy as np
        from scipy.integrate import simpson

        def _compute_cl_task(args):
            key_X, key_Y, bin_i, bin_j, W_X, W_Y, cH_chi2, PS_lz, zz = args
            cl_result = [
                simpson(cH_chi2 * W_X[bin_i, xx] * W_Y[bin_j, xx] * PS_lz[xx], x=zz)
                for xx in range(len(zz))
            ]
            return key_X, key_Y, bin_i, bin_j, cl_result

        # Check existence of power spectrum
        try: self.power_spectra_interpolator
        except AttributeError: raise AttributeError("Power spectra have not been loaded yet")

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min*self.geometric_factor_f_K(self.z_min), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max*self.geometric_factor_f_K(self.z_max), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

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

        # 1) Define lengths and quantities
        zz       = self.z_integration
        n_l      = len(np.atleast_1d(l))
        n_z      = self.n_z_integration
        HH       = self.Hubble
        cH_chi2  = self.c_over_H_over_chi_squared
        Cl       = {}
        for i,ki in enumerate(keys):
            for j,kj in enumerate(keys):
                Cl['%s-%s' %(ki,kj)] = np.zeros((n_bins[i],n_bins[j], n_l))

        # 2) Load power spectra
        power_spectra = self.power_spectra_interpolator
        PS_lz = np.zeros((n_l, n_z))
        for il in range(n_l):
            for iz in range(n_z):
                PS_lz[il,iz] = power_spectra([(zz[iz], (l[il]+0.5)/self.geometric_factor[iz])])
        # Add curvature correction (see arXiv:2302.04507)
        if self.cosmology.K != 0.:
            KK = self.cosmology.K
            factor = np.zeros((n_l,n_z))
            for il,ell in enumerate(l):
                factor[il]=(1-np.sign(KK)*ell**2/(((ell+0.5)/self.geometric_factor)**2+KK))**-0.5
            PS_lz *= factor

        # 3) Parallelized Cl computation
        tasks = []
        key_list = list(keys)
        for index_X in range(n_keys):
            key_X = key_list[index_X]
            W_X = np.array([[windows_to_use[key_X][i,j](zz) for j in range(n_l)] for i in range(n_bins[index_X])])
            for index_Y in range(index_X, n_keys):
                key_Y = key_list[index_Y]
                W_Y = np.array([[windows_to_use[key_Y][i,j](zz) for j in range(n_l)] for i in range(n_bins[index_Y])])
                if key_X == key_Y:
                    for bin_i in range(n_bins[index_X]):
                        for bin_j in range(bin_i, n_bins[index_Y]):
                            tasks.append((key_X, key_Y, bin_i, bin_j, W_X, W_Y, cH_chi2, PS_lz, zz))
                else:
                    for bin_i in range(n_bins[index_X]):
                        for bin_j in range(n_bins[index_Y]):
                            tasks.append((key_X, key_Y, bin_i, bin_j, W_X, W_Y, cH_chi2, PS_lz, zz))

        with ProcessPoolExecutor() as executor:
            results = executor.map(_compute_cl_task, tasks)

        for key_X, key_Y, bin_i, bin_j, cl_result in results:
            Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j] = cl_result
            Cl['%s-%s' % (key_Y, key_X)][bin_j, bin_i] = cl_result

        return Cl
    '''

    # -----------------------------------------------------------------------------------------
    # ANGULAR POWER SPECTRA AUTO CORRELATION LENSING: PRIMA
    # -----------------------------------------------------------------------------------------
    def limber_angular_power_spectra_lensing_auto_prima(
            self, l, s_gal, beta, H_0, omega_m, omega_b,
            windows=None, n_points=20, n_points_x=20, grid_x='mix',
            n_low=5, n_high=5, Delta_z=0.05, z_min=1e-5
    ):
        import numpy as np
        import scipy.interpolate as si
        from astropy.cosmology import FlatLambdaCDM
        from astropy import constants as const

        cosmo = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b)

        # Power spectra must be loaded
        try:
            self.power_spectra_interpolator
        except AttributeError:
            raise AttributeError("Power spectra have not been loaded yet")

        # Convergence checks on (l, k, z)
        assert np.atleast_1d(l).min() > self.k_min * self.geometric_factor_f_K(self.z_min), \
            "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max * self.geometric_factor_f_K(self.z_max), \
            "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:
                    windows_to_use[key] = self.window_function[key]
                except KeyError:
                    raise KeyError("Requested window function '%s' not known" % key)

        # Check window functions
        keys = list(windows_to_use.keys())
        if len(keys) == 0:
            raise AttributeError("No window function has been computed!")
        nkeys = len(keys)
        n_bins = [len(windows_to_use[key]) for key in keys]

        n_l = len(np.atleast_1d(l))
        Cl = {}
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                Cl['%s-%s' % (ki, kj)] = np.zeros((n_bins[i], n_bins[j], n_l))

        # Bias interpolation support grid
        z_interp_bias = np.linspace(z_min, 10.0, 1200)

        # Helpers (unit-safe, vectorized)
        def H(x):
            # km / s / Mpc
            return cosmo.H(np.asarray(x)).value

        def r(x):
            # comoving distance in Mpc
            return cosmo.comoving_distance(np.asarray(x)).value

        def JJ(tracer, x):
            x = np.asarray(x)
            if 'gal' in tracer:
                return np.ones_like(x, dtype=float)
            # lensing magnification Jacobian-like term
            return (1.0 + x) * const.c.to('km/s').value / H(x) + r(x)

        def A_L(chi, z, tracer, r_z1, bs, Hvals):
            # chi, r_z1 in Mpc, Hvals in km/s/Mpc
            if 'gal' in tracer:
                return 0.5 * (5.0 * bs - 2.0) * (r_z1 - chi) / chi
            else:
                conf_H = Hvals / (1.0 + z)
                return 0.5 * (((r_z1 - chi) / chi) * (bs - 2.0) + (1.0 / (1.0 + r_z1 * conf_H)))

        power_spectra = self.power_spectra_interpolator  # P(k,z)

        # Core integrand (kept self-contained like the working version)
        def lensing_int(z1, z2, ll, bin_i, bin_j, WX_, WY_, tX, tY,
                        n_points=30, p=1, z_min=1e-4, grid='mix', show_plot=False):

            # Build x grid (mix of linear + geometric in z)
            if grid == 'mix':
                x1 = np.linspace(z_min * (1 + 0.01), z2, n_points)
                x2 = np.geomspace(z_min, z2 * (1 - 0.01), n_points)
                x = np.zeros((2 * x1.shape[0], x1.shape[1], x1.shape[2]))
                for ii in range(x1.shape[1]):
                    for jj in range(x1.shape[2]):
                        x_ = np.sort(np.unique(np.concatenate([x1[:, ii, jj], x2[:, ii, jj]])))
                        if len(x_) < x.shape[0]:
                            n_ = x.shape[0] - len(x_)
                            x_ = np.sort(np.unique(np.concatenate(
                                [x_, np.linspace(z_min * (1 + 0.05), z2[ii, jj] * (1 - 0.05), n_)]
                            )))
                        x[:, ii, jj] = x_
            elif grid == 'lin':
                x = np.linspace(z_min, z2, n_points)
            elif grid == 'geom':
                x = np.geomspace(z_min, z2, n_points)
            else:
                raise ValueError("Unknown grid '%s'" % grid)

            r1 = r(z1)  # (L,)
            r2 = r(z2)  # (N2,L)
            rx = r(x)  # (Nx,N2,L)
            Hx = H(x)

            # Interp bias inside (match old behavior)
            if 'gal' in tX:
                b1 = si.interp1d(z_interp_bias, s_gal, kind='cubic', bounds_error=False, fill_value=0.0)
            else:
                b1 = si.interp1d(z_interp_bias, beta, kind='cubic', bounds_error=False, fill_value=0.0)

            if 'gal' in tY:
                b2 = si.interp1d(z_interp_bias, s_gal, kind='cubic', bounds_error=False, fill_value=0.0)
            else:
                b2 = si.interp1d(z_interp_bias, beta, kind='cubic', bounds_error=False, fill_value=0.0)

            # c in km/s to match H units
            c_km_s = const.c.to('km/s').value

            t1 = np.transpose(
                c_km_s * A_L(rx, x, tX, r1, b1(x), Hx) *
                A_L(rx, x, tY, r2, b2(x), Hx) *
                (1.0 + x) ** 2 / Hx * rx ** 2,
                (1, 2, 0)
            )  # -> (L, N2, Nx)

            # Power spectrum P(k= (l+0.5)/r, z=x)
            # Build PS_ with shapes aligned to t1
            PS_ = np.squeeze(np.asarray([[[[
                    power_spectra((xx, yy))  # <-- FIX: wrap as a single tuple
                    for xx, yy in zip((ll[lidx] + 0.5) / rx[:, k, i], x[:, k, i])
                ] for i in range(len(z1))]
                    for k in range(z2.shape[0])]
                for lidx in range(len(ll))
            ]))

            my_int = t1[None, :, :, :] * PS_  # (L, L, N2, Nx) broadcasted to (L, L, N2, Nx) -> effectively (L, N2, Nx)

            # --- Integrate over x (inner): keep the l-axis ---
            I1_ = np.asarray([
                [
                    [
                        np.trapz(my_int[l, i2, i1], x=x[:, i2, i1], axis=0)
                        for i1 in range(len(z1))
                    ]
                    for i2 in range(z2.shape[0])
                ]
                for l in range(len(ll))
            ])  # (L, N2, L1)

            # --- Integrate over z2: still per-l ---
            I2_ = np.asarray([
                [
                    np.trapz(WY_[:, i1] * JJ(tY, z2[:, i1]) * I1_[l, :, i1] / r2[:, i1],
                             x=z2[:, i1], axis=0)
                    for i1 in range(len(z1))
                ]
                for l in range(len(ll))
            ])  # (L, L1)

            # --- Integrate over z1: collapse to (L,) ---
            I3_ = np.asarray([
                np.trapz(WX_ * JJ(tX, z1) * I2_[l, :] / r1, x=z1, axis=0)
                for l in range(len(ll))
            ])  # (L,)

            return I3_

        # 3) Compute Cl
        for index_X in range(nkeys):
            key_X = keys[index_X]

            # Adaptive z-grid per bin, denser near window peaks
            bins_ = self.bin_edges[key_X]
            bins_centers_ = (bins_[:-1] + bins_[1:]) / 2.0
            zzs = []
            ll_max = 0
            for bin_i in range(n_bins[index_X]):
                if 'gal' in key_X:
                    maxz = 5.0
                    npts = n_high
                else:
                    maxz = 10.0
                    npts = 2 * n_high

                if max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01) == z_min:
                    n1 = n_points + n_low
                else:
                    n1 = n_points

                if bin_i < n_bins[index_X] - 1:
                    myarr = np.sort(np.unique(np.concatenate([
                        np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)),
                                    bins_[bin_i + 1] * (1 + 5 * Delta_z), n_points),
                        np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low),
                        np.linspace(bins_[bin_i + 1] * (1 + 0.05) + 0.01, maxz, npts)
                    ])))
                else:
                    myarr = np.sort(np.unique(np.concatenate([
                        np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)), maxz, n_points + npts),
                        np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low)
                    ])))
                l_ = len(myarr)
                ll_max = max(ll_max, l_)
                zzs.append(myarr)

            for i, a in enumerate(zzs):
                if len(a) != ll_max:
                    n_ = ll_max - len(a)
                    zzs[i] = np.sort(np.unique(np.concatenate([
                        zzs[i],
                        np.linspace(z_min * (1 + 0.01), max(a) * (1 - 0.01), n_)
                    ])))
            try:
                zzs = np.asarray(zzs)  # (n_bins_X, L)
            except Exception as e:
                print(zzs)
                print(maxz)  # corrected name (not z_max)
                print(e)
                raise

            # Windows for X
            W_X = np.array([
                [windows_to_use[key_X][i, j](zzs[i]) for j in range(n_l)]
                for i in range(n_bins[index_X])
            ])  # (n_bins_X, n_l, L)

            for index_Y in range(nkeys):
                key_Y = keys[index_Y]

                for bin_i in range(n_bins[index_X]):
                    # We only compute diagonal (i==j) as in the old working code
                    for bin_j in range(n_bins[index_Y]):
                        if bin_j == bin_i:
                            # build 2D z2 grid up to each z1 element (shape (n_points_x, L))
                            z2s_ = np.linspace(z_min, zzs[bin_i], n_points)
                            # Windows for Y (on z2 grid)
                            W_Y = np.array([
                                [windows_to_use[key_Y][i, j](z2s_) for j in range(n_l)]
                                for i in range(n_bins[index_Y])
                            ])  # (n_bins_Y, n_l, n_points, L)

                            WY = W_Y[bin_j, 0]  # (n_points, L)
                            WX = W_X[bin_i, 0]  # (L,)

                            pref = l ** 2 * (l + 1) ** 2 / (l + 0.5) ** 4  # (n_l,)
                            I3_vec = lensing_int(
                                zzs[bin_i], z2s_, np.atleast_1d(l), bin_i, bin_j,
                                WX, WY, key_X, key_Y,
                                z_min=z_min, n_points=n_points_x, grid=grid_x
                            )  # (n_l,)

                            Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j, :] = pref * I3_vec  # (n_l,)

        return Cl


h = C.h
z_centers_use = np.asarray(z_centers_use, float).ravel()
k_max = np.asarray(k_max, float).ravel()

# Astropy path (for cross-check)
chi_ast_hMpc = np.asarray(
    [fiducial_universe.comoving_distance(z).value for z in z_centers_use]
) * h
ell_ast = chi_ast_hMpc * k_max
l_max_nl_astropy = np.rint(ell_ast).astype(int)

# C path — vectorized, no scalar casts, no warnings
chi_C_hMpc = np.asarray(C.comoving_distance(z_centers_use), float).ravel()
ell_C = chi_C_hMpc * k_max
l_max_nl_C = np.rint(ell_C).astype(int)

# Compare (float first, then ints)
rel_float = np.max(np.abs((ell_C - ell_ast) / ell_ast))
print("max |Δℓ/ℓ| (float) =", rel_float)

diff_int = l_max_nl_C - l_max_nl_astropy
print("int Δℓ: min/max =", diff_int.min(), diff_int.max())
print("fraction of bins with Δℓ ≠ 0:", np.mean(diff_int != 0))

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from joblib import Parallel, delayed
from multiprocessing import Process


# Compute shot noise matrices for galaxies and GW
	noise_gal = fcc.shot_noise_mat_auto(shot_noise_gal, ll_total)
	noise_GW = fcc.shot_noise_mat_auto(shot_noise_GW, ll_total)

	# Initialize localization noise attenuation matrices
	noise_loc = np.zeros(shape=(n_bins_dl, len(ll_total)))
	noise_loc_auto = np.zeros(shape=(n_bins_dl, len(ll_total)))

	# Parallelization
	def compute_noise_for_bin(i):
		loc_row = np.zeros(len(ll_total))
		loc_auto_row = np.zeros(len(ll_total))

		for l in range(len(ll_total)):
			ell = ll_total[l]
			ell_term = ell * (ell + 1) * (sigma_sn_GW[i] / (2 * np.pi) ** (3 / 2))
			if ell_term < 30:
				loc_row[l] = np.exp(-ell_term)
				loc_auto_row[l] = np.exp(-2 * ell_term)
			else:
				loc_row[l] = np.exp(-30)
				loc_auto_row[l] = np.exp(-30)

		return i, loc_row, loc_auto_row

	def parallel_compute_noise():
		global noise_loc, noise_loc_auto
		with ProcessPoolExecutor() as executor:
			results = executor.map(compute_noise_for_bin, range(n_bins_dl))
			for i, loc_row, loc_auto_row in results:
				noise_loc[i, :] = loc_row
				noise_loc_auto[i, :] = loc_auto_row

	# Execute the parallel computation
	parallel_compute_noise()

	# Downstream array constructions
	noise_loc_mat = np.zeros((n_bins_z, n_bins_dl, len(ll_total)))
	for i in range(n_bins_z):
		for ii in range(n_bins_dl):
			noise_loc_mat[i, ii, :] = noise_loc[ii, :]

	noise_loc_mat_auto = np.zeros((n_bins_dl, n_bins_dl, len(ll_total)))
	for i in range(n_bins_dl):
		for ii in range(i, n_bins_dl):
			noise_loc_mat_auto[i, ii, :] = noise_loc_auto[ii, :]
	for i in range(n_bins_dl):
		for ii in range(i + 1, n_bins_dl):
			noise_loc_mat_auto[ii, i, :] = noise_loc_mat_auto[i, ii, :]

	# Parallelization
	def interpolate_Cl(args):
		i, ii, matrix, axis_type = args
		if axis_type == 'GG':
			Cl_slice = matrix[i, ii]
		elif axis_type == 'GWGW':
			Cl_slice = matrix[i, ii]
		elif axis_type == 'GGW':
			Cl_slice = matrix[i, ii]
		f_interp = si.interp1d(ll, Cl_slice, kind='linear', bounds_error=False, fill_value="extrapolate")
		return (i, ii, axis_type, f_interp(ll_total))


	# Generate tasks for parallel interpolation
	tasks = []
	for i in range(n_bins_z):
		for ii in range(n_bins_z):
			tasks.append((i, ii, Cl_GG, 'GG'))
	for i in range(n_bins_dl):
		for ii in range(n_bins_dl):
			tasks.append((i, ii, Cl_GWGW, 'GWGW'))
	for i in range(n_bins_z):
		for ii in range(n_bins_dl):
			tasks.append((i, ii, Cl_GGW, 'GGW'))

	# Parallel execution
	with Pool() as pool:
		results = pool.map(interpolate_Cl, tasks)

	# Initialize output arrays
	Cl_GG_total = np.zeros((n_bins_z, n_bins_z, len(ll_total)))
	Cl_GWGW_total = np.zeros((n_bins_dl, n_bins_dl, len(ll_total)))
	Cl_GGW_total = np.zeros((n_bins_z, n_bins_dl, len(ll_total)))

	# Fill results
	for i, ii, axis_type, interpolated in results:
		if axis_type == 'GG':
			Cl_GG_total[i, ii, :] = interpolated
		elif axis_type == 'GWGW':
			Cl_GWGW_total[i, ii, :] = interpolated
		elif axis_type == 'GGW':
			Cl_GGW_total[i, ii, :] = interpolated


	# Save all computed Cls and noise terms to disk
	np.save(os.path.join(FLAGS.fout, 'Cl_GG'), Cl_GG_total)
	np.save(os.path.join(FLAGS.fout, 'Cl_GWGW'), Cl_GWGW_total)
	np.save(os.path.join(FLAGS.fout, 'Cl_GGW'), Cl_GGW_total)
	np.save(os.path.join(FLAGS.fout, 'noise_GW'), noise_GW)
	np.save(os.path.join(FLAGS.fout, 'noise_gal'), noise_gal)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_auto'), noise_loc_mat_auto)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_cross'), noise_loc_mat)

	# Apply localization damping matrices to GW-GW and GW-Galaxy spectra
	Cl_GWGW_total *= noise_loc_mat_auto
	Cl_GGW_total *= noise_loc_mat

	# Add shot noise to the auto-correlations
	Cl_GWGW_total += noise_GW
	Cl_GG_total += noise_gal

	# -----------------------------------------------------------------------------------------
	#               COMPUTING PARAMETER DERIVATIVE MATRIX
	# -----------------------------------------------------------------------------------------
	print('\nComputing parameter derivative matrix...\n')
	covariance_matrices = {}


	# Return FULLY PROCESSED BLOCK for any cosmology (same pipeline as fiducial)  # <<< FIX
	def Cl_func_wrapped(cosmo_params, H0, Omega_m, Omega_b, A_s, n_s, alpha_M, alpha_B, w_0, w_a):
		params = deepcopy(cosmo_params)
		params['h'] = H0 / 100.0
		params['Omega_m'] = Omega_m
		params['Omega_b'] = Omega_b
		params['A_s'] = A_s
		params['n_s'] = n_s
		params['parameters_smg'] = f"1.0,{alpha_M},{alpha_B},0.0,1.0"
		C_ = cc.cosmo(**params)

		# raw on ll
		GG, GWGW, GGW = Cl_func(C_, params, gw_params, dl_GW, bin_edges_dl, z_gal, ll,
								bias_gal, bias_GW, save=False)
		# interpolate to ll_total, apply damping + shot noise, assemble block  # <<< FIX
		GGt = _interp_to_ll_total(GG)
		GWt = _interp_to_ll_total(GWGW)
		GGWt = _interp_to_ll_total(GGW)
		GWt *= noise_loc_mat_auto
		GGWt *= noise_loc_mat
		GWt += noise_GW
		GGt += noise_gal
		D_ = n_bins_z + n_bins_dl
		block = np.zeros((D_, D_, len(ll_total)))
		block[:n_bins_z, :n_bins_z, :] = GGt
		block[n_bins_z:, n_bins_z:, :] = GWt
		block[:n_bins_z, n_bins_z:, :] = GGWt
		block[n_bins_z:, :n_bins_z, :] = np.swapaxes(GGWt, 0, 1)
		for k in range(block.shape[2]):
			Mk = block[:, :, k]
			block[:, :, k] = 0.5 * (Mk + Mk.T)
		return block  # <<< returns (D,D,len(ll_total))


	# If fem.compute_parameter_derivatives() uses Cl_func_wrapped, it will now produce BLOCK derivatives on ll_total.
	# Otherwise, compute your derivatives here similarly to the bias case.

	# Compute and save the derivative covariance BLOCKS for each parameter (via your helper)
	fem.compute_parameter_derivatives(parameters, FLAGS, n_bins_z, n_bins_dl, covariance_matrices)


	# -----------------------------------------------------------------------------------------
	#                   COMPUTING THE POWER SPECTRUM
	# -----------------------------------------------------------------------------------------
	print('\nComputing the Power Spectrum...\n')

	# Compute angular power spectra from Cl_func with fiducial cosmological and bias parameters (on ll)
	Cl_GG, Cl_GWGW, Cl_GGW = Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll,
									 b_gal=bias_gal, b_GW=bias_GW, save=True)


	# Interpolate EACH (i,j) curve to ll_total WITHOUT extrapolation  # <<< FIX
	def _interp_to_ll_total(M):
		out = np.zeros((M.shape[0], M.shape[1], len(ll_total)))
		for i in range(M.shape[0]):
			for j in range(M.shape[1]):
				f_interp = si.interp1d(
					ll, M[i, j], kind='linear', bounds_error=False,
					fill_value=(M[i, j, 0], M[i, j, -1])  # <<< FIX: no "extrapolate"
				)
				out[i, j, :] = f_interp(ll_total)
		return out


	Cl_GG_total = _interp_to_ll_total(Cl_GG)
	Cl_GWGW_total = _interp_to_ll_total(Cl_GWGW)
	Cl_GGW_total = _interp_to_ll_total(Cl_GGW)

	# Save all computed Cls and noise terms to disk (interpolated)
	np.save(os.path.join(FLAGS.fout, 'Cl_GG'), Cl_GG_total)
	np.save(os.path.join(FLAGS.fout, 'Cl_GWGW'), Cl_GWGW_total)
	np.save(os.path.join(FLAGS.fout, 'Cl_GGW'), Cl_GGW_total)
	np.save(os.path.join(FLAGS.fout, 'noise_GW'), noise_GW)
	np.save(os.path.join(FLAGS.fout, 'noise_gal'), noise_gal)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_auto'), noise_loc_mat_auto)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_cross'), noise_loc_mat)

	# Apply localization damping matrices to GW-GW and GW-Galaxy spectra (on ll_total)
	Cl_GWGW_total *= noise_loc_mat_auto
	Cl_GGW_total *= noise_loc_mat

	# Add shot noise to the auto-correlations
	Cl_GWGW_total += noise_GW
	Cl_GG_total += noise_gal

	# Assemble the final observable covariance BLOCK per-ℓ (D,D,len(ll_total))  # <<< FIX
	D = n_bins_z + n_bins_dl
	cov_mat = np.zeros((D, D, len(ll_total)))
	cov_mat[:n_bins_z, :n_bins_z, :] = Cl_GG_total
	cov_mat[n_bins_z:, n_bins_z:, :] = Cl_GWGW_total
	cov_mat[:n_bins_z, n_bins_z:, :] = Cl_GGW_total
	cov_mat[n_bins_z:, :n_bins_z, :] = np.swapaxes(Cl_GGW_total, 0, 1)
	# Symmetrize per ℓ (numerical hygiene)
	for k in range(cov_mat.shape[2]):
		M = cov_mat[:, :, k]
		cov_mat[:, :, k] = 0.5 * (M + M.T)

	# Save the fiducial observable covariance block
	np.save(os.path.join(FLAGS.fout, 'cov_mat'), cov_mat)

	# -----------------------------------------------------------------------------------------
	#           COMPUTING DERIVATIVES WITH RESPECT TO BIASES (now BLOCK derivatives)
	# -----------------------------------------------------------------------------------------
	# NOTE: last dimension changed to len(ll_total)  # <<< FIX
	der_b_gal = np.zeros(shape=(n_bins_z, n_bins_dl + n_bins_z, n_bins_dl + n_bins_z, len(ll_total)))


	def compute_partial_derivatives_gal(b_gal, bias_GW, der_b_gal, step):
		for i in range(len(b_gal)):
			print('\nComputing the derivative with respect to the galaxy bias in bin %i...\n' % i)

			def fun(b):
				b_tmp = np.copy(b_gal);
				b_tmp[i] = b
				GG, GWGW, GGW = Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll,
										b_tmp, bias_GW, save=False)
				GGt = _interp_to_ll_total(GG)
				GWt = _interp_to_ll_total(GWGW)
				GGWt = _interp_to_ll_total(GGW)
				GWt *= noise_loc_mat_auto
				GGWt *= noise_loc_mat
				GWt += noise_GW
				GGt += noise_gal
				block = np.zeros((D, D, len(ll_total)))
				block[:n_bins_z, :n_bins_z, :] = GGt
				block[n_bins_z:, n_bins_z:, :] = GWt
				block[:n_bins_z, n_bins_z:, :] = GGWt
				block[n_bins_z:, :n_bins_z, :] = np.swapaxes(GGWt, 0, 1)
				for k in range(block.shape[2]):
					Mk = block[:, :, k]
					block[:, :, k] = 0.5 * (Mk + Mk.T)
				return block

			der_block = (fun(b_gal[i] + step) - fun(b_gal[i] - step)) / (2.0 * step)  # <<< FIX
			der_b_gal[i] = der_block
			np.save(os.path.join(FLAGS.fout, 'der_b_gal_block_bin_%i.npy' % i), der_block)
		return der_b_gal


	der_b_gal_cov_mat = compute_partial_derivatives_gal(bias_gal, bias_GW, der_b_gal, step)

	# GW bias derivatives (BLOCK)
	der_bGW = np.zeros(shape=(n_bins_dl, n_bins_dl + n_bins_z, n_bins_dl + n_bins_z, len(ll_total)))


	def compute_partial_derivatives_GW(b_GW, bias_gal, der_b_GW, step):
		for i in range(len(b_GW)):
			print('\nComputing the derivative with respect to the GW bias in bin %i...\n' % i)

			def fun(b):
				b_tmp = np.copy(b_GW);
				b_tmp[i] = b
				GG, GWGW, GGW = Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll,
										bias_gal, b_tmp, save=False)
				GGt = _interp_to_ll_total(GG)
				GWt = _interp_to_ll_total(GWGW)
				GGWt = _interp_to_ll_total(GGW)
				GWt *= noise_loc_mat_auto
				GGWt *= noise_loc_mat
				GWt += noise_GW
				GGt += noise_gal
				block = np.zeros((D, D, len(ll_total)))
				block[:n_bins_z, :n_bins_z, :] = GGt
				block[n_bins_z:, n_bins_z:, :] = GWt
				block[:n_bins_z, n_bins_z:, :] = GGWt
				block[n_bins_z:, :n_bins_z, :] = np.swapaxes(GGWt, 0, 1)
				for k in range(block.shape[2]):
					Mk = block[:, :, k]
					block[:, :, k] = 0.5 * (Mk + Mk.T)
				return block

			der_block = (fun(b_GW[i] + step) - fun(b_GW[i] - step)) / (2.0 * step)  # <<< FIX
			der_b_GW[i] = der_block
			np.save(os.path.join(FLAGS.fout, 'der_b_GW_block_bin_%i.npy' % i), der_block)
		return der_b_GW


	der_bGW_cov_mat = compute_partial_derivatives_GW(bias_GW, bias_gal, der_bGW, step)

	eigvals, eigvecs = np.linalg.eigh(fisher)
	for i, v in enumerate(eigvals):
		if abs(v) < 1e-12:
			print(f"\nEigenvalue {i} = {v:.3e}")
			print("Degenerate combination of parameters (weights):")
			for j in range(len(eigvecs)):
				print(f"  Param {j}: {eigvecs[j, i]:+.4f}")

	# Compute scaling factor to bring max entry ~1
	scale = 1.0 / np.max(np.abs(fisher))
	print("Rescaling factor:", scale)

	# Rescale
	fisher_scaled = fisher * scale

	# Attempt inversion
	try:
		fisher_inv_scaled = scipy.linalg.inv(fisher_scaled)
		fisher_inv = fisher_inv_scaled / scale
		print("Inversion successful.")
	except np.linalg.LinAlgError:
		print("Fisher matrix is singular—using pseudo-inverse.")
		fisher_inv_scaled = np.linalg.pinv(fisher_scaled)
		fisher_inv = fisher_inv_scaled / scale


	try:
		fisher_inv = scipy.linalg.inv(fisher)
	except np.linalg.LinAlgError:
		print("Fisher matrix is singular ----> pseudo-inverse")
		fisher_inv = np.linalg.pinv(fisher)  # use pseudo-inverse


def compute_auto_block(args):
    j, k, cl_block = args
    return cl_block[j, k]

def vector_cl(cl_cross, cl_auto1, cl_auto2):
    n_bins_1 = len(cl_auto1)
    n_bins_2 = len(cl_auto2)

    length = n_bins_1**2 - np.sum(range(n_bins_1)) + (n_bins_1 * n_bins_2) + n_bins_2**2 - np.sum(range(n_bins_2))
    ell_size = cl_auto1.shape[2]
    vec_cl = np.zeros((length, ell_size))

    i = 0
    auto1_tasks = [(j, k, cl_auto1) for j in range(n_bins_1) for k in range(j, n_bins_1)]
    cross_tasks = [(j, k, cl_cross) for j in range(n_bins_1) for k in range(n_bins_2)]
    auto2_tasks = [(j, k, cl_auto2) for j in range(n_bins_2) for k in range(j, n_bins_2)]

    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(compute_auto_block, auto1_tasks):
            vec_cl[i] = result
            i += 1
        for result in executor.map(compute_auto_block, cross_tasks):
            vec_cl[i] = result
            i += 1
        for result in executor.map(compute_auto_block, auto2_tasks):
            vec_cl[i] = result
            i += 1

    return vec_cl



# Compute the covariance matrix
def unpack_upper_triangle(args):
    i, j, index, vec_cl = args
    return i, j, vec_cl[index]

def covariance_matrix(vec_cl, n_bins_1, n_bins_2):
    dim = n_bins_1 + n_bins_2
    ell_len = vec_cl.shape[1]
    cov_matrix = np.zeros((dim, dim, ell_len))

    tasks = []
    idx = 0

    # Upper triangle - auto1
    for j in range(n_bins_1):
        for k in range(j, n_bins_1):
            tasks.append((j, k, idx, vec_cl))
            idx += 1

    # Cross
    for j in range(n_bins_1):
        for k in range(n_bins_1, dim):
            tasks.append((j, k, idx, vec_cl))
            idx += 1

    # Upper triangle - auto2
    for j in range(n_bins_1, dim):
        for k in range(j, dim):
            tasks.append((j, k, idx, vec_cl))
            idx += 1

    # Fill upper triangle
    with ProcessPoolExecutor(max_workers=4) as executor:
        for j, k, value in executor.map(unpack_upper_triangle, tasks):
            cov_matrix[j, k] = value

    # Symmetrize
    for i in range(dim):
        for j in range(i + 1, dim):
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix

# Compute the fisher matrix
def fisher_worker(i, Cl_i, Cl_der_i, n_par):
    Cl_inv_i = np.linalg.inv(Cl_i)
    trace_row = np.zeros((n_par, n_par))
    for a in range(n_par):
        for b in range(n_par):
            term = Cl_inv_i @ Cl_der_i[a] @ Cl_inv_i @ Cl_der_i[b]
            trace_row[a, b] = np.trace(term)
    return i, trace_row

def fisher_matrix(Cl, Cl_der, ll, f_sky=0.3):
    n_par = len(Cl_der)
    n_ell = len(ll)
    fisher = np.zeros((n_par, n_par))

    # Pack inputs for workers
    args = [(i, Cl[:, :, i], Cl_der[:, :, :, i], n_par) for i in range(n_ell)]

    traces = [None] * n_ell
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, trace_row in executor.map(lambda x: fisher_worker(*x), args):
            traces[i] = trace_row

    # Combine traces and apply (2l+1)/2 weights
    for i in range(n_ell):
        weight = (2 * ll[i] + 1) / 2.0
        fisher += weight * traces[i]

    return f_sky * fisher


# Compute the fisher matrix for different l_max in each bin
def fisher_trace_worker(args):
    i, Cl_slice, Cl_der_slice, n_par = args
    Cl_inv = np.linalg.inv(Cl_slice)
    trace_out = np.zeros((n_par, n_par))
    for a in range(n_par):
        for b in range(n_par):
            product = Cl_inv @ Cl_der_slice[a] @ Cl_inv @ Cl_der_slice[b]
            trace_out[a, b] = np.trace(product)
    return i, trace_out

def fisher_matrix_different_l(Cl, Cl_der, l_min_tot, l_max_bin, f_sky=0.3):
    l_max_bin = l_max_bin[::-1]
    n_par = len(Cl_der)
    fisher = np.zeros((len(l_max_bin), n_par, n_par))

    for j in range(len(l_max_bin)):
        # Determine ℓ range for the bin
        if j == 0:
            ll_aux = np.arange(l_min_tot, l_max_bin[j] + 1)
        else:
            ll_aux = np.arange(l_max_bin[j - 1] + 1, l_max_bin[j] + 1)

        n_ell = len(ll_aux)

        # Prepare parallel tasks
        args = [
            (i, Cl[:, :, ll - l_min_tot], Cl_der[:, :, :, ll - l_min_tot], n_par)
            for i, ll in enumerate(ll_aux)
        ]

        trace_arr = np.zeros((n_ell, n_par, n_par))
        with ProcessPoolExecutor(max_workers=4) as executor:
            for i, trace in executor.map(fisher_trace_worker, args):
                trace_arr[i] = trace

        # Weighted sum over ℓ for this bin
        for a in range(n_par):
            for b in range(n_par):
                fisher[j, a, b] = np.sum(((2 * ll_aux + 1) / 2.0) * trace_arr[:, a, b])

        # Shrink the observable matrix dimensions for next bin
        Cl = Cl[:-1, :-1, :]
        Cl_der = Cl_der[:, :-1, :-1, :]

    fisher_tot = f_sky * np.sum(fisher, axis=0)
    return fisher_tot


def compute_single_derivative_gal(i, b_gal):
    print(f'\nComputing the derivative with respect to the galaxy bias in bin {i}...\n')

    def func_GG(b):
        b_gal_temp = np.copy(b_gal)
        b_gal_temp[i] = b
        return Cl_func(C,cosmo_params,gw_params,dl_GW,bin_edges_dl,z_gal,ll,b_gal= bias_gal_temp, b_GW = bias_GW)[0]

    def func_GWGW(b):
        b_gal_temp = np.copy(b_gal)
        b_gal_temp[i] = b
        return Cl_func(C,cosmo_params,gw_params,dl_GW,bin_edges_dl,z_gal,ll,b_gal= bias_gal_temp, b_GW = bias_GW)[1]

    def func_GGW(b):
        b_gal_temp = np.copy(b_gal)
        b_gal_temp[i] = b
        return Cl_func(C,cosmo_params,gw_params,dl_GW,bin_edges_dl,z_gal,ll,b_gal= bias_gal_temp, b_GW = bias_GW)[2]

    der_b_gal_GG = nd.Derivative(func_GG, step=step)(b_gal[i])
    der_b_gal_GWGW = nd.Derivative(func_GWGW, step=step)(b_gal[i])
    der_b_gal_GGW = nd.Derivative(func_GGW, step=step)(b_gal[i])

    der_b_gal_vec = fcc.vector_cl(cl_cross=der_b_gal_GGW, cl_auto1=der_b_gal_GG, cl_auto2=der_b_gal_GWGW)
    der_b_gal_cov_mat = fcc.covariance_matrix(der_b_gal_vec, n_bins_z, n_bins_dl)

    np.save(os.path.join(FLAGS.fout, f'der_b_gal_cov_mat_bin_{i}.npy'), der_b_gal_cov_mat)

    return i, der_b_gal_cov_mat

def compute_partial_derivatives_gal(b_gal, der_b_gal):
    args = [(i, b_gal) for i in range(len(b_gal))]

    with ProcessPoolExecutor() as executor:
        results = executor.map(lambda p: compute_single_derivative_gal(*p), args)

    for i, cov_mat in results:
        der_b_gal[i] = cov_mat

    return der_b_gal

def compute_partial_derivatives_gal(b_gal, der_b_gal,step):
    """
    Compute numerical derivatives of power spectra with respect to galaxy bias in each bin.

    Parameters:
    - b_gal: Array of galaxy bias values per bin
    - der_b_gal: Output array to store derivative covariance matrices

    Returns:
    - der_b_gal: Updated array with derivative covariance matrices per bias bin
    """

    for i in range(len(b_gal)):
        print('\nComputing the derivative with respect to the galaxy bias in bin %i...\n' % i)

        def func_GG(b):
            b_gal_temp = np.copy(b_gal)
            b_gal_temp[i] = b
            return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal_temp, b_GW=bias_GW)[0]

        def func_GWGW(b):
            b_gal_temp = np.copy(b_gal)
            b_gal_temp[i] = b
            return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal_temp, b_GW=bias_GW)[1]

        def func_GGW(b):
            b_gal_temp = np.copy(b_gal)
            b_gal_temp[i] = b
            return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal_temp, b_GW=bias_GW)[2]

        # Compute finite difference derivatives
        der_b_gal_GG = nd.Derivative(func_GG, step=step)(b_gal[i])
        der_b_gal_GWGW = nd.Derivative(func_GWGW, step=step)(b_gal[i])
        der_b_gal_GGW = nd.Derivative(func_GGW, step=step)(b_gal[i])

        # Construct and store covariance matrix
        der_b_gal_vec = fcc.vector_cl(cl_cross=der_b_gal_GGW, cl_auto1=der_b_gal_GG, cl_auto2=der_b_gal_GWGW)
        der_b_gal_cov_mat = fcc.covariance_matrix(der_b_gal_vec, n_bins_z, n_bins_dl)

        der_b_gal[i] = der_b_gal_cov_mat
        np.save(os.path.join(FLAGS.fout, 'der_b_gal_cov_mat_bin_%i.npy' % i), der_b_gal_cov_mat)

    return der_b_gal

def compute_partial_derivatives_GW(b_GW, der_b_GW,step):
    """
    Compute numerical derivatives of power spectra with respect to GW (gravitational wave) bias parameters in each bin.

    Parameters:
    - b_GW: Array of GW bias values per bin
    - der_b_GW: Output array to store derivative covariance matrices

    Returns:
    - der_b_GW: Updated array with derivative covariance matrices per bias bin
    """

    for i in range(len(b_GW)):
        print('\nComputing the derivative with respect to the GW bias in bin %i...\n' % i)

        # Define internal functions to return power spectra for modified bias
        def func_GG(b):
            b_GW_temp = np.copy(b_GW)
            b_GW_temp[i] = b
            return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal, b_GW=bias_GW_temp)[0]

        def func_GWGW(b):
            b_GW_temp = np.copy(b_GW)
            b_GW_temp[i] = b
            return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal, b_GW=bias_GW_temp)[1]

        def func_GGW(b):
            b_GW_temp = np.copy(b_GW)
            b_GW_temp[i] = b
            return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal, b_GW=bias_GW_temp)[2]

        # Compute numerical derivatives for each power spectrum
        der_b_GW_GG = nd.Derivative(func_GG, step=step)(b_GW[i])
        der_b_GW_GWGW = nd.Derivative(func_GWGW, step=step)(b_GW[i])
        der_b_GW_GGW = nd.Derivative(func_GGW, step=step)(b_GW[i])

        # Assemble derivative vector and covariance matrix
        der_b_GW_vec = fcc.vector_cl(cl_cross=der_b_GW_GGW, cl_auto1=der_b_GW_GG, cl_auto2=der_b_GW_GWGW)
        der_b_GW_cov_mat = fcc.covariance_matrix(der_b_GW_vec, n_bins_z, n_bins_dl)

        # Store result and save to file
        der_b_GW[i] = der_b_GW_cov_mat
        np.save(os.path.join(FLAGS.fout, 'der_b_GW_cov_mat_bin_%i.npy' % i), der_b_GW_cov_mat)

    return der_b_GW

def compute_single_derivative_GW(i, b_GW, step, FLAGS, n_bins_z, n_bins_dl):
    b_GW_temp = np.copy(b_GW)

    def func_GG(b):
        b_GW_temp[i] = b
        return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal, b_GW=bias_GW_temp)[0]

    def func_GWGW(b):
        b_GW_temp[i] = b
        return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal, b_GW=bias_GW_temp)[1]

    def func_GGW(b):
        b_GW_temp[i] = b
        return Cl_func(C, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal=bias_gal, b_GW=bias_GW_temp)[2]

    # Use finite difference manually (central)
    delta = step
    fGG_plus = func_GG(b_GW[i] + delta)
    fGG_minus = func_GG(b_GW[i] - delta)
    der_b_GW_GG = (fGG_plus - fGG_minus) / (2 * delta)

    fGWGW_plus = func_GWGW(b_GW[i] + delta)
    fGWGW_minus = func_GWGW(b_GW[i] - delta)
    der_b_GW_GWGW = (fGWGW_plus - fGWGW_minus) / (2 * delta)

    fGGW_plus = func_GGW(b_GW[i] + delta)
    fGGW_minus = func_GGW(b_GW[i] - delta)
    der_b_GW_GGW = (fGGW_plus - fGGW_minus) / (2 * delta)

    # Assemble and save
    der_vec = fcc.vector_cl(cl_cross=der_b_GW_GGW, cl_auto1=der_b_GW_GG, cl_auto2=der_b_GW_GWGW)
    der_cov = fcc.covariance_matrix(der_vec, n_bins_z, n_bins_dl)

    filename = os.path.join(FLAGS.fout, f'der_b_GW_cov_mat_bin_{i}.npy')
    np.save(filename, der_cov)

    return i, der_cov


def compute_partial_derivatives_GW(b_GW, der_b_GW, step, FLAGS, n_bins_z, n_bins_dl):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(compute_single_derivative_GW, i, b_GW, step, FLAGS, n_bins_z, n_bins_dl)
            for i in range(len(b_GW))
        ]

        for future in futures:
            i, cov_mat = future.result()
            der_b_GW[i] = cov_mat

    return der_b_GW


# -----------------------------------------------------------------------------------------
#            COMPUTING AND PLOTTING GW BIN DISTRIBUTION STATISTICS
# -----------------------------------------------------------------------------------------
# Compute the merger rate distribution and related quantities from luminosity distance bins
z_GW_new, bin_convert_new, ndl_GW_new, n_GW_new, merger_rate_tot_new = fcc.merger_rate_dl_new(
    dl=dl_GW,
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
    C=Hi_Cosmo,
    normalize=False
)

# Integrate the total merger rate over the full luminosity distance range (in Gpc)
n_tot_GW_new = trapezoid(merger_rate_tot_new, dl_GW / 1000) * 4 * np.pi
print('\n ratio n_tot_GW_new: ', (n_tot_GW_new/n_tot_GW)-1)

# Calculate the fraction of GW sources in each luminosity distance bin
bin_frac_GW_new = np.zeros(shape=n_bins_dl)
for i in range(n_bins_dl):
    bin_frac_GW_new[i] = trapezoid(ndl_GW_new[i], dl_GW / 1000)

# Sum all bin fractions to get the total number in bins (should match total GW if complete)
n_GW_bins_new = np.sum(bin_frac_GW_new)
print('\nratio n_GW_bins_new ', (n_GW_bins_new/n_GW_bins) -1)

fem.plot_gw_bin_distributions(
    dl_GW=dl_GW,
    ndl_GW=ndl_GW_new,
    merger_rate_tot=merger_rate_tot_new,
    bin_edges_dl=bin_edges_dl,
    n_bins_dl=n_bins_dl,
    output_path=FLAGS.fout
)
# Print per-bin and mean statistics for GW shot noise
print('\nratio bin_frac_GW', bin_frac_GW_new/bin_frac_GW -1)
shot_noise_GW_new = 1 / bin_frac_GW_new
print('\nratio shot_noise_GW_new', shot_noise_GW_new/shot_noise_GW -1)
print('\nratio bin_frac_GW: ', np.mean(bin_frac_GW_new)/np.mean(bin_frac_GW)-1)
print('\nratio shot_noise_GW : ', np.mean(shot_noise_GW_new)/np.mean(shot_noise_GW)-1)



def summarize_rel_diff(P_camb, P_class, kk, zz, FLAGS,
                       z_slices=(0.0, 0.5, 1.0, 2.0),
                       k_indices=(5, 20, 50)):
    """
    Quantify differences between two P(k,z) arrays on the SAME (z,k) grid.
    rel = P_camb / P_class - 1

    Saves plots in FLAGS.fout instead of displaying them.
    """

    # Ensure 2D arrays oriented as [nz, nk]
    Pc = np.asarray(P_camb)
    Pl = np.asarray(P_class)
    nz, nk = len(zz), len(kk)

    if Pc.shape == (nk, nz): Pc = Pc.T
    if Pl.shape == (nk, nz): Pl = Pl.T
    assert Pc.shape == (nz, nk) and Pl.shape == (nz,
                                                 nk), f"Shapes mismatch: {Pc.shape} vs {Pl.shape}, expected ({nz},{nk})"

    rel = Pc / Pl - 1.0
    finite = np.isfinite(rel)

    # ---------- Global stats ----------
    def summ(a):
        return dict(
            median=float(np.nanmedian(a)),
            mean=float(np.nanmean(a)),
            rms=float(np.nanmean(a ** 2) ** 0.5),
            p05=float(np.nanpercentile(a, 5)),
            p95=float(np.nanpercentile(a, 95)),
            p99=float(np.nanpercentile(a, 99)),
            amin=float(np.nanmin(a)),
            amax=float(np.nanmax(a)),
        )

    global_stats = summ(rel[finite])
    iz_max, ik_max = np.unravel_index(np.nanargmax(np.abs(rel)), rel.shape)
    max_info = dict(
        abs_rel_max=float(np.abs(rel[iz_max, ik_max])),
        rel_value=float(rel[iz_max, ik_max]),
        z_at_max=float(zz[iz_max]),
        k_at_max=float(kk[ik_max]),
        idx=(int(iz_max), int(ik_max))
    )

    per_z = [{'z': float(zz[iz]), **summ(rel[iz, :])} for iz in range(nz)]
    per_k = [{'k': float(kk[ik]), **summ(rel[:, ik])} for ik in range(nk)]

    stats = dict(global_stats=global_stats, max_info=max_info,
                 per_z=per_z, per_k=per_k)

    # ---------- Save plots ----------
    # Heatmap
    plt.figure(figsize=(7.5, 5.5))
    im = plt.imshow(rel, aspect='auto', origin='lower',
                    extent=[kk[0], kk[-1], zz[0], zz[-1]],
                    interpolation='nearest')
    cbar = plt.colorbar(im)
    cbar.set_label(r"$P_{\rm CAMB}/P_{\rm CLASS}-1$")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$z$")
    plt.title("Relative difference heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FLAGS.fout, "rel_diff_heatmap.png"), bbox_inches='tight')
    plt.close()

    # z-slices
    for zt in z_slices:
        iz = int(np.argmin(np.abs(zz - zt)))
        plt.figure(figsize=(7, 4.5))
        plt.plot(kk, rel[iz, :], lw=1.4)
        plt.axhline(0, lw=1, linestyle='--')
        plt.xlabel(r"$k$")
        plt.ylabel(r"$P_{\rm CAMB}/P_{\rm CLASS}-1$")
        plt.title(f"Rel diff at z ≈ {zz[iz]:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(FLAGS.fout, f"rel_diff_z{zz[iz]:.3f}.png"), bbox_inches='tight')
        plt.close()

    # k-slices
    for ik in k_indices:
        if 0 <= ik < nk:
            plt.figure(figsize=(7, 4.5))
            plt.plot(zz, rel[:, ik], lw=1.4)
            plt.axhline(0, lw=1, linestyle='--')
            plt.xlabel(r"$z$")
            plt.ylabel(r"$P_{\rm CAMB}/P_{\rm CLASS}-1$")
            plt.title(f"Rel diff at k={kk[ik]:.4g}")
            plt.tight_layout()
            plt.savefig(os.path.join(FLAGS.fout, f"rel_diff_k{kk[ik]:.4g}.png"), bbox_inches='tight')
            plt.close()

    # ---------- Console diagnostics ----------
    print("\n=== GLOBAL rel = P_CAMB / P_CLASS - 1 ===")
    for k, v in global_stats.items():
        print(f"{k:>6}: {v: .3%}")
    print("\nMax |rel| at (z, k) = ({:.3f}, {:.4g}) : rel = {:+.3%}".format(
        max_info['z_at_max'], max_info['k_at_max'], max_info['rel_value']
    ))

    return rel, stats
# rel, stats = summarize_rel_diff(P_vals_camb, P_vals, kk=kk_nl, zz=zz_nl, FLAGS=FLAGS, z_slices=(0.0, 0.5, 1.0, 2.0),k_indices=(5, 20, 50))


z_GW_old, bin_GW_converted_old, ndl_GW_old, _ , _ = fcc.merger_rate_dl(
    bg=bg,
    dl=dl_GW,  # Mpc
    bin_dl=bin_edges_dl,
    log_dl=log_dl,
    log_delta_dl=log_delta_dl,
    H0=H_0,
    omega_m=Omega_m,
    omega_b=Omega_b,
    A=A,
    Z_0=Z_0,
    Alpha=Alpha,
    Beta=Beta,
    C=Hi_Cosmo,
    normalize=False
)

def relative_diff_percent(new, old):
    new = np.array(new)
    old = np.array(old)
    # Avoid division by zero
    if np.any(old == 0):
        # Handle cases where old is zero, e.g., return a large value or NaN
        # Here, we replace division by zero with NaN to indicate undefined relative difference
        result = np.zeros_like(new, dtype=float)
        non_zero_mask = old != 0
        result[non_zero_mask] = (new[non_zero_mask] - old[non_zero_mask]) / old[non_zero_mask] * 100
        result[~non_zero_mask] = np.nan
        return result
    else:
        return (new - old) / old * 100


rel_hubble = relative_diff_percent(z_GW, z_GW_old)
rel_jac = relative_diff_percent(bin_GW_converted, bin_GW_converted_old)
rel_w = relative_diff_percent(ndl_GW, ndl_GW_old)


print("Differenze relative % z_GW:")
print(rel_hubble)
print("\nDifferenze relative % bin_GW_converted")
print(rel_jac)
print("\nDifferenze relative % ndl_GW:")
print(rel_w)

print('ndl_GW',ndl_GW)

print('info')
print('dimensions: z_GW',len(z_GW),'bin_GW_converted',len(bin_GW_converted),'ndl_GW',len(ndl_GW))
print('dimensions: z_GW',len(z_GW_old), 'bin_GW_converted', len(bin_GW_converted_old), 'ndl_GW', len(ndl_GW_old))



#-----------------------------------------------------------------------------------------
# 	DEFINE FIDUCIAL COSMOLOGICAL MODEL AND COMPUTE CORRESPONDING LUMINOSITY DISTANCES
#-----------------------------------------------------------------------------------------
# Luminosity distance interval, equal to the redshift one assuming fiducial cosmology
Hi_Cosmo = cc.cosmo(**cosmo_params)


def dL_from_C(Hi_Cosmo, z):
    """
    Luminosity distance d_L(z) in Mpc from your colibri cosmology `Hi_Cosmo`.
    Assumes Hi_Cosmo.comoving_distance(z) returns comoving distance in Mpc/h.
    """
    z = np.asarray(z, dtype=float)
    chi_Mpc = np.asarray(Hi_Cosmo.comoving_distance(z)) / Hi_Cosmo.h  # -> Mpc
    return (1.0 + z) * chi_Mpc

dlm_bin = dL_from_C(Hi_Cosmo, z_m_bin_GW)  # min d_L from C
dlM_bin = dL_from_C(Hi_Cosmo, z_M_bin_GW)  # max d_L from C

z_gal = np.linspace(z_m, z_M, 1200)  # Redshift grid for galaxy distribution
dl_GW = np.linspace(dlm, dlM, 1200)  # Luminosity distance grid for gravitational wave sources

#-----------------------------------------------------------------------------------------
#							BIN STRATEGY
#-----------------------------------------------------------------------------------------
def z_from_dL(Hi_Cosmo, dL_Mpc, z_max=10.0, ngrid=20001):
    """
    Invert d_L(z) -> z using a precomputed grid and linear interpolation.
    dL_Mpc: array-like in Mpc.
    Returns z with same shape as dL_Mpc.
    """
    # Build a monotonic (z, dL) grid
    z_grid = np.linspace(0.0, float(z_max), int(ngrid))
    dL_grid = dL_from_C(Hi_Cosmo, z_grid)  # [Mpc]

    # Ensure strict monotonicity for interp1d by uniquifying dL_grid
    # (d_L is monotonic in standard cosmologies; numerical noise can cause ties)
    order = np.argsort(dL_grid)
    dL_sorted = dL_grid[order]
    z_sorted = z_grid[order]
    # Drop any duplicates in dL_sorted
    mask = np.concatenate(([True], np.diff(dL_sorted) > 0))
    dL_unique = dL_sorted[mask]
    z_unique = z_sorted[mask]

    inv = interp1d(dL_unique, z_unique, kind='linear', bounds_error=False, fill_value='extrapolate',assume_sorted=True)

    return inv(np.asarray(dL_Mpc, dtype=float))

bin_int = np.linspace(z_m_bin, z_M_bin, n_bins_z * 1000)  # Fine redshift grid for binning
bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl * 1000)  # Fine luminosity distance grid for GW binning (in Gpc)

# Compute bin edges using the specified strategy and cosmology
bin_edges, bin_edges_dl = fem.compute_bin_edges_new(bin_strategy, n_bins_dl, n_bins_z, bin_int, z_M_bin, dlM_bin, z_m_bin, Hi_Cosmo, A, Z_0, Alpha, Beta, spline) # bin_edges_dl in Gpc (plain floats)

# convert luminosity-distance bin edges (Gpc) to redshift using C
dL_edges_Mpc = 1000.0 * np.asarray(bin_edges_dl,dtype=float)  #  Gpc -> Mpc
bin_z_fiducial = z_from_dL(Hi_Cosmo, dL_edges_Mpc)  # array of z edges

# Compute redshift distribution and total number of galaxies
nz_gal, gal_tot = fem.compute_nz_gal_and_total(gal_det, z_gal, bin_edges, sig_gal, gal_params['spline'])

gal_tot[gal_tot < 0] = 0  # Remove negative values (if any)
n_tot_gal = trapezoid(gal_tot, z_gal)  # Integrate total galaxy distribution

# Compute fraction of galaxies in each redshift bin
bin_frac_gal = np.zeros(shape=(n_bins_z))
for i in range(n_bins_z):
    bin_frac_gal[i] = simpson(nz_gal[i], z_gal)

shot_noise_gal= 1/bin_frac_gal
n_gal_bins = np.sum(bin_frac_gal)  # Sum of galaxy fractions across bins

# Save bin edges for later use
np.save(os.path.join(FLAGS.fout, 'bin_edges_GW_fiducial.npy'), bin_z_fiducial)
np.save(os.path.join(FLAGS.fout, 'bin_edges_GW.npy'), bin_edges_dl)
np.save(os.path.join(FLAGS.fout, 'bin_edges_gal.npy'), bin_edges)
np.save(os.path.join(FLAGS.out,'nz_gal.npy'),nz_gal)

#-----------------------------------------------------------------------------------------
#                PLOTTING THE GALAXY BIN DISTRIBUTION
#-----------------------------------------------------------------------------------------
fem.plot_galaxy_bin_distributions(z_gal, nz_gal, gal_tot, bin_edges, n_bins_z, z_m_bin, z_M_bin, FLAGS.fout)

# Print statistics about galaxy bins
print('\nthe total number of galaxies across all redshift: ', n_tot_gal * 4 * np.pi * f_sky)
print('\nthe total number of galaxies in our bins: ', n_gal_bins * 4 * np.pi * f_sky)
print('\nmean number of galaxies in each bin: ', np.mean(bin_frac_gal))
print('\nmean shot noise in each bin: ', np.mean(shot_noise_gal))

with open(os.path.join(FLAGS.fout, "galaxy_bin_distributions.txt"), "w") as f:
    f.write("Diagnostics for this run\n\n")
    f.write("z_gal ="+str(z_gal.tolist())+"\n")
    f.write("nz_gal ="+str(nz_gal.tolist())+"\n")
    f.write("gal_tot = " + str(gal_tot.tolist()) + "\n")
    f.write("bin_frac_gal = " + str(bin_frac_gal.tolist()) + "\n")
    f.write("shot_noise_gal = " + str(shot_noise_gal.tolist()) + "\n")
    f.write("n_bins_z         = " + str(n_bins_z) + "\n")
    f.write("z_m_bin            = " + str(z_m_bin) + "\n")
    f.write("z_M_bin     = " + str(z_M_bin) + "\n")
    f.write("the total number of galaxies across all redshift  = " + str(n_tot_gal * 4 * np.pi * f_sky) + "\n")
    f.write("the total number of galaxies in our bins     = " + str(n_gal_bins * 4 * np.pi * f_sky) + "\n")
    f.write("mean number of galaxies in each bin     = " + str(np.mean(bin_frac_gal)) + "\n")
    f.write("mean shot noise in each bin    = " + str(np.mean(shot_noise_gal)) + "\n")

print("\nDiagnostics saved!")

#-----------------------------------------------------------------------------------------
# 		DETERMINE REPRESENTATIVE REDSHIFTS AND COMPUTE NONLINEAR POWER SPECTRUM
#-----------------------------------------------------------------------------------------
# Initialize array to store the peak redshift of each galaxy bin
redshift = np.zeros(shape=n_bins_z)
for i in range(n_bins_z):
    a = np.argmax(nz_gal[i])  # Index of maximum value in the redshift distribution
    redshift[i] = z_gal[a]  # Assign corresponding redshift

# Define k and z arrays for evaluating the nonlinear power spectrum
kk_nl = np.geomspace(1e-4, 1e2, 200)  # Logarithmically spaced k values [h/Mpc]
zz_nl = np.linspace(z_m_bin_GW, z_M_bin_GW, 100)  # Linearly spaced redshift values

# 	#_, P_vals_camb = Hi_Cosmo.camb_Pk(z=zz_nl, k=kk_nl, nonlinear=True, halofit='mead2020')
#print('\n prima:',P_vals,'\n')

# Compute nonlinear matter power spectrum using HI_CLASS
_,kk_nl,zz_nl,P_vals = Hi_Cosmo.hi_class_pk(cosmo_params, kk_nl, zz_nl, True) # in (Mpc/h)^3

#print('\n dopo:',P_vals,'\n')
#print((P_vals_camb.T / P_vals) - 1)
#print('\nbackground',bg)

# Interpolate power spectrum over redshift and k
P_interp = RectBivariateSpline(kk_nl, zz_nl, P_vals)

# Use peak redshifts of bins as centers for computing k_max
z_centers_use = redshift

# Compute maximum usable wavenumber at each redshift bin center
k_max = fem.compute_k_max(z_centers_use, P_interp, kk_nl)

#print('\nz_centers_use=',  z_centers_use.tolist())
#print('\nk_max=', k_max.tolist())

z_centers_use = np.asarray(z_centers_use, float).ravel()
k_max = np.asarray(k_max, float).ravel()

#-----------------------------------------------------------------------------------------
#			COMPUTING MULTIPOLE LIMITS AND GW BIN DISTRIBUTION STATISTICS
#-----------------------------------------------------------------------------------------
# Compute maximum multipole l for each bin using comoving distance and k_max
chi_C_hMpc = np.asarray(Hi_Cosmo.comoving_distance(z_centers_use), float).ravel()
ell_C = chi_C_hMpc * k_max
l_max_nl = np.rint(ell_C).astype(int)

#print('\nl_max_nl=',l_max_nl.tolist())

# sanitize edges: 1-D, finite, strictly increasing, unique, and covering data range
edges = np.asarray(bin_edges_dl, float).ravel()
edges = edges[np.isfinite(edges)]
edges = np.unique(edges)  # sorted + deduplicated

# ensure edges cover the data range (recommended)
dl_vals = np.power(10.0, np.asarray(log_dl, float).ravel())  # Gpc if log_dl=log10(Gpc)
emin, emax = edges[0], edges[-1]
dmin, dmax = dl_vals.min(), dl_vals.max()

eps = 1e-12
if dmin < emin:
    edges = np.insert(edges, 0, dmin * (1 - eps))
if dmax > emax:
    edges = np.append(edges, dmax * (1 + eps))

assert np.all(np.diff(edges) > 0), "bin edges must be strictly increasing"

# diagnostics
#counts, _ = np.histogram(dl_vals, bins=edges)
#print("empty bin indices:", np.where(counts == 0)[0])

# Compute localization error parameters for GW bins
# now call your function with the sanitized edges
bin_edges_dl=edges
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

# Define multipole vector with increasing step sizes at higher l
ll = np.sort(np.unique(np.concatenate([
    np.arange(l_min, 20, step=2),
    np.arange(20, 50, step=5),
    np.arange(50, 100, step=10),
    np.arange(100, l_max + 1, step=25)])))
ll[-1] = l_max  # Ensure maximum l is included
ll_total = np.arange(l_min, l_max + 1)

# Compute normalization factor for Cl's
c = ll * (ll + 1.) / (2. * np.pi)

# Save computed arrays
np.save(os.path.join(FLAGS.fout, 'ell_max.npy'), l_max_bin)
np.save(os.path.join(FLAGS.fout, 'loc_nl.npy'), loc_or_nl)

#-----------------------------------------------------------------------------------------
#            COMPUTING AND PLOTTING GW BIN DISTRIBUTION STATISTICS
#-----------------------------------------------------------------------------------------
# Compute the merger rate distribution and related quantities from luminosity distance bins
z_GW, bin_convert, ndl_GW, n_GW, merger_rate_tot = fcc.merger_rate_dl(
    dl=dl_GW,
    bin_dl=bin_edges_dl,
    log_dl=log_dl,
    log_delta_dl=log_delta_dl,
    A=A,
    Z_0=Z_0,
    Alpha=Alpha,
    Beta=Beta,
    C=Hi_Cosmo,
    normalize=False)



def rel_change(new, old):
	new = np.asarray(new, dtype=float)
	old = np.asarray(old, dtype=float)
	out = np.full_like(new, np.nan, dtype=float)
	with np.errstate(divide='ignore', invalid='ignore'):
		np.divide(new, old, out=out, where=(old != 0))
	return out - 1.0


def report_ratio(name, new, old):
	r = rel_change(new, old)  # può essere 1D/2D/3D
	rabs = np.abs(r)
	finite = np.isfinite(rabs)
	print(f"\nratio {name}", r)
	if finite.any():
		print(f"max |ratio| {name}", np.nanmax(rabs))
	else:
		print(f"max |ratio| {name}: all-NaN")


# ---- i tuoi print diventano:
# report_ratio('z_GW', z_GW_new, z_GW)
# report_ratio('bin_GW_converted', bin_GW_converted_new, bin_GW_converted)
# report_ratio('ndl_GW', ndl_GW_new, ndl_GW)  # 2D ok
# report_ratio('n_GW', n_GW_new, n_GW)
# report_ratio('total', total_new, total)

'
# Compute both versions first
S.load_galaxy_clustering_window_functions(bg,h=h, z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal, name='bg')
S.load_galaxy_clustering_window_functions_old(z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal, name='old')

# Then compare
S.compare_galaxy_windows(z=z_gal, ll=ll, outdir=FLAGS.fout,save_overlays=False, save_rel=False)

# Call both loaders first
S.load_gravitational_wave_window_functions(bg,C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, bias=b_GW, name='bg')
S.load_gravitational_wave_window_functions_old(z=z_GW, ndl=ndl_GW, ll=ll, bias=b_GW, H0=H_0, omega_m=Omega_m, omega_b=Omega_b, name='old')

# Compare them
S.compare_gw_windows(z=z_GW, ll=ll, outdir=FLAGS.fout, prefix="GW_W",name_bg='bg', name_legacy='old',save_overlays=False, save_rel=False)


S.compare_galaxy_windows_save(bg=bg, z=z_gal, nz=nz_gal, ll=ll, bias=b_gal,
	outdir=FLAGS.fout, prefix="W_G",
	save_overlays=True, save_rel=True
)
S.compare_gw_windows_save(bg=bg, z=z_GW, ndl=ndl_GW, ll=ll, bias=b_GW,
	H0=H_0, omega_m=Omega_m, omega_b=Omega_b,
	outdir=FLAGS.fout, prefix="W_GW",
	save_overlays=True, save_rel=True

)

S.limber_angular_power_spectra_old( l=ll, windows=None)
S.limber_angular_power_spectra(bg, h=h, l=ll, windows=None)

# Call after windows are loaded and power_spectra_interpolator is set
S.compare_angular_power_spectra(
	bg=bg, h=h, ell=ll,
	outdir=FLAGS.fout, prefix="Cl", spectra=["galaxy", "GW"]
)


# -----------------------------------------------------------------------------------------
# GALAXY LENSING WINDOW FUNCTION
# -----------------------------------------------------------------------------------------

def load_galaxy_lensing_window_functions_old(self, z, n_z, H_0, omega_m, ll, name='galaxy_lensing'):
    constant = 3. * omega_m * (H_0 / const.c) ** 2.
    nz = np.array(n_z)
    z = np.array(z)
    n_bins = len(nz)
    norm_const = simpson(nz, x=z, axis=1)

    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        tmp_interp = si.interp1d(z, nz[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        self.window_function[name].append(
            si.interp1d(self.z_integration, constant * tmp_interp(self.z_integration) / norm_const[galaxy_bin],
                        'cubic', bounds_error=False, fill_value=0.))

    self.window_function[name] = np.tile(np.array(self.window_function[name]).reshape(-1, 1), (1, len(ll)))


# -----------------------------------------------------------------------------------------
# RSD WINDOW FUNCTION
# -----------------------------------------------------------------------------------------
def load_rsd_window_functions_old(self, z, nz, H_0, omega_m, omega_b, ll, name='RSD'):

    cosmo = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b)

    nz = np.array(nz)
    z = np.array(z)
    n_bins = len(nz)
    norm_const = simpson(nz, x=z, axis=1)
    assert np.all(np.diff(
        z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (
        self.dz_windows)
    assert nz.ndim == 2, "'nz' must be 2-dimensional"
    assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
    assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (
        self.z_min)
    assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (
        self.z_max)

    def L0(ll):
        return (2 * ll ** 2 + 2 * ll - 1) / ((2 * ll - 1) * (2 * ll + 3))

    def Lm1(ll):
        return -ll * (ll - 1) / ((2 * ll - 1) * np.sqrt((2 * ll - 3) * (2 * ll + 1)))

    def Lp1(ll):
        return -(ll - 1) * (ll + 2) / ((2 * ll + 3) * np.sqrt((2 * ll + 1) * (2 * ll + 5)))

    # Initialize window
    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        nz_interp = si.interp1d(z, nz[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        f_interp = si.interp1d(z, cosmo.Om(z) ** 0.55, 'cubic', bounds_error=False, fill_value=0.)

        def Wm1(ll):
            return Lm1(ll) * nz_interp(((2 * ll + 1 - 4) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1 - 4) / (2 * ll + 1)) * self.z_integration)

        def Wz(ll):
            return L0(ll) * nz_interp(((2 * ll + 1) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1) / (2 * ll + 1)) * self.z_integration)

        def Wp1(ll):
            return Lp1(ll) * nz_interp(((2 * ll + 1 + 4) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1 + 4) / (2 * ll + 1)) * self.z_integration)

        self.window_function[name].append([si.interp1d(self.z_integration,
                                                       (Wm1(l) + Wz(l) + Wp1(l)) * self.Hubble / const.c /
                                                       norm_const[galaxy_bin], 'cubic', bounds_error=False,
                                                       fill_value=0.) for l in ll])

    self.window_function[name] = np.array(self.window_function[name])

# -----------------------------------------------------------------------------------------
# GRAVITATIONAL WAVES CLUSTERING WINDOW FUNCTION
# -----------------------------------------------------------------------------------------
def load_gravitational_wave_window_functions_old(self, z, ndl, H0, omega_m, omega_b, ll, bias=1.0, name='GW'):

    conversion = FlatLambdaCDM(H0=H0, Om0=omega_m, Ob0=omega_b,Tcmb0=2.7255)

    ndl = np.array(ndl)
    z = np.array(z)
    n_bins = len(ndl)

    norm_const = simpson(ndl, x=conversion.luminosity_distance(z).value, axis=1)
    assert np.all(np.diff(
        z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (
        self.dz_windows)
    assert ndl.ndim == 2, "'nz' must be 2-dimensional"
    assert (ndl.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
    assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (
        self.z_min)
    assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (
        self.z_max)

    if isinstance(bias, float):
        bias = bias * np.ones(n_bins)
    elif isinstance(bias, int):
        bias = np.float(bias) * np.ones(n_bins)
    else:
        assert len(bias) == n_bins, "Number of bias factors different from number of bins"

    # conversion.luminosity_distance in [Mpc]; const.c / self.Hubble in [h/Mpc]
    jac_dl=(1 + self.z_integration) * const.c / self.Hubble + (conversion.luminosity_distance(self.z_integration).value) / (1 + self.z_integration)
    #print('\n GW OLD:')
    #print('Hubble [h/Mpc] ', self.Hubble/const.c)
    #print('r_n_z [Mpc/h]', conversion.luminosity_distance(self.z_integration).value)
    #print('jac_dL mix', jac_dL.tolist())
    #print('norm',norm_const)

    # Initialize window
    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        tmp_interp = si.interp1d(z, ndl[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration) * (jac_dl ) * self.Hubble / const.c / norm_const[galaxy_bin] * bias[galaxy_bin], 'cubic', bounds_error=False,fill_value=0.))

    self.window_function[name] = np.tile(np.array(self.window_function[name]).reshape(-1, 1), (1, len(ll)))

# -----------------------------------------------------------------------------------------
# LSD WINDOW FUNCTION
# -----------------------------------------------------------------------------------------
def load_lsd_window_functions_old(self, z, ndl, H_0, omega_m, omega_b, ll, name='LSD'):

    conversion = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b,Tcmb0=2.7255)

    ndl = np.array(ndl)
    z = np.array(z)
    n_bins = len(ndl)

    #print('omega_z OLD',conversion.Om(z))

    conf_H = (conversion.H(z).value / (1 + z))*1/(0.6781) *1/(const.c) # h/Mpc
    r_conf_H = (conversion.comoving_distance(z).value*0.6781) * conf_H # Mpc/h
    gamma = r_conf_H / (1 + r_conf_H)

    jj = ((1 + z) *(const.c*0.6781) / conversion.H(z).value + (conversion.luminosity_distance(z).value*0.6781) *1/ (1 + z)) # Mpc/h

    norm_const = simpson(ndl, x=conversion.luminosity_distance(z).value, axis=1)

    #print('conf_H OLD', conf_H)
    #print('self.Hubble / const.c',self.Hubble / const.c)
    #print('r_conf_H OLD', r_conf_H)
    #print('gamma OLD', gamma)
    #print('jj OLD', jj)
    #print('norm_const OLD', norm_const)


    assert np.all(np.diff(
        z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (
        self.dz_windows)
    assert ndl.ndim == 2, "'nz' must be 2-dimensional"
    assert (ndl.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
    assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (
        self.z_min)
    assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (
        self.z_max)

    def L0(ll):
        return (2 * ll ** 2 + 2 * ll - 1) / ((2 * ll - 1) * (2 * ll + 3))

    def Lm1(ll):
        return -ll * (ll - 1) / ((2 * ll - 1) * np.sqrt((2 * ll - 3) * (2 * ll + 1)))

    def Lp1(ll):
        return -(ll - 1) * (ll + 2) / ((2 * ll + 3) * np.sqrt((2 * ll + 1) * (2 * ll + 5)))

    # Initialize window
    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        nz_interp = si.interp1d(z, ndl[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        f_interp = si.interp1d(z, conversion.Om(z) ** 0.55 * 2 * gamma * jj, 'cubic', bounds_error=False,fill_value=0.)

        def Wm1(ll):
            return Lm1(ll) * nz_interp(((2 * ll + 1 - 4) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1 - 4) / (2 * ll + 1)) * self.z_integration)

        def Wz(ll):
            return L0(ll) * nz_interp(((2 * ll + 1) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1) / (2 * ll + 1)) * self.z_integration)

        def Wp1(ll):
            return Lp1(ll) * nz_interp(((2 * ll + 1 + 4) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1 + 4) / (2 * ll + 1)) * self.z_integration)

        # self.Hubble / const.c in h/Mpc
        self.window_function[name].append([si.interp1d(self.z_integration,
                                                       (Wm1(l) + Wz(l) + Wp1(l)) * self.Hubble / const.c /
                                                       norm_const[galaxy_bin], 'cubic', bounds_error=False,
                                                       fill_value=0.) for l in ll])

    self.window_function[name] = np.array(self.window_function[name])



#-----------------------------------------------------------------------------------------
# ANGULAR POWER SPECTRA CROSS CORRELATION WITH LENSING
#-----------------------------------------------------------------------------------------

def limber_angular_power_spectra_lensing_cross(self, bg, l, s_gal, beta, windows=None, n_points=20, n_points_x=20,
                                                   grid_x='mix', n_low=5, n_high=5, Delta_z=0.05, z_min=1e-05):
        # Check existence of power spectrum
        try:
            self.power_spectra_interpolator
        except AttributeError:
            raise AttributeError("Power spectra have not been loaded yet")

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min * self.geometric_factor_f_K(
            self.z_min), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max * self.geometric_factor_f_K(
            self.z_max), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Check window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:
                    windows_to_use[key] = self.window_function[key]
                except KeyError:
                    raise KeyError("Requested window function '%s' not known" % key)

        # Check window functions
        keys = windows_to_use.keys()
        if len(keys) == 0: raise AttributeError("No window function has been computed!")
        n_keys = len(keys)
        n_bins = [len(windows_to_use[key]) for key in keys]

        # 1) Define lengths and quantities
        # zz       = self.z_integration
        n_l = len(np.atleast_1d(l))
        # n_z      = self.n_z_integration # this is now n_points

        Cl = {}
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                Cl['%s-%s' % (ki, kj)] = np.zeros((n_bins[i], n_bins[j], n_l))

        z_interp_bias = np.linspace(z_min, 10, 2000)

        def exponential_value(x, alpha_interp):
            def single_exp(x_single):
                integrand = lambda z_: 0.5 * alpha_interp(z_) / (1 + z_)
                integral_val, _ = sint.quad(integrand, 0, x_single, epsabs=1e-6)
                return np.exp(integral_val)

            # Apply scalar integration to each element
            x = np.atleast_1d(x)
            return np.array([single_exp(val) for val in x])

        z_bg = bg['z']
        H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate")  # 1/Mpc
        comoving_distance_interp = interp1d(z_bg, bg["comov. dist."], kind='cubic', bounds_error=False,
                                            fill_value="extrapolate")  # Mpc
        alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False,
                                  fill_value="extrapolate")

        def JJ(tracer, x, H_interp, comoving_distance_interp, alpha_M_interp):

            if 'gal' in tracer:
                return np.ones(len(x))
            else:
                result = []
                for xi in x:
                    H_x = H_interp(xi)  # 1/Mpc
                    r_x = comoving_distance_interp(xi)  # Mpc
                    exp_val = exponential_value(xi, alpha_M_interp)
                    alphaM_x = alpha_M_interp(xi)

                    val = ((1 + xi) / H_x + r_x) * exp_val + r_x * exp_val * 0.5 * alphaM_x
                    result.append(val)

                # print('JJ', result)
                return np.array(result)

        def A_L(chi, z, tracer, r_z1, bs, H_vals):
            """
            Accepts chi,z,bs,H_vals as (Nz2, Nz1) or (1, Nz2, Nz1).
            Returns array shaped (Nz1, Nz2) so it broadcasts with r_z1 (Nz1,1).
            """
            chi = np.asarray(chi)
            z = np.asarray(z)
            bs = np.asarray(bs)
            H_vals = np.asarray(H_vals)

            # print('r_z1',r_z1)
            # print('H_vals',H_vals)

            # If inputs are 3D like (1, Nz2, Nz1), squeeze to (Nz2, Nz1)
            if chi.ndim == 3:
                chi = np.squeeze(chi, axis=0)
                z = np.squeeze(z, axis=0)
                bs = np.squeeze(bs, axis=0)
                H_vals = np.squeeze(H_vals, axis=0)

            # Now we expect (Nz2, Nz1). Transpose to (Nz1, Nz2)
            if chi.ndim == 2:
                chi = chi.T
                z = z.T
                bs = bs.T
                H_vals = H_vals.T
            else:
                raise ValueError("A_L expects 2D or 3D chi.")

            # Ensure bs and H_vals broadcast to (Nz1, Nz2)
            bs = np.broadcast_to(bs, chi.shape)
            H_vals = np.broadcast_to(H_vals, chi.shape)

            if 'gal' in tracer:
                return 0.5 * (5.0 * bs - 2.0) * (r_z1 - chi) / chi  # (Nz1, Nz2)
            else:
                conf_H = H_vals / (1.0 + z)  # 1/Mpc
                return 0.5 * (((r_z1 - chi) / chi) * (bs - 2.0) + 1.0 / (1.0 + r_z1 * conf_H))

        # 2) Load power spectra
        power_spectra = self.power_spectra_interpolator

        # Add curvature correction (see arXiv:2302.04507)
        if self.cosmology.K != 0.:
            KK = self.cosmology.K
            factor = np.zeros((n_l, n_z))
            for il, ell in enumerate(l):
                factor[il] = (1 - np.sign(KK) * ell ** 2 / (((ell + 0.5) / self.geometric_factor) ** 2 + KK)) ** -0.5
            PS_lz *= factor

        def lensing_int(H_interp, comoving_distance_interp, alpha_M_interp, z1, z2, ll, bin_i, bin_j, WX_, WY_, tX, tY,
                        b2, n_points=30, p=1, z_min=1e-04, grid='mix', show_plot=False):

            r_1 = comoving_distance_interp(z1)  # (Nz1,) Mpc
            z2 = np.asarray(z2)  # (Nz2, Nz1)
            r_2 = comoving_distance_interp(z2)  # (Nz2, Nz1) Mpc
            H_x = H_interp(z2)  # (Nz2, Nz1) #1/Mpc

            # print('DOPO')
            # print('r_1',r_1)
            # print('z2',z2)
            # print('r_2',r_2)
            # print('H_x',H_x)

            # JJ needs to accept 2D; it already loops over x, so np.asarray works
            JJ_vals = JJ(tY, z2, H_interp, comoving_distance_interp, alpha_M_interp)  # (Nz2, Nz1)

            A_vals = A_L(r_2, z2, tY, r_1[:, None], b2(z2), H_x)  # all 2D-ish

            # z2 is (Nz2, Nz1)
            z2T = np.asarray(z2).T  # -> (Nz1, Nz2)
            JJ_T = np.asarray(JJ_vals).T  # -> (Nz1, Nz2)

            # print('QUA')
            # print('A_L', A_vals)
            # print('JJ',JJ_T)

            # A_vals already (Nz1, Nz2)
            t1 = A_vals * (1.0 + z2T) * JJ_T  # all (Nz1, Nz2)
            # print('t1 new',t1)

            if show_plot:
                idx = 9
                plt.plot(z1, WX_)
                plt.scatter(z1, WX_)
                plt.plot(z2[:, idx], WY_[:, idx], ls='--')
                plt.scatter(z2[:, idx], WY_[:, idx], ls='--')
                plt.axvline(z1[idx])
                plt.title(str(bin_i) + str(bin_j) + ', ' + tX + '-' + tY)
                plt.yscale('log')
                plt.show()
                plt.close()

                '''
                plt.plot( x[:, idx, idx], t1[idx, idx, :] )
                plt.scatter( x[:, idx, idx], t1[idx, idx, :] )
                plt.yscale('log')
                plt.xscale('log')
                plt.show()
                plt.close()

                plt.plot(x[:, idx, idx], t1[idx, idx, :] )
                plt.scatter(x[:, idx, idx], t1[idx, idx, :] )
                #plt.yscale('log')
                #plt.xscale('log')
                plt.show()
                plt.close()
                '''

            # Vectorized power spectrum lookup expecting points as (k, z)
            Nl = len(ll)
            Nz1 = len(z1)
            Nz2 = z2.shape[0]  # since z2 is (Nz2, Nz1)

            PS_list = []
            for lidx in range(Nl):
                k = (ll[lidx] + 0.5) / r_2  # (Nz2, Nz1)
                # NOTE: (k, z) order for the interpolator
                pts = np.column_stack([k.ravel(), z2.ravel()])  # (Nz2*Nz1, 2)
                P_flat = power_spectra(pts)  # (Nz2*Nz1,)
                P = P_flat.reshape(z2.shape)  # (Nz2, Nz1)
                PS_list.append(P.T)  # -> (Nz1, Nz2) to match t1
            PS_ = np.stack(PS_list, axis=0)  # (Nl, Nz1, Nz2)

            assert t1.shape == (Nz1, Nz2)
            assert PS_.shape == (Nl, Nz1, Nz2)

            t1L = np.repeat(t1[None, :, :], len(ll), axis=0)  # (Nl, Nz1, Nz2)
            Nl, Nz1, Nz2 = t1L.shape

            # Normalize WY_ to (Nl, Nz1, Nz2)
            if WY_.ndim == 2:
                # Expect (Nl, Nz2): same for all z1 rows; lift along Nz1
                if WY_.shape != (Nl, Nz2):
                    raise ValueError(f"Unexpected WY_ shape {WY_.shape}; expected (Nl,Nz2)=({Nl},{Nz2})")
                WY_ = np.broadcast_to(WY_[:, None, :], (Nl, Nz1, Nz2))
            elif WY_.ndim == 3:
                # Could be (Nl, Nz2, Nz1) or already (Nl, Nz1, Nz2)
                if WY_.shape == (Nl, Nz2, Nz1):
                    WY_ = np.transpose(WY_, (0, 2, 1))  # -> (Nl, Nz1, Nz2)
                elif WY_.shape != (Nl, Nz1, Nz2):
                    raise ValueError(f"Unexpected WY_ shape {WY_.shape}; expected (Nl,Nz1,Nz2)=({Nl},{Nz1},{Nz2})")
            else:
                raise ValueError(f"WY_ ndim={WY_.ndim} not supported")

            # PS_ is already (Nl, Nz1, Nz2); keep or broadcast if you want symmetry
            PS_ = np.broadcast_to(PS_, t1L.shape)

            assert WY_.shape == t1L.shape == PS_.shape, (WY_.shape, t1L.shape, PS_.shape)

            # print('DOPO')
            # print('PS_', PS_)
            # print('WY_', WY_)

            my_int = t1L * WY_ * PS_

            I1_ = np.asarray([
                [trapezoid(my_int[l, i1], x=z2T[i1], axis=0) for i1 in range(Nz1)]
                for l in range(Nl)
            ])

            '''
            if show_plot:
                plt.plot(x[:, idx, idx], my_int[15, idx, idx, :] )
                plt.scatter(x[:, idx, idx], my_int[15, idx, idx, :] )
                plt.yscale('log')
                plt.xscale('log')
                plt.show()
                plt.close()

                plt.plot(x[:, idx, idx], my_int[15, idx, idx, :] )
                plt.scatter(x[:, idx, idx], my_int[15, idx, idx, :] )
                plt.show()
                plt.close()
            '''
            # I2_ =  trapezoid( WX_ * I1_/r_1 * const.c / H_interp(z1), x=z1, axis=1 )
            # print('WX new', WX_)
            # print('I1 new', I1_)
            I2_ = trapezoid(WX_ * I1_ / r_1 * 1 / H_interp(z1), x=z1, axis=1)
            # print('I2 new',I2_)
            return I2_

        # 3) load Cls given the source functions
        # 1st key (from 1 to N_keys)

        b_interp_gal = si.interp1d(z_interp_bias, s_gal, 'cubic', bounds_error=False, fill_value=0.)
        b_interp_beta = si.interp1d(z_interp_bias, beta, 'cubic', bounds_error=False, fill_value=0.)

        for index_X in xrange(n_keys):
            key_X = list(keys)[index_X]

            # choose z grid in each bin so that it is denser around the peak

            bins_ = self.bin_edges[key_X]
            # bins_centers_ = (bins_[:-1] + bins_[1:]) / 2
            zzs = []
            ll_ = 0
            for bin_i in xrange(n_bins[index_X]):
                if 'gal' in key_X:
                    max_z = 5  # check this
                    n_pts = n_high
                else:
                    max_z = 10  # check this
                    n_pts = 2 * n_high

                '''
                if max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01) == z_min:
                    n1 = n_points + n_low
                else:
                    n1 = n_points
                '''
                if bin_i < n_bins[index_X]:
                    my_arr = np.sort(np.unique(np.concatenate(
                        [np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)), bins_[bin_i + 1] * (1 + 5 * Delta_z),
                                     n_points),
                         np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low),
                         np.linspace(bins_[bin_i + 1] * (1 + 0.05) + 0.01, max_z, n_pts)])))
                else:
                    my_arr = np.sort(np.unique(np.concatenate(
                        [np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)), max_z, n_points + n_pts),
                         np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low)])))
                l_ = len(my_arr)
                if l_ > ll_:
                    ll_ = l_
                zzs.append(my_arr)

            for i, a in enumerate(zzs):
                # print('len %s: %s'%(i, len(a)))
                if not len(a) == l_:
                    n_ = l_ - len(a)
                    # print('adding %s'%n_)
                    zzs[i] = np.sort(
                        np.unique(np.concatenate([zzs[i], np.linspace(z_min * (1 + 0.01), max(a) * (1 - 0.01), n_)])))
                    # print('new len %s: %s'%(i, len(zzs[i])))
            try:
                zzs = np.asarray(zzs)
            except Exception as e:
                print(zzs)
                print(z_max)
                print(e)

            # now compute window functions
            W_X = np.array([[windows_to_use[key_X][i, j](zzs[i]) for j in range(n_l)] for i in range(n_bins[index_X])])

            # 2nd key (from 1st key to N_keys)
            for index_Y in xrange(n_keys):
                key_Y = list(keys)[index_Y]

                # Symmetry C_{AA}^{ij} == C_{AA}^{ji}
                # NOW THIS IS NOT TRUE FOR LENSING!
                for bin_i in xrange(n_bins[index_X]):
                    my_range = xrange(n_bins[index_Y])
                    for bin_j in my_range:
                        if (bin_j == bin_i):  # and (key_Y==key_X): # and ('gal' in key_Y) and (bin_j<=2)
                            # print( 'computing %s %s %s %s'%(key_Y, key_X, bin_i, bin_j))
                            # this could also made finer by adapting the grid.
                            # For now we leave as it is
                            z2s_ = np.linspace(z_min, zzs[bin_i], n_points)
                            W_Y = np.array([[windows_to_use[key_Y][i, j](z2s_) for j in range(n_l)] for i in
                                            range(n_bins[index_Y])])

                            WY = W_Y[bin_j]
                            WX = W_X[bin_i]
                            b2 = b_interp_gal if 'gal' in key_Y else b_interp_beta

                            Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j, :] = l * (l + 1) / (
                                    (l + 0.5) ** 2) * lensing_int(H_interp, comoving_distance_interp, alpha_M_interp,
                                                                  zzs[bin_i], z2s_, l, bin_i, bin_j, WX, WY,
                                                                  key_X, key_Y, b2, z_min=z_min, n_points=n_points_x,
                                                                  grid=grid_x)

        return Cl

'''
def capture_summary(diff_dict, ref_a, ref_b, tol=0.0, autocorr_only=False, only_nonzero=True):
    lines = []
    for k, D in diff_dict.items():
        A = np.asarray(ref_a[k], dtype=float)
        B = np.asarray(ref_b[k], dtype=float)
        I, J, L = D.shape
        header_written = False
        for i in range(I):
            for j in range(J):
                if autocorr_only and i != j:
                    continue
                rd = D[i, j, :]
                max_rel = np.nanmax(np.abs(rd))
                if only_nonzero and not (max_rel > tol):
                    continue
                if not header_written:
                    lines.append(f"\n[{k}]  bins={I}x{J}, L={L}")
                    header_written = True
                a = A[i, j, :];
                b = B[i, j, :];
                d = a - b
                max_abs = np.nanmax(np.abs(d))
                rms_rel = np.sqrt(np.nanmean(rd ** 2))
                na = np.sqrt(np.dot(a, a));
                nb = np.sqrt(np.dot(b, b))
                cos = (np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else np.nan
                lines.append(f"  bin({i},{j}): max|Δ|={max_abs:.3e}  max|rel|={max_rel:.3e}  "
                             f"RMS_rel={rms_rel:.3e}  cos={cos:.6f}")
    return "\n".join(lines)
'''

#with open(os.path.join(FLAGS.fout, "summary_diff_all.txt"), "w") as f:	f.write(capture_summary(diff_all, Cl_lens_cross, Cl_lens_cross_old, tol=0.0, autocorr_only=False, only_nonzero=True))

#with open(os.path.join(FLAGS.fout, "summary_diff_auto.txt"), "w") as f:
#	f.write(capture_summary(diff_auto, Cl_lens_cross, Cl_lens_cross_prima, tol=0.0, autocorr_only=True, only_nonzero=True))

def rel_diff_same_keys(Cl_a, Cl_b, eps=1e-12):
    """
    Compute relative difference only for keys that are exactly equal (no symmetries).
    """
    out = {}
    common = sorted(set(Cl_a.keys()) & set(Cl_b.keys()))
    if not common:
        raise ValueError(f"No strictly matching keys! A: {list(Cl_a.keys())}, B: {list(Cl_b.keys())}")

    for k in common:
        A = np.asarray(Cl_a[k], float)
        B = np.asarray(Cl_b[k], float)

        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch for {k}: {A.shape} vs {B.shape}")

        denom = np.where(np.abs(A) > eps, A,
                         np.where(np.abs(B) > eps, B, 1.0))
        out[k] = (A - B) / denom
    return out

def summarize_same_keys(diff_dict, Cl_a, Cl_b, tol=0):
    for k in sorted(diff_dict.keys()):
        A = np.asarray(Cl_a[k], float)
        B = np.asarray(Cl_b[k], float)
        D = diff_dict[k]

        I, J, L = D.shape
        printed = False

        for i in range(I):
            for j in range(J):

                rd = D[i, j, :]
                max_rel = np.nanmax(np.abs(rd))

                if max_rel <= tol:
                    continue

                if not printed:
                    print(f"\n[{k}] bins={I}x{J}, L={L}")
                    printed = True

                a = A[i, j, :]
                b = B[i, j, :]
                diff = a - b
                max_abs = np.nanmax(np.abs(diff))
                rms_rel = np.sqrt(np.nanmean(rd ** 2))

                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                cos = np.dot(a, b) / (na * nb) if na > 0 and nb > 0 else np.nan

                print(
                    f"  bin({i},{j}): max|Δ|={max_abs:.3e}  max|rel|={max_rel:.3e}  RMS_rel={rms_rel:.3e}  cos={cos:.6f}")

#-----------------------------------------------------------------------------------------
    # ANGULAR POWER SPECTRA CROSS CORRELATION WITH LENSING OLD
    #-----------------------------------------------------------------------------------------
    def limber_angular_power_spectra_lensing_cross_old(self, l, s_gal, beta, H_0, omega_m, omega_b, windows=None,
                                                   n_points=20, n_points_x=20, grid_x='mix', n_low=5, n_high=5, Delta_z=0.05,z_min=1e-05):

        cosmo = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b,Tcmb0=2.7255)

        # Check existence of power spectrum
        try:
            self.power_spectra_interpolator
        except AttributeError:
            raise AttributeError("Power spectra have not been loaded yet")

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min * self.geometric_factor_f_K(
            self.z_min), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max * self.geometric_factor_f_K(
            self.z_max), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Check window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:
                    windows_to_use[key] = self.window_function[key]
                except KeyError:
                    raise KeyError("Requested window function '%s' not known" % key)

        # Check window functions
        keys = windows_to_use.keys()
        if len(keys) == 0: raise AttributeError("No window function has been computed!")
        nkeys = len(keys)
        n_bins = [len(windows_to_use[key]) for key in keys]

        # 1) Define lengths and quantities
        # zz       = self.z_integration
        n_l = len(np.atleast_1d(l))
        # n_z      = self.nz_integration # this is now n_points

        Cl = {}
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                Cl['%s-%s' % (ki, kj)] = np.zeros((n_bins[i], n_bins[j], n_l))

        z_interp_bias = np.linspace(z_min, 10, 2000)

        def H(x):
            return cosmo.H(x).value #km/s/Mpc

        def r(x):
            return cosmo.comoving_distance(x).value # Mpc

        def JJ(tracer, x):
            if 'gal' in tracer:
                return np.ones_like(x)
            else:
                return (1 + x) * const.c / H(x) + (cosmo.comoving_distance(x).value)

        def A_L(chi, z, tracer, r_z1, bs, Hvals):
            if 'gal' in tracer:
                return 0.5 * (5 * bs - 2) * (r_z1 - chi) / chi
            else:
                conf_H = (Hvals / (1 + z))*1/const.c
                return 0.5 * (((r_z1 - chi) / chi) * (bs - 2) + (1 / (1 + r_z1 * conf_H)))

        # 2) Load power spectra
        power_spectra = self.power_spectra_interpolator

        # Add curvature correction (see arXiv:2302.04507)
        if self.cosmology.K != 0.:
            KK = self.cosmology.K
            factor = np.zeros((n_l, n_z))
            for il, ell in enumerate(l):
                factor[il] = (1 - np.sign(KK) * ell ** 2 / (((ell + 0.5) / self.geometric_factor) ** 2 + KK)) ** -0.5
            PS_lz *= factor

        ############################################################################
        def lensing_int(z1, z2, ll, bin_i, bin_j, WX_, WY_, tX, tY,n_points=30, p=1, z_min=1e-04, grid='mix', show_plot=False):

            r1 = r(z1)
            r2 = r(z2)
            Hx = H(z2)

            if 'gal' in tX:
                b1 = si.interp1d(z_interp_bias, s_gal, 'cubic', bounds_error=False, fill_value=0.)
            else:
                b1 = si.interp1d(z_interp_bias, beta, 'cubic', bounds_error=False, fill_value=0.)

            if 'gal' in tY:
                b2 = si.interp1d(z_interp_bias, s_gal, 'cubic', bounds_error=False, fill_value=0.)
            else:
                b2 = si.interp1d(z_interp_bias, beta, 'cubic', bounds_error=False, fill_value=0.)

            t1 = np.transpose(A_L(r2, z2, tY, r1, b2(z2), Hx) * (1 + z2) * JJ(tY, z2), (1, 0))

            if show_plot:
                idx = 9

                plt.plot(z1, WX_)
                plt.scatter(z1, WX_)
                plt.plot(z2[:, idx], WY_[:, idx], ls='--')
                plt.scatter(z2[:, idx], WY_[:, idx], ls='--')
                plt.axvline(z1[idx])
                plt.title(str(bin_i) + str(bin_j) + ', ' + tX + '-' + tY)
                plt.yscale('log')
                plt.show()
                plt.close()


            PS_ = np.squeeze(np.asarray([[[power_spectra((xx, yy)) for xx, yy in zip((ll[l] + 0.5) / r2[:, i], z2[:, i])] for i in range(len(z1))] for l in range(len(ll))]))

            WY_ = np.transpose(WY_, (0, 2, 1))

            my_int = t1[None, :, :, ] * WY_ * PS_

            I1_ = np.asarray([[np.trapz(my_int[l, i1], x=z2[:, i1], axis=0) for i1 in range(len(z1))] for l in range(len(ll))])

            I2_ = np.trapz(WX_ * I1_ / r1 * const.c / H(z1), x=z1, axis=1)

            return I2_

        ############################################################################
        # 3) load Cls given the source functions
        # 1st key (from 1 to N_keys)
        for index_X in xrange(nkeys):
            key_X = list(keys)[index_X]

            #################
            # choose z grid in each bin so that it is denser around the peak

            bins_ = self.bin_edges[key_X]
            bins_centers_ = (bins_[:-1] + bins_[1:]) / 2
            zzs = []
            ll_ = 0
            for bin_i in xrange(n_bins[index_X]):
                if 'gal' in key_X:
                    maxz = 5  # check this
                    npts = n_high
                else:
                    maxz = 10  # check this
                    npts = 2 * n_high
                if max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01) == z_min:
                    n1 = n_points + n_low
                else:
                    n1 = n_points
                if bin_i < n_bins[index_X]:
                    myarr = np.sort(np.unique(np.concatenate(
                        [np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)), bins_[bin_i + 1] * (1 + 5 * Delta_z),
                                     n_points),
                         np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low),
                         np.linspace(bins_[bin_i + 1] * (1 + 0.05) + 0.01, maxz, npts)])))
                else:
                    myarr = np.sort(np.unique(np.concatenate(
                        [np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)), maxz, n_points + npts),
                         np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low)])))
                l_ = len(myarr)
                if l_ > ll_:
                    ll_ = l_
                zzs.append(myarr)

            for i, a in enumerate(zzs):
                # print('len %s: %s'%(i, len(a)))
                if not len(a) == l_:
                    n_ = l_ - len(a)
                    # print('adding %s'%n_)
                    zzs[i] = np.sort(
                        np.unique(np.concatenate([zzs[i], np.linspace(z_min * (1 + 0.01), max(a) * (1 - 0.01), n_)])))
                    # print('new len %s: %s'%(i, len(zzs[i])))
            try:
                zzs = np.asarray(zzs)
            except Exception as e:
                print(zzs)
                print(zmax)
                print(e)

            # now compute window functions
            W_X = np.array([[windows_to_use[key_X][i, j](zzs[i]) for j in range(n_l)] for i in range(n_bins[index_X])])

            # 2nd key (from 1st key to N_keys)
            for index_Y in xrange(nkeys):
                key_Y = list(keys)[index_Y]

                # Symmetry C_{AA}^{ij} == C_{AA}^{ji}
                # NOW THIS IS NOT TRUE FOR LENSING!
                for bin_i in xrange(n_bins[index_X]):
                    myrange = xrange(n_bins[index_Y])
                    for bin_j in myrange:
                        if (bin_j == bin_i):  # and (key_Y==key_X): # and ('gal' in key_Y) and (bin_j<=2)
                            # print( 'computing %s %s %s %s'%(key_Y, key_X, bin_i, bin_j))
                            # this could also made finer by adapting the grid.
                            # For now we leave as it is
                            z2s_ = np.linspace(z_min, zzs[bin_i], n_points)
                            W_Y = np.array([[windows_to_use[key_Y][i, j](z2s_) for j in range(n_l)] for i in
                                            range(n_bins[index_Y])])

                            WY = W_Y[bin_j]
                            WX = W_X[bin_i]
                            Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j, :] = l * (l + 1) / (
                                        (l + 0.5) ** 2) * lensing_int(zzs[bin_i], z2s_, l, bin_i, bin_j, WX, WY, key_X,
                                                                      key_Y, z_min=z_min, n_points=n_points_x, grid=grid_x)

        return Cl

# -----------------------------------------------------------------------------------------
# ANGULAR POWER SPECTRA AUTO CORRELATION LENSING OLD
# -----------------------------------------------------------------------------------------
def limber_angular_power_spectra_lensing_auto_old(self, l, s_gal, beta, H_0, omega_m, omega_b, windows=None, n_points=20,
                                                  n_points_x=20, grid_x='mix', n_low=5, n_high=5, Delta_z=0.05, z_min=1e-05):

        cosmo = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b, Tcmb0=2.7255)

        # Check existence of power spectrum
        try:
            self.power_spectra_interpolator
        except AttributeError:
            raise AttributeError("Power spectra have not been loaded yet")

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min * self.geometric_factor_f_K(
            self.z_min), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max * self.geometric_factor_f_K(
            self.z_max), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Check window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:
                    windows_to_use[key] = self.window_function[key]
                except KeyError:
                    raise KeyError("Requested window function '%s' not known" % key)

        # Check window functions
        keys = windows_to_use.keys()
        if len(keys) == 0: raise AttributeError("No window function has been computed!")
        nkeys = len(keys)
        n_bins = [len(windows_to_use[key]) for key in keys]

        # 1) Define lengths and quantities
        # zz       = self.z_integration
        n_l = len(np.atleast_1d(l))
        # n_z      = self.nz_integration # this is now n_points

        Cl = {}
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                Cl['%s-%s' % (ki, kj)] = np.zeros((n_bins[i], n_bins[j], n_l))

        z_interp_bias = np.linspace(z_min, 10, 2000)

        def H(x):
            return cosmo.H(x).value

        def r(x):
            return cosmo.comoving_distance(x).value

        def JJ(tracer, x):
            if 'gal' in tracer:
                return np.ones(len(x))
            else:
                return (1 + x) * const.c / H(x) + (cosmo.comoving_distance(x).value)

        def A_L(chi, z, tracer, r_z1, bs, Hvals):
            if 'gal' in tracer:
                return 0.5 * (5 * bs - 2) * (r_z1 - chi) / chi
            else:
                conf_H = (Hvals / (1 + z))*1/const.c
                #print('conf_H OLD', conf_H)
                return 0.5 * (((r_z1 - chi) / chi) * (bs - 2) + (1 / (1 + r_z1 * conf_H)))

        # 2) Load power spectra
        power_spectra = self.power_spectra_interpolator

        # Add curvature correction (see arXiv:2302.04507)
        if self.cosmology.K != 0.:
            KK = self.cosmology.K
            factor = np.zeros((n_l, n_z))
            for il, ell in enumerate(l):
                factor[il] = (1 - np.sign(KK) * ell ** 2 / (((ell + 0.5) / self.geometric_factor) ** 2 + KK)) ** -0.5
            PS_lz *= factor

        ############################################################################
        def lensing_int(z1, z2, ll, bin_i, bin_j, WX_, WY_, tX, tY,
                        n_points=30, p=1, z_min=1e-04, grid='mix', show_plot=False):

            if grid == 'mix':
                x1 = np.linspace(z_min * (1 + 0.01), z2, n_points)
                x2 = np.geomspace(z_min, z2 * (1 - 0.01), n_points)
                x = np.zeros((2 * x1.shape[0], x1.shape[1], x1.shape[2]))
                for i in range(x1.shape[1]):
                    for j in range(x1.shape[2]):
                        x_ = np.sort(np.unique(np.concatenate([x1[:, i, j], x2[:, i, j]])))
                        if len(x_) < len(x):
                            n_ = len(x) - len(x_)
                            x_ = np.sort(np.unique(
                                np.concatenate([x_, np.linspace(z_min * (1 + 0.05), z2[i, j] * (1 - 0.05), n_)])))
                        x[:, i, j] = x_
            elif grid == 'lin':
                x = np.linspace(z_min, z2, n_points)
            elif grid == 'geom':
                # not recommended
                x = np.geomspace(z_min, z2, n_points)

            r1 = r(z1)
            r2 = r(z2)
            rx = r(x)
            Hx = H(x)

            #print('OLD')
            #print('r1',r1)
            #print('r2',r2)
            #print('rx',rx)
            #print('Hx',Hx)

            if 'gal' in tX:
                b1 = si.interp1d(z_interp_bias, s_gal, 'cubic', bounds_error=False, fill_value=0.)
            else:
                b1 = si.interp1d(z_interp_bias, beta, 'cubic', bounds_error=False, fill_value=0.)

            if 'gal' in tY:
                b2 = si.interp1d(z_interp_bias, s_gal, 'cubic', bounds_error=False, fill_value=0.)
            else:
                b2 = si.interp1d(z_interp_bias, beta, 'cubic', bounds_error=False, fill_value=0.)

            #print('OLD')
            #print('A_lx',A_L(rx, x, tX, r1, b1(x), Hx))
            #print('A_ly',A_L(rx, x, tY, r2, b2(x), Hx))
            #print('rx',rx)
            #print('H',const.c / Hx)

            t1 = np.transpose(
                const.c * A_L(rx, x, tX, r1, b1(x), Hx) * A_L(rx, x, tY, r2, b2(x), Hx) * (1 + x) ** 2 / Hx * rx ** 2,
                (1, 2, 0))

            if show_plot:
                idx = 9

                plt.plot(z1, WX_)
                plt.scatter(z1, WX_)
                plt.plot(z2[:, idx], WY_[:, idx], ls='--')
                plt.scatter(z2[:, idx], WY_[:, idx], ls='--')
                plt.axvline(z1[idx])
                plt.title(str(bin_i) + str(bin_j) + ', ' + tX + '-' + tY)
                # plt.yscale('log')
                plt.show()
                plt.close()

                plt.plot(x[:, idx, idx], t1[idx, idx, :])
                plt.scatter(x[:, idx, idx], t1[idx, idx, :])
                plt.yscale('log')
                plt.xscale('log')
                plt.show()
                plt.close()

                plt.plot(x[:, idx, idx], t1[idx, idx, :])
                plt.scatter(x[:, idx, idx], t1[idx, idx, :])
                # plt.yscale('log')
                # plt.xscale('log')
                plt.show()
                plt.close()

            PS_ = np.squeeze(np.asarray([[[[power_spectra((xx, yy)) for xx, yy in
                                            zip((ll[l] + 0.5) / rx[:, k, i], x[:, k, i])] for i in range(len(z1))] for k
                                          in range(len(z2))] for l in range(len(ll))]))

            my_int = t1[None, :, :, :, ] * PS_

            I1_ = np.asarray([[[np.trapz(my_int[l, i2, i1], x=x[:, i2, i1], axis=0) for i1 in range(len(z1))] for i2 in
                               range(len(z2))] for l in range(len(ll))])

            if show_plot:
                plt.plot(x[:, idx, idx], my_int[15, idx, idx, :])
                plt.scatter(x[:, idx, idx], my_int[15, idx, idx, :])
                plt.yscale('log')
                plt.xscale('log')
                plt.show()
                plt.close()

                plt.plot(x[:, idx, idx], my_int[15, idx, idx, :])
                plt.scatter(x[:, idx, idx], my_int[15, idx, idx, :])
                plt.show()
                plt.close()

            I2_ = np.asarray([[np.trapz(WY_[:, i1] * JJ(tY, z2[:, i1]) * I1_[l, :, i1] / r2[:, i1], x=z2[:, i1], axis=0)
                               for i1 in range(len(z1))] for l in range(len(ll))])

            I3_ = np.trapz(WX_ * JJ(tX, z1) * I2_ / r1, x=z1, axis=1)

            return I3_

        ############################################################################
        # 3) load Cls given the source functions
        # 1st key (from 1 to N_keys)
        for index_X in xrange(nkeys):
            key_X = list(keys)[index_X]

            #################
            # choose z grid in each bin so that it is denser around the peak

            bins_ = self.bin_edges[key_X]
            bins_centers_ = (bins_[:-1] + bins_[1:]) / 2
            zzs = []
            ll_ = 0
            for bin_i in xrange(n_bins[index_X]):
                if 'gal' in key_X:
                    maxz = 5  # check this
                    npts = n_high
                else:
                    maxz = 10  # check this
                    npts = 2 * n_high
                if max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01) == z_min:
                    n1 = n_points + n_low
                else:
                    n1 = n_points
                if bin_i < n_bins[index_X]:
                    myarr = np.sort(np.unique(np.concatenate(
                        [np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)), bins_[bin_i + 1] * (1 + 5 * Delta_z),
                                     n_points),
                         np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low),
                         np.linspace(bins_[bin_i + 1] * (1 + 0.05) + 0.01, maxz, npts)])))
                else:
                    myarr = np.sort(np.unique(np.concatenate(
                        [np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)), maxz, n_points + npts),
                         np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low)])))
                l_ = len(myarr)
                if l_ > ll_:
                    ll_ = l_
                zzs.append(myarr)

            for i, a in enumerate(zzs):
                # print('len %s: %s'%(i, len(a)))
                if not len(a) == l_:
                    n_ = l_ - len(a)
                    # print('adding %s'%n_)
                    zzs[i] = np.sort(
                        np.unique(np.concatenate([zzs[i], np.linspace(z_min * (1 + 0.01), max(a) * (1 - 0.01), n_)])))
                    # print('new len %s: %s'%(i, len(zzs[i])))
            try:
                zzs = np.asarray(zzs)
            except Exception as e:
                print(zzs)
                print(zmax)
                print(e)

            # now compute window functions
            W_X = np.array([[windows_to_use[key_X][i, j](zzs[i]) for j in range(n_l)] for i in range(n_bins[index_X])])

            # 2nd key (from 1st key to N_keys)
            for index_Y in xrange(nkeys):
                key_Y = list(keys)[index_Y]

                # Symmetry C_{AA}^{ij} == C_{AA}^{ji}
                # NOW THIS IS NOT TRUE FOR LENSING!
                for bin_i in xrange(n_bins[index_X]):
                    myrange = xrange(n_bins[index_Y])
                    for bin_j in myrange:
                        if (
                                bin_j == bin_i):  # and (key_Y==key_X):                      and ('gal' in key_Y) and (bin_j<=2)
                            # print( 'computing %s %s %s %s'%(key_Y, key_X, bin_i, bin_j))
                            # this could also made finer by adapting the grid.
                            # For now we leave as it is
                            z2s_ = np.linspace(z_min, zzs[bin_i], n_points)
                            W_Y = W_Y = np.array([[windows_to_use[key_Y][i, j](z2s_) for j in range(n_l)] for i in
                                                  range(n_bins[index_Y])])
                            WY = W_Y[bin_j, 0]
                            WX = W_X[bin_i, 0]
                            Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j, :] = l ** 2 * (l + 1) ** 2 / (
                                        (l + 0.5) ** 4) * lensing_int(zzs[bin_i], z2s_, l, bin_i, bin_j, WX, WY, key_X,
                                                                      key_Y, z_min=z_min, n_points=n_points_x, grid=grid_x)

        return Cl


####

print('CONFRONTA NUOVO E VECCHIO')


def compare_Cl_dicts(Cl_new, Cl_old, label_new="new", label_old="old"):
    """
    Cl_new, Cl_old: dict[key] -> array (n_bin_i, n_bin_j, n_l)
    """
    global_max_abs = 0.0
    global_max_rel = 0.0

    print(f"Comparing {label_new} vs {label_old}")

    for key in Cl_old.keys():
        if key not in Cl_new:
            print(f"[WARN] key {key} missing in {label_new}")
            continue

        A = np.asarray(Cl_old[key])
        B = np.asarray(Cl_new[key])

        if A.shape != B.shape:
            print(f"[WARN] shape mismatch for {key}: {A.shape} (old) vs {B.shape} (new)")
            continue

        diff = B - A

        # absolute
        max_abs = np.nanmax(np.abs(diff))

        # relative (only where A != 0)
        rel = np.full_like(diff, np.nan, dtype=float)
        mask = A != 0.0
        rel[mask] = diff[mask] / A[mask]
        max_rel = np.nanmax(np.abs(rel))

        print(f"{key}: max |ΔCℓ| = {max_abs:.3e},   max |ΔCℓ/Cℓ_old| = {max_rel:.3e}")

        global_max_abs = max(global_max_abs, max_abs)
        global_max_rel = max(global_max_rel, max_rel)

    print("--------------------------------------------------")
    print(f"GLOBAL max |ΔCℓ|          = {global_max_abs:.3e}")
    print(f"GLOBAL max |ΔCℓ/Cℓ_old|   = {global_max_rel:.3e}")


'''
# Call both loaders first
S.load_gravitational_wave_window_functions(bg, C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, bias=b_GW, name='bg')
S.load_gravitational_wave_window_functions_old(z=z_GW, ndl=ndl_GW, ll=ll, bias=b_GW, H0=H_0, omega_m=Omega_m, omega_b=Omega_b, name='old')

# Compare them
S.compare_gw_windows(z=z_GW, ll=ll, outdir=FLAGS.fout, prefix="GW_W", name_bg='bg', name_legacy='old',
                     save_overlays=False, save_rel=False)

# --- RSD windows: new vs old ---
S.load_rsd_window_functions(bg, h, z=z_gal, n_z=nz_gal, ll=ll, name='rsd_new')
S.load_rsd_window_functions_old(z=z_gal,nz=nz_gal,H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll, name='rsd_old')

S.compare_galaxy_windows(
    z=z_gal, ll=ll,
    outdir=FLAGS.fout,
    prefix='cmp_rsd',
    name_bg='rsd_new',
    name_legacy='rsd_old',
    save_overlays=False,
    save_rel=False,
    title='RSD'
)
res_rsd = S.compare_windows(
    name_a='rsd_new',
    name_b='rsd_old',
    z_eval=z_gal,
    ells=[0],
    outdir=FLAGS.fout,
    prefix="cmp_rsd",
    plot_overlays=False,
    plot_rel_lines=False,
    plot_rel_heatmap=False
)


# --- LSD windows: new vs old ---
S.load_lsd_window_functions(bg, h,C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, name='lsd_new')
S.load_lsd_window_functions_old(z_GW, ndl_GW, H_0, Omega_m, Omega_b, ll, name='lsd_old')

S.compare_galaxy_windows(
    z=z_GW, ll=ll,
    outdir=FLAGS.fout,
    prefix='cmp_lsd',
    name_bg='lsd_new',
    name_legacy='lsd_old',
    save_overlays=False,
    save_rel=False,
    title='LSD'
)


res_lsd = S.compare_windows(
    name_a='lsd_new',
    name_b='lsd_old',
    z_eval=z_GW,
    ells=[0],
    outdir=FLAGS.fout,
    prefix="cmp_lsd",
    plot_overlays=False,
    plot_rel_lines=False,
    plot_rel_heatmap=False
)


# --- GALAXY LENSING: new vs old ---
S.load_galaxy_lensing_window_functions(
    z=z_gal, n_z=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll, name='gal_new')
S.load_galaxy_lensing_window_functions_old(
    z=z_gal, n_z=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll, name='gal_old')

S.compare_galaxy_windows(
    z=z_gal, ll=ll,
    outdir='./cmp_gal',
    prefix='cmp_gal',
    name_bg='gal_new',
    name_legacy='gal_old',
    save_overlays=False,
    save_rel=False,
    title='GALAXY LENSING'
)


# --- GW LENSING: new vs old ---
S.load_gw_lensing_window_functions(
    bg=bg,C=Hi_Cosmo, z=z_GW,h=h, n_dl=ndl_GW, H_0=H_0, omega_m=Omega_m, ll=ll, name='gw_new'
)
S.load_gw_lensing_window_functions_old(  # your legacy loader
    z_GW, ndl_GW, H_0, Omega_m, Omega_b,  ll, name='gw_old'
)

S.compare_galaxy_windows(
    z=z_GW, ll=ll,
    outdir='./cmp_gw',
    prefix='cmp_gw',
    name_bg='gw_new',
    name_legacy='gw_old',
    save_overlays=False,
    save_rel=False,
    title='GW LENSING'
)
'''

'''
print('\n prima')
start = time.time()
Cl_lens_cross_old = S.limber_angular_power_spectra_lensing_cross_old(l=ll, s_gal=s_gal, beta=beta, H_0=H_0,
                                                                     omega_m=Omega_m, omega_b=Omega_b,
                                                                     windows=None, n_points=n_points,
                                                                     n_points_x=n_points_x,
                                                                     z_min=z_min, grid_x=grid_x,
                                                                     n_low=n_low,
                                                                     n_high=n_high)
end = time.time()
print(f"Time took {end - start:.4f} seconds \n")

compare_Cl_dicts(Cl_new=Cl_lens_cross, Cl_old=Cl_lens_cross_old,label_new="Hi_CLASS+alpha_M", label_old="FlatLCDM")
'''

'''
start = time.time()
Cl_lens_old=S.limber_angular_power_spectra_lensing_auto_old( l=ll, s_gal=s_gal, beta=beta, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, windows=['lensing_gal', 'lensing_GW'],
                                                             n_points=n_points, n_points_x=n_points_x,
                                                             z_min=z_min, grid_x=grid_x, n_low=n_low,
                                                             n_high=n_high, Delta_z=0.05)

end = time.time()
 print(f"Time took {end - start:.4f} seconds \n")

compare_Cl_dicts(Cl_new=Cl_lens, Cl_old=Cl_lens_old,label_new="Hi_CLASS+alpha_M", label_old="FlatLCDM")

key = 'lensing_GW-lensing_GW'

Cl_old_mat = np.asarray(Cl_lens_old[key])  # shape: (n_bin, n_bin, n_l)
Cl_new_mat = np.asarray(Cl_lens[key])

n_bin_i, n_bin_j, n_ell = Cl_old_mat.shape
assert n_bin_i == n_bin_j  # se bins simmetrici

for b in range(n_bin_i):
    Cl_old_arr = Cl_old_mat[b, b, :]  # auto bin b
    Cl_new_arr = Cl_new_mat[b, b, :]

    print(f"\nBin {b}:")
    print("  old min/max:", Cl_old_arr.min(), Cl_old_arr.max())
    print("  new min/max:", Cl_new_arr.min(), Cl_new_arr.max())

ratio = Cl_new_arr / Cl_old_arr
print("  ratio min/max:", ratio.min(), ratio.max())
'''

# --- RSD windows: new vs old ---
S.load_rsd_window_functions(bg, h, z=z_gal, n_z=nz_gal, ll=ll, name='rsd_new')
S.load_rsd_window_functions_old(z=z_gal, nz=nz_gal, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll,name='rsd_old')

S.compare_galaxy_windows(
    z=z_gal, ll=ll,
    outdir=FLAGS.fout,
    prefix='cmp_rsd',
    name_bg='rsd_new',
    name_legacy='rsd_old',
    save_overlays=False,
    save_rel=False,
    title='RSD'
)
res_rsd = S.compare_windows(
    name_a='rsd_new',
    name_b='rsd_old',
    z_eval=z_gal,
    ells=[0],
    outdir=FLAGS.fout,
    prefix="cmp_rsd",
    plot_overlays=False,
    plot_rel_lines=False,
    plot_rel_heatmap=False
)

# --- LSD windows: new vs old ---
S.load_lsd_window_functions(bg, h, C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, name='lsd_new')
S.load_lsd_window_functions_old(z_GW, ndl_GW, H_0, Omega_m, Omega_b, ll, name='lsd_old')

S.compare_galaxy_windows(
    z=z_GW, ll=ll,
    outdir=FLAGS.fout,
    prefix='cmp_lsd',
    name_bg='lsd_new',
    name_legacy='lsd_old',
    save_overlays=False,
    save_rel=False,
    title='LSD'
)

res_lsd = S.compare_windows(
    name_a='lsd_new',
    name_b='lsd_old',
    z_eval=z_GW,
    ells=[0],
    outdir=FLAGS.fout,
    prefix="cmp_lsd",
    plot_overlays=False,
    plot_rel_lines=False,
    plot_rel_heatmap=False
)

# --- GALAXY LENSING: new vs old ---
S.load_galaxy_lensing_window_functions(
    z=z_gal, n_z=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll, name='gal_new')
S.load_galaxy_lensing_window_functions_old(
    z=z_gal, n_z=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll, name='gal_old')

S.compare_galaxy_windows(
    z=z_gal, ll=ll,
    outdir='./cmp_gal',
    prefix='cmp_gal',
    name_bg='gal_new',
    name_legacy='gal_old',
    save_overlays=False,
    save_rel=False,
    title='GALAXY LENSING'
)


###################################### OLD
# -----------------------------------------------------------------------------------------
# GRAVITATIONAL WAVES CLUSTERING WINDOW FUNCTION
# -----------------------------------------------------------------------------------------
def load_gravitational_wave_window_functions_old(self, z, ndl, H0, omega_m, omega_b, ll, bias=1.0, name='GW'):

    conversion = FlatLambdaCDM(H0=H0, Om0=omega_m, Ob0=omega_b, Tcmb0=2.7255)
    ndl = np.array(ndl)
    z = np.array(z)
    n_bins = len(ndl)

    norm_const = simpson(ndl, x=conversion.luminosity_distance(z).value, axis=1) #[Mpc]
    assert np.all(np.diff(
        z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (
        self.dz_windows)
    assert ndl.ndim == 2, "'nz' must be 2-dimensional"
    assert (ndl.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
    assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (
        self.z_min)
    assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (
        self.z_max)

    if isinstance(bias, float):
        bias = bias * np.ones(n_bins)
    elif isinstance(bias, int):
        bias = np.float(bias) * np.ones(n_bins)
    else:
        assert len(bias) == n_bins, "Number of bias factors different from number of bins"

    # conversion.luminosity_distance in [Mpc]; const.c / self.Hubble in [h/Mpc]
    jac_dl = ((1 + self.z_integration) * (const.c / self.Hubble)/0.6781 + (conversion.luminosity_distance(self.z_integration).value) / (1 + self.z_integration)) #[Mpc]
    # print('\n GW OLD:')
    #print('Hubble [h/Mpc] ', self.Hubble/const.c)
    # print('r_n_z [Mpc/h]', conversion.luminosity_distance(self.z_integration).value)
    # print('jac_dL mix', jac_dL.tolist())
    # print('norm',norm_const)
    #print('\n-------------------------------------------------------------\n')

    # Initialize window
    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        tmp_interp = si.interp1d(z, ndl[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        #                                                                         1/Gpc * [Mpc]* [1/Mpc] *[Mpc]
        self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration) * (jac_dl) * ((self.Hubble / const.c)*0.6781) / norm_const[galaxy_bin] * bias[galaxy_bin], 'cubic',
                                                      bounds_error=False, fill_value=0.))

    self.window_function[name] = np.tile(np.array(self.window_function[name]).reshape(-1, 1), (1, len(ll)))

# -----------------------------------------------------------------------------------------
# RSD WINDOW FUNCTION
# -----------------------------------------------------------------------------------------
def load_rsd_window_functions_old(self, z, nz, H_0, omega_m, omega_b, ll, name='RSD'):

    cosmo = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b)

    nz = np.array(nz)
    z = np.array(z)
    n_bins = len(nz)
    norm_const = simpson(nz, x=z, axis=1)
    assert np.all(np.diff(
        z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (
        self.dz_windows)
    assert nz.ndim == 2, "'nz' must be 2-dimensional"
    assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
    assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (
        self.z_min)
    assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (
        self.z_max)

    def L0(ll):
        return (2 * ll ** 2 + 2 * ll - 1) / ((2 * ll - 1) * (2 * ll + 3))

    def Lm1(ll):
        return -ll * (ll - 1) / ((2 * ll - 1) * np.sqrt((2 * ll - 3) * (2 * ll + 1)))

    def Lp1(ll):
        return -(ll - 1) * (ll + 2) / ((2 * ll + 3) * np.sqrt((2 * ll + 1) * (2 * ll + 5)))

    # Initialize window
    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        nz_interp = si.interp1d(z, nz[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        f_interp = si.interp1d(z, cosmo.Om(z) ** 0.55, 'cubic', bounds_error=False, fill_value=0.)

        def Wm1(ll):
            return Lm1(ll) * nz_interp(((2 * ll + 1 - 4) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1 - 4) / (2 * ll + 1)) * self.z_integration)

        def Wz(ll):
            return L0(ll) * nz_interp(((2 * ll + 1) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1) / (2 * ll + 1)) * self.z_integration)

        def Wp1(ll):
            return Lp1(ll) * nz_interp(((2 * ll + 1 + 4) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1 + 4) / (2 * ll + 1)) * self.z_integration)

        self.window_function[name].append([si.interp1d(self.z_integration,
                                                       (Wm1(l) + Wz(l) + Wp1(l)) * self.Hubble / const.c /
                                                       norm_const[galaxy_bin], 'cubic', bounds_error=False,
                                                       fill_value=0.) for l in ll])

    self.window_function[name] = np.array(self.window_function[name])

# -----------------------------------------------------------------------------------------
# LSD WINDOW FUNCTION
# -----------------------------------------------------------------------------------------
def load_lsd_window_functions_old(self, z, ndl, H_0, omega_m, omega_b, ll, name='LSD'):

    conversion = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b, Tcmb0=2.7255)

    ndl = np.array(ndl)
    z = np.array(z)
    n_bins = len(ndl)

    # print('omega_z OLD',conversion.Om(z))

    conf_H = (conversion.H(z).value / (1 + z)) * 1 / (0.6781) * 1 / (const.c)  # h/Mpc
    r_conf_H = (conversion.comoving_distance(z).value * 0.6781) * conf_H  # Mpc/h
    gamma = r_conf_H / (1 + r_conf_H)

    jj = ((1 + z) * (const.c * 0.6781) / conversion.H(z).value + (
                conversion.luminosity_distance(z).value * 0.6781) * 1 / (1 + z))  # Mpc/h

    norm_const = simpson(ndl, x=conversion.luminosity_distance(z).value, axis=1)

    # print('conf_H OLD', conf_H)
    # print('self.Hubble / const.c',self.Hubble / const.c)
    # print('r_conf_H OLD', r_conf_H)
    # print('gamma OLD', gamma)
    # print('jj OLD', jj)
    # print('norm_const OLD', norm_const)

    assert np.all(np.diff(
        z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (
        self.dz_windows)
    assert ndl.ndim == 2, "'nz' must be 2-dimensional"
    assert (ndl.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
    assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (
        self.z_min)
    assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (
        self.z_max)

    def L0(ll):
        return (2 * ll ** 2 + 2 * ll - 1) / ((2 * ll - 1) * (2 * ll + 3))

    def Lm1(ll):
        return -ll * (ll - 1) / ((2 * ll - 1) * np.sqrt((2 * ll - 3) * (2 * ll + 1)))

    def Lp1(ll):
        return -(ll - 1) * (ll + 2) / ((2 * ll + 3) * np.sqrt((2 * ll + 1) * (2 * ll + 5)))

    # Initialize window
    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        nz_interp = si.interp1d(z, ndl[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        f_interp = si.interp1d(z, conversion.Om(z) ** 0.55 * 2 * gamma * jj, 'cubic', bounds_error=False,
                               fill_value=0.)

        def Wm1(ll):
            return Lm1(ll) * nz_interp(((2 * ll + 1 - 4) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1 - 4) / (2 * ll + 1)) * self.z_integration)

        def Wz(ll):
            return L0(ll) * nz_interp(((2 * ll + 1) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1) / (2 * ll + 1)) * self.z_integration)

        def Wp1(ll):
            return Lp1(ll) * nz_interp(((2 * ll + 1 + 4) / (2 * ll + 1)) * self.z_integration) * f_interp(
                ((2 * ll + 1 + 4) / (2 * ll + 1)) * self.z_integration)

        # self.Hubble / const.c in h/Mpc
        self.window_function[name].append([si.interp1d(self.z_integration,
                                                       (Wm1(l) + Wz(l) + Wp1(l)) * self.Hubble / const.c /
                                                       norm_const[galaxy_bin], 'cubic', bounds_error=False,
                                                       fill_value=0.) for l in ll])

    self.window_function[name] = np.array(self.window_function[name])

# -----------------------------------------------------------------------------------------
# GALAXY LENSING WINDOW FUNCTION
# -----------------------------------------------------------------------------------------

def load_galaxy_lensing_window_functions_old(self, z, n_z, H_0, omega_m, ll, name='galaxy_lensing'):
    constant = 3. * omega_m * (H_0 / const.c) ** 2.
    nz = np.array(n_z)
    z = np.array(z)
    n_bins = len(nz)
    norm_const = simpson(nz, x=z, axis=1)

    self.window_function[name] = []
    # Compute window
    for galaxy_bin in xrange(n_bins):
        tmp_interp = si.interp1d(z, nz[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        self.window_function[name].append(
            si.interp1d(self.z_integration, constant * tmp_interp(self.z_integration) / norm_const[galaxy_bin],
                        'cubic', bounds_error=False, fill_value=0.))

    self.window_function[name] = np.tile(np.array(self.window_function[name]).reshape(-1, 1), (1, len(ll)))

# -----------------------------------------------------------------------------------------
# GRAVITATIONAL WAVES LENSING WINDOW FUNCTION
# -----------------------------------------------------------------------------------------
def load_gw_lensing_window_functions_old(self, z, ndl, H_0, omega_m, omega_b, ll, name='lensing_GW'):

    conversion = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=omega_b,Tcmb0=2.7255)
    z = np.array(z)
    n_bins = len(ndl)
    dl_grid = conversion.luminosity_distance(z).value  # Mpc
    norm_const = simpson(ndl, x=dl_grid, axis=1)

    #H_0=67.81
    constant = 3. * omega_m * (H_0 / const.c) ** 2.  #H_0/c in 1/Mpc

    # Initialize window
    self.window_function[name] = []
    # Set windows
    for galaxy_bin in xrange(n_bins):
        # Select which is the function and which are the arguments
        n_z_interp = si.interp1d(z, ndl[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
        self.window_function[name].append(
            si.interp1d(self.z_integration, constant * n_z_interp(self.z_integration) / norm_const[galaxy_bin],'cubic', bounds_error=False, fill_value=0.))

    self.window_function[name] = np.tile(np.array(self.window_function[name]).reshape(-1, 1), (1, len(ll)))


# ------------------------------------------------------
# Simple post-processing + saving + corner plot
# ------------------------------------------------------

def estimate_acceptance_rate(chain):
    """
    Rough estimate of acceptance rate:
    fraction of steps where the state actually changed.
    """
    # chain[i] == chain[i-1] for rejected steps
    moved = np.any(chain[1:] != chain[:-1], axis=1)
    return moved.mean()


def postprocess_mcmc(chain,logp,labels,burnin=5000, thin=10,outdir="mcmc_results",corner_filename="corner_posterior.pdf"):
    """
    Post-process MCMC chain:
      - remove burn-in
      - thin
      - compute mean, std, covariance
      - save everything
      - produce a simple corner plot

    Parameters
    ----------
    chain : array, (N_steps, N_params)
        Full MCMC chain (including burn-in).
    logp : array, (N_steps,)
        Log-posterior at each step.
    labels : list of str
        Parameter labels in the same order as chain columns.
    burnin : int
        Number of initial steps to discard.
    thin : int
        Keep one every 'thin' samples AFTER burn-in.
    outdir : str
        Directory where results and figures are saved.
    corner_filename : str
        Name of the PDF file for the corner plot.

    Returns
    -------
    samples : array, (N_samples_eff, N_params)
        Posterior samples after burn-in & thinning.
    stats : dict
        Dictionary with means, std, covariance, acceptance rate, etc.
    """

    os.makedirs(outdir, exist_ok=True)

    # PROBABILMENTE C'È UN PROBLEMA DI DIMENSIONI QUA DENTRO
    chain = np.asarray(chain)
    logp = np.asarray(logp)

    # --- Save raw chain first ---
    raw_path = os.path.join(outdir, "chain_raw.npz")
    np.savez(raw_path, chain=chain, logp=logp, labels=np.array(labels, dtype=object))
    print(f"[postprocess] Saved raw chain to: {raw_path}")

    # --- Burn-in + thinning ---
    if burnin >= chain.shape[0]:
        raise ValueError("burnin is larger than total chain length!")

    chain_cut = chain[burnin:]
    logp_cut = logp[burnin:]

    samples = chain_cut[::thin]
    logp_eff = logp_cut[::thin]

    print(f"[postprocess] After burn-in ({burnin}) and thinning (every {thin}): "
          f"{samples.shape[0]} samples remain.")

    # --- Acceptance rate (rough) ---
    acc_rate = estimate_acceptance_rate(chain)
    print(f"[postprocess] Estimated acceptance rate (full chain): {acc_rate:.3f}")

    # --- Basic stats ---
    mean_params = np.mean(samples, axis=0)
    std_params = np.std(samples, axis=0, ddof=1)
    cov_params = np.cov(samples.T, ddof=1)

    # --- Save processed samples and stats ---
    post_path = os.path.join(outdir, "chain_posterior.npz")
    np.savez(
        post_path,
        samples=samples,
        logp=logp_eff,
        labels=np.array(labels, dtype=object),
        mean=mean_params,
        std=std_params,
        cov=cov_params,
        acceptance_rate=acc_rate,
        burnin=burnin,
        thin=thin,
    )
    print(f"[postprocess] Saved posterior samples & stats to: {post_path}")

    # --- Save a summary ---
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("MCMC summary\n")
        f.write("====================\n\n")
        f.write(f"Total steps        : {chain.shape[0]}\n")
        f.write(f"Burn-in            : {burnin}\n")
        f.write(f"Thinning factor    : {thin}\n")
        f.write(f"Effective samples  : {samples.shape[0]}\n")
        f.write(f"Acceptance rate    : {acc_rate:.4f}\n\n")

        f.write("Parameter stats (mean ± std):\n")
        for lab, m, s in zip(labels, mean_params, std_params):
            f.write(f"  {lab:10s} = {m:.5g} ± {s:.5g}\n")
    print(f"[postprocess] Saved summary to: {summary_path}")

    # --- Corner plot ---
    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 12},
        use_math_text=True,
        smooth=0.6,
        smooth1d=0.6,
        color="royalblue",
        plot_datapoints=False,
        fill_contours=True,
    )

    fig.suptitle("Posterior distributions", fontsize=18)

    fig_path = os.path.join(outdir, corner_filename)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[postprocess] Saved corner plot to: {fig_path}")

    # Pack stats in a dict for easy reuse in notebooks, etc.
    stats = {
        "mean": mean_params,
        "std": std_params,
        "cov": cov_params,
        "acceptance_rate": acc_rate,
        "N_steps": chain.shape[0],
        "burnin": burnin,
        "thin": thin,
        "N_eff": samples.shape[0],
    }

    return samples, stats

# -------------------------------
#  FLAT LOG-PRIOR logP(θ)
# -------------------------------

def log_prior(theta):
    """
    Simple top-hat (flat) prior on each parameter.

    θ = [H0, Omega_m, Omega_b, As_1e9, n_s,
         alpha_M, alpha_B, w0, wa,
         b_gal, b_GW]

    Returns ln P(θ). If θ is outside the allowed range,
    we return -∞ (log(0)).
    """
    (H0, Omega_m, Omega_b, As_1e9, n_s,
     alpha_M, alpha_B, w0, wa,
     b_gal, b_GW) = theta

    # Example ranges (adapt as you like)
    if not (50.0 < H0 < 80.0):
        return -np.inf
    if not (0.1 < Omega_m < 0.5):
        return -np.inf
    if not (0.03 < Omega_b < 0.07):
        return -np.inf
    if not (1.5 < As_1e9 < 3.5):
        return -np.inf
    if not (0.9 < n_s < 1.1):
        return -np.inf

    if not (-1.0 < alpha_M < 1.0):
        return -np.inf
    if not (-2.0 < alpha_B < 2.0):
        return -np.inf
    if not (-2.0 < w0 < -0.3):
        return -np.inf
    if not (-1.0 < wa < 1.0):
        return -np.inf

    if not (0.1 < b_gal < 5.0):
        return -np.inf
    if not (0.1 < b_GW < 5.0):
        return -np.inf

    # Flat prior within bounds
    return 0.0


# --------------------------------
#  LOG-POSTERIOR logPost(θ)
# --------------------------------

def log_posterior(theta, Cl_fid, Cov_inv):
    """
    Full log-posterior:

        ln P(θ | data) = ln P_prior(θ) + ln L(θ)

    Using the fiducial Cl's as mock data.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_Likelihood(theta, Cl_fid, Cov_inv)
    return lp + ll


# ---------------------------------------------
#  Metropolis–Hastings MCMC SAMPLER
# ---------------------------------------------

def run_mcmc(theta0, n_steps, proposal_cov, Cl_fid, Cov_inv, seed=42):
    """
    Basic Metropolis–Hastings sampler for the posterior.

    Parameters
    ----------
    theta0       : 1D array, initial position in parameter space.
    n_steps      : int, total number of MCMC steps.
    proposal_cov : 2D array, proposal covariance for a Gaussian random walk.
    Cl_fid       : 2D array, fiducial spectra (3, N_ell).
    Cov_inv      : 3D array, inverse covariance (3,3,N_ell).
    seed         : int, random seed for reproducibility.

    Returns
    -------
    chain : array, shape (n_steps, n_params)
        The MCMC chain of θ values.
    logp  : array, shape (n_steps,)
        Log-posterior values at each step.
    """
    rng = np.random.default_rng(seed)

    theta0 = np.array(theta0, dtype=float)
    n_params = theta0.size

    chain = np.zeros((n_steps, n_params))
    logp = np.zeros(n_steps)

    # Evaluate posterior at starting point
    theta_current = theta0.copy()
    # QUI POSSO EVITARE DI CHIAMARE LA FUNZIONE --> A MENO CHE LA PRIOR NON CI STA IN MEZZO
    logp_current = log_posterior(theta_current, Cl_fid, Cov_inv)

    chain[0] = theta_current
    logp[0] = logp_current

    for i in range(1, n_steps+1):
        # Propose new θ' from multivariate Gaussian centered at θ_current
        proposal = rng.multivariate_normal(
            mean=np.zeros(n_params),
            cov=proposal_cov
        )
        # Updating the variables
        theta_prop = theta_current + proposal

        # Compute log posterior at proposal
        logp_prop = log_posterior(theta_prop, Cl_fid, Cov_inv)

        # Acceptance probability
        logA = logp_prop - logp_current
        if np.log(rng.uniform()) < logA:
            # Accept
            theta_current = theta_prop
            logp_current = logp_prop

        # Store current state (whether moved or stayed)
        chain[i] = theta_current
        logp[i] = logp_current

        if (i + 1) % 100 == 0:
            print(f"Step {i + 1}/{n_steps}  logP = {logp_current:.3f}")

    return chain, logp

############################################## OLD
def exp_factor_alphaM(z, alpha_interp):
    """
    Compute exp( ∫_0^z 0.5 * alpha_M(z')/(1+z') dz' ) for scalar or array z.
    - z: float or 1-D array-like of redshifts (must be monotonic increasing if array)
    - alpha_interp: callable returning alpha_M at given z (e.g., scipy interp1d)

    Returns:
      float if z is scalar, or a 1-D np.ndarray with same length as z if array.
    """

    def _cumulative_trapezoid(y, x, initial=0.0):
        dx = np.diff(x)
        area = dx * 0.5 * (y[1:] + y[:-1])
        return np.concatenate(([initial], initial + np.cumsum(area)))

    if np.isscalar(z):
        integrand = lambda zp: 0.5 * alpha_interp(zp) / (1.0 + zp)
        val, _ = sint.quad(integrand, 0.0, float(z), epsabs=1e-6)
        return float(np.exp(val))

    # array case
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError("z must be a scalar or 1-D array")
    if not np.all(np.diff(z) >= 0):
        raise ValueError("z array must be monotonically non-decreasing")

    vals = 0.5 * alpha_interp(z) / (1.0 + z)
    cum_int = _cumulative_trapezoid(vals, z, initial=0.0)
    return np.exp(cum_int)

def exp_factor_alphaM(z, alpha_interp):
    """
    Compute exp( ∫_0^z 0.5 * alpha_M(z')/(1+z') dz' ).
    Vectorised in z. z can be scalar or 1D array (not necessarily sorted).
    """

    z = np.atleast_1d(z).astype(float)

    # Sort z so we can do a cumulative integral once
    order = np.argsort(z)
    z_sorted = z[order]

    # Build integrand on sorted grid
    vals = 0.5 * alpha_interp(z_sorted) / (1.0 + z_sorted)

    # Cumulative integral from min(z) upward
    I_sorted = cumulative_trapezoid(vals, z_sorted, initial=0.0)

    # Map back to original order
    I = np.empty_like(I_sorted)
    I[order] = I_sorted

    out = np.exp(I)
    # Return scalar if input was scalar
    return float(out[0]) if np.isscalar(z) and out.size == 1 else out

def build_exp_alphaM_interp(alpha_interp, z_max=10.0, ngrid=2000):
    z_grid = np.linspace(0.0, float(z_max), int(ngrid))
    vals = 0.5 * alpha_interp(z_grid) / (1.0 + z_grid)
    I = cumulative_trapezoid(vals, z_grid, initial=0.0)
    E_grid = np.exp(I)

    E_of_z = interp1d(
        z_grid, E_grid,
        kind='cubic',
        bounds_error=False,
        fill_value='extrapolate',
        assume_sorted=True,
    )
    return E_of_z

# print('Delta',Delta)

# Sum over ℓ of Δ^T Cov^{-1} Δ
# Log-likelihood
# const = np.array([np.log(np.linalg.det(F[:,:,i])) for i in range(F.shape[2])])
'''
for i in range(F.shape[2]):
    det_direct = np.linalg.det(F[:, :, i])
    sign, logdet_slog = np.linalg.slogdet(F[:, :, i])

    print(i,
          " direct=", np.log(det_direct) if det_direct > 0 else float('nan'),
          " slogdet=", logdet_slog)
'''
# print('det F',np.linalg.slogdet(F.transpose(2, 0, 1))[1],'\n')
# sign, logdetF = np.linalg.slogdet(F_ell)  # both shape (n_ell,)

# Check that F is positive definite for every ℓ
# if not np.all(sign > 0):
#	print("Warning: some determinants of F are non-positive!")

# --------------------------------------
#  COMPUTE  NEW Cl's FOR A GIVEN COSMOLOGY
# --------------------------------------
def Cl_UPDATE(cosmo_params, theta, name):
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
    # start = time.time()

    print('\nUpdating parameters...')
    params = deepcopy(cosmo_params)

    # ΛCDM-like parameters
    params[f'{name}'] = theta[f'{name}']
    # params['omega_m'] = theta['omega_m']
    # params['omega_b'] = theta['omega_b']
    # params['A_s'] = theta['A_s'] * 10 ** (-9)
    # params['n_s'] = theta['n_s']

    # Modified gravity (SMG) parameters
    # x_k, x_b, x_m, x_t, (M_*)^ 2_ini -> '10.0, 0.0, 0.0, 0.0, 1.0' LCDM case
    # params['parameters_smg'] = f"10.0,{theta['alpha_B']},{theta['alpha_M']},0.0,1.0"

    # expansion_smg = cosmo_params['expansion_smg']
    # expansion_smg_list = list(map(float, expansion_smg.split(',')))
    # if len(expansion_smg_list) == 3:
    #	params['expansion_smg'] = f"0.7,{w_0},{w_a}"

    # print('\nThese are the values of the new cosmology:')
    # print(r"COSMO: $H_0$:", params['h'] * 100, r"$\omega_m$:", params['omega_m'], r'$\omega_b$', params['omega_b'],r'$A_s$', params['A_s'], r"$n_s$", params['n_s'])
    # print('MG: \t parameters_smg', params['parameters_smg'], '\t expansion model', params['expansion_smg'], '\n')

    # Cosmology object from CLASS / hi_class
    Hi_Cosmo = cc.cosmo(**params)

    start = time.time()
    Cl_GG_update, Cl_GWGW_update, Cl_GGW_update = Cl_func(Hi_Cosmo, params, gw_params, dl_GW, bin_edges_dl, z_gal, ll,
                                                          theta['bias_gal'], theta['bias_GW'], save=False)

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

    Cl_vector_updated, _, _ = LH_fun.build_vector(n_bins_z, n_bins_dl, Cl_GGW_total, Cl_GWGW_total, Cl_GG_total)

    end = time.time()
    print(f"Time took {end - start:.4f} seconds for the Cl UPDATING\n")
    # Cl_vector_updated,_, _ = LH_fun.build_vector(n_bins_z, n_bins_dl, Cl_GGW_update, Cl_GWGW_update, Cl_GG_update)

    return Cl_vector_updated