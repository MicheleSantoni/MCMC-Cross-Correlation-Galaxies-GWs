import numpy as np

# Mock Cl dictionaries to simulate serial and parallel versions
def mock_compute_cl_serial(keys, n_bins, n_l):
    Cl = {}
    for key_X in keys:
        for key_Y in keys:
            Cl_name = f"{key_X}-{key_Y}"
            Cl[Cl_name] = np.random.rand(n_bins[key_X], n_bins[key_Y], n_l)
    return Cl

def mock_compute_cl_parallel(keys, n_bins, n_l):
    Cl = {}
    for key_X in keys:
        for key_Y in keys:
            Cl_name = f"{key_X}-{key_Y}"
            Cl[Cl_name] = np.random.rand(n_bins[key_X], n_bins[key_Y], n_l)
    return Cl

# Test function to compare the two
def test_compare_cl_outputs():
    keys = ['galaxy', 'GW']
    n_bins = {'galaxy': 3, 'GW': 3}
    n_l = 10

    Cl_serial = mock_compute_cl_serial(keys, n_bins, n_l)
    Cl_parallel = mock_compute_cl_parallel(keys, n_bins, n_l)

    # Now compare
    for key in Cl_serial:
        if not np.allclose(Cl_serial[key], Cl_parallel[key], rtol=1e-5, atol=1e-8):
            return f"Mismatch found in Cl[{key}]"
    return "All Cl matrices match within tolerance."


def test_cl_parallel_vs_serial(serial_func, parallel_func, *args, **kwargs):
    """
    Compare output of serial vs parallel Cl computation functions.

    Parameters:
    - serial_func: Reference function (serial)
    - parallel_func: Function under test (parallel)
    - *args, **kwargs: Arguments passed to both functions

    Returns:
    - None. Raises AssertionError if mismatches found.
    """
    Cl_serial = serial_func(*args, **kwargs)
    Cl_parallel = parallel_func(*args, **kwargs)

    for key in Cl_serial:
        assert np.allclose(Cl_serial[key], Cl_parallel[key], rtol=1e-5, atol=1e-8), \
            f"Mismatch in Cl[{key}]"

    print("✅ Serial and parallel Cl results match!")


test_compare_cl_outputs()



from concurrent.futures import ProcessPoolExecutor
        import numpy as np

        def compute_grid_z(key, bins, n_bins, Deltaz, z_min, z_max, n_points, n_high, n_low):
            zzs = []
            for bin_i in range(n_bins):
                if 'gal' in key:
                    maxz = 5
                    npts = n_high
                else:
                    maxz = 10
                    npts = 2 * n_high

                if max(z_min, bins[bin_i] * (1 - 5 * Deltaz) - 0.01) == z_min:
                    n1 = n_points + n_low
                else:
                    n1 = n_points

                if bin_i < n_bins:
                    myarr = np.sort(np.unique(np.concatenate([
                        np.linspace(max(z_min, bins[bin_i] * (1 - 5 * Deltaz)), bins[bin_i + 1] * (1 + 5 * Deltaz),
                                    n_points),
                        np.linspace(z_min, max(z_min, bins[bin_i] * (1 - 5 * Deltaz) - 0.01), n_low),
                        np.linspace(bins[bin_i + 1] * (1 + 0.05) + 0.01, maxz, npts)
                    ])))
                else:
                    myarr = np.sort(np.unique(np.concatenate([
                        np.linspace(max(z_min, bins[bin_i] * (1 - 5 * Deltaz)), maxz, n_points + npts),
                        np.linspace(z_min, max(z_min, bins[bin_i] * (1 - 5 * Deltaz) - 0.01), n_low)
                    ])))

                zzs.append(myarr)

            l_ = max(len(a) for a in zzs)
            for i, a in enumerate(zzs):
                if len(a) < l_:
                    n_ = l_ - len(a)
                    zzs[i] = np.sort(np.unique(np.concatenate([
                        zzs[i],
                        np.linspace(z_min * (1 + 0.01), max(a) * (1 - 0.01), n_)
                    ])))

            return np.asarray(zzs)

        def compute_cl_entry_full(args):
            (key_X, key_Y, bin_i, bin_j, zzs, windows_to_use, n_l, l, bg,
             z_min, n_points_x, grid_x, n_points) = args

            z2s_ = np.linspace(z_min, zzs[bin_i], n_points)
            W_X = np.array([[windows_to_use[key_X][i, j](zzs[i]) for j in range(n_l)] for i in range(len(zzs))])
            W_Y = np.array([[windows_to_use[key_Y][i, j](z2s_) for j in range(n_l)] for i in range(len(zzs))])

            WX = W_X[bin_i]
            WY = W_Y[bin_j]

            Cl_val = l * (l + 1) / ((l + 0.5) ** 2) * lensing_int(
                bg, zzs[bin_i], z2s_, l, bin_i, bin_j, WX, WY, key_X, key_Y,
                z_min=z_min, n_points=n_points_x, grid=grid_x
            )

            return (key_X, key_Y, bin_i, bin_j, Cl_val)

        def compute_full_cl_parallel(keys, windows_to_use, n_bins, bin_edges,
                                     Deltaz, z_min, z_max, n_points, n_high, n_low,
                                     n_l, l, bg, n_points_x, grid_x):
            Cl = {f"{k1}-{k2}": np.zeros((n_bins[k1], n_bins[k2], len(l))) for k1 in keys for k2 in keys}
            all_args = []

            for key_X in keys:
                bins_ = bin_edges[key_X]
                zzs = compute_grid_z(key_X, bins_, n_bins[key_X], Deltaz, z_min, z_max, n_points, n_high, n_low)

                for key_Y in keys:
                    for bin_i in range(n_bins[key_X]):
                        for bin_j in range(n_bins[key_Y]):
                            if bin_j == bin_i:
                                all_args.append((key_X, key_Y, bin_i, bin_j, zzs,
                                                 windows_to_use, n_l, l, bg,
                                                 z_min, n_points_x, grid_x, n_points))

            with ProcessPoolExecutor() as executor:
                results = executor.map(compute_cl_entry_full, all_args)

            for key_X, key_Y, bin_i, bin_j, Cl_val in results:
                Cl[f"{key_X}-{key_Y}"][bin_i, bin_j, :] = Cl_val

            return Cl

