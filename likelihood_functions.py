import matplotlib.pyplot as plt
import numpy as np
import os
import re

def build_vector(n_bins_z,n_bins_dl,Cl_GGW,Cl_GWGW,Cl_GG,l_max_nl_,l_max_loc):
    # vector_idx: labels for all spectra
    vec_idx = vector_idx(n_bins_z, n_bins_dl)

    # vector_cl: flatten all Cℓ into a single data vector
    vec = vector_cl(l_max_nl_,l_max_loc, cl_cross=Cl_GGW, cl_auto1=Cl_GG, cl_auto2=Cl_GWGW)

    # Build a dict of Cℓ by label   ---> Associa ad ogni termine la sua corretta etichetta
    vec_dict = {label: vec[i] for i, label in enumerate(vec_idx)}
    return vec, vec_dict,vec_idx

def vector_idx(n_bins_1, n_bins_2):
    '''
     Crea un array con le etichettte per i tracers X/Y e i bins: X0X1 -> tracer X e X, bins 0 e 1
    '''
    lenght = n_bins_1 ** 2 - np.sum(range(n_bins_1)) + (n_bins_1 * n_bins_2) + n_bins_2 ** 2 - np.sum(range(n_bins_2))
    vec_cl = np.zeros(shape=(lenght), dtype=object)
    i = 0
    for j in range(n_bins_1):
        for k in range(j, n_bins_1):
            vec_cl[i] = 'X%iX%i' % (j, k)  #GG
            i += 1
    for j in range(n_bins_1):
        for k in range(n_bins_2):
            vec_cl[i] = 'X%iY%i' % (j, k) #GGW
            i += 1
    for j in range(n_bins_2):
        for k in range(j, n_bins_2):
            vec_cl[i] = 'Y%iY%i' % (j, k) #GWGW
            i += 1
    return vec_cl


def vector_cl(l_max_nl_,l_max_loc, cl_cross, cl_auto1, cl_auto2):
    '''
     Crea una matrice di dimensione (Lunghezza tutte le Cl, n_ell)
    '''
    n_bins_1 = len(cl_auto1)
    n_bins_2 = len(cl_auto2)
    lenght = n_bins_1 ** 2 - np.sum(range(n_bins_1)) + (n_bins_1 * n_bins_2) + n_bins_2 ** 2 - np.sum(range(n_bins_2))
    #l_max_global = int(np.min([np.in(l_max_nl_), np.min(l_max_loc)]))
    vec_cl = np.zeros(shape=(lenght, len(cl_auto1[0, 0])))

    i = 0
    for j in range(n_bins_1):
        for k in range(j, n_bins_1):
            lmax = l_max_nl_[min(j, k)]
            vec_cl[i, :lmax] = cl_auto1[j, k, :lmax]
            i += 1

    for j in range(n_bins_1):
        for k in range(n_bins_2):
            lmax = min(l_max_nl_[min(j, k)], l_max_loc[k])
            vec_cl[i, :lmax] = cl_cross[j, k, :lmax]
            i += 1
    for j in range(n_bins_2):
        for k in range(j, n_bins_2):
            lmax = min(l_max_nl_[min(j, k)], l_max_loc[k])
            vec_cl[i, :lmax] = cl_auto2[j, k, :lmax]
            i += 1

    return vec_cl


# -----------------------------------------------------------------------------------------
#                                COVARIANCE BUILDING AND INVERTING
# -----------------------------------------------------------------------------------------

'''
def build_cov_and_inv(n_bins_z,n_bins_dl,vec_fid,vec_idx,ll,f_sky=0.35,delta_ell=1):
    """
        Build and Invert the 3×3 covariance block at each multipole ℓ.

        Parameters
        ----------
        Cov : array, shape (3, 3, N_ell)
            Covariance matrix for [GG, GGW, GWGW] at each ℓ.
        ell : array, shape (N_ell,)
            Multipole values (only used for sanity checks / loops).

        Returns
        -------
        Cov_inv : array, shape (3, 3, N_ell)
            Inverse covariance for each ℓ.
    """
    # Building the full covariance matrix
    tot_len = n_bins_z ** 2 - np.sum(range(n_bins_z)) + n_bins_z * n_bins_dl + n_bins_dl ** 2 - np.sum(range(n_bins_dl))
    cov_mat = np.zeros(shape=(tot_len, tot_len, len(ll)))
    prefactor = 1.0 / ((2 * ll + 1) * f_sky * delta_ell)  # shape (N_ell,)
    print('prefactor',prefactor, 'len(ll)',len(ll))
    #print('vec_fid shape',vec_fid.shape)
    for i in range(tot_len):
        for j in range(tot_len):
            # sort_lab: Label reshuffling helpers
            #print('1:',lab_I1J1(vec_idx, i, j))
            #print('2:',lab_I2J2(vec_idx, i, j))
            #print('3:',lab_I1J2(vec_idx, i, j))
            #print('4:',lab_I2J1(vec_idx, i, j))
            #print('\n')
            cov_mat[i, j] = prefactor * (vec_fid[sort_lab(lab_I1J1(vec_idx, i, j))] * vec_fid[sort_lab(lab_I2J2(vec_idx, i, j))]
                                         + vec_fid[sort_lab(lab_I1J2(vec_idx, i, j))] * vec_fid[sort_lab(lab_I2J1(vec_idx, i, j))])

    #print(cov_mat.shape)
    #for l in range(len(ll)):
    #    eigs = np.linalg.eigvals(cov_mat[:, :, l])
    #    if np.any(eigs <= 0):
    #        print("Non-PD covariance at ℓ =", ll[l])
    
    test_sym = False
    for ell in range(cov_mat.shape[2]):
        A = cov_mat[:, :, ell]
        diff = np.max(np.abs(A - A.T))
        if diff > 1e-10:
            test_sym = True
            print(f"cov_mat NOT symmetric at ell={ell}: {diff:.2e}")

    if test_sym:
        for ell in range(cov_mat.shape[2]):
            cov_mat[:, :, ell] = 0.5 * (cov_mat[:, :, ell] + cov_mat[:, :, ell].T)

    
    #cov_inv = np.zeros_like(cov_mat)
    #for l in range(len(ll)):
    #    cov_inv[:, :, l] = np.linalg.inv(cov_mat[:, :, l])
    

    cov_t = np.transpose(cov_mat, (2, 0, 1))  # (n_ell, n, n)
    cov_inv_t = np.linalg.inv(cov_t)
    cov_inv = np.transpose(cov_inv_t, (1, 2, 0))

    return cov_mat, cov_inv
    
def build_cov_and_inv_single(length,vec_dict,vec_idx,ll,f_sky=0.35,delta_ell=1):
    """
        Build and Invert the 3×3 covariance block at each multipole ℓ.

        Parameters
        ----------
        Cov : array, shape (3, 3, N_ell)
            Covariance matrix for [GG, GGW, GWGW] at each ℓ.
        ell : array, shape (N_ell,)
            Multipole values (only used for sanity checks / loops).

        Returns
        -------
        Cov_inv : array, shape (3, 3, N_ell)
            Inverse covariance for each ℓ.
    """
    # Building the full covariance matrix
    cov_mat = np.zeros(shape=(length, length, len(ll)))
    prefactor = 1.0 / ((2 * ll + 1) * f_sky * delta_ell)  # shape (N_ell,)
    print('prefactor', prefactor, 'len(ll)', len(ll))

    for i in range(length):
        for j in range(length):
            cov_mat[i, j] = prefactor*(vec_dict[sort_lab(lab_I1J1(vec_idx, i, j))] * vec_dict[sort_lab(lab_I2J2(vec_idx, i, j))]
                                       + vec_dict[sort_lab(lab_I1J2(vec_idx, i, j))] * vec_dict[sort_lab(lab_I2J1(vec_idx, i, j))])

    for l in range(len(ll)):
        eigs = np.linalg.eigvals(cov_mat[:, :, l])
        if np.any(eigs <= 0):
            print("Non-PD covariance at ℓ =", ll[l])

    test_sym = False
    for ell in range(cov_mat.shape[2]):
        A = cov_mat[:, :, ell]
        diff = np.max(np.abs(A - A.T))
        if diff > 1e-10:
            test_sym = True
            print(f"cov_mat NOT symmetric at ell={ell}: {diff:.2e}")

    if test_sym:
        for ell in range(cov_mat.shape[2]):
            cov_mat[:, :, ell] = 0.5 * (cov_mat[:, :, ell] + cov_mat[:, :, ell].T)

    #cov_mat: shape (n, n, n_ell)
    # Move ℓ in the first ax
    cov_t = np.transpose(cov_mat, (2, 0, 1))  # (n_ell, n, n)

    # Invert batch
    cov_inv_t = np.linalg.inv(cov_t)  # (n_ell, n, n)

    # Back to the original
    cov_inv = np.transpose(cov_inv_t, (1, 2, 0))  # (n, n, n_ell)

    return cov_mat, cov_inv
'''


def build_cov_and_inv(n_bins_z, n_bins_dl, vec_fid, vec_idx, ll, f_sky=0.35, delta_ell=1):
    tot_len = n_bins_z ** 2 - np.sum(range(n_bins_z)) + n_bins_z * n_bins_dl + n_bins_dl ** 2 - np.sum(range(n_bins_dl))
    cov_mat = np.zeros(shape=(tot_len, tot_len, len(ll)))
    cov_inv = np.zeros(shape=(tot_len, tot_len, len(ll)))
    prefactor = 1.0 / ((2 * ll + 1) * f_sky * delta_ell)

    for i in range(tot_len):
        for j in range(tot_len):
            cov_mat[i, j] = prefactor * (
                    vec_fid[sort_lab(lab_I1J1(vec_idx, i, j))] * vec_fid[sort_lab(lab_I2J2(vec_idx, i, j))]
                    + vec_fid[sort_lab(lab_I1J2(vec_idx, i, j))] * vec_fid[sort_lab(lab_I2J1(vec_idx, i, j))]
            )

    for l in range(len(ll)):
        C = cov_mat[:, :, l]

        # finds the rows/columns with diagonal elements not null
        diag = np.diag(C)
        active = np.where(diag > 0)[0]

        if len(active) == 0:
            continue

        # Extract the sub-mat not null
        C_sub = C[np.ix_(active, active)]

        # Symmetrize
        C_sub = 0.5 * (C_sub + C_sub.T)

        # invert
        C_sub_inv = np.linalg.inv(C_sub)

        # Re-insert
        cov_inv[np.ix_(active, active, [l])] = C_sub_inv[:, :, np.newaxis]

    return cov_mat, cov_inv

def build_cov_and_inv_single(length,vec_dict,vec_idx,ll,f_sky=0.35,delta_ell=1):
    """
        Build and Invert the 3×3 covariance block at each multipole ℓ.

        Parameters
        ----------
        Cov : array, shape (3, 3, N_ell)
            Covariance matrix for [GG, GGW, GWGW] at each ℓ.
        ell : array, shape (N_ell,)
            Multipole values (only used for sanity checks / loops).

        Returns
        -------
        Cov_inv : array, shape (3, 3, N_ell)
            Inverse covariance for each ℓ.
    """
    # Building the full covariance matrix
    cov_mat = np.zeros(shape=(length, length, len(ll)))
    cov_inv = np.zeros(shape=(length, length, len(ll)))
    prefactor = 1.0 / ((2 * ll + 1) * f_sky * delta_ell)  # shape (N_ell,)
    #print('prefactor', prefactor, 'len(ll)', len(ll))

    for i in range(length):
        for j in range(length):
            cov_mat[i, j] = prefactor*(vec_dict[sort_lab(lab_I1J1(vec_idx, i, j))] * vec_dict[sort_lab(lab_I2J2(vec_idx, i, j))]
                                       + vec_dict[sort_lab(lab_I1J2(vec_idx, i, j))] * vec_dict[sort_lab(lab_I2J1(vec_idx, i, j))])


    for l in range(len(ll)):
        C = cov_mat[:, :, l]

        # finds the rows/columns with diagonal elements not null
        diag = np.diag(C)
        active = np.where(diag > 0)[0]

        if len(active) == 0:
            continue

        # Extract the sub-mat not null
        C_sub = C[np.ix_(active, active)]

        # Symmetrize
        C_sub = 0.5 * (C_sub + C_sub.T)

        # invert
        C_sub_inv = np.linalg.inv(C_sub)

        # Re-insert
        cov_inv[np.ix_(active, active, [l])] = C_sub_inv[:, :, np.newaxis]


    return cov_mat, cov_inv


def lab_I1J1(vec_idx, i, j):
    pattern = r'([XY])(\d+)([XY])(\d+)'
    match_i = re.match(pattern, vec_idx[i])
    match_j = re.match(pattern, vec_idx[j])
    if match_i and match_j:
        letter1, num1 = match_i.group(1), match_i.group(2)
        letter2, num2 = match_j.group(1), match_j.group(2)
        return f'{letter1}{num1}{letter2}{num2}'
    else:
        raise ValueError(f"Invalid label format: {vec_idx[i]} or {vec_idx[j]}")


def lab_I2J2(vec_idx, i, j):
    pattern = r'([XY])(\d+)([XY])(\d+)'
    match_i = re.match(pattern, vec_idx[i])
    match_j = re.match(pattern, vec_idx[j])
    if match_i and match_j:
        letter1, num1 = match_i.group(3), match_i.group(4)
        letter2, num2 = match_j.group(3), match_j.group(4)
        return f'{letter1}{num1}{letter2}{num2}'
    else:
        raise ValueError(f"Invalid label format: {vec_idx[i]} or {vec_idx[j]}")


def lab_I1J2(vec_idx, i, j):
    pattern = r'([XY])(\d+)([XY])(\d+)'  # Match: letter-number-letter-number
    match_i = re.match(pattern, vec_idx[i])
    match_j = re.match(pattern, vec_idx[j])
    if match_i and match_j:
        # Take first pair from i, second pair from j
        letter1, num1 = match_i.group(1), match_i.group(2)
        letter2, num2 = match_j.group(3), match_j.group(4)
        return f'{letter1}{num1}{letter2}{num2}'
    else:
        raise ValueError(f"Invalid label format: {vec_idx[i]} or {vec_idx[j]}")


def lab_I2J1(vec_idx, i, j):
    pattern = r'([XY])(\d+)([XY])(\d+)'
    match_i = re.match(pattern, vec_idx[i])
    match_j = re.match(pattern, vec_idx[j])
    if match_i and match_j:
        letter1, num1 = match_i.group(3), match_i.group(4)
        letter2, num2 = match_j.group(1), match_j.group(2)
        return f'{letter1}{num1}{letter2}{num2}'
    else:
        raise ValueError(f"Invalid label format: {vec_idx[i]} or {vec_idx[j]}")


def sort_lab(label):
    '''
        rimette in ordine le labels del tipo: Prima X e poi Y; prima numeri piccoli e poi più alti:
        sort_lab('X2X0') → X0X2
        sort_lab('Y3X1') → X1Y3 (since X < Y )
    '''
    pattern = r'([XY])(\d+)([XY])(\d+)'
    match = re.match(pattern, label)
    if match:
        part1 = (match.group(1), int(match.group(2)))  # e.g., ('X', 10)
        part2 = (match.group(3), int(match.group(4)))  # e.g., ('X', 0)
        # Sort using letter + number
        first, second = sorted([part1, part2])
        return f'{first[0]}{first[1]}{second[0]}{second[1]}'
    else:
        raise ValueError(f"Invalid label format: {label}")

def plot_SNR_xL(ll, snr_l,output_path):
    plt.figure(figsize=(7, 5))
    plt.plot(ll, snr_l, marker='o')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'${\rm SNR}_\ell$')
    plt.title('Signal-to-noise per multipole')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'SNR_per_multipole.pdf'), bbox_inches='tight')
    plt.close()

def plot_SNR_cumulative(ll, snr_l,output_path):
    snr_cum = np.sqrt(np.cumsum(snr_l))

    plt.figure(figsize=(7, 5))
    plt.plot(ll, snr_cum)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'${\rm SNR}(<\ell)$')
    plt.title('SNR cumulative')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'SNR_per_cumulative.pdf'), bbox_inches='tight')
    plt.close()






