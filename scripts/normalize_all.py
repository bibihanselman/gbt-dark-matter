import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from tqdm.auto import tqdm, trange
import json
from multiprocessing import Pool
from functools import partial
from numpy.polynomial import Polynomial
import math
from astropy.coordinates import SkyCoord
import gc
from matplotlib.colors import Normalize
from astropy.visualization import ZScaleInterval
import warnings
warnings.filterwarnings("ignore")
from numba import njit, jit, prange
from numba_progress import ProgressBar

# Constants
c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s

@njit
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0],deg + 1))
    const = np.ones_like(x)
    mat_[:,0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_
    
@jit
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_
 
@jit
def fit_poly(x, y, deg, w=None):
    a = _coeff_mat(x, deg)
    if w is not None:
        a *= w[:, np.newaxis]
        y *= w
    p = _fit_x(a, y)
    return p

@jit
def eval_polynomial(p, x):
    '''
    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0].  Uses Horner's Method.
    '''
    result = 0
    for coeff in p[::-1]:
        result = x * result + coeff
    return result

@njit(cache=True)
def normalize_weighted_numba(data, a=0.1, b=1, exterior=3, order=5, is_decay=True, freq=None, width=200, start=None, stop=None):
    """
    Standard rolling polynomial normalize procedure. Updated.
    """
    xs, ys = data
    
    freq_diff = xs[1] - xs[0]
    ys_mean = ys / np.nanmean(ys)
    for i in range(16, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
        ys_mean[i] = np.nan
        ys_mean[i+15] = np.nan
        ys_mean[i+1] = np.nan
        ys_mean[i+14] = np.nan
    
    xs_trimmed = xs
    if freq is not None:
        index = np.argmin(np.abs(xs - freq))
        width = int(width / freq_diff)
        xs_trimmed = xs[index-width:index+width+1]
    else:
        if start is not None or stop is not None:
            start_index = 0 if start is None else np.argmin(np.abs(xs - start))
            stop_index = len(xs) - 1 if stop is None else np.argmin(np.abs(xs - stop))
            xs_trimmed = xs[start_index:stop_index+1]
    
    new_xs = np.zeros(xs_trimmed.shape[0])
    normalized_spectrum = np.zeros(xs_trimmed.shape[0])
    
    def get_sigma(center):
        if is_decay:
            sigma = v_virial*center/(c*np.sqrt(3))
        else:
            sigma = v_virial*center/(c*np.sqrt(6))
        return sigma

    def round_up_to_nearest_odd(number):
        ceiled_number = math.ceil(number)
        return int(ceiled_number + 1) if ceiled_number % 2 == 0 else int(ceiled_number)

    # Loop through the data points and divide each point by the polynomial 
    for j in range(len(xs_trimmed)):
        x = xs_trimmed[j]
        new_xs[j] = x
        #i = xs.tolist().index(x)
        i = np.where(xs == x)[0][0]
        window = round_up_to_nearest_odd(2 * exterior * get_sigma(x) / freq_diff)
        
        if (i < (window//2)) or (i >= len(xs) - (window//2 + 1)): 
            normalized_spectrum[j] = np.nan # 1
            continue

        ind_xs = np.arange(-(window//2), window//2+1).astype(np.float64)
        current_ys = ys_mean[i-window//2:i+window//2+1]
        idx = np.isfinite(current_ys)

        if math.isnan(ys[i]) or np.all(~idx):
            normalized_spectrum[j] = np.nan
            continue

        # Unweighted fit
        #unweighted_params = np.polyfit(ind_xs[idx], current_ys[idx], order)
        unweighted_p = fit_poly(ind_xs[idx], current_ys[idx], order)
        
        # Weighted fit
        sigma = get_sigma(x)
        weights = 1 - a*np.exp(-(freq_diff*ind_xs[idx])**2 / (2*b*sigma**2))
        #weighted_params = np.polyfit(ind_xs[idx], current_ys[idx], order, w=weights)
        weighted_p = fit_poly(ind_xs[idx], current_ys[idx], order, w=weights)

        unweighted = eval_polynomial(unweighted_p, 0) #np.poly1d(unweighted_params)
        weighted = eval_polynomial(weighted_p, 0) #np.poly1d(weighted_params)

        normalized_spectrum[j] = unweighted/weighted
        
    #new_ys = ys_mean[lower: len(xs)-upper]
    normalized = np.zeros((2, new_xs.shape[0]))
    normalized[0] = new_xs
    normalized[1] = normalized_spectrum
    #old = np.array([new_xs, new_ys])

    return normalized

@njit(parallel=True)
def normalize_all_spectra(spectra, progress, b=1, exterior=3, freq=None, width=100, is_decay=True):
    # Normalize the first spectrum to get the right shape (and warm up the function)
    test_spec = spectra[0]
    test_norm = normalize_weighted_numba(test_spec, b=b, exterior=exterior, freq=freq, width=width, is_decay=is_decay)
    norm_spectra = np.empty((spectra.shape[0], test_norm.shape[1]))
    for i in prange(norm_spectra.shape[0]):
        norm = normalize_weighted_numba(spectra[i], b=b, exterior=exterior, freq=freq, width=width, is_decay=is_decay)
        norm_spectra[i] = norm[1]
        progress.update(1)
    return norm_spectra

@njit(nogil=True, parallel=True)
def quality_cut(spectra, freqs, progress, freq_diff=1500/8192, rho=0.1, is_decay=True):
    mads = np.empty_like(spectra)
    num = 3 if is_decay else 6
    factor = v_earth / c + (v_virial / (c*np.sqrt(num)) * 3) * (1 + v_earth / c)

    for i in prange(spectra.shape[0]):
        for j in range(len(freqs)):
            half_width_bin = int(2 * factor * freqs[j] / freq_diff)
            window = spectra[i][max(0, j-half_width_bin):min(len(freqs)-1, j+half_width_bin)]
            devs = np.zeros(len(window))
            for k in range(len(window)):
                devs[k] = np.abs(window[k]-1)
            mads[i, j] = np.nanmedian(devs)
        
        progress.update(1)
    
    quantiles = np.empty_like(mads)
    for i in range(mads.shape[1]):
        quantile = np.nanquantile(mads[:, i], 1-rho)
        quantiles[:, i] = quantile
    #quantiles = np.nanquantile(mads, 1-rho, axis=0, keepdims=True)
    new_spectra = np.where(mads > quantiles, np.nan, spectra)
    
    return new_spectra

if __name__ == '__main__':
    for bank in range(10):
        for case in ['decay', 'ann']:
            print(f'Bank {bank} {case}')
            data_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/lscx_081526/data/preprocessed/{bank}'
            norm_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/lscx_081526/data/normalized/{case}/{bank}'
            norm_cut_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/lscx_081526/data/normalized_cut/{case}/{bank}'

            files = os.listdir(data_dir)
            freqs = np.load(os.path.join(data_dir, files[0]))[0]
            spectra = np.array([np.load(os.path.join(data_dir, file)) for file in files])

            # Normalize spectra
            print('Normalizing')
            with ProgressBar(total=spectra.shape[0]) as progress:
                norm_spectra = normalize_all_spectra(spectra, progress, is_decay=(case == 'decay'))

            os.makedirs(norm_dir, exist_ok=True)
            for file, spec in zip(files, norm_spectra):
                save_spec = np.array([freqs, spec])
                np.save(os.path.join(norm_dir, file), save_spec)

            # norm_spectra = np.array([np.load(os.path.join(norm_dir, file))[1] for file in files])

            # Quality cut
            print('Quality cut')
            with ProgressBar(total=spectra.shape[0]) as progress:
                new_spectra = quality_cut(norm_spectra, freqs, progress, is_decay=(case == 'decay'))

            os.makedirs(norm_cut_dir, exist_ok=True)
            for file, spec in zip(files, new_spectra):
                save_spec = np.array([freqs, spec])
                np.save(os.path.join(norm_cut_dir, file), save_spec)