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
import argparse
from matplotlib.animation import FuncAnimation
from numba import jit, njit, prange, set_num_threads
import concurrent.futures
import time
from numba_progress import ProgressBar
import yaml
import datetime
from dateutil.relativedelta import relativedelta

os.chdir('/home/bhanselman/bhanselman/scripts')
from normalization_methods import *

# Constants
c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s

# Adapted from: https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
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
    ys_mean = ys / ys.mean()
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
            normalized_spectrum[j] = 1
            continue

        ind_xs = np.arange(-(window//2), window//2+1).astype(np.float64)

        current_ys = ys_mean[i-window//2:i+window//2+1]
        idx = np.isfinite(current_ys)
        
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

@njit(cache=True)
def build_asymmetry(specs, state, mask): #template=False
    '''Form asymmetry from a given state.'''
    inner = (state == 1) & mask
    outer = (state == 1) & ~mask

    # specs = template_spectra if template else spectra

    # inner_mean, outer_mean = get_means(specs, inner, outer)
    inner_slice = specs[inner, :]
    outer_slice = specs[outer, :]
    # inner_mean = np.mean(inner_slice, axis=0)
    # outer_mean = np.mean(outer_slice, axis=0)
    inner_mean = np.zeros(specs.shape[1])
    outer_mean = np.zeros(specs.shape[1])
    
    for j in range(specs.shape[1]):
        inner_mean[j] = inner_slice[:, j].mean()
        outer_mean[j] = outer_slice[:, j].mean()

    asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)

    # if template:
    #     # Smoothing the template with a spline (Aya)
    #     xs_temp = np.arange(len(asymmetry))
    #     asymmetry = scipy.interpolate.make_interp_spline(xs_temp[::15], asymmetry[::15], k=3)(xs_temp)
    
    return asymmetry 

@njit(cache=True)
def wammsd_dynamic(asymmetry, freqs, freq_diff=0.18, factor=1e-3):
    '''Weighted average moving median squared deviation. Window width is in MHz.'''
    # Basic filter (pass middle 6 nodes)
    def filter(i, length):
        if i < (length / 8):
            return 8 * i / length
        elif i > (7 * length / 8):
            return 8 * (1 - i / length)
        else:
            return 1
    
    #msds = list(map(msd, range(500, len(asymmetry)-500)))
    #inds = np.arange(half_width_bin, len(asymmetry)-half_width_bin)
    msds = np.zeros(len(freqs))
    filt = np.zeros(len(freqs))
    for i, freq in enumerate(freqs):
        half_width_bin = int(factor * freq / freq_diff)
        window = asymmetry[max(0, i-half_width_bin):min(len(freqs)-1, i+half_width_bin)]
        devs = np.zeros(len(window))
        for j in range(len(window)):
            devs[j] = np.abs(window[j]) ** 2
        msds[i] = np.median(devs)
        filt[i] = filter(i, len(freqs))

    wammsd = np.average(msds, weights=filt)
    return wammsd #np.sqrt(wammsd) 

@njit(cache=True)
def objective_mad_new(state, spectra, template_spectra, mask, freqs=None, freq_diff=0.18, factor=1e-3):
    '''The objective function.'''
    asymmetry = build_asymmetry(spectra, state, mask)
    template = build_asymmetry(template_spectra, state, mask)

    template_integral = np.sum(np.abs(template) ** 2)
    denom = wammsd_dynamic(asymmetry, freqs, freq_diff=freq_diff, factor=factor) 
    return template_integral / denom

@njit(parallel=True, cache=True)
def cross_entropy(
    f,
    n_pop,
    rho,
    n_gen,
    p0,
    d,
    spectra,
    template_spectra,
    mask,
    progress,
    alpha=1,
    freqs=None,
    factor=1e-3,
    freq_diff=0.18310546875
):
    '''
    Implementation of cross-entropy method for combinatorial optimization of Bernoulli vectors.

    Parameters
    ---
    f: callable
        Performance function (to be maximized).
    n_pop: int
        Vector sample size.
    rho: float
        Quantile of elite performances.
    n_gen: int
        Number of generations.
    p0: array_like, shape (M,)
        Initial probability vector for binary vectors of size M.
    d: int
        Stopping parameter.
    alpha: float, optional
        Alpha value for smooth probability vector updating. Default 1 (no smooth updating).

    Returns
    ---
    x_best: np.ndarray
        Vector with the best performance.
    f_best: np.ndarray
        Best performance.
    perfs: np.ndarray
        History of the best performance at each iteration.
    '''
    perfs = np.zeros(n_gen)
    x_best = np.zeros(n_pop, dtype=np.int64)
    f_best = 0

    np.random.seed(42) # For reproducibility

    for i in range(n_gen): #trange(n_gen):
        # Generate vectors
        xs = np.empty((n_pop, len(p0)), dtype=np.int64)
        for j in range(n_pop):
            for k in range(len(p0)):
                xs[j,k] = np.random.binomial(1, p0[k])

        # Calculate performances
        fs = np.empty(n_pop, dtype=np.float64)
        for j in prange(n_pop):
            fs[j] = f(
                xs[j],
                spectra,
                template_spectra,
                mask,
                freqs=freqs,
                factor=factor,
                freq_diff=freq_diff
            )
        # Elite sample
        gamma = np.quantile(fs, 1 - rho) #1 - float(n_elite) / float(n_pop))
        elite = xs[fs >= gamma]

        # Save best performance
        f_cand = np.max(fs)
        print(f_cand)
        print(p0)
        perfs[i] = f_cand
        if f_cand > f_best:
            f_best = f_cand
            x_best = xs[np.argmax(fs)]

        # Update probabilities - weighted average
        p = np.empty(len(p0), dtype=np.float64)
        weights = fs[fs >= gamma]
        for j in range(len(p0)):
            total = 0.0
            weight_sum = 0.0
            for k in range(elite.shape[0]):
                total += elite[k, j] * weights[k]
                weight_sum += weights[k]
            p[j] = total / weight_sum
        p0 = alpha * p + (1 - alpha) * p0 # Smooth updating

        # Convergence check
        if (i >= d) and np.all(f_cand == perfs[i-d:i]): # Last d iterations
            print(f'Converged at iteration {i}!')
            # Truncate perfs at the current iteration
            perfs = perfs[:i]
            break

        progress.update(1)
        
    return x_best, f_best, perfs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    config_dir = args.config
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)
    
    info_path = config['info_path']
    data_dir = config['data_dir']
    save_dir = config['save_dir']
    sefd = config['sefd']
    is_decay = config['interaction'] == 'decay'
    n_banks = config['n_banks']

    print(is_decay)

    ref_signal_size = 1e-29 if is_decay else 1e-26

    os.makedirs(save_dir, exist_ok=True)

    info = pd.read_csv(info_path)

    # Factor for dynamic wammsd
    num = 3 if is_decay else 6
    factor = v_earth / c + (v_virial / (c*np.sqrt(num)) * 3) * (1 + v_earth / c)

    start_time = datetime.datetime.now()
    
    for bank in range(n_banks):
        print(f'Starting bank {bank}!')
        # Bank info
        bank_dir = os.path.join(data_dir, str(bank))
        bank_info = info[info['source'].apply(lambda l: l + '.npy').isin(os.listdir(bank_dir))]
        print(len(bank_info))

        # All bank spectra, full range
        files = bank_info['source'].apply(lambda l: l + '.npy')

        # Save dir
        opt_dir = os.path.join(save_dir, str(bank))
        os.makedirs(opt_dir, exist_ok=True)

        # Normalized uninjected spectra
        uninj_spectra = np.array([np.load(os.path.join(bank_dir, file)) for file in files])
        freqs = uninj_spectra[0][0]
        freq_diff = freqs[1] - freqs[0]
        ref_freq = (freqs[0] + freqs[-1]) / 2 # Center of bank

        if os.path.exists(os.path.join(opt_dir, 'uninj_norm_spectra.npy')):
            print('Loading uninj norm spectra')
            uninj_norm_spectra = np.load(os.path.join(opt_dir, 'uninj_norm_spectra.npy'))
            #uninj_norm_spectra = np.array([np.load(os.path.join(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/heatmap/052926_01/ann/{bank}', file))[1] for file in files]) #np.load(os.path.join(opt_dir, 'uninj_norm_spectra.npy'))
            #freqs = np.load(os.path.join(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/heatmap/052926_01/ann/{bank}', files[0]))[0]
        else:
            print('Normalizing uninjected')
            with ProgressBar(total=len(files)) as progress:
                uninj_norm_spectra = normalize_all_spectra(uninj_spectra, progress, b=1, exterior=3, is_decay=is_decay)
            np.save(os.path.join(opt_dir, 'uninj_norm_spectra.npy'), uninj_norm_spectra)

        # Templates
        if os.path.exists(os.path.join(opt_dir, 'template_spectra.npy')):
            print('Loading template spectra')
            template_spectra = np.load(os.path.join(opt_dir, 'template_spectra.npy'))
        else:
            # Injections
            spec = uninj_spectra[0]
            inj_spectra = np.empty((len(files), spec.shape[0], spec.shape[1]))
            for i, targ in enumerate(files):
                inj_spectra[i] = inject_calibrated(
                    targ,
                    ref_freq,
                    bank_info,
                    data_dir=bank_dir, # Note: no calibration dir!
                    sefd=sefd,
                    signal_sizes=ref_signal_size, 
                    is_decay=is_decay,
                    template=True
                )
            print('Normalizing templates')
            with ProgressBar(total=len(files)) as progress:
                template_spectra = normalize_all_spectra(inj_spectra, progress, b=1, exterior=3, freq=ref_freq, width=100, is_decay=is_decay)
            np.save(os.path.join(opt_dir, 'template_spectra.npy'), template_spectra)
        
        # Optimization for Doppler and intensity
        for asym_type in ['doppler', 'intensity']:
            print(f'Optimizing {asym_type}')
            inner_all_mask = (bank_info['theta'] < 90).to_numpy() if asym_type == 'doppler' else (bank_info['phi'] < 90).to_numpy()
            asym_dir = os.path.join(opt_dir, asym_type)
            os.makedirs(asym_dir, exist_ok=True)

            if os.path.exists(os.path.join(asym_dir, 'best_state.npy')):
                print('Optimization already complete; loading in results.')
                state_best = np.load(os.path.join(asym_dir, 'best_state.npy'))
                perfs = np.load(os.path.join(asym_dir, 'perfs.npy'))
            else:
                p0 = 0.5 * np.ones(len(bank_info))

                # Run optimization
                with ProgressBar(total=1000) as progress:
                    state_best, f_best, perfs = cross_entropy(
                        objective_mad_new,
                        5000,
                        0.01,
                        1000,
                        p0,
                        10,
                        uninj_norm_spectra,
                        template_spectra,
                        inner_all_mask,
                        progress,
                        alpha=0.5,
                        freqs=freqs,
                        factor=factor,
                        freq_diff=freq_diff
                    )

                # Save optimization results
                np.save(os.path.join(asym_dir, 'best_state.npy'), state_best)
                np.save(os.path.join(asym_dir, 'perfs.npy'), perfs)

            # Save plot of best asymmetry (with old for comparison)
            aya_mask = (bank_info['theta'] < 65).to_numpy() if asym_type == 'doppler' else inner_all_mask
            state_aya = np.ones(len(bank_info)) if asym_type == 'doppler' else np.where(bank_info['phi'].between(70,115), 0, 1)
            asymmetry_aya = build_asymmetry(uninj_norm_spectra, state_aya, aya_mask)
            asymmetry_best = build_asymmetry(uninj_norm_spectra, state_best, inner_all_mask)
            mad_aya = scipy.stats.median_abs_deviation(asymmetry_aya)
            mad_best = scipy.stats.median_abs_deviation(asymmetry_best)
            
            fig, ax = plt.subplots()
            ax.plot(freqs, asymmetry_aya, label=f'aya; MAD={mad_aya:.3g}')
            ax.plot(freqs, asymmetry_best, label=f'best; MAD={mad_best:.3g}')
            ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
            ax.set_ylabel(f'{asym_type.capitalize()} Asymmetry', fontsize=18)
            ax.set_ylim(np.percentile(asymmetry_aya, 2), np.percentile(asymmetry_aya, 98))
            ax.legend()
            fig.savefig(os.path.join(asym_dir, 'best_asymmetry_plot.png'), dpi=300, bbox_inches='tight')
            np.save(os.path.join(asym_dir, 'best_asymmetry.npy'), asymmetry_best)

            # Save trace plot
            fig, ax = plt.subplots()
            ax.plot(range(1, len(perfs)+1), perfs, c='k')
            ax.set_xlim(1, len(perfs)+1)
            ax.set_xlabel('Generation', fontsize=18)
            ax.set_ylabel('Performance', fontsize=18)
            fig.savefig(os.path.join(asym_dir, 'trace.png'), dpi=300, bbox_inches='tight')

            # Save sky plot of best state
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='aitoff')
            ax.grid(zorder=0)

            inner = (state_best == 1) & inner_all_mask
            outer = (state_best == 1) & ~inner_all_mask
            inner_info = bank_info[inner]
            outer_info = bank_info[outer]
            inner_gal = SkyCoord(inner_info['l'], inner_info['b'], unit='deg', frame='galactic')
            outer_gal = SkyCoord(outer_info['l'], outer_info['b'], unit='deg', frame='galactic')
            inner_sc = ax.scatter(inner_gal.l.wrap_at('180d').radian, inner_gal.b.radian, s=10, marker='*', c='k', label='forward')
            outer_sc = ax.scatter(outer_gal.l.wrap_at('180d').radian, outer_gal.b.radian, s=10, marker='*', c='gray', label='backward')
            
            if asym_type == 'doppler':
                # DOPPLER
                ax.plot([0, 0], [np.pi/2, -np.pi/2], c='r', label=r'$\theta=90^{\circ}$')
            else:
                # INTENSITY
                ax.plot([-np.pi/2, -np.pi/2], [-np.pi/2, np.pi/2], c='r')
                ax.plot([np.pi/2, np.pi/2], [np.pi/2, -np.pi/2], c='r', label=r'$\phi=90^{\circ}$')

            ax.legend(bbox_to_anchor=(0.8, 0.1))
            fig.savefig(os.path.join(asym_dir, 'aitoff_best_state.png'), dpi=300, bbox_inches='tight')
    
    end_time = datetime.datetime.now()
    delta = relativedelta(end_time, start_time)
    print(f'Optimization complete in {delta.hours} hrs, {delta.minutes} mins, {delta.seconds} sec')