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

os.chdir('/home/bhanselman/bhanselman/scripts')
from normalization_methods import *

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
def normalize_weighted_numba_parallel(data, a=0.1, b=1, exterior=3, order=5, is_decay=True, freq=None, width=200, start=None, stop=None):
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

def normalized_spectrum_template(file):
    inj_data = inject_calibrated(
        file,
        5650, # reference frequency; doesn't really matter.
        info,
        data_dir='/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/1',
        sefd=10,
        signal_sizes=1e-29,
        template=True
    )
    norm_data = normalize_weighted(
        inj_data,
        freq=5650,
        width=100,
        b=1,
        exterior=3,
        is_decay=is_decay
    )
    return norm_data[1]

def calculate_asymmetry_snr(xs, inj, freq, window=10):
    mask = abs(xs - freq) < window
    window_ys = inj[mask]
    signal = np.max(window_ys)
    noise = scipy.stats.median_abs_deviation(inj[~mask]) #np.std(uninj)
    return signal / noise

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
    
    for j in prange(specs.shape[1]):
        inner_mean[j] = inner_slice[:, j].mean()
        outer_mean[j] = outer_slice[:, j].mean()

    asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)

    # if template:
    #     # Smoothing the template with a spline (Aya)
    #     xs_temp = np.arange(len(asymmetry))
    #     asymmetry = scipy.interpolate.make_interp_spline(xs_temp[::15], asymmetry[::15], k=3)(xs_temp)
    
    return asymmetry

def max_moving_mad(asymmetry, width=100):
    '''Maximum moving median absolute deviation. Window width is in MHz.'''
    half_width_bin = int(width / freq_diff) // 2

    def mad(i):
        window = asymmetry[i-half_width_bin:i+half_width_bin]
        devs = np.abs(window)
        return np.median(devs)
    
    # Basic filter
    def filter(arr):
        length = len(arr)
        xs = np.arange(length)
        for x in xs:
            if x < (length / 8):
                yield 8 * x / length
            elif x > (7 * length / 8):
                yield 8 * (1 - x / length)
            else:
                yield 1
    
    mads = list(map(mad, range(500, len(asymmetry)-500)))
    filt = list(filter(mads))
    mads_filt = np.multiply(mads, filt)
    return np.max(mads_filt)    

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
        devs = np.abs(window) ** 2
        msds[i] = np.median(devs)
        filt[i] = filter(i, len(freqs))

    wammsd = np.average(msds, weights=filt)
    return wammsd #np.sqrt(wammsd) 

def moving_mad_mean(asymmetry, width=100):
    '''Average of moving median absolute deviations. Window width is in MHz.'''
    half_width_bin = int(width / freq_diff) // 2

    def mad_window(i):
        window = asymmetry[i-half_width_bin:i+half_width_bin]
        return scipy.stats.median_abs_deviation(window)
    
    mads = list(map(mad_window, range(half_width_bin, len(asymmetry)-half_width_bin)))
    return np.mean(mads)

@njit(cache=True)
def objective_mad_new(state, spectra, template_spectra, mask, freqs=None, freq_diff=0.18, factor=1e-3):
    '''The objective function.'''
    asymmetry = build_asymmetry(spectra, state, mask)
    template = build_asymmetry(template_spectra, state, mask)

    template_integral = np.sum(np.abs(template) ** 2)
    denom = wammsd_dynamic(asymmetry, freqs, freq_diff=freq_diff, factor=factor) 
    return template_integral / denom

def objective_mad_constraint(state, constraint=None, width=40):
    '''The objective function: WAMMSD subject to template_integral >= constraint.'''
    asymmetry = build_asymmetry(state)
    #template = build_asymmetry(state, template=True)

    #template_integral = np.sum(np.abs(template) ** 2) #template_peak = np.max(template)
    perf = sqrt_wammsd(asymmetry, width=width) 
    return 1 / perf# if (template_integral >= constraint) else 0

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
    gammas = np.zeros(n_gen)
    x_best = np.zeros(n_pop, dtype=np.int64)
    f_best = 0
    n_pop_init = n_pop

    #with Pool(processes=72) as pool:
    for i in range(n_gen): #trange(n_gen):
        # Generate vectors
        #xs = np.random.binomial(n=1, p=p0, size=(n_pop, len(p0)))
        xs = np.empty((n_pop, len(p0)), dtype=np.int64)
        for j in range(n_pop):
            for k in range(len(p0)):
                xs[j,k] = np.random.binomial(1, p0[k])

        # Calculate performances
        # f_partial = partial(f, spectra=spectra, template_spectra=template_spectra, freq_diff=freq_diff)
        # fs = np.array(list(map(f_partial, xs)))
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
        # Calculate new probabilities
        # p = np.mean(elite, axis=0)
        # p = np.sum(elite, axis=0) / elite.shape[0]
        # Save best performance
        f_cand = np.max(fs)
        print(f_cand)
        print(p0)
        perfs[i] = f_cand
        if f_cand > f_best:
            f_best = f_cand
            x_best = xs[np.argmax(fs)]
        
        # # FACE algorithm - adaptive tuning
        # if (gammas[i-1] >= gammas[i]) and (perfs[i-1] >= perfs[i]):
        #     # Convergence check
        #     if (i >= d) and np.all(f_cand == perfs[i-d:i]): # Last d iterations
        #         print(f'Converged at iteration {i}!')
        #         print('Reliable' if n_pop != (5 * n_pop_init) else 'Unreliable')
        #         # Truncate perfs at the current iteration
        #         perfs = perfs[:i]
        #         break
        #     elif n_pop < (5 * n_pop_init):
        #         n_pop += 1
        #         continue
        
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
            print('Reliable' if n_pop != (5 * n_pop_init) else 'Unreliable')
            # Truncate perfs at the current iteration
            perfs = perfs[:i]

        progress.update(1)
        
    return x_best, f_best, perfs

# def generate_movie(asym_bounds=(5000, 5200)):
#     fig = plt.figure(figsize=(10, 8))
#     gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], wspace=0.25)
#     ax1 = fig.add_subplot(gs[0,:], projection='aitoff')
#     ax2 = fig.add_subplot(gs[1,0])
#     ax3 = fig.add_subplot(gs[1,1])

#     ax1.grid(zorder=0)

#     inner_sc = ax1.scatter([], [], s=10, marker='*', c='k', label='inward')
#     outer_sc = ax1.scatter([], [], s=10, marker='*', c='gray', label='outward')
#     ax1.plot([-np.pi/2, -np.pi/2], [-np.pi/2, np.pi/2], c='r')
#     ax1.plot([np.pi/2, np.pi/2], [np.pi/2, -np.pi/2], c='r', label=r'$\phi=90^{\circ}$')

#     ax1.legend(bbox_to_anchor=(0.8, 0.1))

#     iters = ax2.plot(range(1, 100_001), stds, c='k')
#     ax2.set_xlim(1, 100_000)
#     ax2.set_xlabel('Iteration', fontsize=14)
#     ax2.set_ylabel('Objective', fontsize=14)
#     #ax2.set_yscale('log')
#     pt = ax2.scatter([], [], color='b', marker='x', s=50, zorder=2)

#     xmin = np.argmin(np.abs(xs-asym_bounds[0]))
#     xmax = np.argmin(np.abs(xs-asym_bounds[1]))
#     asym, = ax3.plot([], [], c='b')
#     ax3.set_xlim(xs[xmin], xs[xmax])
#     ax3.set_xlabel(r'$\nu$ [MHz]', fontsize=14)
#     ax3.set_ylabel(r'$A_I(\nu)$', fontsize=14)

#     text = ax1.set_title('n=0', fontsize=16)

#     def frame(idx):
#         state = states[idx]

#         inner = state == 1
#         outer = state == -1
#         inner_info = bank_1_info[inner]
#         outer_info = bank_1_info[outer]

#         inner_gal = SkyCoord(inner_info['l'], inner_info['b'], unit='deg', frame='galactic')
#         outer_gal = SkyCoord(outer_info['l'], outer_info['b'], unit='deg', frame='galactic')
        
#         inner_plot = np.array([inner_gal.l.wrap_at('180d').radian, inner_gal.b.radian]).T
#         outer_plot = np.array([outer_gal.l.wrap_at('180d').radian, outer_gal.b.radian]).T
#         inner_sc.set_offsets(inner_plot)
#         outer_sc.set_offsets(outer_plot)

#         asymmetry = build_asymmetry(state)
#         asym.set_data(xs[xmin:xmax], asymmetry[xmin:xmax])
#         ax3.set_ylim(1.1*np.percentile(asymmetry, 2), 1.1*np.percentile(asymmetry, 98))

#         pt.set_offsets([idx+1, stds[idx]])

#         text.set_text(f'i={idx+1}')

#         return inner_sc, outer_sc, pt, text, asym

#     def draw():
#         return frame(0) # i=1

#     def animate(i):
#         return frame((i+1)*100-1) # i=multiple of 100

#     anim = FuncAnimation(
#         fig=fig,
#         func=animate,
#         init_func=draw,
#         frames=1000,
#         blit=True
#     )

#     save_dir = os.path.join(opt_dir, 'movie.gif')
#     anim.save(save_dir, writer='pillow', fps=60, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of subdirectory in which to save results')
    parser.add_argument('-a', '--ann', action='store_true', help='optimize annihilation instead of decay')
    parser.add_argument('-d', '--doppler', action='store_true', help='optimize doppler asymmetry instead of intensity')
    # parser.add_argument('-m', '--movie', action='store_true', help='generate and save a movie of the search')
    args = parser.parse_args()
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    name = args.name
    doppler = args.doppler
    asym_type = 'Doppler' if doppler else 'Intensity'
    case = 'ann' if args.ann else 'decay'
    is_decay = case == 'decay'
    # movie = args.movie

    # Bank 1 info
    bank_1_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/heatmap/052926_01/{case}/1'
    print(bank_1_dir)
    info = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/all_cband_info.csv')
    bank_1_info = info[info['source'].apply(lambda l: l + '.npy').isin(os.listdir(bank_1_dir))]
    inner_all_mask = (bank_1_info['theta'] < 90) if doppler else (bank_1_info['phi'] < 90)
    inner_all_mask = inner_all_mask.to_numpy()

    # All bank 1 spectra, full range
    files = bank_1_info['source'].apply(lambda l: l + '.npy')
    freqs = np.load(os.path.join(bank_1_dir, files[0]))[0]
    freq_diff = freqs[1] - freqs[0]
    spectra = np.array([np.load(os.path.join(bank_1_dir, file))[1] for file in files])

    # Factor for dynamic wammsd
    factor = v_earth / c + (v_virial / (c*np.sqrt(3)) * 3) * (1 + v_earth / c)

    # Prepare template spectra
    opt_dir = os.path.join('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection', name)
    os.makedirs(opt_dir, exist_ok=True)

    if os.path.exists(os.path.join(opt_dir, 'template_spectra.npy')):
        print('Skipping template normalization')
        template_spectra = np.load(os.path.join(opt_dir, 'template_spectra.npy'))
    else:
        with Pool(processes=72) as p:
            template_spectra = np.array(list(tqdm(p.imap_unordered(normalized_spectrum_template, files), total=len(files), desc='Normalizing templates')))
        np.save(os.path.join(opt_dir, 'template_spectra.npy'), template_spectra)

    if os.path.exists(os.path.join(opt_dir, 'best_state.npy')):
        print('Optimization already complete; loading in results.')
        state_best = np.load(os.path.join(opt_dir, 'best_state.npy'))
        perfs = np.load(os.path.join(opt_dir, 'perfs.npy'))
    else:
        p0 = 0.5 * np.ones(len(bank_1_info))
        # Prepare probability vector - cut top 10% stds
        # p0 = np.zeros(len(bank_1_info)) # Vector of zero probabilities
        # spec_stds = np.std(spectra, axis=1)
        # std_cutoff = np.percentile(spec_stds, 90)
        # p0 = np.where(spec_stds > std_cutoff, 0, 0.5) # Set probabilities below the cutoff to 0.5
        # print(f'# spectra: {sum(p0 == 0.5)}') # Sanity check
        # sample = np.load('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/1000_random_targets_070626.npy')
        # p0 = np.where(sample == 1, 0.5, 0)
        # print(f'# spectra: {sum(p0 == 0.5)}') # Sanity check

        # p0 = np.where(spec_stds > std_cutoff, 0, 0.5) # Set probabilities below the cutoff to 0.5
        # print(f'# spectra: {sum(p0 == 0.5)}') # Sanity check
        
        # Weighted probabilities...
        # phis = bank_1_info['phi'].to_numpy()
        # dist = np.abs(90 - phis)
        # dist_standard = (dist - np.mean(dist)) / np.std(dist)
        # p0 = 1 / (1 + np.exp(-dist_standard))

        # Set constraint on template integral
        # state_aya = np.where(bank_1_info['phi'].between(70, 115), 0, 1)
        # constraint_template = build_asymmetry(state_aya, template=True)
        # constraint = np.sum(np.abs(constraint_template) ** 2) / 4
        # print(constraint)

        # Run optimization
        with ProgressBar(total=1000) as progress:
            state_best, f_best, perfs = cross_entropy(
                objective_mad_new,
                5000,
                0.01,
                1000,
                p0,
                10,
                spectra,
                template_spectra,
                inner_all_mask,
                progress,
                alpha=0.5,
                freqs=freqs,
                factor=factor,
                freq_diff=freq_diff
                #constraint=constraint
            )
        #state_best, noise_best, states, stds, Ts = simulated_annealing(objective_fourier, state_init, 100_000, 1.0, 0.99, adaptive=True)

        # Save optimization results
        np.save(os.path.join(opt_dir, 'best_state.npy'), state_best)
        np.save(os.path.join(opt_dir, 'perfs.npy'), perfs)

    # Save plot of best asymmetry
    asymmetry_best = build_asymmetry(spectra, state_best, inner_all_mask)
    fig, ax = plt.subplots()
    ax.plot(freqs, asymmetry_best)
    mad = scipy.stats.median_abs_deviation(asymmetry_best)
    ax.set_title(f'MAD = {mad:.3g}', fontsize=18)
    ax.set_xlabel(r'$\nu$ [MHz]', fontsize=16)
    ax.set_ylabel(f'{asym_type} Asymmetry', fontsize=16)
    ax.set_ylim(np.percentile(asymmetry_best, 2), np.percentile(asymmetry_best, 98))
    fig.savefig(os.path.join(opt_dir, 'best_asymmetry_plot.png'), dpi=300, bbox_inches='tight')
    np.save(os.path.join(opt_dir, 'best_asymmetry.npy'), asymmetry_best)

    # Save trace plot
    fig, ax = plt.subplots()
    ax.plot(range(1, len(perfs)+1), perfs, c='k')
    ax.set_xlim(1, len(perfs)+1)
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Performance', fontsize=14)
    fig.savefig(os.path.join(opt_dir, 'trace.png'), dpi=300, bbox_inches='tight')

    # Save sky plot of best state
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='aitoff')
    ax.grid(zorder=0)

    inner = (state_best == 1) & inner_all_mask
    outer = (state_best == 1) & ~inner_all_mask
    inner_info = bank_1_info[inner]
    outer_info = bank_1_info[outer]
    inner_gal = SkyCoord(inner_info['l'], inner_info['b'], unit='deg', frame='galactic')
    outer_gal = SkyCoord(outer_info['l'], outer_info['b'], unit='deg', frame='galactic')
    inner_sc = ax.scatter(inner_gal.l.wrap_at('180d').radian, inner_gal.b.radian, s=10, marker='*', c='k', label='forward')
    outer_sc = ax.scatter(outer_gal.l.wrap_at('180d').radian, outer_gal.b.radian, s=10, marker='*', c='gray', label='backward')
    
    if doppler:
        # DOPPLER
        ax.plot([0, 0], [np.pi/2, -np.pi/2], c='r', label=r'$\theta=90^{\circ}$')
    # INTENSITY
    else:
        ax.plot([-np.pi/2, -np.pi/2], [-np.pi/2, np.pi/2], c='r')
        ax.plot([np.pi/2, np.pi/2], [np.pi/2, -np.pi/2], c='r', label=r'$\phi=90^{\circ}$')

    ax.legend(bbox_to_anchor=(0.8, 0.1))
    fig.savefig(os.path.join(opt_dir, 'aitoff_best_state.png'), dpi=300, bbox_inches='tight')

    # Save 5650 MHZ SNR plot
    if doppler:
        # DOPPLER
        original_state = np.ones(len(bank_1_info))
    else:
        # INTENSITY
        inner_mask = bank_1_info['phi'] < 70
        outer_mask = bank_1_info['phi'] > 115
        original_state = np.zeros(len(bank_1_info))
        original_state[inner_mask | outer_mask] = 1

    # def build_asymmetry_for_plot(specs, state, mask):
    #     inner = (state == 1) & mask
    #     outer = (state == 1) & ~mask

    #     # inner_mean, outer_mean = get_means(specs, inner, outer)
    #     inner_slice = specs[inner, :]
    #     outer_slice = specs[outer, :]
    #     inner_mean = np.mean(inner_slice, axis=0)
    #     outer_mean = np.mean(outer_slice, axis=0)
        
    #     asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)

    #     return asymmetry
    
    inj_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/norm_1e29_5650_tests/injected'
    freqs = np.load(os.path.join(inj_dir, files[0]))[0]
    inj_spectra = np.array([np.load(os.path.join(inj_dir, file))[1] for file in files])

    aya_mask = (bank_1_info['theta'] < 65).to_numpy() if doppler else inner_all_mask
    asymmetry_inj = build_asymmetry(inj_spectra, original_state, aya_mask)
    asymmetry_inj_best = build_asymmetry(inj_spectra, state_best, inner_all_mask)

    fig, ax = plt.subplots()
    ax.plot(freqs, asymmetry_inj, label='before')
    ax.plot(freqs, asymmetry_inj_best, label='after')

    before_snr = calculate_asymmetry_snr(freqs, asymmetry_inj, 5650)
    after_snr = calculate_asymmetry_snr(freqs, asymmetry_inj_best, 5650)
    ax.text(0.03, 0.97, f'Before: SNR = {before_snr:.2f}\nAfter: SNR = {after_snr:.2f}',
             transform=ax.transAxes, ha='left', va='top', fontsize=16)
    
    ax.legend()
    ax.set_xlabel(r'$\nu$ [MHz]', fontsize=16)
    ax.set_ylabel(f'{asym_type} Asymmetry', fontsize=16)
    
    fig.savefig(os.path.join(opt_dir, '5650_snrs.png'), dpi=300, bbox_inches='tight')

    # # Generate movie
    # if movie:
    #     print('Generating movie...')
    #     generate_movie()
    #     print('Movie saved!')
