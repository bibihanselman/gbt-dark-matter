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
import concurrent.futures
import time

os.chdir('/home/bhanselman/bhanselman/scripts')
from normalization_methods import *


def normalized_spectrum_template(file):
    inj_data = inject_calibrated_cband(
        file,
        5650, # reference frequency; doesn't really matter.
        info,
        data_dir='/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/1',
        calibration_dir='/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/median_sefds/1',
        signal_sizes=1e-29,
        template=True
    )
    norm_data = normalize_weighted(
        inj_data,
        freq=5650,
        width=100,
        b=1,
        exterior=3
    )
    return norm_data[1]

def calculate_asymmetry_snr(xs, inj, freq, window=10):
    mask = abs(xs - freq) < window
    window_ys = inj[mask]
    signal = np.max(window_ys)
    noise = scipy.stats.median_abs_deviation(inj[~mask]) #np.std(uninj)
    return signal / noise

def build_asymmetry(state, template=False):
    '''Form asymmetry from a given state.'''
    inner = (state == 1) & inner_all_mask
    outer = (state == 1) & ~inner_all_mask

    specs = template_spectra if template else spectra

    # inner_mean, outer_mean = get_means(specs, inner, outer)
    inner_slice = specs[inner, :]
    outer_slice = specs[outer, :]
    inner_mean = np.mean(inner_slice, axis=0)
    outer_mean = np.mean(outer_slice, axis=0)

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

def sqrt_wammsd(asymmetry, width=100):
    '''Square root of weighted average moving median squared deviations. Window width is in MHz.'''
    half_width_bin = int(width / freq_diff) // 2

    def msd(i):
        window = asymmetry[i-half_width_bin:i+half_width_bin]
        devs = np.abs(window) ** 2
        return np.median(devs)
    
    # Basic filter
    def filter(arr):
        length = len(arr)
        xs = np.arange(length)
        filt = np.zeros(length)
        for i, x in enumerate(xs):
            if x < (length / 8):
                filt[i] = 8 * x / length
            elif x > (7 * length / 8):
                filt[i] = 8 * (1 - x / length)
            else:
                filt[i] = 1
        return filt
    
    msds = list(map(msd, range(half_width_bin, len(asymmetry)-half_width_bin)))

    filt = filter(msds)
    wammsd = np.average(msds, weights=filt)
    return np.sqrt(wammsd) 

def moving_mad_mean(asymmetry, width=100):
    '''Average of moving median absolute deviations. Window width is in MHz.'''
    half_width_bin = int(width / freq_diff) // 2

    def mad_window(i):
        window = asymmetry[i-half_width_bin:i+half_width_bin]
        return scipy.stats.median_abs_deviation(window)
    
    mads = list(map(mad_window, range(half_width_bin, len(asymmetry)-half_width_bin)))
    return np.mean(mads)

def objective_mad_new(state, beta=50, width=100):
    '''The objective function: template peak / MAD - kurtosis penalty.'''
    asymmetry = build_asymmetry(state)
    template = build_asymmetry(state, template=True)

    template_peak = np.max(template)
    # mad = scipy.stats.median_abs_deviation(asymmetry)
    # devs = np.abs(asymmetry[1024:len(asymmetry)-1024] - np.median(asymmetry[1024:len(asymmetry)-1024])) # Test
    # mad = np.median(devs)
    #mmm = moving_mad_mean(asymmetry, width=width)
    denom = sqrt_wammsd(asymmetry, width=width) 
    #max_moving_mad(asymmetry, width=width) #sqrt_wammsd(asymmetry, width=width)

    # Kurtosis penalty
    # q1, q3 = np.percentile(asymmetry, 25), np.percentile(asymmetry, 75)
    # iqr = q3 - q1
    # asymmetry_cut = asymmetry[(asymmetry > q1 - 3*iqr) & (asymmetry < q3 + 3*iqr)] # Important: 3*IQR
    # kurtosis = scipy.stats.kurtosis(asymmetry_cut)

    return template_peak / denom #mad #- beta * kurtosis
    # std = np.std(asymmetry)
    
    # return template_peak / std

def cross_entropy(f, n_pop, rho, n_gen, p0, alpha=1):
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
    with Pool(processes=15) as pool:
        for i in trange(n_gen):
            # Generate vectors
            xs = np.random.binomial(n=1, p=p0, size=(n_pop, len(p0)))
            # Calculate performances
            fs = np.array(list(pool.imap(f, xs)))
            # Elite sample
            gamma = np.quantile(fs, rho)
            elite = xs[fs >= gamma]
            # Calculate new probabilities
            p = np.mean(elite, axis=0)
            p0 = alpha * p + (1 - alpha) * p0 # Smooth updating
            # Save best performance
            f_cand = np.max(fs)
            perfs[i] = f_cand
            if f_cand > f_best:
                f_best = f_cand
                x_best = xs[np.argmax(fs)]
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
    # parser.add_argument('-m', '--movie', action='store_true', help='generate and save a movie of the search')
    args = parser.parse_args()
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    name = args.name
    # movie = args.movie

    # Bank 1 info
    bank_1_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/heatmap/052926_01/decay/1'
    info = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/all_cband_info.csv')
    bank_1_info = info[info['source'].apply(lambda l: l + '.npy').isin(os.listdir(bank_1_dir))]
    inner_all_mask = bank_1_info['phi'] < 90
    inner_all_mask = inner_all_mask.to_numpy()

    # All bank 1 spectra, full range
    files = bank_1_info['source'].apply(lambda l: l + '.npy')
    xs = np.load(os.path.join(bank_1_dir, files[0]))[0]
    freq_diff = xs[1] - xs[0]
    spectra = np.array([np.load(os.path.join(bank_1_dir, file))[1] for file in files])

    # Prepare template spectra
    opt_dir = os.path.join('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection', name)
    os.makedirs(opt_dir, exist_ok=True)

    if os.path.exists(os.path.join(opt_dir, 'template_spectra.npy')):
        print('Skipping template normalization')
        template_spectra = np.load(os.path.join(opt_dir, 'template_spectra.npy'))
    else:
        with Pool(processes=10) as p:
            template_spectra = np.array(list(tqdm(p.imap(normalized_spectrum_template, files), total=len(files), desc='Normalizing templates')))
        np.save(os.path.join(opt_dir, 'template_spectra.npy'), template_spectra)

    if os.path.exists(os.path.join(opt_dir, 'best_state.npy')):
        print('Optimization already complete; loading in results.')
        state_best = np.load(os.path.join(opt_dir, 'best_state.npy'))
        perfs = np.load(os.path.join(opt_dir, 'perfs.npy'))
    else:
        # p0 = 0.5 * np.ones(len(bank_1_info))
        # Prepare probability vector - cut top 10% stds
        # p0 = np.zeros(len(bank_1_info)) # Vector of zero probabilities
        # spec_stds = np.std(spectra, axis=1)
        # std_cutoff = np.percentile(spec_stds, 90)
        # p0 = np.where(spec_stds > std_cutoff, 0, 0.5) # Set probabilities below the cutoff to 0.5
        # print(f'# spectra: {sum(p0 == 0.5)}') # Sanity check
        sample = np.load('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/1000_random_targets_070626.npy')
        p0 = np.where(sample == 1, 0.5, 0)
        print(f'# spectra: {sum(p0 == 0.5)}') # Sanity check

        # p0 = np.where(spec_stds > std_cutoff, 0, 0.5) # Set probabilities below the cutoff to 0.5
        # print(f'# spectra: {sum(p0 == 0.5)}') # Sanity check

        # Run optimization
        state_best, f_best, perfs = cross_entropy(objective_mad_new, 5000, 0.99, 1000, p0, alpha=0.25)
        #state_best, noise_best, states, stds, Ts = simulated_annealing(objective_fourier, state_init, 100_000, 1.0, 0.99, adaptive=True)

        # Save optimization results
        np.save(os.path.join(opt_dir, 'best_state.npy'), state_best)
        np.save(os.path.join(opt_dir, 'perfs.npy'), perfs)

    # Save plot of best asymmetry
    asymmetry_best = build_asymmetry(state_best)
    fig, ax = plt.subplots()
    ax.plot(xs, asymmetry_best)
    mad = scipy.stats.median_abs_deviation(asymmetry_best)
    ax.set_title(f'MAD = {mad:.3g}', fontsize=18)
    ax.set_xlabel(r'$\nu$ [MHz]', fontsize=16)
    ax.set_ylabel('Intensity Asymmetry', fontsize=16)
    ax.set_ylim(np.percentile(asymmetry_best, 2), np.percentile(asymmetry_best, 98))
    fig.savefig(os.path.join(opt_dir, 'best_asymmetry_plot.png'), dpi=300, bbox_inches='tight')
    np.save(os.path.join(opt_dir, 'best_asymmetry.npy'), asymmetry_best)

    # Save trace plot
    fig, ax = plt.subplots()
    ax.plot(range(1, 1001), perfs, c='k')
    ax.set_xlim(1, 1001)
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
    inner_sc = ax.scatter(inner_gal.l.wrap_at('180d').radian, inner_gal.b.radian, s=10, marker='*', c='k', label='inward')
    outer_sc = ax.scatter(outer_gal.l.wrap_at('180d').radian, outer_gal.b.radian, s=10, marker='*', c='gray', label='outward')
    
    ax.plot([-np.pi/2, -np.pi/2], [-np.pi/2, np.pi/2], c='r')
    ax.plot([np.pi/2, np.pi/2], [np.pi/2, -np.pi/2], c='r', label=r'$\phi=90^{\circ}$')

    ax.legend(bbox_to_anchor=(0.8, 0.1))
    fig.savefig(os.path.join(opt_dir, 'aitoff_best_state.png'), dpi=300, bbox_inches='tight')

    # Save 5650 MHZ SNR plot
    inner_mask = bank_1_info['phi'] < 70
    outer_mask = bank_1_info['phi'] > 115
    original_state = np.zeros(len(bank_1_info))
    original_state[inner_mask | outer_mask] = 1

    def build_asymmetry_for_plot(specs, state):
        inner = (state == 1) & inner_all_mask
        outer = (state == 1) & ~inner_all_mask

        # inner_mean, outer_mean = get_means(specs, inner, outer)
        inner_slice = specs[inner, :]
        outer_slice = specs[outer, :]
        inner_mean = np.mean(inner_slice, axis=0)
        outer_mean = np.mean(outer_slice, axis=0)
        
        asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)

        return asymmetry
    
    inj_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/norm_1e29_5650_tests/injected'
    xs = np.load(os.path.join(inj_dir, files[0]))[0]
    inj_spectra = np.array([np.load(os.path.join(inj_dir, file))[1] for file in files])

    asymmetry_inj = build_asymmetry_for_plot(inj_spectra, original_state)
    asymmetry_inj_best = build_asymmetry_for_plot(inj_spectra, state_best)

    fig, ax = plt.subplots()
    ax.plot(xs, asymmetry_inj, label='before')
    ax.plot(xs, asymmetry_inj_best, label='after')

    before_snr = calculate_asymmetry_snr(xs, asymmetry_inj, 5650)
    after_snr = calculate_asymmetry_snr(xs, asymmetry_inj_best, 5650)
    ax.text(0.03, 0.97, f'Before: SNR = {before_snr:.2f}\nAfter: SNR = {after_snr:.2f}',
             transform=ax.transAxes, ha='left', va='top', fontsize=16)
    
    ax.legend()
    ax.set_xlabel(r'$\nu$ [MHz]', fontsize=16)
    ax.set_ylabel('Intensity Asymmetry', fontsize=16)
    
    fig.savefig(os.path.join(opt_dir, '5650_snrs.png'), dpi=300, bbox_inches='tight')

    # # Generate movie
    # if movie:
    #     print('Generating movie...')
    #     generate_movie()
    #     print('Movie saved!')