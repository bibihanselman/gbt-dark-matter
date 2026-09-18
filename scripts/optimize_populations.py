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
#from numba import njit, prange
import concurrent.futures

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

def next_neighbor(x):
    '''"Flip-or-swap" method.'''
    x_new = x.copy()

    # Pick inner or outer
    if random.random() < 0.5:
        inds = [i for i, s in enumerate(inner_all_mask) if s]
        #inner = True
    else:
        inds = [i for i, s in enumerate(outer_all_mask) if s]
        #inner = False
    
    # Pick flip or swap
    idx = random.choice(inds)
    if random.random() < 0.5: # Flip
        x_new[idx] = (1 - x_new[idx])# if inner else -(1 + x_new[idx])
    else: # Swap
        inds2 = [i for i in inds if x_new[i] != x_new[idx]]
        idx2 = random.choice(inds2)
        x_new[idx], x_new[idx2] = x_new[idx2], x_new[idx]
    
    return x_new

# def next_neighbor(x):
#     '''Bit-flip method.'''
#     x_new = x.copy()

#     # Pick inner or outer
#     if random.random() < 0.5:
#         inds = [i for i, s in enumerate(inner_all_mask) if s]
#         inner = True
#     else:
#         inds = [i for i, s in enumerate(outer_all_mask) if s]
#         inner = False
    
#     idx = random.choice(inds)
#     x_new[idx] = (1 - x_new[idx]) if inner else -(1 + x_new[idx])
    
#     return x_new

# @njit(cache=True)
# def get_means(specs, inner, outer):
#     inner_slice = specs[inner, :]
#     outer_slice = specs[outer, :]
    
#     inner_mean = np.zeros(specs.shape[1])
#     outer_mean = np.zeros(specs.shape[1])
    
#     for j in range(specs.shape[1]):
#         inner_mean[j] = inner_slice[:, j].mean()
#         outer_mean[j] = outer_slice[:, j].mean()
        
#     return inner_mean, outer_mean

def build_asymmetry(state, template=False):
    '''Form asymmetry from a given state.'''
    # inner = state == 1
    # outer = state == -1
    inner = (state == 1) & inner_all_mask
    outer = (state == 1) & outer_all_mask

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

def objective_mad_new(state):
    '''The objective function: template peak / MAD.'''
    asymmetry = build_asymmetry(state)
    template = build_asymmetry(state, template=True)

    template_peak = np.max(template)
    mad = scipy.stats.median_abs_deviation(asymmetry)

    return template_peak / mad

def objective_fourier(state):
    '''Another objective function: square integral of template over spectral power.'''
    asymmetry = build_asymmetry(state)
    template = build_asymmetry(state, template=True)

    template_integral = np.sum(np.abs(template) ** 2)
    fft = np.fft.rfft(asymmetry)
    power = np.sum(np.abs(fft) ** 2)

    return template_integral / power

def simulated_annealing(f, x0, n_iter, T, cooling_rate, adaptive=False):
    '''Implementation of simulated annealing discrete optimization algorithm.'''
    f_best = f(x0)
    x_best = x0

    xs = [x_best]
    fs = [f_best]
    Ts = [T]

    print(f'Initial f: {f_best}')

    for i in trange(1, n_iter, desc='Optimizing'):
        # Generate a new candidate via neighborhood function
        x_new = next_neighbor(x_best)

        # Evaluate the objective
        f_new = f(x_new)
        
        # Delta
        delta = f_new - f_best

        # Accept or reject the candidate
        if delta > 0: # Maximizing
            x_best = x_new
            f_best = f_new
        else:
            p = np.exp(delta / T)
            if random.random() < p:
                x_best = x_new
                f_best = f_new
        
        # Save state
        xs.append(x_new)
        fs.append(f_new)

        # Decrease temperature every 100 iterations
        if i % 100 == 0:
            if adaptive and (fs[-1] - fs[-101]) <= 0:
                T /= cooling_rate
            else:
                T *= cooling_rate
            Ts.append(T)
    
    print(f'Final f: {f_best}')

    # Return best state and full chain
    return x_best, f_best, xs, fs, Ts

def eval_neighbor(x, f=None):
    x_new = next_neighbor(x)
    f_new = f(x_new)
    return x_new, f_new

def parallel_simulated_annealing(f, x0, n_iter, T, cooling_rate, adaptive=False):
    '''Implementation of simulated annealing discrete optimization algorithm with parallelization.'''
    f_best = f(x0)
    x_best = x0

    xs = [x_best]
    fs = [f_best]
    Ts = [T]

    print(f'Initial f: {f_best}')

    eval = partial(eval_neighbor, f=f)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for i in trange(1, n_iter, desc='Optimizing'):
            # Generate 10 new candidates via neighborhood function
            results = executor.map(eval, [x_best for _ in range(10)])

            # Pick the best candidate
            x_new, f_new = max(list(results), key=lambda r: r[1])
            
            # Delta
            delta = f_new - f_best

            # Accept or reject the candidate
            if delta > 0: # Maximizing
                x_best = x_new
                f_best = f_new
            else:
                p = np.exp(delta / T)
                if random.random() < p:
                    x_best = x_new
                    f_best = f_new
            
            # Save state
            xs.append(x_new)
            fs.append(f_new)

            # Decrease temperature every 100 iterations
            if i % 100 == 0:
                if adaptive and (fs[-1] - fs[-101]) <= 0:
                    T /= cooling_rate
                else:
                    T *= cooling_rate
                Ts.append(T)
    
    print(f'Final f: {f_best}')

    # Return best state and full chain
    return x_best, f_best, xs, fs, Ts

def generate_movie(asym_bounds=(5000, 5200)):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], wspace=0.25)
    ax1 = fig.add_subplot(gs[0,:], projection='aitoff')
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.grid(zorder=0)

    inner_sc = ax1.scatter([], [], s=10, marker='*', c='k', label='inward')
    outer_sc = ax1.scatter([], [], s=10, marker='*', c='gray', label='outward')
    ax1.plot([-np.pi/2, -np.pi/2], [-np.pi/2, np.pi/2], c='r')
    ax1.plot([np.pi/2, np.pi/2], [np.pi/2, -np.pi/2], c='r', label=r'$\phi=90^{\circ}$')

    ax1.legend(bbox_to_anchor=(0.8, 0.1))

    iters = ax2.plot(range(1, 100_001), stds, c='k')
    ax2.set_xlim(1, 100_000)
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('Objective', fontsize=14)
    #ax2.set_yscale('log')
    pt = ax2.scatter([], [], color='b', marker='x', s=50, zorder=2)

    xmin = np.argmin(np.abs(xs-asym_bounds[0]))
    xmax = np.argmin(np.abs(xs-asym_bounds[1]))
    asym, = ax3.plot([], [], c='b')
    ax3.set_xlim(xs[xmin], xs[xmax])
    ax3.set_xlabel(r'$\nu$ [MHz]', fontsize=14)
    ax3.set_ylabel(r'$A_I(\nu)$', fontsize=14)

    text = ax1.set_title('n=0', fontsize=16)

    def frame(idx):
        state = states[idx]

        # inner = state == 1
        # outer = state == -1
        inner = (state == 1) & inner_all_mask
        outer = (state == 1) & outer_all_mask
        inner_info = bank_1_info[inner]
        outer_info = bank_1_info[outer]

        inner_gal = SkyCoord(inner_info['l'], inner_info['b'], unit='deg', frame='galactic')
        outer_gal = SkyCoord(outer_info['l'], outer_info['b'], unit='deg', frame='galactic')
        
        inner_plot = np.array([inner_gal.l.wrap_at('180d').radian, inner_gal.b.radian]).T
        outer_plot = np.array([outer_gal.l.wrap_at('180d').radian, outer_gal.b.radian]).T
        inner_sc.set_offsets(inner_plot)
        outer_sc.set_offsets(outer_plot)

        asymmetry = build_asymmetry(state)
        asym.set_data(xs[xmin:xmax], asymmetry[xmin:xmax])
        ax3.set_ylim(1.1*np.percentile(asymmetry, 2), 1.1*np.percentile(asymmetry, 98))

        pt.set_offsets([idx+1, stds[idx]])

        text.set_text(f'i={idx+1}')

        return inner_sc, outer_sc, pt, text, asym

    def draw():
        return frame(0) # i=1

    def animate(i):
        return frame((i+1)*100-1) # i=multiple of 100

    anim = FuncAnimation(
        fig=fig,
        func=animate,
        init_func=draw,
        frames=1000,
        blit=True
    )

    save_dir = os.path.join(opt_dir, 'movie.gif')
    anim.save(save_dir, writer='pillow', fps=60, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of subdirectory in which to save results')
    parser.add_argument('-m', '--movie', action='store_true', help='generate and save a movie of the search')
    args = parser.parse_args()
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    name = args.name
    movie = args.movie

    # Bank 1 info
    bank_1_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/heatmap/052926_01/decay/1'
    info = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/all_cband_info.csv')
    bank_1_info = info[info['source'].apply(lambda l: l + '.npy').isin(os.listdir(bank_1_dir))]
    inner_all_mask = bank_1_info['phi'] < 90
    outer_all_mask = bank_1_info['phi'] > 90

    # All bank 1 spectra, full range
    files = bank_1_info['source'].apply(lambda l: l + '.npy')
    xs = np.load(os.path.join(bank_1_dir, files[0]))[0]
    spectra = np.array([np.load(os.path.join(bank_1_dir, file))[1] for file in files])

    # Prepare template spectra
    opt_dir = os.path.join('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection', name)
    os.makedirs(opt_dir, exist_ok=True)
    #np.save(os.path.join(opt_dir, 'spectra.npy'), spectra)

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
        states = np.load(os.path.join(opt_dir, 'states.npy'))
        stds = np.load(os.path.join(opt_dir, 'stds.npy'))
    else:
        # Initial state
        # inner_mask = bank_1_info['phi'] < 70
        # outer_mask = bank_1_info['phi'] > 115
        # state_init = np.zeros(len(bank_1_info))
        # state_init[inner_mask] = 1
        # state_init[outer_mask] = -1
        state_init = np.load('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/063026/02/best_state.npy')

        # Run optimization
        state_best, noise_best, states, stds, Ts = simulated_annealing(objective_mad_new, state_init, 100_000, 100, 0.99, adaptive=True)

        # Save optimization results
        np.save(os.path.join(opt_dir, 'best_state.npy'), state_best)
        np.save(os.path.join(opt_dir, 'states.npy'), states)
        np.save(os.path.join(opt_dir, 'stds.npy'), stds)
        np.save(os.path.join(opt_dir, 'temps.npy'), Ts)

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

    # Generate movie
    if movie:
        print('Generating movie...')
        generate_movie()
        print('Movie saved!')
