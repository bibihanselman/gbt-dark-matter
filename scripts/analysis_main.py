'''
Model-independent dark matter search.
Author: Benjamin Hanselman
Date: 23 July 2026
'''

import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import itertools
from scipy.interpolate import interp1d, make_interp_spline
import logging
from tqdm.auto import tqdm
from numba import njit, prange, set_num_threads
import matplotlib.colors as mcolors
from numba import jit, njit, prange, set_num_threads
from numba_progress import ProgressBar
from scipy.stats import combine_pvalues
from itertools import chain
import argparse
import yaml

from normalization_methods import *

from whack_a_mole import WhackAMole

num_processes = mp.cpu_count()

c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s
dm_profile = 'nfw'

##### NORMALIZATION #####

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

##### ASYMMETRIES AND P-VALUES #####

@njit(cache=True)
def build_asymmetry(specs, state, mask): #template=False
    '''Form asymmetry from a given state.'''
    inner = (state == 1) & mask
    outer = (state == 1) & ~mask

    inner_slice = specs[inner, :]
    outer_slice = specs[outer, :]
    inner_mean = np.zeros(specs.shape[1])
    outer_mean = np.zeros(specs.shape[1])
    
    for j in range(specs.shape[1]):
        inner_mean[j] = inner_slice[:, j].mean()
        outer_mean[j] = outer_slice[:, j].mean()

    asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)
    
    return asymmetry

def correlate_asymmetry(freqs, spectrum, template, ref_freq=5000):
    def stretch_template(template, stretch_factor):
        if template.shape[0] % 2 == 0:
            template = template[1:] # Make the number of points odd
        k = template.shape[0] // 2
        orig_x = np.arange(-k, k+1)
        interp_func = interp1d(orig_x, template, bounds_error=False, fill_value=0)
        new_k = ((2*k + 1)/stretch_factor - 1)/2 # 2k_1 + 1 = sf*(2k_0 + 1)
        new_x = np.linspace(-new_k, new_k, template.shape[0])

        return interp_func(new_x)

    def get_a(target, template):
        return np.dot(target, template) / (np.dot(template, template) + 1e-12)
    
    a_vals = np.zeros_like(spectrum)
    residuals = np.zeros_like(spectrum)
    lower = (len(template) - 1) // 2
    upper = (len(template) + 1) // 2
    for i in range(len(spectrum)):
        stretched = stretch_template(template, freqs[i] / ref_freq)
        
        if i < lower:
            x = lower - i
            stretched = stretched[x:]
            window = spectrum[:i+upper]
        elif i > (len(spectrum) - upper):
            x = len(spectrum) - upper - i
            stretched = stretched[:x]
            window = spectrum[i-lower:]
        else:
            window = spectrum[i-lower:i+upper]
        
        a = get_a(window, stretched) 
        a_vals[i] = a 
        residuals[i] = np.sqrt(np.mean((a*stretched-window)**2)) 

    return a_vals / (residuals + 1e-12)

def get_pvals_cont(inj, uninj):
    stds = []
    for i in range(len(uninj)):
        if i < 140:
            stds.append(np.std(uninj[:i+140]))
        elif i >= len(uninj) - 140:
            stds.append(np.std(uninj[i-140:]))
        else:
            stds.append(np.std(uninj[i-140:i+140]))
    
    sigmas = inj / stds
    pvals = list(map(lambda sigma: .5*math.erfc(sigma/np.sqrt(2)), sigmas))
    return pvals

def calculate_pvals(iter, freqs, signal_sizes, b=1, exterior=3, **kwargs):
    print(f'Iteration {iter}') 
    # Injections
    inj_spectra = np.empty((len(targs), 2, len(xs)))
    sefd = np.load(os.path.join(calibration_bank_dir, 'median.npy'))[1]
    for i, targ in enumerate(targs):
        inj_spectra[i] = inject_calibrated(
            targ,
            freqs,
            bank_info,
            data_dir=bank_dir,
            sefd=sefd,
            signal_sizes=signal_sizes, 
            is_decay=is_decay
        )

    # Normalization
    with ProgressBar(total=len(targs)) as progress:
        inj_norm_spectra = normalize_all_spectra(inj_spectra, progress, b=b, exterior=exterior, is_decay=is_decay)
    
    # Build asymmetries
    asymmetry_uninj_int = build_asymmetry(uninj_norm_spectra, state_int, int_mask)
    asymmetry_inj_int = build_asymmetry(inj_norm_spectra, state_int, int_mask)
    template_int = build_asymmetry(template_spectra, state_int, int_mask)
    template_smooth_int = make_interp_spline(template_xs[::15], template_int[::15], k=3)(template_xs)

    asymmetry_uninj_dopp = build_asymmetry(uninj_norm_spectra, state_dopp, dopp_mask)
    asymmetry_inj_dopp = build_asymmetry(inj_norm_spectra, state_dopp, dopp_mask)
    template_dopp = build_asymmetry(template_spectra, state_dopp, dopp_mask)
    template_smooth_dopp = make_interp_spline(template_xs[::15], template_dopp[::15], k=3)(template_xs)

    corr_uninj_int = correlate_asymmetry(xs, asymmetry_uninj_int, template_smooth_int, **kwargs)
    corr_inj_int = correlate_asymmetry(xs, asymmetry_inj_int, template_smooth_int, **kwargs)

    corr_uninj_dopp = correlate_asymmetry(xs, asymmetry_uninj_dopp, template_smooth_dopp, **kwargs)
    corr_inj_dopp = correlate_asymmetry(xs, asymmetry_inj_dopp, template_smooth_dopp, **kwargs)

    pvals_int = get_pvals_cont(corr_inj_int, corr_uninj_int)
    pvals_dopp = get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)
    pvals_comb = np.array([combine_pvalues([pval_int, pval_dopp])[1] for pval_int, pval_dopp in zip(pvals_int, pvals_dopp)]) #np.multiply(pvals_int, pvals_dopp)

    def get_pval(freq):
        idx = np.argmin(np.abs(freq - xs))
        pvals_comb_filt = pvals_comb[pvals_comb != 0]
        pval = np.min(pvals_comb_filt[idx-20:idx+20])
        return pval
    
    pvals = np.array(list(map(get_pval, freqs)))
            
    ################ PLOTTING IN CASE I NEED TO DEBUG ################
    
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    sfigs = fig.subfigures(1, 2, width_ratios=[2,1])
    
    axs = sfigs[0].subplots(2, 2, sharex=True)
    axs_pvals = sfigs[1].subplots(3, 1, sharex=True)
                                 
    axs[0][0].plot(xs, asymmetry_uninj_dopp, label='raw')
    axs[0][0].plot(xs, asymmetry_inj_dopp, label='injected')
    axs[0][0].set_title('Doppler')
    
    axs[0][1].plot(xs, asymmetry_uninj_int, label='raw')
    axs[0][1].plot(xs, asymmetry_inj_int, label='injected')
    axs[0][1].legend()
    axs[0][1].set_title('Intensity')
    
    axs[1][0].plot(xs, corr_uninj_dopp, c='gray', label='raw')
    axs[1][0].plot(xs, corr_inj_dopp, c='k', label='injected')
    
    axs[1][1].plot(xs, corr_uninj_int, c='gray', label='raw')
    axs[1][1].plot(xs, corr_inj_int, c='k', label='injected')
    axs[1][1].legend()
    
    pvals_list = [pvals_dopp, pvals_int, pvals_comb]
    titles = ['Doppler', 'Intensity', 'Combined']
    
    # p value plot
    for ax, pvals_cont, title in zip(axs_pvals.flat, pvals_list, titles):        
        i = 0
        while .5*math.erfc((i+1)/np.sqrt(2)) > min(pvals_cont):
            label = '$n\sigma$' if not i else None
            ax.axhline(.5*math.erfc((i+1)/np.sqrt(2)), c='red', ls='--', lw=0.5, label=label)
            i += 1
        for freq in freqs:
            ax.axvspan(freq-3, freq+3, color='gray', alpha=0.3, linewidth=0.5)
        ax.plot(xs, pvals_cont, c='k')
        ax.set_yscale('log')
        ax.set_ylim(min(pvals_cont), 1)
        ax.set_title(title)
        ax.legend()
    
    sfigs[0].suptitle('Asymmetries and Correlations')
    sfigs[1].suptitle('P-Values')
    
    save_path = os.path.join(root_bank_dir, 'plots')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'iter_{iter}.png'))
    
    ##################################################################
    
    # Save some data
    path_to_save = os.path.join(root_bank_dir, f'pvals/iter_{iter}')
    os.makedirs(path_to_save, exist_ok=True)
    
    np.save(os.path.join(path_to_save, 'doppler_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)]))
    np.save(os.path.join(path_to_save, 'intensity_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_int, corr_uninj_int)]))
    np.save(os.path.join(path_to_save, 'combined_pvals.npy'), np.array([xs, pvals_cont]))
    
    return pvals

def suggest(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    sign = (ys[-1] < three_sigma)
    xs_opp = xs[(ys < three_sigma) != sign]
    if len(xs_opp) == 0:
        return xs[-1] / 10 if sign else xs[-1] * 10
    else:
        x0, x1 = xs_opp[-1], xs[-1]
        return (x0 + x1) / 2

def get_limit(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    xs_above = xs[ys < three_sigma]
    xs_below = xs[ys > three_sigma]
    if not (xs_above.size and xs_below.size):
        # Pick the value closest to the 3-sigma crossing
        return xs[np.argmin(np.abs(ys - three_sigma))]
    above = np.where(xs == np.min(xs_above))[0][0]
    below = np.where(xs == np.max(xs_below))[0][0]
    # Linear interpolation
    return ((ys[below] - three_sigma) * xs[above] - (ys[above] - three_sigma) * xs[below]) / (ys[below] - ys[above])

def get_f(nu):
    kb = 1.38e-23
    h = 6.626e-34
    t_cmb = 2.725
    t_bkg = 1.19 * (1000 / nu) ** 2.62
    x_cmb = (h * nu * 1e6) / (kb * t_cmb)
    x_bkg = (h * nu * 1e6) / (kb * t_bkg)
    f_cmb = 1 / (np.exp(x_cmb) - 1)
    f_bkg = 1 / (np.exp(x_bkg) - 1)
    return f_cmb + f_bkg

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
    analysis_dir = config['analysis_dir']
    state_dir = config['population_dir']
    calibration_dir = config['calibration_dir']
    whack_dir = config['whack_save_dir']
    sefd = config.get('sefd', 10)
    size_init = float(config.get('size_init', 1e-30))
    ref_signal_size = float(config.get('ref_signal_size', 1e-28))
    case = config.get('case', 'decay')
    is_decay = case == 'decay'

    root_dir = os.path.join(analysis_dir, case)

    info = pd.read_csv(info_path)

    banks = os.listdir(data_dir)

    three_sigma = .5*math.erfc(3/np.sqrt(2))

    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    start_time = datetime.datetime.now()

    for bank in banks:
        print(f'Bank {bank} start!')
        root_bank_dir = os.path.join(root_dir, bank)
        os.makedirs(root_bank_dir, exist_ok=True)
        calibration_bank_dir = os.path.join(calibration_dir, bank)
        
        # Load in template spectra
        bank_dir = os.path.join(data_dir, bank)
        bank_info = info[info['source'].apply(lambda l: l + '.npy').isin(os.listdir(bank_dir))]
        
        # Load in xs
        xs = np.load(os.path.join(bank_dir, os.listdir(bank_dir)[0]))[0]
        freq_diff = xs[1] - xs[0]
        ref_freq = (xs[0] + xs[-1]) / 2 # Center of bank

        # Optimized states
        state_int_all = np.load(os.path.join(state_dir, case, bank, 'intensity/best_state.npy'))
        state_dopp_all = np.load(os.path.join(state_dir, case, bank, 'doppler/best_state.npy'))
        # Exclude the spectra appearing in neither state, and prepare states/masks for asymmetry formation
        state_mask = (state_int_all == 1) | (state_dopp_all == 1)
        bank_info = bank_info[state_mask]
        targs = bank_info['source'].apply(lambda l: l + '.npy')
        state_int, state_dopp = state_int_all[state_mask], state_dopp_all[state_mask]
        int_mask = (bank_info['phi'] < 90).to_numpy()
        dopp_mask = (bank_info['theta'] < 90).to_numpy()

        # Normalize uninjected spectra
        uninj_spectra = np.array([np.load(os.path.join(bank_dir, targ)) for targ in targs])
        if os.path.exists(os.path.join(root_bank_dir, 'uninj_norm_spectra.npy')):
            print('Loading uninj norm spectra')
            uninj_norm_spectra = np.load(os.path.join(root_bank_dir, 'uninj_norm_spectra.npy'))
        else:
            print('Normalizing uninjected')
            with ProgressBar(total=len(targs)) as progress:
                uninj_norm_spectra = normalize_all_spectra(uninj_spectra, progress, b=1, exterior=3)
            np.save(os.path.join(root_bank_dir, 'uninj_norm_spectra.npy'), uninj_norm_spectra)

        # Templates
        if os.path.exists(os.path.join(root_bank_dir, 'template_spectra.npy')):
            print('Loading template spectra')
            template_spectra = np.load(os.path.join(root_bank_dir, 'template_spectra.npy'))
        else:
            # Injections
            inj_spectra = np.empty((len(targs), 2, len(xs)))   
            for i, targ in enumerate(targs):
                inj_spectra[i] = inject_calibrated(
                    targ,
                    ref_freq,
                    bank_info,
                    data_dir=bank_dir,
                    sefd=sefd,
                    signal_sizes=ref_signal_size, 
                    is_decay=is_decay,
                    template=True
                )
            print('Normalizing templates')
            with ProgressBar(total=len(targs)) as progress:
                template_spectra = normalize_all_spectra(inj_spectra, progress, b=1, exterior=3, freq=ref_freq, width=100)
            np.save(os.path.join(root_bank_dir, 'template_spectra.npy'), template_spectra)
        
        template_xs = np.arange(template_spectra.shape[1])
        
        # Uninjected p-values
        print('Calculating uninjected p-values...')

        asymmetry_uninj_int = build_asymmetry(uninj_norm_spectra, state_int, int_mask)
        template_int = build_asymmetry(template_spectra, state_int, int_mask)
        template_smooth_int = make_interp_spline(template_xs[::15], template_int[::15], k=3)(template_xs)

        asymmetry_uninj_dopp = build_asymmetry(uninj_norm_spectra, state_dopp, dopp_mask)
        template_dopp = build_asymmetry(template_spectra, state_dopp, dopp_mask)
        template_smooth_dopp = make_interp_spline(template_xs[::15], template_dopp[::15], k=3)(template_xs)

        corr_uninj_int = correlate_asymmetry(xs, asymmetry_uninj_int, template_smooth_int, ref_freq=ref_freq)
        corr_uninj_dopp = correlate_asymmetry(xs, asymmetry_uninj_dopp, template_smooth_dopp, ref_freq=ref_freq)

        pvals_int = get_pvals_cont(corr_uninj_int, corr_uninj_int)
        pvals_dopp = get_pvals_cont(corr_uninj_dopp, corr_uninj_dopp)
        pvals_comb = np.array([combine_pvalues([pval_int, pval_dopp])[1] for pval_int, pval_dopp in zip(pvals_int, pvals_dopp)])

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(8,10), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.3)

        min_pval = min(chain(pvals_int, pvals_dopp, pvals_comb))
        for ax, pval, title in zip(axs, [pvals_int, pvals_dopp, pvals_comb], ['Intensity', 'Doppler', 'Combined']):
            ax.plot(xs, pval, c='k')
            ax.set_title(title, fontsize=18)
            
            i = 0
            while .5*math.erfc((i+1)/np.sqrt(2)) > min_pval:
                label = '$n\sigma$' if not i else None
                ax.axhline(.5*math.erfc((i+1)/np.sqrt(2)), c='r', ls='--', lw=0.5, label=label)
                i += 1

            ax.set_yscale('log')
            ax.set_ylim(min_pval, 1)
            if title == 'Combined':
                ax.legend(loc='lower right')

        axs[2].set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        axs[1].set_ylabel('$p$', fontsize=18)
        save_path = os.path.join(root_bank_dir, 'plots')
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, 'uninjected_pvals.png'), dpi=300, bbox_inches='tight')    

        # Save uninjected pvalues
        save_path = os.path.join(root_bank_dir, 'pvals')
        os.makedirs(save_path, exist_ok=True)
        save_pvals_dopp = np.array([xs, pvals_dopp])
        np.save(os.path.join(save_path, 'uninj_dopp_pvals.npy'), save_pvals_dopp)
        save_pvals_int = np.array([xs, pvals_int])
        np.save(os.path.join(save_path, 'uninj_int_pvals.npy'), save_pvals_int)
        save_pvals_comb = np.array([xs, pvals_comb])
        np.save(os.path.join(save_path, 'uninj_combined_pvals.npy'), save_pvals_comb)
        
        # Feed data into whack-a-mole
        # whack_save_path = os.path.join(root_bank_dir, 'whack_a_mole.pdf')
        os.makedirs(os.path.join(whack_dir, case), exist_ok=True)
        whack_save_path = os.path.join(whack_dir, case, f'bank_{bank}.pdf')
        if os.path.exists(whack_save_path):
            print('Skipping whack-a-mole!')
        else:
            print('Running whack-a-mole...')
            whack_config = {
                'freqs': xs,
                'uninj_norm_spectra': uninj_norm_spectra,
                'uninj_spectra': np.array([np.load(os.path.join(bank_dir, targ))[1] for targ in targs]),
                'template_spectra': template_spectra,
                'template_freq': ref_freq,
                'state_dopp': state_dopp,
                'state_int': state_int,
                'info': bank_info
            }
            whack = WhackAMole(whack_config)
            whack.run(save_path=whack_save_path)

        print('Analysis start!')

        start = int(round(xs[0] + 75, -1)) # Round to nearest 10
        stop = int(round(xs[-1] - 75, -1))
        spacing = 100
        step = 10
        
        start_freqs = np.arange(start, start+spacing, step)
        
        for start_freq in start_freqs:
            if os.path.exists(os.path.join(root_bank_dir, f'{start_freq}_limits.npy')):
                # Skip
                print(f'Skipping {start_freq} MHz.')
            else:
                print(f'Analyzing start freq of {start_freq} MHz')
                # Frequencies
                freqs = np.arange(start_freq, stop, spacing)

                # "Exponential" search
                s = [size_init] * len(freqs)
                signal_sizes, pvals = [], []

                for i in range(8):
                    print(s)
                    p = calculate_pvals(i, freqs, s, ref_freq=ref_freq)
                    print(p)
                    signal_sizes.append(s)
                    pvals.append(p.tolist())
                    s = [suggest(s_, p_) for s_, p_ in zip(zip(*signal_sizes), zip(*pvals))]

                # Save signal sizes and pvals at each frequency
                signal_sizes_arr, pvals_arr = np.array(signal_sizes).T, np.array(pvals).T
                for signal_sizes_, pvals_, freq in zip(signal_sizes_arr, pvals_arr, freqs):
                    save_arr = np.array([signal_sizes_, pvals_])
                    np.save(os.path.join(root_bank_dir, f'{freq}_pvals.npy'), save_arr)
                
                # Linearly interpolate to 3-sigma at each frequency
                lims = [get_limit(s_, p_) for s_, p_ in zip(zip(*signal_sizes), zip(*pvals))]
                limits = np.array([freqs, lims])
                np.save(os.path.join(root_bank_dir, f'{start_freq}_limits.npy'), limits)
        
        # Combine all limits
        limits_all = np.concatenate(
            [np.load(os.path.join(root_bank_dir, f'{start_freq}_limits.npy')) for start_freq in start_freqs],
            axis=1
        )
        sort_indices = np.argsort(limits_all[0])
        limits_all = limits_all[:, sort_indices]
        np.save(os.path.join(root_bank_dir, 'limits.npy'), limits_all)

        # Exclusion plot
        plot_freqs = np.arange(limits_all[0,0]-step/2, limits_all[0,-1]+step, step)
        fig, ax = plt.subplots(figsize=(8,4))

        ax.stairs(2 * limits_all[1], plot_freqs, color='k', label='1-photon states')
        ax.stairs(limits_all[1], plot_freqs, color='k', ls='--', label='2-photon states')
        get_f_func = np.vectorize(get_f)
        fs = get_f_func(limits_all[0])
        ax.stairs(2 / (1 + fs) * limits_all[1], plot_freqs, color='b', label='1-photon states with stimulated emission')

        ax.set_yscale('log')
        ax.set_xlim(limits_all[0,0]-step/2, limits_all[0,-1]+step/2)
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        ax.grid()

        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        ax.set_ylabel(r'$\lambda$ [s$^{-1}$]' if is_decay else r'$\langle \sigma v \rangle$ [cm$^3$s$^{-1}$]', fontsize=18)
        fig.savefig(os.path.join(root_bank_dir, 'plots', 'exclusion.png'), dpi=300, bbox_inches='tight')    

    end_time = datetime.datetime.now()
    delta = relativedelta(end_time, start_time)
    print(f'Study complete in {delta.hours} hrs, {delta.minutes} mins, {delta.seconds} sec')