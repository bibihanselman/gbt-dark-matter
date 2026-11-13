'''
Model-independent dark matter search: limit finder.
Author: Benjamin Hanselman
Date: 13 November 2026
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
from scipy.integrate import quad

from whack_a_mole import WhackAMole

num_processes = mp.cpu_count()

c = 299792
v_virial = 250 # km/s
v_earth = 225 # km/s
dm_profile = 'nfw'

##### TEMPLATES #####

### Integrals

R_s = 7.9 # distance from the center of galaxy to the sun

# Parameters for Navarro-Frenk-White
rho_c_nfw = 1.4e7
r_c_nfw = 16.1


def r_prime(r, b, l):
    # find the coordinate-transformed radius
    return math.sqrt(r*r + R_s*R_s - 2*R_s*r*math.cos(b)*math.cos(l))

def nfw(r_prime):
    ratio = r_prime / r_c_nfw
    return rho_c_nfw / (ratio * (1+ratio) * (1+ratio))

def integrate(func, target_b_deg, target_l_deg, annihilation=False):
    target_b, target_l = math.radians(target_b_deg), math.radians(target_l_deg)
    def integrand(r):
        val = func(r_prime(r, target_b, target_l))
        return val*val if annihilation else val
    return quad(integrand, 0, float('inf'))

decay_amp = integrate(nfw, 90, 90)[0]
ann_amp = integrate(nfw, 90, 90, annihilation=True)[0]

decay_std = v_virial / (c*np.sqrt(3))
ann_std = v_virial / (c*np.sqrt(6))

def inject_template(xs, targ, info, is_decay=True):
    amp_ratio = decay_amp if is_decay else ann_amp
    col_name = 'nfw' if is_decay else 'nfw_sq'
    std = decay_std if is_decay else ann_std

    bank_info = info.set_index('time')

    theta = bank_info.loc[targ[:-4], 'theta']
    radians_from_v_earth = np.radians(theta)
    shift_factor = v_earth / c * np.cos(radians_from_v_earth)
    sig_loc = shift_factor
    integral = bank_info.loc[targ[:-4], col_name]
    sig_amp =  integral / amp_ratio * 1e-2 * (1 + shift_factor)
    sig_std = std * (1 + shift_factor)
    spec = 1 + sig_amp*np.exp(-(xs - sig_loc)**2 / (2*sig_std**2))

    injected = np.array([xs, spec])
    return injected

##### INJECTION #####

def inject_calibrated(
    name,
    sig_locs,
    info,
    data_dir=None,
    save_dir=None,
    sefd=None,
    signal_sizes=1e-28,
    is_decay=True,
):
    '''General routine for signal injection.'''
    freq, spec = np.load(os.path.join(data_dir, name))
    
    targ_info = info.set_index('time')
    
    theta = targ_info.loc[name[:-4], 'theta']
    shift_factor = get_shift_factor(theta, v_earth)
        
    if sefd is None:
        sefd = 10 * np.ones_like(spec)

    if not isinstance(sefd, (list, np.ndarray)):
        sefd = sefd * np.ones_like(spec)

    if not isinstance(sig_locs, (list, np.ndarray)):
        sig_locs = [sig_locs]

    if not isinstance(signal_sizes, (list, np.ndarray)):
        signal_sizes = [signal_sizes] * len(sig_locs)  
    
    if len(sig_locs) != len(signal_sizes):
        raise ValueError("sig_locs and signal_sizes must have the same length.")

    for sig_loc, signal_size in zip(sig_locs, signal_sizes):
        # Calculate line intensity
        if is_decay:
            sig_std = v_virial*sig_loc/(c*np.sqrt(3))
            sig_amp_ratio = 2.75e24*signal_size / (v_virial*(sig_loc/1e3)**3) # convert MHz to GHz # Units: Jy
            col_name = dm_profile
        else:
            sig_std = v_virial*sig_loc/(c*np.sqrt(6))
            sig_amp_ratio = 9.41e14*signal_size / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz # Units: Jy
            col_name = dm_profile+'_sq'
        
        # Multiply by line-integrated halo density
        sig_amp = sig_amp_ratio * targ_info.loc[name[:-4], col_name]
        
        # Doppler shift
        sig_loc, sig_std, sig_amp = sig_loc * shift_factor, sig_std * shift_factor, sig_amp / shift_factor
        
        # Add the Gaussian to the spectrum
        g = sig_amp*np.exp(-(freq - sig_loc)**2 / (2*sig_std**2))
        spec = spec * (sefd + g) / sefd
    
    # Save/return the injected spectrum
    injected = np.array([freq, spec])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, name), injected)
    return injected

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
def normalize_weighted_numba(data, a=0.1, b=1, exterior=3, order=5, is_decay=True, template=False):
    """
    Standard rolling polynomial normalize procedure. Updated.
    """
    xs, ys = data
    
    freq_diff = xs[1] - xs[0]
    ys_copy = ys.copy()
    if not template:
        for i in range(16, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
            ys[i] = np.nan
            ys[i+15] = np.nan
            ys[i+1] = np.nan
            ys[i+14] = np.nan
        
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
    for i, x in enumerate(xs):
        center = 1 if template else x
        sigma = get_sigma(center)
        window = round_up_to_nearest_odd(2 * exterior * sigma / freq_diff)
        
        if (i < (window//2)) or (i >= len(xs) - (window//2 + 1)): 
            normalized_spectrum[i] = 1
            continue
            
        if np.isnan(ys_copy[i]): # checking if the spectrum was nan BEFORE excising the coarse channel valleys
            normalized_spectrum[i] = np.nan
            continue

        ind_xs = np.arange(-(window//2), window//2+1).astype(np.float64)

        current_ys = ys[i-window//2:i+window//2+1]
        idx = np.isfinite(current_ys)
        
        # Unweighted fit
        unweighted_p = fit_poly(ind_xs[idx], current_ys[idx], order)
        
        # Weighted fit
        weights = 1 - a*np.exp(-(freq_diff*ind_xs[idx])**2 / (2*b*sigma**2))
        weighted_p = fit_poly(ind_xs[idx], current_ys[idx], order, w=weights)

        unweighted = eval_polynomial(unweighted_p, 0) #np.poly1d(unweighted_params)
        weighted = eval_polynomial(weighted_p, 0) #np.poly1d(weighted_params)

        normalized_spectrum[j] = unweighted/weighted
        
    normalized = np.zeros((2, new_xs.shape[0]))
    normalized[0] = new_xs
    normalized[1] = normalized_spectrum

    return normalized

@njit(parallel=True)
def normalize_all_spectra(spectra, progress=None, b=1, exterior=3, is_decay=True, template=False):
    norm_spectra = np.empty((spectra.shape[0], spectra.shape[2]))
    for i in prange(norm_spectra.shape[0]):
        norm = normalize_weighted_numba(spectra[i], b=b, exterior=exterior, is_decay=is_decay, template=template)
        norm_spectra[i] = norm[1]
        if progress:
            progress.update(1)
    return norm_spectra

##### QUALITY CUT #####

@njit(nogil=True)
def quality_cut(freqs, spectra, progress=None, freq_diff=1500/8192, rho=0.1):
    mads = np.empty_like(spectra)

    for i in prange(spectra.shape[0]):
        for j in range(len(freqs)):
            half_width_bin = int(2 * factor * freqs[j] / freq_diff)
            window = spectra[i][max(0, j-half_width_bin):min(len(freqs)-1, j+half_width_bin)]
            devs = np.zeros(len(window))
            for k in range(len(window)):
                devs[k] = np.abs(window[k]-1)
            mads[i, j] = np.nanmedian(devs)
        
        if progress:
            progress.update(1)
    
    quantiles = np.empty_like(mads)
    for i in prange(mads.shape[1]):
        quantile = np.nanquantile(mads[:, i], 1-rho)
        quantiles[:, i] = quantile
    new_spectra = np.where(mads > quantiles, np.nan, spectra)
    
    return new_spectra

##### ASYMMETRIES AND P-VALUES #####

@njit(cache=True)
def build_asymmetry(specs, state, mask, time_mask=None):
    '''Form asymmetry from a given state.'''

    if time_mask is not None:
        inner_mean_pre = np.nanmean(specs[(state == 1) & mask & time_mask], axis=0)
        inner_mean_post = np.nanmean(specs[(state == 1) & mask & ~time_mask], axis=0)
        outer_mean_pre = np.nanmean(specs[(state == 1) & ~mask & time_mask], axis=0)
        outer_mean_post = np.nanmean(specs[(state == 1) & ~mask & ~time_mask], axis=0)

        asymmetry = (inner_mean_pre + inner_mean_post - outer_mean_pre - outer_mean_post) / (inner_mean_pre + inner_mean_post + outer_mean_pre + outer_mean_post)

    else:
        inner_mean = np.nanmean(specs[(state == 1) & mask], axis=0)
        outer_mean = np.nanmean(specs[(state == 1) & ~mask], axis=0)

        asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)

    return asymmetry

def correlate_asymmetry(xs, template_xs, spectrum, template):

    def get_a(target, template):
        target = np.nan_to_num(target)
        return np.dot(target, template) / (np.dot(template, template) + 1e-12)
    
    a_vals = np.zeros_like(spectrum)

    for i, freq in enumerate(xs):
        half_width_bin = float((3 * factor * freq) // freq_diff)
        new_xs = np.arange(-half_width_bin, half_width_bin+1) * freq_diff + freq
        stretched = np.interp(new_xs, freq * template_xs, template)

        lower = (len(template) - 1) // 2
        upper = (len(template) + 1) // 2
        
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

    return a_vals

def get_pval_spectrum(xs, inj, uninj):
    stds = np.zeros_like(uninj)

    for i, freq in enumerate(xs):
        half_width_bin = (6 * factor * freq) // freq_diff
        if i < half_width_bin:
            window = uninj[:i+half_width_bin+1]
        elif i >= len(xs) - half_width:
            window = uninj[i-half_width_bin:]
        else:
            window = uninj[i-half_width_bin:i+half_width_bin+1]
        stds[i] = np.nanstd(window)
    
    sigmas = inj / stds
    pvals = list(map(lambda sigma: .5*math.erfc(sigma/np.sqrt(2)), sigmas))
    return pvals

def calculate_pvals(iter, freqs, signal_sizes, b=1, exterior=3, bank=0, **kwargs):
    print(f'Iteration {iter}')

    # Injections
    inj_spectra = np.empty((len(targs), 2, len(xs)))
    if bank < 6:
        sefd_all = np.load(os.path.join(calibration_bank_dir, 'median_smoothed.npy'))[1]
    else:
        path_pre = os.path.join(calibration_bank_dir, 'pre23_median_smoothed.npy')
        path_post = os.path.join(calibration_bank_dir, 'post23_median_smoothed.npy')
        if os.path.exists(path_pre):
            sefd_pre = np.load(path_pre)[1]
        if os.path.exists(path_post):
            sefd_post = np.load(path_post)[1]

    for i, targ in enumerate(targs):
        if bank < 6:
            sefd = sefd_all
        else:
            sefd = sefd_pre if time_mask[i] else sefd_post
        
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
    inj_norm_spectra = normalize_all_spectra(inj_spectra, b=b, exterior=exterior, is_decay=is_decay)
    
    # Build asymmetries
    asymmetry_uninj_int = build_asymmetry(uninj_norm_spectra, state_int, int_mask, time_mask=time_mask)
    asymmetry_inj_int = build_asymmetry(inj_norm_spectra, state_int, int_mask, time_mask=time_mask)
    template_int = build_asymmetry(template_spectra, state_int, int_mask, time_mask=time_mask)

    asymmetry_uninj_dopp = build_asymmetry(uninj_norm_spectra, state_dopp, dopp_mask, time_mask=time_mask)
    asymmetry_inj_dopp = build_asymmetry(inj_norm_spectra, state_dopp, dopp_mask, time_mask=time_mask)
    template_dopp = build_asymmetry(template_spectra, state_dopp, dopp_mask, time_mask=time_mask)

    corr_uninj_int = correlate_asymmetry(xs, template_xs, asymmetry_uninj_int, template_int)
    corr_inj_int = correlate_asymmetry(xs, template_xs, asymmetry_inj_int, template_int)

    corr_uninj_dopp = correlate_asymmetry(xs, template_xs, asymmetry_uninj_dopp, template_dopp)
    corr_inj_dopp = correlate_asymmetry(xs, template_xs, asymmetry_inj_dopp, template_dopp)

    pvals_int = get_pval_spectrum(xs, corr_inj_int, corr_uninj_int)
    pvals_dopp = get_pval_spectrum(xs, corr_inj_dopp, corr_uninj_dopp)
    pvals_comb = np.array([combine_pvalues([pval_int, pval_dopp])[1] for pval_int, pval_dopp in zip(pvals_int, pvals_dopp)]) #np.multiply(pvals_int, pvals_dopp)

    def get_pval(freq):
        idx = np.argmin(np.abs(freq - xs))
        pvals_comb_filt = np.where(pvals_comb == 0, np.nan, pvals_comb)
        pval = np.nanmin(pvals_comb_filt[idx-20:idx+20])
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
    
    save_path = os.path.join(root_bank_dir, 'plots', str(freqs[0])) # tail: start freq
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'iter_{iter}.png'))
    
    ##################################################################
    
    # Save some data
    path_to_save = os.path.join(root_bank_dir, 'pvals', str(freqs[0]), f'iter_{iter}')
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
    size_init = float(config.get('size_init', 1e-30))
    ref_signal_size = float(config.get('ref_signal_size', 1e-2))
    case = config.get('case', 'decay')
    is_decay = case == 'decay'

    root_dir = os.path.join(analysis_dir, case)

    info = pd.read_csv(info_path, sep='\t')

    banks = sorted([int(bank) for bank in os.listdir(data_dir)])

    three_sigma = .5*math.erfc(3/np.sqrt(2))

    blanking_ranges = [
        [
            (750, 1320),
            (1520, 1660),
            (1880, 2260)
        ],
        [
            (1650, 1950),
            (2170, 2260),
            (2310, 2520),
            (2720, 3160)
        ],
        [
            (3560, 3950), #4200
            (7970, 8440)
        ],
        [
            (7500, 8160),
            (10430, 10600),
            (11500, 12380)
        ]
    ]
    start_banks = [0, 1, 2, 6]

    start_time = time.perf_counter()

    for bank in banks:
        print(f'Bank {bank} start!')
        root_bank_dir = os.path.join(root_dir, str(bank))
        os.makedirs(root_bank_dir, exist_ok=True)
        calibration_bank_dir = os.path.join(calibration_dir, str(bank))
        
        # Load in template spectra
        bank_dir = os.path.join(data_dir, str(bank))
        bank_info = info[info['source'].apply(lambda l: l + '.npy').isin(os.listdir(bank_dir))]
        
        # Load in xs
        xs = np.load(os.path.join(bank_dir, os.listdir(bank_dir)[0]))[0]

        # Optimized states
        state_int_all = np.load(os.path.join(state_dir, case, str(bank), 'intensity/best_state.npy'))
        state_dopp_all = np.load(os.path.join(state_dir, case, str(bank), 'doppler/best_state.npy'))
        # Exclude the spectra appearing in neither state, and prepare states/masks for asymmetry formation
        state_mask = (state_int_all == 1) | (state_dopp_all == 1)
        bank_info = bank_info[state_mask]
        targs = bank_info['source'].apply(lambda l: l + '.npy')
        state_int, state_dopp = state_int_all[state_mask], state_dopp_all[state_mask]
        int_mask = (bank_info['phi'] < 90).to_numpy()
        dopp_mask = (bank_info['theta'] < 90).to_numpy()
        time_mask = bank_info['epoch'].apply(lambda x: int(x[4:6]) < 23).to_numpy() if bank >= 6 else None

        # Normalize uninjected spectra
        uninj_spectra = np.array([np.load(os.path.join(bank_dir, targ)) for targ in targs])

        band_idx = max([i for i, start_bank in enumerate(start_banks) if start_bank <= bank])
        condition = np.any(
            [
                (xs > blank[0]) & (xs < blank[1])
                for blank in blanking_ranges[band_idx]
            ],
            axis=0
        )

        if os.path.exists(os.path.join(root_bank_dir, 'uninj_norm_spectra.npy')):
            print('Loading uninj norm spectra')
            uninj_norm_spectra = np.load(os.path.join(root_bank_dir, 'uninj_norm_spectra.npy'))
        else:
            print('Normalizing uninjected')
            
            # Normalize
            with ProgressBar(total=len(targs)) as progress:
                uninj_norm_spectra = normalize_all_spectra(uninj_spectra, progress=progress, b=1, exterior=3)
            
            # Quality cut
            with ProgressBar(total=len(targs)) as progress:
                uninj_norm_spectra = quality_cut(xs, uninj_norm_spectra, progress=progress)            
            
            # Blanking ranges
            uninj_norm_spectra = np.where(condition, np.nan, uninj_norm_spectra)

            # Save!
            np.save(os.path.join(root_bank_dir, 'uninj_norm_spectra.npy'), uninj_norm_spectra)

        # Templates
        template_xs = np.linspace(1-3*factor, 1+3*factor, 500)
        if os.path.exists(os.path.join(root_bank_dir, 'template_spectra.npy')):
            print('Loading template spectra')
            template_spectra = np.load(os.path.join(root_bank_dir, 'template_spectra.npy'))
        else:
            # Injections
            inj_spectra = np.array([inject_template(xs, targ, bank_info) for targ in targs])
            print('Normalizing templates')
            with ProgressBar(total=len(files)) as progress:
                template_spectra = normalize_all_spectra(inj_spectra, progress, b=1, exterior=3, is_decay=is_decay)
            np.save(os.path.join(root_bank_dir, 'template_spectra.npy'), template_spectra)
                
        # Uninjected p-values
        print('Calculating uninjected p-values...')

        asymmetry_uninj_int = build_asymmetry(uninj_norm_spectra, state_int, int_mask, time_mask=time_mask)
        template_int = build_asymmetry(template_spectra, state_int, int_mask, time_mask=time_mask)

        asymmetry_uninj_dopp = build_asymmetry(uninj_norm_spectra, state_dopp, dopp_mask, time_mask=time_mask)
        template_dopp = build_asymmetry(template_spectra, state_dopp, dopp_mask, time_mask=time_mask)

        corr_uninj_int = correlate_asymmetry(xs, template_xs, asymmetry_uninj_int, template_int)
        corr_uninj_dopp = correlate_asymmetry(xs, template_xs, asymmetry_uninj_dopp, template_smooth_dopp)

        pvals_int = get_pvals_cont(xs, corr_uninj_int, corr_uninj_int)
        pvals_dopp = get_pvals_cont(xs, corr_uninj_dopp, corr_uninj_dopp)
        pvals_comb = np.array([combine_pvalues([pval_int, pval_dopp])[1] for pval_int, pval_dopp in zip(pvals_int, pvals_dopp)])

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(8,10), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.3)

        min_pval = np.nanmin(np.array([pvals_int, pvals_dopp, pvals_comb]))
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
                'state_dopp': state_dopp,
                'state_int': state_int,
                'info': bank_info,
                'blanking_ranges': blanking_ranges[band_idx]
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
                power = 3 if is_decay else 4
                s = [size_init * (freq / 1e3) ** power for freq in freqs]
                signal_sizes, pvals = [], []

                for i in range(8):
                    print(s)
                    p = calculate_pvals(i, freqs, s, bank=bank)
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
        # get_f_func = np.vectorize(get_f)
        # fs = get_f_func(limits_all[0])
        # ax.stairs(2 / (1 + fs) * limits_all[1], plot_freqs, color='b', label='1-photon states with stimulated emission')

        ax.set_yscale('log')
        ax.set_xlim(limits_all[0,0]-step/2, limits_all[0,-1]+step/2)
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        ax.grid()

        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        ax.set_ylabel(r'$\lambda$ [s$^{-1}$]' if is_decay else r'$\langle \sigma v \rangle$ [cm$^3$s$^{-1}$]', fontsize=18)
        fig.savefig(os.path.join(root_bank_dir, 'plots', 'exclusion.png'), dpi=300, bbox_inches='tight')    

    t = datetime.timedelta(seconds=time.perf_counter() - start_time)
    print(f'Study complete in {t}')