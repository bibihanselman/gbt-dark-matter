import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from dateutil.relativedelta import relativedelta
import itertools
from scipy.interpolate import interp1d, make_interp_spline, PchipInterpolator, Akima1DInterpolator
from scipy.optimize import brentq
import logging
from tqdm.auto import tqdm
from numba import njit, prange, set_num_threads
from scipy.special import erfcinv
import matplotlib.colors as mcolors
from numba import jit, njit, prange, set_num_threads
from numba_progress import ProgressBar
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import combine_pvalues
from itertools import chain

from normalization_methods import *

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

def correlate_asymmetry(freqs, spectrum, template):
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

@njit(parallel=True)
def normalize_all_spectra(spectra, progress, b=1, exterior=3, freq=None, width=100):
    # Normalize the first spectrum to get the right shape (and warm up the function)
    test_spec = spectra[0]
    test_norm = normalize_weighted_numba(test_spec, b=b, exterior=exterior, freq=freq, width=width)
    norm_spectra = np.empty((spectra.shape[0], test_norm.shape[1]))
    for i in prange(norm_spectra.shape[0]):
        norm = normalize_weighted_numba(spectra[i], b=b, exterior=exterior, freq=freq, width=width)
        norm_spectra[i] = norm[1]
        progress.update(1)
    return norm_spectra

def calculate_corrs(iter, freqs, signal_sizes, b=1, exterior=3, **kwargs):
    print(f'Iteration {iter}') 
    # Injections
    inj_spectra = np.empty((len(targs), spec.shape[0], spec.shape[1]))   
    for i, targ in enumerate(targs):
        inj_spectra[i] = inject_calibrated_cband(
            targ,
            freqs,
            bank_1_info,
            data_dir=uninjected_dir,
            calibration_dir=calibration_dir,
            signal_sizes=signal_sizes, 
            is_decay=is_decay
        )

    # Normalization
    with ProgressBar(total=len(targs)) as progress:
        inj_norm_spectra = normalize_all_spectra(inj_spectra, progress, b=b, exterior=exterior)
    
    # Build asymmetries
    asymmetry_uninj_int = build_asymmetry(uninj_norm_spectra, state_int, int_mask)
    asymmetry_inj_int = build_asymmetry(inj_norm_spectra, state_int, int_mask)
    template_int = build_asymmetry(template_spectra, state_int, int_mask)
    template_smooth_int = make_interp_spline(template_xs[::15], template_int[::15], k=3)(template_xs)

    asymmetry_uninj_dopp = build_asymmetry(uninj_norm_spectra, state_dopp, dopp_mask)
    asymmetry_inj_dopp = build_asymmetry(inj_norm_spectra, state_dopp, dopp_mask)
    template_dopp = build_asymmetry(template_spectra, state_dopp, dopp_mask)
    template_smooth_dopp = make_interp_spline(template_xs[::15], template_dopp[::15], k=3)(template_xs)

    corr_uninj_int = correlate_asymmetry(xs, asymmetry_uninj_int, template_smooth_int)
    corr_inj_int = correlate_asymmetry(xs, asymmetry_inj_int, template_smooth_int)

    corr_uninj_dopp = correlate_asymmetry(xs, asymmetry_uninj_dopp, template_smooth_dopp)
    corr_inj_dopp = correlate_asymmetry(xs, asymmetry_inj_dopp, template_smooth_dopp)

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
    
    save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/iter_{iter}.png'
    os.makedirs(save_path.replace(save_path.split('/')[-1], ''), exist_ok=True)
    plt.savefig(save_path)
    
    ##################################################################
    
    # Save some data
    path_to_save = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/pvals/iter_{iter}'
    os.makedirs(path_to_save, exist_ok=True)
    
    np.save(os.path.join(path_to_save, 'doppler_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)]))
    np.save(os.path.join(path_to_save, 'intensity_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_int, corr_uninj_int)]))
    np.save(os.path.join(path_to_save, 'combined_pvals.npy'), np.array([xs, pvals_cont]))
    
    corrs = np.sqrt(2) * erfcinv(2 * pvals) #np.array(list(map(lambda pval: np.sqrt(2) * erfcinv(2 * pval), pvals)))
    return corrs

def plot_freq_pvals(freq, limit):
    signal_sizes, corrs = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/{int(freq)}_corrs.npy')

    fig, ax = plt.subplots(figsize=(8,5))

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(signal_sizes)))
    for i in range(len(signal_sizes)):
        ax.scatter(signal_sizes[i], corrs[i], marker='x', color=colors[i], s=100)

    bounds = np.arange(len(signal_sizes)+1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    discrete_cmap = mcolors.ListedColormap(colors)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps['viridis']), ax=ax) # Ticks in the middle of intervals
    cbar.set_label('Iteration', fontsize=18)
    cbar.set_ticks(bounds[:-1]+0.5)
    cbar.set_ticklabels(bounds[:-1])

    fit = Polynomial.fit(signal_sizes, corrs, 1)
    b, m = fit.convert().coef
    r = np.corrcoef(signal_sizes, corrs)[0][1]
    ax.plot(signal_sizes, fit(signal_sizes), c='k', ls=':', label=f'm={m:.2g}, b={b:.2f}, r={r:.2f}')

    ax.axhline(3, c='red', ls='--')
    ax.set_xlabel('$\lambda$ [s$^{-1}$]', fontsize=18)
    ax.set_ylabel(r'$S(\nu) / \sigma$', fontsize=18)
    ax.set_title(rf'$\nu$ = {int(freq)} MHz; $\lambda$ = {limit:.3g}', fontsize=18)
    ax.legend()

    return fig

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    is_decay = True
    ref_freq = 5650

    case = 'decay' if is_decay else 'ann'
    
    order = 5

    dir_name = f'analysis_tests/bank_1_071826_exp_{case}'
    root_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}'
    os.makedirs(root_dir, exist_ok=True)

    start_time = datetime.datetime.now()

    # Load in template spectra
    info = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/all_cband_info.csv')
    bank_1_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/1'
    bank_1_info = info[info['source'].apply(lambda l: l + '.npy').isin(os.listdir(bank_1_dir))]

    # Frequencies
    freqs = np.arange(4850, 6100, 100)
    # Optimized states
    state_int_all = np.load('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/071626/03/best_state.npy')
    state_dopp_all = np.load('/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/071626/04/best_state.npy')
    # Exclude the spectra appearing in neither state, and prepare states/masks for asymmetry formation
    state_mask = (state_int_all == 1) | (state_dopp_all == 1)
    targs = bank_1_info.loc[state_mask, 'source'].apply(lambda l: l + '.npy')
    state_int, state_dopp = state_int_all[state_mask], state_dopp_all[state_mask]
    int_mask = (bank_1_info.loc[state_mask, 'phi'] < 90).to_numpy()
    dopp_mask = (bank_1_info.loc[state_mask, 'theta'] < 90).to_numpy()

    # Normalized uninjected spectra
    uninjected_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/1'
    uninj_spectra = np.array([np.load(os.path.join(uninjected_dir, targ)) for targ in targs])
    if os.path.exists(os.path.join(root_dir, 'uninj_norm_spectra.npy')):
        print('Loading uninj norm spectra')
        uninj_norm_spectra = np.load(os.path.join(root_dir, 'uninj_norm_spectra.npy'))
    else:
        print('Normalizing uninjected')
        with ProgressBar(total=len(targs)) as progress:
            uninj_norm_spectra = normalize_all_spectra(uninj_spectra, progress, b=1, exterior=3)
        np.save(os.path.join(root_dir, 'uninj_norm_spectra.npy'), uninj_norm_spectra)

    # Load in xs
    spec = uninj_spectra[0]
    xs = spec[0]

    # Templates
    if os.path.exists(os.path.join(root_dir, 'template_spectra.npy')):
        print('Loading template spectra')
        template_spectra = np.load(os.path.join(root_dir, 'template_spectra.npy'))
    else:
        # Injections
        inj_spectra = np.empty((len(targs), spec.shape[0], spec.shape[1]))   
        for i, targ in enumerate(targs):
            inj_spectra[i] = inject_calibrated_cband(
                targ,
                ref_freq, # 5650 MHz
                bank_1_info,
                data_dir=uninjected_dir, # Note: no calibration dir!
                signal_sizes=1e-29, 
                is_decay=is_decay,
                template=True
            )
        print('Normalizing templates')
        with ProgressBar(total=len(targs)) as progress:
            template_spectra = normalize_all_spectra(inj_spectra, progress, b=1, exterior=3, freq=ref_freq, width=100)
        np.save(os.path.join(root_dir, 'template_spectra.npy'), template_spectra)
    
    inj_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/norm_1e29_5650_tests/injected'
    template_xs = np.load(os.path.join(inj_dir, targs[0]))[0]
    
    # Uninjected p-values
    print('Calculating uninjected p-values...')

    asymmetry_uninj_int = build_asymmetry(uninj_norm_spectra, state_int, int_mask)
    template_int = build_asymmetry(template_spectra, state_int, int_mask)
    template_smooth_int = make_interp_spline(template_xs[::15], template_int[::15], k=3)(template_xs)

    asymmetry_uninj_dopp = build_asymmetry(uninj_norm_spectra, state_dopp, dopp_mask)
    template_dopp = build_asymmetry(template_spectra, state_dopp, dopp_mask)
    template_smooth_dopp = make_interp_spline(template_xs[::15], template_dopp[::15], k=3)(template_xs)

    corr_uninj_int = correlate_asymmetry(xs, asymmetry_uninj_int, template_smooth_int)
    corr_uninj_dopp = correlate_asymmetry(xs, asymmetry_uninj_dopp, template_smooth_dopp)

    pvals_int = get_pvals_cont(corr_uninj_int, corr_uninj_int)
    pvals_dopp = get_pvals_cont(corr_uninj_dopp, corr_uninj_dopp)
    pvals_comb = np.multiply(pvals_int, pvals_dopp)
    pvals_fisher = np.array([combine_pvalues([pval_int, pval_dopp])[1] for pval_int, pval_dopp in zip(pvals_int, pvals_dopp)])

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8,10), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3)

    min_pval = min(chain(pvals_int, pvals_dopp, pvals_comb, pvals_fisher))
    for ax, pval, title in zip(axs, [pvals_int, pvals_dopp, pvals_comb], ['Intensity', 'Doppler', 'Combined']):
        ax.plot(xs, pval, c='k', label='multiply' if (title=='Combined') else None)
        if title == 'Combined':
            ax.plot(xs, pvals_fisher, c='orange', label='fisher')
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
    save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/'
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, 'uninjected_pvals.png'), dpi=300, bbox_inches='tight')    

    # Save uninjected pvalues
    save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/pvals/'
    os.makedirs(save_path, exist_ok=True)
    save_pvals_dopp = np.array([xs, pvals_dopp])
    np.save(os.path.join(save_path, 'uninj_dopp_pvals.npy'), save_pvals_dopp)
    save_pvals_int = np.array([xs, pvals_int])
    np.save(os.path.join(save_path, 'uninj_int_pvals.npy'), save_pvals_int)
    save_pvals_comb = np.array([xs, pvals_fisher])
    np.save(os.path.join(save_path, 'uninj_combined_pvals.npy'), save_pvals_comb)
    
    print('Mini analysis start!')

    calibration_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/median_sefds/1'

    def suggest(xs, ys):
        xs, ys = np.array(xs), np.array(ys)
        sign = (ys[-1] > 3.0)
        xs_opp = xs[(ys > 3.0) != sign]
        if len(xs_opp) == 0:
            return xs[-1] / 10 if sign else xs[-1] * 10
        else:
            x0, x1 = xs_opp[-1], xs[-1]
            return (x0 + x1) / 2

    if os.path.exists(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/{freqs[0]}_corrs.npy'):
        # Skip
        limits = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/limits.npy')
    else:
        # "Exponential" search
        s = [1e-30] * len(freqs)
        signal_sizes, corrs = [], []

        for i in range(8):
            print(s)
            c = calculate_corrs(i, freqs, s)
            signal_sizes.append(s)
            corrs.append(c.tolist())
            s = [suggest(s_, c_) for s_, c_ in zip(zip(*signal_sizes), zip(*corrs))]

        # x0 = 30.0 * np.ones_like(freqs) # Starting signal size
        # print(x0)
        # y0 = calculate_corrs(0, freqs, 10 ** -x0)
        
        # x1 = -np.log10((10 ** -x0) * 3 / y0) # Regula falsi
        # print(x1)
        # y1 = calculate_corrs(1, freqs, 10 ** -x1)

        # signal_sizes = [x0, x1]
        # corrs = [y0, y1]
        # # 3 more iterations: secant method in -log(signal size)
        # for i in range(3):
        #     x2 = x1 - (y1 - 3) * (x1 - x0) / (y1 - y0)
        #     print(x2)
        #     y0 = y1
        #     y1 = calculate_corrs(i+2, freqs, 10 ** -x2)
        #     x0, x1 = x1, x2
        #     signal_sizes.append(x1)
        #     corrs.append(y1)
        
        # Regula falsi
        # signal_sizes, corrs = [], []
        # x_ = 1e-30 * np.ones_like(freqs) # Starting signal size
        # for i in range(5):
        #     y_ = calculate_corrs(i, freqs, x_)
        #     signal_sizes.append(x_)
        #     corrs.append(y_)
        #     x_ = x_ * 3 / y_
        
        # Get limits with one more interpolation
        # x_final = x1 - (y1 - 3) * (x1 - x0) / (y1 - y0)
        # limits = np.array([freqs, 10 ** -x_final]) # convert log
        limits = np.array([freqs, s]) # The binary search suggestion after the final iteration
        np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/limits.npy', limits)

        # Save signal sizes and pvals at each frequency
        signal_sizes_arr, corrs_arr = np.array(signal_sizes).T, np.array(corrs).T
        for signal_sizes, corrs, freq in zip(signal_sizes_arr, corrs_arr, freqs):
            save_arr = np.array([signal_sizes, corrs])
            np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/{freq}_corrs.npy', save_arr)

    # Save a pdf of combined correlation vs. signal size for all freqs
    pdf = PdfPages(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/results.pdf')

    for freq, limit in zip(*limits):
        fig = plot_freq_pvals(freq, limit)
        pdf.savefig(fig, bbox_inches='tight', dpi=500)
        plt.close(fig)
    
    pdf.close()

    end_time = datetime.datetime.now()
    delta = relativedelta(end_time, start_time)
    print(f'Study complete in {delta.hours} hrs, {delta.minutes} mins, {delta.seconds} sec')