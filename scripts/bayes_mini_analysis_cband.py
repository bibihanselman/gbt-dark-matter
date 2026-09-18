#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
import random
from functools import partial
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_interp_spline
import matplotlib as mpl
import datetime
from dateutil.relativedelta import relativedelta
import itertools
from scipy.interpolate import interp1d, make_interp_spline, PchipInterpolator, Akima1DInterpolator
from scipy.optimize import brentq
import logging
import gc
from tqdm.auto import tqdm, trange
from bayes_opt import BayesianOptimization, acquisition

from normalization_methods import *

num_processes = 20#mp.cpu_count()

info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/all_cband_info.csv'
info = pd.read_csv(info_dir)

c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s
dm_profile = 'nfw'

def sigma_filter(data, window_size=15):
    for i in range(window_size//2, len(data)-window_size//2):
        window_data = data[i-window_size//2:i+window_size//2+1]
        median = np.median(window_data)
        std = np.std(window_data)
        if abs(data[i] - median) > 3 * std:
            data[i] = median
    return data

def build_asymmetry(data_dir, angle, lower, upper):
    spectra = os.listdir(data_dir)
    inner = info[info[angle] < lower]
    outer = info[info[angle] > upper]
    inner_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in inner['source'].tolist()])
    outer_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in outer['source'].tolist()])
    inner_mean = np.mean(inner_spectra, axis=0)
    outer_mean = np.mean(outer_spectra, axis=0)
    asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)
    xs = np.load(os.path.join(data_dir, spectra[0]))[0]
    return xs, asymmetry

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
  
    
def sigma_filter(data, window_size=15):
    for i in range(window_size//2, len(data)-window_size//2):
        window_data = data[i-window_size//2:i+window_size//2+1]
        median = np.median(window_data)
        std = np.std(window_data)
        if abs(data[i] - median) > 3 * std:
            data[i] = median
    return data

    
def calculate_pval(normalize_func, freq, data_dir=None, save_dir=None, i=0, b=1, exterior=3, **kwargs):
    spectra = os.listdir(data_dir)
    
    skip = (os.path.exists(save_dir)) and (len(os.listdir(save_dir)) == len(spectra))

    if skip:
        print(f'Freq {freq} at iteration {dir_name.split("/")[-1]} already normalized. Skipping!')
    else:
        with mp.Pool(processes=num_processes) as p:
            normalize_func_partial = partial(
                normalize_func,
                data_dir=data_dir,
                save_dir=save_dir,
                freq=freq,
                b=b,
                exterior=exterior,
                **kwargs
            )
            _ = list(p.imap_unordered(normalize_func_partial, spectra))

    # Build asymmetries
    final_asym_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/uninjected_asymmetries/asymmetries/{freq}/b_{b:.2f}_ext_{exterior:.2f}'
    uninj_dopp = np.load(os.path.join(final_asym_dir, 'doppler.npy'))[1]
    uninj_int = np.load(os.path.join(final_asym_dir, 'intensity.npy'))[1]      
    xs, inj_dopp = build_asymmetry(save_dir, 'theta', *theta_cutoffs)
    xs, inj_int = build_asymmetry(save_dir, 'phi', *phi_cutoffs)

    # Calculating correlation spectra
    final_corr_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/uninjected_asymmetries/correlations/{freq}/b_{b:.2f}_ext_{exterior:.2f}'
    corr_uninj_dopp = np.load(os.path.join(final_corr_dir, 'doppler.npy'))[1]
    corr_uninj_int = np.load(os.path.join(final_corr_dir, 'intensity.npy'))[1]
    
    final_temp_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b:.2f}_ext_{exterior:.2f}'
    temp_dopp = np.load(os.path.join(final_temp_dir, 'doppler_template.npy'))[1]
    temp_int = np.load(os.path.join(final_temp_dir, 'intensity_template.npy'))[1]
    
    corr_inj_dopp = correlate_asymmetry(xs, inj_dopp, temp_dopp)
    corr_inj_int = correlate_asymmetry(xs, inj_int, temp_int)
    
    # Calculate p-values
    pvals_dopp = get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)
    pvals_int = get_pvals_cont(corr_inj_int, corr_uninj_int)
    pvals_comb = np.multiply(pvals_dopp, pvals_int)
    
    idx = np.argmin(np.abs(freq - xs))
    pval = np.min(pvals_comb[idx-20:idx+20])
            
    ################ PLOTTING IN CASE I NEED TO DEBUG ################
    
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    sfigs = fig.subfigures(1, 2, width_ratios=[2,1])
    
    axs = sfigs[0].subplots(2, 2, sharex=True)
    axs_pvals = sfigs[1].subplots(3, 1, sharex=True)
                                 
    axs[0][0].plot(xs, uninj_dopp, label='raw')
    axs[0][0].plot(xs, inj_dopp, label='injected')
    axs[0][0].set_title('Doppler')
    
    axs[0][1].plot(xs, uninj_int, label='raw')
    axs[0][1].plot(xs, inj_int, label='injected')
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
        j = 0
        while .5*math.erfc((j+1)/np.sqrt(2)) > min(pvals_cont):
            label = r'$n\sigma$' if not j else None
            ax.axhline(.5*math.erfc((j+1)/np.sqrt(2)), c='red', ls='--', lw=0.5, label=label)
            j += 1
        ax.axvspan(freq-3, freq+3, color='gray', alpha=0.3, linewidth=0.5)
        ax.plot(xs, pvals_cont, c='k')
        ax.set_yscale('log')
        ax.set_ylim(min(pvals_cont), 1)
        ax.set_title(title)
        ax.legend()
    
    sfigs[0].suptitle('Asymmetries and Correlations')
    sfigs[1].suptitle('P-Values')
    
    save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{freq}/b_{b:.2f}_ext_{exterior:.2f}/iter_{i}.png'
    os.makedirs(save_path.replace(save_path.split('/')[-1], ''), exist_ok=True)
    plt.savefig(save_path)
    
    ##################################################################
    
    # Save some data
    path_to_save = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/pvals/{freq}/b_{b:.2f}_ext_{exterior:.2f}/iter_{i}'
    os.makedirs(path_to_save, exist_ok=True)
    
    np.save(os.path.join(path_to_save, 'doppler_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)]))
    np.save(os.path.join(path_to_save, 'intensity_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_int, corr_uninj_int)]))
    np.save(os.path.join(path_to_save, 'combined_pvals.npy'), np.array([xs, pvals_cont]))
    
    return pval

def analyze(i, signal_size, freq=8000, b=1, exterior=3, **kwargs):    
    injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/injected/{freq}/b_{b:.2f}_ext_{exterior:.2f}/iter_{i}'
    inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/normalized_injected/{freq}/b_{b:.2f}_ext_{exterior:.2f}/iter_{i}'

    # Inject the signals
    if os.path.exists(injected_dir):
        print(f'Iteration iter_{i} for params ({b}, {exterior}) already injected. Skipping!')
    else:
        for targ in targs:
            # For injected asymmetries
            inject_calibrated_cband(
                targ,
                freq,
                info,
                data_dir=uninjected_dir,
                save_dir=injected_dir,
                calibration_dir=calibration_dir,
                signal_sizes=signal_size, 
                is_decay=is_decay
            )
        
    return calculate_pval(
        normalize_weighted,
        freq,
        width=60,
        data_dir=injected_dir,
        save_dir=inj_save_dir,
        b=b,
        exterior=exterior,
        i=i,
        **kwargs
    )

def get_limit(pval_array, interp_type='spline'):
    three_sigma = np.log10(.5*math.erfc(3/np.sqrt(2)))

    sizes, vals = pval_array
    # Throw out zero p-vals
    mask = vals != 0
    sizes, vals = sizes[mask], vals[mask]
    # Convert to log
    sizes, vals = np.log10(sizes), np.log10(vals)
    sort_indices = np.argsort(sizes)
    sizes, vals = sizes[sort_indices], vals[sort_indices]
    #print(sizes, vals)
    
    if interp_type == 'spline':
        spline = make_interp_spline(sizes, vals, k=3)
        xs = np.linspace(min(sizes), max(sizes), 1000)
        fit = interp1d(xs, spline(xs), bounds_error=False, fill_value='extrapolate')
    elif interp_type == 'pchip':
        fit = PchipInterpolator(sizes, vals, extrapolate=True)
    elif interp_type == 'akima':
        fit = Akima1DInterpolator(sizes, vals, extrapolate=True)
    elif interp_type == 'makima':
        fit = Akima1DInterpolator(sizes, vals, method='makima', extrapolate=True)
        
    def target(x):
        return fit(x) - three_sigma
    
    try:
        limit = brentq(target, min(sizes), max(sizes))
    except:
        logging.warning(f'Root could not be found in the search interval, pvals={pval_array}')
        limit = sizes[np.argmin(np.abs(vals - three_sigma))]
    
    return 10 ** limit

def prepare_asymmetries(b, exterior, freq):
    # Templates
    temp_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/normalized/b_{b:.2f}_ext_{exterior:.2f}'
    final_temp_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b:.2f}_ext_{exterior:.2f}'

    if not os.path.exists(final_temp_dir):
        with mp.Pool(processes=num_processes) as p:
            normalize_func_partial = partial(
                normalize_weighted,
                data_dir=template_dir,
                save_dir=temp_save_dir,
                freq=ref_freq,
                width=40,
                b=b,
                exterior=exterior,
                is_decay=is_decay
            )
            _ = list(p.imap_unordered(normalize_func_partial, targs))

    xs, temp_dopp = build_asymmetry(temp_save_dir, 'theta', *theta_cutoffs)
    xs, temp_int = build_asymmetry(temp_save_dir, 'phi', *phi_cutoffs)

    # Smoothing the templates with a spline (Aya)
    data_dopp = [xs[::15], temp_dopp[::15]]
    data_int = [xs[::15], temp_int[::15]]
    temp_dopp = make_interp_spline(*data_dopp, k=3)(xs)
    temp_int = make_interp_spline(*data_int, k=3)(xs)
    
    # Save the asymmetries
    os.makedirs(final_temp_dir, exist_ok=True)
    save_dopp = np.array([xs, temp_dopp])
    np.save(os.path.join(final_temp_dir, 'doppler_template.npy'), save_dopp)
    save_int = np.array([xs, temp_int])
    np.save(os.path.join(final_temp_dir, 'intensity_template.npy'), save_int)

    # Plot for good measure
    plot_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/templates/{freq}'
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(*save_dopp, c='k')
    axs[0].axvline(ref_freq, c='r', ls='--')
    axs[0].set_xlabel(r'$\nu$ [MHz]', fontsize=16)
    axs[0].set_ylabel('Asymmetry', fontsize=16)
    axs[0].set_title('Doppler', fontsize=16)

    axs[1].plot(*save_int, c='k')
    axs[1].axvline(ref_freq, c='r', ls='--')
    axs[1].set_title('Intensity', fontsize=16)

    fig.suptitle(f'b={b:.2f}, exterior={exterior:.2f}', fontsize=18)

    fig.savefig(os.path.join(plot_dir, f'b_{b:.2f}_ext_{exterior:.2f}.png'), dpi=300, bbox_inches='tight')

    # Uninjected asymmetries
    uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/normalized_uninjected/{freq}/b_{b:.2f}_ext_{exterior:.2f}'
    final_asym_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/uninjected_asymmetries/asymmetries/{freq}/b_{b:.2f}_ext_{exterior:.2f}'
    final_corr_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/uninjected_asymmetries/correlations/{freq}/b_{b:.2f}_ext_{exterior:.2f}'

    if not os.path.exists(final_asym_dir):
        with mp.Pool(processes=num_processes) as p:
            normalize_func_partial = partial(
                normalize_weighted,
                data_dir=bank_1_dir,
                save_dir=uninj_save_dir,
                freq=freq,
                width=60,
                b=b,
                exterior=exterior,
                is_decay=is_decay
            )
            _ = list(p.imap_unordered(normalize_func_partial, targs))

    xs, uninj_dopp = build_asymmetry(uninj_save_dir, 'theta', *theta_cutoffs)
    xs, uninj_int = build_asymmetry(uninj_save_dir, 'phi', *phi_cutoffs)
    
    # Correlation
    temp_dopp = np.load(os.path.join(final_temp_dir, 'doppler_template.npy'))[1]
    temp_int = np.load(os.path.join(final_temp_dir, 'intensity_template.npy'))[1]
    corr_uninj_dopp = correlate_asymmetry(xs, uninj_dopp, temp_dopp)
    corr_uninj_int = correlate_asymmetry(xs, uninj_int, temp_int)

    # Save
    os.makedirs(final_asym_dir, exist_ok=True)
    os.makedirs(final_corr_dir, exist_ok=True)

    save_dopp = np.array([xs, uninj_dopp])
    np.save(os.path.join(final_asym_dir, 'doppler.npy'), save_dopp)
    save_int = np.array([xs, uninj_int])
    np.save(os.path.join(final_asym_dir, 'intensity.npy'), save_int)
    save_corr_dopp = np.array([xs, corr_uninj_dopp])
    np.save(os.path.join(final_corr_dir, 'doppler.npy'), save_corr_dopp)
    save_corr_int = np.array([xs, corr_uninj_int])
    np.save(os.path.join(final_corr_dir, 'intensity.npy'), save_corr_int)

    # Plot
    plot_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/uninjected_asymmetries/{freq}'
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(*save_dopp, c='k')
    axs[0].set_xlabel(r'$\nu$ [MHz]', fontsize=16)
    axs[0].set_ylabel('Asymmetry', fontsize=16)
    axs[0].set_title('Doppler', fontsize=16)

    axs[1].plot(*save_int, c='k')
    axs[1].set_title('Intensity', fontsize=16)

    fig.suptitle(f'b={b:.2f}, exterior={exterior:.2f}', fontsize=18)

    fig.savefig(os.path.join(plot_dir, f'b_{b:.2f}_ext_{exterior:.2f}.png'), dpi=300, bbox_inches='tight')

def normalize_number(value, desired_min, desired_max, original_min=0, original_max=1):
    return(value - original_min) * (desired_max - desired_min) / (original_max - original_min) + desired_min

def objective(b, exterior, freq=5800, **kwargs):
    b = normalize_number(b, *b_lims)
    exterior = normalize_number(exterior, *ext_lims)

    if os.path.exists(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/pval_arrays/{freq}/b_{b:.2f}_ext_{exterior:.2f}.npy'):
        save_arr = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/pval_arrays/{freq}/b_{b:.2f}_ext_{exterior:.2f}.npy')
    else:
        # Form templates and uninjected asymmetries for this freq and (b, exterior)
        prepare_asymmetries(b, exterior, freq)
        
        # Binary search over signal sizes (5 iterations)
        left, right = -31, -29
        signal_sizes = []
        pvals = []
        
        for i in range(5):
            mid = (left + right) / 2
            signal_size = 10 ** mid

            print(f'Iteration {i} signal size: {signal_size}')
            
            pval = analyze(i, signal_size, freq=freq, b=b, exterior=exterior, **kwargs)
            log_pval = -np.log10(pval)

            if log_pval < log_three_sigma:
                left = mid
            else:
                right = mid
            
            signal_sizes.append(signal_size)
            pvals.append(pval)
        
        save_arr = np.array([signal_sizes, pvals])
        os.makedirs(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/pval_arrays/{freq}', exist_ok=True)
        np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/pval_arrays/{freq}/b_{b:.2f}_ext_{exterior:.2f}.npy', save_arr)

    # Calculate limit
    limit = get_limit(save_arr)
    print(limit)
    
    gc.collect()

    return -np.log10(limit)

if __name__ == '__main__':
    is_decay = True
    ref_signal_size = 1e-29 if is_decay else 5e-25
    ref_freq = 5800
    power = 3 if is_decay else 4

    states_dir_1 = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/snr_evals_cband/grid_weighted_1e29_decay/states/1/'
    states_dir_2 = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/snr_evals_cband/grid_weighted_1e29_decay_2/states/1/'
    bs = np.array([0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75])
    exts = np.arange(2, 8.5, 0.5)
    xx, yy = np.meshgrid(bs, exts, indexing='ij')
    
    targets = []
    ddir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file:
        for line in file:
            targets.append(line.strip())
    print(f'Targets: {targets}')
    
    case = 'decay' if is_decay else 'ann'

    theta_cutoffs = (65, 65)
    phi_cutoffs = (70, 115)

    b_lims = (0.05, 1.95)
    ext_lims = (2, 8)
    
    order = 5
    #orders = np.arange(3, 8)
    # maybe consider this later, but order 5 has been the best in my experience, so start with that. 6/5/26

    #for order in orders:
    dir_name = f'snr_evals_cband/bayes_mini_analysis_bank_1_weighted_{case}'

    start_time = datetime.datetime.now()

    # Form templates for each choice of parameters
    bank_1_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/1'
    template_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/injected'

    targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(bank_1_dir)]
    
    for targ in targs:
        # Injections with reference size at 5800 MHz
        inject_calibrated_cband(
            targ,
            ref_freq,
            info,
            data_dir=bank_1_dir,
            save_dir=template_dir,
            signal_sizes=ref_signal_size,
            is_decay=is_decay,
            template=True
        )

    uninjected_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/1'
    calibration_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/median_sefds/1'

    xs = np.load(os.path.join(uninjected_dir, os.listdir(uninjected_dir)[0]))[0]
    start = int(round(xs[0] + 200, -2)) # Round to nearest 100
    stop = int(round(xs[-1] - 200, -2))

    freqs = np.arange(start, stop, 100)
    print(f'Injection frequencies (n={len(freqs)}): {freqs}')
    
    log_three_sigma = -np.log10(.5*math.erfc(3/np.sqrt(2)))

    for freq in freqs: 
        print(f'Evaluating {freq} MHz')
        freq_start_time = datetime.datetime.now()
        #print(f'Evaluating order {order}')
        
        # Freeze the keywords
        objective_func = partial(
            objective,
            freq=freq,
            order=order,
            is_decay=is_decay
        )
            
        acq = acquisition.UpperConfidenceBound(kappa=4) #2 #4

        pbounds = {'b': (0, 1), 'exterior': (0, 1)}
        optimizer = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds=pbounds,
            verbose=0,
            random_state=1,
            allow_duplicate_points=True
        )

        if os.path.exists(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/optimizers/{freq}.json'):
            # Skip optimization; load the optimizer
            optimizer.load_state(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/optimizers/{freq}.json')
        else:
            # 5 random points
            random.seed(42)
            random_points = [
                {
                    'b': random.random(),
                    'exterior': random.random()
                }
                for _ in range(5)
            ]
            for point in tqdm(random_points, desc='Initializing'):
                target = objective_func(**point)
                optimizer.register(params=point, target=target)
            
            for _ in trange(20, desc='Optimizing'):
                next_point = optimizer.suggest()
                target = objective_func(**next_point)
                optimizer.register(params=next_point, target=target)
            
            # Save the results of the optimization
            os.makedirs(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/optimizers', exist_ok=True)
            optimizer.save_state(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/optimizers/{freq}.json')

        # Plot the posterior
        x = np.linspace(pbounds['b'][0], pbounds['b'][1], 1000)
        y = np.linspace(pbounds['exterior'][0], pbounds['exterior'][1], 1000)
        xy = np.array([[x_i, y_j] for y_j in y for x_i in x])
        plot_x = np.linspace(*b_lims, 1000)
        plot_y = np.linspace(*ext_lims, 1000)
        X, Y = np.meshgrid(plot_x, plot_y)

        Z_est = optimizer._gp.predict(xy).reshape(X.shape)

        fig, ax = plt.subplots()
        im = ax.pcolormesh(X, Y, Z_est, cmap=mpl.colormaps['magma'])
        cbar = fig.colorbar(im)
        cbar.set_label('-log($\\lambda$ [s$^{-1}$])', fontsize=16)
        ax.contour(X, Y, Z_est, levels=5, colors='w', linewidths=0.5, linestyles='-')

        max_ = optimizer.max
        res = optimizer.res
        max_x, max_y = normalize_number(max_["params"]['b'], *b_lims), normalize_number(max_["params"]['exterior'], *ext_lims)
        x_ = normalize_number(np.array([r["params"]['b'] for r in res]), *b_lims)
        y_ = normalize_number(np.array([r["params"]['exterior'] for r in res]), *ext_lims)
        x_, y_ = list(zip(*[(x_i, y_j) for x_i, y_j in zip(x_, y_) if x_i != max_x or y_j != max_y]))

        ax.scatter(x_, y_, c='white', s=60, edgecolors='black')
        ax.scatter(max_x, max_y, c='red', s=120, marker='*', edgecolors='black')

        ax.set_xlabel('b', fontsize=16)
        ax.set_ylabel('exterior', fontsize=16)
        ax.set_title(f'$\\nu$ = {freq} MHz', fontsize=20) #f'order = {file.split("_")[1]}'
        ax.text(0.03, 0.97, f'$\\lambda_\\mathrm{{min}}$ = {(10 ** -(max_["target"])):.2e}',
                transform=ax.transAxes, ha='left', va='top', fontsize=14, color='w')

        save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{freq}'
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, 'results.png'), dpi=300, bbox_inches='tight')

        # Comparison plot
        fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(12, 6))
        tails = ['snrs.npy', 'pvals.npy', 'results.npy']
        labels = [
            'SNR\n$\\lambda = 1 \\times 10^{-29}$ $\\mathrm{s}^{-1}$',
            '-$\\log(p)$\n$\\lambda = 1 \\times 10^{-29}$ $\\mathrm{s}^{-1}$',
            '-log($\\lambda$ [s$^{-1}$])'
        ]

        for ax, tail, label in zip(axs, tails, labels):
            if tail == 'results.npy':
                pcm = ax.pcolormesh(X, Y, Z_est, cmap='magma')
            else:
                grid = np.concatenate(
                    [
                        np.load(os.path.join(states_dir_1, str(freq), tail))[:, :-1], #remove exterior=5
                        np.load(os.path.join(states_dir_2, str(freq), tail))
                    ],
                    axis=1
                )
                pcm = ax.pcolormesh(xx, yy, grid, cmap='magma')
            cbar = fig.colorbar(pcm, ax=ax, location='top')
            cbar.set_label(label, fontsize=16)
            if tail == 'snrs.npy':
                ax.set_xlabel('$b$', fontsize=16)
                ax.set_ylabel('exterior', fontsize=16)
        
        fig.savefig(os.path.join(save_path, 'results_comparison.png'), dpi=300, bbox_inches='tight')

        freq_end_time = datetime.datetime.now()
        delta_freq = relativedelta(freq_end_time, freq_start_time)
        print(f'Freq {freq} done in {delta_freq.hours} hrs, {delta_freq.minutes} mins, {delta_freq.seconds} sec')

    # Aggregate results over all frequencies
    def get_Z(file):
        pbounds = {'b': (0, 1), 'exterior': (0, 1)}
        optimizer = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds=pbounds,
            verbose=0,
            random_state=1,
            allow_duplicate_points=True
        )
        optimizer.load_state(os.path.join(optimizer_dir, file))
        
        Z_est = optimizer._gp.predict(xy).reshape(X.shape)
        return Z_est

    def get_max_pt(file):
        pbounds = {'b': (0, 1), 'exterior': (0, 1)}
        optimizer = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds=pbounds,
            verbose=0,
            random_state=1,
            allow_duplicate_points=True
        )
        optimizer.load_state(os.path.join(optimizer_dir, file))
        
        max_ = optimizer.max
        max_x, max_y = normalize_number(max_["params"]['b'], *b_lims), normalize_number(max_["params"]['exterior'], *ext_lims)
        return max_x, max_y
    
    fig, ax = plt.subplots()
    optimizer_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/optimizers'
    states = os.listdir(optimizer_dir)
    Z_mean = np.mean(list(map(get_Z, states)), axis=0)
    xmax, ymax = list(zip(*list(map(get_max_pt, states))))

    im = ax.pcolormesh(X, Y, Z_mean, cmap='magma')
    cbar = fig.colorbar(im)
    cbar.set_label('-log($\\lambda$ [s$^{-1}$])', fontsize=16)
    ax.contour(X, Y, Z_mean, levels=5, colors='w', linewidths=0.5, linestyles='-')

    ax.scatter(xmax, ymax, c='white', s=60, edgecolors='black')

    ax.set_xlabel('b', fontsize=16)
    ax.set_ylabel('exterior', fontsize=16)
    ax.set_title('Bank 1 Results', fontsize=20)

    fig.savefig(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/results.png', dpi=300, bbox_inches='tight')

    # Comparison plot
    fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(12, 6))

    for ax, tail, label in zip(axs, tails, labels):
        if tail == 'results.npy':
            pcm = ax.pcolormesh(X, Y, Z_mean, cmap='magma')
        else:
            grid = np.array([
                np.concatenate(
                    [
                        np.load(os.path.join(states_dir_1, str(freq), tail))[:, :-1], #remove exterior=5
                        np.load(os.path.join(states_dir_2, str(freq), tail))
                    ],
                    axis=1
                )
                for freq in freqs
            ])
            mean_grid = np.mean(grid, axis=0)
            pcm = ax.pcolormesh(xx, yy, mean_grid, cmap='magma')
        cbar = fig.colorbar(pcm, ax=ax, location='top')
        cbar.set_label(label, fontsize=16)
        if tail == 'snrs.npy':
            ax.set_xlabel('$b$', fontsize=16)
            ax.set_ylabel('exterior', fontsize=16)
    
    fig.suptitle('Bank 1 Results', fontsize=20)
    fig.savefig(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/results_comparison.png', dpi=300, bbox_inches='tight')
    
    end_time = datetime.datetime.now()
    delta = relativedelta(end_time, start_time)
    print(f'Study complete in {delta.hours} hrs, {delta.minutes} mins, {delta.seconds} sec')