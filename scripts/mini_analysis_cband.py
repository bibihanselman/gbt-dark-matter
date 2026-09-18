#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
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
from tqdm.auto import tqdm

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

    
def calculate_pval(normalize_func, freq, data_dir=None, save_dir=None, i=0, p=None, b=1, exterior=3, **kwargs):
    spectra = os.listdir(data_dir)
    
    skip = (os.path.exists(save_dir)) and (len(os.listdir(save_dir)) == len(spectra))

    if skip:
        print(f'Freq {freq} at iteration {dir_name.split("/")[-1]} already normalized. Skipping!')
    else:
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
    final_asym_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/uninjected_asymmetries/asymmetries/b_{b}_ext_{exterior}'
    uninj_dopp = np.load(os.path.join(final_asym_dir, 'doppler.npy'))[1]
    uninj_int = np.load(os.path.join(final_asym_dir, 'intensity.npy'))[1]      
    xs, inj_dopp = build_asymmetry(save_dir, 'theta', *theta_cutoffs)
    xs, inj_int = build_asymmetry(save_dir, 'phi', *phi_cutoffs)

    # Calculating correlation spectra
    final_corr_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/uninjected_asymmetries/correlations/b_{b}_ext_{exterior}'
    corr_uninj_dopp = np.load(os.path.join(final_corr_dir, 'doppler.npy'))[1]
    corr_uninj_int = np.load(os.path.join(final_corr_dir, 'intensity.npy'))[1]
    
    final_temp_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b}_ext_{exterior}'
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
    
    save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/b_{b}_ext_{exterior}/iter_{i}.png'
    os.makedirs(save_path.replace(save_path.split('/')[-1], ''), exist_ok=True)
    plt.savefig(save_path)
    
    ##################################################################
    
    # Save some data
    path_to_save = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/pvals/b_{b}_ext_{exterior}/iter_{i}'
    os.makedirs(path_to_save, exist_ok=True)
    
    np.save(os.path.join(path_to_save, 'doppler_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)]))
    np.save(os.path.join(path_to_save, 'intensity_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_int, corr_uninj_int)]))
    np.save(os.path.join(path_to_save, 'combined_pvals.npy'), np.array([xs, pvals_cont]))
    
    return pval

def analyze(i, signal_size, freq=8000, b=1, exterior=3, **kwargs):    
    injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/injected/b_{b}_ext_{exterior}/iter_{i}'
    inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/normalized_injected/b_{b}_ext_{exterior}/iter_{i}'

    # Inject the signals
    if os.path.exists(injected_dir):
        print(f'Iteration iter_{i} for params ({b}, {exterior}) already injected. Skipping!')
    else:
        for targ in targs:
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
        is_decay=is_decay,
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

def objective(b, exterior, **kwargs):
    if os.path.exists(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/b_{b}_ext_{exterior}.npy'):
        save_arr = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/b_{b}_ext_{exterior}.npy')
    else:
        left, right = -31, -29
        signal_sizes = []
        pvals = []
        
        # Binary search (5 iterations)
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
        os.makedirs(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/', exist_ok=True)
        np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/b_{b}_ext_{exterior}.npy', save_arr)

    # Calculate limit
    limit = get_limit(save_arr)
    print(limit)
    
    return -np.log10(limit)

if __name__ == '__main__':
    is_decay = True
    ref_signal_size = 1e-29 if is_decay else 5e-25
    ref_freq = 5500
    power = 3 if is_decay else 4
    
    targets = []
    ddir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file:
        for line in file:
            targets.append(line.strip())
    print(f'Targets: {targets}')
    
    case = 'decay' if is_decay else 'ann'

    theta_cutoffs = (65, 65)
    phi_cutoffs = (70, 115)
    
    freq = 5500
    order = 5
    #orders = np.arange(3, 8)
    # maybe consider this later, but order 5 has been the best in my experience, so start with that. 6/5/26

    bs = np.array([0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75])
    exts = np.arange(2, 8.5, 0.5) #(2, 5.5, 0.5)
    grid_params = list(itertools.product(bs, exts)) #b, exterior

    xx, yy = np.meshgrid(bs, exts, indexing='ij')

    #for order in orders:
    dir_name = f'snr_evals_cband/mini_analysis_5500_weighted_{case}'

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

    # Templates and uninjected asymmetries
    for b, exterior in tqdm(grid_params, desc='Forming templates/uninjected asymmetries'):
        # Templates
        temp_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/normalized/b_{b}_ext_{exterior}'
        final_temp_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b}_ext_{exterior}'

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
        plot_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/templates'
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

        fig.suptitle(f'b={b}, exterior={exterior}', fontsize=18)

        fig.savefig(os.path.join(plot_dir, f'b_{b}_ext_{exterior}.png'), dpi=300, bbox_inches='tight')

        # Uninjected asymmetries
        uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/normalized_uninjected/b_{b}_ext_{exterior}'
        final_asym_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/uninjected_asymmetries/asymmetries/b_{b}_ext_{exterior}'
        final_corr_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/uninjected_asymmetries/correlations/b_{b}_ext_{exterior}'

        if not os.path.exists(final_asym_dir):
            with mp.Pool(processes=num_processes) as p:
                normalize_func_partial = partial(
                    normalize_weighted,
                    data_dir=bank_1_dir,
                    save_dir=uninj_save_dir,
                    freq=ref_freq,
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
        plot_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/uninjected_asymmetries'
        os.makedirs(plot_dir, exist_ok=True)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].plot(*save_dopp, c='k')
        axs[0].set_xlabel(r'$\nu$ [MHz]', fontsize=16)
        axs[0].set_ylabel('Asymmetry', fontsize=16)
        axs[0].set_title('Doppler', fontsize=16)

        axs[1].plot(*save_int, c='k')
        axs[1].set_title('Intensity', fontsize=16)

        fig.suptitle(f'b={b}, exterior={exterior}', fontsize=18)

        fig.savefig(os.path.join(plot_dir, f'b_{b}_ext_{exterior}.png'), dpi=300, bbox_inches='tight')

    print('Mini analysis start!')

    uninjected_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/1'
    calibration_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/median_sefds/1'
    
    log_three_sigma = -np.log10(.5*math.erfc(3/np.sqrt(2)))

    with mp.Pool(processes=num_processes) as p:
        objective_func = partial(objective, p=p)
        result = list(tqdm(itertools.starmap(objective_func, grid_params), total=len(grid_params)))

    limit_grid = np.array(result).reshape(xx.shape)
    np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/results.npy', limit_grid)

    # Plot grid
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xx, yy, limit_grid, cmap='viridis')
    cbar = fig.colorbar(pcm, ax=ax, location='top')
    cbar.set_label(r'-log($\lambda$ [s$^{-1}$])', fontsize=16)
    ax.set_xlabel('$b$', fontsize=16)
    ax.set_ylabel('exterior', fontsize=16)
    
    fig.savefig(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/results.png', dpi=300, bbox_inches='tight')
    
    end_time = datetime.datetime.now()
    delta = relativedelta(end_time, start_time)
    print(f'Study complete in {delta.hours} hrs, {delta.minutes} mins, {delta.seconds} sec')