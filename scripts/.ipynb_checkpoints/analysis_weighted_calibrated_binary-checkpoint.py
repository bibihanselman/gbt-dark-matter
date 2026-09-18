'''
X-band analysis script.
Author: Benjamin Hanselman
Date: 4 November 2025
'''

import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d, make_interp_spline

from normalization_methods import *


num_processes = mp.cpu_count()

info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/all_xband_info.csv'
info = pd.read_csv(info_dir)


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
        stretched = stretch_template(template, freqs[i] / template_freq)
        
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

    
def calculate_pval(normalize_func, freq, data_dir=None, save_dir=None, dir_name=None, p=None, **kwargs):
    spectra = os.listdir(data_dir)
    
    skip = (os.path.exists(save_dir)) and (len(os.listdir(save_dir)) == len(spectra))

    if skip:
        print(f'Start freq {freqs[0]} at signal size {dir_name.split("/")[-1]} already normalized. Skipping!')
    else:
        normalize_func_partial = partial(normalize_func, freq=freq, data_dir=data_dir, save_dir=save_dir, **kwargs)
        _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing injected'))
        
    xs, inj_dopp = build_asymmetry(save_dir, 'theta', *theta_cutoffs)
    inj_dopp = sigma_filter(inj_dopp)
    xs, inj_int = build_asymmetry(save_dir, 'phi', *phi_cutoffs)
    inj_int = sigma_filter(inj_int)
    
    # Calculating correlation spectra
    corr_inj_dopp = correlate_asymmetry(xs, inj_dopp, temp_dopp)
    corr_uninj_dopp = correlate_asymmetry(xs, uninj_dopp, temp_dopp)
    corr_inj_int = correlate_asymmetry(xs, inj_int, temp_int)
    corr_uninj_int = correlate_asymmetry(xs, uninj_int, temp_int)
    
    # Calculate p-values
    pvals_dopp = get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)
    pvals_int = get_pvals_cont(corr_inj_int, corr_uninj_dopp)
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
        i = 0
        while .5*math.erfc((i+1)/np.sqrt(2)) > min(pvals_cont):
            label = '$n\sigma$' if not i else None
            ax.axhline(.5*math.erfc((i+1)/np.sqrt(2)), c='red', ls='--', lw=0.5, label=label)
            i += 1
        ax.axvspan(freq-3, freq+3, color='gray', alpha=0.3, linewidth=0.5)
        ax.plot(xs, pvals_cont, c='k')
        ax.set_yscale('log')
        ax.set_ylim(min(pvals_cont), 1)
        ax.set_title(title)
        ax.legend()
    
    sfigs[0].suptitle('Asymmetries and Correlations')
    sfigs[1].suptitle('P-Values')
    
    save_path = f'/home/bhanselman/asymmetry_plots/{dir_name}'
    os.makedirs(save_path.replace(save_path.split('/')[-1], ''), exist_ok=True)
    plt.savefig(f'{save_path}.png')
    
    ##################################################################
    
    # Save some data
    path_to_save = f'{root}/{dir_name}/'
    
    np.save(os.path.join(path_to_save, 'doppler_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)]))
    np.save(os.path.join(path_to_save, 'intensity_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_int, corr_uninj_int)]))
    np.save(os.path.join(path_to_save, 'combined_pvals.npy'), np.array([xs, pvals_cont]))
    
    return pval


def analyze(signal_size, freq=8000, **kwargs):
    print(f'Evaluating signal size {signal_size}')
    
    dir_name = f'{analysis_dir}/{freq}/{signal_size}'

    injected_dir = f'{root}/{dir_name}/injected/'
    inj_save_dir = f'{root}/{dir_name}/normalized_injected/'

    # Inject the signals
    targs = os.listdir(uninjected_dir)
        
    if os.path.exists(injected_dir):
        print('Directory for this freq and signal size already exists. Skipping!')
    else:
        for targ in targs:
            calibration_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/median_sefds/{bank}'
            inject_calibrated(targ,
                              freq,
                              info,
                              data_dir=uninjected_dir,
                              save_dir=injected_dir,
                              signal_size=signal_size, 
                              is_decay=is_decay
                  )
        
    return calculate_pval(
        normalize_weighted,
        freq,
        data_dir=injected_dir,
        save_dir=inj_save_dir,
        width=50,
        a=a,
        exterior=exterior,
        dir_name=dir_name,
        is_decay=is_decay,
        **kwargs
    )
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of analysis run')
    parser.add_argument('bank', help='bank to analyze')
    parser.add_argument('lower', type=float, help='log(lower endpoint) for binary search')
    parser.add_argument('upper', type=float, help='log(upper endpoint) for binary search')
    parser.add_argument('-d', '--decay', action='store_true', help='perform decay instead of annihilation search')
    parser.add_argument('-n', '--niter', default=6, type=int, help='number of binary search iterations, default 6')
    args = parser.parse_args()
    
    bank = args.bank
    is_decay = args.decay
    case = 'decay' if is_decay else 'ann'
    
    signal_size_endpoints = [args.lower, args.upper]
    template_signal_size = 1e-28 if is_decay else 1e-25
    
    root = '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625'
    analysis_dir = f'analyses/{name}/{bank}/{case}'
    
    uninjected_dir = f'{root}/data/preprocessed/{bank}/'
    uninj_spectra = os.listdir(uninjected_dir)
    xs = np.load(os.path.join(uninjected_dir, uninj_spectra[0]))[0]

    start = int(round(xs[0] + 75, -1)) # Round to nearest 10
    stop = int(round(xs[-1] - 75, -1))
    step = 10
    
    freqs = np.arange(start, stop, step)
    
    a = 0.1
    exterior = 8
    
    theta_cutoffs = (65, 65)
    phi_cutoffs = (70, 115)
    
    # Form the uninjected asymmetries (used for all freqs and signal sizes)
    print('Forming uninjected asymmetries')
    uninj_save_dir = f'{root}/{analysis_dir}/normalized_uninjected/'
        
    if os.path.exists(uninj_save_dir):
        print('Uninjected spectra already normalized. Skipping!')
    else:
        with mp.Pool(processes=num_processes) as p:
            normalize_func_partial = partial(normalize_weighted,
                                             data_dir=uninjected_dir,
                                             save_dir=uninj_save_dir,
                                             start=start-75,
                                             stop=stop+75,
                                             a=a,
                                             exterior=exterior,
                                             is_decay=is_decay
                                            )
            _ = list(tqdm(p.imap_unordered(normalize_func_partial, uninj_spectra), total=len(uninj_spectra), desc=f'Normalizing uninjected'))

    xs, uninj_dopp = build_asymmetry(uninj_save_dir, 'theta', *theta_cutoffs)
    uninj_dopp = sigma_filter(uninj_dopp)
    xs, uninj_int = build_asymmetry(uninj_save_dir, 'phi', *phi_cutoffs)
    uninj_int = sigma_filter(uninj_int)
    
    # Save the asymmetries for good measure
    save_dopp = np.array([xs, uninj_dopp])
    np.save(f'{root}/{analysis_dir}/uninj_dopp_asymmetry.npy', save_dopp)
    save_int = np.array([xs, uninj_int])
    np.save(f'{root}/{analysis_dir}/uninj_int_asymmetry.npy', save_int)

    print('Uninjected asymmetries done!')
    
    # Form the master templates
    print('Forming master templates')
    template_dir = f'{root}/{analysis_dir}/template/'
    temp_save_dir = f'{root}/{analysis_dir}/normalized_template/'
    template_freq = start + 80
    
    for targ in uninj_spectra:
        inject(targ, template_freq, info, data_dir=uninjected_dir, save_dir=template_dir, signal_size=template_signal_size, is_decay=is_decay, template=True)
    
    if os.path.exists(temp_save_dir):
        print('Template spectra already normalized. Skipping!')
    else:
        with mp.Pool(processes=num_processes) as p:
            normalize_func_partial = partial(normalize_weighted,
                                             data_dir=template_dir,
                                             save_dir=temp_save_dir,
                                             freq=template_freq,
                                             width=80,
                                             a=a,
                                             exterior=exterior,
                                             is_decay=is_decay
                                            )
            _ = list(tqdm(p.imap_unordered(normalize_func_partial, uninj_spectra), total=len(uninj_spectra), desc=f'Normalizing template'))

    xs, temp_dopp = build_asymmetry(temp_save_dir, 'theta', *theta_cutoffs)
    xs, temp_int = build_asymmetry(temp_save_dir, 'phi', *phi_cutoffs)

    # Smoothing the templates with a spline (Aya)
    data_dopp = [xs[::15], temp_dopp[::15]]
    data_int = [xs[::15], temp_int[::15]]
    temp_dopp = make_interp_spline(*data_dopp, k=3)(xs)
    temp_int = make_interp_spline(*data_int, k=3)(xs)
    
    # Save the asymmetries for good measure
    save_dopp = np.array([xs, temp_dopp])
    np.save(f'{root}/{analysis_dir}/doppler_template.npy', save_dopp)
    save_int = np.array([xs, temp_int])
    np.save(f'{root}/{analysis_dir}/intensity_template.npy', save_int)

    print('Templates done!')
    
    n = args.niter
    log_three_sigma = -np.log10(.5*math.erfc(3/np.sqrt(2)))
    
    # Begin searching the analysis grid
    with mp.Pool(processes=num_processes) as p:
        for freq in tqdm(freqs):
            print(f'Analyzing freq of {freq} MHz')
            
            left, right = signal_size_endpoints
            signal_sizes = []
            pvals = []
            
            # Binary search
            for i in range(n):
                mid = (left + right) / 2
                signal_size = 10 ** mid
                pval = analyze(signal_size, freq=freq, p=p)
                log_pval = -np.log10(pval)
                if log_pval < log_three_sigma:
                    left = mid
                else:
                    right = mid
                signal_sizes.append(signal_size)
                pvals.append(pval)
                    
            save_arr = np.array([signal_sizes, pvals])
            np.save(f'{root}/{analysis_dir}/{freq}_pvals.npy', save_arr)