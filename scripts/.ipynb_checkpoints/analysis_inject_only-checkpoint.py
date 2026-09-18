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

c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s
dm_profile = 'nfw'


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

    
def calculate_pval(normalize_func, freqs, data_dir=None, save_dir=None, dir_name=None, **kwargs):
    spectra = os.listdir(data_dir)
    
    skip = (os.path.exists(save_dir)) and (len(os.listdir(save_dir)) == len(spectra))

    if skip:
        print(f'Start freq {freqs[0]} at signal size {dir_name.split("/")[-1]} already normalized. Skipping!')
    else:
        with mp.Pool(processes=num_processes) as p:
            normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, **kwargs)
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
    
    # Cross correlate doppler and intensity
    # corr_inj_comb = correlate(corr_inj_int, corr_inj_dopp, mode='same')
    # corr_uninj_comb = correlate(corr_uninj_int, corr_uninj_dopp, mode='same')
    
    # Calculate p-values
    pvals_dopp = get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)
    pvals_int = get_pvals_cont(corr_inj_int, corr_uninj_dopp)
    pvals_comb = np.multiply(pvals_dopp, pvals_int)
    
    def get_pvals(freq):
        idx = np.argmin(np.abs(freq - xs))
        pval = np.min(pvals_comb[idx-20:idx+20])
        return pval
        
    pvals = list(map(get_pvals, freqs))
    
    ################ PLOTTING IN CASE I NEED TO DEBUG ################
    
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    sfigs = fig.subfigures(1, 2, width_ratios=[2,1])
    
    axs = sfigs[0].subplots(2, 2, sharex=True)
    axs_pvals = sfigs[1].subplots(3, 1, sharex=True)
                                 
    axs[0][0].plot(xs, uninj_dopp, label='raw')
    axs[0][0].plot(xs, inj_dopp, label='injected')
    #axs[0][0].plot(xs, temp_dopp, label='template')
    axs[0][0].set_title('Doppler')
    
    axs[0][1].plot(xs, uninj_int, label='raw')
    axs[0][1].plot(xs, inj_int, label='injected')
    #axs[0][1].plot(xs, temp_int, label='template')
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
    
    save_path = f'/home/bhanselman/asymmetry_plots/{dir_name}'
    os.makedirs(save_path.replace(save_path.split('/')[-1], ''), exist_ok=True)
    plt.savefig(f'{save_path}.png')
    
    ##################################################################
    
    # Save some data
    path_to_save = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/'
    
    np.save(os.path.join(path_to_save, 'doppler_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_dopp, corr_uninj_dopp)]))
    np.save(os.path.join(path_to_save, 'intensity_pvals.npy'), np.array([xs, get_pvals_cont(corr_inj_int, corr_uninj_int)]))
    np.save(os.path.join(path_to_save, 'combined_pvals.npy'), np.array([xs, pvals_cont]))
    
    return pvals


def analyze(signal_size, start_freq=8000):
    print(f'Evaluating signal size {signal_size}')
    
    dir_name = f'{analysis_dir}/start_{start_freq}/{signal_size}'

    injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/injected/'
    inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_injected/'

    # Inject the signals
    targs = os.listdir(uninjected_dir)
    
    skip = os.path.exists(injected_dir)
    
    if skip:
        print('Directory for this freq and signal size already exists. Skipping!')
    else:
        for targ in targs:
            inject_spaced(targ, start_freq, stop, spacing, info, data_dir=uninjected_dir, save_dir=injected_dir, signal_size=signal_size, is_decay=is_decay)
    
    freqs = np.arange(start_freq, stop, spacing)
    
    return calculate_pval(
        normalize_dynamic,
        freqs,
        data_dir=injected_dir,
        save_dir=inj_save_dir,
        start=start-75,
        stop=stop+75,
        interior=interior,
        exterior=exterior,
        dir_name=dir_name,
        is_decay=is_decay
    )
    

if __name__ == '__main__':
    full_band = True
    start = 8000
    stop = 8500
    spacing = 100
    step = 1
        
    banks = [0, 1, 2, 3]
    signal_sizes = [4e-28, 2e-28, 1e-28, 5e-29, 3e-29, 2e-29, 1e-29, 5e-30]
    # Decay: [4e-28, 2e-28, 1e-28, 5e-29, 3e-29, 2e-29, 1e-29, 5e-30]
    # Annihilation: [1e-24, 5e-25, 2e-25, 1e-25, 5e-26, 3e-26, 2e-26, 1e-26]
    is_decay = True
    decay_or_ann = 'decay' if is_decay else 'ann'
    print(decay_or_ann)

    # Begin searching the analysis grid
    for bank in banks:
        print(f'Injecting bank {bank}')
        uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed/{bank}/'
        uninj_spectra = os.listdir(uninjected_dir)

        xs = np.load(os.path.join(uninjected_dir, uninj_spectra[0]))[0]
        if full_band:
            start = int(round(xs[0] + 75, -1)) # Round to nearest 10
            stop = int(round(xs[-1] - 75, -1))
            print(start, stop)

        start_freqs = np.arange(start, start+spacing, step)
        
        for start_freq in tqdm(start_freqs):
            print(f'Injecting start freq of {start_freq} MHz')
            injected_dir_root = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/injected/{bank}/{decay_or_ann}/start_{start_freq}'

            for signal_size in signal_sizes:
                injected_dir = os.path.join(injected_dir_root, f'{signal_size}')
                with mp.Pool(processes=num_processes) as p:
                    inject_spaced_partial = partial(inject_spaced, start=start_freq, stop=stop, spacing=spacing, info=info, data_dir=uninjected_dir, save_dir=injected_dir, signal_size=signal_size, is_decay=is_decay)
                    _ = list(tqdm(p.imap_unordered(inject_spaced_partial, uninj_spectra), total=len(uninj_spectra), desc='Injecting'))
