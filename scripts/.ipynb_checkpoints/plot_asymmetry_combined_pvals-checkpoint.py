import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
import random
from functools import partial
from scipy.optimize import curve_fit
import itertools
from tqdm import tqdm, trange
from functools import partial
import random
from bayes_opt import BayesianOptimization, acquisition
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, correlate
from scipy.stats import percentileofscore
from scipy.interpolate import interp1d

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


def correlate_asymmetry(freqs, spectrum, template, pad=False):
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
    orig_freq = freqs[len(spectrum) // 2]
    for i in range(len(spectrum)):
        central_freq = freqs[i]
        stretched = stretch_template(template, central_freq / orig_freq)
        
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
    if pad:
        a_vals = np.pad(a_vals, len(stretched//2))
        residuals = np.pad(residuals, len(stretched)//2, mode='constant', constant_values=1)

    return a_vals / (residuals + 1e-12)


def plot_asymmetry(normalize_func, data_dirs=None, save_dirs=None, fit_window=None, freq=8720, do_normalize=True, std_cut=None, fit_type='poly', fit_order=4, name=None, **kwargs):
    spectra = os.listdir(data_dirs[2])
    if do_normalize:
        with mp.Pool(processes=num_processes) as p:
            for data_dir, save_dir in zip(data_dirs, save_dirs):
                normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
                _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing {data_dir.split("/")[-2]} with {normalize_func.__name__}'))
    
        # Cut the uninjected+injected normalized spectra with the highest stds
        if std_cut:
            for save_dir in save_dirs:
                files = os.listdir(save_dir)
                spectra = [np.load(os.path.join(save_dir, file))[1] for file in files]
                stds = [np.std(spec) for spec in spectra]
                for file, std in zip(files, stds):
                    if percentileofscore(stds, std) > std_cut:
                        os.remove(os.path.join(save_dir, file))
            print(len(os.listdir(save_dir)))
        
    def sigma_filter(data, window_size=15):
        for i in range(window_size//2, len(data)-window_size//2):
            window_data = data[i-window_size//2:i+window_size//2+1]
            median = np.median(window_data)
            std = np.std(window_data)
            if abs(data[i] - median) > 3 * std:
                data[i] = median
        return data
    
    uninj_save_dir, inj_save_dir, temp_save_dir = save_dirs

    xs, uninj_dopp = build_asymmetry(uninj_save_dir, 'theta', 65, 65)
    xs, inj_dopp = build_asymmetry(inj_save_dir, 'theta', 65, 65)
    xs, temp_dopp = build_asymmetry(temp_save_dir, 'theta', 65, 65)
    
    uninj_dopp, inj_dopp, temp_dopp = list(map(sigma_filter, [uninj_dopp, inj_dopp, temp_dopp]))
    
    xs, uninj_int = build_asymmetry(uninj_save_dir, 'phi', 70, 115)
    xs, inj_int = build_asymmetry(inj_save_dir, 'phi', 70, 115)
    xs, temp_int = build_asymmetry(temp_save_dir, 'phi', 70, 115)
    
    uninj_int, inj_int, temp_int = list(map(sigma_filter, [uninj_int, inj_int, temp_int]))

    
    from scipy.interpolate import make_interp_spline

    # Smoothing the templates with a spline (Aya)
    data_int = [xs[::15], temp_int[::15]]
    data_dopp = [xs[::15], temp_dopp[::15]]
    temp_int = make_interp_spline(*data_int, k=3)(xs)
    temp_dopp = make_interp_spline(*data_dopp, k=3)(xs)
    
    # Calculating correlation spectra
    corr_inj_int = correlate_asymmetry(xs, inj_int, temp_int)
    corr_uninj_int = correlate_asymmetry(xs, uninj_int, temp_int)
    corr_inj_dopp = correlate_asymmetry(xs, inj_dopp, temp_dopp)
    corr_uninj_dopp = correlate_asymmetry(xs, uninj_dopp, temp_dopp)
    
    # Cross correlate doppler and intensity
    corr_inj_comb = correlate(corr_inj_int, corr_inj_dopp, mode='same')
    corr_uninj_comb = correlate(corr_uninj_int, corr_uninj_dopp, mode='same')
    
    print(len(corr_inj_comb))
    
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    sfigs = fig.subfigures(1, 2, width_ratios=[2,1])
    
    axs = sfigs[0].subplots(2, 2, sharex=True)
    axs_pvals = sfigs[1].subplots(3, 1, sharex=True)
                                 
    axs[0][0].plot(xs, uninj_dopp, label='raw')
    axs[0][0].plot(xs, inj_dopp, label='injected')
    axs[0][0].plot(xs, temp_dopp, label='template')
    axs[0][0].set_title('Doppler')
    
    axs[0][1].plot(xs, uninj_int, label='raw')
    axs[0][1].plot(xs, inj_int, label='injected')
    axs[0][1].plot(xs, temp_int, label='template')
    axs[0][1].legend()
    axs[0][1].set_title('Intensity')
    
    axs[1][0].plot(xs, corr_uninj_dopp, c='gray', label='raw')
    axs[1][0].plot(xs, corr_inj_dopp, c='k', label='injected')
    
    axs[1][1].plot(xs, corr_uninj_int, c='gray', label='raw')
    axs[1][1].plot(xs, corr_inj_int, c='k', label='injected')
    axs[1][1].legend()
    
    corrs_list = [
        (corr_inj_dopp, corr_uninj_dopp),
        (corr_inj_int, corr_uninj_int),
        (corr_inj_comb, corr_uninj_comb)
    ]
    titles = ['Doppler', 'Intensity', 'Combined']
    
    # p value plot
    for ax, corrs, title in zip(axs_pvals.flat, corrs_list, titles):
        corr_inj, corr_uninj = corrs
        
        for i in range(15):
            label = '$n\sigma$' if not i else None
            ax.axhline(.5*math.erfc((i+1)/np.sqrt(2)), c='red', ls='--', lw=0.5, label=label)

        sigmas = corr_inj / np.std(corr_uninj)
        #print(max(sigmas))
        pvals = list(map(lambda sigma: .5*math.erfc(sigma/np.sqrt(2)), sigmas))
        ax.axvspan(freq-3, freq+3, color='gray', alpha=0.3, linewidth = 0.5)
        ax.plot(xs, pvals, c='k')
        ax.set_yscale('log')
        ax.set_ylim(min(pvals), 1)
        ax.set_title(title)
        ax.legend()
    
    sfigs[0].suptitle('Asymmetries and Correlations')
    sfigs[1].suptitle('P-Values')
    
    plt.savefig(f'/home/bhanselman/asymmetry_plots/{name}.png')


if __name__ == '__main__':
    targets = []
    ddir ='/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file: #random_names
        for line in file:
            targets.append(line.strip())
    #print(f'Targets: {targets}')
    
    # Values
    #interior = 1.4
    exterior = 3
    freq = 8000
    bank = 0
    name = '8000_worse_doppler_maybe'
    dir_name = f'asymmetry_tests/{name}' #'asymmetry_tests/10650_template_0.015_decay'
    
    uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed/{bank}/'
    injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/injected/'
    template_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/template/'
    uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_uninjected/'
    inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_injected/'
    temp_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_template/'
    
    
    # Get the targets
    targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
    print(f'# of targets in bank {bank}: {len(targs)}')
    
    do_inject = True
    is_decay = True
    if do_inject:
        signal_size = 2e-29
        for targ in targs:
            inject(targ, freq, info, data_dir=uninjected_dir, save_dir=injected_dir, signal_size=signal_size, is_decay=is_decay)
            inject(targ, freq, info, data_dir=uninjected_dir, save_dir=template_dir, signal_size=signal_size, is_decay=is_decay, template=True)
    
    plot_asymmetry(
        normalize_weighted, #normalize_dynamic
        fit_window=None, #20,
        freq=freq,
        width=80,
        a=0.1,
        #interior=interior,
        exterior=exterior,
        order=5,
        data_dirs=(uninjected_dir, injected_dir, template_dir),
        save_dirs=(uninj_save_dir, inj_save_dir, temp_save_dir),
        do_normalize=True,
        #std_cut=90,
        fit_type='none',
        is_decay=is_decay,
        name=name
        # fit_order=4
    )