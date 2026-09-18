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
from scipy.signal import savgol_filter
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


def correlate(freqs, spectrum, template, pad=False):
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


def plot_asymmetry(normalize_func, data_dirs=None, save_dirs=None, fit_window=None, freq=8720, do_normalize=True, std_cut=None, fit_type='poly', fit_order=4, **kwargs):
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

    
    uninj_save_dir, inj_save_dir, temp_save_dir = save_dirs
    #print('Building asymmetries')
    xs, uninj_ys = build_asymmetry(uninj_save_dir, 'phi', 70, 115)
    xs, inj_ys = build_asymmetry(inj_save_dir, 'phi', 70, 115)
    xs, temp_ys = build_asymmetry(temp_save_dir, 'phi', 70, 115)
    
    # Smoothing the template with a spline (Aya)
    data = [xs[::15], temp_ys[::15]]
    from scipy.interpolate import make_interp_spline
    temp_ys = make_interp_spline(*data, k=3)(xs)
    
    # Calculating correlation spectrum
    corr_inj = correlate(xs, inj_ys, temp_ys)
    corr_uninj = correlate(xs, uninj_ys, temp_ys)
    
    # Plotting
    
    if fit_window:
        mask = abs(xs - freq) < fit_window
        window = [xs[mask], inj_ys[mask]]
        bounds = (min(window[0]), max(window[0]))

        if fit_type == 'poly':
            params = np.polyfit(*window, fit_order) #quadratic for intensity
            fit = np.poly1d(params)(window[0])
        elif fit_type == 'gaussian':
            # Trying a Gaussian instead of quadratic. 7/19/25
            def gaussian(x, amplitude, mean, stddev):
                return amplitude * np.exp(-((x - mean) / (2 * stddev))**2)

            params, _ = curve_fit(
                gaussian,
                *window,
                bounds=((0, bounds[0], 0),(np.inf, bounds[1], np.inf))
            )
            print(params)

            fit = gaussian(window[0], *params)
        elif fit_type == 'poly_gaussian':
            def poly_gaussian(x, amplitude, mean, stddev, *coeffs):
                y = amplitude * np.exp(-((x - mean) / (2 * stddev))**2)
                for i, coeff in enumerate(coeffs):
                    y += coeff * x ** i
                return y

            guesses = np.ones_like(range(fit_order+1)).tolist()
            fit_bounds = (
                (0, bounds[0], 0, *[-np.inf for _ in guesses]),
                (np.inf, bounds[1], np.inf, *[np.inf for _ in guesses])
            )

            params, _ = curve_fit(
                poly_gaussian,
                *window,
                bounds=fit_bounds,
                p0=[0.01, freq, 10, *guesses]
            )
            print(params)

            amp, mean, std, *coeffs = params
            fit = poly_gaussian(window[0], amp, mean, std, *coeffs)
        elif fit_type == 'savgol':
            #Sigma filter...
            window_size = 15
            window_copy = window[1]
            for i in range(window_size//2, len(window_copy)-window_size//2):
                window_data = window_copy[i-window_size//2:i+window_size//2+1]
                median = np.median(window_data)
                std = np.std(window_data)
                if abs(window_copy[i] - median) > 3 * std:
                    window_copy[i] = median
            filt_window = savgol_filter(window_copy, 30, fit_order)

    fig, axs = plt.subplots(3, 1, sharex=True)
    plt.subplots_adjust(wspace=0)
    axs[0].plot(xs, uninj_ys, label='raw')
    axs[0].plot(xs, inj_ys, label='injected')
    axs[0].plot(xs, temp_ys, label='template')
    
    if fit_type == 'savgol':
        axs[0].plot(window[0], filt_window, color='red', label='smoothed')
    elif fit_type != 'none':
        axs[0].plot(window[0], fit, color='red', label='fit')
    
    if fit_window:
        axs[0].axvspan(*bounds, color='k', alpha=0.2)
    axs[0].legend()
    
    axs[1].plot(xs, corr_uninj, c='gray', label='raw')
    axs[1].plot(xs, corr_inj, c='k', label='injected')
    axs[1].legend()
    
    # p value plot
    for i in range(15):
        label = '$n\sigma$' if not i else None
        axs[2].axhline(.5*math.erfc((i+1)/np.sqrt(2)), c='red', ls='--', lw=0.5, label=label)
        
    sigmas = corr_inj / np.std(corr_uninj)
    #print(max(sigmas))
    pvals = list(map(lambda sigma: .5*math.erfc(sigma/np.sqrt(2)), sigmas))
    axs[2].axvspan(freq-3, freq+3, color='gray', alpha=0.3, linewidth = 0.5)
    axs[2].plot(xs, pvals, c='k')
    axs[2].set_yscale('log')
    axs[2].set_ylim(min(pvals), 1)
    axs[2].legend()
    
    plt.savefig(f'/home/bhanselman/asymmetry_plots/{freq}_template_corr_0.015_decay.png')


if __name__ == '__main__':
    targets = []
    ddir ='/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file: #random_names
        for line in file:
            targets.append(line.strip())
    #print(f'Targets: {targets}')
    
    # Values
    nsigma = 1.2
    window_factor = 0.015
    freq = 10650
    bank = 2
    dir_name = 'asymmetry_tests/10650_template_0.015_decay'
    
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
    if do_inject:
        for targ in targs:
            signal_size = 5e-25 * (freq / 8500)**4 # same strength as a 1e-27 signal at 8500 MHz

            inject(targ, freq, info, data_dir=uninjected_dir, save_dir=injected_dir, signal_size=signal_size, is_decay=False)
            inject(targ, freq, info, data_dir=uninjected_dir, save_dir=template_dir, signal_size=signal_size, is_decay=False, template=True)
    
    plot_asymmetry(
        normalize_dynamic,
        fit_window=None, #20,
        freq=freq,
        width=100,
        nsigma=nsigma,
        #window=window,
        window_factor=window_factor,
        order=5,
        data_dirs=(uninjected_dir, injected_dir, template_dir),
        save_dirs=(uninj_save_dir, inj_save_dir, temp_save_dir),
        do_normalize=True,
        std_cut=90,
        fit_type='none',
        # fit_order=4
    )