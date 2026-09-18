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


def plot_asymmetry(normalize_func, uninjected_dir=None, injected_dir=None, uninj_save_dir=None, inj_save_dir=None, fit_window=None, freq=8720, do_normalize=True, std_cut=None, fit_type='poly', fit_order=4, **kwargs):
    spectra = os.listdir(injected_dir)
    
    if do_normalize:
        with mp.Pool(processes=num_processes) as p:
            for data_dir, save_dir in ((uninjected_dir, uninj_save_dir), (injected_dir, inj_save_dir)):
                normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
                _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing {data_dir.split("/")[-2]} with {normalize_func.__name__}'))
    
    # Cut the uninjected+injected normalized spectra with the highest stds
    if std_cut:
        for data_dir in (uninj_save_dir, inj_save_dir):
            files = os.listdir(data_dir)
            spectra = [np.load(os.path.join(data_dir, file))[1] for file in files]
            stds = [np.std(spec) for spec in spectra]
            for file, std in zip(files, stds):
                if percentileofscore(stds, std) > std_cut:
                    os.remove(os.path.join(data_dir, file))
        print(len(os.listdir(data_dir)))
            
    
    #print('Building asymmetries')
    xs, uninj_ys = build_asymmetry(uninj_save_dir, 'phi', 70, 115)
    xs, inj_ys = build_asymmetry(inj_save_dir, 'phi', 70, 115)
    
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

    fig, ax = plt.subplots()
    ax.plot(xs, uninj_ys, label='raw')
    ax.plot(xs, inj_ys, label='injected')
    
    if fit_type == 'savgol':
        ax.plot(window[0], filt_window, color='red', label='smoothed')
    elif fit_type != 'none':
        ax.plot(window[0], fit, color='red', label='fit')
    
    if fit_window:
        ax.axvspan(*bounds, color='k', alpha=0.2)
    ax.legend()
    
    plt.savefig(f'/home/bhanselman/asymmetry_plots/{freq}_spike_test_no_cut.png')


if __name__ == '__main__':
    targets = []
    ddir ='/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file: #random_names
        for line in file:
            targets.append(line.strip())
    #print(f'Targets: {targets}')
    
    # Values
    nsigma = 1.2300773402677938
    window_factor = 0.044421412846963396
    freq = 9450
    bank = 1
    dir_name = 'asymmetry_wtf'
    
    uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed/{bank}/'
    injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/injected/'
    uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_uninjected/'
    inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_injected/'
    
    # Get the targets
    #targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
    targs = os.listdir(uninjected_dir)
    print(f'# of targets in bank {bank}: {len(targs)}')
    
    do_inject = True
    if do_inject:
        for targ in targs:
            inject(targ, freq, info, data_dir=uninjected_dir, save_dir=injected_dir, signal_size=1e-28)

    plot_asymmetry(
        normalize_dynamic,
        fit_window=None, #20,
        freq=freq,
        width=100,
        nsigma=nsigma,
        #window=window,
        window_factor = window_factor,
        order=5,
        uninjected_dir=uninjected_dir,
        injected_dir=injected_dir,
        uninj_save_dir=uninj_save_dir,
        inj_save_dir=inj_save_dir,
        do_normalize=True,
        std_cut=90,
        fit_type='none',
        # fit_order=4
    )