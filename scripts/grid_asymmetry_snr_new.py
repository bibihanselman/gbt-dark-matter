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
from bayes_opt import BayesianOptimization, acquisition
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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


def calculate_snr(normalize_func, uninjected_dir=None, injected_dir=None, uninj_save_dir=None, inj_save_dir=None, fit_window=None, fit_type='quadratic', freq=8720, plot=True, dir_name=None, **kwargs):
    spectra = os.listdir(injected_dir)
    
    with mp.Pool(processes=num_processes) as p:
        for data_dir, save_dir in ((uninjected_dir, uninj_save_dir), (injected_dir, inj_save_dir)):
            normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
            _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing {data_dir.split("/")[-2]} with {normalize_func.__name__}'))
    
    #print('Building asymmetries')
    xs, uninj_ys = build_asymmetry(uninj_save_dir, 'phi', 70, 115)
    xs, inj_ys = build_asymmetry(inj_save_dir, 'phi', 70, 115)
    
    def sigma_filter(data, window_size):
        for i in range(window_size//2, len(data)-window_size//2):
            window_data = data[i-window_size//2:i+window_size//2+1]
            median = np.median(window_data)
            std = np.std(window_data)
            if abs(data[i] - median) > 3 * std:
                data[i] = median
        return data
    
    #print('Calculating SNR')
    noise_window = 3 * fit_window
    # Calculate noise
    mask = abs(xs - freq) < noise_window
    window = uninj_ys[mask]
    filt_noise_window = sigma_filter(window, 15)
    noise = np.std(filt_noise_window)
        
    # Calculate signal amplitude
    mask = abs(xs - freq) < fit_window
    window = [xs[mask], inj_ys[mask]]
    bounds = [min(window[0]), max(window[0])]
    fit = None
    
    if fit_type == 'quadratic':
        params = np.polyfit(*window, 2) #quadratic for intensity
        fit = np.poly1d(params)(window[0])
        signal = np.max(fit)# - np.mean(window) #maybe subtract mean? 7/16/25
    elif fit_type == 'gaussian':
        # Trying a Gaussian instead of quadratic. 7/19/25
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / (2 * stddev))**2)

        try:
            params, _ = curve_fit(
                gaussian,
                *window,
                bounds=((0, bounds[0], 0),(np.inf, bounds[1], np.inf)))
            print(params)
            fit = gaussian(window[0], *params)
            signal = np.max(fit)# - np.mean(window) #maybe subtract mean? 7/16/25
        except:
            print('Gaussian fit failed...')
            signal = 0
    elif fit_type == 'savgol':
        #Sigma filter...
        window_copy = sigma_filter(window[1], 15)
        #Savgol filter...
        filt_window = savgol_filter(window_copy, 30, 4)
        signal = np.max(filt_window)
    else:
        signal = np.max(window[1])
    
    if signal < 0:
        signal = 0
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(xs, uninj_ys, label='raw')
        ax.plot(xs, inj_ys, label='injected')
        if fit is not None:
            ax.plot(window[0], fit, color='red', label='fit')
        elif fit_type == 'savgol':
            ax.plot(window[0], filt_window, color='red', label='smoothed')
        else:
            index = window[1].tolist().index(signal)
            if not isinstance(index, int):
                index = index[0]
            ax.scatter(window[0][index], signal, marker='x', color='red', label='max')
        ax.axvspan(*bounds, color='k', alpha=0.2)
        ax.legend()
        
        save_path = f'/home/bhanselman/asymmetry_plots/{dir_name}/{freq}'
        os.makedirs(save_path, exist_ok=True)
        
        nsigma, a, window_factor = kwargs.get('nsigma'), kwargs.get('a'), kwargs.get('window_factor')
        if nsigma:
            nsigma = round(nsigma, 4)
        if a:
            a = round(a, 4)
        if window_factor:
            window_factor = round(window_factor, 4)
        name = f'{nsigma}_{window_factor}' if normalize_func.__name__ == 'normalize_dynamic' else f'{a}_{window_factor}'
        fig.savefig(os.path.join(save_path, name) + '.png')
    
    # Return SNR
    return signal / noise


if __name__ == '__main__':
    freq_starts = [7900, 9050, 10050]
    banks = [0, 1, 2]
    
    #targets = random.sample(info['source'].tolist(), 1000)
    targets = []
    ddir ='/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file:
        for line in file:
            targets.append(line.strip())
    print(f'Targets: {targets}')
    
    
    # Values for the grid search
    pg_params = list(itertools.product(np.arange(1.0, 2.1, 0.5), np.arange(0.025, 0.050, 0.005)))
    
    for bank in banks:    
        dir_name = f'snr_grid_avg_more_windows_{bank}'
        print(f'Evaluating bank {bank}')
        freqs = np.arange(freq_starts[bank], freq_starts[bank] + 1000, 100)

        for freq in freqs:
            print(f'Evaluating {freq} MHz')

            uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/{bank}/preprocessed_despiked/'
            injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/{dir_name}/injected/'
            uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/{dir_name}/normalized_uninjected/'
            inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/{dir_name}/normalized_injected/'

            # Get the targets
            targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
            print(f'# of targets in bank {bank}: {len(targs)}')
            
            signal_size = 1e-26 if bank == '2' else 1e-27
            # Inject those fuckers
            for targ in targs:
                inject(targ, freq, info, data_dir=uninjected_dir, save_dir=injected_dir, signal_size=signal_size)

            snrs = []
            for pg_param in tqdm(pg_params):
                nsigma, window_factor = pg_param
                snr = calculate_snr(
                    normalize_dynamic,
                    fit_window=20,
                    freq=freq,
                    width=80,
                    nsigma=nsigma,
                    window_factor=window_factor,
                    order=5,
                    uninjected_dir=uninjected_dir,
                    injected_dir=injected_dir,
                    uninj_save_dir=uninj_save_dir,
                    inj_save_dir=inj_save_dir,
                    dir_name=dir_name,
                    fit_type='savgol'
                )
                snrs.append(snr)


            # Save the SNRs for the bank
            np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/{dir_name}/snrs_{freq}.npy', snrs) 