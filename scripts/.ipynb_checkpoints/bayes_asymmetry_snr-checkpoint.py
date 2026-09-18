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
from scipy.stats import percentileofscore

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


def calculate_snr(normalize_func, uninjected_dir=None, injected_dir=None, uninj_save_dir=None, inj_save_dir=None, fit_window=None, fit_type='quadratic', freq=8720, plot=True, dir_name=None, bank=None, **kwargs):
    spectra = os.listdir(injected_dir)
    
    print(f'params: {kwargs.get("interior")} {kwargs.get("exterior")}')
    with mp.Pool(processes=num_processes) as p:
        for data_dir, save_dir in ((uninjected_dir, uninj_save_dir), (injected_dir, inj_save_dir)):
            normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
            _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing {data_dir.split("/")[-2]} with {normalize_func.__name__}'))
    
    # # Cut the uninjected+injected normalized spectra with the highest stds
    # if std_cut:
    #     for data_dir in (uninj_save_dir, inj_save_dir):
    #         files = os.listdir(data_dir)
    #         spectra = [np.load(os.path.join(data_dir, file))[1] for file in files]
    #         stds = [np.std(spec) for spec in spectra]
    #         for file, std in zip(files, stds):
    #             if percentileofscore(stds, std) > std_cut:
    #                 os.remove(os.path.join(data_dir, file))
    
    def sigma_filter(data, window_size=15):
        for i in range(window_size//2, len(data)-window_size//2):
            window_data = data[i-window_size//2:i+window_size//2+1]
            median = np.median(window_data)
            std = np.std(window_data)
            if abs(data[i] - median) > 3 * std:
                data[i] = median
        return data
    
    #print('Building asymmetries')
    xs, uninj_ys = build_asymmetry(uninj_save_dir, 'phi', 70, 115)
    xs, inj_ys = build_asymmetry(inj_save_dir, 'phi', 70, 115)
    
    uninj_ys, inj_ys = sigma_filter(uninj_ys), sigma_filter(inj_ys)
    
    uninj_ys_filt = sigma_filter(uninj_ys, 15)
    inj_ys_filt = sigma_filter(inj_ys, 15)

    #print('Calculating SNR')
    noise_window = 3 * fit_window
    # Calculate noise
    mask = abs(xs - freq) < noise_window
    window = uninj_ys_filt[mask]
    noise = np.std(window)
        
    # Calculate signal amplitude
    mask = abs(xs - freq) < fit_window
    window = [xs[mask], inj_ys_filt[mask]]
    bounds = [min(window[0]), max(window[0])]
    fit = None
    
    if fit_type == 'quadratic':
        params = np.polyfit(*window, 2) #quadratic for intensity
        fit = np.poly1d(params)(window[0])
        signal = np.max(fit)# - np.mean(window)
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
            signal = np.max(fit)# - np.mean(window)
        except:
            print('Gaussian fit failed...')
            signal = 0
    elif fit_type == 'savgol':
        #Savgol filter...
        filt_window = savgol_filter(window[1], 30, 4)
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
        
        interior, a, exterior = kwargs.get('interior'), kwargs.get('a'), kwargs.get('exterior')
        if interior:
            interior = round(interior, 4)
        if a:
            a = round(a, 4)
        if exterior:
            exterior = round(exterior, 4)
        name = f'{interior}_{exterior}' if normalize_func.__name__ == 'normalize_dynamic' else f'{a}_{exterior}'
        fig.savefig(os.path.join(save_path, name) + '.png')
    
    # Return SNR
    return signal / noise

    
def normalize_number(value, original_min, original_max, desired_min, desired_max):
    return(value - original_min) * (desired_max - desired_min) / (original_max - original_min) + desired_min

def objective(interior, exterior, **kwargs):
    
    interior = normalize_number(interior, 0, 1, 1, 2) #1, 2
    exterior = normalize_number(exterior, 0, 1, 3, 9) #(window, 0, 1, 131, 301)
    
    return calculate_snr(
        normalize_dynamic, #normalize
        fit_window=20,
        width=70,
        interior=interior,
        exterior=exterior, #window=window
        **kwargs
    )

def objective_weighted(a, exterior, **kwargs):
    
    def normalize_number(value, original_min, original_max, desired_min, desired_max):
        return (value - original_min) * (desired_max - desired_min) / (original_max - original_min) + desired_min
    
    a = normalize_number(a, 0, 1, 0.05, 1)
    exterior = normalize_number(exterior, 0, 1, 3, 9) #(window, 0, 1, 131, 301)
    
    return calculate_snr(
        normalize_weighted, #normalize
        fit_window=20,
        width=70,
        a=a,
        exterior=exterior, #window=window
        **kwargs
    )

if __name__ == '__main__':
    freqs = np.arange(7900, 8900, 200) #[8720, 9950]#, 11080]
    bank = 0
    is_decay = True
    ref_signal_size = 1e-28 if is_decay else 5e-25
    power = 3 if is_decay else 4
    
    targets = []
    ddir ='/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file: #random_names.txt #400
        for line in file:
            targets.append(line.strip())
    print(f'Targets: {targets}')
    
    case = 'decay' if is_decay else 'ann'
    dir_name = f'snr_evals/bayes_weighted_{case}/{bank}'
    
    for freq in freqs:#for freq, bank in zip(freqs, banks): #for order in orders:
        print(f'Evaluating {freq} MHz (bank {bank})')
        #print(f'Evaluating order {order}')
        
        uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed/{bank}/'
        injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/{freq}/injected/'
        uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/{freq}/normalized_uninjected/'
        inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/{freq}/normalized_injected/'

        # Get the targets
        targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
        #print(f'# of targets in bank {bank}: {len(targs)}')
        
        # Inject the signals
        for targ in targs:
            signal_size = ref_signal_size * (freq / 8500)**power
            inject(targ, freq, info, data_dir=uninjected_dir, save_dir=injected_dir, signal_size=signal_size, is_decay=is_decay)
        
        # Freeze the keywords
        objective_func = partial(
            objective_weighted, #objective
            uninjected_dir=uninjected_dir,
            injected_dir=injected_dir,
            uninj_save_dir=uninj_save_dir,
            inj_save_dir=inj_save_dir,
            fit_type='savgol',
            dir_name=dir_name,
            bank=bank,
            freq=freq,
            is_decay=is_decay
        )
            
        acq = acquisition.UpperConfidenceBound(kappa=2) #4

        optimizer = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds={'a': (0, 1), 'exterior': (0, 1)}, #nsigma #{'nsigma': (0.5, 2.5), 'window': (131, 301, int)},
            verbose=0,
            random_state=1,
            allow_duplicate_points=True
        )

        for _ in trange(15): #25
            next_point = optimizer.suggest()
            target = objective_func(**next_point)
            optimizer.register(params=next_point, target=target)
        
        # Save the results of the optimization
        optimizer.save_state(f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/optimized_{freq}.json')