import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import random
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from scipy.optimize import curve_fit
from tqdm import tqdm, trange
from bayes_opt import BayesianOptimization, acquisition
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import percentileofscore

from normalization_methods import *


num_processes = mp.cpu_count()

info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/all_xband_info.csv'
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

def build_asymmetry_modified(data_dir, angle, lower, upper, spectra=None):
    if spectra is None:
        spectra = os.listdir(data_dir)
    
    inner_pre = info[(info[angle] < lower) & (info['time'].apply(lambda t: convert_to_seconds(t) < 5.18e9))]
    inner_post = info[(info[angle] < lower) & (info['time'].apply(lambda t: convert_to_seconds(t) >= 5.18e9))]
    outer_pre = info[(info[angle] > upper) & (info['time'].apply(lambda t: convert_to_seconds(t) < 5.18e9))]
    outer_post = info[(info[angle] > upper) & (info['time'].apply(lambda t: convert_to_seconds(t) >= 5.18e9))]

    inner_pre_spectra = [np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in inner_pre['source'].tolist()]
    inner_post_spectra = [np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in inner_post['source'].tolist()]
    outer_pre_spectra = [np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in outer_pre['source'].tolist()]
    outer_post_spectra = [np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in outer_post['source'].tolist()]

    inner_pre_mean = np.mean(inner_pre_spectra, axis=0)
    inner_post_mean = np.mean(inner_post_spectra, axis=0)
    outer_pre_mean = np.mean(outer_pre_spectra, axis=0)
    outer_post_mean = np.mean(outer_post_spectra, axis=0)

    asymmetry = (inner_pre_mean + inner_post_mean - outer_pre_mean - outer_post_mean) / (inner_pre_mean + inner_post_mean + outer_pre_mean + outer_post_mean)
    xs = np.load(os.path.join(data_dir, spectra[0]))[0]
    asymmetry = sigma_filter(asymmetry)
    return xs, asymmetry


def calculate_snr(normalize_func, uninjected_dir=None, injected_dir=None, uninj_save_dir=None, inj_save_dir=None, fit_window=None, fit_type='quadratic', freq=8720, plot=True, dir_name=None, **kwargs):
    spectra = os.listdir(injected_dir)
    
    with mp.Pool(processes=num_processes) as p:
        for data_dir, save_dir in ((uninjected_dir, uninj_save_dir), (injected_dir, inj_save_dir)):
            normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
            _ = list(p.imap_unordered(normalize_func_partial, spectra))
    
    #print('Building asymmetries')
    xs, uninj_ys = build_asymmetry_modified(uninj_save_dir, 'phi', 70, 115)
    xs, inj_ys = build_asymmetry_modified(inj_save_dir, 'phi', 70, 115)

    #print('Calculating SNR')
    noise_window = 3 * fit_window
    # Calculate noise
    mask = abs(xs - freq) < noise_window
    window = uninj_ys[mask]
    noise = np.std(window)
        
    # Calculate signal amplitude
    mask = abs(xs - freq) < fit_window
    window = [xs[mask], inj_ys[mask]]
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
        
        save_path = f'/home/bhanselman/bhanselman/asymmetry_plots/{dir_name}/{freq}'
        os.makedirs(save_path, exist_ok=True)
        
        exterior = round(kwargs.get('exterior'), 3)
        name = str(exterior)
        fig.savefig(os.path.join(save_path, name) + '.png')
    
    # Return SNR
    return signal / noise

    
def normalize_number(value, original_min, original_max, desired_min, desired_max):
    return(value - original_min) * (desired_max - desired_min) / (original_max - original_min) + desired_min

def objective_weighted(exterior, **kwargs):
    exterior = normalize_number(exterior, 0, 1, 3, 9) #(window, 0, 1, 131, 301)
    
    return calculate_snr(
        normalize_weighted, #normalize
        fit_window=20,
        width=70,
        exterior=exterior, #window=window
        **kwargs
    )

if __name__ == '__main__':
    freqs = np.arange(9050, 10100, 100)
    bank = 1
    is_decay = True
    ref_signal_size = 1e-28 if is_decay else 5e-25
    power = 3 if is_decay else 4
    
    targets = []
    ddir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file:
        for line in file:
            targets.append(line.strip())
    print(f'Targets: {targets}')
    
    case = 'decay' if is_decay else 'ann'
    
    orders = np.arange(3, 8)
    calibration_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/median_sefds/{bank}'

    for order in orders:
        dir_name = f'snr_evals/bayes_weighted_modified_{case}/order_{order}'

        for freq in freqs:
            print(f'Evaluating {freq} MHz (bank {bank})')
            #print(f'Evaluating order {order}')
            
            uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed/{bank}/'
            injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/injected/'
            uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_uninjected/'
            inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_injected/'

            # Get the targets
            targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
            #print(f'# of targets in bank {bank}: {len(targs)}')
            
            # Inject the signals
            for targ in targs:
                signal_size = ref_signal_size * (freq / 8500)**power
                inject_calibrated(
                    targ,
                    freq,
                    info,
                    data_dir=uninjected_dir,
                    save_dir=injected_dir,
                    calibration_dir=calibration_dir,
                    signal_size=signal_size,
                    is_decay=is_decay
                )
            
            # Freeze the keywords
            objective_func = partial(
                objective_weighted, #objective
                uninjected_dir=uninjected_dir,
                injected_dir=injected_dir,
                uninj_save_dir=uninj_save_dir,
                inj_save_dir=inj_save_dir,
                fit_type='savgol',
                dir_name=dir_name,
                freq=freq,
                order=order,
                is_decay=is_decay
            )
                
            acq = acquisition.UpperConfidenceBound(kappa=2) #4

            optimizer = BayesianOptimization(
                f=None,
                acquisition_function=acq,
                pbounds={'exterior': (0, 1)},
                verbose=0,
                random_state=1,
                allow_duplicate_points=True
            )

            # 5 random points
            random.seed(42)
            random_points = [random.random() for _ in range(5)]
            for point in tqdm(random_points, desc='Initializing'):
                target = objective_func(point)
                optimizer.register(params=point, target=target)
            
            for _ in trange(10, desc='Optimizing'):
                next_point = optimizer.suggest()
                target = objective_func(**next_point)
                optimizer.register(params=next_point, target=target)
            
            # Save the results of the optimization
            optimizer.save_state(f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/optimized_{freq}.json')