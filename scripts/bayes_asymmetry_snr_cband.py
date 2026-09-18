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
import matplotlib as mpl
import datetime
from dateutil.relativedelta import relativedelta
import scipy

from normalization_methods import *

num_processes = 10#mp.cpu_count()

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

def calculate_snr(
        normalize_func,
        uninjected_dir=None,
        injected_dir=None,
        uninj_save_dir=None,
        inj_save_dir=None,
        fit_window=None,
        fit_type='quadratic',
        freq=8720,
        plot=True,
        dir_name=None,
        bank=0,
        **kwargs
    ):
    spectra = os.listdir(injected_dir)
    
    with mp.Pool(processes=num_processes) as p:
        for data_dir, save_dir in ((uninjected_dir, uninj_save_dir), (injected_dir, inj_save_dir)):
            normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
            _ = list(p.imap_unordered(normalize_func_partial, spectra))
    
    #print('Building asymmetries')
    xs, uninj_ys = build_asymmetry(uninj_save_dir, 'phi', 70, 115)
    xs, inj_ys = build_asymmetry(inj_save_dir, 'phi', 70, 115)

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
    
    # CALCULATE!
    snr = signal / noise

    if plot:
        fig, ax = plt.subplots()
        ax.plot(xs, inj_ys, color='r', label='injected')
        ax.plot(xs, uninj_ys, color='k', label='raw')
        if fit is not None:
            ax.plot(window[0], fit, color='blue', label='fit')
        elif fit_type == 'savgol':
            ax.plot(window[0], filt_window, color='blue', label='smoothed')
        else:
            index = np.argmax(window[1])
            ax.scatter(window[0][index], signal, marker='x', color='blue', label='max')
        ax.axvspan(*bounds, color='k', alpha=0.2)
        
        ax_inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
        ax_inset.hist(uninj_ys, density=True, color='gray', bins=20)
        mean, std = np.mean(uninj_ys), np.std(uninj_ys)
        xmin, xmax = ax_inset.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = scipy.stats.norm.pdf(x, mean, std)
        ax_inset.plot(x, p, c='k')
        ax_inset.set_yticks([])
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
        
        ax.text(0.03, 0.97, f'SNR = {round(snr, 2)}\n$\\sigma$ = {f"{std:.3g}"}',
                transform=ax.transAxes, ha='left', va='top', fontsize=16)

        ax.legend(loc='lower left')
        b = round(kwargs.get('b'), 3)
        exterior = round(kwargs.get('exterior'), 3)

        ax.set_xlabel('$\\nu$ [MHz]', fontsize=16)
        ax.set_ylabel('Asymmetry', fontsize=16)
        ax.set_title(f'b={b}, exterior={exterior}', fontsize=18)
        
        save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/{freq}'
        os.makedirs(save_path, exist_ok=True)
        
        fig.savefig(os.path.join(save_path, f'b_{b}_ext_{exterior}.png'), dpi=300, bbox_inches='tight')
    
    return snr

def normalize_number(value, desired_min, desired_max, original_min=0, original_max=1):
    return(value - original_min) * (desired_max - desired_min) / (original_max - original_min) + desired_min

def objective_weighted(b, exterior, **kwargs):
    b = normalize_number(b, 0.05, 1.95) #(window, 0, 1, 131, 301)
    exterior = normalize_number(exterior, 2, 5) #(window, 0, 1, 131, 301)
    
    return calculate_snr(
        normalize_weighted, #normalize
        fit_window=10,
        width=60,
        b=b,
        exterior=exterior, #window=window
        **kwargs
    )

if __name__ == '__main__':
    is_decay = True
    ref_signal_size = 3e-29 if is_decay else 5e-25
    power = 3 if is_decay else 4
    
    targets = []
    ddir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file:
        for line in file:
            targets.append(line.strip())
    print(f'Targets: {targets}')
    
    case = 'decay' if is_decay else 'ann'
    
    order = 5
    #orders = np.arange(3, 8)
    # maybe consider this later, but order 5 has been the best in my experience, so start with that. 6/5/26

    #for order in orders:
    dir_name = f'snr_evals_cband/bayes_weighted_{case}/'

    start_time = datetime.datetime.now()

    for bank in range(4):
        bank_start_time = datetime.datetime.now()
        print(f'Bank {bank} start!')

        uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/{bank}'
        injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/injected/{bank}'
        uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/normalized_uninjected/{bank}'
        inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/normalized_injected/{bank}'
        calibration_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/median_sefds/{bank}'

        # Get the targets
        targs = [targ for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
        print(f'# of targets in bank {bank}: {len(targs)}')
        targ_info = info[info['source'].isin(targs)]
        inner_targs = targ_info[targ_info['phi'] < 70]
        outer_targs = targ_info[targ_info['phi'] > 115]
        print(f'# of inner targets: {len(inner_targs)}')
        print(f'# of outer targets: {len(outer_targs)}')
        
        xs = np.load(os.path.join(uninjected_dir, os.listdir(uninjected_dir)[0]))[0]
        start = int(round(xs[0] + 200, -2)) # Round to nearest 100
        stop = int(round(xs[-1] - 200, -2))
        
        freqs = np.arange(start, stop, 100)
        print(f'Injection frequencies (n={len(freqs)}): {freqs}')

        for freq in freqs:
            print(f'Evaluating {freq} MHz (bank {bank})')
            freq_start_time = datetime.datetime.now()
            #print(f'Evaluating order {order}')
            
            # Inject the signals
            for targ in targs:
                signal_size = ref_signal_size * (freq / 5800)**power
                inject_calibrated_cband(
                    targ + '.npy',
                    freq,
                    info,
                    data_dir=uninjected_dir,
                    save_dir=injected_dir,
                    calibration_dir=calibration_dir,
                    signal_sizes=signal_size,
                    is_decay=is_decay
                )
            
            # Freeze the keywords
            objective_func = partial(
                objective_weighted, #objective
                uninjected_dir=uninjected_dir,
                injected_dir=injected_dir,
                uninj_save_dir=uninj_save_dir,
                inj_save_dir=inj_save_dir,
                fit_type=None,
                dir_name=dir_name,
                bank=bank,
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
            
            for _ in trange(15, desc='Optimizing'):
                next_point = optimizer.suggest()
                target = objective_func(**next_point)
                optimizer.register(params=next_point, target=target)
            
            # Save the results of the optimization
            os.makedirs(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}', exist_ok=True)
            optimizer.save_state(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/optimized_{freq}.json')

            # Plot the posterior
            x = np.linspace(pbounds['b'][0], pbounds['b'][1], 1000)
            y = np.linspace(pbounds['exterior'][0], pbounds['exterior'][1], 1000)
            xy = np.array([[x_i, y_j] for y_j in y for x_i in x])
            plot_x = np.linspace(0.05, 1.95, 1000)
            plot_y = np.linspace(2, 5, 1000)
            X, Y = np.meshgrid(plot_x, plot_y)

            Z_est = optimizer._gp.predict(xy).reshape(X.shape)

            fig, ax = plt.subplots()
            im = ax.pcolormesh(X, Y, Z_est, cmap=mpl.colormaps['magma'])
            cbar = fig.colorbar(im)
            cbar.set_label('SNR', fontsize=14)
            ax.contour(X, Y, Z_est, levels=5, colors='w', linewidths=0.5, linestyles='-')

            max_ = optimizer.max
            res = optimizer.res
            max_x, max_y = normalize_number(max_["params"]['b'], 0.05, 1.95), normalize_number(max_["params"]['exterior'], 2, 5)
            x_ = normalize_number(np.array([r["params"]['b'] for r in res]), 0.05, 1.95)
            y_ = normalize_number(np.array([r["params"]['exterior'] for r in res]), 2, 5)
            x_, y_ = list(zip(*[(x_i, y_j) for x_i, y_j in zip(x_, y_) if x_i != max_x or y_j != max_y]))

            ax.scatter(x_, y_, c='white', s=60, edgecolors='black')
            ax.scatter(max_x, max_y, c='red', s=120, marker='*', edgecolors='black')

            ax.set_xlabel('b', fontsize=14)
            ax.set_ylabel('exterior', fontsize=14)
            ax.set_title(f'$\\nu$ = {freq} MHz', fontsize=18) #f'order = {file.split("_")[1]}'
            ax.text(0.03, 0.97, f'$\\mathrm{{SNR}}_\\mathrm{{max}}$ = {round(max_["target"], 2)}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=14, color='w')

            save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/{freq}'
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, 'results.png'), dpi=300, bbox_inches='tight')

            freq_end_time = datetime.datetime.now()
            delta_freq = relativedelta(freq_end_time, freq_start_time)
            print(f'Freq {freq} done in {delta_freq.hours} hrs, {delta_freq.minutes} mins, {delta_freq.seconds} sec')
        
        bank_end_time = datetime.datetime.now()
        delta_bank = relativedelta(bank_end_time, bank_start_time)
        print(f'Bank {bank} done in {delta_bank.hours} hrs, {delta_bank.minutes} mins, {delta_bank.seconds} sec')
    
    end_time = datetime.datetime.now()
    delta = relativedelta(end_time, start_time)
    print(f'Study complete in {delta.hours} hrs, {delta.minutes} mins, {delta.seconds} sec')