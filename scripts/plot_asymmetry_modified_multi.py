import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from tqdm import tqdm, trange
from functools import partial
import random
import matplotlib.pyplot as plt


from normalization_methods import *


num_processes = mp.cpu_count()

info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/all_xband_info.csv'
info = pd.read_csv(info_dir)

def sigma_filter(data, window_size=15):
    for i in range(window_size//2, len(data)-window_size//2):
        window_data = data[i-window_size//2:i+window_size//2+1]
        median = np.median(window_data)
        std = np.std(window_data)
        if abs(data[i] - median) > 3 * std:
            data[i] = median
    return data

def build_asymmetry(data_dir, angle, lower, upper, spectra=None):
    if spectra is None:
        spectra = os.listdir(data_dir)
    inner = info[info[angle] < lower]
    outer = info[info[angle] > upper]
    inner_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in inner['source'].tolist()])
    outer_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in outer['source'].tolist()])    
    inner_mean = np.mean(inner_spectra, axis=0)
    outer_mean = np.mean(outer_spectra, axis=0)
    asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)
    xs = np.load(os.path.join(data_dir, spectra[0]))[0]
    asymmetry = sigma_filter(asymmetry)
    return xs, asymmetry

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

def plot_asymmetry(normalize_func, data_dirs=None, save_dirs=None, snr_freqs=None, **kwargs):
    spectra = os.listdir(injected_dir)
    
    with mp.Pool(processes=num_processes) as p:
        for data_dir, save_dir in zip(data_dirs, save_dirs):
            skip = (os.path.exists(save_dir)) and (len(os.listdir(save_dir)) == len(spectra))
            if skip:
                print('Already normalized, skipping')
            else:
                normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, **kwargs)
                _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing {data_dir.split("/")[-2]} with {normalize_func.__name__}'))
    
    uninj_save_dir, inj_save_dir = save_dirs
    
    for mode in ('doppler', 'intensity'):
        args = ('theta', 65, 65) if mode == 'doppler' else ('phi', 70, 115)
        xs, uninj = build_asymmetry(uninj_save_dir, *args)
        xs, inj = build_asymmetry(inj_save_dir, *args)
        
        xs, uninj_mod = build_asymmetry_modified(uninj_save_dir, *args)
        xs, inj_mod = build_asymmetry_modified(inj_save_dir, *args)

        # Calculate SNRs
        def calculate_snrs(freq):
            mask = (xs >= freq - 50) & (xs <= freq + 50)
            snr = np.max(inj[mask]) / np.std(uninj[mask])
            snr_mod = np.max(inj_mod[mask]) / np.std(uninj_mod[mask])
            factor = snr_mod / snr
            return freq, snr, snr_mod, factor

        snr_list = list(map(calculate_snrs, snr_freqs))
        snr_df = pd.DataFrame(snr_list, columns=['Frequency', 'SNR', 'SNR_Modified', 'Factor'])
        snr_df.to_csv(f'/home/bhanselman/bhanselman/asymmetry_plots/modified_tests/bank1_{mode}_1e28_snrs.csv', index=False)
        
        # Plotting
        fig, ax = plt.subplots()
        ax.plot(xs, uninj, color='green', label='raw')
        #ax.plot(xs, inj, color='green', label='injected')
        ax.plot(xs, uninj_mod, color='blue', label='raw_mod')
        #ax.plot(xs, inj_mod, color='blue', label='injected_mod')
        ax.legend()
        
        plt.savefig(f'/home/bhanselman/bhanselman/asymmetry_plots/modified_tests/bank1_{mode}_raw.png', dpi=300)


if __name__ == '__main__':
    targets = []
    ddir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file:
        for line in file:
            targets.append(line.strip())
    
    # Values
    exterior = 8
    freqs = np.arange(9050, 10100, 100)
    bank = 1
    dir_name = 'modified_asymmetry_test_multi'
    
    uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed/{bank}/'
    injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/injected/'
    uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_uninjected/'
    inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_injected/'
    
    # Get the targets
    #targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
    targs = os.listdir(uninjected_dir)
    print(f'# of targets in bank {bank}: {len(targs)}')
    
    calibration_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/median_sefds/{bank}'
    do_inject = True
    if do_inject:
        for targ in targs:
            inject_calibrated(
                targ,
                freqs,
                info,
                data_dir=uninjected_dir,
                save_dir=injected_dir,
                calibration_dir=calibration_dir,
                signal_size=1e-28
            )

    plot_asymmetry(
        normalize_weighted,
        start=9000,
        stop=10100,
        exterior=exterior,
        order=5,
        data_dirs=(uninjected_dir, injected_dir),
        save_dirs=(uninj_save_dir, inj_save_dir),
        snr_freqs=freqs
    )