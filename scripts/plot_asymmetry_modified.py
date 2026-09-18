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

    print(f'% of inner post spectra: {len(inner_post_spectra) / (len(inner_pre_spectra) + len(inner_post_spectra))},\
           % of outer post spectra: {len(outer_post_spectra) / (len(outer_pre_spectra) + len(outer_post_spectra))}')

    inner_pre_mean = np.mean(inner_pre_spectra, axis=0)
    inner_post_mean = np.mean(inner_post_spectra, axis=0)
    outer_pre_mean = np.mean(outer_pre_spectra, axis=0)
    outer_post_mean = np.mean(outer_post_spectra, axis=0)

    asymmetry = (inner_pre_mean + inner_post_mean - outer_pre_mean - outer_post_mean) / (inner_pre_mean + inner_post_mean + outer_pre_mean + outer_post_mean)
    xs = np.load(os.path.join(data_dir, spectra[0]))[0]
    asymmetry = sigma_filter(asymmetry)
    return xs, asymmetry

def plot_asymmetry(normalize_func, data_dirs=None, save_dirs=None, freq=8720, do_normalize=True, **kwargs):
    spectra = os.listdir(injected_dir)
    
    if do_normalize:
        with mp.Pool(processes=num_processes) as p:
            for data_dir, save_dir in zip(data_dirs, save_dirs):
                normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
                _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing {data_dir.split("/")[-2]} with {normalize_func.__name__}'))
    
    uninj_save_dir, inj_save_dir = save_dirs

    # xs, uninj_dopp = build_asymmetry(uninj_save_dir, 'theta', 65, 65)
    # xs, inj_dopp = build_asymmetry(inj_save_dir, 'theta', 65, 65)
    
    # uninj_dopp, inj_dopp = list(map(sigma_filter, [uninj_dopp, inj_dopp]))
    
    xs, uninj_int = build_asymmetry(uninj_save_dir, 'theta', 65, 65)
    xs, inj_int = build_asymmetry(inj_save_dir, 'theta', 65, 65)
    
    # pre22B = info[info['time'].apply(lambda t: convert_to_seconds(t) < 5.18e9)]['source'].tolist()
    # post22B = info[info['time'].apply(lambda t: convert_to_seconds(t) >= 5.18e9)]['source'].tolist()

    # xs, uninj_int_pre = build_asymmetry(uninj_save_dir, 'phi', 70, 115, spectra=[file for file in os.listdir(uninj_save_dir) if file[:-4] in pre22B])
    # xs, inj_int_pre = build_asymmetry(inj_save_dir, 'phi', 70, 115, spectra=[file for file in os.listdir(uninj_save_dir) if file[:-4] in pre22B])
    
    # xs, uninj_int_post = build_asymmetry(uninj_save_dir, 'phi', 70, 115, spectra=[file for file in os.listdir(uninj_save_dir) if file[:-4] in post22B])
    # xs, inj_int_post = build_asymmetry(inj_save_dir, 'phi', 70, 115, spectra=[file for file in os.listdir(uninj_save_dir) if file[:-4] in post22B])
    
    xs, uninj_int_mod = build_asymmetry_modified(uninj_save_dir, 'theta', 65, 65)
    xs, inj_int_mod = build_asymmetry_modified(inj_save_dir, 'theta', 65, 65)

    order = int(math.floor(math.log10(max(abs(inj_int_mod)))))
    uninj_int_mod = uninj_int_mod * 10**-order
    inj_int_mod = inj_int_mod * 10**-order

    # Calculate SNRs
    snr = np.max(inj_int) / np.std(uninj_int)
    snr_mod = np.max(inj_int_mod) / np.std(uninj_int_mod)
    print(f'SNR (original): {snr}, SNR (modified): {snr_mod}')
    
    # Plotting
    fig, ax = plt.subplots()
    #ax.plot(xs, uninj_int, color='green', linestyle='--', label='raw')
    #ax.plot(xs, inj_int, color='green', label='injected')
    ax.plot(xs, uninj_int_mod, color='k', linestyle='--', label='raw')
    ax.plot(xs, inj_int_mod, color='r', label='injected')
    ax.legend()
    ax.set_xlabel(r'$\nu$ [MHz]', fontsize=16)
    ax.set_ylabel(rf'Asymmetry ($\times 10^{{{order}}}$)', fontsize=16)
    
    #plt.savefig(f'/home/bhanselman/bhanselman/asymmetry_plots/{freq}_modified_test_new_formula_doppler_1e28.png', dpi=300)
    plt.savefig(f'/home/dataadmin/GBTData/SharedDataDirectory/{freq}_modified_aas.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 12})

    targets = []
    ddir ='/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file: #random_names
        for line in file:
            targets.append(line.strip())
    #print(f'Targets: {targets}')
    
    # Values
    exterior = 8
    freq = 9450
    bank = 1
    dir_name = 'modified_asymmetry_test_new_formula'
    
    uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed/{bank}/'
    injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/injected/'
    uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_uninjected/'
    inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/{dir_name}/normalized_injected/'
    
    # Get the targets
    #targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
    targs = os.listdir(uninjected_dir)
    print(f'# of targets in bank {bank}: {len(targs)}')
    
    calibration_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/median_sefds/{bank}'
    do_inject = False
    if do_inject:
        for targ in targs:
            inject_calibrated(
                targ,
                freq,
                info,
                data_dir=uninjected_dir,
                save_dir=injected_dir,
                calibration_dir=calibration_dir,
                signal_sizes=1e-28
            )

    plot_asymmetry(
        normalize_weighted,
        freq=freq,
        width=80,
        exterior=exterior,
        order=5,
        data_dirs=(uninjected_dir, injected_dir),
        save_dirs=(uninj_save_dir, inj_save_dir),
        do_normalize=False,
    )