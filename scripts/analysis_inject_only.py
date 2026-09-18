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


if __name__ == '__main__':
    full_band = True
    start = 8000
    stop = 8500
    spacing = 100
    step = 1
        
    banks = [0, 1, 2, 3]
    signal_sizes = [1e-24, 5e-25, 2e-25, 1e-25, 5e-26, 3e-26, 2e-26, 1e-26]
    # Decay: [4e-28, 2e-28, 1e-28, 5e-29, 3e-29, 2e-29, 1e-29, 5e-30]
    # Annihilation: [1e-24, 5e-25, 2e-25, 1e-25, 5e-26, 3e-26, 2e-26, 1e-26]
    is_decay = False
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
