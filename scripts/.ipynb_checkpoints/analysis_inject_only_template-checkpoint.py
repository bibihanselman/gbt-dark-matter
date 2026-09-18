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
    freq = 8000   
    bank = 0
    is_decay = True
    decay_or_ann = 'decay' if is_decay else 'ann'
    signal_size = 1e-28 if is_decay else 1e-25
    print(decay_or_ann)
    
    width = 80
    uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed/{bank}/'
    uninj_spectra = os.listdir(uninjected_dir)
    
    lower, upper = freq - width, freq + width
    trimmed_uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/trimmed/{decay_or_ann}/'
    os.makedirs(trimmed_uninjected_dir, exist_ok=True)
    
    for file in uninj_spectra:
        spec = np.load(os.path.join(uninjected_dir, file))
        trimmed = spec[:,np.argmin(np.abs(spec[0]-lower)):np.argmin(np.abs(spec[0]-upper))]
        np.save(os.path.join(trimmed_uninjected_dir, file), trimmed)

    injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/template/{decay_or_ann}/'

    with mp.Pool(processes=num_processes) as p:
        inject_partial = partial(inject, sig_loc=freq, info=info, data_dir=trimmed_uninjected_dir, save_dir=injected_dir, signal_size=signal_size, template=True, is_decay=is_decay)
        _ = list(tqdm(p.imap_unordered(inject_partial, uninj_spectra), total=len(uninj_spectra), desc='Injecting'))
