from functools import partial
import gc
from tqdm.auto import tqdm
import pandas as pd
import os, time, traceback
from multiprocessing import Pool
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
import h5py
from blimpy import Waterfall
from numpy.polynomial import Polynomial
from itertools import groupby
import json
import warnings
from scipy.integrate import quad
import math
import argparse
warnings.filterwarnings("ignore")

### Integrals

R_s = 7.9 # distance from the center of galaxy to the sun

# Parameters for Navarro-Frenk-White
rho_c_nfw = 1.4e7
r_c_nfw = 16.1

# Parameters for Isothermal (UNKNOWN)
rho_c_iso = rho_c_nfw
r_c_iso = r_c_nfw

# Parameters for Burkert
rho_c_burk = 3.68e7
r_c_burk = 9.06

# Parameters for Moore
rho_c_moore = 1.46e6
r_c_moore = 16.5

def r_prime(r, b, l):
    # find the coordinate-transformed radius
    return math.sqrt(r*r + R_s*R_s - 2*R_s*r*math.cos(b)*math.cos(l))

def nfw(r_prime):
    ratio = r_prime / r_c_nfw
    return rho_c_nfw / (ratio * (1+ratio) * (1+ratio))

def isothermal(r_prime):
    return rho_c_iso / (1 + (r_prime / r_c_iso) * (r_prime / r_c_iso))

def burkert(r_prime):
    ratio = r_prime / r_c_burk
    return rho_c_burk / ((1+ratio) * (1 + ratio * ratio))

def moore(r_prime):
    ratio_raised = math.pow(r_prime/r_c_moore, 3/2)
    return rho_c_moore / (ratio_raised * (1 + ratio_raised))

def integrate(func, target_b_deg, target_l_deg, annihilation=False):
    target_b, target_l = math.radians(target_b_deg), math.radians(target_l_deg)
    def integrand(r):
        val = func(r_prime(r, target_b, target_l))
        return val*val if annihilation else val
    return quad(integrand, 0, float('inf'))

models = {
    'nfw': nfw,
    'iso': isothermal,
    'burkert': burkert,
    'moore': moore
}

### Preprocessing functions

def despike(ys, cell_size=1024):
    '''Remove the spike at the midpoint of the polyphase filterbank.'''
    for i in range(cell_size//2-1, len(ys), cell_size):
        ys[i] = np.mean([ys[i-1], ys[i+2]])
        ys[i+1] = np.mean([ys[i], ys[i+2]])
    return ys

def rebin(xs, ys, cell_size=16):
    '''Reduce spectral resolution (default: 16 bins/coarse channel).'''
    rebin_size=1024//cell_size
    rebinned_spec = []
    for i in range(len(ys)//rebin_size):
        average = np.mean(ys[i*rebin_size:i*rebin_size+rebin_size])
        rebinned_spec.append(average)

    rebinned_freq = []
    for i in range(len(xs)//rebin_size):
        average = np.mean(xs[i*rebin_size:i*rebin_size+rebin_size])
        rebinned_freq.append(average)
    return rebinned_freq, rebinned_spec

def sigma_filter(data_, window_size=15):
    '''Apply a sigma filter to remove residual spikes.'''
    data = data_.copy()
    for i in range(window_size//2, len(data)-window_size//2):
        window_data = data[i-window_size//2:i+window_size//2+1]
        median = np.median(window_data)
        std = np.std(window_data)
        if abs(data[i] - median) > 3 * std:
            data[i] = median
    return data

def comb(spec):
    '''Eliminate the polyphase filterbank structure in each coarse channel.'''
    num_windows = len(spec) // 16
    reshaped_power = np.reshape(spec, (num_windows, 16))

    new_power = []
    for i in range(len(reshaped_power)):
        p = reshaped_power[i]
        c = rebinned_comb
        new_power.append(p / c)
    
    return np.concatenate(new_power)

def cliff_correct(ys):
    '''Eliminate power discontinuities between data blocks.'''
    n_nodes = len(ys) // 1024
    reshaped = np.reshape(ys, (n_nodes, 1024)).copy()

    for i in range(n_nodes-1):
        left = reshaped[i]
        right = reshaped[i+1]
        diff = np.abs(right[0] - left[-1])
        avg = (np.abs(right[1] - right[0]) + np.abs(left[-1] - left[-2])) / 2
        if diff > 5 * avg: # Condition for cliff correction
            # Fit polynomial to middle 12 points of coarse channel
            p = Polynomial.fit(np.arange(12), left[-14:-2], 1) #linear
            y_fit = p(16)
            a = y_fit / right[2]
            for j in range(i+1, n_nodes):
                reshaped[j] *= a
    
    out = np.concatenate(reshaped)
    return out

def sort_key(l):
    return l.split('/')[3], l.split('/')[-1].split('_')[-4:-2]

def get_info(loc):
    '''Obtain metadata for observation target.'''
    source = loc.split('/')[-1].split('_')[-2]
    epoch = loc.split('/')[3]
    time = '_'.join(loc.split('/')[-1].split('_')[-4:-2])

    with h5py.File(loc, 'r') as f:
        ha = f['data'].attrs['src_raj']
        angle = Angle(ha, u.hourangle)
        ra = angle.degree
        dec = f['data'].attrs['src_dej']

    icrs_coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    galactic_coords = icrs_coords.galactic
    l, b = galactic_coords.l.deg, galactic_coords.b.deg
    v_sun = SkyCoord(90, 0, unit='deg', frame='galactic')
    v_gc = SkyCoord(0, 0, unit='deg', frame='galactic')
    theta = v_sun.separation(galactic_coords).deg
    phi = v_gc.separation(galactic_coords).deg

    cols = ('source', 'epoch', 'time', 'ra', 'dec', 'l', 'b', 'theta', 'phi')
    data = (source, epoch, time, ra, dec, l, b, theta, phi)
    data_dict = dict(zip(cols, data))

    for model, func in models.items():
        decay_int, _ = integrate(func, b, l, annihilation=False)
        ann_int, _ = integrate(func, b, l, annihilation=True)
        data_dict[model] = decay_int
        data_dict[f'{model}_sq'] = ann_int

    return data_dict

def process(loc):
    '''Apply all preprocessing steps to one spectrum.'''
    try:
        wf = Waterfall(loc)
    except:
        log(f'File is corrupted: {loc}')
        return None
    xs, ys = wf.grab_data()        
    xs = xs[::-1]  # reverse the x-axis
    ys = ys[..., ::-1]  # reverse the y-axis
    ys_avg = np.mean(ys, axis=0)
    ys_despiked = despike(ys_avg)
    xs_rebin, ys_rebin = rebin(xs, ys_despiked)
    ys_combed = comb(ys_rebin)
    ys_filt = sigma_filter(ys_combed)
    rebinned = np.array([xs_rebin, ys_filt])
    return rebinned
        
def get_files(locs):
    '''Process and splice all spectra in a single bank.'''
    # def process_wrapper(loc):
    #     try:
    #         return process(loc)
    #     except Exception as e:
    #         log(f'Error processing {loc}, skipping: {e}')
    #         return None
    
    specs = list(map(process, locs))
    specs = [spec for spec in specs if spec is not None] # Remove failed files
    bank_spec = np.concatenate(specs, axis=1)
    sort_indices = np.argsort(bank_spec[0])
    bank_spec = bank_spec[:, sort_indices]
    bank_spec[1] = cliff_correct(bank_spec[1])
    return bank_spec

### Preprocessing/splicing routine

# Standard bank ranges (hard coded but might make customizable later 5/27/26)
ranges = {
    '0': 6144,
    '1': 8192,
    '2': 8192,
    '3': 7168
}

def log(msg):
    with open(log_path, "a", buffering=1) as f:
        f.write(msg + "\n")

def save_bank(items, name=None):
    bank, locs = items
    
    epochs = [list(i) for j, i in groupby(sorted(locs, key=sort_key), key=sort_key)]
    locs = max(epochs, key=len)

    try:
        bank_spec = get_files(locs)

        # Throw out spectrum if the start frequency is below 3000 MHz
        if bank_spec[0][0] < 3000:
            log(f"Spectrum falls outside C band. Skipping bank {bank} with name {name}.")
            return
        
        # Throw out spectrum if there are frequency gaps between data blocks
        diffs = np.diff(bank_spec[0])
        if max(diffs) > 2 * diffs[0]:
            log(f"Spliced spectrum has a frequency gap. Skipping bank {bank} with name {name}.")
            return
        
        size = len(bank_spec[0])

        # Trim bank 0 and 3 spectra to 6 and 7 nodes respectively (ad hoc)
        if bank == '0' and size == 8192:
            bank_spec = bank_spec[:, 2048:]
        if bank == '3' and size == 8192:
            bank_spec = bank_spec[:, :-1024]
        
        # Throw out spectrum that is the wrong size
        if len(bank_spec[0]) != ranges[bank]:
            log(f"Spectrum is the wrong size ({len(bank_spec[0])}, expected {ranges[bank]}). Skipping bank {bank} with name {name}.")
            return

        # Save
        os.makedirs(os.path.join(path_to_save, str(bank)), exist_ok=True)
        np.save(os.path.join(path_to_save, str(bank), name), bank_spec)
        del bank_spec

    except Exception as e:
        log(f'Something weird happened with {name}, bank {bank}: {e}')

def splice_banks(items):
    name, locs = items
    
    data = get_info(locs[0])

    banks = {}
    for loc in locs:
        bank = loc.split('/')[4][-2]
        if bank not in banks:
            banks[bank] = []
        banks[bank].append(loc)
    banks_copy = {}
    for bank in banks.keys():
        new_bank = 3 - int(bank) #int(max(banks.keys(), key=int)) - int(bank)
        new_bank = str(new_bank)
        banks_copy[new_bank] = banks[bank]
    banks = banks_copy

    save_bank_partial = partial(save_bank, name=name)
    _ = list(map(save_bank_partial, banks.items()))
    del banks, banks_copy
    gc.collect()
    return name, data

# with Pool(maxtasksperchild=20) as p:
#     result = dict(tqdm(p.imap_unordered(splice_banks, groups.items()), total=len(groups)))

# # Save info dataframe
# df = pd.DataFrame.from_dict(result, orient='index')
# df.to_csv('/home/dataadmin/GBTData/SharedDataDirectory/spliced_cband_final_2/all_cband_info.csv', index=False)

def splice_banks_logged(items):
    name, locs = items
    pid = os.getpid()
    t0 = time.time()

    log(f"START pid={pid} name={name} nlocs={len(locs)}\n")

    try:
        out = splice_banks(items)
        log(f"DONE  pid={pid} name={name} dt={time.time() - t0:.1f}s\n")
        return out

    except Exception:
        with open(log_path, "a", buffering=1) as f:
            f.write(f"ERROR pid={pid} name={name}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        return name, {"error": "see splice_debug.log"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, help='batch size')
    parser.add_argument('nproc', type=int, help='number of CPUs')
    args = parser.parse_args()

    batch_size = args.batch_size
    nproc = args.nproc

    root_dir = "/home/dataadmin/GBTData/SharedDataDirectory/cband_052726"
    os.makedirs(root_dir, exist_ok=True)
    path_to_save = os.path.join(root_dir, 'data/preprocessed')
    log_path = os.path.join(root_dir, "splice_debug.log")

    rebinned_comb = np.load('/home/dataadmin/GBTData/SharedDataDirectory/rebinned_comb.npy')

    # Load groups dictionary
    groups_path = '/home/dataadmin/GBTData/SharedDataDirectory/cband_groups.json'
    with open(groups_path, 'r') as f:
        groups = json.load(f)

    items = list(groups.items())
    result = {}

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        log(f"batch {i // batch_size + 1}/{len(items) // batch_size + 1} start")

        with Pool(processes=nproc, maxtasksperchild=5) as p:
            for name, data in tqdm(
                p.imap_unordered(splice_banks_logged, batch, chunksize=1),
                total=len(batch),
                desc=f"batch {i // batch_size + 1}/{len(items) // batch_size + 1}"
            ):
                result[name] = data
        
        log(f"batch {i // batch_size + 1}/{len(items) // batch_size + 1} complete")

        pd.DataFrame.from_dict(result, orient="index").to_csv(
            os.path.join(root_dir, "all_cband_info_partial.csv"),
            index=False
        )
        gc.collect()

    df = pd.DataFrame.from_dict(result, orient="index")
    df.to_csv(os.path.join(root_dir, "all_cband_info.csv"), index=False)