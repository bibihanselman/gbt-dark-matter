import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_interp_spline, PchipInterpolator, Akima1DInterpolator
from scipy.optimize import brentq
import logging

def get_limit(freq, pval_array, interp_type='spline'):
    three_sigma = np.log10(.5*math.erfc(3/np.sqrt(2)))

    sizes, vals = pval_array
    # Throw out zero p-vals
    mask = vals != 0
    sizes, vals = sizes[mask], vals[mask]
    # Convert to log
    sizes, vals = np.log10(sizes), np.log10(vals)
    sort_indices = np.argsort(sizes)
    sizes, vals = sizes[sort_indices], vals[sort_indices]
    #print(sizes, vals)
    
    if interp_type == 'spline':
        spline = make_interp_spline(sizes, vals, k=3)
        xs = np.linspace(min(sizes), max(sizes), 1000)
        fit = interp1d(xs, spline(xs), bounds_error=False, fill_value='extrapolate')
    elif interp_type == 'pchip':
        fit = PchipInterpolator(sizes, vals, extrapolate=True)
    elif interp_type == 'akima':
        fit = Akima1DInterpolator(sizes, vals, extrapolate=True)
    elif interp_type == 'makima':
        fit = Akima1DInterpolator(sizes, vals, method='makima', extrapolate=True)
        
    def target(x):
        return fit(x) - three_sigma
    
    try:
        limit = brentq(target, min(sizes), max(sizes))
    except:
        logging.warning(f'Root could not be found in the search interval, freq={freq}, pvals={pval_array}')
        limit = sizes[np.argmin(np.abs(vals - three_sigma))]
    
    return 10 ** limit

def condition(file):
    return 'pvals' in file and 'uninj' not in file

def key(file):
    return float(file.split('_')[0])

def get_limits(case, bank):
    data_dir = os.path.join(analysis_dir, case, str(bank))
    files = [file for file in os.listdir(data_dir) if condition(file)]
    files = sorted(files, key=key)
    pval_arrays = [np.load(os.path.join(data_dir, file)) for file in files]
    freqs = list(map(key, files))
    limits = [get_limit(freq, pval_array) for freq, pval_array in zip(freqs, pval_arrays)]

    return freqs, limits

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of the analysis to plot')
    args = parser.parse_args()
    name = args.name

    analysis_dir = os.path.join('/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/analyses', name)

    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})
    
    for bank in range(3):
        fig, axs = plt.subplots(2, 1, figsize=(15,8), sharex=True)
        plt.subplots_adjust(hspace=0.15)
        
        freqs, decay_limits = get_limits('decay', bank)
        freqs, ann_limits = get_limits('ann', bank)

        flags = [(decay_limits[i] == 1 or ann_limits[i] == 1) for i in range(len(freqs))]
        freq_diff = freqs[1] - freqs[0]

        for ax, case, limits in zip(axs, ['decay', 'ann'], [decay_limits, ann_limits]):
            
            for i in range(len(freqs)):
                if flags[i]:
                    ax.axvspan(freqs[i], freqs[i]+freq_diff, color='red', alpha=.3)
                else:
                    ax.plot([freqs[i], freqs[i]+freq_diff], [limits[i], limits[i]], c='k')
                if i != len(freqs) - 1:
                    ax.plot([freqs[i+1], freqs[i+1]], [limits[i], limits[i+1]], c='k')


            label = r'$\lambda$ [s$^{-1}$]' if case == 'decay' else r'$\langle \sigma v \rangle$ [cm$^3$s$^{-1}$]'
            ax.set_ylabel(label, fontsize=18)
            if case == 'ann':
                ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
            ax.set_yscale('log')
            
            ax.set_xlim(freqs[0], freqs[-1]+freq_diff)
            limits_filt = [limit for flag, limit in zip(flags, limits) if not flag]
            log_min, log_max = np.log10(min(limits_filt)), np.log10(max(limits_filt))
            plot_min, plot_max = 10 ** math.floor(log_min), 10 ** math.ceil(log_max)
            ax.set_ylim(plot_min, plot_max)

            ax.grid()

        plt.savefig(f'/home/bhanselman/bhanselman/images/{name}_exclusion_bank_{bank}.png', bbox_inches='tight', dpi=500)