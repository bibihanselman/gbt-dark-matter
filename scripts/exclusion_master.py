import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_interp_spline, PchipInterpolator, Akima1DInterpolator
from scipy.optimize import brentq
import logging
from collections import Counter

def get_f(freq):
    x = freq / 5.685e4
    return 1 / (math.exp(x) - 1)

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

    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16})
    
    fig, axs = plt.subplots(2, 1, figsize=(20,8), sharex=True)
    plt.subplots_adjust(hspace=0.15)
    
    # colors = ['blue', 'orange', 'green']
    
    # Get overlapping frequency ranges between banks
    def get_bank_freqs(bank):
        data_dir = os.path.join(analysis_dir, 'decay', str(bank))
        files = [file for file in os.listdir(data_dir) if condition(file)]
        files = sorted(files, key=key)
        freqs = list(map(key, files))
        return freqs

    def elements_in_at_least_two_lists(list_of_lists):
        all_unique_elements = []
        for sublist in list_of_lists:
            for element in set(sublist):
                all_unique_elements.append(element)
                
        counts = Counter(all_unique_elements)
        
        result = {element for element, count in counts.items() if count >= 2}
        
        return sorted(list(result))

    def get_overlap_endpoints(data):
        sublists = []
        current_sublist = [data[0]]

        for i in range(1, len(data)):
            if data[i] - data[i-1] > 10:
                sublists.append(current_sublist)
                current_sublist = [data[i]]
            else:
                current_sublist.append(data[i])
        
        sublists.append(current_sublist)
        return [(sublist[0], sublist[-1]) for sublist in sublists]

    bank_freqs = [get_bank_freqs(bank) for bank in range(3)]
    overlap = elements_in_at_least_two_lists(bank_freqs)
    endpoints = get_overlap_endpoints(overlap)

    for ax, case in zip(axs, ['decay', 'ann']):
        mins = []
        maxs = []

        for bank in range(3):
            freqs, limits = get_limits(case, bank)
            freq_diff = freqs[1] - freqs[0]
            
            mins.append(min([limit * 2/(1+get_f(freqs[0])) for limit in limits]))
            maxs.append(max([limit * 2 for limit in limits]))

            for i in range(len(freqs)):
                # 2-photon, 1-photon, 1-photon w/ Bose enhancement
                constants = [2, 1, 2/(1+get_f(freqs[i]))]
                colors = ['k', 'r', 'b']
                linestyles = ['-', '--', '-']
                labels = ['1-photon states', '2-photon states', '1-photon states with stimulated emission']
                
                if freqs[i] not in overlap:
                    for constant, color, ls, label in zip(constants, colors, linestyles, labels):
                        ax.plot([freqs[i], freqs[i]+freq_diff], [constant*limits[i], constant*limits[i]], ls=ls, c=color, label=label)
                        if i == 0 or freqs[i-1] in overlap:
                            ax.plot([freqs[i], freqs[i]], [1, constant*limits[i]], ls=ls, c=color)
                        if i == len(freqs) - 1 or freqs[i+1] in overlap:
                            ax.plot([freqs[i]+freq_diff, freqs[i]+freq_diff], [constant*limits[i], 1], ls=ls, c=color)
                        else:
                            ax.plot([freqs[i]+freq_diff, freqs[i]+freq_diff], [constant*limits[i], constant*limits[i+1]], ls=ls, c=color)

        for lst in endpoints:
            ax.axvspan(lst[0], lst[1]+freq_diff, color='red', alpha=0.3, linewidth=0)
        
        label = r'$\lambda$ [s$^{-1}$]' if case == 'decay' else r'$\langle \sigma v \rangle$ [cm$^3$s$^{-1}$]'
        ax.set_ylabel(label, fontsize=22)
        if case == 'ann':
            ax.set_xlabel(r'$\nu$ [MHz]', fontsize=22)
        ax.set_yscale('log')
        
        # Manual exclusion of the bandpass edges
        ax.set_xlim(8050, 11050)

        log_min, log_max = np.log10(min(mins)), np.log10(max(maxs))
        plot_min, plot_max = 10 ** math.floor(log_min), 10 ** math.ceil(log_max)
        ax.set_ylim(plot_min, plot_max)

        ax.grid()
        import matplotlib.ticker as tck
        ax.xaxis.set_minor_locator(tck.MultipleLocator(100))

    handles, labels = axs[0].get_legend_handles_labels()
    unique_handles, unique_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
        
    axs[1].legend(unique_handles, unique_labels, fontsize=14, loc='lower right')

    plt.savefig(f'/home/bhanselman/bhanselman/images/{name}_exclusion_master_aas_v2.png', bbox_inches='tight', dpi=500)