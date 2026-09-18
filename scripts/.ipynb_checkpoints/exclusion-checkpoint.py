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

def get_limit(pval_array):
    sizes, vals = pval_array
    
    three_sigma = np.log10(.5*math.erfc(3/np.sqrt(2)))

    above = np.array([sizes[vals > three_sigma], vals[vals > three_sigma]])
    below = np.array([sizes[vals < three_sigma], vals[vals < three_sigma]])

    above_sort_indices = np.argsort(above[0])[::-1]
    above = above[:, above_sort_indices]
    below_sort_indices = np.argsort(below[0])[::-1]
    below = below[:, below_sort_indices]

    #print(above, below)

    if above.size > 0 and below.size > 0:
        ys = [below[0,-1], above[0,0]]
        xs = [below[1,-1], above[1,0]]

        print(xs, ys)

        return np.interp(three_sigma, xs, ys)

    elif above.size > 0:
        above_pts = above[:,:3]
        fit = np.polyfit(above_pts[1], above_pts[0], 1)

        return np.poly1d(fit)(three_sigma)

    elif below.size > 0:
        below_pts = below[:,-3:]
        fit = np.polyfit(below_pts[1], below_pts[0], 1)

        return np.poly1d(fit)(three_sigma)


if __name__ == '__main__':
    analysis_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/analyses/8000_8500_mhz/decay'
    pval_arrays = [np.load(os.path.join(analysis_dir, file)) for file in os.listdir(analysis_dir) if 'pvals' in file]
    freqs = [int(file.split('_')[0]) for file in os.listdir(analysis_dir) if 'pvals' in file]
    
    exclusion = [get_limit(pval_array) for pval_array in pval_arrays]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(15,4))

    for i in range(len(freqs)):
        ax.plot([freqs[i], freqs[i+1]], [decay_limits[i], decay_limits[i]], c='k')
        if i != len(freqs) - 1:
            ax.plot([freqs[i+1], freqs[i+1]], [decay_limits[i], decay_limits[i+1]], c='k')

    ax.set_ylabel(r'log($\lambda$ [s$^{-1}$])', fontsize=18)
    ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
    
    freq_diff = freqs[1] - freqs[0]
    ax.set_xlim(freqs[0], freqs[-1]+freq_diff)

    plt.savefig('/home/bhanselman/images/second_analysis_exclusion_decay.png', bbox_inches='tight', dpi=500)