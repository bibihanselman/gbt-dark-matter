import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from tqdm import tqdm
from astropy.visualization import ZScaleInterval

info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/all_xband_info.csv'

def round_up_to_nearest_10(number):
    return math.ceil(number / 10) * 10

def convert_to_seconds(l):
    mjd, seconds = l.split('_')
    mjd, seconds = int(mjd), int(seconds)
    return mjd * 86400 + seconds

def plot_heatmap(case, bank, data_dir, xs, start_freq=8000, width=100, n_bins=50, time=False):
    mask = (xs >= start_freq) & (xs < start_freq + width)
    nans = np.full(xs.shape, np.nan)[mask]
    end_freq = xs[mask][-1]

    def create_map(bin_column):
        for i in range(n_bins):
            sources = info[info[bin_column] == i]['source'].tolist()
            spectra = np.array([np.load(os.path.join(data_dir, file))[1][mask] for file in os.listdir(data_dir) if file[:-4] in sources])
            if spectra.size:
                avg = np.mean(spectra, axis=0)
                yield avg
            else:
                yield nans
    
    if time:
        time_map = list(create_map('time_mjd_bin'))

        fig, ax = plt.subplots(figsize=(12,6))

        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(np.array(time_map))
        vmax = max(vmax, 2 - vmin)
        ndigits = -int(math.floor(math.log10(vmax - 1))) + 1
        vmax = round(vmax, ndigits)
        vmin = 2 - vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
        extent = [np.min(xs[mask]), np.max(xs[mask]), min(info['time_mjd']), max(info['time_mjd'])]

        im = ax.imshow(time_map, extent=extent, origin='lower', norm=norm, aspect='auto')
        
        ax.set_ylabel('Modified JD', fontsize=18)
        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        fig.suptitle(f'Bank {bank} {case}: {round(start_freq)}-{round(end_freq)} MHz', fontsize=20)

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.0335, pad=0.04, ticks=np.linspace(vmin, vmax, 5))
        cbar.set_label('Normalized Power', fontsize=18)
        
    else:
        theta_map = list(create_map('cos_theta_bin'))
        phi_map = list(create_map('cos_phi_bin'))
    
        fig, axs = plt.subplots(2, 1, figsize=(12,6))
        plt.subplots_adjust(hspace=0.15)

        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(np.array(theta_map))
        vmax = max(vmax, 2 - vmin)
        ndigits = -int(math.floor(math.log10(vmax - 1))) + 1
        vmax = round(vmax, ndigits)
        vmin = 2 - vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
        extent = [np.min(xs[mask]), np.max(xs[mask]), -1, 1]

        im0 = axs[0].imshow(theta_map, extent=extent, origin='lower', norm=norm, aspect='auto')
        im1 = axs[1].imshow(phi_map, extent=extent, origin='lower', norm=norm, aspect='auto')
        
        axs[0].set_ylabel(r'$\mathrm{cos}({\theta})$', fontsize=18)
        axs[1].set_ylabel(r'$\mathrm{cos}({\phi})$', fontsize=18)
        axs[1].set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        fig.suptitle(f'Bank {bank} {case}: {round(start_freq)}-{round(end_freq)} MHz', fontsize=20)
        
        for ax in axs:
            ax.set_yticks([-1,0,1])
            ax.set_aspect('auto')

        #tick_locs = ax[0].get_xticks
        axs[0].set_xticklabels([])
        #axs[1].set_xticklabels([round(loc * freqs + min(xs)) for loc in tick_locs])
        cbar = fig.colorbar(im0, ax=axs.ravel().tolist(), orientation='vertical', fraction=0.0335, pad=0.04, ticks=np.linspace(vmin, vmax, 5))
        cbar.set_label('Normalized Power', fontsize=18)
    
    return fig

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('bank', help='bank index for which to produce heatmaps')
    parser.add_argument('--name', help='name of the analysis to plot')
    parser.add_argument('--bins', type=int, default=50, help='number of cos theta/phi bins, default 50')
    parser.add_argument('-t', '--time', action='store_true', help='sort by time instead of theta/phi')
    args = parser.parse_args()
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    name = args.name
    n_bins = args.bins
    time = args.time
    
    analysis_dir = os.path.join('/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/analyses', name)
    
    info = pd.read_csv(info_dir)
    info['time_mjd'] = [int(mjd.split('_')[0]) for mjd in info['time']]
    info['cos_theta'] = [math.cos(math.radians(theta)) for theta in info['theta']]
    info['cos_phi'] = [math.cos(math.radians(phi)) for phi in info['phi']]
    
    bins = np.linspace(-1, 1, n_bins+1)
    time_bins = np.linspace(min(info['time_mjd']), max(info['time_mjd']), n_bins+1)
    info['time_mjd_bin'] = pd.cut(info['time_mjd'], time_bins, labels=False)
    info['cos_theta_bin'] = pd.cut(info['cos_theta'], bins, labels=False)
    info['cos_phi_bin'] = pd.cut(info['cos_phi'], bins, labels=False)
    
    cases = ['decay', 'ann']
    banks = np.arange(3)

    #info = info[info['time'].apply(lambda t: convert_to_seconds(t) >= 5.18e9)]

    for case in cases:
        pdf = PdfPages(f'/home/bhanselman/bhanselman/heatmaps/analysis_20251130/time/{name}_{case}_time_heatmaps.pdf')
        
        for bank in banks:
            data_dir = os.path.join(analysis_dir, case, str(bank), 'normalized_uninjected')
            xs = np.load(os.path.join(data_dir, os.listdir(data_dir)[0]))[0]
            width = 100
            start_freqs = np.arange(round_up_to_nearest_10(xs[0]), xs[-1], width)

            for start_freq in tqdm(start_freqs, desc=f'Bank {bank} {case}'):
                fig = plot_heatmap(case, bank, data_dir, xs, start_freq=start_freq, width=width, n_bins=n_bins, time=time)
                pdf.savefig(fig, bbox_inches='tight', dpi=500)
                plt.close(fig)

        pdf.close()