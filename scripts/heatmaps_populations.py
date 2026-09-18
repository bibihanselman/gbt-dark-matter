import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from tqdm.auto import tqdm
from astropy.visualization import ZScaleInterval
from normalization_methods import *
import argparse
import warnings
warnings.filterwarnings("ignore")

def round_up_to_nearest_10(number):
    return math.ceil(number / 10) * 10

def plot_heatmap(case, bank, data_dir, xs, info, start_freq=8000, width=100, n_bins=50, var='cos_phi'):
    mask = (xs >= start_freq) & (xs < start_freq + width)
    nans = np.full(xs.shape, np.nan)[mask]
    end_freq = xs[mask][-1]

    info['time_mjd'] = info['time'].apply(lambda mjd: int(mjd.split('_')[0])) #[int(mjd.split('_')[0]) for mjd in info['time']]
    info['cos_theta'] = info['theta'].apply(lambda theta: math.cos(math.radians(theta))) #[math.cos(math.radians(theta)) for theta in info['theta']]
    info['cos_phi'] = info['phi'].apply(lambda phi: math.cos(math.radians(phi))) #[math.cos(math.radians(phi)) for phi in info['phi']]

    bounds = (min(info['time_mjd']), max(info['time_mjd'])) if var == 'time_mjd' else (-1, 1)
    bins = np.linspace(*bounds, n_bins+1)
    info['time_mjd_bin'] = pd.cut(info['time_mjd'], bins, labels=False)
    info['cos_theta_bin'] = pd.cut(info['cos_theta'], bins, labels=False)
    info['cos_phi_bin'] = pd.cut(info['cos_phi'], bins, labels=False)

    def create_map(bin_column):
        for i in range(n_bins):
            sources = info[info[bin_column] == i]['source'].tolist()
            spectra = np.array([np.load(os.path.join(data_dir, file))[1][mask] for file in os.listdir(data_dir) if file[:-4] in sources])
            if spectra.size:
                avg = np.mean(spectra, axis=0)
                yield avg
            else:
                yield nans
        
    heat_map = list(create_map(f'{var}_bin'))

    fig = plt.figure(figsize=(12, 6), layout="tight")
    gs = fig.add_gridspec(1, 2, wspace=0, width_ratios=[3, 1])
    #plt.subplots_adjust(hspace=0.15)

    ax = fig.add_subplot(gs[0])
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(np.array(heat_map))
    vmax = max(vmax, 2 - vmin)
    ndigits = -int(math.floor(math.log10(vmax - 1))) + 1
    vmax = round(vmax, ndigits)
    vmin = 2 - vmax
    norm = Normalize(vmin=vmin, vmax=vmax)
    extent = [np.min(xs[mask]), np.max(xs[mask]), *bounds]

    #im0 = axs[0].imshow(theta_map, extent=extent, origin='lower', norm=norm, aspect='auto')
    im1 = ax.imshow(heat_map, extent=extent, origin='lower', norm=norm, aspect='auto')
    
    #axs[0].set_ylabel(r'$\mathrm{cos}({\theta})$', fontsize=18)
    if var == 'cos_phi':
        label = r'$\mathrm{cos}({\phi})$'
    elif var == 'cos_theta':
        label = r'$\mathrm{cos}({\theta})$'
    else:
        label = 'Modified JD'

    ax.set_ylabel(label, fontsize=18)
    ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
    # for i in bins:
    #     ax.axhline(i, c='r')

    hist_ax = fig.add_subplot(gs[1])
    hist_ax.hist(info[var], bins=bins, facecolor='lightgray', edgecolor='k', orientation='horizontal')
    hist_ax.set_xlabel('Counts', fontsize=18)
    hist_ax.set_yticks([])
    hist_ax.set_ylim(*bounds)

    if var != 'time_mjd':
        ax.axhline(0, c='r') # Inward/outward division
        hist_ax.axhline(0, c='r') # Inward/outward division
        ax.set_yticks(np.linspace(-1, 1, 9))

    fig.suptitle(f'Bank {bank} {case}: {start_freq:.0f}-{end_freq:.0f} MHz', fontsize=20)
    ax.set_aspect('auto')

    #tick_locs = ax[0].get_xticks
    #axs[0].set_xticklabels([])
    #axs[1].set_xticklabels([round(loc * freqs + min(xs)) for loc in tick_locs])
    cax = fig.add_axes([1, 0.117, 0.025, 0.778])
    cbar = fig.colorbar(im1, cax=cax, ax=ax, orientation='vertical', ticks=np.linspace(vmin, vmax, 5))
    cbar.set_label('Normalized Power', fontsize=18)
    
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='aya', help='run name')
    parser.add_argument('--bins', type=int, default=50, help='number of cos theta/phi bins, default 50')
    parser.add_argument('--width', type=int, default=100, help='frequency interval in MHz, default 100')
    args = parser.parse_args()
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    run_name = args.run
    n_bins = args.bins
    width = args.width

    run_name_dir = run_name.replace('_', '/')

    info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/all_cband_info.csv'

    info = pd.read_csv(info_dir)
    bank_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/heatmap/052926_01/decay/1'
    bank_1_info = info[info['source'].apply(lambda l: l+'.npy').isin(os.listdir(bank_dir))]

    if run_name == 'aya':
        info = bank_1_info[~bank_1_info['phi'].between(70, 115)]
    else:
        state = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/{run_name_dir}/best_state.npy')
        info = bank_1_info[state == 1]

    xs = np.load(os.path.join(bank_dir, os.listdir(bank_dir)[0]))[0]    
    start_freqs = np.arange(round_up_to_nearest_10(xs[0])+20, xs[-1], 100)

    for var in ['cos_theta', 'time_mjd']:
        os.makedirs(f'/home/bhanselman/bhanselman/heatmaps/{run_name}', exist_ok=True)
        pdf = PdfPages(f'/home/bhanselman/bhanselman/heatmaps/{run_name}/{var}.pdf')

        for start_freq in tqdm(start_freqs, desc=var):
            fig = plot_heatmap(
                'decay',
                1,
                bank_dir,
                xs,
                info,
                start_freq=start_freq,
                width=width,
                n_bins=n_bins,
                var=var
            )
            pdf.savefig(fig, bbox_inches='tight', dpi=500)
            plt.close(fig)

        pdf.close()
