import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm.auto import tqdm
from astropy.visualization import ZScaleInterval
from normalization_methods import *
import multiprocessing as mp
import argparse
import warnings
warnings.filterwarnings("ignore")

def round_up_to_nearest_10(number):
    return math.ceil(number / 10) * 10

def convert_to_seconds(l):
    mjd, seconds = l.split('_')
    mjd, seconds = int(mjd), int(seconds)
    return mjd * 86400 + seconds

def plot_heatmap(case, bank, all_spectra, xs, info, start_freq=8000, width=100, n_bins=50, time=False, resid=None):
    mask = (xs >= start_freq) & (xs < start_freq + width)
    nans = np.full(xs.shape, np.nan)[mask]
    end_freq = xs[mask][-1]

    info['time_mjd'] = info['time'].apply(lambda mjd: int(mjd.split('_')[0])) #[int(mjd.split('_')[0]) for mjd in info['time']]
    info['cos_theta'] = info['theta'].apply(lambda theta: math.cos(math.radians(theta))) #[math.cos(math.radians(theta)) for theta in info['theta']]
    info['cos_phi'] = info['phi'].apply(lambda phi: math.cos(math.radians(phi))) #[math.cos(math.radians(phi)) for phi in info['phi']]

    var_list = ['time_mjd'] if time else ['cos_theta', 'cos_phi']

    fig = plt.figure(figsize=(12, 6), layout="tight")
    gs = fig.add_gridspec(len(var_list), 2, wspace=0, hspace=0.2, width_ratios=[3, 1])
    axs = gs.subplots()

    def create_map(bin_column):
        for i in range(n_bins):
            bin_mask = (info[bin_column] == i).to_numpy()
            spectra = all_spectra[bin_mask][:, mask]
            if spectra.size:
                avg = np.mean(spectra, axis=0)
                yield avg
            else:
                yield nans

    for i, var in enumerate(var_list):
        bounds = (min(info['time_mjd']), max(info['time_mjd'])) if var == 'time_mjd' else (-1, 1)
        bins = np.linspace(*bounds, n_bins+1)
        info[f'{var}_bin'] = pd.cut(info[var], bins, labels=False)
            
        heat_map = np.array(list(create_map(f'{var}_bin')))
        if resid is not None:
            heat_map = heat_map - resid[mask]

        ax = axs[i, 0]
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(np.array(heat_map))
        if resid is not None:
            vmax = max(vmax, -vmin)
            ndigits = -int(math.floor(math.log10(vmax))) + 1
            vmax = round(vmax, ndigits)
            vmin = -vmax
        else:
            vmax = max(vmax, 2-vmin)
            ndigits = -int(math.floor(math.log10(vmax - 1))) + 1
            vmax = round(vmax, ndigits)
            vmin = 2 - vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
        extent = [np.min(xs[mask]), np.max(xs[mask]), *bounds]

        #im0 = axs[0].imshow(theta_map, extent=extent, origin='lower', norm=norm, aspect='auto')
        im1 = ax.imshow(heat_map, extent=extent, cmap='RdBu' if resid is not None else 'viridis', origin='lower', norm=norm, aspect='auto', interpolation='none')
        
        #axs[0].set_ylabel(r'$\mathrm{cos}({\theta})$', fontsize=18)
        if var == 'cos_phi':
            label = r'$\mathrm{cos}({\phi})$'
        elif var == 'cos_theta':
            label = r'$\mathrm{cos}({\theta})$'
        else:
            label = 'Modified JD'

        ax.set_ylabel(label, fontsize=18)
        # for i in bins:
        #     ax.axhline(i, c='r')

        hist_ax = axs[i, 1]
        hist_ax.hist(info[var], bins=bins, facecolor='lightgray', edgecolor='k', orientation='horizontal')
        hist_ax.set_yticks([])
        hist_ax.set_ylim(*bounds)

        if var != 'time_mjd':
            ax.axhline(0, c='r') # Inward/outward division
            hist_ax.axhline(0, c='r') # Inward/outward division
            ax.set_yticks(np.linspace(-1, 1, 5))

    fig.suptitle(f'Bank {bank} {case}: {start_freq:.0f}-{end_freq:.0f} MHz', fontsize=20)
    #ax.set_aspect('auto')

    axs[-1, 0].set_xlabel(r'$\nu$ [MHz]', fontsize=18)
    axs[-1, 1].set_xlabel('Counts', fontsize=18)

    cax = inset_axes(
        hist_ax,
        width="15%",
        height=f"{100 * len(var_list) + 20 * (len(var_list)-1)}%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=hist_ax.transAxes,
        borderpad=0
    )
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar.set_label('Normalized Power Residual' if resid is not None else 'Normalized Power', fontsize=18)
    
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of heatmap directory')
    #parser.add_argument('prepath', help='preprocessed spectrum directory')
    parser.add_argument('normpath', help='normalized spectrum directory')
    parser.add_argument('--run', default='aya', help='name of population optimization run')
    parser.add_argument('--bins', type=int, default=50, help='number of cos theta/phi bins, default 50')
    parser.add_argument('--width', type=int, default=100, help='frequency interval in MHz, default 100')
    #parser.add_argument('--nproc', type=int, default=0, help='number of processes for normalization multiprocessing, default cpu count')
    parser.add_argument('-t', '--time', action='store_true', help='sort by time instead of theta/phi')
    args = parser.parse_args()
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    name = args.name
    data_dir = args.normpath
    run = args.run
    n_bins = args.bins
    width = args.width
    time = args.time
    # nproc = args.nproc
    # if not nproc:
    #     nproc = mp.cpu_count()
    
    info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/all_cband_info.csv'
    info = pd.read_csv(info_dir)
    
    cases = ['decay', 'ann']
    banks = np.arange(4)

    for case in cases:
        is_decay = case == 'decay'
        print(f'{case} start')

        pdf = PdfPages(f'/home/bhanselman/bhanselman/heatmaps/{name}_{case}.pdf')
        
        for bank in banks:
            # Normalize the preprocessed data
            # bank_dir = os.path.join(data_dir, str(bank))
            # save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/heatmap/{name}/{case}/{bank}'

            # if os.path.exists(save_dir):
            #     print(f'Bank {bank} already normalized. Skipping!')
            # else:
            #     with mp.Pool(processes=nproc) as p:
            #         normalize_partial = partial(
            #             normalize_weighted,
            #             data_dir=bank_dir,
            #             save_dir=save_dir,
            #             is_decay=is_decay
            #         )
            #         _ = list(tqdm(
            #             p.imap_unordered(normalize_partial, os.listdir(bank_dir)), 
            #             total=len(os.listdir(bank_dir)),
            #             desc=f'Normalizing bank {bank}'
            #         ))

            # xs = np.load(os.path.join(save_dir, os.listdir(save_dir)[0]))[0]   
            # start_freqs = np.arange(round_up_to_nearest_10(xs[0]), xs[-1], width)

            bank_dir = os.path.join(data_dir, case, str(bank))
            bank_info = info[info['source'].apply(lambda l: l+'.npy').isin(os.listdir(bank_dir))]
            files = bank_info['source'].apply(lambda l: l+'.npy').to_numpy()

            if run != 'aya':
                state = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/target_selection/{run}/best_state.npy')
                bank_info = bank_info[state == 1]

            xs = np.load(os.path.join(bank_dir, os.listdir(bank_dir)[0]))[0]    
            start_freqs = np.arange(round_up_to_nearest_10(xs[0]), xs[-1], 100)

            all_spectra = np.array([np.load(os.path.join(bank_dir, file))[1] for file in files]) 

            for start_freq in tqdm(start_freqs, desc=f'Bank {bank} {case}'):
                fig = plot_heatmap(
                    case,
                    bank,
                    all_spectra,
                    xs,
                    bank_info,
                    start_freq=start_freq,
                    width=width,
                    n_bins=n_bins,
                    time=time,
                    resid=np.nanmean(all_spectra, axis=0)
                )
                pdf.savefig(fig, bbox_inches='tight', dpi=72)
                plt.close(fig)

        pdf.close()