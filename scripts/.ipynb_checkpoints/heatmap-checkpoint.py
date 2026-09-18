import numpy as np
import pandas as pd
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import trange

def plot_heatmap(name, vmin=0.99, vmax=1.01, n_bins=50):
    info = pd.read_csv(info_dir)
    info['cos_theta'] = [math.cos(math.radians(theta)) for theta in info['theta']]
    info['cos_phi'] = [math.cos(math.radians(phi)) for phi in info['phi']]
    bins = np.linspace(-1, 1, n_bins+1)
    info['cos_theta_bin'] = pd.cut(info['cos_theta'],  bins, labels=False)
    info['cos_phi_bin'] = pd.cut(info['cos_phi'], bins, labels=False)
    
    xs = np.load(os.path.join(data_dir, os.listdir(data_dir)[0]))[0] # might change later
    nans = np.full(xs.shape, np.nan)
    theta_map = []
    phi_map = []

    for i in trange(n_bins):
        thetas = info[info['cos_theta_bin'] == i]['source'].tolist()
        phis = info[info['cos_phi_bin'] == i]['source'].tolist()
        theta_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in os.listdir(data_dir) if file[:-4] in thetas])
        if theta_spectra.size:
            avg = np.mean(theta_spectra, axis=0)
            theta_map.append(avg)
        else:
            theta_map.append(nans)
        phi_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in os.listdir(data_dir) if file[:-4] in phis])
        if phi_spectra.size:
            avg = np.mean(phi_spectra, axis=0)
            phi_map.append(avg)
        else:
            phi_map.append(nans)
    
    fig, axs = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.01)

    norm = Normalize(vmin=vmin, vmax=vmax)
    extent = [np.min(xs), np.max(xs), -1, 1]

    im0 = axs[0].imshow(theta_map, extent=extent, origin='lower', norm=norm)
    im1 = axs[1].imshow(phi_map, extent=extent, origin='lower', norm=norm)
    
    axs[0].set_ylabel('$\mathrm{cos}({\\theta})$')
    axs[1].set_ylabel('$\mathrm{cos}({\phi})$')
    axs[1].set_xlabel('Frequency (MHz)')
    
    for ax in axs:
        freq_diff = xs[1] - xs[0]
        # Set ticks "intelligently"
        #labels = np.arange(8505, 8535, 5)
        #ax.set_xticks((labels - min(xs)) * (1 / freq_diff))
        ax.set_yticks([-1,0,1])
        #ax.set_yticklabels([-1,0,1])
        #ax.set_facecolor(mpl.colormaps['viridis'](0))
        ax.set_aspect(len(xs)/n_bins * 1.5)

    #tick_locs = ax[0].get_xticks
    axs[0].set_xticklabels([])
    #axs[1].set_xticklabels([round(loc * freqs + min(xs)) for loc in tick_locs])
    cbar = fig.colorbar(im0, ax=axs.ravel().tolist(), orientation='vertical', fraction=0.0335, pad=0.04, ticks=np.linspace(vmin, vmax, 5))
    cbar.set_label('Normalized Power')
    
    plt.savefig(f'/home/bhanselman/heatmaps/{name}.png', bbox_inches='tight', dpi=500)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('bank', help='bank index for which to produce heatmaps')
    parser.add_argument('name', help='name of the heatmap file')
    parser.add_argument('-i', '--injected', action='store_true', help='plot the injected data, default uninjected')
    parser.add_argument('--vmin', type=float, default=0.99, help='colorbar minimum, default 0.99')
    parser.add_argument('--vmax', type=float, default=1.01, help='colorbar maximum, default 1.01')
    parser.add_argument('--bins', type=int, default=50, help='number of cos theta/phi bins, default 50')
    args = parser.parse_args()
    
    #bank = args.bank
    name = args.name
    injected = args.injected
    
    # data_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/{bank}/'
    # data_dir += 'normalized_injected_w' if args.injected else 'normalized_uninjected_w'
    data_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/11080_test_0.015/normalized_uninjected'
    info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/all_xband_info.csv'
    
    plot_heatmap(name, vmin=vmin, vmax=vmax, n_bins=n_bins)