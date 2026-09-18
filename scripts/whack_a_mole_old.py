import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import itertools
import matplotlib as mpl
mpl.use('pgf')
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from astropy.visualization import ZScaleInterval
from scipy.signal import find_peaks
from copy import deepcopy

def _get_nan_slice_endpoints(arr):
    nan_mask = np.isnan(arr)
    padded_mask = np.concatenate(([False], nan_mask, [False]))
    boundaries = np.flatnonzero(padded_mask[1:] != padded_mask[:-1])
    start_indices = boundaries[::2]
    end_indices_exclusive = boundaries[1::2]
    if len(end_indices_exclusive):
        if end_indices_exclusive[-1] == len(nan_mask):
            end_indices_exclusive[-1] -= 1

    return list(zip(start_indices, end_indices_exclusive))

class WhackAMole:
    '''
    Class for analyzing candidate dark matter signals.
    '''

    def __init__(self, config, window_size=280, save=False):
        self.root = config['root_dir']
        os.makedirs(self.root, exist_ok=True)
        self.analysis_dir = config['analysis_dir']
        self.info_path = config['info_path']
        self.window_size = window_size
        self.save = save

        self._dopp_corr = np.load(os.path.join(self.analysis_dir, 'uninj_dopp_correlation.npy'))
        self._int_corr = np.load(os.path.join(self.analysis_dir, 'uninj_int_correlation.npy'))
        self.dopp_corr = deepcopy(self._dopp_corr)
        self.int_corr = deepcopy(self._int_corr)

        self.norm_data_dir = os.path.join(self.analysis_dir, 'normalized_uninjected')
        self.normalized_spectra = [np.load(os.path.join(self.norm_data_dir, file)) for file in os.listdir(self.norm_data_dir)]

        self.data_dir = config['raw_data_dir']
        self.raw_spectra = [np.load(os.path.join(self.data_dir, file)) for file in os.listdir(self.data_dir)]

    def set_window_size(self, window_size):
        '''
        Set rolling window size for p-value calculation.
        '''
        self.window_size = window_size
    
    def calc_combined_pvals(self, window_size=None):
        '''
        Calculate combined p-value spectrum from Doppler and intensity correlation spectra.
        '''
        if not window_size:
            window_size = self.window_size
        
        xs = self.dopp_corr[0]

        def calc_pvals(uninj):
            window_half_size = window_size // 2
            stds = []
            for i in range(len(uninj)):
                if i < window_half_size:
                    slice = uninj[:i+window_half_size]
                elif i >= len(uninj) - window_half_size:
                    slice = uninj[i-window_half_size:]
                else:
                    slice = uninj[i-window_half_size:i+window_half_size]
                
                if np.all(np.isnan(slice)):
                    stds.append(stds[-1])
                else:
                    stds.append(np.nanstd(slice))
        
            sigmas = uninj / stds
            pvals = list(map(lambda sigma: .5*math.erfc(sigma/np.sqrt(2)), sigmas))
            return pvals
        
        return xs, np.multiply(calc_pvals(self.dopp_corr[1]), calc_pvals(self.int_corr[1]))

    def plot_pval_spectrum(self, freq=None, sc=None, title=None, fig=None):
        '''
        Plot p-value spectrum.
        '''
        xs, pvals = self.calc_combined_pvals()
        if freq:
            mask = np.abs(freq - xs) < 100
            xs, pvals = xs[mask], pvals[mask]
        
        if fig is None:
            fig = Figure()
        ax = fig.subplots()
        
        i = 0
        while .5*math.erfc((i+1)/np.sqrt(2)) > min(pvals):
            label = r'$n\sigma$' if not i else None
            ax.axhline(.5*math.erfc((i+1)/np.sqrt(2)), c='red', ls='--', lw=0.5, label=label)
            i += 1
                
        ax.plot(xs, pvals, c='k')

        if sc is not None:
            label = 'hit' if len(sc) == 1 else 'hits'
            hit_pvals = [pvals[xs.tolist().index(x)] for x in sc]
            ax.scatter(sc, hit_pvals, c='blue', s=100, marker='x', label=label)
        
        ax.set_yscale('log')
        if freq:
            ax.set_xlim(freq-100, freq+100)
            ax.axvspan(freq-3, freq+3, color='gray', alpha=0.3, linewidth=0.5)

        for start, end in _get_nan_slice_endpoints(pvals):
            ax.axvspan(xs[start], xs[end], color='red', alpha=0.3, linewidth=0)

        min_pval = np.nanmin(pvals)
        ax.set_ylim(10 ** (math.floor(np.log10(min_pval))), 1)

        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        ax.set_ylabel('$p$', fontsize=18)
        if not title:
            title = 'P-Value Spectrum'
        ax.set_title(title, fontsize=20)
        ax.legend()

        if self.save:
            plt.savefig(os.path.join(self.root, f'{freq}_pval_spectrum'))

        return fig

    def plot_stacked_normalized_spectra(self, freq=None, fig=None):
        '''
        Plot stacked normalized spectra.
        '''
        xs = self.normalized_spectra[0][0]
        if freq:
            mask = np.abs(freq - xs) < 100

        if fig is None:
            fig = Figure()
        ax = fig.subplots()

        for spec in self.normalized_spectra:
            freqs, powers = spec
            if freq:
                freqs, powers = freqs[mask], powers[mask]
            powers = powers / np.median(powers)
            ax.plot(freqs, powers, c='k', alpha=0.3, rasterized=True)

        if freq:
            ax.set_xlim(freq-100, freq+100)

        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        ax.set_ylabel('Normalized Power', fontsize=18)
        ax.set_title('Stacked Normalized Spectra', fontsize=20)

        if self.save:
            plt.savefig(os.path.join(self.root, f'{freq}_stacked_normalized_spectra'))
        
        return fig

    def plot_stacked_raw_spectra(self, freq=None, fig=None):
        '''
        Plot stacked raw spectra.
        '''
        xs = self.raw_spectra[0][0]
        if freq:
            mask = np.abs(freq - xs) < 100

        if fig is None:
            fig = Figure()
        ax = fig.subplots()

        for spec in self.raw_spectra:
            freqs, powers = spec
            if freq:
                freqs, powers = freqs[mask], powers[mask]
            powers = powers / np.median(powers)
            ax.plot(freqs, powers, c='k', alpha=0.3, rasterized=True)

        if freq:
            ax.set_xlim(freq-100, freq+100)
        
        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        ax.set_ylabel('Median-Normalized Power', fontsize=18)
        ax.set_title('Stacked Raw Spectra', fontsize=20)

        if self.save:
            plt.savefig(os.path.join(self.root, f'{freq}_stacked_raw_spectra'))
        
        return fig

    def plot_heatmaps(self, freq=None, n_time_bins=50, n_bins=50, fig=None):
        info = pd.read_csv(self.info_path)

        info['time_mjd'] = [int(mjd.split('_')[0]) for mjd in info['time']]
        info['cos_theta'] = [math.cos(math.radians(theta)) for theta in info['theta']]
        info['cos_phi'] = [math.cos(math.radians(phi)) for phi in info['phi']]
        
        time_bins = np.linspace(min(info['time_mjd']), max(info['time_mjd']), n_time_bins+1)
        bins = np.linspace(-1, 1, n_bins+1)
        info['time_mjd_bin'] = pd.cut(info['time_mjd'], time_bins, labels=False)
        info['cos_theta_bin'] = pd.cut(info['cos_theta'], bins, labels=False)
        info['cos_phi_bin'] = pd.cut(info['cos_phi'], bins, labels=False)
        
        xs = self.normalized_spectra[0][0]
        if freq:
            mask = (xs >= freq-50) & (xs < freq+50)
        else:
            mask = np.full(xs.shape, True)

        nans = np.full(xs.shape, np.nan)[mask]

        def create_map(bin_column, n_bins_):
            for i in range(n_bins_):
                sources = info[info[bin_column] == i]['source'].tolist()
                spectra = np.array([np.load(os.path.join(self.norm_data_dir, file))[1][mask] for file in os.listdir(self.norm_data_dir) if file[:-4] in sources])
                if spectra.size:
                    avg = np.mean(spectra, axis=0)
                    yield avg
                else:
                    yield nans
        
        time_map = list(create_map('time_mjd_bin', n_time_bins))
        theta_map = list(create_map('cos_theta_bin', n_bins))
        phi_map = list(create_map('cos_phi_bin', n_bins))
        
        if fig is None:
            fig = Figure(figsize=(18, 18))
        axs = fig.subplots(3, 1)

        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(np.array(theta_map))
        vmax = max(vmax, 2 - vmin)
        ndigits = -int(math.floor(math.log10(vmax - 1))) + 1
        vmax = round(vmax, ndigits)
        vmin = 2 - vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
        extent = [np.min(xs[mask]), np.max(xs[mask]), -1, 1]
        time_extent = [np.min(xs[mask]), np.max(xs[mask]), min(info['time_mjd']), max(info['time_mjd'])]

        im0 = axs[0].imshow(theta_map, extent=extent, origin='lower', norm=norm, aspect='auto')
        im1 = axs[1].imshow(phi_map, extent=extent, origin='lower', norm=norm, aspect='auto')
        im_time = axs[2].imshow(time_map, extent=time_extent, origin='lower', norm=norm, aspect='auto')
        
        axs[0].set_ylabel(r'$\mathrm{cos}({\theta})$', fontsize=18)
        axs[1].set_ylabel(r'$\mathrm{cos}({\phi})$', fontsize=18)
        axs[1].set_ylabel(r'$\mathrm{cos}({\phi})$', fontsize=18)
        axs[2].set_ylabel('Modified JD', fontsize=18)
        axs[2].set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        fig.suptitle(f'Heatmaps', fontsize=20)
        
        for ax in [axs[0], axs[1]]:
            ax.set_xticklabels([])
            ax.set_yticks([-1,0,1])

        cbar = fig.colorbar(im0, ax=axs.ravel().tolist(), orientation='horizontal', fraction=0.0335, pad=0.04)
        cbar.set_label('Normalized Power', fontsize=18)
        
        return fig
    
    def get_hits(self, pval_threshold=None, distance=20):
        '''
        Get candidate hits below a specified p-value threshold.
        '''
        if pval_threshold is None:
            pval_threshold = .5*math.erfc(3/np.sqrt(2)) # 3-sigma threshold
        pval_threshold = -np.log10(pval_threshold)
        xs, pvals = self.calc_combined_pvals()
        log_pvals = -np.log10(np.nan_to_num(np.array(pvals), nan=1))
        hits, _ = find_peaks(log_pvals, height=pval_threshold, distance=distance)
        return xs[hits], pvals[hits]

    def _report(self, freq):
        '''
        Create report figure for one candidate signal.
        '''
        # Figure with pval spectrum, stacked normalized spectra, and stacked raw spectra in one column,
        # and heatmaps in the other column.
        fig = plt.figure(constrained_layout=True, figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig)        

        pval_subfig = fig.add_subfigure(gs[0, 0])
        stacked_norm_subfig = fig.add_subfigure(gs[1, 0])
        stacked_raw_subfig = fig.add_subfigure(gs[2, 0])
        heatmap_subfig = fig.add_subfigure(gs[:, 1])

        self.plot_pval_spectrum(freq=freq, fig=pval_subfig)
        self.plot_stacked_normalized_spectra(freq=freq, fig=stacked_norm_subfig)
        self.plot_stacked_raw_spectra(freq=freq, fig=stacked_raw_subfig)
        self.plot_heatmaps(freq=freq, fig=heatmap_subfig)

        fig.suptitle(f'Candidate Signal Report: {freq:.2f} MHz', fontsize=24)
        return fig

    def generate_report(self, freqs, save_dir):
        '''
        Generate report for candidate signal(s).
        '''
        if not isinstance(freqs, (list, np.ndarray)):
            freqs = [freqs]
        
        pdf = PdfPages(save_dir)

        for freq in freqs:
            fig = self._report(freq)
            pdf.savefig(fig)
            plt.close(fig)
        
        pdf.close()
        print(f'Report saved at {save_dir}')
    
    def exclude(self, freqs, width=20):
        if not isinstance(freqs, (list, np.ndarray)):
            freqs = [freqs]
        
        xs = self.dopp_corr[0]
        for freq in freqs:
            idx = np.argmin(np.abs(freq - xs))
            self.dopp_corr[1][idx-width:idx+width] = np.nan
            self.int_corr[1][idx-width:idx+width] = np.nan
    
    def run(self, save_path=None):
        '''
        Perform the entire whack-a-mole procedure;
        i.e., identify, plot and eliminate p-value hits until none are left.
        '''
        if save_path is None:
            save_path = 'whack_a_mole_results.pdf'
        save_dir = os.path.join(self.root, save_path)
        pdf = PdfPages(save_dir)
        
        i = 0
        while True:
            freqs, _ = self.get_hits()
            
            print(f'Round {i}')
            
            fig = self.plot_pval_spectrum(sc=freqs, title=None)#f'Round {i}: P-Value Spectrum')
            pdf.savefig(fig)
            plt.close(fig)

            if not len(freqs):
                break

            for freq in freqs:
                fig = self._report(freq)
                pdf.savefig(fig)
                plt.close(fig)
                
            self.exclude(freqs)
            i += 1
            if i >= 9:
                print('Maximum number of rounds reached.')
                break
        
        pdf.close()

if __name__ == '__main__':
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 0.5

    config = {
        'root_dir': '/home/dataadmin/GBTData/SharedDataDirectory',#'/home/bhanselman/bhanselman/images/whack_a_mole',
        'info_path': '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/all_xband_info.csv',
    }
    cases = ['decay', 'ann']
    banks = np.arange(3)
    for case, bank in itertools.product(cases, banks):
        print(f'Case: {case}, bank: {bank}')
        config['analysis_dir'] = os.path.join(
            '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/analyses/analysis_20251130', case, str(bank))
        config['raw_data_dir'] = os.path.join(
            '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed', str(bank))
        save_path = f'{case}_bank_{bank}.pdf'
        
        whack_a_mole = WhackAMole(config, window_size=800)
        whack_a_mole.run(save_path=save_path)
    # case = 'decay'
    # bank = 1
    # print(f'Case: {case}, bank: {bank}')
    # config['analysis_dir'] = os.path.join(
    #     '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/analyses/analysis_20251130', case, str(bank))
    # config['raw_data_dir'] = os.path.join(
    #     '/home/dataadmin/GBTData/SharedDataDirectory/xband_072625/data/preprocessed', str(bank))
    # save_path = f'{case}_bank_{bank}_aas.pdf'
    
    # whack_a_mole = WhackAMole(config, window_size=800)
    # whack_a_mole.run(save_path=save_path)
    
