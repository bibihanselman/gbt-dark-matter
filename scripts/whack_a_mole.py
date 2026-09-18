import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import itertools
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from astropy.visualization import ZScaleInterval
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, make_interp_spline
from scipy.stats import combine_pvalues
from copy import deepcopy

def _get_nan_slice_endpoints(arr):
    nan_mask = np.isnan(arr)
    padded_mask = np.concatenate(([False], nan_mask, [False]))
    diffs = np.diff(padded_mask.astype(int))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    if len(ends):
        if ends[-1] == len(nan_mask):
            ends[-1] -= 1
    return list(zip(starts, ends))

class WhackAMole:
    '''
    Class for analyzing candidate dark matter signals.
    '''

    def __init__(self, config, window_size=280, save=False):
        plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

        self.freqs = config['freqs']
        self.uninj_norm_spectra = config['uninj_norm_spectra']
        self.uninj_spectra = config['uninj_spectra']
        self.template_spectra = config['template_spectra']
        self.ref_freq = config['template_freq']

        self.state_dopp = config['state_dopp']
        self.state_int = config['state_int']
        
        self.info = config['info']

        self.asymmetry_int = self.build_asymmetry(self.uninj_norm_spectra, self.state_int, self.info['phi'] < 90)
        template_int = self.build_asymmetry(self.template_spectra, self.state_int, self.info['phi'] < 90)
        template_xs = np.arange(len(template_int))
        self.template_int = make_interp_spline(template_xs[::15], template_int[::15], k=3)(template_xs)

        self.asymmetry_dopp = self.build_asymmetry(self.uninj_norm_spectra, self.state_dopp, self.info['theta'] < 90)
        template_dopp = self.build_asymmetry(self.template_spectra, self.state_dopp, self.info['theta'] < 90)
        template_xs = np.arange(len(template_dopp))
        self.template_dopp = make_interp_spline(template_xs[::15], template_dopp[::15], k=3)(template_xs)

        self.window_size = window_size
        self.save = save

    def set_window_size(self, window_size):
        '''
        Set rolling window size for p-value calculation.
        '''
        self.window_size = window_size
    
    def build_asymmetry(self, specs, state, mask): #template=False
        '''Form asymmetry from a given state.'''
        inner = (state == 1) & mask
        outer = (state == 1) & ~mask

        inner_slice = specs[inner, :]
        outer_slice = specs[outer, :]
        inner_mean = np.zeros(specs.shape[1])
        outer_mean = np.zeros(specs.shape[1])
        
        for j in range(specs.shape[1]):
            inner_mean[j] = inner_slice[:, j].mean()
            outer_mean[j] = outer_slice[:, j].mean()

        asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)
        
        return asymmetry

    def correlate_asymmetry(self, spectrum, template):
        def stretch_template(template, stretch_factor):
            if template.shape[0] % 2 == 0:
                template = template[1:] # Make the number of points odd
            k = template.shape[0] // 2
            orig_x = np.arange(-k, k+1)
            interp_func = interp1d(orig_x, template, bounds_error=False, fill_value=0)
            new_k = ((2*k + 1)/stretch_factor - 1)/2 # 2k_1 + 1 = sf*(2k_0 + 1)
            new_x = np.linspace(-new_k, new_k, template.shape[0])

            return interp_func(new_x)

        def get_a(target, template):
            return np.dot(target, template) / (np.dot(template, template) + 1e-12)
        
        a_vals = np.zeros_like(spectrum)
        residuals = np.zeros_like(spectrum)
        lower = (len(template) - 1) // 2
        upper = (len(template) + 1) // 2
        for i in range(len(spectrum)):
            if math.isnan(spectrum[i]):
                # Set correlation to nan whenever the asymmetry is nan
                a_vals[i] = np.nan
                residuals[i] = 1
            else:
                stretched = stretch_template(template, self.freqs[i] / self.ref_freq)
                
                if i < lower:
                    x = lower - i
                    stretched = stretched[x:]
                    window = spectrum[:i+upper]
                elif i > (len(spectrum) - upper):
                    x = len(spectrum) - upper - i
                    stretched = stretched[:x]
                    window = spectrum[i-lower:]
                else:
                    window = spectrum[i-lower:i+upper]
                
                nan_mask = np.isfinite(window)
                a = get_a(window[nan_mask], stretched[nan_mask]) 
                a_vals[i] = a 
                residuals[i] = np.sqrt(np.nanmean((a*stretched-window)**2)) 

        return a_vals / (residuals + 1e-12)
        
    def calc_combined_pvals(self, window_size=None):
        '''
        Calculate combined p-value spectrum from Doppler and intensity asymmetry spectra.
        '''
        if not window_size:
            window_size = self.window_size
        
        corr_dopp = self.correlate_asymmetry(self.asymmetry_dopp, self.template_dopp)
        corr_int = self.correlate_asymmetry(self.asymmetry_int, self.template_int)
        
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
        
        return np.array([
            combine_pvalues([pval_dopp, pval_int])[1]
            for pval_dopp, pval_int in zip(calc_pvals(corr_dopp), calc_pvals(corr_int))
        ])

    def plot_pval_spectrum(self, freq=None, sc=None, title=None, fig=None):
        '''
        Plot p-value spectrum.
        '''
        pvals = self.calc_combined_pvals()
        if freq:
            mask = np.abs(freq - self.freqs) < 100
            xs, pvals = self.freqs[mask], pvals[mask]
        else:
            xs = self.freqs
        
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
        print(min_pval)
        if min_pval != 0.0:
            ax.set_ylim(10 ** (math.floor(np.log10(min_pval))), 1)
        else:
            print('what the fuck')

        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        ax.set_ylabel('$p$', fontsize=18)
        if not title:
            title = 'P-Value Spectrum'
        ax.set_title(title, fontsize=20)
        ax.legend()

        if self.save:
            plt.savefig(f'{freq}_pval_spectrum')

        return fig

    def plot_stacked_normalized_spectra(self, freq=None, fig=None):
        '''
        Plot stacked normalized spectra.
        '''
        if freq:
            mask = np.abs(freq - self.freqs) < 100
            xs = self.freqs[mask]
        else:
            xs = self.freqs

        if fig is None:
            fig = Figure()
        ax = fig.subplots()

        for spec in self.uninj_norm_spectra:
            if freq:
                spec = spec[mask]
            powers = spec / np.median(spec)
            ax.plot(xs, powers, c='k', alpha=0.3, rasterized=True)

        if freq:
            ax.set_xlim(freq-100, freq+100)

        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        ax.set_ylabel('Normalized Power', fontsize=18)
        ax.set_title('Stacked Normalized Spectra', fontsize=20)

        if self.save:
            plt.savefig(f'{freq}_stacked_normalized_spectra')
        
        return fig

    def plot_stacked_raw_spectra(self, freq=None, fig=None):
        '''
        Plot stacked raw spectra.
        '''
        if freq:
            mask = np.abs(freq - self.freqs) < 100
            xs = self.freqs[mask]
        else:
            xs = self.freqs

        if fig is None:
            fig = Figure()
        ax = fig.subplots()

        for spec in self.uninj_spectra:
            if freq:
                spec = spec[mask]
            powers = spec / np.median(spec)
            ax.plot(xs, powers, c='k', alpha=0.3, rasterized=True)

        if freq:
            ax.set_xlim(freq-100, freq+100)
        
        ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
        ax.set_ylabel('Median-Normalized Power', fontsize=18)
        ax.set_title('Stacked Raw Spectra', fontsize=20)

        if self.save:
            plt.savefig(f'{freq}_stacked_raw_spectra')
        
        return fig

    def plot_heatmaps(self, freq=None, n_bins=50, resid=False, fig=None):
        if freq:
            mask = (self.freqs >= freq-50) & (self.freqs < freq+50)
        else:
            mask = np.full(self.freqs.shape, True)

        nans = np.full(self.freqs.shape, np.nan)[mask]

        self.info['time_mjd'] = self.info['time'].apply(lambda mjd: int(mjd.split('_')[0])) #[int(mjd.split('_')[0]) for mjd in info['time']]
        self.info['cos_theta'] = self.info['theta'].apply(lambda theta: math.cos(math.radians(theta))) #[math.cos(math.radians(theta)) for theta in info['theta']]
        self.info['cos_phi'] = self.info['phi'].apply(lambda phi: math.cos(math.radians(phi))) #[math.cos(math.radians(phi)) for phi in info['phi']]

        var_list = ['cos_theta', 'cos_phi', 'time_mjd']

        if fig is None:
            fig = Figure(figsize=(18, 18))
        axs = fig.subplots(3, 1)

        def create_map(bin_column, cond):
            for i in range(n_bins):
                bin_mask = cond & (self.info[bin_column] == i).to_numpy()
                spectra = self.uninj_norm_spectra[bin_mask][:, mask]
                if spectra.size:
                    avg = np.mean(spectra, axis=0)
                    yield avg
                else:
                    yield nans

        for i, var in enumerate(var_list):
            bounds = (min(self.info['time_mjd']), max(self.info['time_mjd'])) if var == 'time_mjd' else (-1, 1)
            bins = np.linspace(*bounds, n_bins+1)
            self.info[f'{var}_bin'] = pd.cut(self.info[var], bins, labels=False)

            if var == 'cos_theta':
                state = self.state_dopp
                label = r'$\mathrm{cos}({\theta})$'
            elif var == 'cos_phi':
                state = self.state_int
                label = r'$\mathrm{cos}({\phi})$'
            else:
                state = np.ones(self.uninj_norm_spectra.shape[0])
                label = 'Modified JD'
                
            heat_map = np.array(list(create_map(f'{var}_bin', state == 1)))
            if resid:
                heat_map = heat_map - np.nanmean(self.uninj_norm_spectra[state == 1][:, mask], axis=0)

            ax = axs[i]
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(np.array(heat_map))
            if resid:
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
            extent = [np.min(self.freqs[mask]), np.max(self.freqs[mask]), *bounds]

            im1 = ax.imshow(heat_map, extent=extent, cmap='RdBu' if resid else 'viridis', origin='lower', norm=norm, aspect='auto', interpolation='none')

            ax.set_ylabel(label, fontsize=18)

            if var != 'time_mjd':
                #ax.axhline(0, c='r') # Inward/outward division
                ax.set_yticks(np.linspace(-1, 1, 5))

        fig.suptitle(f'Heatmaps', fontsize=20)

        axs[-1].set_xlabel(r'$\nu$ [MHz]', fontsize=18)

        cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation='horizontal', fraction=0.0335, pad=0.04)
        cbar.set_label('Normalized Power Residual' if resid else 'Normalized Power', fontsize=18)
        
        return fig
    
    def get_hits(self, pval_threshold=None, distance=20):
        '''
        Get candidate hits below a specified p-value threshold.
        '''
        if pval_threshold is None:
            pval_threshold = .5*math.erfc(3/np.sqrt(2)) # 3-sigma threshold
        pval_threshold = -np.log10(pval_threshold)
        pvals = self.calc_combined_pvals()
        log_pvals = -np.log10(np.nan_to_num(np.array(pvals), nan=1))
        hits, _ = find_peaks(log_pvals, height=pval_threshold, distance=distance)
        return self.freqs[hits], pvals[hits]

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
        self.plot_heatmaps(freq=freq, resid=True, fig=heatmap_subfig)

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
    
    def exclude(self, freqs, width=50):
        if not isinstance(freqs, (list, np.ndarray)):
            freqs = [freqs]
        
        for freq in freqs:
            idx = np.argmin(np.abs(freq - self.freqs))
            self.asymmetry_dopp[idx-width:idx+width] = np.nan
            self.asymmetry_int[idx-width:idx+width] = np.nan
    
    def run(self, save_path=None):
        '''
        Perform the entire whack-a-mole procedure;
        i.e., identify, plot and eliminate p-value hits until none are left.
        '''
        if save_path is None:
            save_path = 'whack_a_mole_results.pdf'
        pdf = PdfPages(save_path)
        
        i = 0
        while True:
            freqs, _ = self.get_hits()
            
            fig = self.plot_pval_spectrum(sc=freqs, title=f'Round {i}: P-Value Spectrum')
            pdf.savefig(fig)
            plt.close(fig)

            if not len(freqs):
                break
            
            print(f'Round {i}')

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