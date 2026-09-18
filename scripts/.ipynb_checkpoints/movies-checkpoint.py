import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gc

C = 299792
V_VIRIAL = 250 # km/s

def _round_up_to_nearest_odd(number):
    ceiled_number = math.ceil(number)
    return int(ceiled_number + 1) if ceiled_number % 2 == 0 else int(ceiled_number)

class NormalizedSpectrum:
    """Normalized spectrum class."""
    
    def __init__(self,
                 path,
                 mode='dynamic',
                 freq=None,
                 width=200,
                 **kwargs):
        """
        Preprocessing and normalization.
        
        Parameters:
        path (str): Path to spectrum file
        mode (str, optional): Dynamic or static normalization, default dynamic
        freq (float or None, optional): Center frequency of region to normalize, default None (normalize the entire spectrum)
        width (float or None, optional): Half width (in MHz) of region to normalize, default 200
        
        """
        
        xs, ys = np.load(path)
        self.freq_diff = xs[1] - xs[0]
        offset = 0
        if freq:
            index = np.argmin(np.abs(xs - freq))
            width = int(width / self.freq_diff)
            xs = xs[index-width:index+width]
            ys = ys[index-width:index+width]
            offset = 16 - (index-width) % 16

        ys_mean = ys / ys.mean()
        for i in range(offset, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
            ys_mean[i] = np.NaN
            ys_mean[i+15] = np.NaN
            ys_mean[i+1] = np.NaN
            ys_mean[i+14] = np.NaN
        
        self.spectrum = np.array([xs, ys])
        self.xs, self.ys = xs, ys
        self.ys_mean = ys_mean
        self.mean_spectrum = np.array([xs, ys_mean])
        
        self.mode = mode
        
        # Normalization
        print('Normalizing spectrum...')

        if mode == 'dynamic':
            self.normalize_dynamic(**kwargs)
        else:
            self.normalize_static(**kwargs)
        
        print('Normalization complete!')

    def normalize_dynamic(self,
                          nsigma=1.5,
                          window_factor=0.02,
                          order=5,
                          is_decay=True):
        
        self.is_decay = is_decay
        self.window_factor = window_factor
        self.nsigma = nsigma
        
        new_xs = []
        normalized_spectrum = []
        pg_params = []
        p_params = []

        # Loop through the data points and divide each point by the polynomial 
        for i, x in enumerate(self.xs):
            window = _round_up_to_nearest_odd(window_factor * x)
            if i < (window//2):
                continue
            if i > len(self.xs) - (window//2 + 1):
                break
            ind_xs = np.arange(-(window//2), window//2+1)
            current_ys = self.ys_mean[i-window//2:i+window//2+1]
            idx = np.isfinite(current_ys)

            # Center bounds
            lower_bound = -(window//2) * self.freq_diff
            upper_bound = (window//2) * self.freq_diff
            #guesses = np.ones_like(range(order+1)).tolist()
            guesses = [1, *np.zeros(order)]
            bounds = (
                (0, lower_bound, *[-np.inf for _ in guesses]),
                (np.inf, upper_bound, *[np.inf for _ in guesses])
            )

            # Fit the defined function to the data using curve fitting
            fit_func_wrapper = partial(self._poly_gaussian, window_center = x)
            #polynomial + gaussian
            parameters, covariance = curve_fit(
                fit_func_wrapper,
                ind_xs[idx],
                current_ys[idx],
                p0=[0.001, 0, *guesses],
                bounds=bounds,
                maxfev=1000000)
            #polynomial + guassian

            fit_center_ = parameters[1]
            cent = fit_center_ / self.freq_diff
            sig = nsigma * self._get_sigma(x + fit_center_) / self.freq_diff # 3 sigma 
            mask1 = ind_xs[idx] < (cent - sig)
            mask2 = ind_xs[idx] > (cent + sig)
            combined_mask = mask1 | mask2
            xs_masked = ind_xs[idx][combined_mask]
            ys_masked = current_ys[idx][combined_mask]

            #polynomial
            parameters2, covariance = curve_fit(self._poly, xs_masked, ys_masked, p0=guesses, maxfev=1000000)   

            p = self._poly(0, *parameters2)
            pg = fit_func_wrapper(0, *parameters)
            new_xs.append(x)
            normalized_spectrum.append(pg/p)
            pg_params.append(parameters)
            p_params.append(parameters2)
        
        self.norm_spectrum = np.array([new_xs, normalized_spectrum])
        self.n_params = pg_params
        self.d_params = p_params

    def normalize_static(self,
                         nsigma=1.5,
                         window=201,
                         order=5,
                         is_decay=True):
        
        self.is_decay = is_decay
        self.window = window
        self.nsigma = nsigma
        
        ind_xs = np.arange(-(window//2), window//2+1)
        lower = (window - 1) // 2
        upper = (window + 1) // 2
        
        # Center bounds
        lower_bound = -(window//2) * self.freq_diff
        upper_bound = (window//2) * self.freq_diff
        #guesses = np.ones_like(range(order+1)).tolist()
        guesses = [1, *np.zeros(order)]
        bounds = (
            (0, lower_bound, *[-np.inf for _ in guesses]),
            (np.inf, upper_bound, *[np.inf for _ in guesses])
        )
        
        new_xs = []
        normalized_spectrum = []
        pg_params = []
        p_params = []

        # Loop through the data points and divide each point by the polynomial 
        for i in range(lower, len(self.xs)-upper):
            
            x = self.xs[i]
            current_ys = self.ys_mean[i-lower:i+upper]
            idx = np.isfinite(current_ys)
            
            # Fit the defined function to the data using curve fitting
            fit_func_wrapper = partial(self._poly_gaussian, window_center = x)
            #polynomial + gaussian
            parameters, covariance = curve_fit(
                fit_func_wrapper,
                ind_xs[idx],
                current_ys[idx],
                p0=[0.1, 0, *guesses],
                bounds=bounds,
                maxfev=1000000)
            #polynomial + guassian

            fit_center_ = parameters[1]
            cent = fit_center_ / self.freq_diff
            sig = nsigma * self._get_sigma(x + fit_center_) / self.freq_diff # 3 sigma 
            mask1 = ind_xs[idx] < (cent - sig)
            mask2 = ind_xs[idx] > (cent + sig)
            combined_mask = mask1 | mask2
            xs_masked = ind_xs[idx][combined_mask]
            ys_masked = current_ys[idx][combined_mask]

            #polynomial
            parameters2, covariance = curve_fit(self._poly, xs_masked, ys_masked, p0=guesses, maxfev=1000000)   

            p = self._poly(0, *parameters2)
            pg = fit_func_wrapper(0, *parameters)
            new_xs.append(x)
            normalized_spectrum.append(pg/p)
            pg_params.append(parameters)
            p_params.append(parameters2)
        
        self.norm_spectrum = np.array([new_xs, normalized_spectrum])
        self.n_params = pg_params
        self.d_params = p_params
    
    def _get_sigma(self, center):
        if self.is_decay:
            sigma = V_VIRIAL*center/(C*np.sqrt(3))
        else:
            sigma = V_VIRIAL*center/(C*np.sqrt(6))
        return sigma

    def _poly_gaussian(self, x, A, center, *coeffs, window_center=0):
        ''' Weighted polynomial plus Gaussian
        '''
        sigma = self._get_sigma(window_center + center)
        y = (A**2)*np.exp((-(self.freq_diff*x-center)**2)/(2*sigma**2))
        y += self._poly(x, *coeffs)
        return y

    def _poly(self, x, *coeffs): #polynomial
        ''' 
        '''
        y = 0
        for i, coeff in enumerate(coeffs):
            y += coeff * x ** i
        return y
    
    def get_spectrum(self):
        return self.spectrum
    
    def get_norm_spectrum(self):
        return self.norm_spectrum
    
    def generate_movie(self, save_dir=None):
        ######## INITIALIZE THE PLOT ########

        fig, axs = plt.subplots(2, 1, sharex=True)
        plt.subplots_adjust(wspace=0)

        pg, = axs[0].plot([], [], color='blue', label='pg')
        p, = axs[0].plot([], [], color='red', ls='--',label='p')

        line_0 = axs[0].axvline(0, ls='--', color='k')
        line_1 = axs[1].axvline(0, ls='--', color='k')

        window_box = axs[0].axvspan(0, 0, color='blue', alpha=0.2, label='window')
        masked_box = axs[0].axvspan(0, 0, color='gray', alpha=0.2, label='masked')

        norm_spectrum, = axs[1].plot([], [], color='k', label='pg/p')

        xs, ys_mean = self.mean_spectrum
        idx = np.isfinite(ys_mean)
        xs_norm, ys_norm = self.norm_spectrum

        axs[0].set_xlim(xs[0], xs[-1])
        axs[0].set_ylim([min(ys_mean), max(ys_mean)])
        axs[1].set_xlim(xs[0], xs[-1])
        axs[1].set_ylim([min(ys_norm), max(ys_norm)])
        axs[0].legend(loc='upper right')
        axs[1].legend()

        #####################################

        def draw():
            spectrum, = axs[0].plot(xs[idx], ys_mean[idx], ls='', color='k', marker='o', markersize=1)
            box1 = axs[1].axvspan(min(xs), min(xs_norm), color='gray')
            box2 = axs[1].axvspan(max(xs_norm), max(xs), color='gray')
            return spectrum, box1, box2

        def animate(i):
            x = xs_norm[i]
            if self.mode == 'dynamic':
                window = _round_up_to_nearest_odd(self.window_factor * x)
            else:
                window = self.window
            xmin = x - (window//2) * self.freq_diff
            xmax = x + (window//2) * self.freq_diff
            y_coords = window_box.get_xy()[:,1]
            window_box.set_xy([(xmin, y_coords[0]),
                              (xmin, y_coords[1]),
                              (xmax, y_coords[2]),
                              (xmax, y_coords[3])])

            i_prime = xs.tolist().index(x)
            full_ind_xs = np.arange(-i_prime, len(xs)-i_prime)

            fit_center_ = self.n_params[i][1]
            cent = fit_center_ / self.freq_diff
            sig = self.nsigma * self._get_sigma(x + fit_center_) / self.freq_diff # 3 sigma 
            mask1 = full_ind_xs > (cent - sig)
            mask2 = full_ind_xs < (cent + sig)
            combined_mask = mask1 & mask2
            xs_masked = xs[combined_mask]
            xmin_masked, xmax_masked = xs_masked[0], xs_masked[-1]
            y_coords = masked_box.get_xy()[:,1]
            masked_box.set_xy([(xmin_masked, y_coords[0]),
                              (xmin_masked, y_coords[1]),
                              (xmax_masked, y_coords[2]),
                              (xmax_masked, y_coords[3])])

            fit_func_wrapper = partial(self._poly_gaussian, window_center=x)
            parameters, parameters2 = self.n_params[i], self.d_params[i]
            pg.set_data(xs, fit_func_wrapper(full_ind_xs, *parameters))
            p.set_data(xs, self._poly(full_ind_xs, *parameters2))

            norm_spectrum.set_data(xs_norm[:i], ys_norm[:i])

            line_0.set_xdata([x, x])
            line_1.set_xdata([x, x])
            
            gc.collect()
            
            return pg, p, norm_spectrum, line_0, line_1, window_box, masked_box

        print('Writing gif...')
        anim = FuncAnimation(fig=fig,
                             func=animate,
                             init_func=draw,
                             frames=len(xs_norm),
                             interval=100,
                             blit=True)
        if save_dir:
            save_dir_head = save_dir.replace(save_dir.split('/')[-1], '')
            os.makedirs(save_dir_head, exist_ok=True)
        else:
            save_dir = f'norm_{self.mode}.gif'
        anim.save(save_dir)
        print('gif saved!')