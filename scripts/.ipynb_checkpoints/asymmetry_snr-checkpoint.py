import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
import random
from functools import partial
from scipy.optimize import curve_fit
import itertools
from tqdm import tqdm
from functools import partial
import random

num_processes = mp.cpu_count()

uninjected_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/0/preprocessed/'
injected_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/0/injected_28/'
uninj_save_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_evals/uninjected/'
inj_save_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_evals/injected/'
info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/all_xband_info.csv'
info = pd.read_csv(info_dir)

spectra = random.sample(os.listdir(uninjected_dir), 1000) # Random sample of 1000 spectra - should be good enough

c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s
dm_profile = 'nfw'

# Normalization techniques
def normalize(data, data_dir=None, save_dir=None, window=111, freq=None, width=200, nsigma=2, order=5, is_decay=True):
    if data_dir:
        xs, ys = np.load(os.path.join(data_dir, data))
    else:
        xs, ys = data
    freq_diff = xs[1] - xs[0]
    if freq:
        #index = int((freq - xs[0]) / freq_diff)
        index = np.argmin(np.abs(xs - freq))
        width = int(width / freq_diff)
        xs = xs[index-width:index+width]
        ys = ys[index-width:index+width]
        offset = 16 - (index-width) % 16
    ind_xs = np.arange(-(window//2), window//2+1)
    lower = (window - 1) // 2
    upper = (window + 1) // 2
    ys_mean = ys / ys.mean()
    for i in range(offset, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
        ys_mean[i] = np.NaN
        ys_mean[i+15] = np.NaN
        ys_mean[i+1] = np.NaN
        ys_mean[i+14] = np.NaN
    
    def get_sigma(center):
        if is_decay:
            sigma = v_virial*center/(c*np.sqrt(3))
        else:
            sigma = v_virial*center/(c*np.sqrt(6))
        return sigma
    
    def fit_func(x, A, center, *coeffs, window_center=0):
        ''' Weighted polynomial plus Gaussian
        '''
        sigma = get_sigma(window_center + center)
        y = (A**2)*np.exp((-(freq_diff*x-center)**2)/(2*sigma**2))
        y += fit_func2(x, *coeffs)
        return y
        
    def fit_func2(x, *coeffs): #polynomial
        ''' 
        '''
        y = 0
        for i, coeff in enumerate(coeffs):
            y += coeff * x ** i
        return y

    normalized_spectrum = []
    # Center bounds
    lower_bound = -(window//2) * freq_diff
    upper_bound = (window//2+1) * freq_diff
    #print(-(window//2), (window//2+1))
    guesses = np.ones_like(range(order+1)).tolist()
    bounds = (
        (0, lower_bound, *[-np.inf for _ in guesses]),
        (np.inf, upper_bound, *[np.inf for _ in guesses])
    )
    
    #print(f'n:{len(xs)-lower-upper}')
    # Loop through the data points and divide each point by the polynomial 
    for i in range(lower, len(xs)-upper):
        
        #print(f'{i}/{len(xs)-upper-lower}')
        current_ys = ys_mean[i-lower:i+upper]
        idx = np.isfinite(current_ys)

        # Fit the defined function to the data using curve fitting
        fit_func_wrapper = partial(fit_func, window_center = xs[i])
        #polynomial + gaussian
        parameters, covariance = curve_fit(
            fit_func_wrapper,
            ind_xs[idx],
            current_ys[idx],
            p0=[2, 1, *guesses],
            bounds=bounds,
            maxfev=1000000)
        #polynomial + guassian
        # fit_A_, fit_b_, fit_c_, fit_d_, fit_f_, fit_g_, fit_h_, fit_center_= parameters

        # cent = ind_xs[np.argmin(np.abs(ind_xs-fit_center_/freq_diff))]
        fit_center_ = parameters[1]
        cent = fit_center_/freq_diff
        sig = nsigma * get_sigma(xs[i] + fit_center_)/freq_diff # 3 sigma 
        # xs_1 = ind_xs[idx][:np.argmin(np.abs(ind_xs[idx] - (cent-sig)))]
        # xs_2 = ind_xs[idx][np.argmin(np.abs(ind_xs[idx] - (cent+sig))):]
        # ys_1 = current_ys[idx][:np.argmin(np.abs(ind_xs[idx] - (cent-sig)))]
        # ys_2 = current_ys[idx][np.argmin(np.abs(ind_xs[idx] - (cent+sig))):]
        #sig = 12 # testing the old range
        mask1 = ind_xs[idx] < (cent - sig)
        mask2 = ind_xs[idx] > (cent + sig)
        combined_mask = mask1 | mask2
        xs_masked = ind_xs[idx][combined_mask]
        ys_masked = current_ys[idx][combined_mask]
        #print(f'Center (bins): {cent}, 3-sigma (bins): {sig}, xs: {xs_masked}')
        
        #polynomial
        # parameters2, covariance = curve_fit(fit_func2, np.concatenate((xs_1, xs_2)), np.concatenate((ys_1, ys_2)),  p0=[1,1,1,1,1,0], maxfev=1000000)
        parameters2, covariance = curve_fit(fit_func2, xs_masked, ys_masked, p0=guesses, maxfev=1000000)
        #polynomial
        # fit_b, fit_c, fit_d, fit_f, fit_g, fit_h = parameters2      
        
        p = fit_func2(0, *parameters2)
        pg = fit_func_wrapper(0, *parameters)
        #fit_ys = fit_func(ind_xs, fit_A_, fit_b_, fit_c_, fit_d_, fit_f_, fit_g_, fit_h_, fit_center_)
        # chi_squared = np.sum((current_ys[idx] - fit_ys[idx])**2/fit_ys[idx])
        # chi_squareds.append(chi_squared)
        normalized_spectrum.append(pg/p)
    
    new_xs = xs[lower: len(xs)-upper]
    normalized = np.array([new_xs, normalized_spectrum])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, data), normalized)
    return normalized

    
def normalize_unconst(data, data_dir=None, save_dir=None, window=111, freq=None, width=200, nsigma=2, order=5, is_decay=True):
    if data_dir:
        xs, ys = np.load(os.path.join(data_dir, data))
    else:
        xs, ys = data
    freq_diff = xs[1] - xs[0]
    if freq:
        #index = int((freq - xs[0]) / freq_diff)
        index = np.argmin(np.abs(xs - freq))
        width = int(width / freq_diff)
        xs = xs[index-width:index+width]
        ys = ys[index-width:index+width]
        offset = 16 - (index-width) % 16
    ind_xs = np.arange(-(window//2), window//2+1)
    lower = (window - 1) // 2
    upper = (window + 1) // 2
    ys_mean = ys / ys.mean()
    for i in range(offset, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
        ys_mean[i] = np.NaN
        ys_mean[i+15] = np.NaN
        ys_mean[i+1] = np.NaN
        ys_mean[i+14] = np.NaN
    
    def get_sigma(center):
        if is_decay:
            sigma = v_virial*center/(c*np.sqrt(3))
        else:
            sigma = v_virial*center/(c*np.sqrt(6))
        return sigma
    
    def fit_func(x, A, center, *coeffs, window_center=0):
        ''' Weighted polynomial plus Gaussian
        '''
        sigma = get_sigma(window_center + center)
        y = A*np.exp((-(freq_diff*x-center)**2)/(2*sigma**2))
        y += fit_func2(x, *coeffs)
        return y
        
    def fit_func2(x, *coeffs): #polynomial
        ''' 
        '''
        y = 0
        for i, coeff in enumerate(coeffs):
            y += coeff * x ** i
        return y

    normalized_spectrum = []
    # Center bounds
    lower_bound = -(window//2) * freq_diff
    upper_bound = (window//2+1) * freq_diff
    #print(-(window//2), (window//2+1))
    guesses = np.ones_like(range(order+1)).tolist()
    bounds = (
        (-np.inf, lower_bound, *[-np.inf for _ in guesses]),
        (np.inf, upper_bound, *[np.inf for _ in guesses])
    )
    
    #print(f'n:{len(xs)-lower-upper}')
    # Loop through the data points and divide each point by the polynomial 
    for i in range(lower, len(xs)-upper):
        
        #print(f'{i}/{len(xs)-upper-lower}')
        current_ys = ys_mean[i-lower:i+upper]
        idx = np.isfinite(current_ys)

        # Fit the defined function to the data using curve fitting
        fit_func_wrapper = partial(fit_func, window_center = xs[i])
        #polynomial + gaussian
        parameters, covariance = curve_fit(
            fit_func_wrapper,
            ind_xs[idx],
            current_ys[idx],
            p0=[2, 1, *guesses],
            bounds=bounds,
            maxfev=1000000)
        #polynomial + guassian
        # fit_A_, fit_b_, fit_c_, fit_d_, fit_f_, fit_g_, fit_h_, fit_center_= parameters

        # cent = ind_xs[np.argmin(np.abs(ind_xs-fit_center_/freq_diff))]
        fit_center_ = parameters[1]
        cent = fit_center_/freq_diff
        sig = nsigma * get_sigma(xs[i] + fit_center_)/freq_diff # 3 sigma 
        # xs_1 = ind_xs[idx][:np.argmin(np.abs(ind_xs[idx] - (cent-sig)))]
        # xs_2 = ind_xs[idx][np.argmin(np.abs(ind_xs[idx] - (cent+sig))):]
        # ys_1 = current_ys[idx][:np.argmin(np.abs(ind_xs[idx] - (cent-sig)))]
        # ys_2 = current_ys[idx][np.argmin(np.abs(ind_xs[idx] - (cent+sig))):]
        #sig = 12 # testing the old range
        mask1 = ind_xs[idx] < (cent - sig)
        mask2 = ind_xs[idx] > (cent + sig)
        combined_mask = mask1 | mask2
        xs_masked = ind_xs[idx][combined_mask]
        ys_masked = current_ys[idx][combined_mask]
        #print(f'Center (bins): {cent}, 3-sigma (bins): {sig}, xs: {xs_masked}')
        
        #polynomial
        # parameters2, covariance = curve_fit(fit_func2, np.concatenate((xs_1, xs_2)), np.concatenate((ys_1, ys_2)),  p0=[1,1,1,1,1,0], maxfev=1000000)
        parameters2, covariance = curve_fit(fit_func2, xs_masked, ys_masked, p0=guesses, maxfev=1000000)
        #polynomial
        # fit_b, fit_c, fit_d, fit_f, fit_g, fit_h = parameters2      
        
        p = fit_func2(0, *parameters2)
        pg = fit_func_wrapper(0, *parameters)
        #fit_ys = fit_func(ind_xs, fit_A_, fit_b_, fit_c_, fit_d_, fit_f_, fit_g_, fit_h_, fit_center_)
        # chi_squared = np.sum((current_ys[idx] - fit_ys[idx])**2/fit_ys[idx])
        # chi_squareds.append(chi_squared)
        normalized_spectrum.append(pg/p)
    
    new_xs = xs[lower: len(xs)-upper]
    normalized = np.array([new_xs, normalized_spectrum])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, data), normalized)
    return normalized
    

def normalize_weighted(data, data_dir=None, save_dir=None, a=0.1, b=1, window=261, freq=None, width=200, order=5, is_decay=True):#xs, ys, a, b, sigma, order, window, rb_size=1):
    """
    Standard rolling polynomial normalize procedure. Updated.
    """
    if data_dir:
        xs, ys = np.load(os.path.join(data_dir, data))
    else:
        xs, ys = data
    freq_diff = xs[1] - xs[0]
    offset = 0
    if freq:
        #index = int((freq - xs[0]) / freq_diff)
        index = np.argmin(np.abs(xs - freq))
        width = int(width / freq_diff)
        xs = xs[index-width:index+width]
        ys = ys[index-width:index+width]
        offset = 16 - (index-width) % 16
    ind_xs = np.arange(-(window//2), window//2+1)
    lower = (window - 1) // 2
    upper = (window + 1) // 2
    ys_mean = ys / ys.mean()
    for i in range(offset, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
        ys_mean[i] = np.NaN
        ys_mean[i+15] = np.NaN
        ys_mean[i+1] = np.NaN
        ys_mean[i+14] = np.NaN
    
    def get_sigma(center):
        if is_decay:
            sigma = v_virial*center/(c*np.sqrt(3))
        else:
            sigma = v_virial*center/(c*np.sqrt(6))
        return sigma

    normalized_spectrum = []

    # Loop through the data points and divide each point by the polynomial 
    for i in range(lower, len(xs)-upper):

        current_ys = ys_mean[i-lower:i+upper]
        idx = np.isfinite(current_ys)

        # Unweighted fit
        unweighted_params = np.polyfit(ind_xs[idx], current_ys[idx], order)
        
        # Weighted fit
        sigma = get_sigma(xs[i])
        weights = 1 - a*np.exp(-(freq_diff*ind_xs[idx])**2 / (2*b*sigma**2))
        weighted_params = np.polyfit(ind_xs[idx], current_ys[idx], order, w=weights)
        
        unweighted = np.poly1d(unweighted_params)(0)
        weighted = np.poly1d(weighted_params)(0)
        
        normalized_spectrum.append(unweighted/weighted)
    
    new_xs = xs[lower: len(xs)-upper]
    #new_ys = ys_mean[lower: len(xs)-upper]
    normalized = np.array([new_xs, normalized_spectrum])
    #old = np.array([new_xs, new_ys])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, data), normalized)
    return normalized


def build_asymmetry(data_dir, angle, lower, upper):
    inner = info[info[angle] < lower]
    outer = info[info[angle] > upper]
    inner_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in inner['source'].tolist()])
    outer_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in outer['source'].tolist()])
    inner_mean = np.mean(inner_spectra, axis=0)
    outer_mean = np.mean(outer_spectra, axis=0)
    asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)
    xs = np.load(os.path.join(data_dir, spectra[0]))[0]
    return xs, asymmetry


def calculate_snr(normalize_func, fit_window=None, freq=8720, **kwargs):
    with mp.Pool(processes=num_processes) as p:
        for data_dir, save_dir in ((uninjected_dir, uninj_save_dir), (injected_dir, inj_save_dir)):
            normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
            _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing {data_dir.split("/")[-2]} with {normalize_func.__name__}'))
    
    print('Building asymmetries')
    xs, uninj_ys = build_asymmetry(uninj_save_dir, 'phi', 70, 115)
    xs, inj_ys = build_asymmetry(inj_save_dir, 'phi', 70, 115)
    
    print('Calculating SNR')
    noise_window = 3 * fit_window
    # Calculate noise
    mask = abs(xs - freq) < noise_window
    window = uninj_ys[mask]
    noise = np.std(window)
        
    # Calculate signal amplitude
    mask = abs(xs - freq) < fit_window
    window = [xs[mask], inj_ys[mask]]
    params = np.polyfit(*window, 2) #quadratic for intensity
    fit = np.poly1d(params)(window[0])
    signal = np.max(fit)
    
    # Return SNR
    return signal / noise

                                       
if __name__ == '__main__':
    pg_params = list(itertools.product(np.arange(1.0, 2.1, 0.5), np.arange(131, 200, 30)))
    w_params = list(itertools.product(np.arange(0.1, 0.31, 0.1), np.arange(131, 200, 30)))

    pg_results = []
    pg_unconst_results = []
    w_results = []
    
    for pg_param, w_param in zip(pg_params, w_params):
        
        print(f'pg: {pg_param}, w: {w_param}')
        nsigma, window = pg_param
        pg_snr = calculate_snr(
            normalize,
            fit_window=10,
            freq=8720,
            width=40,
            nsigma=nsigma,
            window=window,
            order=5
        )
        pg_unconst_snr = calculate_snr(
            normalize_unconst,
            fit_window=10,
            freq=8720,
            width=40,
            nsigma=nsigma,
            window=window,
            order=5
        )
        a, window = w_param
        w_snr = calculate_snr(
            normalize_weighted,
            fit_window=10,
            freq=8720,
            width=40,
            a=a,
            window=window,
            order=5
        )
        pg_results.append(pg_snr)
        pg_unconst_results.append(pg_unconst_snr)
        w_results.append(w_snr)
    
    np.save('/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_evals/pg_snrs', np.array(pg_results))
    np.save('/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_evals/pg_u_snrs', np.array(pg_unconst_results))
    np.save('/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_evals/w_snrs', np.array(w_results))
