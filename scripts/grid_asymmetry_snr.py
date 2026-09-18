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
from tqdm import tqdm, trange
from functools import partial
import random
from bayes_opt import BayesianOptimization, acquisition
import math


num_processes = mp.cpu_count()

info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/all_xband_info.csv'
info = pd.read_csv(info_dir)

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
    upper_bound = (window//2) * freq_diff
    guesses = np.ones_like(range(order+1)).tolist()
    bounds = (
        (0, lower_bound, *[-np.inf for _ in guesses]),
        (np.inf, upper_bound, *[np.inf for _ in guesses])
    )
    
    # Loop through the data points and divide each point by the polynomial 
    for i in range(lower, len(xs)-upper):
        
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

    
# def normalize_unconst(data, data_dir=None, save_dir=None, window=111, freq=None, width=200, nsigma=2, order=5, is_decay=True):
#     if data_dir:
#         xs, ys = np.load(os.path.join(data_dir, data))
#     else:
#         xs, ys = data
#     freq_diff = xs[1] - xs[0]
#     if freq:
#         #index = int((freq - xs[0]) / freq_diff)
#         index = np.argmin(np.abs(xs - freq))
#         width = int(width / freq_diff)
#         xs = xs[index-width:index+width]
#         ys = ys[index-width:index+width]
#         offset = 16 - (index-width) % 16
#     ind_xs = np.arange(-(window//2), window//2+1)
#     lower = (window - 1) // 2
#     upper = (window + 1) // 2
#     ys_mean = ys / ys.mean()
#     for i in range(offset, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
#         ys_mean[i] = np.NaN
#         ys_mean[i+15] = np.NaN
#         ys_mean[i+1] = np.NaN
#         ys_mean[i+14] = np.NaN
    
#     def get_sigma(center):
#         if is_decay:
#             sigma = v_virial*center/(c*np.sqrt(3))
#         else:
#             sigma = v_virial*center/(c*np.sqrt(6))
#         return sigma
    
#     def fit_func(x, A, center, *coeffs, window_center=0):
#         ''' Weighted polynomial plus Gaussian
#         '''
#         sigma = get_sigma(window_center + center)
#         y = A*np.exp((-(freq_diff*x-center)**2)/(2*sigma**2))
#         y += fit_func2(x, *coeffs)
#         return y
        
#     def fit_func2(x, *coeffs): #polynomial
#         ''' 
#         '''
#         y = 0
#         for i, coeff in enumerate(coeffs):
#             y += coeff * x ** i
#         return y

#     normalized_spectrum = []
#     # Center bounds
#     lower_bound = -(window//2) * freq_diff
#     upper_bound = (window//2+1) * freq_diff
#     #print(-(window//2), (window//2+1))
#     guesses = np.ones_like(range(order+1)).tolist()
#     bounds = (
#         (-np.inf, lower_bound, *[-np.inf for _ in guesses]),
#         (np.inf, upper_bound, *[np.inf for _ in guesses])
#     )
    
#     #print(f'n:{len(xs)-lower-upper}')
#     # Loop through the data points and divide each point by the polynomial 
#     for i in range(lower, len(xs)-upper):
        
#         #print(f'{i}/{len(xs)-upper-lower}')
#         current_ys = ys_mean[i-lower:i+upper]
#         idx = np.isfinite(current_ys)

#         # Fit the defined function to the data using curve fitting
#         fit_func_wrapper = partial(fit_func, window_center = xs[i])
#         #polynomial + gaussian
#         parameters, covariance = curve_fit(
#             fit_func_wrapper,
#             ind_xs[idx],
#             current_ys[idx],
#             p0=[2, 1, *guesses],
#             bounds=bounds,
#             maxfev=1000000)
#         #polynomial + guassian
#         # fit_A_, fit_b_, fit_c_, fit_d_, fit_f_, fit_g_, fit_h_, fit_center_= parameters

#         # cent = ind_xs[np.argmin(np.abs(ind_xs-fit_center_/freq_diff))]
#         fit_center_ = parameters[1]
#         cent = fit_center_/freq_diff
#         sig = nsigma * get_sigma(xs[i] + fit_center_)/freq_diff # 3 sigma 
#         # xs_1 = ind_xs[idx][:np.argmin(np.abs(ind_xs[idx] - (cent-sig)))]
#         # xs_2 = ind_xs[idx][np.argmin(np.abs(ind_xs[idx] - (cent+sig))):]
#         # ys_1 = current_ys[idx][:np.argmin(np.abs(ind_xs[idx] - (cent-sig)))]
#         # ys_2 = current_ys[idx][np.argmin(np.abs(ind_xs[idx] - (cent+sig))):]
#         #sig = 12 # testing the old range
#         mask1 = ind_xs[idx] < (cent - sig)
#         mask2 = ind_xs[idx] > (cent + sig)
#         combined_mask = mask1 | mask2
#         xs_masked = ind_xs[idx][combined_mask]
#         ys_masked = current_ys[idx][combined_mask]
#         #print(f'Center (bins): {cent}, 3-sigma (bins): {sig}, xs: {xs_masked}')
        
#         #polynomial
#         # parameters2, covariance = curve_fit(fit_func2, np.concatenate((xs_1, xs_2)), np.concatenate((ys_1, ys_2)),  p0=[1,1,1,1,1,0], maxfev=1000000)
#         parameters2, covariance = curve_fit(fit_func2, xs_masked, ys_masked, p0=guesses, maxfev=1000000)
#         #polynomial
#         # fit_b, fit_c, fit_d, fit_f, fit_g, fit_h = parameters2      
        
#         p = fit_func2(0, *parameters2)
#         pg = fit_func_wrapper(0, *parameters)
#         #fit_ys = fit_func(ind_xs, fit_A_, fit_b_, fit_c_, fit_d_, fit_f_, fit_g_, fit_h_, fit_center_)
#         # chi_squared = np.sum((current_ys[idx] - fit_ys[idx])**2/fit_ys[idx])
#         # chi_squareds.append(chi_squared)
#         normalized_spectrum.append(pg/p)
    
#     new_xs = xs[lower: len(xs)-upper]
#     normalized = np.array([new_xs, normalized_spectrum])
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         np.save(os.path.join(save_dir, data), normalized)
#     return normalized
    

# def normalize_weighted(data, data_dir=None, save_dir=None, a=0.1, b=1, window=261, freq=None, width=200, order=5, is_decay=True):#xs, ys, a, b, sigma, order, window, rb_size=1):
#     """
#     Standard rolling polynomial normalize procedure. Updated.
#     """
#     if data_dir:
#         xs, ys = np.load(os.path.join(data_dir, data))
#     else:
#         xs, ys = data
#     freq_diff = xs[1] - xs[0]
#     offset = 0
#     if freq:
#         #index = int((freq - xs[0]) / freq_diff)
#         index = np.argmin(np.abs(xs - freq))
#         width = int(width / freq_diff)
#         xs = xs[index-width:index+width]
#         ys = ys[index-width:index+width]
#         offset = 16 - (index-width) % 16
#     ind_xs = np.arange(-(window//2), window//2+1)
#     lower = (window - 1) // 2
#     upper = (window + 1) // 2
#     ys_mean = ys / ys.mean()
#     for i in range(offset, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
#         ys_mean[i] = np.NaN
#         ys_mean[i+15] = np.NaN
#         ys_mean[i+1] = np.NaN
#         ys_mean[i+14] = np.NaN
    
#     def get_sigma(center):
#         if is_decay:
#             sigma = v_virial*center/(c*np.sqrt(3))
#         else:
#             sigma = v_virial*center/(c*np.sqrt(6))
#         return sigma

#     normalized_spectrum = []

#     # Loop through the data points and divide each point by the polynomial 
#     for i in range(lower, len(xs)-upper):

#         current_ys = ys_mean[i-lower:i+upper]
#         idx = np.isfinite(current_ys)

#         # Unweighted fit
#         unweighted_params = np.polyfit(ind_xs[idx], current_ys[idx], order)
        
#         # Weighted fit
#         sigma = get_sigma(xs[i])
#         weights = 1 - a*np.exp(-(freq_diff*ind_xs[idx])**2 / (2*b*sigma**2))
#         weighted_params = np.polyfit(ind_xs[idx], current_ys[idx], order, w=weights)
        
#         unweighted = np.poly1d(unweighted_params)(0)
#         weighted = np.poly1d(weighted_params)(0)
        
#         normalized_spectrum.append(unweighted/weighted)
    
#     new_xs = xs[lower: len(xs)-upper]
#     #new_ys = ys_mean[lower: len(xs)-upper]
#     normalized = np.array([new_xs, normalized_spectrum])
#     #old = np.array([new_xs, new_ys])
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         np.save(os.path.join(save_dir, data), normalized)
#     return normalized


def build_asymmetry(data_dir, angle, lower, upper, spectra):
    inner = info[info[angle] < lower]
    outer = info[info[angle] > upper]
    inner_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in inner['source'].tolist()])
    outer_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in outer['source'].tolist()])
    inner_mean = np.mean(inner_spectra, axis=0)
    outer_mean = np.mean(outer_spectra, axis=0)
    asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)
    xs = np.load(os.path.join(data_dir, spectra[0]))[0]
    return xs, asymmetry


def calculate_snr(normalize_func, uninjected_dir=None, injected_dir=None, uninj_save_dir=None, inj_save_dir=None, fit_window=None, freq=8720, **kwargs):
    spectra = os.listdir(injected_dir)
    
    with mp.Pool(processes=num_processes) as p:
        for data_dir, save_dir in ((uninjected_dir, uninj_save_dir), (injected_dir, inj_save_dir)):
            normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
            _ = list(tqdm(p.imap_unordered(normalize_func_partial, spectra), total=len(spectra), desc=f'Normalizing {data_dir.split("/")[-2]} with {normalize_func.__name__}'))
    
    #print('Building asymmetries')
    xs, uninj_ys = build_asymmetry(uninj_save_dir, 'phi', 70, 115, spectra)
    xs, inj_ys = build_asymmetry(inj_save_dir, 'phi', 70, 115, spectra)
    
    #print('Calculating SNR')
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
    signal = np.max(fit)# - np.mean(window) #maybe subtract mean? 7/16/25
    if signal < 0: signal = 0
    
    # Return SNR
    return signal / noise


def inject(name, sig_loc, data_dir=None, save_dir=None, signal_size=1e-28, is_decay=True):
    if data_dir:
        freq, spec = np.load(os.path.join(data_dir, name))
    else:
        freq, spec = name
    
    spec_med = np.median(spec)
    targ_info = info.set_index('source')
    theta = targ_info.loc[name[:-4], 'theta']
        
    if is_decay:
        sig_std = v_virial*sig_loc/(c*np.sqrt(3))
        sig_amp_ratio = 2.75e23*signal_size / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
        col_name = dm_profile
    else:
        sig_std = v_virial*sig_loc/(c*np.sqrt(6))
        sig_amp_ratio = 3.55e28*signal_size / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
        col_name = dm_profile+'_sq'
    
    amp_ratio = sig_amp_ratio * targ_info.loc[name[:-4], col_name]
    sig_amp = spec_med * amp_ratio
    
    shift_factor = get_shift_factor(theta, v_earth)
    sig_loc, sig_std, sig_amp = sig_loc*shift_factor, sig_std*shift_factor, sig_amp / shift_factor
    new_spec = spec + sig_amp*np.exp(-(freq - sig_loc)**2 / (2*sig_std**2))
    
    injected = np.array([freq, new_spec])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, name), injected)
    return injected
    

def get_shift_factor(theta, v_earth):
    beta = v_earth / c
    radians_from_v_earth = np.radians(theta)
    return 1 + beta*np.cos(radians_from_v_earth)


# def objective(nsigma, window, **kwargs):
#     def round_up_to_nearest_odd(number):
#         ceiled_number = math.ceil(number)
#         return int(ceiled_number + 1) if ceiled_number % 2 == 0 else int(ceiled_number)
#     window = round_up_to_nearest_odd(window)
    
#     return calculate_snr(
#         normalize,
#         fit_window=10,
#         width=60,
#         nsigma=nsigma,
#         window=window,
#         order=5,
#         **kwargs
#     )

if __name__ == '__main__':
    freqs = [8720, 9520, 10500]
    banks = ['0', '1', '2']
    
    #targets = random.sample(info['source'].tolist(), 1000)
    targets = []
    ddir ='/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/'
    with open(os.path.join(ddir, 'random_names.txt'), 'r') as file:
        for line in file:
            targets.append(line.strip())
    print(f'Targets: {targets}')
    
    # Values for the grid search
    pg_params = list(itertools.product(np.arange(1.0, 2.1, 0.5), np.arange(161, 261, 30)))
    
    for freq, bank in zip(freqs, banks):
        print(f'Evaluating {freq} MHz (bank {bank})')
        
        uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/{bank}/preprocessed/'
        injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_grid_new/{bank}/injected/'
        uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_grid_new/{bank}/normalized_uninjected/'
        inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_grid_new/{bank}/normalized_injected/'
        
        # Get the targets
        targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
        print(f'# of targets in bank {bank}: {len(targs)}')
        
        # Inject those fuckers
        for targ in targs:
            inject(targ, freq, data_dir=uninjected_dir, save_dir=injected_dir)
        
        snrs = []
        for pg_param in tqdm(pg_params):
            nsigma, window = pg_param
            snr = calculate_snr(
                normalize,
                fit_window=10,
                freq=freq,
                width=60,
                nsigma=nsigma,
                window=window,
                order=5,
                uninjected_dir=uninjected_dir,
                injected_dir=injected_dir,
                uninj_save_dir=uninj_save_dir,
                inj_save_dir=inj_save_dir,
            )
            snrs.append(snr)
            
        
        # Save the SNRs for the bank
        np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/snr_grid_new/snrs_{bank}_{freq}.npy', snrs) 