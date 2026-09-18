import numpy as np
import pandas as pd
import os
from functools import partial
from scipy.optimize import curve_fit
import itertools
from tqdm import tqdm, trange
import math
import matplotlib.pyplot as plt

# Constants
c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s
dm_profile = 'nfw'


######### INJECTION #########

def convert_to_seconds(l):
    mjd, seconds = l.split('_')
    mjd, seconds = int(mjd), int(seconds)
    return mjd * 86400 + seconds

def get_shift_factor(theta, v_earth):
    beta = v_earth / c
    radians_from_v_earth = np.radians(theta)
    return 1 + beta*np.cos(radians_from_v_earth)


def inject_calibrated(name, sig_loc, info, data_dir=None, save_dir=None, calibration_dir=None, signal_size=1e-28, is_decay=True, template=False):
    freq, spec = np.load(os.path.join(data_dir, name))
    
    targ_info = info.set_index('source')
    time = convert_to_seconds(targ_info.loc[name[:-4], 'time'])
    if time < 5.18e9:
        sefd_path = os.path.join(calibration_dir, 'pre22B_median.npy')
    else:
        sefd_path = os.path.join(calibration_dir, 'post22B_median.npy')
    
    if template:
        spec = np.ones_like(spec)
        sefd = spec
    else:
        sefd = np.load(sefd_path)[1]
        
    # Calculate line intensity
    if is_decay:
        sig_std = v_virial*sig_loc/(c*np.sqrt(3))
        sig_amp_ratio = 2.75e24*signal_size / (v_virial*(sig_loc/1e3)**3) # convert MHz to GHz # Units: Jy
        col_name = dm_profile
    else:
        sig_std = v_virial*sig_loc/(c*np.sqrt(6))
        sig_amp_ratio = 9.41e14*signal_size / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz # Units: Jy
        col_name = dm_profile+'_sq'
    
    # Multiply by line-integrated halo density
    sig_amp = sig_amp_ratio * targ_info.loc[name[:-4], col_name]
    
    # Doppler shift
    theta = targ_info.loc[name[:-4], 'theta']
    shift_factor = get_shift_factor(theta, v_earth)
    sig_loc, sig_std, sig_amp = sig_loc * shift_factor, sig_std * shift_factor, sig_amp / shift_factor
    
    # Add the Gaussian to the spectrum
    G = sig_amp*np.exp(-(freq - sig_loc)**2 / (2*sig_std**2))
    new_spec = spec * (sefd + G) / sefd
    
    # Save/return the injected spectrum
    injected = np.array([freq, new_spec])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, name), injected)
    return injected


def inject(name, sig_loc, info, data_dir=None, save_dir=None, signal_size=1e-28, is_decay=True, template=False):
    freq, spec = np.load(os.path.join(data_dir, name))
    
    if template:
        spec = np.ones_like(spec)
        
    #spec_med = np.median(spec)
    targ_info = info.set_index('source')
    theta = targ_info.loc[name[:-4], 'theta']
        
    if is_decay:
        sig_std = v_virial*sig_loc/(c*np.sqrt(3))
        sig_amp_ratio = 2.75e23*signal_size / (v_virial*(sig_loc/1e3)**3) # convert MHz to GHz
        col_name = dm_profile
    else:
        sig_std = v_virial*sig_loc/(c*np.sqrt(6))
        sig_amp_ratio = 9.41e13*signal_size / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
        col_name = dm_profile+'_sq'
    
    amp_ratio = sig_amp_ratio * targ_info.loc[name[:-4], col_name]
    #sig_amp = spec_med * amp_ratio
    
    shift_factor = get_shift_factor(theta, v_earth)
    sig_loc, sig_std, amp_rato = sig_loc*shift_factor, sig_std*shift_factor, amp_ratio / shift_factor
    
    bkg = spec[np.argmin(abs(freq - sig_loc))]
    sig_amp = bkg * amp_ratio
    
    new_spec = spec + sig_amp*np.exp(-(freq - sig_loc)**2 / (2*sig_std**2))
    
    injected = np.array([freq, new_spec])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, name), injected)
    return injected
    

def inject_spaced(name, start, stop, spacing, info, data_dir=None, save_dir=None, signal_size=1e-28, is_decay=True, template=False):
    freq, spec = np.load(os.path.join(data_dir, name))
    
    if template:
        spec = np.ones_like(spec)
        
    #spec_med = np.median(spec)
    targ_info = info.set_index('source')
    theta = targ_info.loc[name[:-4], 'theta']
    
    sig_locs = np.arange(start, stop, spacing)
    for sig_loc in sig_locs:
        if is_decay:
            sig_std = v_virial*sig_loc/(c*np.sqrt(3))
            sig_amp_ratio = 2.75e23*signal_size / (v_virial*(sig_loc/1e3)**3) # convert MHz to GHz
            col_name = dm_profile
        else:
            sig_std = v_virial*sig_loc/(c*np.sqrt(6))
            sig_amp_ratio = 9.41e13*signal_size / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
            col_name = dm_profile+'_sq'

        amp_ratio = sig_amp_ratio * targ_info.loc[name[:-4], col_name]
        #sig_amp = spec_med * amp_ratio

        shift_factor = get_shift_factor(theta, v_earth)
        sig_loc, sig_std, amp_rato = sig_loc*shift_factor, sig_std*shift_factor, amp_ratio / shift_factor

        bkg = spec[np.argmin(abs(freq - sig_loc))]
        sig_amp = bkg * amp_ratio

        spec = spec + sig_amp*np.exp(-(freq - sig_loc)**2 / (2*sig_std**2))
    
    injected = np.array([freq, spec])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, name), injected)
    return injected

######### NORMALIZATION #########

def prepare(xs, ys, freq_diff, freq=None, width=200, start=None, stop=None):
    '''Prepare and optionally clip the spectrum.'''
    offset = 0
    if freq:
        index = np.argmin(np.abs(xs - freq))
        width = int(width / freq_diff)
        xs_trimmed = xs[index-width:index+width+1]
    else:
        if start or stop:
            start_index = 0 if start is None else np.argmin(np.abs(xs - start))
            stop_index = len(xs) - 1 if stop is None else np.argmin(np.abs(xs - stop))
            xs_trimmed = xs[start_index:stop_index+1]

    ys_mean = ys / ys.mean()
    for i in range(16, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
        ys_mean[i] = np.NaN
        ys_mean[i+15] = np.NaN
        ys_mean[i+1] = np.NaN
        ys_mean[i+14] = np.NaN
    
    return xs_trimmed, ys_mean

def round_up_to_nearest_odd(number):
    ceiled_number = math.ceil(number)
    return int(ceiled_number + 1) if ceiled_number % 2 == 0 else int(ceiled_number)

    
#Dynamic normalize
def normalize_dynamic(data, data_dir=None, save_dir=None, interior=2, exterior=3, order=5, is_decay=True, **kwargs):
    if data_dir:
        xs, ys = np.load(os.path.join(data_dir, data))
    else:
        xs, ys = data
    
    freq_diff = xs[1] - xs[0]
    xs, ys_mean = prepare(xs, ys, freq_diff, **kwargs)

    new_xs = []
    normalized_spectrum = []
    
    def get_sigma(center):
        if is_decay:
            sigma = v_virial*center/(c*np.sqrt(3))
        else:
            sigma = v_virial*center/(c*np.sqrt(6))
        return sigma
    
    def fit_func(x, A, center, *coeffs, window_center=0):
        ''' Polynomial plus Gaussian.'''
        sigma = get_sigma(window_center + center)
        y = (A**2)*np.exp((-(freq_diff*x-center)**2)/(2*sigma**2))
        y += fit_func2(x, *coeffs)
        return y

    def fit_func2(x, *coeffs): #polynomial
        '''Polynomial.'''
        y = 0
        for i, coeff in enumerate(coeffs):
            y += coeff * x ** i
        return y
    
    # Loop through the data points and divide each point by the polynomial 
    for i, x in enumerate(xs):
        window = round_up_to_nearest_odd(2 * exterior * get_sigma(x) / freq_diff)
        if i < (window//2):
            continue
        if i >= len(xs) - (window//2 + 1):
            break
        ind_xs = np.arange(-(window//2), window//2+1)
        #print(f'{i}/{len(xs)-upper-lower}')
        current_ys = ys_mean[i-window//2:i+window//2+1]
        idx = np.isfinite(current_ys)
        
        # Center bounds
        lower_bound = -(window//2) * freq_diff
        upper_bound = (window//2) * freq_diff
        #guesses = np.ones_like(range(order+1)).tolist()
        guesses = [1, *np.zeros(order)]
        bounds = (
            (0, lower_bound, *[-np.inf for _ in guesses]),
            (np.inf, upper_bound, *[np.inf for _ in guesses])
        )
        
        # Fit the defined function to the data using curve fitting
        fit_func_wrapper = partial(fit_func, window_center=x)
        #polynomial + gaussian
        parameters, covariance = curve_fit(
            fit_func_wrapper,
            ind_xs[idx],
            current_ys[idx],
            p0=[0.1, 0, *guesses],
            bounds=bounds,
            maxfev=1000000)

        # cent = ind_xs[np.argmin(np.abs(ind_xs-fit_center_/freq_diff))]
        fit_center_ = parameters[1]
        cent = fit_center_/freq_diff
        sig = interior * get_sigma(x + fit_center_)/freq_diff # 3 sigma 
        mask1 = ind_xs[idx] < (cent - sig)
        mask2 = ind_xs[idx] > (cent + sig)
        combined_mask = mask1 | mask2
        xs_masked = ind_xs[idx][combined_mask]
        ys_masked = current_ys[idx][combined_mask]
        
        #parameters2, covariance = curve_fit(fit_func2, xs_masked, ys_masked, p0=guesses, maxfev=1000000)
        parameters2 = np.polyfit(xs_masked, ys_masked, order)
        
        #p = fit_func2(0, *parameters2)
        p = np.poly1d(parameters2)(0)
        pg = fit_func_wrapper(0, *parameters)
        new_xs.append(x)
        normalized_spectrum.append(pg/p)
        
        del current_ys, parameters, parameters2
           
    normalized = np.array([new_xs, normalized_spectrum])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, data), normalized)
        
    return normalized


#Old normalize
def normalize(data, data_dir=None, save_dir=None, window=111, freq=None, width=200, nsigma=2, order=5, is_decay=True):
    if data_dir:
        xs, ys = np.load(os.path.join(data_dir, data))
    else:
        xs, ys = data
    freq_diff = xs[1] - xs[0]
    offset = 0
    
    if freq:
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
    #guesses = np.ones_like(range(order+1)).tolist()
    guesses = [1, *np.zeros(order)]
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
            p0=[0.001, 0, *guesses],
            bounds=bounds,
            maxfev=1000000)

        fit_center_ = parameters[1]
        cent = fit_center_/freq_diff
        sig = nsigma * get_sigma(xs[i] + fit_center_)/freq_diff # 3 sigma 
        mask1 = ind_xs[idx] < (cent - sig)
        mask2 = ind_xs[idx] > (cent + sig)
        combined_mask = mask1 | mask2
        xs_masked = ind_xs[idx][combined_mask]
        ys_masked = current_ys[idx][combined_mask]
        
        parameters2, covariance = curve_fit(fit_func2, xs_masked, ys_masked, p0=guesses, maxfev=1000000)  
        
        p = fit_func2(0, *parameters2)
        pg = fit_func_wrapper(0, *parameters)
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

        fit_center_ = parameters[1]
        cent = fit_center_/freq_diff
        sig = nsigma * get_sigma(xs[i] + fit_center_)/freq_diff # 3 sigma 
        mask1 = ind_xs[idx] < (cent - sig)
        mask2 = ind_xs[idx] > (cent + sig)
        combined_mask = mask1 | mask2
        xs_masked = ind_xs[idx][combined_mask]
        ys_masked = current_ys[idx][combined_mask]

        parameters2, covariance = curve_fit(fit_func2, xs_masked, ys_masked, p0=guesses, maxfev=1000000)   
        
        p = fit_func2(0, *parameters2)
        pg = fit_func_wrapper(0, *parameters)
        normalized_spectrum.append(pg/p)
    
    new_xs = xs[lower: len(xs)-upper]
    normalized = np.array([new_xs, normalized_spectrum])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, data), normalized)
    return normalized
    

def normalize_weighted(data, data_dir=None, save_dir=None, a=0.1, b=1, exterior=3, order=5, is_decay=True, **kwargs):
    """
    Standard rolling polynomial normalize procedure. Updated.
    """
    if data_dir:
        xs, ys = np.load(os.path.join(data_dir, data))
    else:
        xs, ys = data
    
    freq_diff = xs[1] - xs[0]
    xs_trimmed, ys_mean = prepare(xs, ys, freq_diff, **kwargs)
    
    new_xs = []
    normalized_spectrum = []
    
    def get_sigma(center):
        if is_decay:
            sigma = v_virial*center/(c*np.sqrt(3))
        else:
            sigma = v_virial*center/(c*np.sqrt(6))
        return sigma
    
    # Loop through the data points and divide each point by the polynomial 
    for x in xs_trimmed:
        i = xs.tolist().index(x)
        window = round_up_to_nearest_odd(2 * exterior * get_sigma(x) / freq_diff)
        ind_xs = np.arange(-(window//2), window//2+1)
        #print(f'{i}/{len(xs)-upper-lower}')
        current_ys = ys_mean[i-window//2:i+window//2+1]
        idx = np.isfinite(current_ys)

        # Unweighted fit
        unweighted_params = np.polyfit(ind_xs[idx], current_ys[idx], order)
        
        # Weighted fit
        sigma = get_sigma(xs[i])
        weights = 1 - a*np.exp(-(freq_diff*ind_xs[idx])**2 / (2*b*sigma**2))
        weighted_params = np.polyfit(ind_xs[idx], current_ys[idx], order, w=weights)
        
        unweighted = np.poly1d(unweighted_params)(0)
        weighted = np.poly1d(weighted_params)(0)
        
        new_xs.append(x)
        normalized_spectrum.append(unweighted/weighted)
    
    #new_ys = ys_mean[lower: len(xs)-upper]
    normalized = np.array([new_xs, normalized_spectrum])
    #old = np.array([new_xs, new_ys])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, data), normalized)
    return normalized
