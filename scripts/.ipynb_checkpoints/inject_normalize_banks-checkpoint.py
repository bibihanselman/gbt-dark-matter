import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
import random
from functools import partial
from scipy.optimize import curve_fit
from tqdm import tqdm
import math
import argparse

c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s
dm_profile = 'nfw'

def inject(xs, ys, theta, sig_loc, signal_size, v_earth, v_virial, is_decay, name, targ_info, sig_amp, sig_std):
    shift_factor = get_shift_factor(theta, v_earth)
    sig_loc, sig_std, sig_amp = sig_loc*shift_factor, sig_std*shift_factor, sig_amp / shift_factor
    return (ys + sig_amp*np.exp(-(xs - sig_loc)**2 / (2*sig_std**2)))
    
def get_shift_factor(theta, v_earth):
    beta = v_earth / c
    radians_from_v_earth = np.radians(theta)
    return 1 + beta*np.cos(radians_from_v_earth)

def inject_one(name, sig_loc, data_dir=None, save_dir=None, signal_size=1e-28, is_decay=True):
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

def inject_spaced_one(name, data_dir=None, save_dir=None, signal_size=1e-27, loc=7520, spacing=100, template=False, is_decay=True):
    freq, spec = np.load(os.path.join(data_dir, name))
    
    if loc == 'start':
        loc = freq[0] + 20
    
    def get_vals(sig_loc):
        if is_decay:
            sig_std = v_virial*sig_loc/(c*np.sqrt(3))
            sig_amp_ratio = 2.75e23*signal_size / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
            col_name = dm_profile
            return (sig_std, sig_amp_ratio, col_name)
        else:
            sig_std = v_virial*sig_loc/(c*np.sqrt(6))
            sig_amp_ratio = 3.55e28*signal_size / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
            col_name = dm_profile+'_sq'
            return (sig_std, sig_amp_ratio, col_name)
    targ_info = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/all_xband_info.csv').set_index('source')

    if template:
        spec = np.ones_like(spec)
    spec_med = np.median(spec)
    theta = targ_info.loc[name[:-4], 'theta']

    loc_loop = loc
    end = freq[freq.size-20]
    # salted = inject(freq, spec, theta, loc_loop, signal_size, v_earth, v_virial, is_decay, name, targ_info, amp, sig_std)
    while loc_loop <= end:
        sig_std, sig_amp_ratio, col_name = get_vals(loc_loop)
        amp_ratio = sig_amp_ratio*targ_info.loc[name[:-4], col_name]
        amp = spec_med*amp_ratio
        salted = inject(freq, spec, theta, loc_loop, signal_size, v_earth, v_virial, is_decay, name, targ_info, amp, sig_std)
        spec = salted
        loc_loop = loc_loop + spacing
    injected = np.array([freq, spec])
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, name), injected)
    return injected

def normalize_dynamic(data, data_dir=None, save_dir=None, window_factor=0.02, freq=None, width=200, nsigma=2, order=5, is_decay=True):
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
    
    new_xs = []
    normalized_spectrum = []
    
    def round_up_to_nearest_odd(number):
        ceiled_number = math.ceil(number)
        return int(ceiled_number + 1) if ceiled_number % 2 == 0 else int(ceiled_number)
    
    # Loop through the data points and divide each point by the polynomial 
    for i, x in enumerate(xs):
        window = round_up_to_nearest_odd(window_factor * x)
        if i < (window//2):
            continue
        if i > len(xs) - (window//2 + 1):
            break
        ind_xs = np.arange(-(window//2), window//2+1)
        #print(f'{i}/{len(xs)-upper-lower}')
        current_ys = ys_mean[i-window//2:i+window//2+1]
        idx = np.isfinite(current_ys)
        
        # Center bounds
        lower_bound = -(window//2) * freq_diff
        upper_bound = (window//2) * freq_diff
        guesses = np.ones_like(range(order+1)).tolist()
        bounds = (
            (0, lower_bound, *[-np.inf for _ in guesses]),
            (np.inf, upper_bound, *[np.inf for _ in guesses])
        )
        
        # Fit the defined function to the data using curve fitting
        fit_func_wrapper = partial(fit_func, window_center = x)
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
        sig = nsigma * get_sigma(x + fit_center_)/freq_diff # 3 sigma 
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
        new_xs.append(x)
        normalized_spectrum.append(pg/p)
    
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, data), normalized)
    return normalized

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--injected", help="normalize the injected data",
                        action="store_true")
    args = parser.parse_args()
    
    num_processes = mp.cpu_count()
    print(num_processes)
    with mp.Pool() as p:
        banks = ['0', '1', '2', '3']
        for bank in banks:
            cleaned_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/{bank}/preprocessed_despiked'
            spectra = os.listdir(cleaned_dir) #injected_dir

            injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/{bank}/injected_pg_dynamic_27'
            normalized_uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/{bank}/normalized_uninjected_pg_dynamic_27'
            normalized_injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/xband_071025/spliced/{bank}/normalized_injected_pg_dynamic_27'
            
            inject_func = partial(
                inject_spaced_one,
                data_dir=cleaned_dir,
                save_dir=injected_dir,
                signal_size=1e-27,
                loc='start',
                spacing=100,
            )
            
            in_dir = injected_dir if args.injected else cleaned_dir
            out_dir = normalized_injected_dir if args.injected else normalized_uninjected_dir
            
            normalize_func = partial(
                normalize_dynamic,
                data_dir=in_dir,
                save_dir=out_dir,
                nsigma=1.5,
                window_factor=0.0275,
                # window=161,
                # freq=11080,
                # width=70,
                order=5
            )
            if args.injected:
                print(f'Injecting and normalizing bank {bank}')
                injected = list(tqdm(p.imap_unordered(inject_func, spectra), total=len(spectra)))
            else:
                print(f'Normalizing bank {bank}')
            normalized = list(tqdm(p.imap_unordered(normalize_func, spectra), total=len(spectra)))