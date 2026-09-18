import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from scipy.optimize import curve_fit
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, make_interp_spline
import matplotlib as mpl
import datetime
from dateutil.relativedelta import relativedelta
import scipy
import itertools

from normalization_methods import *

num_processes = 20#mp.cpu_count()

info_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/all_cband_info.csv'
info = pd.read_csv(info_dir)

c = 299792
v_virial = 250 # km/s
v_earth  = 225 # km/s
dm_profile = 'nfw'

def sigma_filter(data, window_size=15):
    for i in range(window_size//2, len(data)-window_size//2):
        window_data = data[i-window_size//2:i+window_size//2+1]
        median = np.median(window_data)
        std = np.std(window_data)
        if abs(data[i] - median) > 3 * std:
            data[i] = median
    return data

def build_asymmetry(data_dir, angle, lower, upper):
    spectra = os.listdir(data_dir)
    inner = info[info[angle] < lower]
    outer = info[info[angle] > upper]
    inner_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in inner['source'].tolist()])
    outer_spectra = np.array([np.load(os.path.join(data_dir, file))[1] for file in spectra if file[:-4] in outer['source'].tolist()])
    inner_mean = np.mean(inner_spectra, axis=0)
    outer_mean = np.mean(outer_spectra, axis=0)
    asymmetry = (inner_mean - outer_mean) / (inner_mean + outer_mean)
    xs = np.load(os.path.join(data_dir, spectra[0]))[0]
    return xs, asymmetry

def correlate_asymmetry(freqs, spectrum, template):
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
        stretched = stretch_template(template, freqs[i] / ref_freq)
        
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
        
        a = get_a(window, stretched) 
        a_vals[i] = a 
        residuals[i] = np.sqrt(np.mean((a*stretched-window)**2)) 

    return a_vals / (residuals + 1e-12)

def get_pvals_cont(inj, uninj, width=140):
    stds = []
    for i in range(len(uninj)):
        if i < width:
            stds.append(np.std(uninj[:i+width]))
        elif i >= len(uninj) - width:
            stds.append(np.std(uninj[i-width:]))
        else:
            stds.append(np.std(uninj[i-width:i+width]))
    
    sigmas = inj / stds
    pvals = list(map(lambda sigma: .5*math.erfc(sigma/np.sqrt(2)), sigmas))
    return pvals

def plot_asymmetry(ax, xs, uninj_ys, inj_ys, snr, b, exterior):
    ax.plot(xs, inj_ys, color='r', label='injected')
    ax.plot(xs, uninj_ys, color='k', label='raw')
    
    ax_inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    ax_inset.hist(uninj_ys, density=True, color='gray', bins=20)
    mean, std = np.mean(uninj_ys), np.std(uninj_ys)
    xmin, xmax = ax_inset.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mean, std)
    ax_inset.plot(x, p, c='k')
    ax_inset.set_yticks([])
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    
    ax.text(0.03, 0.97, f'SNR = {snr:.2f}\n$\\sigma$ = {f"{std:.3g}"}',
            transform=ax.transAxes, ha='left', va='top', fontsize=16)

    ax.legend(loc='lower left')

    ax.set_xlabel('$\\nu$ [MHz]', fontsize=16)
    ax.set_ylabel('Asymmetry', fontsize=16)
    ax.set_title(f'b={b}, exterior={exterior}', fontsize=18)

def plot_pvals(ax, xs, pvals_comb, freq, b, exterior):      
    i = 0
    while .5*math.erfc((i+1)/np.sqrt(2)) > min(pvals_comb):
        label = r'$n\sigma$' if not i else None
        ax.axhline(.5*math.erfc((i+1)/np.sqrt(2)), c='red', ls='--', lw=0.5, label=label)
        i += 1
            
    ax.plot(xs, pvals_comb, c='k')
    
    idx = np.argmin(np.abs(freq - xs))
    pval_idx = np.argmin(pvals_comb[idx-20:idx+20])
    x = xs[idx+pval_idx-20]
    pval = np.min(pvals_comb[idx-20:idx+20])
    ax.scatter(x, pval, c='blue', s=100, marker='x', label=label)
    
    ax.set_yscale('log')
    ax.axvspan(freq-3, freq+3, color='gray', alpha=0.3, linewidth=0.5)

    ax.set_ylim(10 ** (math.floor(np.log10(min(pvals_comb[pvals_comb != 0])))), 1)

    log_pval = -np.log10(pval)
    ax.text(0.03, 0.03, f'$-\\log(p)$ = {log_pval:.2f}',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=16)

    ax.set_xlabel(r'$\nu$ [MHz]', fontsize=18)
    ax.set_ylabel('$p$', fontsize=18)
    ax.set_title(f'b={b}, exterior={exterior}', fontsize=20)
    ax.legend(loc='lower right')

def calculate_snr(
        normalize_func,
        uninjected_dir=None,
        injected_dir=None,
        uninj_save_dir=None,
        inj_save_dir=None,
        fit_window=None,
        fit_type='quadratic',
        freq=8720,
        plot=True,
        dir_name=None,
        bank=0,
        **kwargs
    ):
    spectra = os.listdir(injected_dir)
    
    with mp.Pool(processes=num_processes) as p:
        for data_dir, save_dir in ((uninjected_dir, uninj_save_dir), (injected_dir, inj_save_dir)):
            normalize_func_partial = partial(normalize_func, data_dir=data_dir, save_dir=save_dir, freq=freq, **kwargs)
            _ = list(p.imap_unordered(normalize_func_partial, spectra))
    
    #print('Building asymmetries')
    xs, uninj_ys = build_asymmetry(uninj_save_dir, 'phi', *phi_cutoffs)
    xs, inj_ys = build_asymmetry(inj_save_dir, 'phi', *phi_cutoffs)
    xs, uninj_ys_dopp = build_asymmetry(uninj_save_dir, 'theta', *theta_cutoffs)
    xs, inj_ys_dopp = build_asymmetry(inj_save_dir, 'theta', *theta_cutoffs)

    # Save the intensity asymmetries
    b = round(kwargs.get('b'), 3)
    exterior = round(kwargs.get('exterior'), 3)
    asym_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/asymmetries/{bank}/{freq}'
    os.makedirs(asym_dir, exist_ok=True)
    np.save(os.path.join(asym_dir, f'b_{b}_ext_{exterior}_uninj'), np.array([xs, uninj_ys]))
    np.save(os.path.join(asym_dir, f'b_{b}_ext_{exterior}_inj'), np.array([xs, inj_ys]))

    #print('Calculating SNR')
    noise_window = 3 * fit_window
    # Calculate noise
    mask = abs(xs - freq) < noise_window
    window = uninj_ys[mask]
    noise = np.std(window)
        
    # Calculate signal amplitude
    mask = abs(xs - freq) < fit_window
    window = [xs[mask], inj_ys[mask]]
    bounds = [min(window[0]), max(window[0])]
    fit = None
    
    if fit_type == 'quadratic':
        params = np.polyfit(*window, 2) #quadratic for intensity
        fit = np.poly1d(params)(window[0])
        signal = np.max(fit)# - np.mean(window)
    elif fit_type == 'gaussian':
        # Trying a Gaussian instead of quadratic. 7/19/25
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / (2 * stddev))**2)

        try:
            params, _ = curve_fit(
                gaussian,
                *window,
                bounds=((0, bounds[0], 0),(np.inf, bounds[1], np.inf)))
            print(params)
            fit = gaussian(window[0], *params)
            signal = np.max(fit)# - np.mean(window)
        except:
            print('Gaussian fit failed...')
            signal = 0
    elif fit_type == 'savgol':
        #Savgol filter...
        filt_window = savgol_filter(window[1], 30, 4)
        signal = np.max(filt_window)
    else:
        signal = np.max(window[1])
    
    if signal < 0:
        signal = 0
    
    # CALCULATE!
    snr = signal / noise

    if plot:
        fig, ax = plt.subplots()

        plot_asymmetry(ax, xs, uninj_ys, inj_ys, snr, b, exterior)

        save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/{freq}/asymmetries'
        os.makedirs(save_path, exist_ok=True)
        
        fig.savefig(os.path.join(save_path, f'b_{b}_ext_{exterior}.png'), dpi=300, bbox_inches='tight')
    
    # Load in templates
    temp_dopp = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b}_ext_{exterior}/doppler_template.npy')[1]
    temp_int = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b}_ext_{exterior}/intensity_template.npy')[1]

    # Calculate FOM spectra
    corr_uninj_dopp = correlate_asymmetry(xs, uninj_ys_dopp, temp_dopp)
    corr_uninj_int = correlate_asymmetry(xs, uninj_ys, temp_int)
    corr_inj_dopp = correlate_asymmetry(xs, inj_ys_dopp, temp_dopp)
    corr_inj_int = correlate_asymmetry(xs, inj_ys, temp_int)
    
    # Calculate p-values
    pvals_dopp = get_pvals_cont(corr_inj_dopp, corr_uninj_dopp, width=300)
    pvals_int = get_pvals_cont(corr_inj_int, corr_uninj_int, width=300)
    pvals_comb = np.multiply(pvals_dopp, pvals_int)

    # Save p-values
    pval_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/pvals/{bank}/{freq}'
    os.makedirs(pval_dir, exist_ok=True)
    np.save(os.path.join(pval_dir, f'b_{b}_ext_{exterior}'), np.array([xs, pvals_comb]))

    # Calculate injection p-value
    idx = np.argmin(np.abs(freq - xs))
    window = pvals_comb[idx-20:idx+20]
    window_nonzero = window[window != 0]
    pval = np.min(window_nonzero)

    if plot:
        fig, ax = plt.subplots()

        plot_pvals(ax, xs, pvals_comb, freq, b, exterior)

        save_path = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/{freq}/pvals'
        os.makedirs(save_path, exist_ok=True)

        fig.savefig(os.path.join(save_path, f'b_{b}_ext_{exterior}.png'), dpi=300, bbox_inches='tight')

    return snr, noise, pval

def normalize_number(value, desired_min, desired_max, original_min=0, original_max=1):
    return(value - original_min) * (desired_max - desired_min) / (original_max - original_min) + desired_min

def objective_weighted(b, exterior, **kwargs):
    return calculate_snr(
        normalize_weighted, #normalize
        fit_window=10,
        width=60,
        b=b,
        exterior=exterior, #window=window
        **kwargs
    )

if __name__ == '__main__':
    is_decay = True
    ref_signal_size = 1e-28 if is_decay else 5e-25
    ref_freq = 5800
    power = 3 if is_decay else 4
    
    targets = []
    ddir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/'
    with open(os.path.join(ddir, 'snr_targets_200.txt'), 'r') as file:
        for line in file:
            targets.append(line.strip())
    print(f'Targets: {targets}')
    
    case = 'decay' if is_decay else 'ann'

    theta_cutoffs = (65, 65)
    phi_cutoffs = (70, 115)
    
    order = 5
    #orders = np.arange(3, 8)
    # maybe consider this later, but order 5 has been the best in my experience, so start with that. 6/5/26

    bs = np.array([0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75])
    exts = np.arange(2, 8.5, 0.5) #(2, 5.5, 0.5)
    grid_params = list(itertools.product(bs, exts)) #b, exterior

    xx, yy = np.meshgrid(bs, exts, indexing='ij')

    #for order in orders:
    dir_name = f'snr_evals_cband/grid_weighted_1e28_{case}_bank_3'

    start_time = datetime.datetime.now()

    # Form templates for each choice of parameters
    bank_1_dir = '/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/1'
    template_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/injected'

    targs = [targ + '.npy' for targ in targets if targ + '.npy' in os.listdir(bank_1_dir)]
    
    for targ in targs:
        # Injections with reference size at 5800 MHz
        inject_calibrated_cband(
            targ,
            ref_freq,
            info,
            data_dir=bank_1_dir,
            save_dir=template_dir,
            signal_sizes=ref_signal_size,
            is_decay=is_decay,
            template=True
        )

    for b, exterior in tqdm(grid_params, desc='Forming templates'):
        temp_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/normalized/b_{b}_ext_{exterior}'
        final_temp_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b}_ext_{exterior}'

        if os.path.exists(final_temp_dir):
            continue
        else:
            with mp.Pool(processes=num_processes) as p:
                normalize_func_partial = partial(
                    normalize_weighted,
                    data_dir=template_dir,
                    save_dir=temp_save_dir,
                    freq=ref_freq,
                    width=40,
                    b=b,
                    exterior=exterior,
                    is_decay=is_decay
                )
                _ = list(p.imap_unordered(normalize_func_partial, targs))

            xs, temp_dopp = build_asymmetry(temp_save_dir, 'theta', *theta_cutoffs)
            xs, temp_int = build_asymmetry(temp_save_dir, 'phi', *phi_cutoffs)

            # Smoothing the templates with a spline (Aya)
            data_dopp = [xs[::15], temp_dopp[::15]]
            data_int = [xs[::15], temp_int[::15]]
            temp_dopp = make_interp_spline(*data_dopp, k=3)(xs)
            temp_int = make_interp_spline(*data_int, k=3)(xs)
            
            # Save the asymmetries
            os.makedirs(final_temp_dir, exist_ok=True)
            save_dopp = np.array([xs, temp_dopp])
            np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b}_ext_{exterior}/doppler_template.npy', save_dopp)
            save_int = np.array([xs, temp_int])
            np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/templates/asymmetries/b_{b}_ext_{exterior}/intensity_template.npy', save_int)

            # Plot for good measure
            plot_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/templates'
            os.makedirs(plot_dir, exist_ok=True)

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].plot(*save_dopp, c='k')
            axs[0].axvline(ref_freq, c='r', ls='--')
            axs[0].set_xlabel(r'$\nu$ [MHz]', fontsize=16)
            axs[0].set_ylabel('Asymmetry', fontsize=16)
            axs[0].set_title('Doppler', fontsize=16)

            axs[1].plot(*save_int, c='k')
            axs[1].axvline(ref_freq, c='r', ls='--')
            axs[1].set_title('Intensity', fontsize=16)

            fig.suptitle(f'b={b}, exterior={exterior}', fontsize=18)

            fig.savefig(os.path.join(plot_dir, f'b_{b}_ext_{exterior}.png'), dpi=300, bbox_inches='tight')

    #for bank in range(4):
    for bank in [3]:
        bank_start_time = datetime.datetime.now()
        print(f'Bank {bank} start!')

        uninjected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/data/preprocessed/{bank}'
        injected_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/injected/{bank}'
        uninj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/normalized_uninjected/{bank}'
        inj_save_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/normalized_injected/{bank}'

        # Get the targets
        targs = [targ for targ in targets if targ + '.npy' in os.listdir(uninjected_dir)]
        print(f'# of targets in bank {bank}: {len(targs)}')
        targ_info = info[info['source'].isin(targs)]
        inner_targs = targ_info[targ_info['phi'] < 70]
        outer_targs = targ_info[targ_info['phi'] > 115]
        print(f'# of inner targets: {len(inner_targs)}')
        print(f'# of outer targets: {len(outer_targs)}')
        
        xs = np.load(os.path.join(uninjected_dir, os.listdir(uninjected_dir)[0]))[0]
        start = int(round(xs[0] + 200, -2)) # Round to nearest 100
        stop = int(round(xs[-1] - 200, -2))

        freqs = np.arange(start, stop, 100)
        print(f'Injection frequencies (n={len(freqs)}): {freqs}')

        for freq in freqs:
            print(f'Evaluating {freq} MHz (bank {bank})')
            freq_start_time = datetime.datetime.now()
            #print(f'Evaluating order {order}')
            
            # Inject the signals
            for targ in targs:
                signal_size = ref_signal_size * (freq / ref_freq)**power
                inject_calibrated_cband(
                    targ + '.npy',
                    freq,
                    info,
                    data_dir=uninjected_dir,
                    save_dir=injected_dir,
                    signal_sizes=signal_size,
                    is_decay=is_decay
                )
            
            # Freeze the keywords
            objective_func = partial(
                objective_weighted, #objective
                uninjected_dir=uninjected_dir,
                injected_dir=injected_dir,
                uninj_save_dir=uninj_save_dir,
                inj_save_dir=inj_save_dir,
                fit_type=None,
                dir_name=dir_name,
                bank=bank,
                freq=freq,
                order=order,
                is_decay=is_decay
            )
            
            if os.path.exists(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/{freq}/results.png'):
                snr_grid = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/{freq}/snrs.npy')
                std_grid = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/{freq}/stds.npy')
                pval_grid = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/{freq}/pvals.npy')
            else:
                # Compute objective function over the grid
                result = list(tqdm(itertools.starmap(objective_func, grid_params), total=len(grid_params)))
                # Unpack the result
                snrs, stds, pvals = [np.array(x) for x in zip(*result)]

                # Save the results
                snr_grid = snrs.reshape(xx.shape)
                std_grid = stds.reshape(xx.shape)
                pval_grid = -np.log10(pvals.reshape(xx.shape))

                os.makedirs(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/{freq}/', exist_ok=True)
                np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/{freq}/snrs.npy', snr_grid)
                np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/{freq}/stds.npy', std_grid)
                np.save(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/{freq}/pvals.npy', pval_grid)

            # Plot the results
            fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(15, 6))
            grids = [snr_grid, std_grid, pval_grid]
            labels = ['SNR', r'$\sigma$', r'-$\log(p)$']

            for ax, grid, label in zip(axs, grids, labels):
                pcm = ax.pcolormesh(xx, yy, grid, cmap='viridis')
                cbar = fig.colorbar(pcm, ax=ax, location='top')
                cbar.set_label(label, fontsize=16)
                if label == 'SNR':
                    ax.set_xlabel('$b$', fontsize=16)
                    ax.set_ylabel('exterior', fontsize=16)
            
            fig.suptitle(f'$\\nu$ = {freq} MHz', fontsize=18)
            fig.savefig(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/{freq}/results.png', dpi=300, bbox_inches='tight')

            # asym_fig, asym_axs = plt.subplots(len(exts), len(bs), layout='constrained', figsize=(8*len(bs), 6*(len(exts))))
            
            # for param, snr, ax in zip(grid_params, snrs, asym_axs.flat):
            #     b, exterior = param
            #     # Intensity asymmetries
            #     xs, uninj_ys = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/asymmetries/{bank}/{freq}/b_{b}_ext_{exterior}_uninj.npy')
            #     xs, inj_ys = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/asymmetries/{bank}/{freq}/b_{b}_ext_{exterior}_inj.npy')
            #     plot_asymmetry(ax, xs, uninj_ys, inj_ys, snr, b, exterior)
            
            # asym_fig.savefig(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/{freq}/asymmetries.png', dpi=300, bbox_inches='tight')

            # pval_fig, pval_axs = plt.subplots(len(exts), len(bs), layout='constrained', figsize=(8*len(bs), 6*(len(exts))))
            
            # for param, ax in zip(grid_params, pval_axs.flat):
            #     b, exterior = param
            #     xs, pvals = np.load(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/pvals/{bank}/{freq}/b_{b}_ext_{exterior}.npy')
            #     plot_pvals(ax, xs, pvals, freq, b, exterior)
            
            # pval_fig.savefig(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/{freq}/p_values.png', dpi=300, bbox_inches='tight')

            freq_end_time = datetime.datetime.now()
            delta_freq = relativedelta(freq_end_time, freq_start_time)
            print(f'Freq {freq} done in {delta_freq.hours} hrs, {delta_freq.minutes} mins, {delta_freq.seconds} sec')
        
        # Plot and save the aggregate results (over all frequencies) for the current bank
        states_dir = f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/states/{bank}/'
        fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(15, 6))
        tails = ['snrs.npy', 'stds.npy', 'pvals.npy']
        labels = ['SNR', r'$\sigma$', r'-$\log(p)$']

        for ax, tail, label in zip(axs, tails, labels):
            grid = np.array([
                np.load(os.path.join(states_dir, str(freq), tail))
                for freq in freqs
            ])
            sum_grid = np.sum(grid, axis=0)
            pcm = ax.pcolormesh(xx, yy, sum_grid, cmap='viridis')
            cbar = fig.colorbar(pcm, ax=ax, location='top')
            cbar.set_label(label, fontsize=16)
            if label == 'SNR':
                ax.set_xlabel('$b$', fontsize=16)
                ax.set_ylabel('exterior', fontsize=16)
        
        fig.suptitle(f'Bank {bank} Results', fontsize=18)
        fig.savefig(f'/home/dataadmin/GBTData/SharedDataDirectory/cband_052726/{dir_name}/plots/{bank}/results.png', dpi=300, bbox_inches='tight')

        bank_end_time = datetime.datetime.now()
        delta_bank = relativedelta(bank_end_time, bank_start_time)
        print(f'Bank {bank} done in {delta_bank.hours} hrs, {delta_bank.minutes} mins, {delta_bank.seconds} sec')
    
    end_time = datetime.datetime.now()
    delta = relativedelta(end_time, start_time)
    print(f'Study complete in {delta.hours} hrs, {delta.minutes} mins, {delta.seconds} sec')