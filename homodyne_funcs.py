# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab. Homodyne system
# homodyne_funcs.py
# Set of functions to analyse homodyne data.
#
# Marcial Becerril, @ 12 April 2024
# Latest Revision: 12 Apr 2024, 12:50 GMT-6
#
# TODO list:
# Functions missing:
#   + Visualizer tool
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import os
import time
import sys

import numpy as np
from tqdm import tqdm
from os import walk

from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter, freqz
from scipy import signal
from scipy import optimize
from scipy.optimize import curve_fit

from astropy.io import fits

from matplotlib.pyplot import *

sys.path.append('../')
from misc.msg_custom import *
from data_processing import *
from fit_psd import *


def get_df(I, Q, didf, dqdf, I0, Q0):
    """
    Get the df thrpugh the magic formula to get 'df'.
    Parameters
    ----------
    Input:
        I/Q : array
            I/Q timestreams.
        I0/Q0 : float
            I0/Q0 at the resonance frequency.
        didf/dqdf : float
            I/Q sweep gradient.
    Output:
        df : array
            Resonance frequency shift.
        dIQ : array
            Gradient magitude.
    ----------
    """

    dIQ = didf**2 + dqdf**2
    df = [ ( ((I[i] - I0)*didf) + ((Q[i] - Q0)*dqdf) ) / dIQ for i in range(len(I)) ]

    return df, dIQ

def get_psd(df, fs, method='mean'):
    """
    Compute the PSD.
    Parameters
    ----------
    df : array
        Fractional frequency [Hz].
    fs : float
        Samplig frequency [Hz].
    method : string
        PSD method.
    ----------
    """

    psd = [signal.periodogram(df[i], fs)[1]  for i in range(len(df))]
    freqs = signal.periodogram(df[0], fs)[0]

    if method == 'mean':
        total_psd = np.average(psd, axis=0)
    elif method == 'median':
        total_psd = np.median(psd, axis=0)

    return freqs[2:], total_psd[2:]

def fit_mix_psd(f, psd_mix, f0, Qr, trim_range=[0.2, 9e4], plot_name="", n_pts=500, smooth_params=[21, 3]):
    """
    Get the pixed PSD.
    Parameters
    ----------
    f : array
        Frequency [Hz].
    psd_on : array 
        PSD on resonance.
    psd_off : array
        PSD off resonance [System noise].
    f0 : float
        Resonance frequency.
    Qr : float
        Total quality factor.
    trim_range[opt] : list
        Useful frequencies to work on.
    plot_name[opt] : string
        Plot name of the PSD.
    ----------
    """

    # Create fit object
    fit_psd_obj = fit_psd(plot_name=plot_name, n_pts=n_pts, smooth_params=smooth_params)

    ioff()

    psd_clean = psd_mix[np.where(f>trim_range[0])[0][0]:np.where(f>trim_range[1])[0][0]]
    psd_trim = psd_clean[psd_clean>0]

    fm = f[np.where(f>trim_range[0])[0][0]:np.where(f>trim_range[1])[0][0]][psd_clean>0]
    fm = np.array(fm)

    try:
        print('I N I T   P A R A M S')
        print('---------------------')
        msg(f'Qr: {Qr:.0f}', 'info')
        msg(f'f0: {f0:.0f} Hz', 'info')
        #msg(f'amp [not used]: {amp_noise:.3f} [Hz^2/Hz]', 'info')

        # P E R F O R M   T H E   F I T
        # -----------------------------
        fit_psd_obj.apply_psd_fit(fm, psd_trim, f0, Qr, inter=True)

        ion()

        # F I T    R E S U L T S
        # -----------------------------
        fit_single_psd = {}
        fit_single_psd['gr'] = fit_psd_obj.gr_noise
        fit_single_psd['tau'] = fit_psd_obj.tau
        fit_single_psd['tls_a'] = fit_psd_obj.tls_a
        fit_single_psd['tls_b'] = fit_psd_obj.tls_b
        fit_single_psd['amp_noise'] = fit_psd_obj.amp_noise

        print('F I T   R E S U L T S')
        msg(f'tau: {fit_psd_obj.tau*1e6:.1f} us', 'info')
        msg(f'GR noise: {fit_psd_obj.gr_noise:.3f} Hz^2/Hz', 'info')
        msg(f'tls_b: {fit_psd_obj.tls_b:.3f}', 'info')
        msg(f'tls_a: {fit_psd_obj.tls_a:.3f}', 'info')
        msg(f'amp_noise: {fit_psd_obj.amp_noise:.3f}', 'info')

        # Get the 1/f knee
        # Generation-Recombination noise
        gr = gr_noise(fm, fit_psd_obj.gr_noise, fit_psd_obj.tau, Qr, f0)
        # TLS noise
        tls = tls_noise(fm, fit_psd_obj.tls_a, fit_psd_obj.tls_b, fit_psd_obj.tau, Qr, f0)

        # Select frequencies below 1kHz.
        gr = gr[fm < 1000]
        tls = tls[fm < 1000]
        # Get GR-TLS intersection
        f_knee = fm[np.argmin(np.abs(gr-tls))]
        
        msg(f'1/f knee: {f_knee:.3f} Hz', 'info')

        return fit_single_psd, f_knee

    except Exception as e:
        print(e)
        return None, None

def get_didf_dqdf(freqs, I, Q, f0, **kwargs):
    """
    Get the gradients didf/dqdf.
    Parameters
    ----------
    Input:
        f : array
            Frequency sweep.
        I/Q : array
            I/Q from the sweep.
        f0 : float
            Resonance frequency.
    Output:
        I0/Q0 : float
            I/Q at the resonance frequency
        didf/dqdf : float
            I/Q gradients.
    ----------
    """
    # Key arguments
    # ----------------------------------------------
    # Smooth number of points
    n_smooth = kwargs.pop('n_smooth', 51)
    # Smooth order
    order = kwargs.pop('order', 3)
    # ----------------------------------------------

    # f0 index
    f0_ind = np.argmin(np.abs(freqs - (f0)))

    f0 = freqs[f0_ind]
    I0 = I[f0_ind]
    Q0 = Q[f0_ind]

    # Smooth the sweep
    I = savgol_filter(I, n_smooth, order)
    Q = savgol_filter(Q, n_smooth, order)

    didf = np.gradient(I, freqs)[f0_ind]
    dqdf = np.gradient(Q, freqs)[f0_ind]

    aI = I[f0_ind]-didf*freqs[f0_ind]
    aQ = Q[f0_ind]-dqdf*freqs[f0_ind]

    yi = didf*freqs + aI
    yq = dqdf*freqs + aQ

    return I0, Q0, didf, dqdf

def get_homodyne_filenames(diry):
    """
    Get timestream files.
    Parameters
    ----------
    Input:
        diry : string
            Homodyne directory.
    Output:
        sweep_path : string
            Sweep filename.
        sweep_hr_path :
            High resolution sweep filename.
        ON_path :
            ON resonance timestreams filename.
            High/Low resolution.
        OFF_path :
            OFF resonance timestreams filename.
            High/Low resolution.
    ----------
    """
    files = os.listdir(diry)

    sweep_path = ""
    sweep_hr_path = ""
    ON_path = []
    OFF_path = []

    for i in files:
        k = i.lower()
        if k == "sweep.fits":
            sweep_path = i
        elif k == "sweep_hr.fits":
            sweep_hr_path = i
        else:
            freq = ''
            mode = ''
            n = ''
            cnt = 0
            for j in k:
                if j == '_':
                    cnt += 1
                elif j == '.':
                    cnt = 0
                elif cnt == 1:
                    freq = freq + j
                elif cnt == 3:
                    mode = mode + j
                elif cnt == 6:
                    n = n + j
            if n == '':
                n = 0
            if mode == "on":
                ON_path.append([i,int(freq),int(n)])
            elif mode == "off":
                OFF_path.append([i,int(freq),int(n)])

    return sweep_path, sweep_hr_path, ON_path, OFF_path

def get_homodyne_data(directory, avoid=[[], []]):
    """
    Get homodyne data.
    Parameters
    ----------
    Input:
        directory : string
            Get homodyne data.
    ----------
    """
    # Get filenames
    sweep_path, sweep_hr_path, timestream_on, timestream_off = get_homodyne_filenames(directory)

    # Get sweep: header and data
    sweep_fits = fits.getdata(os.path.join(directory, sweep_path))
    sweep_hdul = fits.open(os.path.join(directory, sweep_path))
    sweep_hdr = sweep_hdul[1].header

    f_s21 = sweep_fits.field(0)
    sweep_I = sweep_fits.field(1)
    sweep_Q = sweep_fits.field(2)

    s21 = sweep_I + 1j*sweep_Q

    # Get the PSD
    (I_low_on, Q_low_on), (I_high_on, Q_high_on), (ts_low_on, ts_high_on), (hr_low_on, hr_high_on) = get_noise(directory, timestream_on, avoid=avoid)
    (I_low_off, Q_low_off), (I_high_off, Q_high_off), (ts_low_off, ts_high_off), (hr_low_off, hr_high_off) = get_noise(directory, timestream_off, avoid=avoid)

    return (f_s21, s21, sweep_hdr), (ts_low_on, I_low_on, Q_low_on, hr_low_on), (ts_high_on, I_high_on, Q_high_on, hr_high_on), \
    (ts_low_off, I_low_off, Q_low_off, hr_low_off), (ts_high_off, I_high_off, Q_high_off, hr_high_off)

# Mix high and low PSD data
def mix_psd(freqs, psd, xp=800):
    # Sort the frequencies in the list
    idx_one = []
    for i in range(len(freqs)):
        idx_one.append(freqs[i][-1])

    sort_list = list(np.argsort(idx_one))
    # Arrays
    full_freq = []
    full_psd = []
    # Get intersections
    # Frequencies
    ff0 = freqs[sort_list.index(0)]
    ff1 = freqs[sort_list.index(1)]
    # PSD
    pf0 = psd[sort_list.index(0)]
    pf1 = psd[sort_list.index(1)]

    if xp is None:
        xp = ff0[-1]

    xp_idx0 = np.where(ff0>=xp)[0][0]
    xp_idx1 = np.where(ff1>=xp)[0][0]

    full_freq = np.concatenate((ff0[:xp_idx0], ff1[xp_idx1:]))
    full_psd = np.concatenate((pf0[:xp_idx0], pf1[xp_idx1:]))

    return full_freq, full_psd

def get_noise(diry, files, avoid=[[], []]):
    """
    Get noise data from high/low frequency sampling rate.
    Parameters
    ----------
    diry : string
        Data directory.
    files : array
        Timestream high/low frequency filenames.
    avoid [opt] : list
        List the index number of the avoided timestreams.
    ----------
    """

    # Get fil indices
    idx_file = []
    for idx in files:
        idx_file.append(idx[1])

    # High-frequency
    high_path = os.path.join(diry, files[np.argmax(idx_file)][0])
    ts_high, I_high, Q_high, hr_high = get_noise_from_single_file(high_path, avoid=avoid[1])
    # Low-frequency
    low_path = os.path.join(diry, files[np.argmin(idx_file)][0])
    ts_low, I_low, Q_low, hr_low = get_noise_from_single_file(low_path, avoid=avoid[0])

    return (I_low, Q_low), (I_high, Q_high), (ts_low, ts_high), (hr_low, hr_high)

def get_noise_from_single_file(path, deglitch=False, avoid=[[0, 1], []], **kwargs):
    """
    Get noise data from a filename.
    Parameters
    ----------
    path : string
        Noise filename.
    deglitch[opt] : bool
        Apply deglitching filter
    avoid[opt] : list
        List of timestreams to avoid.
    ----------
    """
    # Key arguments
    # ----------------------------------------------
    # Deglitching window size.
    win_size = kwargs.pop('win_size', 350)
    # Sigma threshold
    sigma_thresh = kwargs.pop('sigma_thresh', 3.5)
    # Number of points to define a peak
    peak_pts = kwargs.pop('peak_pts', 4)
    # Number of points to define a peak
    verbose = kwargs.pop('verbose', False)
    # ----------------------------------------------

    # Get timestream data
    data_fits = fits.getdata(path)

    # Get timestream header
    hdul = fits.open(path)
    hdr = hdul[1].header
    hdul.close()

    fs = hdr['SAMPLERA']

    # Load timestream data
    I = [data_fits.field(2*i) for i in range(int(len(data_fits[0])/2))]
    Q = [data_fits.field(2*i+1) for i in range(int(len(data_fits[0])/2))]

    # Deglitch
    Id, Qd = [], []
    for i in range(len(I)):
        if not i in avoid:
            if deglitch:
                i_t, c1 = cr_filter(I[i], win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                q_t, c2 = cr_filter(Q[i], win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
            else:
                i_t = I[i]
                q_t = Q[i]

            Id.append(i_t)
            Qd.append(q_t)

        #print('Reading...', i)
    #print('All read')

    tm = np.arange(0, (1/fs)*len(i_t), 1/fs)

    return tm, Id, Qd, hdr
