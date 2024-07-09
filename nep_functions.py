# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab. Homodyne system
# homodyne.py
# Class to read/process data taken by the homodyne system.
#
# Marcial Becerril, @ 26 March 2024
# Latest Revision: 26 Mar 2024, 21:10 GMT-6
#
# TODO list:
# Functions missing:
#	+ Savgol filter
# + Visualizer tool
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import time
import sys
import os

import lmfit

import numpy as np
from tqdm import tqdm
from os import walk

from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter, freqz
from scipy import signal
from scipy import optimize
from scipy.optimize import curve_fit

import scipy.signal
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit

from scipy import signal
from scipy import interpolate

from astropy.io import fits

from matplotlib.pyplot import *
rc('font', family='serif', size='18')

sys.path.append('../')
from misc.msg_custom import *

sys.path.append('../')
from data_processing import *

sys.path.append('../../Documents/SMD-Chip/')
from fit_psd import *

sys.path.append('../../Downloads/KID_Analyser-master/')
from dataRed import dataRed

# Cable delay
tau = 50e-9
# Boltzmann constant
Kb = 1.380649e-23
# Planck constant
h = 6.62607015e-34


def fitCirc(x, y):
    """
        Fit the IQ circle.
        This code belongs to Andreas Papagiorgou
    """

    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculate distances from centre (xc_1, yc_1)
    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1-R_1)**2)

    return xc_1, yc_1, R_1


def cable_delay(f, ar, ai):

    S21 = (ar+1j*ai)*np.exp(-1j*2*np.pi*f*tau)

    return S21


def resonator_model(f, fr, ar, ai, Qr, Qc, phi, non):
#def resonator_model(f, fr, Amp, Qr, Qc, phi, non):

    A = (ar + 1j*ai)*np.exp(-1j*2*np.pi*f*tau)

    # Fractional frequency shift of non-linear resonator
    y0s = Qr*(f - fr)/fr
    #y = Qr*(f - fr)/fr
    y = np.zeros_like(y0s)

    for i, y0 in enumerate(y0s):
        coeffs = [4.0, -4.0*y0, 1.0, -(y0+non)]
        y_roots = np.roots(coeffs)
        # Extract real roots only. From [ref Pete Code]
        # If all the roots are real, take the maximum
        y_single = y_roots[np.where(abs(y_roots.imag) < 1e-5)].real.max()
        y[i] = y_single

    B = 1 - (Qr/(Qc*np.exp(1j*phi))) / (1 + 2j*y )

    S21 = A*B

    return S21


def phase_angle(f, theta_0, fr, Qr):

    theta = -theta_0 + 2*np.arctan( 2*Qr*(1 - (f/fr) ) )
    return theta


class ResGaoModel(lmfit.model.Model):
    __doc__ = "resonator Gao model" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='rgm_', *args, **kwargs):
        """
            Resonator model  model
            Parameters
            ----------
            model : string
                Resonator model type: linear or non-linear
            prefix : string
                Model prefix. 'rgm_' by default
            ----------
        """
        super().__init__(resonator_model, *args, **kwargs)

        self.prefix = prefix


    def guess(self, f, data, **kwargs):
        """
            Guessing resonator parameters
            Parameters
            ----------

            ----------
        """

        ar, ai, Qr, fr, theta_0, Qc, phi = coarse_fit_res(f, data)

        # We define the boundaries
        self.set_param_hint('%sar' % self.prefix, value=ar)
        self.set_param_hint('%sai' % self.prefix, value=ai)
        #self.set_param_hint('%stau' % self.prefix, value=tau)

        #self.set_param_hint('%sAmp' % self.prefix, value=np.mean(np.abs(data)), min=0)
        #self.set_param_hint('%sC' % self.prefix, value=1.25, min=0)
        self.set_param_hint('%sQr' % self.prefix, value=Qr, min=100)
        self.set_param_hint('%sfr' % self.prefix, value=fr, min=f[0], max=f[-1])
        self.set_param_hint('%stheta_0' % self.prefix, value=theta_0, min=-20*np.pi, max=20*np.pi)
        self.set_param_hint('%sdelta' % self.prefix, value=50e3, min=0, vary=True)
        print('%sdelta' % self.prefix+'+'+'%sQr' % self.prefix)
        #self.set_param_hint('%sQc' % self.prefix, value=Qc, min=100, expr='%sdelta' % self.prefix+'+'+'%sQr' % self.prefix)
        self.set_param_hint('%sQc' % self.prefix, expr='%sdelta' % self.prefix+'+'+'%sQr' % self.prefix)
        self.set_param_hint('%sphi' % self.prefix, value=phi, min=-20*np.pi, max=20*np.pi)
        self.set_param_hint('%snon' % self.prefix, value=0.1, min=0.0, max=3.0)

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class CableDelay(lmfit.model.Model):
    __doc__ = "cable delay" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='cd_', *args, **kwargs):
        """
            Cable delay model
            Parameters
            ----------
            prefix : string
                Model prefix. 'cd_' by default
            ----------
        """
        super().__init__(cable_delay, *args, **kwargs)

        self.prefix = prefix


    def guess(self, f, data, **kwargs):
        """
            Guessing resonator parameters
            Parameters
            ----------

            ----------
        """

        # Delay
        #tau_guess = 45e-9

        # Amplitude
        ar_guess = np.mean(data.real)
        ai_guess = np.mean(data.imag)

        # We define the boundaries
        #self.set_param_hint('%stau' % self.prefix, value=tau_guess, min=35e-9, max=50e-9)
        self.set_param_hint('%sar' % self.prefix, value=ar_guess)#, min=np.min(data.real), max=np.max(data.real))
        self.set_param_hint('%sai' % self.prefix, value=ai_guess)#, min=np.min(data.imag), max=np.max(data.imag))

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class PhaseAngle(lmfit.model.Model):
    __doc__ = "phase angle" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='pa_', *args, **kwargs):
        """
            Phase Angle model
            Parameters
            ----------
            prefix : string
                Model prefix. 'pa_' by default
            ----------
        """
        super().__init__(phase_angle, *args, **kwargs)

        self.prefix = prefix


    def guess(self, f, data, **kwargs):
        """
            Guessing resonator parameters
            Parameters
            ----------

            ----------
        """

        # Total Q
        Qr_guess = 5e4
        # Resonance frequency
        fr_guess = f[int(len(f)/2)]
        # Theta delay
        theta_0_guess = 0

        # We define the boundaries
        self.set_param_hint('%sQr' % self.prefix, value=Qr_guess, min=100,)
        self.set_param_hint('%sfr' % self.prefix, value=fr_guess, min=f[0], max=f[-1])
        self.set_param_hint('%stheta_0' % self.prefix, value=theta_0_guess, min=-20*np.pi, max=20*np.pi)

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


# Fit procedure following Gao PhD thesis
def coarse_fit_res(f, data):

    I = data.real
    Q = data.imag

    s21 = I + 1j*Q

    # 1. Remove the cable delay
    cable_delay_model = CableDelay()

    f_cable = np.concatenate((f[:10], f[-10:]))
    I_cable = np.concatenate((I[:10], I[-10:]))
    Q_cable = np.concatenate((Q[:10], Q[-10:]))

    s21_cable = I_cable + 1j*Q_cable

    guess = cable_delay_model.guess(f_cable, s21_cable)
    cable_res = cable_delay_model.fit(s21_cable, params=guess, f=f_cable)
    cable_res.params.pretty_print()

    fit_s21 = cable_delay_model.eval(params=cable_res.params, f=f)
    #guess_s21 = cable_delay_model.eval(params=guess, f=f)
    #tau = cable_res.values['cd_tau']
    ar = cable_res.values['cd_ar']
    ai = cable_res.values['cd_ai']

    s21_no_cable = s21/fit_s21

    I_no_cable = s21_no_cable.real
    Q_no_cable = s21_no_cable.imag

    #I_no_cable = I - fit_s21.real
    #Q_no_cable = Q - fit_s21.imag

    # 2. Derotate
    #s21_no_cable = I_no_cable + 1j*Q_no_cable
    idx_f0 = np.argmin( np.abs(s21) )

    f0n = f[idx_f0]
    I0 = I_no_cable[idx_f0]
    Q0 = Q_no_cable[idx_f0]

    sel = np.abs(f-f0n) < 10e4
    xc, yc, r = fitCirc(I_no_cable[sel], Q_no_cable[sel])

    theta = np.arctan2(Q0-yc, I0-xc)
    I_derot = (I_no_cable-xc)*np.cos(-theta)-(Q_no_cable-yc)*np.sin(-theta)
    Q_derot = (I_no_cable-xc)*np.sin(-theta)+(Q_no_cable-yc)*np.cos(-theta)

    # 3. Fit the phase angle 400e3
    sel2 = np.abs(f-f0n) < 400e3
    phase_angle_model = PhaseAngle()

    theta = np.arctan2(Q_derot[sel2], I_derot[sel2])

    for i in range(len(theta)):
        if f[sel2][i] < f0n and theta[i] < 0:
            theta[i] = theta[i] + 2*np.pi
        elif f[sel2][i] > f0n and theta[i] > 0:
            theta[i] = theta[i] - 2*np.pi

    guess = phase_angle_model.guess(f[sel2], theta)
    phase_res = phase_angle_model.fit(theta, params=guess, f=f[sel2])
    phase_res.params.pretty_print()

    fit_theta = phase_angle_model.eval(params=phase_res.params, f=f[sel2])
    #guess_s21 = cable_delay_model.eval(params=guess, f=f)
    Qr = phase_res.values['pa_Qr']
    fr = phase_res.values['pa_fr']
    theta_0 = phase_res.values['pa_theta_0']

    # 4. Get Qc
    mag_zc = np.sqrt(xc**2 + yc**2)
    arg_zc = np.arctan2(yc, xc)

    Qc = Qr*(mag_zc + r)/(2*r)
    phi = theta_0 - arg_zc

    return ar, ai, Qr, fr, theta_0, Qc, phi


def combined_model(freqs, gr_noise, tau_qp, tls_a, tls_b, f0, Qr, amp_noise):
    # Ruido Generación-Recombinación
    gr = gr_noise/(1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)
    # Ruido TLS
    tls = tls_a*freqs**tls_b / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)
    # Ruido del amplificador
    amp = amp_noise
    # Ruido Total
    return gr + tls + amp


def fit_psd_noise(psd_freqs, psd_df, f0, Qr, amp_noise,
                gr_guess=None,  tauqp_guess=None,   tlsa_guess=None,  tlsb_guess=-1.0,
                gr_min=0,       tauqp_min=0,        tlsa_min=-np.inf, tlsb_min=-2.0,
                gr_max=np.inf,  tauqp_max=np.inf,   tlsa_max=np.inf,  tlsb_max=-0.1,
                sigma = None):


    if gr_guess is None:
        gr_guess = 0.01
    if tauqp_guess is None:
        tauqp_guess = 1./(psd_freqs.max()-psd_freqs.min())
    if tlsa_guess is None:
        tlsa_guess = psd_df[-1]
    if tlsb_guess is None:
        tlsb_guess = 1.5

    guess = np.array([gr_guess, tauqp_guess, tlsa_guess,  tlsb_guess ])
    bounds = np.array([ [gr_min, tauqp_min, tlsa_min,  tlsb_min ],
                        [gr_max, tauqp_max, tlsa_max,  tlsb_max ]])

    if sigma is None:
        sigma = (1 / abs(np.gradient(psd_freqs)))

    pval, pcov = curve_fit(combined_model, psd_freqs, psd_df, guess, bounds=bounds, sigma=sigma)
    (gr_noise,tau_qp,tls_a,tls_b) = pval

    psd_fit = combined_model(psd_freqs, gr_noise, tau_qp, tls_a, tls_b)

    return gr_noise, tau_qp, amp_noise, tls_a, tls_b, psd_fit


def get_homodyne_data(diry, kid, temp, atten, mode='BB'):
    """
        Get homodyne sweep
    """

    if mode == 'Dark':
        # Look for the file
        filename = diry+'KID_K'+str(kid).zfill(3)+'/Set_Temperature_'+str(temp).zfill(3)+'_mK/Set_Attenuation_'+str(atten)+'dB/'
    elif mode == 'BB':
        filename = diry+'KID_K'+str(kid).zfill(3)+'/Set_Temperature_'+str(temp).zfill(3)+'_K/Set_Attenuation_'+str(atten)+'dB/'

    #try:
    sweep_path, sweep_hr_path, time_on, time_off = get_homodyne_filenames(filename)

    # Get sweep: header and data
    sweep_fits = fits.getdata(os.path.join(filename, sweep_path))
    sweep_hdul = fits.open(os.path.join(filename, sweep_path))
    sweep_hdr = sweep_hdul[1].header

    freq_s21 = sweep_fits.field(0)
    sweep_I = sweep_fits.field(1)
    sweep_Q = sweep_fits.field(2)

    s21 = sweep_I + 1j*sweep_Q

    # Get the PSD
    f_on, psd_on, df_on, ts_on, hr_on_low, hr_on_high = get_noise(filename, time_on)
    f_off, psd_off, df_off, ts_off, hr_off_low, hr_off_high = get_noise(filename, time_off)

    return freq_s21, s21, f_on, psd_on, f_off, psd_off, ts_on, ts_off, df_on, df_off, sweep_hdr, hr_on_low, hr_on_high, hr_off_low, hr_off_high


def get_noise_from_single_file(path, avoid=[0, 1]):

    # Get timestream data
    data_fits = fits.getdata(path)

    # Get timestream header
    hdul = fits.open(path)
    hdr = hdul[1].header

    fs = hdr['SAMPLERA']

    # Load timestream data
    I = [data_fits.field(2*i) for i in range(int(len(data_fits[0])/2))]
    Q = [data_fits.field(2*i+1) for i in range(int(len(data_fits[0])/2))]

    # Deglitch
    Id, Qd = [], []
    for i in range(len(I)):
        if not i in avoid:
            i_t, c1 = cr_filter(I[i], win_size=350, sigma_thresh=3.5, peak_pts=4, verbose=False)
            q_t, c2 = cr_filter(Q[i], win_size=350, sigma_thresh=3.5, peak_pts=4, verbose=False)

            """
            figure()
            plot(I[i])
            plot(i_t)
            """

            Id.append(i_t)
            Qd.append(q_t)

    tm = np.arange(0, (1/fs)*len(i_t), 1/fs)

    # Apply the magic formula
    I0 = hdr['IF0']
    Q0 = hdr['QF0']

    dqdf = hdr['DQDF']
    didf = hdr['DIDF']

    df, didq_mag = get_df(Id, Qd, didf, dqdf, I0, Q0)

    print(fs)
    freq_psd, psd = get_psd(df, fs)

    return freq_psd, psd, df, (tm, Id, Qd), hdr


def get_noise(diry, files):

    idx_file = []
    for idx in files:
        idx_file.append(idx[1])

    # ----------------------------------------------------------
    # High
    high_path = os.path.join(diry, files[np.argmax(idx_file)][0])
    f_high, psd_high, df_high, ts_high, hr_high = get_noise_from_single_file(high_path)
    # Low
    low_path = os.path.join(diry, files[np.argmin(idx_file)][0])
    f_low, psd_low, df_low, ts_low, hr_low = get_noise_from_single_file(low_path)

    f, psd = mix_psd([f_low, f_high], [psd_low, psd_high])

    return f, psd, (df_low, df_high), (ts_low, ts_high), hr_low, hr_high


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


def get_psd(df, fs, join='mean'):

    psd = [signal.periodogram(df[i], fs)[1]  for i in range(len(df))]
    freqs = signal.periodogram(df[0], fs)[0]

    if join == 'mean':
        join_psd = np.average(psd, axis=0)

    return freqs[2:], join_psd[2:]


def get_df(I, Q, didf, dqdf, I0, Q0):
    """
        Get df through the magic formula.
    """
    didq_mag = (didf**2)+(dqdf**2)
    df = [ ( ((I[i]-I0)*didf) + ((Q[i]-Q0)*dqdf) ) / didq_mag for i in range(len(I)) ]

    return df, didq_mag


# Get the data of the directory
def get_homodyne_filenames(diry):

    files = os.listdir(diry)

    sweep_path = ""
    sweep_hr_path = ""
    ONTemp = []
    OFFTemp = []

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
                ONTemp.append([i,int(freq),int(n)])
            elif mode == "off":
                OFFTemp.append([i,int(freq),int(n)])

    return sweep_path, sweep_hr_path, ONTemp, OFFTemp


def get_vna_data(diry, kid, temp, atten, mode, thresh=1e5):
    """
        ONLY FOR SEGMENTED VNA.
    """
    if mode == 'BB':
        print('Mode BB')
        filename = diry + 'VNA_Sweeps/'+'S21_Segmented_BBTEMP'+str(temp).zfill(3)+'_K_POW0.0_dB_ATT'+str(atten).zfill(2)+'.0_dB.fits'
    elif mode == 'Dark':
        filename = diry + 'VNA_Sweeps/'+'S21_Segmented_BTEMP0'+str(temp).zfill(3)+'_mK_POW0.0_dB_ATT'+str(atten).zfill(2)+'.0_dB.fits'

    data_fits = fits.getdata(filename)

    hdul = fits.open(filename)
    hdr = hdul[1].header

    f = data_fits.field(0)
    I = data_fits.field(1)
    Q = data_fits.field(2)

    # Find KIDs
    n_kids = np.where(np.diff(f)>thresh)[0]
    n_kids = np.concatenate(([0], n_kids, [-1]))

    # Get detector
    from_idx = n_kids[kid]+1
    to_idx = n_kids[kid+1]
    # Get individual detector
    f_k = f[from_idx:to_idx]
    # I, Q
    I_k = I[from_idx:to_idx]
    Q_k = Q[from_idx:to_idx]
    s21_k =  I_k + 1j*Q_k

    return f_k, s21_k, hdr


def fit_resonator(f, s21, n=3.5, plot_res=True):

    # Fit resonator
    res_model = ResGaoModel()

    # Choose the region close to the detector
    guess = res_model.guess(f, s21)

    lw = guess['rgm_fr'].value/guess['rgm_Qr'].value

    # Get middle point
    I = s21.real
    Q = s21.imag
    di = np.diff(I)
    dq = np.diff(Q)
    di_dq = np.sqrt(di**2 + dq**2)
    #m_idx = np.argmax(di_dq)
    m_idx = np.argmin(np.abs(s21))

    #from_idx = guess['rgm_fr'].value-n*lw
    from_idx = f[m_idx]-n*lw
    if from_idx < f[0]:
        from_idx = 0
    else:
        from_idx = np.where(f>=from_idx)[0][0]
    #from_idx = 0

    to_idx = f[m_idx]+n*lw
    if to_idx > f[-1]:
        to_idx = len(f)
    else:
        to_idx = np.where(f>=to_idx)[0][0]

    print(from_idx, to_idx)

    guess = res_model.guess(f[from_idx:to_idx], s21[from_idx:to_idx])

    result = res_model.fit(s21[from_idx:to_idx], params=guess, f=f[from_idx:to_idx])
    fit_s21 = res_model.eval(params=result.params, f=f)

    ar = result.values['rgm_ar']
    ai = result.values['rgm_ai']
    fr = result.values['rgm_fr']
    Qr = result.values['rgm_Qr']
    Qc = result.values['rgm_Qc']
    phi = result.values['rgm_phi']
    non = result.values['rgm_non']

    Qi = Qr*Qc / (Qc - Qr)

    fit_kid = {}
    fit_kid['ar'] = ar
    fit_kid['ai'] = ai
    fit_kid['fr'] = fr
    fit_kid['Qr'] = Qr
    fit_kid['Qc'] = Qc
    fit_kid['Qi'] = Qi
    fit_kid['phi'] = phi
    fit_kid['non'] = non

    if plot_res:
        figure()

        subplots_adjust(
            top=0.98,
            bottom=0.075,
            left=0.05,
            right=0.98,
            hspace=0.2,
            wspace=0.2
        )

        subplot(1,2,1)
        plot(f[from_idx:to_idx], 20*np.log10(np.abs(s21[from_idx:to_idx])), 'b.-')
        plot(f[from_idx:to_idx], 20*np.log10(np.abs(fit_s21[from_idx:to_idx])), 'k-')
        xlabel('Frequency[Hz]')
        ylabel('dB')
        subplot(1,2,2)
        plot(s21.real[from_idx:to_idx], s21.imag[from_idx:to_idx], 'r.-')
        plot(fit_s21.real[from_idx:to_idx], fit_s21.imag[from_idx:to_idx], 'k-')
        plot(I[m_idx], Q[m_idx], 'ko')
        xlabel('I')
        ylabel('Q')

    return fit_kid


def get_nqp(N0, T, Delta):
    nqp = 2 * N0 * np.sqrt( 2*np.pi*Kb*T*Delta ) * np.exp(-(Delta/(Kb*T)))
    return nqp


def get_NEP(f, Sa, tqp, S, Qr, f0, Delta, eta=0.6):
    NEP = np.sqrt(Sa) * (( (eta*tqp/Delta)*(np.abs(S)) )**(-1)) * np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)
    return NEP


def get_BB_NEP(f, Sa, tqp, S, Qr, f0):
    NEP = np.sqrt(Sa) * ( (np.abs(S))**(-1)) * np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)
    return NEP


def derot_phase(f0, I0, Q0, fs, I_sweep, Q_sweep, I, Q, mode, f0_ref):

    # Get f0
    #idx_f0 = int(len(fs)/2)
    #f0 = fs[idx_f0]

    #I0 = I_sweep[idx_f0]
    #Q0 = Q_sweep[idx_f0]

    sel = np.abs(fs-f0) < 8e4

    xc, yc, r = fitCirc(I_sweep[sel], Q_sweep[sel])

    theta = np.arctan2(Q0-yc, I0-xc)

    I_sweep_derot = (I_sweep-xc)*np.cos(-theta)-(Q_sweep-yc)*np.sin(-theta)
    Q_sweep_derot = (I_sweep-xc)*np.sin(-theta)+(Q_sweep-yc)*np.cos(-theta)

    pha = np.arctan2(Q_sweep_derot, I_sweep_derot)
    #figure()
    #plot(fs, pha)
    for i in range(len(pha)):
        if fs[i] < f0 and pha[i] < 0:
            pha[i] = pha[i] + 2*np.pi
        elif fs[i] > f0 and pha[i] > 0:
            pha[i] = pha[i] - 2*np.pi

    f_mdl = interpolate.interp1d(pha[20:-20], fs[20:-20], fill_value='extrapolate')

    figure()
    #plot(I_sweep, Q_sweep)
    #plot(I,Q, 'r,')

    #plot( np.arctan2(Q_sweep_derot, I_sweep_derot), fs)
    plot( fs, pha)
    xlabel('Freq [Hz]')
    ylabel('Phase [rad]')
    #axhline(f0)

    I_derot = (I-xc)*np.cos(-theta) - (Q-yc)*np.sin(-theta)
    Q_derot = (I-xc)*np.sin(-theta) + (Q-yc)*np.cos(-theta)

    phase = np.arctan2(Q_derot, I_derot)

    if mode == 'off':
        for i in range(len(phase)):
            if phase[i] > 0:
                phase[i] = phase[i] - 2*np.pi

    df_derot = f_mdl(phase)-f0_ref

    """
    plot(I_sweep_derot, Q_sweep_derot)
    plot(I_derot,Q_derot, 'b,')
    """

    plot(f_mdl(phase), phase, 'r.' )

    return df_derot, phase, I_sweep_derot, Q_sweep_derot, I_derot, Q_derot


def get_temp_f0(data_dir, kids, temps, atten_vna, mode='BB'):

    all_f0 = []

    for kid in kids:

        Nqps = []
        f0s = []

        kid_f0s = []

        for T in temps:

            f_vna, s21_vna, hr_vna = get_vna_data(data_dir, kid, T, atten_vna[kid], mode) # VNA

            try:
                fit_res = np.load('DEF-K'+str(kid).zfill(3)+'-T_'+str(T)+'-A_'+str(atten_vna[kid]).zfill(3)+'.npy', allow_pickle=True).item()
            except Exception as e:
                print(e)
                fit_res = fit_resonator(f_vna, s21_vna, n=10)
                np.save('DEF-K'+str(kid).zfill(3)+'-T_'+str(T)+'-A_'+str(atten_vna[kid]).zfill(3), fit_res)

            f0 = fit_res['fr']

            kid_f0s.append(f0)

        all_f0.append(kid_f0s)

    return all_f0


def get_dF_dNqp_from_homodyne(data_dir, kids, temps, atten_vna, Delta, N0, V, plot_kid=False):

    # Get VNA sweeps
    red = dataRed()

    dF0_dNqps = []
    all_f0 = []

    for kid in kids:

        Nqps = []
        f0s = []

        for T in temps:

            f_vna, s21_vna, hr_vna = get_vna_data(data_dir, kid, T, atten_vna[kid], mode='Dark') # VNA

            try:
                fit_res = np.load('DEF-K'+str(kid).zfill(3)+'-T_'+str(T)+'-A_'+str(atten_vna[kid]).zfill(3)+'.npy', allow_pickle=True).item()
            except Exception as e:
                print(e)
                fit_res = fit_resonator(f_vna, s21_vna, n=10)
                np.save('DEF-K'+str(kid).zfill(3)+'-T_'+str(T)+'-A_'+str(atten_vna[kid]).zfill(3), fit_res)

            f0 = fit_res['fr']

            nqp = get_nqp(N0, 1e-3*T, Delta)
            Nqp = nqp * V

            Nqps.append(Nqp)
            f0s.append(f0)

        if plot_kid:
            #figure()
            #plot(Nqps, (f0s - f0s[0])/f0s[0], 'rs-')
            all_f0.append((f0s - f0s[0])/f0s[0])

        dF0_dNqp, intercept = np.polyfit(Nqps[-3:], f0s[-3:], 1)
        Nqps_fit = np.linspace(Nqps[0], Nqps[-1], 1000)

        print(f0s[-3:])

        """
        if plot_kid:
            plot(Nqps_fit, Nqps_fit*dF0_dNqp + intercept, 'k')
        """

        dF0_dNqps.append(dF0_dNqp)

    plot(Nqps, np.mean(all_f0, axis=0), 'ks-')
    errorbar(Nqps, np.mean(all_f0, axis=0), yerr=np.std(all_f0, axis=0), capsize=2, color='k')
    fill_between(Nqps, np.mean(all_f0, axis=0)+np.std(all_f0, axis=0),  np.mean(all_f0, axis=0)-np.std(all_f0, axis=0), alpha=0.5, color='blue')
    xlabel(r'N$_{qp}$')
    ylabel(r'(F$_{\theta}$(T) - F$_{\theta}$(80mK))/F$_{\theta}$(80mK)')
    title(r'Normalized frequency shift vs N$_{qp}$')

    return Nqps, dF0_dNqps


def get_dP_dNqp_from_vna(diry, kids, temps, atten_vna, Delta, N0, V, plot_iq=True):

    dP_dNqps = []
    for kid in kids:

        print('+++++++++++++++++++++++++++++++++')
        print(kid)

        f_vna, s21_vna, hr_vna = get_vna_data(diry, kid, temps[0], atten_vna[kid]) # VNA

        try:
            fit_res = np.load('DEF-K'+str(kid).zfill(3)+'-T_'+str(temps[0])+'-A_'+str(atten_vna[kid]).zfill(3)+'.npy', allow_pickle=True).item()
        except:
            fit_res = fit_resonator(f_vna, s21_vna, n=10)
            np.save('DEF-K'+str(kid).zfill(3)+'-T_'+str(temps[0])+'-A_'+str(atten_vna[kid]).zfill(3), fit_res)

        #fr = fit_res['fr']
        #f0_idx = np.where(f_vna >= fr)[0][0]

        I_sweep = s21_vna.real
        Q_sweep = s21_vna.imag

        f0_idx = np.argmax(savgol_filter( (np.diff(I_sweep)/np.diff(f_vna))**2 + (np.diff(Q_sweep)/np.diff(f_vna))**2, 31, 3))
        fr_r = f_vna_r[f0_idx_r]

        I0 = I_sweep[f0_idx]
        Q0 = Q_sweep[f0_idx]

        fr = f_vna[f0_idx]
        sel = np.abs(f_vna-fr) < 8e4

        xc, yc, r = fitCirc(I_sweep[sel], Q_sweep[sel])
        theta_ref = np.arctan2(Q0-yc, I0-xc)+np.pi

        I_sweep_derot = (I_sweep-xc)*np.cos(-theta_ref)-(Q_sweep-yc)*np.sin(-theta_ref)
        Q_sweep_derot = (I_sweep-xc)*np.sin(-theta_ref)+(Q_sweep-yc)*np.cos(-theta_ref)

        I0_derot = (I0-xc)*np.cos(-theta_ref)-(Q0-yc)*np.sin(-theta_ref)
        Q0_derot = (I0-xc)*np.sin(-theta_ref)+(Q0-yc)*np.cos(-theta_ref)

        if plot_iq:
            figure()
            plot(I_sweep_derot, Q_sweep_derot, '.-')
            plot(I0_derot, Q0_derot, 'ro')

        #temps = [140, 180, 220, 240, 260, 280, 300]#, 320, 340, 360]#, 380]
        phases = [0]

        for t in temps[1:]:

            f_vna_r, s21_vna_r, hr_vna = get_vna_data(diry, kid, t, atten_vna[kid]) # VNA

            try:
                fit_res_r = np.load('DEF-K'+str(kid).zfill(3)+'-T_'+str(t)+'-A_'+str(atten_vna[kid]).zfill(3)+'.npy', allow_pickle=True).item()
            except:
                fit_res_r = fit_resonator(f_vna_r, s21_vna_r, n=10)
                np.save('DEF-K'+str(kid).zfill(3)+'-T_'+str(t)+'-A_'+str(atten_vna[kid]).zfill(3), fit_res_r)

            fr_r = fr #fit_res['fr']
            f0_idx_r = np.where(f_vna_r >= fr_r)[0][0]

            I_sweep_r = s21_vna_r.real
            Q_sweep_r = s21_vna_r.imag

            #f0_idx_r = np.argmax(savgol_filter( (np.diff(I_sweep_r[30:-30])/np.diff(f_vna_r[30:-30]))**2 + (np.diff(Q_sweep_r[30:-30])/np.diff(f_vna_r[30:-30]))**2, 21, 3))
            #figure()
            #plot(savgol_filter( (np.diff(I_sweep_r[30:-30])/np.diff(f_vna_r[30:-30]))**2 + (np.diff(Q_sweep_r[30:-30])/np.diff(f_vna_r[30:-30]))**2, 51, 3))
            #fr_r = f_vna_r[f0_idx_r]

            I0_r = I_sweep_r[f0_idx_r]
            Q0_r = Q_sweep_r[f0_idx_r]

            sel_r = np.abs(f_vna_r-fr_r) < 8e4
            xc_r, yc_r, r_r = fitCirc(I_sweep_r[sel_r], Q_sweep_r[sel_r])

            I_sweep_derot_r = (I_sweep_r-xc)*np.cos(-theta_ref)-(Q_sweep_r-yc)*np.sin(-theta_ref)
            Q_sweep_derot_r = (I_sweep_r-xc)*np.sin(-theta_ref)+(Q_sweep_r-yc)*np.cos(-theta_ref)

            I0_derot_r = (I0_r-xc)*np.cos(-theta_ref)-(Q0_r-yc)*np.sin(-theta_ref)
            Q0_derot_r = (I0_r-xc)*np.sin(-theta_ref)+(Q0_r-yc)*np.cos(-theta_ref)

            if plot_iq:
                plot(I_sweep_derot_r, Q_sweep_derot_r, '.-')
                plot(I0_derot_r, Q0_derot_r, 'ko')

            phase = np.arctan2(Q0_derot_r, -I0_derot_r)
            if phase < 0:
                phase = 2*np.pi + phase

            phases.append(phase)

        if plot_iq:
            grid()
            xlabel('I')
            ylabel('Q')

        phases = np.array(phases)

        nqp = get_nqp(N0, 1e-3*np.array(temps), Delta)
        Nqps = V*nqp

        #figure()
        #plot(Nqps, 180*phases/np.pi, 'rs-')

        dP_dNqp, intercept = np.polyfit(Nqps[2:-2], phases[2:-2], 1)
        Nqps_fit = np.linspace(Nqps[0], Nqps[-1], 1000)

        if plot_iq:
            figure()
            title('K'+str(kid).zfill(3))
            plot(Nqps, 180*phases/np.pi, 'rs')
            plot(Nqps_fit, 180*(Nqps_fit*dP_dNqp + intercept)/np.pi, 'k')
            xlabel('Nqp')
            ylabel('phase[deg]')
            grid()

        dP_dNqps.append(dP_dNqp)
        #np.save('dP_dNqps.npy', dP_dNqps)

        return Nqps, dP_dNqps

# Derotate PSD
def get_psd_derot(I_ts, Q_ts, freq_s21, ts, hr, header_ts, mode):

    Fs = hr['SAMPLERA']
    f0_fits = header_ts["F0FOUND"]
    f0_ref = hr["SYNTHFRE"]

    I0 = header_ts["IF0"]
    Q0 = header_ts["QF0"]

    df_derots, phases = [], []
    for i in range(len(ts[1])):

        I = ts[1][i]
        Q = ts[2][i]

        df_derot, phase, I_sweep_derot, Q_sweep_derot, I_derot, Q_derot = derot_phase(f0_fits, I0, Q0, freq_s21, I_ts, Q_ts, I, Q, mode, f0_ref)

        df_derots.append(df_derot)
        phases.append(phase)

    freq_derot, psd_derot = get_psd(df_derots, Fs)
    freq_phase, psd_phase = get_psd(phases, Fs)

    return freq_derot, psd_derot, df_derots, freq_phase, psd_phase


def get_phase_derot(Is, Qs, fss, ts_low, ts_high, hr_high, hr_low, header_ts, mode):

    freq_derot_on_low, psd_derot_on_low, df_derots_on_low, freq_phase_on_low, psd_phase_on_low = get_psd_derot(Is, Qs, fss, ts_low, hr_low, header_ts, mode)
    freq_derot_on_high, psd_derot_on_high, df_derots_on_high, freq_phase_on_high, psd_phase_on_high = get_psd_derot(Is, Qs, fss, ts_high, hr_high, header_ts, mode)

    freq_derot_on, psd_derot_on = mix_psd([freq_derot_on_low, freq_derot_on_high], [psd_derot_on_low, psd_derot_on_high])
    freq_phase_on, psd_phase_on = mix_psd([freq_phase_on_low, freq_phase_on_high], [psd_phase_on_low, psd_phase_on_high])

    return freq_derot_on, psd_derot_on, freq_phase_on, psd_phase_on




"""
subplots_adjust(
    top=0.96,
    bottom=0.075,
    left=0.05,
    right=0.98,
    hspace=0.25,
    wspace=0.2
)

subplot(211)
title('df derotated')
loglog(freq_derot_on, psd_derot_on, label=str(temp)+' mK')
loglog(freq_derot_off, psd_derot_off)
xlabel(r'Frequency[Hz]')
ylabel(r'PSD [Hz$^2$/Hz]')
grid(True, which="both", ls="-")
xlim([1, 2e5])
ylim([0.01, 2000])
legend()

subplot(212)
title('Phase derotated')
loglog(freq_phase_on, psd_phase_on)
loglog(freq_phase_off, psd_phase_off)
xlabel(r'Frequency[Hz]')
ylabel(r'PSD [rad$^2$/Hz]')
grid(True, which="both", ls="-")
xlim([1, 2e5])
ylim([1e-11, 3e-6])
"""





def process_noise( freq_phase_on, phase_on, phase_off , f0, Qr, kid, temp, atten, ):

    noise_from = 7.5e4
    noise_to = 8.0e4
    noise_idx_from = np.where(freq_phase_on>noise_from)[0][0]
    noise_idx_to = np.where(freq_phase_on<noise_to)[0][-1]
    one_sigma = np.std(phase_on[noise_idx_from:noise_idx_to])
    mu_noise = np.mean(phase_on[noise_idx_from:noise_idx_to])

    amp_noise = np.median(phase_on[noise_idx_from:noise_idx_to])

    trim_high = 9e4
    trim_low = 4

    fitPSD = fit_psd()
    ioff()
    psd_clean = phase_on[np.where(freq_phase_on>trim_low)[0][0]:np.where(freq_phase_on>trim_high)[0][0]] - phase_off[np.where(freq_phase_on>trim_low)[0][0]:np.where(freq_phase_on>trim_high)[0][0]]

    psd_clean_for_nep = phase_on - phase_off

    psd_clean_e = psd_clean[psd_clean>0]
    fm = freq_phase_on[np.where(freq_phase_on>trim_low)[0][0]:np.where(freq_phase_on>trim_high)[0][0]][psd_clean>0]

    amp_idx_from = np.where(fm>noise_from)[0][0]
    amp_idx_to = np.where(fm<noise_to)[0][-1]
    amp_noise = np.nanmedian(psd_clean_e[amp_idx_from:amp_idx_to])
    print('++++++++++++++++++++++++++++++++++++++++')
    print(amp_noise)

    try:
        fitPSD.fit_psd(fm, psd_clean_e, f0, Qr, amp_noise, inter=True)
        #fitPSD.fit_psd(freq_derot_on[np.where(freq_derot_on>trim_low)[0][0]:np.where(freq_derot_on>trim_high)[0][0]], psd_derot_on[np.where(freq_derot_on>trim_low)[0][0]:np.where(freq_derot_on>trim_high)[0][0]], f0, Qr, amp_noise, inter=True)
        ion()

        loglog(fm, psd_clean_e)

        fit_PSD = combined_model(freq_phase_on, fitPSD.gr_noise, fitPSD.tau, fitPSD.tls_a, fitPSD.tls_b, f0, Qr, amp_noise)
        loglog(freq_phase_on, fit_PSD, 'k', lw=1.5)

        np.save('PSD-PHASE-CLEAN-K'+str(kid).zfill(3)+'-T_'+str(temp).zfill(3)+'-A_'+str(atten).zfill(3), [freq_phase_on[psd_clean_for_nep>0], psd_clean_for_nep[psd_clean_for_nep>0]] )

        print('*******************')
        print('Qr: ', Qr)
        print('f0: ', f0)
        print('tau[us]: ', fitPSD.tau*1e6 )

        fit_the_psd = {}
        fit_the_psd['gr'] = fitPSD.gr_noise
        fit_the_psd['tau'] = fitPSD.tau
        fit_the_psd['tls_a'] = fitPSD.tls_a
        fit_the_psd['tls_b'] = fitPSD.tls_b

        np.save('DEF-PHASE-PSD-K'+str(kid).zfill(3)+'-T_'+str(temp).zfill(3)+'-A_'+str(atten).zfill(3), fit_the_psd )

        # Ruido Generación-Recombinación
        gr = fitPSD.gr_noise/(1.+(2*np.pi*freq_phase_on*fitPSD.tau)**2) / (1.+(2*np.pi*freq_phase_on*Qr/np.pi/f0)**2)
        # Ruido TLS
        tls = fitPSD.tls_a*freq_phase_on**fitPSD.tls_b / (1.+(2*np.pi*freq_phase_on*Qr/np.pi/f0)**2)

        #subplot(3,1,n+1)
        loglog(freq_phase_on, gr, 'r-', label = 'GR')
        loglog(freq_phase_on, tls, 'b-', label = 'TLS')
        loglog(freq_phase_on, amp_noise*np.ones_like(freq_phase_on), 'g-', label='amp')
        axhline(fitPSD.gr_noise, color='k', linestyle='dashed')

        grid(True, which="both", ls="-")
        xlabel('Frequency[Hz]')
        ylabel(r'PSD [Hz$^2$/Hz]')

        return freq_phase_on[psd_clean_for_nep>0], psd_clean_for_nep[psd_clean_for_nep>0], fitPSD.tau, fitPSD.gr_noise, fitPSD, fit_PSD, freq_phase_on

    except:
        return freq_phase_on[psd_clean_for_nep>0], psd_clean_for_nep[psd_clean_for_nep>0], None, None, None, None, None



#nep = get_NEP( fm, psd_clean_e, fitPSD.tau, np.abs(dF0_dNqps[kid]), Qr, f0 )
#nep = get_NEP( freq_phase_on[psd_clean_for_nep>0], psd_clean_for_nep[psd_clean_for_nep>0], fitPSD.tau, np.abs(dP_dNqps[kid]), Qr, f0 )

#np.save('NEP-PHASE-K'+str(kid).zfill(3)+'-T_'+str(temp).zfill(3)+'-A_'+str(atten).zfill(3), [freq_phase_on[psd_clean_for_nep>0], nep])
