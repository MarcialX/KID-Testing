# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab.
# misc_functions.py
# Diverse functions
#
# Marcial Becerril, @ 28 May 2024
# Latest Revision: 28 May 2024, 12:50 GMT-6
#
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #

import sys
import numpy as np

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
import scipy.integrate as integrate

from physics.physical_constants import *


def combined_model(freqs, gr_noise, tau_qp, tls_a, tls_b, f0, Qr, amp_noise):
    # Ruido Generación-Recombinación
    gr = gr_noise/(1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)
    # Ruido TLS
    tls = tls_a*freqs**tls_b / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)
    # Ruido del amplificador
    amp = amp_noise
    # Ruido Total
    return gr + tls + amp

def f0_vs_pwr_model(P, a, b):
    """
    Responsivity model as a power-law with an 'b' index
    Parameters
    ----------
    Input:
        P : float
            Power in Watts.
        a : float
            Proportionality constant.
        b : float
            Power-law index
    ----------
    """
    return a*P**b

def bb2pwr(T, nu):
    """
    Get the power from the blackbody assuming a throughput (A*Omega) equals to 1.
    Parameters
    ----------
    Input:
        T : float
            Blackbody temperature [K].
        nu : float
            Bandwidth [Hz].
    ----------
    """
    return Kb * np.array(T) * nu

def planck(nu, T):
    '''
    From Tom Brien fucntion.
    Defines Planck function in frequency space
    inputs:
      nu: Frequency in Hertz
      T: Temperature in Kelvin
    output:
      B: Spectral radiance in W m^-2 sr^-1 Hz^-1
    '''
    B = 2*h*nu**3/c**2 * 1 / (np.exp((h*nu)/(Kb*T)) - 1)
    return B

def get_BB_NEP(f, Sa, tqp, S, Qr, f0):
    """
    Get the NEP from BB.
    """
    NEP = np.sqrt(Sa) * ( (np.abs(S))**(-1)) * np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)
    return NEP

def get_Dark_NEP(f, Sa, tqp, S, Qr, f0, Delta, eta=0.6):
    """
    Get the Dark NEP.
    """
    NEP = np.sqrt(Sa) * (( (eta*tqp/Delta)*(np.abs(S)) )**(-1)) * np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)
    return NEP

def get_Delta(Tc):
    """
    Get energy gap, for a T << Tc.
    """
    Delta = 3.528*Kb*Tc/2
    return Delta

def get_nqp(N0, T, Delta):
    """
    Get the quasiparticle density.
    """
    nqp = 2 * N0 * np.sqrt( 2*np.pi*Kb*T*Delta ) * np.exp(-(Delta/(Kb*T)))
    return nqp

def get_power_from_FTS(diry, kid, T):
    """
    Get the power from the FTS.
    """

    fts = np.load(diry+'/KID'+str(kid).zfill(3)+'.npy')
    f = fts[:,0][1:]
    tx = fts[:,1][1:]

    fl = 350
    """
    if nu == 150e9:
        fl = 220
    else:
        fl = 150
    """

    f_sel = f[np.where(f<fl)[0]]
    tx_sel = tx[np.where(f<fl)[0]]

    spec = tx_sel*planck(f_sel*1e9, T)

    nu_max = f_sel[np.argmax(tx_sel*1e9)]*1e9

    """
    figure()
    plot(f_sel, spec, label=str(kid).zfill(3))
    axvline(nu_max)
    """

    #Ao = 1
    Ao = (c/nu_max)**2
    P = integrate.trapz(spec, f_sel*1e9) * Ao

    return P
