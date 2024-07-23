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
from scipy import integrate

from matplotlib.pyplot import *

from physics.physical_constants import *


def spectra_noise_model(freqs, gr_level, tau_qp, tls_a, tls_b, Qr=20e3, f0=1e9, amp_noise=0):
    """
    Spectra noise model.
    Parameters
    ----------
    freqs : array 
        Frequency array [Hz]
    gr_level : float
        Generation-Recombination noise level.
    tau_qp : float
        Quasiparticle lifetime.
    tls_a, tls_b : float
        Parameters that make up the 1/f noise component.
    Qr : float
        Total quality factor.
    f0 : float
        Resonance frequency.
    amp_noise : float
        Amplifier noise.
    ----------
    """
    # Generation-Recombination noise
    gr = gr_noise(freqs, gr_level, tau_qp, Qr, f0)
    # TLS noise
    tls = tls_noise(freqs, tls_a, tls_b, Qr, f0)
    # Amplifier noise
    amp = amp_noise

    # Total noise
    return gr + tls + amp

def gr_noise(freqs, gr_level, tau_qp, Qr, f0):
    """
    Generation-Recombination noise.
    Parameters:
    -----------
    freqs : array 
        Frequency array [Hz]
    gr_noise : float
        Generation-Recombination noise level.
    tau_qp : float
        Quasiparticle lifetime.
    Qr : float
        Total quality factor.
    f0 : float
        Resonance frequency.
    -----------
    """

    return gr_level/(1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)

def tls_noise(freqs, tls_a, tls_b, Qr, f0):
    """
    Generation-Recombination noise.
    Parameters:
    -----------
    freqs : array 
        Frequency array [Hz]
    tls_a, tls_b : float
        Parameters that make up the 1/f noise component.
    Qr : float
        Total quality factor.
    f0 : float
        Resonance frequency.
    -----------
    """

    return tls_a*freqs**(tls_b) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)

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

def get_power_from_FTS(diry, kid, T, n_pols=2):
    """
    Get the power from the FTS.
    """

    fts = np.load(diry+'/K'+str(kid).zfill(3)+'.npy')
    f = fts[:,0][1:]
    tx = fts[:,1][1:]

    fl = 350
    f_sel = f[np.where(f<fl)[0]]
    tx_sel = tx[np.where(f<fl)[0]]

    spec = tx_sel*planck(f_sel*1e9, T)
    nu_max = f_sel[np.argmax(tx_sel*1e9)]*1e9

    """
    figure()
    plot(f_sel, spec, label=str(kid).zfill(3))
    #axvline(nu_max)
    print('Central freq: ', nu_max)
    show()
    """

    #Ao = 1
    Ao = ((c/nu_max)**2)/n_pols
    #print(c/nu_max)
    P = integrate.trapezoid(spec, f_sel*1e9) * Ao

    return P

def gaussian(x, A, mu, sigma, offset):
    """
    Gaussian function
    Parameters
    ----------
    x : int/float/array
    A : float
        Amplitude
    mu : float
        Mean
    sigma : float
        Dispersion
    offset : float
        Offset
    ----------
    """
    return offset+A*_np.exp(-((x-mu)**2)/(2*sigma**2))

def twoD_Gaussian(pos, amplitude, xo, yo, sigma, offset):
    """
    2D Gaussian function.
    Parameters
    ---------
        Input:
            pos:     	[list] Points of the 2D map
            amplitude:  [float] Amplitude
            xo, yo:     [float] Gaussian profile position
            sigma:      [float] Dispersion profile
            offset:     [float] Offset profile
        Ouput:
            g:			[list] Gaussian profile unwraped in one dimension
    ---------
    """
	
    x = pos[0]
    y = pos[1]
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude*np.exp(-(((x - xo)**2/2./sigma**2) + ((y - yo)**2/2./sigma**2)))

    return g.ravel()

def twoD_ElGaussian(pos, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    2D Elliptical Gaussian function.
    Parameters
    ---------
        Input:
            pos:             	[list] Points of the 2D map
            amplitude:          [float] Amplitude
            xo, yo:             [float] Gaussian profile position
            sigma_x, sigma_y:   [float] X-Y Dispersion profile
            theta:              [float] Major axis inclination
            offset:             [float] Offset profile
        Ouput:
            g:			[list] Gaussian profile unwraped in one dimension
    ---------
    """
    
    x = pos[0]
    y = pos[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset+amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return g.ravel()
