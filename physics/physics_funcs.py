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

from .physical_constants import *


def nep_gr(Nqp, tqp, Tc, eta=0.6):
    """
    Generation-Recombination noise.
    Parameters
    ----------
    Nqp : float
        Number of quasiparticles.
    tqp : float
        Quasiparticle lifetime.
    Tc : float
        Critical temperature.
    eta : float
        Conversion efficiency photon energy - quasiparticles.
    ----------
    """

    Delta = get_Delta(Tc)
    NEP_gr = (2*Delta/eta)*np.sqrt(Nqp/tqp)
    return NEP_gr

def nep_rec(f0, P):
    """
    Optical Recombination noise.
    Parameters
    ----------
    f0 : float
        Central frequency.
    P : float
        Optical power.
    ----------
    """

    NEP_rec = np.sqrt(2*h*f0*P)
    return NEP_rec

def nep_shot(f0, P, eta=0.6):
    """
    NEP shot noise.
    Parameters
    ----------
    f0 : float
        Central frequency.
    P : float
        Power
    eta : float
        Optical efficiency.
    ----------
    """

    NEP_shot = np.sqrt(2*h*f0*P/eta)
    return NEP_shot

def nep_wave(dnu, P, n_pols=1):
    """
    NEP wave noise.
    Parameters
    ----------
    dnu : float
        Bandwidth.
    P : float
        Power
    n_pols : int
        Number of polarizations.
    ----------
    """

    NEP_wave = np.sqrt(2*P**2/dnu/n_pols)
    return NEP_wave

def nep_photon(P, eta, dnu, f0, n_pols=1):
    """
    NEP photon noise.
    Parameters
    ----------
    P : float
        Power
    dnu : float
        Bandwidth.
    f0 : float
        Central frequency
    ----------
    """

    total_nep = np.sqrt( nep_shot(f0, P, eta=eta)**2 + nep_wave(dnu, P, n_pols=n_pols)**2 )
    return total_nep

def tr(T, Tc, t0):
    """
    Get tau_r time as described in Pan et al. 2023.
    Parameters
    ----------
    T : float
        Base temperature.
    Tc : float
        Critical temperature.
    t0 : float
        Material-dependent characteristic electron-phonon
        interaction time and can be modified by impurity 
        scattering.
    ----------
    """

    Delta = get_Delta(Tc)
    tr = (t0/np.sqrt(np.pi))*(Kb*Tc/(2*Delta))**(5/2) * np.sqrt(Tc/T) * np.exp(Delta/Kb/T)
    return tr

def spectra_noise_model(freqs, gr_level, tau_qp, tls_a, tls_b, amp_noise=0, Qr=20e3, f0=1e9):
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

    """
    # Generation-Recombination noise
    gr = gr_noise(freqs, gr_level, tau_qp, Qr, f0)
    # TLS noise
    tls = tls_noise(freqs, tls_a, tls_b, tau_qp, Qr, f0)
    # Amplifier noise
    amp = amp_noise
    """

    relax = relaxation(freqs, tau_qp, Qr, f0)
    Sxx = amp_noise + (gr_level*np.ones_like(freqs) + (tls_a*freqs**(tls_b))) * relax
    
    # Total noise
    return Sxx

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

    #return gr_level / (1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)
    relax = relaxation(freqs, tau_qp, Qr, f0)
    return gr_level * relax

def tls_noise(freqs, tls_a, tls_b, tau_qp, Qr, f0):
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

    #return (tls_a*freqs**(tls_b)) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)
    #return (tls_a*freqs**(tls_b)) / (1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)
    relax = relaxation(freqs, tau_qp, Qr, f0)
    return (tls_a*freqs**(tls_b)) * relax

def relaxation(freqs, tau_qp, Qr, f0):

    relax_factor = 1 / (1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*Qr/np.pi/f0)**2)
    return relax_factor

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
    Parameters
    ----------
    f : array
        Frequency array [Hz].
    Sa : float
        Power Spectrum Density [Hz²/Hz].
    tqp : float
        Quasiparticle lifetime [s].
    S : float
        Responsivity [Hz/W].
    Qr : float
        Total quality factor.
    f0 : float
        Resonance frequency.
    ----------
    """

    return np.sqrt(Sa) * ( (np.abs(S))**(-1)) * np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)

def get_Dark_NEP(f, Sa, tqp, S, Qr, f0, Delta, eta=0.6):
    """
    Get the Dark NEP.
    Parameters
    ----------
    f : array
        Frequency array [Hz].
    Sa : float
        Power Spectrum Density [Hz²/Hz].
    tqp : float
        Quasiparticle lifetime [s].
    S : float
        Responsivity [Hz/W].
    Qr : float
        Total quality factor.
    f0 : float
        Resonance frequency.
    Delta : float
        Binding energy.
    eta(opt) : float
        Optical efficiency.
    ----------
    """

    return np.sqrt(Sa) * (( (eta*tqp/Delta)*(np.abs(S)) )**(-1)) * np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)

def get_Delta(Tc):
    """
    Get energy gap, for a T << Tc.
    Parameters
    ----------
    Tc : float
        Critical Temperature.
    ----------
    """

    return 3.528*Kb*Tc/2

def get_nqp(N0, T, Delta):
    """
    Get the quasiparticle density.
    Parameters
    ----------
    N0 : float
        Single spin density of states at the Fermi level.
    T : float
        Base temperature.    
    Delta : float
        Energy gap.
    ----------
    """
    return 2 * N0 * np.sqrt( 2*np.pi*Kb*T*Delta ) * np.exp(-(Delta/(Kb*T)))

def n_occ(freq, T):
    """
    Photon occupation number as defined in
    https://github.com/chill90/BoloCalc
    Parameters
    ----------
    freq : float/array
        Frequency array [Hz].
    temp : float/array
        Temperature [K].
    ----------
    """

    return 1/(np.exp((h*freq)/(Kb*T))-1)

def dPdT(freq, tx):
    """
    Incident power fluctuations due to fluctuations
    in CMB temperature.
    Ref: Hill et al 2019 (SO Collaboration)
    Parameters
    ----------
    freq : float/array
        Frequency array [Hz]
    tx : 
        Transmission efficiency [K]
    ----------
    """

    a = (1/Kb) * ((h*freq/Tcmb)**2) * (n_occ(freq, Tcmb)**2) * np.exp((h*freq)/(Kb*Tcmb)) * tx
    return integrate.trapezoid(a, freq)

def load_tx(diry, kid, freq_upper_limit=350):
    """
    Load transmission efficieny file (FTS spectra).
    Paramteres
    ----------
    diry : string
        Directory
    kid : string
        Detector number.
    freq_upper_limit : float
        Frequency upper limit [GHz]. Frequency beyond is discarded.
    ----------
    """

    # Load transmission.
    fts = np.load(diry+'/K'+str(kid).zfill(3)+'.npy')
    f = fts[:,0][1:]
    tx = fts[:,1][1:]

    # Filter transmission, just select up to an upper limit.
    f_sel = f[np.where(f < freq_upper_limit)[0]]
    tx_sel = tx[np.where(f < freq_upper_limit)[0]]

    return f_sel, tx_sel

def AOmega(f0, modes=1):
    """
    Get throughput AOmega.
    Parameters
    ----------
    f0 : float
        Central frequency.
    modes : int
        Number of polarizations.
        1 : for linear polarization-sensitive detectors
    ----------
    """

    # Assuming beam filled.
    AO = modes*((c/f0)**2)/2
    return AO

def get_power_from_FTS(diry, kid, T, n_pols=2):
    """
    Get the power from the FTS.
    Parameters
    ----------
    diry : string
        Data directory
    kid : int
        KID number.    
    T : float
        Blackbody temperature.
    n_pols : int
        # of modes.
    ----------
    """

    # Load transmission.
    f_sel, tx_sel = load_tx(diry, kid, freq_upper_limit=350)

    # Transmission * BB(T)
    spec = tx_sel*planck(f_sel*1e9, T)
    # Get the central frequency
    f0 = f_sel[np.argmax(tx_sel*1e9)]*1e9
    print(f'F0: {f0:.2f} GHz')

    # A*Omega assuming beam fill.
    AO = AOmega(f0, modes=1)

    """
    figure()
    plot(f_sel, spec, label=str(kid).zfill(3))
    #axvline(nu_max)
    print('Central freq: ', nu_max)
    show()
    """

    # Get total power
    P = integrate.trapezoid(spec, f_sel*1e9) * AO

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
    return offset+A*np.exp(-((x-mu)**2)/(2*sigma**2))

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
