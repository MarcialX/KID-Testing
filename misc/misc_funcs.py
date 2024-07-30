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

from matplotlib.pyplot import *

from physics.physical_constants import *



def lin_binning(freq_psd, psd, w=10):
    """
    Linear binning PSD.
    Parameters
    -----------
    freq_psd : array
        Frequency [Hz].
    psd : array
        Power Spectral Density [Hz²/Hz].
    w : int
        Size binning.
    -----------
    """
    n_psd = []
    n_freq = []
    psd_accum = 0
    freq_accum = 0
    for i, p in enumerate(psd):
        if i%w == 0 and i != 0:
            n_psd.append(psd_accum/w)
            n_freq.append(freq_accum/w)
            psd_accum = 0
            freq_accum = 0
        psd_accum += p
        freq_accum += freq_psd[i]
    
    n_freq = np.array(n_freq)
    n_psd = np.array(n_psd)

    return n_freq, n_psd

def log_binning(freq_psd, psd, n_pts=500):
    """
    Logarithmic binning for PSD.
    Parameters
    -----------
    freq_psd : array
        Frequency [Hz].
    psd : array
        Power Spectral Density [Hz²/Hz].
    n_pts : int
        Number of points.
    -----------
    """

    start = freq_psd[0]
    stop = freq_psd[-1]

    central_pts = np.logspace(np.log10(start), np.log10(stop), n_pts+1)
    
    n_freq = []
    n_psd = []
    for i in range(n_pts):
        idx_start = np.where(freq_psd > central_pts[i])[0][0]
        idx_stop = np.where(freq_psd <= central_pts[i+1])[0][-1] + 1

        if not np.isnan(np.mean(freq_psd[idx_start:idx_stop])):
            n_freq.append(np.mean(freq_psd[idx_start:idx_stop]))
            n_psd.append(np.median(psd[idx_start:idx_stop]))

    n_freq = np.array(n_freq)
    n_psd = np.array(n_psd)

    return n_freq, n_psd