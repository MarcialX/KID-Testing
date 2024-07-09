# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab. IQ functions
# iq_funcs.py
# Variety of functions to process IQ data.
#
# Marcial Becerril, @ 23 May 2024
# Latest Revision: 23 May 2024, 10:50 GMT-6
#
# TODO list:
#
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #

import os
import sys
import time
import lmfit

import numpy as np

from scipy.signal import savgol_filter

from scipy import signal
from scipy import interpolate

sys.path.append('../')
from misc.msg_custom import *
from misc.timeout import timeout

from matplotlib.pyplot import *
#ion()


# G E N E R A L   F U N C T I O N S
# ---------------------------------------------------------
def fitCirc(x, y):
    """
        Fit the IQ circle.
        This code belongs to Andreas Papagiorgou.
        Parameters
        ----------
        x, y : array
            I/Q sweep frequency.
        ----------
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
