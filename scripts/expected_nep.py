# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Get expected NEP
#
# Marcial Becerril, @ 17 Jul 2024
# Latest Revision: 17 Jul 2024, 11:37 GMT-6
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

sys.path.append('../KID-Testing')
from homodyne import *

from matplotlib.pyplot import *
#rc('font', family='serif', size='14')
#ion()



# Boltzmann constant [J K^-1]
Kb = 1.380649e-23
# Planck constant [J Hz^-1]
h = 6.62607015e-34
# Light speed [m/s]
c = 299792458

n = 1
npb = 0.6
nu = 118e9  # GHz
dnu = 1e9*(125-110)
Tc = 1.3

# Power vector
P = np.linspace(0, 20e-12, 1000)

# Shot noise
NEP_shot = np.sqrt(2*h*nu*P/n)

# Wave noise
NEP_wave = np.sqrt(2*P**2/dnu)

# GR noise
Delta = 3.528*Kb*Tc
NEP_gr = np.sqrt(2*P*Delta/(n*npb))
#NEP_gr = 0

NEP = np.sqrt(NEP_shot**2 + NEP_wave**2 + NEP_gr**2)

plot(1e12*P, NEP)
grid(True, which="both", ls="-")


# Measured data
pwr = np.load('/home/marcial/Documents/SOUK-devices/ANL-TG-TopRight/fit_res_dict/responsivity-powers-Blackbody.npy')
att = ['44', '42']

for t, bb in enumerate([0, 10, 15, 20, 25, 30, 40, 50, 60, 70]):
    for k, kid in enumerate([0, 2]):
        try:
            f, n = np.load('/home/marcial/Documents/SOUK-devices/ANL-TG-TopRight/fit_res_dict/neps_pts-K00'+str(kid)+'-B'+str(bb).zfill(3)+'-A'+att[k]+'.0.npy')
            if kid == 0:
                plot(1e12*pwr[kid][t], n[2], 'rs')
            else:
                plot(1e12*pwr[kid][t], n[2], 'bs')
        except Exception as e:
            print(e)
