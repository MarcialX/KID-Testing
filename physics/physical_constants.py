# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab.
# physical_constants.py
# Physical constants
#
# Marcial Becerril, @ 23 May 2024
# Latest Revision: 23 May 2024, 11:00 GMT-6
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

import numpy as np

# Boltzmann constant [J K^-1]
Kb = 1.380649e-23
# Planck constant [J Hz^-1]
h = 6.62607015e-34
# Light speed [m/s]
c = 299792458
# CMB temperature [K]
Tcmb = 2.725

# Single spin density of states at the Fermi level [um^-3 J^-1]
N0s = {
        'Al'    : 1.07e29,
        'TiN'   : 2.43e29
}

# Tc [K]
Tcs = {
        'Al'    : 1.3,
}
