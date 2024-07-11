# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Find overdriven attenuations. EXAMPLE
# find_overdriven_example.py
#
# Marcial Becerril, @ 11 Jun 2024
# Latest Revision: 11 Jun 2024, 13:08 GMT-6
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
ion()
rc('font', family='serif', size='14')

sys.path.append('../KID-Testing')
from homodyne import *

project_name = 'ALH-Ti10Al30-etch_cl-2105241100'
data_path = "/home/marcial/detdaq/2105241100-Ti10Al30_ClEtch/20240603_Dark_Data_Auto/"

h = Homodyne(data_path, only_vna=True, proj_name=project_name)

path = '/home/marcial/Documents/SOUK-devices/'+project_name
h.load_fit(path)
h.find_overdriven_atts(80)