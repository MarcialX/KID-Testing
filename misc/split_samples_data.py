# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab. Split the data in folders per sample
# split_samples_data.py
# Separate the data samples in different folders
#
# Marcial Becerril, @ 26 March 2024
# Latest Revision: 26 Mar 2024, 21:10 GMT-6
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

import numpy as np
from os import walk
import os

sys.path.append('../')
from misc.msg_custom import *

# Dirfile
#diry = '/home/marcial/detdaq/Fb-Sp-TiN/20240515_Dark_Data_Auto/'
diry = '/home/marcial/detdaq/ALH-SLIM-24-SO/20240516_Dark_Data_Auto/'
sub_diry = 'VNA_Sweeps'

filenames = next(walk(diry + sub_diry), (None, None, []))[2]

for filename in filenames:
    file_pieces = filename.split('_')[-1]

    if file_pieces[:-5].isnumeric():
        n = int(file_pieces[:-5])
        new_diry = diry[:-1]+'_'+str(n)
        try:
            os.system('mkdir '+new_diry)
            os.system('mkdir '+new_diry+'/'+sub_diry)
        except Exception as e:
        	msg('Directory not created. '+str(e), 'warn')
        os.system('mv '+diry+sub_diry+'/'+filename+' '+new_diry+'/'+sub_diry+'/'+filename)
