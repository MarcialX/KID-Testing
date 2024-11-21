# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Script to plot f0 vs drive power
# f0_vs_drive_pwr.py
#
# Marcial Becerril, @ 17 Nov 2024
# Latest Revision: 17 Nov 2024, 15:29 GMT-6
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #

import os
import numpy as np 

import argparse
import yaml

from matplotlib.pyplot import *
ion()

import sys
sys.path.append('/home/marcial/Documents/KID-Testing/')
from homodyne import *


# G L O B A L   P A R A M E T E R S
# --------------------------------------------------
root_diry = "/home/marcial/Documents/SOUK-devices/"

# Amber chips
#chips = ['SOUK-RV1-Amber-BT111024']

chips = ['SOUK-RV1-Chip0-BT-01102024',
         'SOUK-RV1-Chip2-BT-200924',
         '240912-SOUK_Dark_Chip6-50mK-figs',
         'SOUK-RV1-Chip8_BB-110924',
         'SOUK-RV1-Chip11',
         'SOUK-RV1-Amber-BT111024',
         'SOUK-RV2-Chip2-BB-221024',
         'SOUK-RV2-Amber-BT-221024',
         'SOUK-RV3-Chip2-BT-291024',
         'SOUK-RV3-Amber-BT-291024',
         'SOUK-CPW1-141020241230-Chip2-BT',
         'SOUK-RV4-Chip6-Run2',
         'SOUK-RV4-Chip9-Run2',
         'SOUK-CPW2-211020241300-Chip7-BT'
         ]

extra_folder = ['',
                '',
                '/home/marcial/detdaq/AlTiAl-Dark-110924/SOUK_Dark_Chip6/240912/vna_sweeps/',
                '',
                '/home/marcial/detdaq/AlTiAl-Dark-Eagle/SOUK_Dark_RV1Chip11/241007/vna_sweeps/',
                '',
                '',
                '',
                '',
                '',
                '',
                '/home/marcial/detdaq/RV4-Chip6-Run2/241107/vna_sweeps/',
                '/home/marcial/detdaq/RV4-Chip9/241107/vna_sweeps/',
                ''
               ]

data_path = [
            "/home/marcial/detdaq/RV1-Chip0/20240930_Dark_Data_Auto/",
            "/home/marcial/detdaq/RV1-Chip2/20240920_Dark_Data_Auto/",
            "",
            "/home/marcial/detdaq/RV1-Chip8/20240911_Blackbody_Data_AutoBB/",
            "",
            "/home/marcial/detdaq/RV1-Amber/20241011_Dark_Data_Auto/",
            "/home/marcial/detdaq/_RV2_Chip2/20241022_Dark_Data_Auto/",
            "/home/marcial/detdaq/RV2-Amber-NI/20241022_Dark_Data_Auto/",
            "/home/marcial/detdaq/RV3_021020241400_Chip2/20241029_Dark_Data_Auto/",
            "/home/marcial/detdaq/RV3_021020241400_Amber/20241029_Dark_Data_Auto/",
            "/home/marcial/detdaq/CPW1-1410241230_Chip2/20241111_Dark_Data_Auto/",
            "/home/marcial/Documents/SOUK-devices/SOUK-RV4-Chip6-Run2/",
            '/home/marcial/Documents/SOUK-devices/SOUK-RV4-Chip9-Run2/',
            "/home/marcial/detdaq/CPW2-211020241300-Chip7/20241111_Dark_Data_Auto/"
            ]


labels = ['RV1-Chip 0 @ 80mK [Dark]: ALH(AlTiAL)+SOUK',
          'RV1-Chip 2 @ 80mK [Dark]: ALH(AlTiAL)+SOUK',
          'RV1-Chip 6 @ 50mK [Dark]: ALH(AlTiAL)+SOUK',
          'RV1-Chip 8 @ 80mK [Opt BB:6.7k]: ALH(AlTiAL)+SOUK',
          'RV1-Chip 11 @ 80mK [Dark]: ALH(AlTiAL) [SOUK too shallow]',
          'RV1-ALH @ 80mK [Dark]: ALH(AlTiAl)',
          'RV2-NI-Chip 2 @ 90mK [Dark]: SOUK',
          'RV2-NI-ALH @ 90mK [Dark]: ALH(AlTiAl)',
          'RV3-NI-Chip 2 @ 80mK [Dark]: SOUK ',
          'RV3-NI-ALH @ 80mK [Dark]: ALH(AlTiAl)',
          'CPW1-Chip 2 @ 80mK [Dark]: SOUK',
          'CPW1-Chip 6 @ 26mK [Dark]: SOUK',
          'CPW1-Chip 9 @ 34mK [Dark]: SOUK',
          'CPW2-Chip 7 @ 80mK [Dark]: SOUK'
          ]

cryo = ['Aloysius',
        'Aloysius',
        'Eagle',
        'Aloysius',
        'Eagle',
        'Aloysius',
        'Aloysius',
        'Aloysius',
        'Aloysius',
        'Aloysius',
        'Aloysius',
        'Eagle',
        'Eagle',
        'Aloysius'
       ]

ignore = [['K001', 'K002', 'K003', 'K017', 'K018', 'K019', 'K020', 'K021', 'K022', 'K023', 'K024'],
          ['K001', 'K002', 'K003'],
          ['K001', 'K002', 'K003', 'K004', 'K006', 'K012', 'K013', 'K018', 'K019', 'K020', 'K021', 'K022', 'K023', 'K024'],
          [],
          ['K001', 'K002', 'K003'],
          ['K004', 'K005', 'K006', 'K007', 'K008', 'K009', 'K010', 'K011', 'K012', 'K013', 'K014', 'K015'],
          [],
          ['K000', 'K001', 'K002', 'K003', 'K004', 'K005', 'K008', 'K009', 'K010', 'K011', 'K012', 'K013'],
          [],
          ['K000', 'K001', 'K002', 'K003', 'K004', 'K005', 'K008', 'K009', 'K010', 'K011', 'K012', 'K013', 'K014'],
          [],
          [],
          ['K002'],
          []
         ]

symbol = ['s',
          's',
          's',
          's',
          's',
          's',
          '^',
          '^',
          '*',
          '*',
          'o',
          'o',
          'o',
          'D'
         ]

# R E A D   D A T A
# --------------------------------------------------
summ = {}
for c, chip in enumerate(chips):

    if not chip in summ:
        summ[chip] = {}

    if cryo[c] in ['Aloysius', 'Elmo']:

        with open(root_diry + chip +"/params.yaml", 'r') as f:
            data = yaml.safe_load(f)
            over_attens = data['ATT_OVRDRN']

        h = Homodyne(data_path[c], only_vna=True, proj_name='Find_KIDs_testing', create_folder=False)

        path = root_diry + chip + '/fit_res_dict/vna/'
        fit_files = os.listdir(path)

        for kid in range(len(over_attens)):
            kid_name = 'K'+str(kid).zfill(3)
            att_name = 'A'+str(int(over_attens[kid]))+'.0'

            if not kid_name in ignore[c]:

                if not kid_name in summ[chip]:
                    summ[chip][kid_name] = {}

                opt_temps = []
                for fit_file in fit_files:
                    if (kid_name in fit_file) and (att_name in fit_file):
                        if fit_file[9] == 'B':
                            nzeros = 3
                        else:
                            nzeros = 4
                        opt_temps.append(fit_file[9:9+nzeros+1])
                
                if len(opt_temps)>0:
                    temp_name = sorted(opt_temps)[0]

                    file_name = 'fit-'+kid_name+'-'+temp_name+'-'+att_name+'-S0.npy'
                    fit_data = np.load(path+file_name, allow_pickle=True).item()

                    print(temp_name)
                    # Get drive power
                    extra_att = h.data['vna'][kid_name][temp_name][att_name]['header'][0]['ATT_UC'] + \
                                h.data['vna'][kid_name][temp_name][att_name]['header'][0]['ATT_C'] + \
                                h.data['vna'][kid_name][temp_name][att_name]['header'][0]['ATT_RT']
                    
                    vna_pwr = h.data['vna'][kid_name][temp_name][att_name]['header'][0]['VNAPOWER']
                    drive_power = -1*(float(att_name[1:])+extra_att)+vna_pwr

                    summ[chip][kid_name]['pwr'] = drive_power
                    summ[chip][kid_name]['qc'] = fit_data['Qc']
                    summ[chip][kid_name]['qi'] = fit_data['Qi']
                    summ[chip][kid_name]['fr'] = fit_data['fr']
    
    elif cryo[c] == 'Eagle':

        try:

            filenames = os.listdir(root_diry + chip)
            for filename in filenames:
                if filename.endswith('.npy'):
                    data_eagle = np.load(root_diry + chip + '/' + filename, allow_pickle=True).item()

            over_attens_data = np.load(data_path[c]+'over_attens.npy', allow_pickle=True).item()

            for k, kid in enumerate(data_eagle.keys()):

                if not kid in ignore[c]:

                    qis = data_eagle[kid]['Qi']
                    atts = data_eagle[kid]['in_att']

                    qis_max_idx = np.argmax(qis)
                    
                    qi = qis[qis_max_idx]
                    att_max = atts[qis_max_idx]

                    print(kid)
                    print('Qi max', qi)
                    print('Att max: ', att_max) 

                    if not kid in summ[chip]:
                        summ[chip][kid] = {}
                    
                    summ[chip][kid]['qi'] = qi
                    summ[chip][kid]['pwr'] = over_attens_data['FIXATT']-1*over_attens_data['ATTOVER'][k]

                    # Get f0
                    extra_files = os.listdir(extra_folder[c])
                    for extra_file in extra_files:
                        if extra_file.startswith('segmented'):
                            meas = np.load(extra_folder[c]+extra_file)

                            freq = meas['frequency_data'][0]
                            s21 = meas['complex_data'][0]

                            f0 = freq[k][np.argmax(np.abs(s21[k]))]
                            summ[chip][kid]['fr'] = f0
                            
                            break

            print(data_eagle.keys())

        except:
            pass


cmap_obj = matplotlib.cm.get_cmap('rainbow')
norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(summ.keys()))

for m, c in enumerate(summ.keys()):
    f0s, pwr = [], []
    for k in summ[c].keys():
        if 'pwr' in summ[c][k]:
            f0s.append(summ[c][k]['fr'])
            pwr.append(summ[c][k]['pwr'])
    
    f0s = np.array(f0s)
    pwr = np.array(pwr)
    plot(f0s*1e-6, pwr, symbol[m], label=labels[m], color=cmap_obj(norm_color(m)))


"""
# Add manual
# Chip 9
f0 = 1e3*np.array([1.3569, 1.4350, 1.5192, 1.5943, 1.6752, 1.7496, 1.8289, 1.9004, 1.9867, 2.0606, 2.1435, 2.2198, 2.3060, 2.3917, 2.4934, 2.5951])
qi = np.array([203.34, 178.356, 129.848, 135.504, 80.072, 141.561, 284.651, 92.691, 104.614, 91.646, 41.661, 85.535, 40.726, 48.145, 32.999, 324.323])
plot(f0, qi, 'D', label='CPW2-Chip 9 @ 24mK [Dark]: SOUK', color=cmap_obj(norm_color(m+1)))
# Chip 7
f0 = 1e3*np.array([1.8578, 1.9756, 2.0171, 2.0883, 2.1144, 2.1873, 2.2049, 2.2128, 2.2727, 2.3069, 2.3668, 2.4103, 2.4654, 2.5080, 2.6020, 2.8519])
qi = np.array([389.119, 64.876, 92.081, 69.307, 91.116, 48.249, 58.011, 76.017, 62.208, 54.807, 18.184, 65.170, 30.410, 42.195, 58.567, 66.498])
plot(f0, qi, 'D', label='SF1-Chip 7 @ 24mK [Dark]: SOUK', color=cmap_obj(norm_color(m+2)))
# Chip 8
f0 = 1e3*np.array([1.9685, 2.0606, 2.1854, 2.2904, 2.3318, 2.3474, 2.4220, 2.5060, 2.5156, 2.6120, 2.7057, 2.8206])
qi = np.array([332.979, 184.356, 90.319, 50.616, 134.721, 47.998, 83.458, 27.850, 86.774, 74.333, 53.654, 39.161])
plot(f0, qi, 'D', label='SF1-Chip 8 @ 24mK [Dark]: SOUK', color=cmap_obj(norm_color(m+3)))
"""

xlabel('Resonance frequency [MHz]')
ylabel('Drive power [dBm]')
grid(True, which="both", ls="-")
legend(fontsize=10, ncols=2)
