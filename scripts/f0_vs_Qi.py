# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Script to plot f0 vs Qis
# f0_vs_Qis.py
#
# Marcial Becerril, @ 15 Nov 2024
# Latest Revision: 15 Nov 2024, 06:39 GMT-6
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

#from homodyne import *


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

                    s_qi = []
                    s_qc = []
                    s_fr = []
                    # Get list of atts
                    for fit_file in fit_files: 
                        if kid_name in fit_file and temp_name in fit_file:
                            att_from_file = int(fit_file.split('-')[3][1:-2])
                            if att_from_file >= over_attens[kid]:

                                #file_name = 'fit-'+kid_name+'-'+temp_name+'-'+att_name+'-S0.npy'
                                fit_name = att_from_file
                                fit_data = np.load(path+fit_file, allow_pickle=True).item()

                                print(kid_name)
                                print(temp_name)

                                if fit_data['Qi'] < 1e6:
                                    s_qc.append(fit_data['Qc'])
                                    s_qi.append(fit_data['Qi'])
                                    s_fr.append(fit_data['fr'])

                    print(s_qi)

                    summ[chip][kid_name]['qi'] = np.max(s_qi)
                    summ[chip][kid_name]['qc'] = np.mean(s_qc)
                    summ[chip][kid_name]['fr'] = s_fr[np.argmax(s_qi)]
    
    elif cryo[c] == 'Eagle':

        filenames = os.listdir(root_diry + chip)
        for filename in filenames:
            if filename.endswith('.npy'):
                data_eagle = np.load(root_diry + chip + '/' + filename, allow_pickle=True).item()

        for k, kid in enumerate(data_eagle.keys()):

            if not kid in ignore[c]:

                qis = data_eagle[kid]['Qi']
                qcs = data_eagle[kid]['Qc']
                atts = data_eagle[kid]['in_att']

                qis_max_idx = np.argmax(qis)
                qcs_mean_idx = np.argmax(qcs)
                
                qi = qis[qis_max_idx]
                qc = qcs[qcs_mean_idx]
                att_max = atts[qis_max_idx]

                print(kid)
                print('Qi max', qi)
                print('Qc mean', qc)
                print('Att max: ', att_max) 

                if not kid in summ[chip]:
                    summ[chip][kid] = {}
                
                summ[chip][kid]['qi'] = qi
                summ[chip][kid]['qc'] = qc

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


cmap_obj = matplotlib.cm.get_cmap('tab20')
norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(summ.keys())+3)

for m, c in enumerate(summ.keys()):
    f0s, qis, qcs = [], [], []
    for k in summ[c].keys():
        if 'qi' in summ[c][k]:
            f0s.append(summ[c][k]['fr'])
            qis.append(summ[c][k]['qi'])
            qcs.append(summ[c][k]['qc'])
    
    f0s = np.array(f0s)
    qis = np.array(qis)
    qcs = np.array(qcs)
    #plot(f0s*1e-6, qis*1e-3, symbol[m], label=labels[m], color=cmap_obj(norm_color(m)))
    plot(f0s*1e-6, qcs*1e-3, symbol[m], label=labels[m], color=cmap_obj(norm_color(m)))


# Add manual
# Chip 9
f0 = 1e3*np.array([1.3569, 1.4350, 1.5192, 1.5943, 1.6752, 1.7496, 1.8289, 1.9004, 1.9867, 2.0606, 2.1435, 2.2198, 2.3060, 2.3917, 2.4934, 2.5951])
qi = np.array([203.34, 178.356, 129.848, 135.504, 80.072, 141.561, 284.651, 92.691, 104.614, 91.646, 41.661, 85.535, 40.726, 48.145, 32.999, 324.323])
qc = np.array([34.387, 94.769, 60.151, 45.346, 61.234, 62.229, 68.477, 80.018, 59.066, 88.388, 65.977, 88.806, 108.502, 92.989, 85.900, 22.654])
plot(f0, qc, 'D', label='CPW2-Chip 9 @ 24mK [Dark]: SOUK', color=cmap_obj(norm_color(m+1)))
# Chip 7
f0 = 1e3*np.array([1.8578, 1.9756, 2.0171, 2.0883, 2.1144, 2.1873, 2.2049, 2.2128, 2.2727, 2.3069, 2.3668, 2.4103, 2.4654, 2.5080, 2.6020, 2.8519])
qi = np.array([389.119, 64.876, 92.081, 69.307, 91.116, 48.249, 58.011, 76.017, 62.208, 54.807, 18.184, 65.170, 30.410, 42.195, 58.567, 66.498])
qc = np.array([42.083, 46.706, 69.949, 49.503, 36.648, 67.584, 50.981, 83.696, 61.438, 44.281, 104.858, 42.543, 39.120, 77.705, 61.859, 69.638])
plot(f0, qc, 'D', label='SF1-Chip 7 @ 24mK [Dark]: SOUK', color=cmap_obj(norm_color(m+2)))
# Chip 8
f0 = 1e3*np.array([1.9685, 2.0606, 2.1854, 2.2904, 2.3318, 2.3474, 2.4220, 2.5060, 2.5156, 2.6120, 2.7057, 2.8206])
qi = np.array([332.979, 184.356, 90.319, 50.616, 134.721, 47.998, 83.458, 27.850, 86.774, 74.333, 53.654, 39.161])
qc = np.array([66.607, 49.532, 29.664, 73.815, 50.398, 88.682, 38.680, 30.026, 33.872, 85.696, 93.573, 37.819])
plot(f0, qc, 'D', label='SF1-Chip 8 @ 24mK [Dark]: SOUK', color=cmap_obj(norm_color(m+3)))


xlabel('Resonance frequency [MHz]')
ylabel('Qc [parts per thousand]')
grid(True, which="both", ls="-")
legend(fontsize=10, ncols=2)
