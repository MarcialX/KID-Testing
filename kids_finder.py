#! /home/marcial/.venv/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID finder tool.
# kids_finder.py
#
# Marcial Becerril, @ 16 Aug 2024
# Latest Revision: 30 Sep 2024, 13:05 GMT-6
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

from matplotlib.pyplot import *
 #rc('font', family='serif', size='14')

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
rc('font', family='serif', size='16')

import sys
sys.path.append('../')
from homodyne import *

import argparse


# R E A D   A R G U M E N T S
# --------------------------------------------------
parser = argparse.ArgumentParser()
# Data paths
parser.add_argument('--path', '-p', help="Data directory", type=str, required=True)
parser.add_argument('--name', '-n', help="Tonelist name", type=str, required=False)
# Data properties
parser.add_argument('--temp', '-t', help="Temperature", type=int, required=False)
parser.add_argument('--atte', '-a', help="Attenuation", type=int, required=False)
# Finding properties
parser.add_argument('--down_factor', '-d', help="Downsampling factor", type=int, required=False)
parser.add_argument('--baseline_params', '-b', help="Baseline filter params", type=tuple, required=False)
parser.add_argument('--total_q', '-qr', help="Total quality factor", type=list, required=False)
parser.add_argument('--coupling_q', '-qc', help="Coupling quality factor", type=list, required=False)

args = parser.parse_args()

path = args.path
if path[-1] != '/':
    path += '/'

tonelist_name = args.name

temp = args.temp
atte = args.atte

down_factor = args.down_factor
baseline_params = args.baseline_params
qr = args.total_q
qc = args.coupling_q

# Default variables
if temp is None:
    temp = 80
if atte is None:
    atte = 40

if down_factor is None:
    down_factor = 35
if baseline_params is None:
    baseline_params = (501, 5)
if qr is None:
    qr = [1500, 150000]
if qc is None:
    qc = [1000, 150000]

# L O A D   D A T A
# --------------------------------------------------
h = Homodyne(path, only_vna=True, proj_name=tonelist_name, create_folder=False)

type_data, nzeros = h._get_meas_type()

temp_str = type_data + str(temp).zfill(nzeros)
atten_str = f"A{atte:.1f}"

f, s21 = h.data['vna']['full'][temp_str][atten_str]['data'][0]

# F I N D   K I D S
# --------------------------------------------------
ioff()
h.find_kids(f, s21, down_factor=down_factor, baseline_params=baseline_params, Qr_lim=qr, Qc_lim=qc, inter=True)
show()

