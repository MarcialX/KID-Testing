#! /home/marcial/.venv/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KID finder tool.
# kids_finder.py
#
# Marcial Becerril, @ 16 Aug 2024
# Latest Revision: 16 Aug 2024, 13:05 GMT-6
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
parser.add_argument('--path', '-p', help="Data directory", type=str, required=True)
parser.add_argument('--temp', '-t', help="Temperature", type=int, required=False)
parser.add_argument('--atte', '-a', help="Attenuation", type=int, required=False)
parser.add_argument('--mode', '-m', help="Blackbody and dark meas", type=int, required=False)

args = parser.parse_args()

path = args.path
if path[-1] != '/':
    path += '/'

temp = args.temp
atte = args.atte
mode = args.mode

if temp is None:
    temp = 80
if atte is None:
    atte = 40
if mode is None:
    mode = "D"

# L O A D   D A T A
# --------------------------------------------------
h = Homodyne(path, only_vna=True, proj_name='Find_KIDs_testing', create_folder=False)

n_zeros = 0
if mode == "D":
    n_zeros = 4
elif mode == "B":
    n_zeros = 3

temp_str = mode+str(temp).zfill(n_zeros)
atten_str = f"A{atte:.1f}"

f, s21 = h.data['vna']['full'][temp_str][atten_str]['data'][0]

# F I N D   K I D S
# --------------------------------------------------
ioff()
h.find_kids(f, s21, inter=True)
show()

