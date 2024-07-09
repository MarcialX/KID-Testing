# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Create a new working directory for a new experiment
# kids_workdir.py
#
# Marcial Becerril, @ 09 Jul 2024
# Latest Revision: 09 Jul 2024, 14:16 GMT-6
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


# R E A D   A R G U M E N T S
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', help="Data directory", type=str, required=True)
parser.add_argument('--dest', '-d', help="Destination directory", type=str, required=False)
parser.add_argument('--template', '-t', help="Load a defined template", type=str, required=False)

args = parser.parse_args()

path = args.path 
dest = args.dest
template = args.template

# In case some of the arguments are not defined
# Destination folder.
if dest is None:
    dest = "/home/marcial/Documents/SOUK-devices"
    print('Default destination directory: /home/marcial/Documents/SOUK-devices')

else: 
    folders = dest.split("/")
    folders = [f for f in folders if f != ""]
    flag_real_dir = True
    folder_conca = ""
    for folder in folders:
        folder_conca += '/'+folder
        if not os.path.isdir(folder_conca):
            os.system('mkdir '+folder_conca)
            print('New folder created: '+folder)

# Template
if template is None:
        

