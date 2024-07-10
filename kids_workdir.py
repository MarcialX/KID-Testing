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


# G L O B A L   P A R A M E T E R S
# --------------------------------------------------
PROJ_NAME_DEFAULT = "DEV-EXP"
PARAMS_FILE = "params.yaml"
DATA_PATH_KEY = "DATA_FOLDER"
DEST_PATH_KEY = "PROJECT_FOLDER"
NAME_KEY = "PROJECT_NAME"

# R E A D   A R G U M E N T S
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', help="Data directory", type=str, required=True)
parser.add_argument('--name', '-n', help="Project name", type=str, required=False)
parser.add_argument('--dest', '-d', help="Destination directory", type=str, required=False)
parser.add_argument('--template', '-t', help="Load a defined template", type=str, required=False)

args = parser.parse_args()

path = args.path 
name = args.name
dest = args.dest
template = args.template

# In case some of the arguments are not defined
# - Destination folder
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

# - Define name
if name is None:
    name = PROJ_NAME_DEFAULT

project_path = dest + '/' + name

#   + Create the project folder
if not os.path.isdir(project_path):
    os.system('mkdir '+project_path)

# Create the params.yaml
# 'params.yaml' contains the general parameters for the experiment
os.system('cp '+"./default_files/"+PARAMS_FILE+" "+project_path)
# Modify params.yaml file

with open(project_path+'/'+PARAMS_FILE, 'r') as f:
    lines = f.readlines()
    updated_lines = []
    for line in lines:
        if DATA_PATH_KEY in line:
            line = line[:line.index(":") + 1] + ' "' + path + '"\n'
        if DEST_PATH_KEY in line:
            line = line[:line.index(":") + 1] + ' "' + dest + '/"\n'
        if NAME_KEY in line:
            line = line[:line.index(":") + 1] + ' "' + name + '"\n'
    
        updated_lines.append(line)

with open(project_path+'/'+PARAMS_FILE, 'w') as wf:
    for line in updated_lines:
        wf.write(line)

# Template
if template is None:
    # Use all the default templates
    templates = next(os.walk('./default_files/'), (None, None, []))[2]
    for template in templates:
        if template != PARAMS_FILE:
            os.system('cp '+"./default_files/"+template+" "+project_path)