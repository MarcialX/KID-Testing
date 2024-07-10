# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Interpreter to run KID-Testing analysis tasks
# do_task.py
#
# Marcial Becerril, @ 10 Jul 2024
# Latest Revision: 10 Jul 2024, 13:06 GMT-6
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

from homodyne import *

from misc.msg_custom import *


# G L O B A L   P A R A M E T E R S
# --------------------------------------------------
PARAMS_FILE = "params.yaml"
BACKUP_FILE = "proj_bkp"


# R E A D   A R G U M E N T S
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--tasks', '-t', help="Tasks file", type=str, required=True)

args = parser.parse_args()

tasks = args.tasks

# Loading tasks file
# --------------------------------------------------
msg('Loading tasks file', 'info')
try:
    with open(tasks) as file:
        tasks_data = yaml.safe_load(file)
    msg('Tasks file loaded', 'ok')
except Exception as e:
    msg('Fail loading tasks file. '+str(e), 'fail')

# Data selection
# --------------------------------------------------
TEMPS = tasks_data['TEMPS']
ATTS = tasks_data['ATTS']
KIDS = tasks_data['KIDS']

SAMPLES = tasks_data['SAMPLES']
ATT_OVRDRN = tasks_data['ATT_OVRDRN']

LOAD_PROJ = tasks_data['LOAD_PROJ']
SAVE_PROJ = tasks_data['SAVE_PROJ']

if KIDS == "all":
    kids = None

# Load gral params file
msg('Loading general params file', 'info')
try:
    with open(PARAMS_FILE) as file:
        gral_params = yaml.safe_load(file)
    msg('General params file loaded', 'ok')
except Exception as e:
    msg('Fail loading general params file. '+str(e), 'fail')

data_path = gral_params['DATA_FOLDER']
project_name = gral_params['PROJECT_NAME']
project_path = gral_params['PROJECT_FOLDER']


# R U N   T A S K S
# --------------------------------------------------
# Create the homodyne object
if LOAD_PROJ:
    used_project = project_path 
else: 
    used_project = data_path

h = Homodyne(used_project, work_dir=project_path, proj_name=project_name, load_saved=LOAD_PROJ)

TASKS = tasks_data['TASKS']

for s, step in enumerate(TASKS):

    # Task name
    task_name = TASKS[step]['name']
    # Task parameters
    task_params = TASKS[step]['params']

    # Run the step
    if task_name == "fit_res":
        
        type_data = task_params['type']
        n = task_params['n']

        if KIDS == "all":
            kids = None 
            kid = 'K000'

        if TEMPS == "all":
            temps = h.data[type_data][kid].keys()
        else:
            temps = TEMPS

        for temp in temps:
            if ATTS == "all":
                attens = h.data[type_data][kid][temp].keys()
            else:
                attens = ATTS

            for atten in attens:
                if SAMPLES == "all":
                    samples = h.data[type_data][kid][temp][atten]['data'].keys()
                else:
                    samples = SAMPLES

                for sample in samples:
                    if type_data == 'vna':
                        print(atten, temp, sample)
                        print('----------------')
                        h.fit_vna_resonators(kids, temp, atten, sample=sample, n=n, verbose=True)

        """
        # Fit resonators from VNA
        step_att = 2
        attens = np.arange(0, 60+step_att, step_att)
        btemp = 80

        samples = 3
        for sample in range(samples):
            for atten in attens:
                h.fit_vna_resonators(None, btemp, atten, sample=sample, n=3.5, verbose=True)
        """
