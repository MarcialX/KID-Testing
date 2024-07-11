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

# In case all the resonators are selected this is the
# general definition. However, it changes slightly for
# task: "fit_res"
if KIDS == "all":
    kids = None 
if TEMPS == "all":
    temps = None 
if ATTS == "all":
    atts = None
if SAMPLES == "all":
    samples = None

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

    print("T A S K : ", task_name)

    # Run the step
    # ---------------------------------
    # ---> Fit resonator
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
                        h.fit_vna_resonators(kids, temp, atten, sample=sample, n=n, verbose=True)

    # ---> Load fit
    elif task_name == "load_fit_res":

        path = task_params['path']
        if path is None:
            path = project_path+'/'+project_name

        type_data = task_params['type']

        h.load_fit(folder=path, data_type=type_data)        

    # ---> Merge sample results
    elif task_name == "merge_vna":

        xls_report = task_params['xls_report']

        h.merge_fit_res(kids, temps, atts, samples)

        if xls_report:
            h.vna_xls_report()

    # ---> Summary plots pt. 1
    elif task_name == "summary_plots_1":

        # Quality factors vs drive power
        q_vs_pwr = task_params['Q_vs_pwr']
        h.plot_qs_vs_drive_power(kids, temps, atts, samples)





